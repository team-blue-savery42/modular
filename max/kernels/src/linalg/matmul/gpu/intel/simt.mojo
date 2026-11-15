# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Sean Avery. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Intel GPU matmul without DPAS: a register-tiled GEMM in shared local
memory, for the integrated GPUs (Xe-LP, Xe-LPG), which have no DPAS.

Each work-group of `SIMT_THREADS` work-items computes a `SIMT_BM` x
`SIMT_BN` tile of C: its work-items hold `SIMT_TM` x `SIMT_TN` outputs each
in float32 registers and step through K in slices of `SIMT_BK`, which the
work-group stages in shared local memory (converted to float32, K-major, so
the inner loop reads each operand as a vector) while it prefetches the next
slice into registers. A sub-group (16 consecutive work-items) shares one row
group of the tile, so its reads of the A slice are broadcasts and its reads
of the B slice are consecutive vectors.

Shapes: A is M x K and B is K x N (or N x K with `transpose_b`), row-major,
with K a multiple of 4 (and N, for B as K x N); edges in M and N are
masked, the tail of K is zero-filled.

Launch: `simt_grid(m, n)` work-groups of `SIMT_THREADS` work-items, in x
(M tiles fastest, so neighbouring work-groups share B's tile).
"""

from std.math import ceildiv
from std.memory import stack_allocation
from std.memory.address_space import AddressSpace

from layout import Coord
from max.gpu import barrier, block_idx, thread_idx

from ....utils import ElementwiseComputeFn, ElementwiseEpilogueFn

comptime SIMT_THREADS = 256
"""Work-items per work-group (16 x 16, within the 512 an Xe-LP allows)."""
comptime SIMT_TM = 4
comptime SIMT_TN = 4
comptime SIMT_BM = 16 * SIMT_TM
comptime SIMT_BN = 16 * SIMT_TN
comptime SIMT_BK = 16


def simt_supported(m: Int, n: Int, k: Int, transpose_b: Bool) -> Bool:
    """Whether `simt_kernel_fn` takes this shape (see the module)."""
    return (
        m > 1
        and n >= 1
        and k >= 4
        and k % 4 == 0
        and (transpose_b or n % 4 == 0)
        and m < (1 << 31)
        and n < (1 << 31)
        and k < (1 << 31)
    )


@always_inline
def simt_grid(m: Int, n: Int) -> Int:
    """Work-groups to launch for an M x N output."""
    return ceildiv(m, SIMT_BM) * ceildiv(n, SIMT_BN)


@always_inline
def _store_one[
    c_type: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    c: MutPointer[Scalar[c_type], MutAnyOrigin],
    n: Int,
    row: Int,
    col: Int,
    value: Scalar[c_type],
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
):
    comptime if has_epilogue_fn:
        epilogue_fn[c_type, 1, alignment=1](Coord(row, col), value)
    else:
        var v = value
        comptime if has_compute_fn:
            v = compute_fn[c_type, 1, alignment=1](Coord(row, col), v)
        c.unsafe_store(row * n + col, v)


@always_inline
def _load_rows[
    dtype: DType
](
    p: MutPointer[Scalar[dtype], MutAnyOrigin],
    row: Int,
    rows: Int,
    K: Int,
    k0: Int,
    l_k: Int,
) -> SIMD[DType.float32, 4]:
    """Four consecutive K values of a row, as float32; zeros outside."""
    if row < rows and k0 + l_k < K:
        return p.unsafe_load[width=4](row * K + k0 + l_k).cast[DType.float32]()
    return SIMD[DType.float32, 4](0)


@always_inline
def _fetch[
    dtype: DType, transpose_b: Bool
](
    a: MutPointer[Scalar[dtype], MutAnyOrigin],
    b: MutPointer[Scalar[dtype], MutAnyOrigin],
    a_reg: MutPointer[
        Scalar[DType.float32], _, address_space=AddressSpace.LOCAL
    ],
    b_reg: MutPointer[
        Scalar[DType.float32], _, address_space=AddressSpace.LOCAL
    ],
    t: Int,
    m0: Int,
    n0: Int,
    M: Int,
    N: Int,
    K: Int,
    k0: Int,
):
    """Loads the work-item's share of the A and B slices at `k0` into
    registers: vectors of 4 consecutive K values of rows of A (and of B^T),
    or of 4 consecutive columns of rows of B."""
    comptime f32 = DType.float32
    comptime NA = SIMT_BM * SIMT_BK // (4 * SIMT_THREADS)
    comptime NB = SIMT_BN * SIMT_BK // (4 * SIMT_THREADS)
    comptime for v in range(NA):
        var idx = t + v * SIMT_THREADS
        a_reg.unsafe_store(
            4 * v,
            _load_rows[dtype](
                a,
                m0 + idx // (SIMT_BK // 4),
                M,
                K,
                k0,
                (idx % (SIMT_BK // 4)) * 4,
            ),
        )
    comptime for v in range(NB):
        var idx = t + v * SIMT_THREADS
        comptime if transpose_b:
            b_reg.unsafe_store(
                4 * v,
                _load_rows[dtype](
                    b,
                    n0 + idx // (SIMT_BK // 4),
                    N,
                    K,
                    k0,
                    (idx % (SIMT_BK // 4)) * 4,
                ),
            )
        else:
            var kr = k0 + idx // (SIMT_BN // 4)
            var col = n0 + (idx % (SIMT_BN // 4)) * 4
            var x = SIMD[f32, 4](0)
            if kr < K and col < N:
                x = b.unsafe_load[width=4](kr * N + col).cast[f32]()
            b_reg.unsafe_store(4 * v, x)


def simt_kernel_fn[
    a_type: DType,
    c_type: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    transpose_b: Bool,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    c: MutPointer[Scalar[c_type], MutAnyOrigin],
    a: MutPointer[Scalar[a_type], MutAnyOrigin],
    b: MutPointer[Scalar[a_type], MutAnyOrigin],
    m: Int64,
    n: Int64,
    k: Int64,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
):
    """`C = A * B` (see the module), with MAX's output closures (pass both
    with `host_args`; `has_epilogue_fn` and `has_compute_fn` say which one is
    used, if any). Accumulates in float32."""
    comptime TM = SIMT_TM
    comptime TN = SIMT_TN
    comptime BM = SIMT_BM
    comptime BN = SIMT_BN
    comptime BK = SIMT_BK
    comptime f32 = DType.float32

    var M = Int(m)
    var N = Int(n)
    var K = Int(k)
    var t = Int(thread_idx.x)
    var tx = t % 16
    var ty = t // 16

    var tiles_m = ceildiv(M, BM)
    var tile = Int(block_idx.x)
    var m0 = (tile % tiles_m) * BM
    var n0 = (tile // tiles_m) * BN

    # Shared slices, K-major: a_s[kk * BM + i], b_s[kk * BN + j], float32.
    var a_s = stack_allocation[
        BK * BM, f32, address_space=AddressSpace.SHARED
    ]()
    var b_s = stack_allocation[
        BK * BN, f32, address_space=AddressSpace.SHARED
    ]()

    var acc = stack_allocation[TM * TN, f32, address_space=AddressSpace.LOCAL]()
    comptime for i in range(TM * TN):
        acc.unsafe_store(i, Scalar[f32](0))

    # Each work-item moves NA vectors of 4 consecutive K values of rows of
    # the A slice (BM rows x BK) per step, and NB of the B slice.
    comptime NA = BM * BK // (4 * SIMT_THREADS)
    comptime NB = BN * BK // (4 * SIMT_THREADS)
    var a_reg = stack_allocation[
        NA * 4, f32, address_space=AddressSpace.LOCAL
    ]()
    var b_reg = stack_allocation[
        NB * 4, f32, address_space=AddressSpace.LOCAL
    ]()

    _fetch[a_type, transpose_b](a, b, a_reg, b_reg, t, m0, n0, M, N, K, 0)

    var steps = ceildiv(K, BK)
    for step in range(steps):
        comptime for v in range(NA):
            var idx = t + v * SIMT_THREADS
            var row = idx // (BK // 4)
            var kq = (idx % (BK // 4)) * 4
            comptime for q in range(4):
                a_s.unsafe_store(
                    (kq + q) * BM + row, a_reg.unsafe_load(4 * v + q)
                )
        comptime for v in range(NB):
            var idx = t + v * SIMT_THREADS
            comptime if transpose_b:
                var row = idx // (BK // 4)
                var kq = (idx % (BK // 4)) * 4
                comptime for q in range(4):
                    b_s.unsafe_store(
                        (kq + q) * BN + row, b_reg.unsafe_load(4 * v + q)
                    )
            else:
                b_s.unsafe_store(
                    (idx // (BN // 4)) * BN + (idx % (BN // 4)) * 4,
                    b_reg.unsafe_load[width=4](4 * v),
                )
        barrier()
        if step + 1 < steps:
            _fetch[a_type, transpose_b](
                a, b, a_reg, b_reg, t, m0, n0, M, N, K, (step + 1) * BK
            )
        comptime for kk in range(BK):
            var av = a_s.unsafe_load[width=TM](kk * BM + ty * TM)
            var bv = b_s.unsafe_load[width=TN](kk * BN + tx * TN)
            comptime for i in range(TM):
                comptime for j in range(TN):
                    comptime idx = i * TN + j
                    acc.unsafe_store(idx, acc.unsafe_load(idx) + av[i] * bv[j])
        barrier()

    comptime for i in range(TM):
        var row = m0 + ty * TM + i
        if row < M:
            comptime for j in range(TN):
                var col = n0 + tx * TN + j
                if col < N:
                    _store_one[has_epilogue_fn, has_compute_fn](
                        c,
                        N,
                        row,
                        col,
                        acc.unsafe_load(i * TN + j).cast[c_type](),
                        epilogue_fn,
                        compute_fn,
                    )
