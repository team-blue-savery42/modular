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

# ===----------------------------------------------------------------------=== #
# ===----------------------------------------------------------------------=== #
"""Intel Xe2 GEMM: `C = A * B` with DPAS and 2D block I/O.

A is M x K row-major; B is K x N row-major, or (with `transpose_b`) given as
its transpose, N x K row-major; C is M x N row-major. A and B are float16 or
bfloat16 (the same type), the products accumulate in float32, and C is
float32 or the input type (rounded once, at the store).

Tiling (comptime parameters, defaults for `bmg-g31`, the Arc Pro B70): a
work-group computes a `wg_m` x `wg_n` tile of C (256 x 256) with
`(wg_m / sg_m) * (wg_n / sg_n)` sub-groups of 16 work-items (32), each owning
an `sg_m` x `sg_n` tile (32 x 64) held in registers as `sg_m / 8` x `sg_n / 16`
DPAS accumulators (`SIMD[float32, 8]` per lane: 128 floats per work-item, so
the kernel wants IGC's large (256) GRF mode, which the driver plugin's auto
large-GRF option allows). K advances 32 at a time: per step a sub-group loads
its A rows with one `block_load[16, sg_m, 2]` and its B columns with
`block_load_transform[16, 32, 2]` (VNNI, `mma()`'s B as loaded) per 32
columns, or, for a transposed B, `block_load_transpose[8, 32]` of the 32-bit
pairs, which is the packed B; then `2 * (sg_m / 8) * (sg_n / 16)` DPAS calls
(`mma()`, M = 8, N = K = 16); `k_step` (32) is the K per loop iteration.
There is no shared local memory: each work-group prefetches
(`block_prefetch`) its whole A and B tiles `prefetch` - 1 steps ahead, split
over its sub-groups in blocks of 8 rows x 32 columns, so the loads of the step
hit the cache. With `split_barrier` (the default for a transposed B) the
sub-groups of a work-group arrive at a work-group split barrier
(`SPV_INTEL_split_barrier`) at the start of each iteration and wait for it at
the end, which keeps them within one iteration of each other, so the tiles
one sub-group prefetched are still in the cache when the others load them.
Work-groups are ordered in groups of `group_m` tile rows (the "grouped
ordering" of Triton's tutorial 03) so that consecutive work-groups share A
and B tiles in L2. This is the shape of intel-xpu-backend-for-triton's Xe2
GEMM (256 x 256 x 32 work-group tile, 32 sub-groups as 8 x 4, two steps of 2D
block prefetch, large GRF), read from its compiled kernels; no code is taken
from it.

Defaults for `bmg-g31` (from a tile sweep on the B70,
`plugins/intel/utils/bench-gemm`): 256 x 256 work-group tiles of 32 x 64
sub-group tiles, prefetch distance 2, `group_m` 8, one K step of 32 per
iteration, the split barrier for a transposed B only. When an output has fewer
256 x 256 tiles than the B70 has Xe-cores (`gemm_small_tiles`), work-groups of
256 x `SMALL_WG_N` (128) fill more of them: launch
`gemm_kernel[..., wg_n=SMALL_WG_N]` with `gemm_grid[256, SMALL_WG_N]` and
`gemm_block[256, SMALL_WG_N]`. The caller gives the device's Xe-core count
(`gemm_small_tiles(m, n, xe_cores)`).

Skinny outputs (M <= `SKINNY_MAX_M`, decode; up to `SKINNY_WIDE_MAX_M` with
32- and 64-row tiles for batched decode) stream B once: `skinny_kernel`
gives each work-group `SKINNY_SG_N` columns of C and `skinny_ks` sub-groups
that split K between them (interleaved by 32), then sums their partial tiles
in shared local memory (see `skinny_kernel`).

`gemm_kernel_fn` and `skinny_kernel_fn` are the same kernels with MAX's
output closures (`_matmul_gpu`'s `epilogue_fn`, which stores each output
itself, or `compute_fn`, which maps it before the store); they take the
sizes as `Int64`, for `DeviceContext.enqueue_function`.

Edges need no special code: block loads read zeros outside the matrices (so a
K that is not a multiple of 32 adds zero products) and block stores skip
elements outside C. The hardware's 2D block rules do constrain the shapes
(`gemm_supported`): every row pitch must be a multiple of 16 bytes and every
matrix at least 64 bytes wide, so N and K must be multiples of 8 and at
least 32; the base addresses must be 64-byte aligned (device allocations
are). The kernel needs sub-group size 16 (DPAS and 2D block I/O assert it).

Launch: `gemm_grid(m, n)` work-groups of `gemm_block()` work-items, in x
(with the same tile parameters as the kernel).

FP8 weights: B^T in float8_e4m3fn (N x K bytes) with one float32 scale, A
bfloat16, C bfloat16 or float32: `C = scale * A * B`. Xe2's DPAS has no
8-bit float type. `skinny_kernel_f8` (M <= 16) reads B^T's bytes in each K
step with one `block_load_transpose[8, 32]` per 32 columns (32-bit words of
4 K values: half the bytes of a bfloat16 B^T step) and widens them in
registers to the packed bfloat16 B of the bfloat16 path
(`_f8_to_bf16_pairs`: exact, as float16 bits of the value / 256, widened to
float32 and truncated to bfloat16); the scale (times 256) multiplies the
float32 accumulators before the store. Larger M widens B^T once into a
bfloat16 copy (`f8_widen_kernel`) and runs the bfloat16 GEMM on it with the
same scaling (`gemm_kernel_f8_widened`): widening in the GEMM's registers
repeats the work for every sub-group row of a work-group (2.5-4x the
bfloat16 GEMM's time at M = 512 on the B70, against 1.2-1.75x this way).
Shapes: `gemm_f8_supported` (K a multiple of 16 and at least 64).
"""

from std.ffi import external_call
from std.math import ceildiv
from std.memory import bitcast, stack_allocation
from std.memory.address_space import AddressSpace
from std.sys import size_of

from layout import Coord
from max.gpu import barrier, block_dim, block_idx, thread_idx
from max.gpu.compute.mma import mma
from max.gpu.memory.block_io_intel import (
    block_load,
    block_load_transform,
    block_load_transpose,
    block_prefetch,
    block_store,
)

from ....utils import (
    ElementwiseComputeFn,
    ElementwiseEpilogueFn,
    no_compute_fn,
    no_epilogue_fn,
)

comptime K_STEP = 32
"""K per load step (two DPAS K = 16 halves); a loop iteration does
`k_step // K_STEP` of them."""

# SPIR-V scopes and memory semantics (the SPIR-V spec's enums).
comptime _WORKGROUP = Int32(2)
comptime _NO_SEMANTICS = Int32(0)


@inline(.always)
def _barrier_arrive():
    """`OpControlBarrierArriveINTEL` (SPV_INTEL_split_barrier), work-group
    execution scope, no memory semantics: signals this sub-group reached
    the barrier and keeps going."""
    external_call["_Z33__spirv_ControlBarrierArriveINTELiii", NoneType](
        _WORKGROUP, _WORKGROUP, _NO_SEMANTICS
    )


@inline(.always)
def _barrier_wait():
    """`OpControlBarrierWaitINTEL`: waits until every sub-group of the
    work-group arrived."""
    external_call["_Z31__spirv_ControlBarrierWaitINTELiii", NoneType](
        _WORKGROUP, _WORKGROUP, _NO_SEMANTICS
    )


comptime SMALL_WG_N = 128
"""Work-group tile columns for outputs with few tiles (`gemm_small_tiles`)."""


def gemm_small_tiles(m: Int, n: Int, xe_cores: Int) -> Bool:
    """Whether an M x N output should use 256 x `SMALL_WG_N` work-group
    tiles: it has fewer 256 x 256 tiles than the GPU has Xe-cores
    (`xe_cores`: Level Zero's slices x sub-slices, which MAX's
    `DeviceAttribute.MULTIPROCESSOR_COUNT` and the Intel device features
    report; 32 on the B70, where 1024^3 runs 1.5x faster with these tiles,
    while 2048^3 and 512 x 4096 x 4096, with 32 or more tiles, run
    slower)."""
    return gemm_grid(m, n) < xe_cores


def gemm_block[
    wg_m: Int = 256, wg_n: Int = 256, sg_m: Int = 32, sg_n: Int = 64
]() -> Int:
    """Work-items per work-group: 16 per sub-group."""
    return (wg_m // sg_m) * (wg_n // sg_n) * 16


def gemm_grid[wg_m: Int = 256, wg_n: Int = 256](m: Int, n: Int) -> Int:
    """Work-groups for an M x N output."""
    return ceildiv(m, wg_m) * ceildiv(n, wg_n)


def gemm_supported[
    a_type: DType, c_type: DType
](m: Int, n: Int, k: Int) -> Bool:
    """Whether the kernel takes these types and this shape: 2D block I/O
    needs every row pitch a multiple of 16 bytes and every matrix at least
    64 bytes wide (N, K multiples of 8 and >= 32 for 16-bit inputs), and
    (SPV_INTEL_2d_block_io) every matrix at most 2^24 bytes wide and 2^24
    rows high: A is K elements wide and M high, B N wide and K high (B^T K
    wide and N high), C N wide in C's type and M high."""
    var types = (a_type == DType.float16 or a_type == DType.bfloat16) and (
        c_type == DType.float32 or c_type == a_type
    )
    comptime limit = 1 << 24
    return (
        types
        and m >= 1
        and n >= 32
        and k >= 32
        and n % 8 == 0
        and k % 8 == 0
        and m <= limit
        and n <= limit
        and k <= limit
        and k * size_of[a_type]() <= limit
        and n * size_of[a_type]() <= limit
        and n * size_of[c_type]() <= limit
    )


def gemm_f8_supported[c_type: DType](m: Int, n: Int, k: Int) -> Bool:
    """Whether the FP8-weight kernels take this shape (bfloat16 A,
    float8_e4m3fn B^T, `c_type` C): `gemm_supported`'s rules for A and C,
    and B^T's rows of K bytes at least 64 bytes wide with a pitch of a
    multiple of 16 bytes (K a multiple of 16 and at least 64)."""
    return (
        gemm_supported[DType.bfloat16, c_type](m, n, k)
        and k % 16 == 0
        and k >= 64
    )


comptime F8_UNIT = Float32(256)
"""`_f8_to_bf16_pairs` gives each float8_e4m3fn value / 256; the FP8
kernels multiply their accumulators by the weight scale times this."""


@inline(.always)
def _f8_to_bf16_pairs(words: SIMD[DType.uint32, 8]) -> SIMD[DType.uint32, 16]:
    """Widens 8 words of 4 float8_e4m3fn values each (K order from the low
    byte) to the 16 words of bfloat16 pairs (K order from the low half) of
    the same values / 256, exactly: each byte's sign, exponent and mantissa
    become float16 bits of the value / 256 (float16 has e4m3's 4 exponent
    bits in its low ones and room for its subnormals), the hardware widens
    float16 to float32, and the upper half of a float32 with 3 mantissa bits
    is that bfloat16. (NaN, 0x7f / 0xff, becomes 480 / 256; checkpoints
    hold none.)"""
    # Bytes 0, 1 and bytes 2, 3 of each word, one per 16-bit half.
    var lo = (words & 0xFF) | ((words & 0xFF00) << 8)
    var hi = ((words >> 16) & 0xFF) | ((words >> 8) & 0xFF0000)
    lo = ((lo & 0x00800080) << 8) | ((lo & 0x007F007F) << 7)
    hi = ((hi & 0x00800080) << 8) | ((hi & 0x007F007F) << 7)
    var lo32 = bitcast[DType.uint32, 16](
        bitcast[DType.float16, 16](lo).cast[DType.float32]()
    )
    var hi32 = bitcast[DType.uint32, 16](
        bitcast[DType.float16, 16](hi).cast[DType.float32]()
    )
    var lo_even, lo_odd = lo32.deinterleave()
    var hi_even, hi_odd = hi32.deinterleave()
    var lo_pair = (lo_even >> 16) | (lo_odd & 0xFFFF0000)
    var hi_pair = (hi_even >> 16) | (hi_odd & 0xFFFF0000)
    return lo_pair.interleave(hi_pair)


@inline(.always)
def _prefetch_step[
    a_type: DType, //, transpose_b: Bool, wg_m: Int, wg_n: Int, sgs: Int
](
    a: MutPointer[Scalar[a_type], MutAnyOrigin],
    b: MutPointer[Scalar[a_type], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    wm: Int,
    wn: Int,
    sg: Int,
    kp: Int,
):
    """Prefetches the work-group's A and B tiles of the K step at `kp`,
    split over its `sgs` sub-groups in blocks of 8 rows x 32 columns."""
    # A: wg_m x 32.
    comptime a_blocks = wg_m // 8
    comptime for j in range(ceildiv(a_blocks, sgs)):
        var i = sg + j * sgs
        if i < a_blocks:
            block_prefetch[16, 8, 2](a, k, m, k, kp, wm + 8 * i)
    comptime if transpose_b:
        # B^T: wg_n x 32, split like A.
        comptime t_blocks = wg_n // 8
        comptime for j in range(ceildiv(t_blocks, sgs)):
            var i = sg + j * sgs
            if i < t_blocks:
                block_prefetch[16, 8, 2](b, k, n, k, kp, wn + 8 * i)
    else:
        # B: 32 x wg_n.
        comptime b_blocks = 4 * (wg_n // 32)
        comptime for j in range(ceildiv(b_blocks, sgs)):
            var i = sg + j * sgs
            if i < b_blocks:
                block_prefetch[16, 8, 2](
                    b, n, k, n, wn + 32 * (i // 4), kp + 8 * (i % 4)
                )


@inline(.always)
def _k_step[
    a_type: DType, b_type: DType, //, transpose_b: Bool, sg_m: Int, sg_n: Int
](
    acc: MutPointer[Scalar[DType.float32], _, address_space=AddressSpace.LOCAL],
    a: MutPointer[Scalar[a_type], MutAnyOrigin],
    b: MutPointer[Scalar[b_type], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    kk: Int,
    m0: Int,
    n0: Int,
):
    """One K step (32) of a sub-group's `sg_m` x `sg_n` tile at (`m0`,
    `n0`): its block loads of A and B at K offset `kk` and
    `2 * (sg_m / 8) * (sg_n / 16)` DPAS calls into `acc`."""
    comptime rows = sg_m // 8  # DPAS row blocks (M = 8)
    comptime cols = sg_n // 16  # DPAS column blocks (N = 16)
    comptime a_vals = 2 * sg_m  # per lane: two 16-wide blocks of sg_m rows
    comptime b_vals = 64  # per lane: two 16-wide blocks of 32 rows
    # A rows by block loads of at most 32 rows (the hardware's limit): lane
    # values [16-wide block h][row] as one block load gives them, `sg_m` rows.
    comptime a_load = 32 if sg_m > 32 else sg_m
    var av = SIMD[a_type, a_vals]()
    comptime for c in range(sg_m // a_load):
        var part = rebind[SIMD[a_type, 2 * a_load]](
            block_load[16, a_load, 2](a, k, m, k, kk, m0 + a_load * c)
        )
        comptime for h in range(2):
            av = av.insert[offset=h * sg_m + a_load * c](
                part.slice[a_load, offset=h * a_load]()
            )
    comptime if b_type == DType.float8_e4m3fn:
        # FP8 B^T as 32-bit words of 4 K values: per 32 columns of C one
        # transposed load covers the whole step (lane l gets B^T rows l and
        # l + 16, element 2c + i is row l + 16i, word c), widened to the
        # bfloat16 pairs of both K halves (`_f8_to_bf16_pairs`).
        var bw = b.unsafe_bitcast[UInt32]()
        var k4 = k // 4
        comptime for j in range(sg_n // 32):
            var v = rebind[SIMD[DType.uint32, 16]](
                block_load_transpose[8, 32](bw, k4, n, k4, kk // 4, n0 + 32 * j)
            )
            comptime for i in range(2):
                var words = SIMD[DType.uint32, 8]()
                comptime for w in range(8):
                    words[w] = v[2 * w + i]
                var pairs = _f8_to_bf16_pairs(words)
                comptime cb = 2 * j + i
                comptime for h in range(2):
                    var bp = pairs.slice[8, offset=8 * h]()
                    comptime for r in range(rows):
                        var af = av.slice[8, offset=h * sg_m + 8 * r]()
                        comptime ai = 8 * (r * cols + cb)
                        var d = SIMD[DType.float32, 8]()
                        mma(d, af, bp, acc.unsafe_load[width=8](ai))
                        acc.unsafe_store(ai, d)
    elif transpose_b:
        # B as 32-bit pairs along K. Per 32 columns of C and K half h: lane
        # l gets B^T rows (columns of B) l and l + 16, 8 words of K pairs
        # each (element 2c + i is row l + 16i, word c): the packed B of two
        # DPAS column blocks.
        var bt = b.unsafe_bitcast[UInt32]()
        var k2 = k // 2
        comptime for h in range(2):
            comptime for j in range(sg_n // 32):
                var v = rebind[SIMD[DType.uint32, 16]](
                    block_load_transpose[8, 32](
                        bt, k2, n, k2, kk // 2 + 8 * h, n0 + 32 * j
                    )
                )
                comptime for i in range(2):
                    var bp = SIMD[DType.uint32, 8]()
                    comptime for w in range(8):
                        bp[w] = v[2 * w + i]
                    comptime cb = 2 * j + i
                    comptime for r in range(rows):
                        var af = av.slice[8, offset=h * sg_m + 8 * r]()
                        comptime ai = 8 * (r * cols + cb)
                        var d = SIMD[DType.float32, 8]()
                        mma(d, af, bp, acc.unsafe_load[width=8](ai))
                        acc.unsafe_store(ai, d)
    else:
        comptime for j in range(sg_n // 32):
            var bv = rebind[SIMD[a_type, b_vals]](
                block_load_transform[16, 32, 2](b, n, k, n, n0 + 32 * j, kk)
            )
            comptime for h in range(2):
                comptime for i in range(2):
                    var bf = bv.slice[16, offset=32 * i + 16 * h]()
                    comptime cb = 2 * j + i
                    comptime for r in range(rows):
                        var af = av.slice[8, offset=h * sg_m + 8 * r]()
                        comptime ai = 8 * (r * cols + cb)
                        var d = SIMD[DType.float32, 8]()
                        mma(d, af, bf, acc.unsafe_load[width=8](ai))
                        acc.unsafe_store(ai, d)


@inline(.always)
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
    """Stores one output (in bounds) of C, through MAX's closures."""
    comptime if has_epilogue_fn:
        epilogue_fn[c_type, 1, alignment=1](Coord(row, col), value)
    else:
        var v = value
        comptime if has_compute_fn:
            v = compute_fn[c_type, 1, alignment=1](Coord(row, col), v)
        c.unsafe_store(row * n + col, v)


@inline(.always)
def _store_tile[
    c_type: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    rows: Int,
    cols: Int,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    c: MutPointer[Scalar[c_type], MutAnyOrigin],
    acc: MutPointer[Scalar[DType.float32], _, address_space=AddressSpace.LOCAL],
    m: Int,
    n: Int,
    m0: Int,
    n0: Int,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
):
    """Stores a sub-group's tile. Each accumulator is 8 rows x 16 columns,
    row r of column l in element r of lane l: block_store's layout, which
    skips elements outside C. With a closure, each lane hands its in-bounds
    outputs to it one at a time."""
    comptime for r in range(rows):
        comptime for cb in range(cols):
            var v = acc.unsafe_load[width=8](8 * (r * cols + cb)).cast[c_type]()
            comptime if has_epilogue_fn or has_compute_fn:
                var col = n0 + 16 * cb + Int(thread_idx.x) % 16
                comptime for e in range(8):
                    var row = m0 + 8 * r + e
                    if row < m and col < n:
                        _store_one[has_epilogue_fn, has_compute_fn](
                            c, n, row, col, v[e], epilogue_fn, compute_fn
                        )
            else:
                block_store[16, 8](c, n, m, n, n0 + 16 * cb, m0 + 8 * r, v)


@inline(.always)
def _gemm_impl[
    a_type: DType,
    c_type: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    transpose_b: Bool,
    wg_m: Int,
    wg_n: Int,
    sg_m: Int,
    sg_n: Int,
    prefetch: Int,
    group_m: Int,
    k_step: Int,
    split_barrier: Bool,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
    scaled: Bool = False,
](
    c: MutPointer[Scalar[c_type], MutAnyOrigin],
    a: MutPointer[Scalar[a_type], MutAnyOrigin],
    b: MutPointer[Scalar[a_type], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    scale: Float32,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
):
    _check_types[a_type, a_type, c_type, transpose_b]()
    comptime assert (
        sg_m == 8 or sg_m == 16 or sg_m == 32
    ), "Intel GEMM: sg_m is 8, 16 or 32 (one block load of A rows)"
    comptime assert sg_n % 32 == 0, "Intel GEMM: sg_n is a multiple of 32"
    comptime assert (
        wg_m % sg_m == 0 and wg_n % sg_n == 0
    ), "Intel GEMM: the sub-group tile divides the work-group tile"
    comptime assert (
        wg_m % 8 == 0 and wg_n % 32 == 0
    ), "Intel GEMM: prefetch splits the tiles into 8 x 32 blocks"
    comptime assert (
        k_step % K_STEP == 0 and k_step <= 128
    ), "Intel GEMM: k_step is 32, 64, 96 or 128"
    comptime sgs_n = wg_n // sg_n
    comptime sgs = (wg_m // sg_m) * sgs_n
    comptime rows = sg_m // 8  # DPAS row blocks (M = 8)
    comptime cols = sg_n // 16  # DPAS column blocks (N = 16)

    # Grouped work-group order (Triton's tutorial 03): group_m tile rows at a
    # time, column-major inside a group.
    var grid_m = ceildiv(m, wg_m)
    var grid_n = ceildiv(n, wg_n)
    var pid = Int(block_idx.x)
    var in_group = group_m * grid_n
    var first_m = (pid // in_group) * group_m
    var size_m = min(grid_m - first_m, group_m)
    var pid_m = first_m + (pid % in_group) % size_m
    var pid_n = (pid % in_group) // size_m
    var wm = pid_m * wg_m
    var wn = pid_n * wg_n

    var sg = Int(thread_idx.x) // 16
    var m0 = wm + (sg // sgs_n) * sg_m
    var n0 = wn + (sg % sgs_n) * sg_n

    var acc = stack_allocation[
        rows * cols * 8, DType.float32, address_space=AddressSpace.LOCAL
    ]()
    comptime for i in range(rows * cols):
        acc.unsafe_store(8 * i, SIMD[DType.float32, 8](0))

    comptime if prefetch > 0:
        comptime for d in range(prefetch - 1):
            _prefetch_step[transpose_b, wg_m, wg_n, sgs](
                a, b, m, n, k, wm, wn, sg, d * K_STEP
            )

    # The loop runs at least once (gemm_supported: k >= 32), written so:
    # with a zero-trip path the optimizer merges the converted outputs of
    # both paths in phis at the loop exit, which keeps all of them live at
    # once (bfloat16 C spilled).
    var k0 = 0
    while True:
        comptime if split_barrier:
            _barrier_arrive()
        comptime for u in range(k_step // K_STEP):
            var kk = k0 + u * K_STEP
            comptime if prefetch > 0:
                _prefetch_step[transpose_b, wg_m, wg_n, sgs](
                    a, b, m, n, k, wm, wn, sg, kk + (prefetch - 1) * K_STEP
                )
            _k_step[transpose_b, sg_m, sg_n](acc, a, b, m, n, k, kk, m0, n0)
        comptime if split_barrier:
            _barrier_wait()
        k0 += k_step
        if k0 >= k:
            break

    _scale_acc[scaled, rows * cols](acc, scale)
    _store_tile[rows, cols, has_epilogue_fn, has_compute_fn](
        c, acc, m, n, m0, n0, epilogue_fn, compute_fn
    )


def gemm_kernel[
    a_type: DType,
    c_type: DType,
    transpose_b: Bool = False,
    wg_m: Int = 256,
    wg_n: Int = 256,
    sg_m: Int = 32,
    sg_n: Int = 64,
    prefetch: Int = 2,
    group_m: Int = 8,
    k_step: Int = 32,
    split_barrier: Bool = transpose_b,
](
    c: MutPointer[Scalar[c_type], MutAnyOrigin],
    a: MutPointer[Scalar[a_type], MutAnyOrigin],
    b: MutPointer[Scalar[a_type], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    """`C = A * B` (see the module): one `wg_m` x `wg_n` tile of C per
    work-group of `gemm_block()` work-items.

    Parameters:
        a_type: float16 or bfloat16, the type of A and B.
        c_type: float32 or `a_type`.
        transpose_b: B is given as its N x K transpose.
        wg_m: Rows of C per work-group.
        wg_n: Columns of C per work-group.
        sg_m: Rows of C per sub-group (8, 16 or 32).
        sg_n: Columns of C per sub-group (a multiple of 32).
        prefetch: Prefetch stages: the first `prefetch` - 1 K steps (of
            32) are prefetched before the loop, and each step prefetches the
            one `prefetch` - 1 steps ahead of it, so the default 2 prefetches
            one step ahead and 1 the step about to be loaded; 0 for none.
        group_m: Tile rows per group of the work-group order.
        k_step: K per loop iteration (32, 64, 96 or 128).
        split_barrier: Keep the sub-groups of a work-group within one
            iteration of each other with a split barrier.

    Args:
        c: C, M x N, row-major.
        a: A, M x K, row-major.
        b: B, K x N row-major, or N x K row-major with `transpose_b`.
        m: M.
        n: N.
        k: K.
    """
    _gemm_impl[
        transpose_b,
        wg_m,
        wg_n,
        sg_m,
        sg_n,
        prefetch,
        group_m,
        k_step,
        split_barrier,
        False,
        False,
    ](c, a, b, m, n, k, Float32(1), no_epilogue_fn, no_compute_fn)


def gemm_kernel_fn[
    a_type: DType,
    c_type: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    transpose_b: Bool,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
    wg_m: Int = 256,
    wg_n: Int = 256,
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
    """`gemm_kernel` with its defaults and MAX's output closures (pass both
    with `host_args`; `has_epilogue_fn` and `has_compute_fn` say which one
    is used, if any)."""
    _gemm_impl[
        transpose_b,
        wg_m,
        wg_n,
        32,
        64,
        2,
        8,
        K_STEP,
        transpose_b,
        has_epilogue_fn,
        has_compute_fn,
    ](c, a, b, Int(m), Int(n), Int(k), Float32(1), epilogue_fn, compute_fn)


def gemm_kernel_f8_widened[
    c_type: DType, wg_n: Int = 256
](
    c: MutPointer[Scalar[c_type], MutAnyOrigin],
    a: MutPointer[Scalar[DType.bfloat16], MutAnyOrigin],
    b: MutPointer[Scalar[DType.bfloat16], MutAnyOrigin],
    m: Int64,
    n: Int64,
    k: Int64,
    scale: Float32,
):
    """`gemm_kernel_f8` on a B^T that `f8_widen_kernel` widened to bfloat16
    (the values / 256): the bfloat16 GEMM, its accumulators times `scale`
    (times `F8_UNIT`) before the store."""
    _gemm_impl[
        True,
        256,
        wg_n,
        32,
        64,
        2,
        8,
        K_STEP,
        True,
        False,
        False,
        True,
    ](c, a, b, Int(m), Int(n), Int(k), scale, no_epilogue_fn, no_compute_fn)


comptime F8_WIDEN_WORDS = 8
"""Words of 4 float8_e4m3fn values `f8_widen_kernel` widens per work-item."""


def f8_widen_kernel(
    dst: MutPointer[Scalar[DType.bfloat16], MutAnyOrigin],
    src: MutPointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin],
    count: Int64,
):
    """Widens `count` float8_e4m3fn values (a multiple of 32) to bfloat16
    values / 256 (`_f8_to_bf16_pairs`), 32 per work-item: launch
    `ceildiv(count, 32 * block)` work-groups of `block` work-items."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i * 4 * F8_WIDEN_WORDS >= Int(count):
        return
    var words = src.unsafe_bitcast[UInt32]().unsafe_load[width=F8_WIDEN_WORDS](
        i * F8_WIDEN_WORDS
    )
    dst.unsafe_bitcast[UInt32]().unsafe_store(
        i * 2 * F8_WIDEN_WORDS, _f8_to_bf16_pairs(words)
    )


@inline(.always)
def _check_types[
    a_type: DType, b_type: DType, c_type: DType, transpose_b: Bool
]():
    """The kernels' types: float16 or bfloat16 A and B of one type, or
    bfloat16 A with a float8_e4m3fn B^T; C float32 or A's type."""
    comptime assert (
        (a_type == DType.float16 or a_type == DType.bfloat16)
        and b_type == a_type
    ) or (
        a_type == DType.bfloat16
        and b_type == DType.float8_e4m3fn
        and transpose_b
    ), (
        "Intel GEMM: A and B are float16 or bfloat16, or B^T is"
        " float8_e4m3fn with a bfloat16 A"
    )
    comptime assert (
        c_type == DType.float32 or c_type == a_type
    ), "Intel GEMM: C is float32 or the input type"


@inline(.always)
def _scale_acc[
    apply: Bool, n: Int
](
    acc: MutPointer[Scalar[DType.float32], _, address_space=AddressSpace.LOCAL],
    scale: Float32,
):
    """With `apply` (FP8 weights), multiplies the `n` accumulators by the
    weight scale (times `F8_UNIT`); nothing otherwise."""
    comptime if apply:
        var f = scale * F8_UNIT
        comptime for i in range(n):
            acc.unsafe_store(8 * i, acc.unsafe_load[width=8](8 * i) * f)


comptime SKINNY_MAX_M = 16
"""Largest M `skinny_kernel` takes by default (two DPAS row blocks)."""

comptime SKINNY_WIDE_MAX_M = 64
"""Largest M of the wider skinny tiles (`sg_m` 32 and 64: batched decode)."""


def skinny_wide_ks[transpose_b: Bool, sg_m: Int]() -> Int:
    """Sub-groups per work-group of the wide skinny tiles: `skinny_ks`, or half
    of it for B^T at 64 rows, which keeps the partial tiles in shared local
    memory at 64 KiB."""
    comptime if transpose_b and sg_m == 64:
        return 8
    return skinny_ks[transpose_b]()


comptime SKINNY_SG_N = 32
"""Columns of C per `skinny_kernel` work-group (and sub-group)."""


def skinny_ks[transpose_b: Bool]() -> Int:
    """Sub-groups per `skinny_kernel` work-group, splitting K between them:
    4 for B row-major, 16 for B^T (measured on the B70, with
    `skinny_prefetch`)."""
    return 16 if transpose_b else 4


def skinny_prefetch[transpose_b: Bool]() -> Int:
    """`skinny_kernel`'s prefetch distance: 1 for B row-major, none for B^T
    (prefetching B^T helps only while B fits in L2)."""
    return 0 if transpose_b else 1


def skinny_supported[
    a_type: DType, c_type: DType
](m: Int, n: Int, k: Int) -> Bool:
    """Whether `skinny_kernel` takes this shape: `gemm_supported`'s and
    M <= `SKINNY_MAX_M`."""
    return m <= SKINNY_MAX_M and gemm_supported[a_type, c_type](m, n, k)


def skinny_grid[sg_n: Int = SKINNY_SG_N](n: Int) -> Int:
    """Work-groups for an output of N columns."""
    return ceildiv(n, sg_n)


def skinny_block[ks: Int]() -> Int:
    """Work-items per work-group: 16 per sub-group."""
    return ks * 16


@inline(.always)
def _skinny_impl[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    transpose_b: Bool,
    sg_m: Int,
    sg_n: Int,
    ks: Int,
    prefetch: Int,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    c: MutPointer[Scalar[c_type], MutAnyOrigin],
    a: MutPointer[Scalar[a_type], MutAnyOrigin],
    b: MutPointer[Scalar[b_type], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    scale: Float32,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
):
    _check_types[a_type, b_type, c_type, transpose_b]()
    comptime assert (
        sg_m == 8 or sg_m == 16 or sg_m == 32 or sg_m == 64
    ), "Intel skinny GEMM: sg_m is 8, 16, 32 or 64"
    comptime assert sg_n % 32 == 0, "Intel GEMM: sg_n is a multiple of 32"
    comptime rows = sg_m // 8
    comptime cols = sg_n // 16
    # Outputs per sub-group: (rows * cols) accumulators x 8 elements x 16
    # lanes, which the ks * 16 work-items reduce `per` each.
    comptime tile = rows * cols * 8 * 16
    comptime per = tile // (ks * 16)
    comptime assert (
        per >= 1 and tile % (ks * 16) == 0
    ), "Intel skinny GEMM: ks * 16 divides sg_m * sg_n"

    var sg = Int(thread_idx.x) // 16
    var n0 = Int(block_idx.x) * sg_n
    var steps = ceildiv(k, K_STEP)

    var acc = stack_allocation[
        rows * cols * 8, DType.float32, address_space=AddressSpace.LOCAL
    ]()
    comptime for i in range(rows * cols):
        acc.unsafe_store(8 * i, SIMD[DType.float32, 8](0))

    # Sub-group sg takes K steps sg, sg + ks, ...: at any time the
    # work-group reads ks adjacent steps of its columns. It prefetches its
    # own B tile `prefetch` of its steps ahead (A is small and stays cached).
    comptime for d in range(prefetch):
        _skinny_prefetch[transpose_b, sg_n](b, n, k, n0, (sg + d * ks) * K_STEP)
    var s = sg
    while s < steps:
        comptime if prefetch > 0:
            _skinny_prefetch[transpose_b, sg_n](
                b, n, k, n0, (s + prefetch * ks) * K_STEP
            )
        _k_step[transpose_b, sg_m, sg_n](acc, a, b, m, n, k, s * K_STEP, 0, n0)
        s += ks
    _scale_acc[b_type == DType.float8_e4m3fn, rows * cols](acc, scale)
    _skinny_store[rows, cols, ks, has_epilogue_fn, has_compute_fn](
        c, acc, m, n, n0, epilogue_fn, compute_fn
    )


@inline(.always)
def _skinny_store[
    c_type: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    rows: Int,
    cols: Int,
    ks: Int,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    c: MutPointer[Scalar[c_type], MutAnyOrigin],
    acc: MutPointer[Scalar[DType.float32], _, address_space=AddressSpace.LOCAL],
    m: Int,
    n: Int,
    n0: Int,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
):
    """Stores a skinny work-group's tile at columns `n0`: each sub-group's
    `rows * cols` accumulators directly when `ks` is 1, else the sum of the
    `ks` sub-groups' partial tiles."""
    comptime tile = rows * cols * 8 * 16
    comptime per = tile // (ks * 16)
    var sg = Int(thread_idx.x) // 16
    comptime if ks == 1:
        _store_tile[rows, cols, has_epilogue_fn, has_compute_fn](
            c, acc, m, n, 0, n0, epilogue_fn, compute_fn
        )
    else:
        # Sum the ks partial tiles in shared local memory: sub-group s's
        # element e of accumulator i, lane l at s * tile + (8 i + e) * 16 + l.
        var red = stack_allocation[
            ks * tile, DType.float32, address_space=AddressSpace.SHARED
        ]()
        var lane = Int(thread_idx.x) % 16
        comptime for i in range(rows * cols):
            var v = acc.unsafe_load[width=8](8 * i)
            comptime for e in range(8):
                red.unsafe_store(sg * tile + (8 * i + e) * 16 + lane, v[e])
        barrier()
        comptime for w in range(per):
            var j = Int(thread_idx.x) + w * ks * 16
            var sum = red.unsafe_load(j)
            comptime for p in range(1, ks):
                sum += red.unsafe_load(p * tile + j)
            var blk = j // 128
            var row = 8 * (blk // cols) + (j // 16) % 8
            var col = n0 + 16 * (blk % cols) + j % 16
            if row < m and col < n:
                _store_one[has_epilogue_fn, has_compute_fn](
                    c, n, row, col, sum.cast[c_type](), epilogue_fn, compute_fn
                )


@inline(.always)
def _skinny_prefetch[
    b_type: DType, //, transpose_b: Bool, sg_n: Int
](
    b: MutPointer[Scalar[b_type], MutAnyOrigin],
    n: Int,
    k: Int,
    n0: Int,
    kp: Int,
):
    """Prefetches a sub-group's B tile (32 K x `sg_n` columns) at K `kp`."""
    comptime if transpose_b:
        # B^T rows n0 .. n0 + sg_n, 32 K columns each.
        comptime for j in range(sg_n // 32):
            block_prefetch[16, 32, 2](b, k, n, k, kp, n0 + 32 * j)
    else:
        comptime for j in range(sg_n // 32):
            block_prefetch[16, 32, 2](b, n, k, n, n0 + 32 * j, kp)


def skinny_kernel[
    a_type: DType,
    c_type: DType,
    transpose_b: Bool = False,
    sg_m: Int = 16,
    sg_n: Int = SKINNY_SG_N,
    ks: Int = skinny_ks[transpose_b](),
    prefetch: Int = skinny_prefetch[transpose_b](),
](
    c: MutPointer[Scalar[c_type], MutAnyOrigin],
    a: MutPointer[Scalar[a_type], MutAnyOrigin],
    b: MutPointer[Scalar[a_type], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    """`C = A * B` for M <= `sg_m` (decode): each work-group of `ks`
    sub-groups computes `sg_m` x `sg_n` of C, its sub-groups taking every
    `ks`-th K step of 32 (each its own DPAS tile, `gemm_kernel`'s loads),
    then summing their tiles in shared local memory. Launch
    `skinny_grid[sg_n](n)` work-groups of `skinny_block[ks]()` work-items
    (defaults: `skinny_grid(n)`, `skinny_block[skinny_ks[transpose_b]()]()`).

    Parameters:
        a_type: float16 or bfloat16, the type of A and B.
        c_type: float32 or `a_type`.
        transpose_b: B is given as its N x K transpose.
        sg_m: Rows of C computed (8 or 16; M <= sg_m).
        sg_n: Columns of C per work-group (a multiple of 32).
        ks: Sub-groups per work-group, splitting K.
        prefetch: Steps of a sub-group to prefetch B ahead; 0 for none.

    Args:
        c: C, M x N, row-major.
        a: A, M x K, row-major.
        b: B, K x N row-major, or N x K row-major with `transpose_b`.
        m: M.
        n: N.
        k: K.
    """
    _skinny_impl[transpose_b, sg_m, sg_n, ks, prefetch, False, False](
        c, a, b, m, n, k, Float32(1), no_epilogue_fn, no_compute_fn
    )


def skinny_kernel_fn[
    a_type: DType,
    c_type: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    transpose_b: Bool,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
    sg_m: Int = 16,
    ks: Int = skinny_ks[transpose_b](),
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
    """`skinny_kernel` with its defaults and MAX's output closures."""
    _skinny_impl[
        transpose_b,
        sg_m,
        SKINNY_SG_N,
        ks,
        skinny_prefetch[transpose_b](),
        has_epilogue_fn,
        has_compute_fn,
    ](c, a, b, Int(m), Int(n), Int(k), Float32(1), epilogue_fn, compute_fn)


def skinny_kernel_f8[
    c_type: DType,
    sg_m: Int = 16,
    sg_n: Int = SKINNY_SG_N,
    ks: Int = skinny_ks[True](),
](
    c: MutPointer[Scalar[c_type], MutAnyOrigin],
    a: MutPointer[Scalar[DType.bfloat16], MutAnyOrigin],
    b: MutPointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin],
    m: Int64,
    n: Int64,
    k: Int64,
    scale: Float32,
):
    """`C = scale * A * B` for M <= `sg_m` with bfloat16 A and
    float8_e4m3fn B^T (N x K) (see the module): `skinny_kernel`'s B^T
    defaults, launched with `skinny_grid(n)` x `skinny_block[ks]()`."""
    _skinny_impl[True, sg_m, sg_n, ks, 0, False, False](
        c, a, b, Int(m), Int(n), Int(k), scale, no_epilogue_fn, no_compute_fn
    )


comptime F4_UNIT = Float32(16384)
"""`_f4_to_bf16_pairs` gives each E2M1 value times its block scale / 2**22:
the value / 2**14 and the E4M3 scale / 256. With the accumulators scaled by
`F8_UNIT` (256, `_scale_acc`) the FP4 kernels take the weight scale times
this (2**14) as their scale."""


def gemm_f4_supported[c_type: DType](m: Int, n: Int, k: Int) -> Bool:
    """Whether the FP4-weight kernels take this shape (bfloat16 A, B^T of
    packed E2M1 codes with E4M3 block scales per 16, `c_type` C):
    `gemm_supported`'s rules for A and C, and K a multiple of the FP4
    skinny kernel's step (64) and at least 128 (B^T's rows of K / 2 bytes
    at least 64 bytes wide)."""
    return (
        gemm_supported[DType.bfloat16, c_type](m, n, k)
        and k % 64 == 0
        and k >= 128
    )


@inline(.always)
def _e4m3_scale(byte: UInt8) -> Float32:
    """A float8_e4m3fn block scale / 256, its sign dropped (MAX's NVFP4
    dequantization takes the scale's magnitude): the exponent and mantissa
    bits placed as float16's give the value / 256 exactly."""
    var h = (byte & 0x7F).cast[DType.uint16]() << 7
    return bitcast[DType.float16, 1](h).cast[DType.float32]()


@inline(.always)
def _f4_to_bf16_pairs(
    words: SIMD[DType.uint32, 4], s0: Float32, s1: Float32
) -> SIMD[DType.uint32, 16]:
    """Widens 4 words of 8 E2M1 codes each (K order from the low nibble of
    the low byte) to the 16 words of bfloat16 pairs (K order from the low
    half) of the values times their block scale (`s0` for K 0-15, `s1` for
    K 16-31) / 2**14, exactly: a code's sign, exponent and mantissa placed
    as float16's give the value / 2**14 (its subnormal 0.5 included), the
    hardware widens float16 to float32, and the product with an E4M3 scale
    has at most 6 significant bits, so its upper half is that bfloat16."""
    var nib = SIMD[DType.uint16, 32]()
    comptime for w in range(4):
        comptime for j in range(8):
            nib[8 * w + j] = ((words[w] >> UInt32(4 * j)) & 0xF).cast[
                DType.uint16
            ]()
    var h = ((nib & 0x8) << 12) | ((nib & 0x7) << 9)
    var f = bitcast[DType.float16, 32](h).cast[DType.float32]()
    var sc = SIMD[DType.float32, 32](s1)
    comptime for e in range(16):
        sc[e] = s0
    var u = bitcast[DType.uint32, 32](f * sc)
    var even, odd = u.deinterleave()
    return (even >> 16) | (odd & 0xFFFF0000)


@inline(.always)
def _f4_word_to_bf16_pairs(
    word: UInt32, scale: Float32
) -> SIMD[DType.uint32, 4]:
    """`_f4_to_bf16_pairs` for one word of 8 E2M1 codes sharing one block
    scale: the 4 words of bfloat16 pairs of the values times `scale` /
    2**14 (K order from the low nibble, and the low half)."""
    comptime shifts = SIMD[DType.uint32, 8](0, 4, 8, 12, 16, 20, 24, 28)
    var nib = ((SIMD[DType.uint32, 8](word) >> shifts) & 0xF).cast[
        DType.uint16
    ]()
    var h = ((nib & 0x8) << 12) | ((nib & 0x7) << 9)
    var f = bitcast[DType.float16, 8](h).cast[DType.float32]() * scale
    var u = bitcast[DType.uint32, 8](f)
    var even, odd = u.deinterleave()
    return (even >> 16) | (odd & 0xFFFF0000)


comptime F4_K_STEP = 64
"""K per step of the FP4 skinny kernel: one transposed load of 8 words of 8
E2M1 codes covers 64 values of each B^T row."""


@inline(.always)
def _k_step_f4[
    sg_m: Int, sg_n: Int
](
    acc: MutPointer[Scalar[DType.float32], _, address_space=AddressSpace.LOCAL],
    a: MutPointer[Scalar[DType.bfloat16], MutAnyOrigin],
    b: MutPointer[Scalar[DType.uint8], MutAnyOrigin],
    scales: MutPointer[Scalar[DType.uint8], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    kk: Int,
    n0: Int,
):
    """One `F4_K_STEP` (64) of a sub-group's tile with an FP4 B^T: per 32
    columns of C one transposed load of 8 words of 8 E2M1 codes covers the
    step (lane l gets B^T rows l and l + 16, element 2c + i is row l + 16i,
    word c: words 0-3 are K 0-31, words 4-7 K 32-63), and each row's four
    block scales of the step (one aligned 4-byte load) widen each 32-value
    half to the bfloat16 pairs of its two 16-value halves
    (`_f4_to_bf16_pairs`), multiplied by the A of that half."""
    comptime rows = sg_m // 8
    comptime cols = sg_n // 16
    comptime a_vals = 2 * sg_m
    comptime a_load = 32 if sg_m > 32 else sg_m
    # Both halves' A: half t at offset t * a_vals.
    var avs = SIMD[DType.bfloat16, 2 * a_vals]()
    comptime for t in range(2):
        comptime for c in range(sg_m // a_load):
            var part = rebind[SIMD[DType.bfloat16, 2 * a_load]](
                block_load[16, a_load, 2](
                    a, k, m, k, kk + K_STEP * t, a_load * c
                )
            )
            comptime for h in range(2):
                avs = avs.insert[offset=t * a_vals + h * sg_m + a_load * c](
                    part.slice[a_load, offset=h * a_load]()
                )
    var bw = b.unsafe_bitcast[UInt32]()
    var k8 = k // 8
    var k16 = k // 16
    var lane = Int(thread_idx.x) % 16
    comptime for j in range(sg_n // 32):
        var v = rebind[SIMD[DType.uint32, 16]](
            block_load_transpose[8, 32](bw, k8, n, k8, kk // 8, n0 + 32 * j)
        )
        comptime for i in range(2):
            # The row's four scales of the step in one aligned load: K and
            # `kk` are multiples of 64. Rows past N read row N - 1 (their
            # columns aren't stored).
            var row = min(n0 + 32 * j + lane + 16 * i, n - 1)
            var quad = scales.unsafe_bitcast[UInt32]().unsafe_load(
                (row * k16 + kk // 16) // 4
            )
            comptime cb = 2 * j + i
            comptime for t in range(2):
                var words = SIMD[DType.uint32, 4]()
                comptime for w in range(4):
                    words[w] = v[2 * (4 * t + w) + i]
                var pairs = _f4_to_bf16_pairs(
                    words,
                    _e4m3_scale(
                        ((quad >> UInt32(16 * t)) & 0xFF).cast[DType.uint8]()
                    ),
                    _e4m3_scale(
                        ((quad >> UInt32(16 * t + 8)) & 0xFF).cast[
                            DType.uint8
                        ]()
                    ),
                )
                comptime for h in range(2):
                    var bp = pairs.slice[8, offset=8 * h]()
                    comptime for r in range(rows):
                        var af = avs.slice[
                            8, offset=t * a_vals + h * sg_m + 8 * r
                        ]()
                        comptime ai = 8 * (r * cols + cb)
                        var d = SIMD[DType.float32, 8]()
                        mma(d, af, bp, acc.unsafe_load[width=8](ai))
                        acc.unsafe_store(ai, d)


def skinny_kernel_f4[
    c_type: DType,
    sg_m: Int = 16,
    sg_n: Int = SKINNY_SG_N,
    ks: Int = skinny_ks[True](),
](
    c: MutPointer[Scalar[c_type], MutAnyOrigin],
    a: MutPointer[Scalar[DType.bfloat16], MutAnyOrigin],
    b: MutPointer[Scalar[DType.uint8], MutAnyOrigin],
    scales: MutPointer[Scalar[DType.uint8], MutAnyOrigin],
    m: Int64,
    n: Int64,
    k: Int64,
    scale: Float32,
):
    """`C = scale * A * B` for M <= `sg_m` with bfloat16 A and an FP4 B^T
    (N x K / 2 bytes of E2M1 codes, N x K / 16 E4M3 block scales); `scale`
    is the weight scale times `F4_UNIT`. Launched with `skinny_grid(n)` x
    `skinny_block[ks]()`; sub-group sg takes K steps sg, sg + ks, ...."""
    comptime rows = sg_m // 8
    comptime cols = sg_n // 16
    var sg = Int(thread_idx.x) // 16
    var n0 = Int(block_idx.x) * sg_n
    var kk = Int(k)
    var steps = kk // F4_K_STEP
    var acc = stack_allocation[
        rows * cols * 8, DType.float32, address_space=AddressSpace.LOCAL
    ]()
    comptime for i in range(rows * cols):
        acc.unsafe_store(8 * i, SIMD[DType.float32, 8](0))
    var s = sg
    while s < steps:
        _k_step_f4[sg_m, sg_n](
            acc, a, b, scales, Int(m), Int(n), kk, s * F4_K_STEP, n0
        )
        s += ks
    _scale_acc[True, rows * cols](acc, scale)
    _skinny_store[rows, cols, ks, False, False](
        c, acc, Int(m), Int(n), n0, no_epilogue_fn, no_compute_fn
    )


comptime F4_WIDEN_WORDS = 4
"""Words of 8 E2M1 codes `f4_widen_kernel` widens per work-item."""


def f4_widen_kernel(
    dst: MutPointer[Scalar[DType.bfloat16], MutAnyOrigin],
    src: MutPointer[Scalar[DType.uint8], MutAnyOrigin],
    scales: MutPointer[Scalar[DType.uint8], MutAnyOrigin],
    count: Int64,
):
    """Widens `count` E2M1 values (a multiple of 32, two per byte of `src`,
    with one E4M3 scale per 16 consecutive values in `scales`) to bfloat16
    values times their scale / 2**22. A work-item takes `F4_WIDEN_WORDS`
    words of 8 codes, a work-group apart, so adjacent work-items read
    adjacent words and write adjacent 16-byte runs of pairs: launch
    `ceildiv(count, 8 * F4_WIDEN_WORDS * block)` work-groups of `block`
    work-items."""
    var block = Int(block_dim.x)
    var words = Int(count) // 8
    var base = Int(block_idx.x) * block * F4_WIDEN_WORDS + Int(thread_idx.x)
    var src32 = src.unsafe_bitcast[UInt32]()
    var dst32 = dst.unsafe_bitcast[UInt32]()
    comptime for t in range(F4_WIDEN_WORDS):
        var w = base + t * block
        if w < words:
            # A word's 8 values lie in one 16-value block: scale w / 2.
            dst32.unsafe_store(
                4 * w,
                _f4_word_to_bf16_pairs(
                    src32.unsafe_load(w),
                    _e4m3_scale(scales.unsafe_load(w // 2)),
                ),
            )
