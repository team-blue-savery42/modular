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

# The generic GEVM of MAX's matmul (linalg.gemv's gevm_kernel: M = 1 with B
# row-major) on the B70 through `_matmul_gpu` and DeviceContext (the HAL
# compat layer), against the CPU. Its inputs are float32, which the Intel GEMM
# declines on every Intel target, so the B70 runs MAX's GEVM too. It launched
# blocks of 1024 // warp_size warps whose final lane-group sum spanned more
# lanes than a sub-group has; the block is now at most warp_size warps.
# Small integers in [-2, 2] give exact sums, so each output must equal the
# CPU's; outputs start as NaN. Prints PASS/FAIL per case.
# run-e2e: device-context
# run-e2e: packages //max/kernels/src/linalg:linalg

from std.memory import alloc
from std.utils.numerics import nan

from layout import Coord, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu
from max.gpu.host import DeviceContext

comptime PLAIN = 0
comptime EPILOGUE = 1
comptime COMPUTE = 2


struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed * 0x9E3779B97F4A7C15 + 1

    def next(mut self) -> UInt64:
        self.s ^= self.s >> 12
        self.s ^= self.s << 25
        self.s ^= self.s >> 27
        return self.s * 0x2545F4914F6CDD1D

    def value(mut self, small: Bool) -> Float32:
        """An integer in [-2, 2], or uniform in [-1, 1]."""
        if small:
            return Float32(Int((self.next() >> 33) % 5) - 2)
        return Float32(Float64(self.next() >> 40) / 16777216.0) * 2.0 - 1.0


def scale(row: Int) -> Int:
    return -2 if row % 2 == 1 else 2


def run_case[
    a_type: DType, c_type: DType, transpose_b: Bool, mode: Int = PLAIN
](
    ctx: DeviceContext,
    m: Int,
    n: Int,
    k: Int,
    small: Bool,
    a_offset: Int = 0,
) raises -> Bool:
    """`a_offset` elements of padding before A (an unaligned base)."""
    var ha = alloc[Scalar[a_type]]({length = m * k + a_offset}).unsafe_leak()
    var hb = alloc[Scalar[a_type]]({length = k * n}).unsafe_leak()
    var hc = alloc[Scalar[c_type]]({length = m * n}).unsafe_leak()
    var fa = alloc[Float64]({length = m * k}).unsafe_leak()
    var fb = alloc[Float64]({length = k * n}).unsafe_leak()
    var rng = Rng(UInt64(m * 1000003 + n * 1009 + k))
    for i in range(m * k):
        var v = rng.value(small).cast[a_type]()
        ha[unsafe_offset=a_offset + i] = v
        fa[unsafe_offset=i] = v.cast[DType.float64]()
    for p in range(k):
        for j in range(n):
            var v = rng.value(small).cast[a_type]()
            fb[unsafe_offset=p * n + j] = v.cast[DType.float64]()
            comptime if transpose_b:
                hb[unsafe_offset=j * k + p] = v
            else:
                hb[unsafe_offset=p * n + j] = v
    for i in range(m * n):
        hc[unsafe_offset=i] = nan[c_type]()

    var da = ctx.enqueue_create_buffer[a_type](m * k + a_offset)
    var db = ctx.enqueue_create_buffer[a_type](k * n)
    var dc = ctx.enqueue_create_buffer[c_type](m * n)
    for i in range(a_offset):
        ha[unsafe_offset=i] = 0
    ctx.enqueue_copy(da, ha)
    ctx.enqueue_copy(db, hb)
    ctx.enqueue_copy(dc, hc)
    var a = TileTensor(
        da.unsafe_ptr().unsafe_offset(a_offset), row_major(m, k)
    ).as_imm()
    var b_rows = n if transpose_b else k
    var b_cols = k if transpose_b else n
    var b = TileTensor(db.unsafe_ptr(), row_major(b_rows, b_cols)).as_imm()
    var c = TileTensor(dc.unsafe_ptr(), row_major(m, n))

    comptime if mode == EPILOGUE:
        # The epilogue stores every output; `c` is a shape over a null base.
        var null_c: Optional[
            MutPointer[Scalar[c_type], MutUntrackedOrigin]
        ] = None
        var c_shape = TileTensor(null_c._unsafe_nullable(), row_major(m, n))

        @inline(.always)
        def epilogue_fn[
            _dtype: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[_dtype, width]) {var c}:
            comptime assert idx.rank == 2
            var s = SIMD[_dtype, width](scale(Int(idx[0])))
            c.store(idx, (val * s).cast[c_type]())

        _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
            c_shape, a, b, epilogue_fn, ctx
        )
    elif mode == COMPUTE:

        def compute_fn[
            _dtype: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[_dtype, width]) -> SIMD[_dtype, width]:
            comptime assert idx.rank == 2
            return val * SIMD[_dtype, width](scale(Int(idx[0])))

        _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
            c, a, b, compute_fn, ctx
        )
    else:
        _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](c, a, b, ctx)
    ctx.enqueue_copy(hc, dc)
    ctx.synchronize()

    var full = m * n * k <= 999 * 520 * 328
    var count = m * n if full else 4096
    var srng = Rng(7)
    var bad = 0
    var worst = 0.0
    var tol = 1.0 / 1024.0 if c_type == DType.float32 else 1.0 / 128.0
    for t in range(count):
        var i: Int
        var j: Int
        if full:
            i = t // n
            j = t % n
        else:
            i = Int(srng.next() % UInt64(m))
            j = Int(srng.next() % UInt64(n))
        var exact = 0.0
        var mag = 0.0
        for p in range(k):
            var x = fa[unsafe_offset=i * k + p] * fb[unsafe_offset=p * n + j]
            exact += x
            mag += abs(x)
        var s = Float64(scale(i)) if mode != PLAIN else 1.0
        var got = hc[unsafe_offset=i * n + j].cast[DType.float64]()
        if small:
            # The kernels round the sum once to C; the closures scale that
            # by a power of two (exact).
            var want = Float32(exact).cast[c_type]().cast[DType.float64]() * s
            if got != want:
                if bad < 3:
                    print("  C[", i, ",", j, "] =", got, "want", want)
                bad += 1
        else:
            var e = abs(got - exact * s) / (abs(s) * mag if mag > 0 else 1.0)
            if not (e <= tol):
                if bad < 3:
                    print("  C[", i, ",", j, "] =", got, "ref", exact * s)
                bad += 1
                e = 1.0
            worst = max(worst, e)
    var name = String(
        a_type,
        "->",
        c_type,
        " B",
        "^T" if transpose_b else "",
        " ",
        m,
        "x",
        n,
        "x",
        k,
        " small" if small else " random",
        " epilogue" if mode
        == EPILOGUE else (" compute" if mode == COMPUTE else ""),
        String(" a+", a_offset) if a_offset else "",
    )
    if bad == 0:
        print("PASS", name, "checked", count, "worst", worst / tol, "of tol")
    else:
        print("FAIL", name, bad, "of", count, "wrong")
    ha.unsafe_free()
    hb.unsafe_free()
    hc.unsafe_free()
    fa.unsafe_free()
    fb.unsafe_free()
    return bad == 0


def run_type[
    a_type: DType, c_type: DType, transpose_b: Bool
](ctx: DeviceContext) raises -> Int:
    var fails = 0
    # (m, n, k, small integers)
    var cases: List[Tuple[Int, Int, Int, Bool]] = [
        (1, 4096, 4096, False),
        (7, 520, 328, True),
        (16, 4096, 4096, False),
        (16, 40, 40, True),
        (33, 40, 40, True),
        (999, 520, 328, True),
        (1024, 1024, 1024, False),
        (2048, 2048, 512, True),
    ]
    for c in cases:
        if not run_case[a_type, c_type, transpose_b](
            ctx, c[0], c[1], c[2], c[3]
        ):
            fails += 1
    return fails


def run_closures[
    a_type: DType, c_type: DType, transpose_b: Bool
](ctx: DeviceContext) raises -> Int:
    var fails = 0
    var cases: List[Tuple[Int, Int, Int]] = [
        (5, 520, 328),
        (16, 1000, 1000),
        (999, 520, 328),
        (2048, 2048, 512),
    ]
    for c in cases:
        if not run_case[a_type, c_type, transpose_b, EPILOGUE](
            ctx, c[0], c[1], c[2], True
        ):
            fails += 1
        if not run_case[a_type, c_type, transpose_b, COMPUTE](
            ctx, c[0], c[1], c[2], True
        ):
            fails += 1
    return fails


def main() raises:
    var fails = 0
    with DeviceContext() as ctx:
        for shape in [(1, 4096, 4096), (1, 256, 256), (1, 64, 64), (1, 16, 16)]:
            if not run_case[DType.float32, DType.float32, False](
                ctx, shape[0], shape[1], shape[2], True
            ):
                fails += 1
        if not run_case[DType.float32, DType.float32, True](
            ctx, 1, 4096, 4096, True
        ):
            fails += 1
    print("gevm: failures", fails)
    if fails:
        raise Error("gevm: wrong results")
