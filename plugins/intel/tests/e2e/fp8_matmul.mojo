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

# The Intel FP8-weight kernels (bfloat16 A, float8_e4m3fn B^T, one
# float32 weight scale) through the HAL client, against the CPU:
# `skinny_kernel_f8` (M <= 16; 16 and 8 sub-groups) and, for larger M,
# `f8_widen_kernel` then `gemm_kernel_f8_widened` (as
# `matmul_f8_dispatch_intel` runs them):
#   every byte: A's rows one-hot, so C[r, j] = scale * e4m3(B^T[j, r]) for
#     all 256 codes but the two NaNs, exact (float32 C);
#   random A in [-1, 1] and random codes: |c - ref| <= tol * sum|a b| *
#     scale, the reference and the sum in float64 from the rounded inputs
#     and the codes' exact values (tol 2^-10 for float32 C, 2^-7 for
#     bfloat16), every output up to 520 x 336, else 4,096 sampled;
# at Qwen3.5-0.8B's FP8 projection shapes (in_proj_qkv + in_proj_z 8192 x
# 1024, out_proj and o_proj 1024 x 2048, q/k/v 5120 x 1024) for decode
# (M = 1, 4, 16: the skinny kernel, sg_m 16 and 8, 16 and 8 sub-groups) and
# prefill (M = 17, 128, 512: widen and GEMM, both tile classes; the GEMM
# path also at M <= 16), and edges (K = 80, not a multiple
# of 32; N = 40). Also gemm_f8_supported's limits. Prints PASS/FAIL per case.
# run-e2e: packages //max/kernels/src/linalg:linalg

from std.math import ceildiv
from std.memory import alloc, bitcast
from std.sys import size_of

from linalg.matmul.gpu.intel import (
    SKINNY_MAX_M,
    SMALL_WG_N,
    gemm_block,
    gemm_f8_supported,
    gemm_grid,
    F8_WIDEN_WORDS,
    f8_widen_kernel,
    gemm_kernel_f8_widened,
    skinny_block,
    skinny_grid,
    skinny_kernel_f8,
    skinny_ks,
)

from hal_gpu import GpuContext, GpuQueue, launch, load, open_context
from max._driver.max_license.hal.plugin import FunctionHandle

comptime bf16 = DType.bfloat16
comptime f8 = DType.float8_e4m3fn


struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed * 0x9E3779B97F4A7C15 + 1

    def next(mut self) -> UInt64:
        self.s ^= self.s >> 12
        self.s ^= self.s << 25
        self.s ^= self.s >> 27
        return self.s * 0x2545F4914F6CDD1D

    def uniform(mut self) -> Float32:
        """Uniform in [-1, 1]."""
        return Float32(Float64(self.next() >> 40) / 16777216.0) * 2.0 - 1.0

    def code(mut self) -> UInt8:
        """A float8_e4m3fn code, not NaN."""
        while True:
            var c = UInt8((self.next() >> 32) & 0xFF)
            if (c & 0x7F) != 0x7F:
                return c


def e4m3(c: UInt8) -> Float64:
    """float8_e4m3fn's value of a code (not NaN), from its definition: bias
    7, subnormals below exponent 1."""
    var e = Int((c >> 3) & 0xF)
    var mant = Float64(Int(c & 7))
    var v: Float64
    if e == 0:
        v = mant / 8.0 * 2.0**-6
    else:
        v = (1.0 + mant / 8.0) * 2.0 ** Float64(e - 7)
    return -v if (c & 0x80) else v


def skinny16[
    c_type: DType
](
    c: MutPointer[Scalar[c_type], MutAnyOrigin],
    a: MutPointer[Scalar[bf16], MutAnyOrigin],
    b: MutPointer[Scalar[f8], MutAnyOrigin],
    m: Int64,
    n: Int64,
    k: Int64,
    scale: UInt64,
):
    skinny_kernel_f8[c_type](
        c, a, b, m, n, k, bitcast[DType.float32](UInt32(scale))
    )


def skinny8[
    c_type: DType
](
    c: MutPointer[Scalar[c_type], MutAnyOrigin],
    a: MutPointer[Scalar[bf16], MutAnyOrigin],
    b: MutPointer[Scalar[f8], MutAnyOrigin],
    m: Int64,
    n: Int64,
    k: Int64,
    scale: UInt64,
):
    skinny_kernel_f8[c_type, sg_m=8](
        c, a, b, m, n, k, bitcast[DType.float32](UInt32(scale))
    )


def skinny16_ks8[
    c_type: DType
](
    c: MutPointer[Scalar[c_type], MutAnyOrigin],
    a: MutPointer[Scalar[bf16], MutAnyOrigin],
    b: MutPointer[Scalar[f8], MutAnyOrigin],
    m: Int64,
    n: Int64,
    k: Int64,
    scale: UInt64,
):
    skinny_kernel_f8[c_type, ks=8](
        c, a, b, m, n, k, bitcast[DType.float32](UInt32(scale))
    )


def gemm[
    c_type: DType, wg_n: Int
](
    c: MutPointer[Scalar[c_type], MutAnyOrigin],
    a: MutPointer[Scalar[bf16], MutAnyOrigin],
    b: MutPointer[Scalar[bf16], MutAnyOrigin],
    m: Int64,
    n: Int64,
    k: Int64,
    scale: UInt64,
):
    gemm_kernel_f8_widened[c_type, wg_n](
        c, a, b, m, n, k, bitcast[DType.float32](UInt32(scale))
    )


def run_case[
    c_type: DType, kind: Int
](
    mut ctx: GpuContext,
    queue: GpuQueue,
    f: FunctionHandle,
    widen: FunctionHandle,
    m: Int,
    n: Int,
    k: Int,
    one_hot: Bool,
    scale: Float32,
) raises -> Bool:
    """kind: 16 or 8 the skinny kernel's sg_m (161: sg_m 16 with 8
    sub-groups), 256 or SMALL_WG_N the widened GEMM's work-group columns."""
    if not gemm_f8_supported[c_type](m, n, k):
        raise Error("shape not supported")
    var ha = alloc[Scalar[bf16]]({length = m * k}).unsafe_leak()
    var hb = alloc[UInt8]({length = n * k}).unsafe_leak()
    var hc = alloc[Scalar[c_type]]({length = m * n}).unsafe_leak()
    var fa = alloc[Float64]({length = m * k}).unsafe_leak()
    var fb = alloc[Float64]({length = n * k}).unsafe_leak()
    var rng = Rng(UInt64(m * 1000003 + n * 1009 + k + kind))
    for i in range(m):
        for p in range(k):
            var v: Scalar[bf16]
            if one_hot:
                v = Scalar[bf16](1) if p == i else Scalar[bf16](0)
            else:
                v = rng.uniform().cast[bf16]()
            ha[unsafe_offset=i * k + p] = v
            fa[unsafe_offset=i * k + p] = v.cast[DType.float64]()
    for j in range(n):
        for p in range(k):
            var code: UInt8
            if one_hot:
                code = UInt8((j + 16 * p) & 0xFF)
                if (code & 0x7F) == 0x7F:
                    code = 0
            else:
                code = rng.code()
            hb[unsafe_offset=j * k + p] = code
            fb[unsafe_offset=j * k + p] = e4m3(code)
    var da = ctx.alloc_sync(UInt64(m * k * 2))
    var db = ctx.alloc_sync(UInt64(n * k))
    var dc = ctx.alloc_sync(UInt64(m * n * size_of[Scalar[c_type]]()))
    queue.copy_to_device(da.view(), ha.unsafe_bitcast[UInt8]())
    queue.copy_to_device(db.view(), hb)
    queue.set_memory(dc.view(), 0xFF)
    var skinny = kind == 16 or kind == 8 or kind == 161
    var grid: Int
    var block: Int
    var words_b = ctx.memory_get_address(db)
    if skinny:
        grid = skinny_grid(n)
        block = (
            skinny_block[8]() if kind
            == 161 else skinny_block[skinny_ks[True]()]()
        )
    elif kind == 256:
        grid = gemm_grid(m, n)
        block = gemm_block()
    else:
        grid = gemm_grid[256, SMALL_WG_N](m, n)
        block = gemm_block[256, SMALL_WG_N]()
    var dw = ctx.alloc_sync(UInt64(n * k * 2 if not skinny else 64))
    if not skinny:
        launch(
            queue,
            widen,
            ceildiv(n * k, 4 * F8_WIDEN_WORDS * 256),
            256,
            [ctx.memory_get_address(dw), words_b, UInt64(n * k)],
        )
        words_b = ctx.memory_get_address(dw)
    launch(
        queue,
        f,
        grid,
        block,
        [
            ctx.memory_get_address(dc),
            ctx.memory_get_address(da),
            words_b,
            UInt64(m),
            UInt64(n),
            UInt64(k),
            UInt64(bitcast[DType.uint32](scale)),
        ],
    )
    queue.copy_from_device(hc.unsafe_bitcast[UInt8](), dc.view())
    queue.synchronize()

    var full = m * n <= 520 * 336
    var count = m * n if full else 4096
    var srng = Rng(7)
    var bad = 0
    var worst = 0.0
    var tol = 1.0 / 1024.0 if c_type == DType.float32 else 1.0 / 128.0
    var s = Float64(scale)
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
            var x = fa[unsafe_offset=i * k + p] * fb[unsafe_offset=j * k + p]
            exact += x
            mag += abs(x)
        exact *= s
        mag *= s
        var got = hc[unsafe_offset=i * n + j].cast[DType.float64]()
        var ok: Bool
        var e = 0.0
        if one_hot:
            ok = got == Float32(exact).cast[c_type]().cast[DType.float64]()
        else:
            e = abs(got - exact) / (mag if mag > 0 else 1.0)
            ok = e <= tol
        if not ok:
            if bad < 3:
                print("  C[", i, ",", j, "] =", got, "ref", exact)
            bad += 1
            e = 1.0
        worst = max(worst, e)
    var name = String(
        "f8 ",
        c_type,
        " ",
        m,
        "x",
        n,
        "x",
        k,
        " one-hot" if one_hot else " random",
        String(" skinny ", kind) if skinny else String(" wg ", kind),
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
    ctx.free_sync(da^)
    ctx.free_sync(db^)
    ctx.free_sync(dc^)
    ctx.free_sync(dw^)
    return bad == 0


def run_type[c_type: DType](mut ctx: GpuContext, queue: GpuQueue) raises -> Int:
    comptime k16 = skinny16[c_type]
    var f16 = load[type_of(k16), k16](ctx)
    comptime k8 = skinny8[c_type]
    var f8_ = load[type_of(k8), k8](ctx)
    comptime kg = gemm[c_type, 256]
    var fg = load[type_of(kg), kg](ctx)
    comptime ks = gemm[c_type, SMALL_WG_N]
    var fs = load[type_of(ks), ks](ctx)
    comptime k16_8 = skinny16_ks8[c_type]
    var f16_8 = load[type_of(k16_8), k16_8](ctx)
    var fw = load[type_of(f8_widen_kernel), f8_widen_kernel](ctx)
    var fails = 0
    # Every code, one-hot rows (M = 16 reads columns 0-15 of B^T).
    if not run_case[c_type, 16](ctx, queue, f16, fw, 16, 256, 64, True, 1.0):
        fails += 1
    if not run_case[c_type, 256](ctx, queue, fg, fw, 16, 256, 64, True, 1.0):
        fails += 1
    if not run_case[c_type, SMALL_WG_N](
        ctx, queue, fs, fw, 16, 256, 64, True, 1.0
    ):
        fails += 1
    # (m, n, k)
    var cases: List[Tuple[Int, Int, Int]] = [
        (1, 8192, 1024),
        (1, 1024, 2048),
        (1, 5120, 1024),
        (4, 8192, 1024),
        (16, 1024, 2048),
        (3, 40, 80),
        (17, 8192, 1024),
        (128, 1024, 2048),
        (512, 5120, 1024),
        (300, 520, 336),
        (33, 40, 80),
    ]
    var scale = Float32(0.0123)
    for c in cases:
        var m = c[0]
        var n = c[1]
        var k = c[2]
        if m <= SKINNY_MAX_M:
            if not run_case[c_type, 16](
                ctx, queue, f16, fw, m, n, k, False, scale
            ):
                fails += 1
            if not run_case[c_type, 161](
                ctx, queue, f16_8, fw, m, n, k, False, scale
            ):
                fails += 1
            if m <= 8 and not run_case[c_type, 8](
                ctx, queue, f8_, fw, m, n, k, False, scale
            ):
                fails += 1
        if not run_case[c_type, 256](ctx, queue, fg, fw, m, n, k, False, scale):
            fails += 1
        if not run_case[c_type, SMALL_WG_N](
            ctx, queue, fs, fw, m, n, k, False, scale
        ):
            fails += 1
    return fails


def main() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var fails = 0
    var limits: List[Tuple[Bool, Bool, String]] = [
        (gemm_f8_supported[bf16](1, 1024, 64), True, "K = 64"),
        (gemm_f8_supported[bf16](1, 1024, 48), False, "K = 48"),
        (gemm_f8_supported[bf16](1, 1024, 72), False, "K = 72"),
        (gemm_f8_supported[bf16](1, 1028, 1024), False, "N = 1028"),
    ]
    for t in limits:
        if t[0] == t[1]:
            print("PASS gemm_f8_supported", t[2], t[0])
        else:
            print("FAIL gemm_f8_supported", t[2], t[0], "expected", t[1])
            fails += 1
    fails += run_type[DType.float32](ctx[], queue[])
    fails += run_type[bf16](ctx[], queue[])
    print("fp8_matmul: failures", fails)
    if fails:
        raise Error("fp8_matmul: wrong results")
