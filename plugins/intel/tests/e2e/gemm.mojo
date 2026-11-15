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

# The Intel GEMM (`linalg.matmul.gpu.intel`) through the HAL client,
# against the CPU. For float16 and bfloat16 inputs, float32 and 16-bit
# outputs, and B row-major and transposed (8 kernels), on:
#   small integers in [-2, 2]: every output exact (the float32 sums are
#     exact; a 16-bit output must equal the exact sum rounded once);
#   random values in [-1, 1]: |c - ref| <= tol * sum|a b| with ref and the
#     sum in float64 from the rounded inputs, tol 2^-10 for float32 outputs
#     and 2^-7 for 16-bit ones (oneMKL's check).
# Shapes: square, transformer-layer (512x4096x4096, 16x4096x4096) and edges where M, N and
# K are not multiples of the 256 x 256 x 32 tiles (all outputs checked up to
# 999x520x328; 4,096 sampled outputs above). Every case runs both tile
# classes: 256 x 256 work-groups ("wg 256") and 256 x SMALL_WG_N
# ("wg 128", what gemm_small_tiles picks for small outputs), each with the
# kernel's defaults (the split barrier for B^T). Cases with M <= 16 also run
# the skinny kernel ("skinny 16", and "skinny 8" for M <= 8). Prints
# PASS/FAIL per case.
# run-e2e: packages //max/kernels/src/linalg:linalg

from std.math import ceildiv
from std.memory import alloc
from std.sys import size_of

from linalg.matmul.gpu.intel import (
    SKINNY_MAX_M,
    SMALL_WG_N,
    gemm_block,
    gemm_grid,
    gemm_kernel,
    gemm_supported,
    skinny_block,
    skinny_grid,
    skinny_kernel,
    skinny_ks,
)

from hal_gpu import GpuContext, GpuQueue, launch, load, open_context
from max._driver.max_license.hal.plugin import FunctionHandle


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

    def small(mut self) -> Float32:
        """An integer in [-2, 2]."""
        return Float32(Int((self.next() >> 33) % 5) - 2)


def run_case[
    a_type: DType, c_type: DType, transpose_b: Bool, wg_n: Int, skinny: Int = 0
](
    mut ctx: GpuContext,
    queue: GpuQueue,
    f: FunctionHandle,
    m: Int,
    n: Int,
    k: Int,
    small: Bool,
) raises -> Bool:
    if not gemm_supported[a_type, c_type](m, n, k):
        raise Error("shape not supported")
    var ha = alloc[Scalar[a_type]]({length = m * k}).unsafe_leak()
    var hb = alloc[Scalar[a_type]]({length = k * n}).unsafe_leak()
    var hc = alloc[Scalar[c_type]]({length = m * n}).unsafe_leak()
    # The rounded inputs as float32, B row-major (K x N), for the reference.
    var fa = alloc[Float32]({length = m * k}).unsafe_leak()
    var fb = alloc[Float32]({length = k * n}).unsafe_leak()
    var rng = Rng(UInt64(m * 1000003 + n * 1009 + k))
    for i in range(m * k):
        var v = (rng.small() if small else rng.uniform()).cast[a_type]()
        ha[unsafe_offset=i] = v
        fa[unsafe_offset=i] = v.cast[DType.float32]()
    for p in range(k):
        for j in range(n):
            var v = (rng.small() if small else rng.uniform()).cast[a_type]()
            fb[unsafe_offset=p * n + j] = v.cast[DType.float32]()
            comptime if transpose_b:
                hb[unsafe_offset=j * k + p] = v
            else:
                hb[unsafe_offset=p * n + j] = v
    comptime ab = size_of[Scalar[a_type]]()
    comptime cb = size_of[Scalar[c_type]]()
    var da = ctx.alloc_sync(UInt64(m * k * ab))
    var db = ctx.alloc_sync(UInt64(k * n * ab))
    var dc = ctx.alloc_sync(UInt64(m * n * cb))
    queue.copy_to_device(da.view(), ha.unsafe_bitcast[UInt8]())
    queue.copy_to_device(db.view(), hb.unsafe_bitcast[UInt8]())
    # NaN everywhere, so an output the kernel misses fails the check.
    queue.set_memory(dc.view(), 0xFF)
    # skinny: the skinny kernel's sg_m (8 or 16), or 0 for gemm_kernel.
    launch(
        queue,
        f,
        skinny_grid(n) if skinny else gemm_grid[256, wg_n](m, n),
        skinny_block[skinny_ks[transpose_b]()]() if skinny else gemm_block[
            256, wg_n
        ](),
        [
            ctx.memory_get_address(dc),
            ctx.memory_get_address(da),
            ctx.memory_get_address(db),
            UInt64(m),
            UInt64(n),
            UInt64(k),
        ],
    )
    queue.copy_from_device(hc.unsafe_bitcast[UInt8](), dc.view())
    queue.synchronize()

    # Every output up to 999x520x328, else 4,096 sampled ones.
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
            var x = Float64(fa[unsafe_offset=i * k + p]) * Float64(
                fb[unsafe_offset=p * n + j]
            )
            exact += x
            mag += abs(x)
        var got = hc[unsafe_offset=i * n + j].cast[DType.float64]()
        if small:
            var want = Float32(exact).cast[c_type]().cast[DType.float64]()
            if got != want:
                if bad < 3:
                    print("  C[", i, ",", j, "] =", got, "want", want)
                bad += 1
        else:
            var e = abs(got - exact) / (mag if mag > 0 else 1.0)
            if not (e <= tol):
                if bad < 3:
                    print("  C[", i, ",", j, "] =", got, "ref", exact)
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
        String(" skinny ", skinny) if skinny else String(" wg ", wg_n),
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
](mut ctx: GpuContext, queue: GpuQueue) raises -> Int:
    comptime kern = gemm_kernel[a_type, c_type, transpose_b]
    var f = load[type_of(kern), kern](ctx)
    comptime kern_s = gemm_kernel[a_type, c_type, transpose_b, wg_n=SMALL_WG_N]
    var fs = load[type_of(kern_s), kern_s](ctx)
    comptime kern_k16 = skinny_kernel[a_type, c_type, transpose_b]
    var fk16 = load[type_of(kern_k16), kern_k16](ctx)
    comptime kern_k8 = skinny_kernel[a_type, c_type, transpose_b, sg_m=8]
    var fk8 = load[type_of(kern_k8), kern_k8](ctx)
    var fails = 0
    # (m, n, k, small integers)
    var cases: List[Tuple[Int, Int, Int, Bool]] = [
        (256, 256, 256, True),
        (999, 520, 328, True),
        (33, 40, 40, True),
        (999, 520, 328, False),
        (1024, 1024, 1024, False),
        (1000, 1000, 1000, True),
        (512, 4096, 4096, False),
        (16, 4096, 4096, False),
        (1, 4096, 4096, False),
        (7, 520, 328, True),
        (13, 40, 40, True),
        (16, 1000, 1000, True),
    ]
    for c in cases:
        if not run_case[a_type, c_type, transpose_b, 256](
            ctx, queue, f, c[0], c[1], c[2], c[3]
        ):
            fails += 1
        if not run_case[a_type, c_type, transpose_b, SMALL_WG_N](
            ctx, queue, fs, c[0], c[1], c[2], c[3]
        ):
            fails += 1
        if c[0] <= SKINNY_MAX_M and not run_case[
            a_type, c_type, transpose_b, 256, 16
        ](ctx, queue, fk16, c[0], c[1], c[2], c[3]):
            fails += 1
        if c[0] <= 8 and not run_case[a_type, c_type, transpose_b, 256, 8](
            ctx, queue, fk8, c[0], c[1], c[2], c[3]
        ):
            fails += 1
    return fails


def main() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var fails = 0
    # gemm_supported's 2D block I/O limits (SPV_INTEL_2d_block_io: at most
    # 2^24 bytes wide and 2^24 rows): in bytes, so a float32 C allows N up
    # to 2^22 and 16-bit A and B allow N, K up to 2^23.
    var limits: List[Tuple[Bool, Bool, String]] = [
        (
            gemm_supported[DType.float16, DType.float32](16, 1 << 22, 64),
            True,
            "f32 C, N = 2^22",
        ),
        (
            gemm_supported[DType.float16, DType.float32](16, (1 << 22) + 8, 64),
            False,
            "f32 C, N = 2^22 + 8",
        ),
        (
            gemm_supported[DType.float16, DType.float16](16, (1 << 22) + 8, 64),
            True,
            "f16 C, N = 2^22 + 8",
        ),
        (
            gemm_supported[DType.bfloat16, DType.bfloat16](
                16, (1 << 23) + 8, 64
            ),
            False,
            "bf16, N = 2^23 + 8",
        ),
        (
            gemm_supported[DType.float16, DType.float32](16, 64, 1 << 23),
            True,
            "f16 A, K = 2^23",
        ),
        (
            gemm_supported[DType.float16, DType.float32](16, 64, (1 << 23) + 8),
            False,
            "f16 A, K = 2^23 + 8",
        ),
        (
            gemm_supported[DType.float16, DType.float32]((1 << 24) + 1, 64, 64),
            False,
            "M = 2^24 + 1",
        ),
    ]
    for t in limits:
        if t[0] == t[1]:
            print("PASS gemm_supported", t[2], t[0])
        else:
            print("FAIL gemm_supported", t[2], t[0], "expected", t[1])
            fails += 1
    fails += run_type[DType.float16, DType.float32, False](ctx[], queue[])
    fails += run_type[DType.float16, DType.float16, False](ctx[], queue[])
    fails += run_type[DType.bfloat16, DType.float32, False](ctx[], queue[])
    fails += run_type[DType.bfloat16, DType.bfloat16, False](ctx[], queue[])
    fails += run_type[DType.float16, DType.float32, True](ctx[], queue[])
    fails += run_type[DType.float16, DType.float16, True](ctx[], queue[])
    fails += run_type[DType.bfloat16, DType.float32, True](ctx[], queue[])
    fails += run_type[DType.bfloat16, DType.bfloat16, True](ctx[], queue[])
    print("gemm: failures", fails)
    if fails:
        raise Error("gemm: wrong results")
