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

# MAX's test_mandelbrot on the B70: why its iteration count is not
# NVIDIA's. The test's kernel, as written, sums 4,096 x 4,096 float64
# iteration counts; its expected sum, 4,687,767,697, is what NVIDIA computes.
# Mojo marks `min_x + x * scale_x` and `min_y + row * scale_y` `contract`;
# NVPTX fuses each into one fma.rn.f64, while LLVM's SPIR-V backend emits
# ContractionOff (no `opencl.enable.FP_CONTRACT` metadata), so the B70
# rounds the product and the sum separately. The loop's own fmas are
# explicit (ComplexSIMD.squared_add, squared_norm) and fused on both.
# `contract` allows fusing but doesn't require it, so both sums are right.
#
# This runs the kernel twice: `as_written` (cx and cy as the test writes
# them) must give 4,687,759,364, the CPU's float64 sum with cx and cy
# rounded twice, and `fused` (cx and cy with an explicit fma) must give
# NVIDIA's 4,687,767,697, the CPU's sum with them fused. The CPU sums are
# from mandelbrot_ref.c in this directory (four variants, with and without
# fma).

from std.algorithm import vectorize
from std.complex import ComplexSIMD
from std.math import ceildiv, iota
from std.memory import alloc
from std.sys.info import simd_width_of

from hal_gpu import launch, load, open_context
from max.gpu import global_idx

comptime float_type = DType.float64
comptime int_type = DType.int
comptime width = 4096
comptime height = 4096
comptime MAX_ITERS = 1000
comptime BLOCK_SIZE = 32

comptime min_x = -2.0
comptime max_x = 0.47
comptime min_y = -1.12
comptime max_y = 1.12


@inline(.always)
def mandelbrot_kernel[
    simd_width: SIMDLength
](c: ComplexSIMD[float_type, simd_width]) -> SIMD[int_type, simd_width]:
    """The test's inner loop, unchanged."""
    var z = ComplexSIMD[float_type, simd_width](0, 0)
    var iters = SIMD[int_type, simd_width](0)

    var in_set_mask = SIMD[.bool, simd_width](fill=True)
    for _ in range(MAX_ITERS):
        if not in_set_mask.reduce_or():
            break
        in_set_mask = z.squared_norm().le(4)
        iters = in_set_mask.select(iters + 1, iters)
        z = z.squared_add(c)

    return iters


def mandelbrot[
    fused: Bool
](out_ptr: MutPointer[Scalar[int_type], MutAnyOrigin]):
    var row = global_idx.x
    if row >= height:
        return

    comptime scale_x = (max_x - min_x) / width
    comptime scale_y = (max_y - min_y) / height

    @inline(.always)
    def compute_vector[simd_width: Int](col: Int) {imm}:
        if col >= width:
            return
        var x = Scalar[float_type](col) + iota[float_type, simd_width]()
        var y = SIMD[float_type, 1](row)
        var cx: SIMD[float_type, simd_width]
        var cy: SIMD[float_type, simd_width]
        comptime if fused:
            cx = x.fma(
                SIMD[float_type, simd_width](scale_x),
                SIMD[float_type, simd_width](min_x),
            )
            cy = SIMD[float_type, simd_width](y).fma(
                SIMD[float_type, simd_width](scale_y),
                SIMD[float_type, simd_width](min_y),
            )
        else:
            # The test's expressions.
            cx = min_x + x * scale_x
            cy = min_y + y * SIMD[float_type, simd_width](scale_y)
        var c = ComplexSIMD[float_type, simd_width](cx, cy)
        out_ptr.unsafe_store(
            row * width + col, mandelbrot_kernel[simd_width](c)
        )

    vectorize[simd_width_of[float_type]()](width, compute_vector)


def run[fused: Bool](name: String, expected: Int) raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    comptime kernel = mandelbrot[fused]
    var function = load[type_of(kernel), kernel](ctx[])
    comptime count = width * height
    var out_dev = ctx[].alloc_sync(UInt64(8 * count))
    launch(
        queue[],
        function,
        ceildiv(height, BLOCK_SIZE),
        BLOCK_SIZE,
        [ctx[].memory_get_address(out_dev)],
    )
    var host = alloc[Scalar[int_type]]({length = count}).unsafe_leak()
    queue[].copy_from_device(host.unsafe_bitcast[UInt8](), out_dev.view())
    queue[].synchronize()
    var total = 0
    for i in range(count):
        total += Int(host[unsafe_offset=i])
    host.unsafe_free()
    ctx[].unload_function(function)
    ctx[].free_sync(out_dev^)
    if total != expected:
        print("FAIL", name, "sum", total, "expected", expected, flush=True)
        raise Error(name + ": wrong sum")
    print("PASS", name, "sum", total, flush=True)


def main() raises:
    run[False]("as_written", 4687759364)
    run[True]("fused", 4687767697)
