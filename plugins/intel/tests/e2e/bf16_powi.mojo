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

# The computation of MAX's test_pow_bf16 (x ** n for a runtime integer n,
# and x ** SIMD[bfloat16](n)) on its inputs (i - 128 + 0.001 for i < 256)
# at widths 1, 4 and 8. Each device result is compared bit for bit with the
# same bfloat16 computation on the host, and counted against the test's own
# check (float32 x ** n on the host, atol 1e-4, rtol 2e-2) as "far".
from std.memory import alloc

from hal_gpu import GpuContext, launch, load, open_context

comptime N = 256


def kernel[
    w: Int, bf16_exponent: Bool
](
    input: MutPointer[Float32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    exponent: Int,
):
    for i in range(0, N, w):
        var val = input.unsafe_load[width=w](i).cast[DType.bfloat16]()
        var result: SIMD[DType.bfloat16, w]
        comptime if bf16_exponent:
            result = val ** SIMD[DType.bfloat16, w](exponent)
        else:
            result = val**exponent
        output.unsafe_store[width=w](i, result.cast[DType.float32]())


def host[bf16_exponent: Bool](x: Float32, exponent: Int) -> Float32:
    var val = x.cast[DType.bfloat16]()
    comptime if bf16_exponent:
        return (val ** BFloat16(exponent)).cast[DType.float32]()
    else:
        return (val**exponent).cast[DType.float32]()


def close(got: Float32, want: Float32) -> Bool:
    # As test_pow_bf16's assert_almost_equal(atol=1e-04, rtol=2e-02).
    if got.to_bits() == want.to_bits() or (got != got and want != want):
        return True
    return abs(got - want) <= 1e-04 + 2e-02 * abs(want)


def check[
    w: Int, bf16_exponent: Bool
](mut ctx: GpuContext, exponent: Int) raises -> Int:
    var queue = ctx.create_queue()
    comptime k = kernel[w, bf16_exponent]
    var function = load[type_of(k), k](ctx)
    var d_in = ctx.alloc_sync(UInt64(4 * N))
    var d_out = ctx.alloc_sync(UInt64(4 * N))
    var h_in = alloc[Float32]({length = N}).unsafe_leak()
    var h_out = alloc[Float32]({length = N}).unsafe_leak()
    for i in range(N):
        h_in[unsafe_offset=i] = Float32(i - N // 2) + 0.001
        h_out[unsafe_offset=i] = 0
    queue[].copy_to_device(d_in.view(), h_in.unsafe_bitcast[UInt8]())
    queue[].copy_to_device(d_out.view(), h_out.unsafe_bitcast[UInt8]())
    launch(
        queue[],
        function,
        1,
        1,
        [
            ctx.memory_get_address(d_in),
            ctx.memory_get_address(d_out),
            UInt64(exponent),
        ],
    )
    queue[].copy_from_device(h_out.unsafe_bitcast[UInt8](), d_out.view())
    queue[].synchronize()
    var wrong = 0
    var far = 0
    for i in range(N):
        var x = h_in[unsafe_offset=i]
        var got = h_out[unsafe_offset=i]
        var want = host[bf16_exponent](x, exponent)
        var want32: Float32
        comptime if bf16_exponent:
            want32 = x ** Float32(exponent)
        else:
            want32 = x**exponent
        if not close(got, want32):
            far += 1
        if got.to_bits() != want.to_bits():
            if wrong < 3:
                print(
                    "  w",
                    w,
                    "bf16 exponent" if bf16_exponent else "int exponent",
                    exponent,
                    "x",
                    x,
                    "device",
                    got,
                    "host",
                    want,
                )
            wrong += 1
    print(
        "w",
        w,
        "bf16 exponent" if bf16_exponent else "int exponent",
        exponent,
        "wrong",
        wrong,
        "of",
        N,
    )
    print(
        "w",
        w,
        "bf16 exponent" if bf16_exponent else "int exponent",
        exponent,
        "far",
        far,
        "of",
        N,
    )
    h_in.unsafe_free()
    h_out.unsafe_free()
    ctx.unload_function(function)
    ctx.free_sync(d_in^)
    ctx.free_sync(d_out^)
    return wrong


def main() raises:
    var ctx = open_context()
    var wrong = 0
    for e in [-1, 2, 3, 5, 6]:
        wrong += check[1, False](ctx[], e)
        wrong += check[4, False](ctx[], e)
        wrong += check[8, False](ctx[], e)
    for e in [2, 3, 5, 6]:
        wrong += check[1, True](ctx[], e)
        wrong += check[4, True](ctx[], e)
        wrong += check[8, True](ctx[], e)
    if wrong:
        raise Error(String(wrong, " wrong results"))
    print("PASS")
