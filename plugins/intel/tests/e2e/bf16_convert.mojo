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

# bfloat16 <-> float32 conversions on the B70 against the host, bit for bit:
# all 65,536 bfloat16 values widened to float32, and a float32 sample
# rounded to bfloat16 (every upper half with the lower half at each rounding
# edge: 0x0000, 0x0001, 0x7fff, 0x8000, 0x8001, 0xffff, so every sign,
# exponent, mantissa, tie, carry into the exponent, infinity, NaN and
# subnormal), at widths 1, 4, 8 and 16; and float64 to bfloat16 for the
# float64 values at and one float64 ulp either side of each float32 tie (the
# host rounds through float32, so twice), at the same widths.
from std.memory import alloc
from std.sys.defines import get_defined_int

from hal_gpu import GpuContext, launch, load, open_context

comptime BF16_COUNT = 65536
comptime LOWERS = 6
comptime F32_COUNT = BF16_COUNT * LOWERS
comptime F64_COUNT = BF16_COUNT * 3


def extend_kernel[
    w: Int
](
    src: MutPointer[BFloat16, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
):
    for i in range(0, BF16_COUNT, w):
        dst.unsafe_store[width=w](
            i, src.unsafe_load[width=w](i).cast[DType.float32]()
        )


def round_kernel[
    w: Int
](
    src: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[BFloat16, MutAnyOrigin],
):
    for i in range(0, F32_COUNT, w):
        dst.unsafe_store[width=w](
            i, src.unsafe_load[width=w](i).cast[DType.bfloat16]()
        )


def round64_kernel[
    w: Int
](
    src: MutPointer[Float64, MutAnyOrigin],
    dst: MutPointer[BFloat16, MutAnyOrigin],
):
    for i in range(0, F64_COUNT, w):
        dst.unsafe_store[width=w](
            i, src.unsafe_load[width=w](i).cast[DType.bfloat16]()
        )


def lower_half(j: Int) -> UInt32:
    if j == 0:
        return 0x0000
    if j == 1:
        return 0x0001
    if j == 2:
        return 0x7FFF
    if j == 3:
        return 0x8000
    if j == 4:
        return 0x8001
    return 0xFFFF


def check_extend[w: Int](mut ctx: GpuContext) raises -> Int:
    var queue = ctx.create_queue()
    comptime k = extend_kernel[w]
    var function = load[type_of(k), k](ctx)
    var d_src = ctx.alloc_sync(UInt64(2 * BF16_COUNT))
    var d_dst = ctx.alloc_sync(UInt64(4 * BF16_COUNT))
    var h_src = alloc[UInt16]({length = BF16_COUNT}).unsafe_leak()
    var h_dst = alloc[UInt32]({length = BF16_COUNT}).unsafe_leak()
    for i in range(BF16_COUNT):
        h_src[unsafe_offset=i] = UInt16(i)
        h_dst[unsafe_offset=i] = 0xDEADBEEF
    queue[].copy_to_device(d_src.view(), h_src.unsafe_bitcast[UInt8]())
    queue[].copy_to_device(d_dst.view(), h_dst.unsafe_bitcast[UInt8]())
    launch(
        queue[],
        function,
        1,
        1,
        [ctx.memory_get_address(d_src), ctx.memory_get_address(d_dst)],
    )
    queue[].copy_from_device(h_dst.unsafe_bitcast[UInt8](), d_dst.view())
    queue[].synchronize()
    var as_bf16 = h_src.unsafe_bitcast[BFloat16]()
    var wrong = 0
    for i in range(BF16_COUNT):
        var want = (
            as_bf16[unsafe_offset=i]
            .cast[DType.float32]()
            .to_bits()
            .cast[DType.uint32]()
        )
        var got = h_dst[unsafe_offset=i]
        if got != want:
            if wrong < 3:
                print(
                    "  extend w",
                    w,
                    "bits",
                    hex(i),
                    "device",
                    hex(got),
                    "host",
                    hex(want),
                )
            wrong += 1
    print(
        "PASS" if wrong == 0 else "FAIL",
        "extend w",
        w,
        "wrong",
        wrong,
        "of",
        BF16_COUNT,
    )
    h_src.unsafe_free()
    h_dst.unsafe_free()
    ctx.unload_function(function)
    ctx.free_sync(d_src^)
    ctx.free_sync(d_dst^)
    return wrong


def check_round[w: Int](mut ctx: GpuContext) raises -> Int:
    var queue = ctx.create_queue()
    comptime k = round_kernel[w]
    var function = load[type_of(k), k](ctx)
    var d_src = ctx.alloc_sync(UInt64(4 * F32_COUNT))
    var d_dst = ctx.alloc_sync(UInt64(2 * F32_COUNT))
    var h_src = alloc[UInt32]({length = F32_COUNT}).unsafe_leak()
    var h_dst = alloc[UInt16]({length = F32_COUNT}).unsafe_leak()
    for upper in range(BF16_COUNT):
        for j in range(LOWERS):
            h_src[unsafe_offset=upper * LOWERS + j] = (
                UInt32(upper) << 16
            ) | lower_half(j)
    for i in range(F32_COUNT):
        h_dst[unsafe_offset=i] = 0xBEEF
    queue[].copy_to_device(d_src.view(), h_src.unsafe_bitcast[UInt8]())
    queue[].copy_to_device(d_dst.view(), h_dst.unsafe_bitcast[UInt8]())
    launch(
        queue[],
        function,
        1,
        1,
        [ctx.memory_get_address(d_src), ctx.memory_get_address(d_dst)],
    )
    queue[].copy_from_device(h_dst.unsafe_bitcast[UInt8](), d_dst.view())
    queue[].synchronize()
    var as_f32 = h_src.unsafe_bitcast[Float32]()
    var wrong = 0
    for i in range(F32_COUNT):
        var want = (
            as_f32[unsafe_offset=i]
            .cast[DType.bfloat16]()
            .to_bits()
            .cast[DType.uint16]()
        )
        var got = h_dst[unsafe_offset=i]
        if got != want:
            if wrong < 3:
                print(
                    "  round w",
                    w,
                    "bits",
                    hex(h_src[unsafe_offset=i]),
                    "device",
                    hex(got),
                    "host",
                    hex(want),
                )
            wrong += 1
    print(
        "PASS" if wrong == 0 else "FAIL",
        "round w",
        w,
        "wrong",
        wrong,
        "of",
        F32_COUNT,
    )
    h_src.unsafe_free()
    h_dst.unsafe_free()
    ctx.unload_function(function)
    ctx.free_sync(d_src^)
    ctx.free_sync(d_dst^)
    return wrong


def check_round64[w: Int](mut ctx: GpuContext) raises -> Int:
    var queue = ctx.create_queue()
    comptime k = round64_kernel[w]
    var function = load[type_of(k), k](ctx)
    var d_src = ctx.alloc_sync(UInt64(8 * F64_COUNT))
    var d_dst = ctx.alloc_sync(UInt64(2 * F64_COUNT))
    var h_src = alloc[UInt64]({length = F64_COUNT}).unsafe_leak()
    var h_dst = alloc[UInt16]({length = F64_COUNT}).unsafe_leak()
    for upper in range(BF16_COUNT):
        var tie = Float32(from_bits=(UInt32(upper) << 16) | 0x8000)
        var bits = Float64(tie).to_bits().cast[DType.uint64]()
        h_src[unsafe_offset=upper * 3] = bits - 1
        h_src[unsafe_offset=upper * 3 + 1] = bits
        h_src[unsafe_offset=upper * 3 + 2] = bits + 1
    for i in range(F64_COUNT):
        h_dst[unsafe_offset=i] = 0xBEEF
    queue[].copy_to_device(d_src.view(), h_src.unsafe_bitcast[UInt8]())
    queue[].copy_to_device(d_dst.view(), h_dst.unsafe_bitcast[UInt8]())
    launch(
        queue[],
        function,
        1,
        1,
        [ctx.memory_get_address(d_src), ctx.memory_get_address(d_dst)],
    )
    queue[].copy_from_device(h_dst.unsafe_bitcast[UInt8](), d_dst.view())
    queue[].synchronize()
    var as_f64 = h_src.unsafe_bitcast[Float64]()
    var wrong = 0
    for i in range(F64_COUNT):
        var want = (
            as_f64[unsafe_offset=i]
            .cast[DType.bfloat16]()
            .to_bits()
            .cast[DType.uint16]()
        )
        var got = h_dst[unsafe_offset=i]
        if got != want:
            if wrong < 3:
                print(
                    "  round64 w",
                    w,
                    "bits",
                    hex(h_src[unsafe_offset=i]),
                    "device",
                    hex(got),
                    "host",
                    hex(want),
                )
            wrong += 1
    print(
        "PASS" if wrong == 0 else "FAIL",
        "round64 w",
        w,
        "wrong",
        wrong,
        "of",
        F64_COUNT,
    )
    h_src.unsafe_free()
    h_dst.unsafe_free()
    ctx.unload_function(function)
    ctx.free_sync(d_src^)
    ctx.free_sync(d_dst^)
    return wrong


def main() raises:
    var ctx = open_context()
    var wrong = 0
    wrong += check_extend[1](ctx[])
    wrong += check_extend[4](ctx[])
    wrong += check_extend[8](ctx[])
    wrong += check_extend[16](ctx[])
    wrong += check_round[1](ctx[])
    wrong += check_round[4](ctx[])
    wrong += check_round[8](ctx[])
    wrong += check_round[16](ctx[])
    # -D FP64=0 on a device without float64 (Xe-LP): IGC rejects the kernel.
    comptime if get_defined_int["FP64", 1]() != 0:
        wrong += check_round64[1](ctx[])
        wrong += check_round64[4](ctx[])
        wrong += check_round64[8](ctx[])
        wrong += check_round64[16](ctx[])
    if wrong:
        raise Error(String(wrong, " wrong results"))
    print("PASS")
