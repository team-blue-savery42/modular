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

# MAX's `apply_packed_bitmask` (nn/gather_scatter.mojo; the sampler's
# grammar mask, `mo.apply_packed_bitmask`) on the B70 through DeviceContext
# (the HAL compat layer: MAX's kernel takes one), against the CPU. Compiling
# it for Intel stopped LLVM's SPIR-V backend with "VReg's low-level type and
# register class have different sizes" (`OpBitwiseAndV` on `<8 x s32>` from
# `<2 x s64>` operands) when MAX's graph compiler built the sampler for the
# B70. Logits [2, 1000] (float32 and bfloat16, a -inf every 7th),
# a random packed mask; kept lanes and -inf lanes keep the logit, the others
# get the fill value.
# run-e2e: device-context
# run-e2e: packages //max/kernels/src/nn:nn

from std.memory import alloc
from std.utils.numerics import neg_inf

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.gather_scatter import apply_packed_bitmask

comptime B = 2
comptime V = 1000
comptime W = (V + 31) // 32


def run[dtype: DType](ctx: DeviceContext) raises -> Int:
    var h_logits = alloc[Scalar[dtype]]({length = B * V}).unsafe_leak()
    var h_out = alloc[Scalar[dtype]]({length = B * V}).unsafe_leak()
    var h_packed = alloc[Int32]({length = B * W}).unsafe_leak()
    var s: UInt64 = 0x9E3779B97F4A7C15
    for i in range(B * V):
        s ^= s << 13
        s ^= s >> 7
        s ^= s << 17
        var x = Float32(Int(s % 2001)) / 100.0 - 10.0
        h_logits[unsafe_offset=i] = (
            neg_inf[dtype]() if i % 7 == 3 else x.cast[dtype]()
        )
    for i in range(B * W):
        s ^= s << 13
        s ^= s >> 7
        s ^= s << 17
        h_packed[unsafe_offset=i] = Int32(Int(s & 0xFFFFFFFF) - (1 << 31))
    var d_logits = ctx.enqueue_create_buffer[dtype](B * V)
    var d_out = ctx.enqueue_create_buffer[dtype](B * V)
    var d_packed = ctx.enqueue_create_buffer[DType.int32](B * W)
    ctx.enqueue_copy(d_logits, h_logits)
    ctx.enqueue_copy(d_packed, h_packed)
    var fill = Scalar[dtype](-1000.0)
    apply_packed_bitmask(
        TileTensor(d_out, row_major(B, V)),
        TileTensor(d_logits, row_major(B, V)).as_imm(),
        TileTensor(d_packed, row_major(B, W)).as_imm(),
        fill,
        ctx,
    )
    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()
    var wrong = 0
    for b in range(B):
        for v in range(V):
            var x = h_logits[unsafe_offset=b * V + v]
            var word = h_packed[unsafe_offset=b * W + v // 32]
            var keep = ((word >> Int32(v % 32)) & 1) != 0
            var want = x if (keep or x == neg_inf[dtype]()) else fill
            var got = h_out[unsafe_offset=b * V + v]
            if got != want:
                if wrong < 3:
                    print("  ", dtype, "b", b, "v", v, "got", got, "want", want)
                wrong += 1
    print(
        "PASS" if wrong == 0 else "FAIL",
        "apply_packed_bitmask",
        dtype,
        "wrong",
        wrong,
        "of",
        B * V,
    )
    h_logits.unsafe_free()
    h_out.unsafe_free()
    h_packed.unsafe_free()
    return wrong


def main() raises:
    with DeviceContext() as ctx:
        var wrong = run[DType.float32](ctx) + run[DType.bfloat16](ctx)
        print("packed_bitmask: failures", wrong)
