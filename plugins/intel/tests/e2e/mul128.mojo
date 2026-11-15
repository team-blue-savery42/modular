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

# 128-bit integer multiplies on the B70, device against host, operands loaded
# from memory. Each form below is one 128-bit `mul` in the optimized IR (forms
# 1-3 with an `and` of the 128-bit constant 2^64 - 1). Prints the wrong count
# per form, then PASS if all are 0. While the plugin emitted them as OpIMul on a
# 128-bit integer (SPV_INTEL_arbitrary_precision_integers), form 0 was right and
# forms 1-3 wrong on the B70, with IGC's defaults and with -cl-opt-disable;
# since the plugin splits every 128-bit integer into 64-bit halves all four are
# right. In check-e2e.
from std.memory import alloc
from std.random import random_ui64, seed

from hal_gpu import launch, load, open_context

comptime N = 64
comptime K = 4
comptime M64 = UInt128(UInt64.MAX)


@always_inline
def form[k: Int](a: UInt128, b: UInt128) -> UInt128:
    comptime if k == 0:
        # The full product's low 128 bits.
        return a * b
    elif k == 1:
        # 64 x 64 -> 128 (zero-extended operands).
        return UInt128(a.cast[DType.uint64]()) * UInt128(b.cast[DType.uint64]())
    elif k == 2:
        # dragonbox's _umul128_upper64 after LLVM's optimizer: the low half
        # of a 128-bit value, masked, times a zero-extended 64-bit value,
        # shifted down.
        return ((a & M64) * UInt128(b.cast[DType.uint64]())) >> 64
    else:
        # The same product, kept whole.
        return (a & M64) * UInt128(b.cast[DType.uint64]())


def kernel(
    src: MutPointer[UInt128, MutAnyOrigin],
    dst: MutPointer[UInt128, MutAnyOrigin],
):
    for i in range(N):
        var a = src[unsafe_offset=2 * i]
        var b = src[unsafe_offset=2 * i + 1]
        comptime for k in range(K):
            dst[unsafe_offset=K * i + k] = form[k](a, b)


def main() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var function = load[type_of(kernel), kernel](ctx[])
    var d_src = ctx[].alloc_sync(UInt64(32 * N))
    var d_dst = ctx[].alloc_sync(UInt64(16 * K * N))
    var h_src = alloc[UInt128]({length = 2 * N}).unsafe_leak()
    var h_dst = alloc[UInt128]({length = K * N}).unsafe_leak()
    seed(17)
    for i in range(2 * N):
        var hi = random_ui64(0, UInt64.MAX)
        var lo = random_ui64(0, UInt64.MAX)
        # Some operands with a zero half, some small.
        if i % 8 == 1:
            hi = 0
        if i % 8 == 3:
            lo = 0
        if i % 8 == 5:
            hi = 0
            lo = lo >> 40
        h_src[unsafe_offset=i] = (UInt128(hi) << 64) | UInt128(lo)
    queue[].copy_to_device(d_src.view(), h_src.unsafe_bitcast[UInt8]())
    launch(
        queue[],
        function,
        1,
        1,
        [ctx[].memory_get_address(d_src), ctx[].memory_get_address(d_dst)],
    )
    queue[].copy_from_device(h_dst.unsafe_bitcast[UInt8](), d_dst.view())
    queue[].synchronize()
    var wrong = List[Int](length=K, fill=0)
    for i in range(N):
        var a = h_src[unsafe_offset=2 * i]
        var b = h_src[unsafe_offset=2 * i + 1]
        comptime for k in range(K):
            var g = h_dst[unsafe_offset=K * i + k]
            var w = form[k](a, b)
            if g != w:
                wrong[k] += 1
                if wrong[k] <= 2:
                    print(
                        "form",
                        k,
                        "i",
                        i,
                        "a",
                        a,
                        "b",
                        b,
                        "device",
                        g,
                        "host",
                        w,
                    )
    var all_right = True
    for k in range(K):
        print("form", k, "wrong", wrong[k], "of", N)
        if wrong[k] != 0:
            all_right = False
    print("PASS" if all_right else "FAIL")
