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

# IGC loses the index of `for i, byte in enumerate(s.bytes())`. The iterator's
# next value is an Optional, `{ { i64, [8 x i8] }, i8 }`, merged by a phi
# whose "none" side is the constant `{ undef, 0 }`; the index is its i64.
# IGC's input IR has it, but with IGC's defaults it became `undef` (the
# stores of it below went, and `data[i]` read at an undefined offset), and
# with -cl-opt-disable it read as 0; the loop itself runs 3 times. With
# -D ASSERT=all, data[i]'s bounds check failed on the undefined index ("index
# 2 is out of bounds, valid range is 0 to 2", which is in bounds). The
# compiler plugin now splits aggregate phis into scalar phis, and
# the indices are 0 1 2 in all three builds.
from std.memory import alloc

from hal_gpu import launch, load, open_context


def kernel(p: MutPointer[Int32, MutAnyOrigin]):
    comptime expected = StaticString("0.0")
    var data = Array[UInt8, expected.byte_length()](fill=0)
    data[0] = UInt8(p[unsafe_offset=0])
    var bad = 0
    var n = 0
    for i, byte in enumerate(expected.bytes()):
        p[unsafe_offset=4 + n] = Int32(i)
        n += 1
        if byte != data[i]:
            bad += 1
    p[unsafe_offset=1] = Int32(bad)
    p[unsafe_offset=2] = Int32(n)


def main() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var function = load[type_of(kernel), kernel](ctx[])
    var d = ctx[].alloc_sync(UInt64(64))
    var h = alloc[Int32]({length = 16}).unsafe_leak()
    for k in range(16):
        h[unsafe_offset=k] = -1
    h[unsafe_offset=0] = 0
    queue[].copy_to_device(d.view(), h.unsafe_bitcast[UInt8]())
    launch(queue[], function, 1, 1, [ctx[].memory_get_address(d)])
    queue[].copy_from_device(h.unsafe_bitcast[UInt8](), d.view())
    queue[].synchronize()
    print("iterations", h[unsafe_offset=2], "(expected 3)")
    print(
        "indices",
        h[unsafe_offset=4],
        h[unsafe_offset=5],
        h[unsafe_offset=6],
        "(expected 0 1 2)",
    )
    print("bytes that differ", h[unsafe_offset=1], "(expected 3)")
    var right = (
        h[unsafe_offset=2] == 3
        and h[unsafe_offset=4] == 0
        and (h[unsafe_offset=5] == 1 and h[unsafe_offset=6] == 2)
    )
    print("ok" if right else "wrong: indices")
