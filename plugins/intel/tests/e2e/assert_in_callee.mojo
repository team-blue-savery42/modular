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

# Run with -D ASSERT=all: an assertion fails in a function that returns a value.
# abort() on Intel raises the driver plugin's flag and returns, and every
# function unwinds after it, so the callee hands its caller poison. Before the
# fix the caller went on with it: here each work-item printed "went on with
# index 0" after its assertion. A reference or index used that way can fault.
# Now each work-item has an aborted flag that the caller checks after the call,
# and returns: the message prints once per work-item, nothing after the call
# runs, and the next synchronize fails.
from std.memory import alloc

from hal_gpu import launch, load, open_context


@no_inline
def checked_index(i: Int) -> Int:
    debug_assert(i < 2, "index ", i, " out of range")
    return i * 1000000


def kernel(p: MutPointer[Int32, MutAnyOrigin]):
    var i = checked_index(Int(p[unsafe_offset=0]) + 5)
    print("went on with index", i)
    p[unsafe_offset=i] = 1


def main() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var function = load[type_of(kernel), kernel](ctx[])
    var d = ctx[].alloc_sync(UInt64(8))
    var h = alloc[Int32]({length = 2}).unsafe_leak()
    h[unsafe_offset=0] = 0
    h[unsafe_offset=1] = 0
    queue[].copy_to_device(d.view(), h.unsafe_bitcast[UInt8]())
    launch(queue[], function, 1, 2, [ctx[].memory_get_address(d)])
    queue[].synchronize()
    print("not reached")
