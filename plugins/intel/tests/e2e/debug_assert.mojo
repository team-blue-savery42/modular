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

# A failing debug_assert on the device (run with -D ASSERT=all): each failing
# work-item prints its message once, on one whole line with its block, thread
# and source location, then the next synchronize fails. 2 blocks of 4
# work-items all fail. test_debug_assert_gpu_error read a null pointer here
# (IGC 2.41): IGC wrote a constant SourceLocation's length over its file
# name, and work-items ran on past the assertion reporter into other code.
from std.memory import alloc

from hal_gpu import launch, load, open_context


def kernel(p: MutPointer[Int32, MutAnyOrigin]):
    debug_assert(
        p[unsafe_offset=0] == 0,
        "forcing 100% failure with value ",
        p[unsafe_offset=0],
    )
    p[unsafe_offset=1] = 1


def main() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var function = load[type_of(kernel), kernel](ctx[])
    var d = ctx[].alloc_sync(UInt64(8))
    var h = alloc[Int32]({length = 2}).unsafe_leak()
    h[unsafe_offset=0] = 7
    h[unsafe_offset=1] = 0
    queue[].copy_to_device(d.view(), h.unsafe_bitcast[UInt8]())
    launch(queue[], function, 2, 4, [ctx[].memory_get_address(d)])
    queue[].synchronize()
    print("not reached")
