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

# Port of the stdlib's Mojo/stdlib/test/memory/pointer/
# test_unsafe_from_address_overflow.mojo to the HAL client. The
# stdlib test expects a debug assertion when an address too wide for a
# 32-bit SHARED pointer is converted; Intel's SHARED pointers are 64-bit, as
# Metal's are as wide as Int, so there is no guard to trip (the stdlib marks
# the test incompatible on Apple GPUs for that reason). Expected: no
# assertion, and the run completes.

from hal_gpu import launch, load, open_context


def _kernel(addr: Int64):
    var p = Pointer[Int, MutAnyOrigin, address_space=.SHARED](
        unsafe_from_address=Int(addr)
    )
    _ = p


def main() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var function = load[type_of(_kernel), _kernel](ctx[])
    launch(queue[], function, 1, 1, [UInt64(1 << 33)])
    queue[].synchronize()
    print("completed without an assertion")
