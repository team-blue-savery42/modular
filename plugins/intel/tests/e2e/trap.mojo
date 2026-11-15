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

# Port of the stdlib's Mojo/stdlib/test/os/test_trap_gpu.mojo to the HAL
# client: abort() on the GPU prints its message once (thread 0 of
# block 0, of 2 blocks of 32), and the run fails. On Intel the failure comes
# from the driver plugin's abort flag at the next synchronize.

from std.os import abort

from hal_gpu import launch, load, open_context


def abort_kernel():
    abort("gpu abort message")


def main() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var function = load[type_of(abort_kernel), abort_kernel](ctx[])
    launch(queue[], function, 2, 32, [])
    queue[].synchronize()
    print("is never reached")
