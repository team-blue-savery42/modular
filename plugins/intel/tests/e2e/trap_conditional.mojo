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

# Diagnostic for the trap port: abort() behind a run-time condition, as the
# driver plugin's AbortFailsTheNextSynchronize corpus kernel does.

from std.os import abort
from std.time import sleep

from hal_gpu import launch, load, open_context


def maybe_abort(p: MutPointer[Int32, MutAnyOrigin]):
    if p[0] == 7:
        abort("conditional abort")
    p[1] = 1


def main() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var function = load[type_of(maybe_abort), maybe_abort](ctx[])
    var buf = ctx[].alloc_sync(UInt64(64))
    queue[].fill(buf.view(), value=UInt64(7), value_size=UInt64(4))
    launch(queue[], function, 1, 1, [ctx[].memory_get_address(buf)])
    for attempt in range(2):
        try:
            queue[].synchronize()
            print("synchronize", attempt, "succeeded: abort not reported")
        except e:
            print("synchronize", attempt, "failed:", e)
        sleep(0.1)
