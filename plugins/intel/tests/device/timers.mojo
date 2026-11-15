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

# Timers and sleep on the B70: compiled for the Intel target, then run
# by utils/l0/run.sh as one work-item. Run with utils/check-device.

from std.compile import compile_info
from std.sys.info import CompilationTarget
from std.time import global_perf_counter_ns, perf_counter_ns, sleep

from max._plugin.intel.targets import ArcProB70


def timers(p: Pointer[Int32, MutAnyOrigin]):
    var start = global_perf_counter_ns()
    var local_start = perf_counter_ns()
    sleep(0.05)
    var end = global_perf_counter_ns()
    var local_end = perf_counter_ns()
    # p[1]: the 50 ms sleep in whole milliseconds; p[2]: perf_counter_ns
    # moved forward too.
    p[1] = Int32((end - start) // 1_000_000)
    p[2] = Int32(local_end > local_start)


def main() raises:
    comptime b70 = CompilationTarget[_mlir_value=ArcProB70.mlir_target]()
    with open("out/timers.spv", "w") as f:
        f.write(compile_info[timers, target=b70, emission_kind="object"]().asm)
