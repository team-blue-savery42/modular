# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

# ===----------------------------------------------------------------------=== #
# ===----------------------------------------------------------------------=== #
"""Timers and sleep for Intel GPUs, from the device clock.

OpenCL's `clock_read_device` (SPV_KHR_shader_clock, `OpReadClockKHR` at device
scope) reads a 64-bit counter that every Xe-core sees alike. On the Arc Pro
B70 it counts at 2.795-2.80 GHz, measured against the host over idle gaps of
1 to 5 s: the 2.8 GHz maximum GPU clock, not the 19.2 MHz timestamp that
Level Zero reports for kernel timestamps. The GPU held 2.8 GHz while idle in
that measurement, so whether the counter keeps its rate when the GPU clocks
down is unverified.
"""

from std.ffi import external_call
from std.sys.info import CompilationTarget


def _clock_hz[target: CompilationTarget]() -> UInt64:
    """The device clock's rate on `target`, in ticks per second."""
    comptime if target._arch() == "bmg-g21":
        # Arc B580: its rated maximum GPU clock; not measured.
        return 2_850_000_000
    else:
        # Arc Pro B70 (bmg-g31): measured.
        return 2_800_000_000


@inline(.always)
def _clock_ticks() -> UInt64:
    return external_call["_Z17clock_read_devicev", UInt64]()


@inline(.always)
def _global_perf_counter_ns[hz: UInt64]() -> UInt64:
    """Converts the device clock to nanoseconds without overflowing: whole
    seconds and the remainder apart."""
    var ticks = _clock_ticks()
    return (ticks // hz) * 1_000_000_000 + (ticks % hz) * 1_000_000_000 // hz


@inline(.always)
def _sleep[hz: UInt64](sec: Float64):
    """Spins on the device clock until `sec` has passed: Intel GPUs have no
    sleep instruction that OpenCL exposes."""
    var ticks = UInt64(sec * Float64(hz))
    var start = _clock_ticks()
    while _clock_ticks() - start < ticks:
        pass
