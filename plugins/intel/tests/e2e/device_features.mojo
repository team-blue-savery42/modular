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

# The Intel device features through the HAL client. The Intel HAL extension set
# (`hal/extensions/intel.mojo`) resolves the Level Zero plugin's
# `M_driver_intel_device_features`; this branches on the capability with
# `conforms_to`, prints what the B70 reports (utils/check-e2e checks it against
# what Level Zero reports), and times the query: the plugin asks Level Zero once
# per device and answers later calls from its cache (goal: <= 1 us).

from std.benchmark import keep
from std.memory import ArcPointer
from std.os import getenv
from std.time import perf_counter_ns

from max._driver import Driver
from max._driver.max_license.hal.device import Device, get_device_spec
from max._driver.max_license.hal.extensions.intel import (
    INTEL_FEATURE_NO,
    INTEL_FEATURE_YES,
    IntelDeviceFeatures,
    SupportsIntelDeviceFeaturesV1,
)

comptime Spec = get_device_spec[0]()


def yes_no(value: Int32) -> String:
    if value == INTEL_FEATURE_YES:
        return "yes"
    if value == INTEL_FEATURE_NO:
        return "no"
    return "unknown"


def main() raises:
    var spec = getenv("MODULAR_DRIVER_PLUGINS")
    if not spec:
        raise Error("MODULAR_DRIVER_PLUGINS not set")
    var driver = Driver.create(spec)
    var device: ArcPointer[Device[Spec]] = driver[].get_device[0]()
    comptime Ext = type_of(device[].extensions())
    comptime if conforms_to(Ext, SupportsIntelDeviceFeaturesV1):
        var query = device[].extensions().intel_device_features()
        if not query:
            print("FAIL the plugin has no M_driver_intel_device_features")
            return
        ref raw = device[]._raw[]
        var handle = device[]._handle

        var start = perf_counter_ns()
        var f = query.value().device_features(raw, handle)
        var first_ns = perf_counter_ns() - start

        var sizes = String()
        for size in range(1, 64):
            if f.supports_sub_group_size(size):
                sizes += String(" ", size)
        print("version:", f.version, "size:", f.size)
        print("sub-group sizes:" + sizes)
        print("dpas:", yes_no(f.dpas))
        print("dp4a:", yes_no(f.dp4a))
        print("2d block load:", yes_no(f.block_2d_load))
        print("2d block store:", yes_no(f.block_2d_store))
        print("slm per work-group:", f.max_shared_local_memory)
        print(
            String(
                "eus: ",
                f.eu_count(),
                " (",
                f.num_eus_per_subslice,
                " x ",
                f.num_subslices_per_slice,
                " x ",
                f.num_slices,
                ")",
            )
        )
        print("threads per eu:", f.num_threads_per_eu)
        print("simd width:", f.physical_simd_width)
        print("max clock mhz:", f.max_clock_mhz)
        print(
            "memory bus width:",
            f.memory_bus_width,
            "clock mhz:",
            f.memory_clock_mhz,
        )
        print("l2 bytes:", f.l2_cache_bytes)

        # Cached calls: 20 batches of 100,000, the median batch's cost per
        # call. Every answer must equal the first.
        comptime batch = 100_000
        var per_call = List[Float64]()
        var differ = 0
        for _ in range(20):
            start = perf_counter_ns()
            for _ in range(batch):
                var again = query.value().device_features(raw, handle)
                keep(again)
                if (
                    again.sub_group_sizes != f.sub_group_sizes
                    or again.l2_cache_bytes != f.l2_cache_bytes
                ):
                    differ += 1
            per_call.append(Float64(perf_counter_ns() - start) / Float64(batch))
        sort(per_call)
        var median_ns = per_call[len(per_call) // 2]
        print(
            "query first call ns:",
            first_ns,
            "cached ns per call (median of 20 x 100000):",
            median_ns,
            "min:",
            per_call[0],
            "max:",
            per_call[len(per_call) - 1],
        )
        if differ != 0:
            print("FAIL cached answers differ:", differ)
        elif median_ns <= 1000.0:
            print("PASS device_features")
        else:
            print("FAIL cached query over 1 us")
    else:
        print("FAIL the device's extensions lack SupportsIntelDeviceFeaturesV1")
