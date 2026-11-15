# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""HAL extensions of Intel GPUs on the Level Zero driver plugin.

`SupportsIntelDeviceFeaturesV1`: the device's run-time features (sub-group
sizes, DPAS, DP4A, 2D block I/O, machine sizes) through the plugin's
`M_driver_intel_device_features`, an entry point outside the HAL C API
(`plugins/intel/driver/IntelDeviceFeatures.h`). Compile-time choices still
come from the target (`bmg-g31`):

```mojo
comptime Ext = type_of(device.extensions())
comptime if conforms_to(Ext, SupportsIntelDeviceFeaturesV1):
    var query = device.extensions().intel_device_features()
    if query:
        var features = query.value().device_features(raw_driver, handle)
```
"""

from std.collections import OptionalReg
from std.ffi import _DLHandle
from std.memory import MutPointer
from std.sys import size_of

from ..plugin import RawDriver
from ..plugins import (
    DeviceHandle,
    HALFunction,
    PluginResultCode,
    RawPluginComponent,
)
from ..status import STATUS_SUCCESS, STATUS_UNKNOWN_SYMBOL, HALError
from .traits import HALExtensions

comptime INTEL_DEVICE_FEATURES_VERSION_1: UInt32 = 1
"""The first version of `IntelDeviceFeatures`."""

comptime INTEL_FEATURE_UNKNOWN: Int32 = -1
"""A yes/no feature the driver doesn't report (its Level Zero extension is
missing). Numeric fields the driver doesn't report are 0."""
comptime INTEL_FEATURE_NO: Int32 = 0
comptime INTEL_FEATURE_YES: Int32 = 1


@fieldwise_init
struct IntelDeviceFeatures(TrivialRegisterPassable):
    """Mirrors C `M_driver_intel_device_features`, version 1 (80 bytes; the
    plugin asserts the size). Yes/no fields hold `INTEL_FEATURE_*`."""

    var version: UInt32
    var size: UInt32
    # Bit s is set when sub-group size s is supported.
    var sub_group_sizes: UInt64
    var dpas: Int32
    var dp4a: Int32
    var block_2d_load: Int32
    var block_2d_store: Int32
    # Bytes per work-group.
    var max_shared_local_memory: UInt32
    var num_slices: UInt32
    # Xe-cores per slice.
    var num_subslices_per_slice: UInt32
    var num_eus_per_subslice: UInt32
    var num_threads_per_eu: UInt32
    var physical_simd_width: UInt32
    var max_clock_mhz: UInt32
    var memory_bus_width: UInt32
    var memory_clock_mhz: UInt32
    var reserved: UInt32
    var l2_cache_bytes: UInt64

    @staticmethod
    def request() -> Self:
        """A version 1 request: version and size set, the rest zero."""
        return Self(
            INTEL_DEVICE_FEATURES_VERSION_1,
            UInt32(size_of[Self]()),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def supports_sub_group_size(self, size: Int) -> Bool:
        """Whether the device runs kernels at sub-group size `size`."""
        if size <= 0 or size >= 64:
            return False
        return (self.sub_group_sizes >> UInt64(size)) & 1 == 1

    def eu_count(self) -> Int:
        """EUs: slices x Xe-cores per slice x EUs per Xe-core."""
        return (
            Int(self.num_slices)
            * Int(self.num_subslices_per_slice)
            * Int(self.num_eus_per_subslice)
        )


trait IntelDeviceFeaturesV1(TrivialRegisterPassable):
    # Status messages are driver-scoped, hence the `RawDriver` argument.
    def device_features(
        self, driver: RawDriver, device: DeviceHandle
    ) raises HALError -> IntelDeviceFeatures:
        """The device's features; queried once by the plugin, then cached."""
        ...


trait SupportsIntelDeviceFeaturesV1(Deinitable, Movable):
    """Capability: Intel device features (`M_driver_intel_device_features`).

    `intel_device_features()` returns `None` for a plugin without the entry
    point, such as an older Level Zero plugin or the CPU plugin.
    """

    comptime IntelDeviceFeaturesTy: IntelDeviceFeaturesV1

    def intel_device_features(
        self,
    ) -> OptionalReg[Self.IntelDeviceFeaturesTy]:
        ...


struct RawPluginIntelDeviceFeatures(IntelDeviceFeaturesV1, RawPluginComponent):
    """`M_driver_intel_device_features` of the Level Zero driver plugin."""

    var query: HALFunction[
        "M_driver_intel_device_features",
        def(
            device: DeviceHandle,
            features: MutPointer[IntelDeviceFeatures, _],
        ) thin abi("C") -> PluginResultCode,
        required=True,
    ]

    @staticmethod
    def load(
        lib: _DLHandle, so_path: StringSlice
    ) raises HALError -> OptionalReg[Self]:
        """Resolves the entry point, or None when `lib` doesn't export it."""
        try:
            return Self(lib, so_path)
        except e:
            if e.status == STATUS_UNKNOWN_SYMBOL:
                return None
            raise e^

    def device_features(
        self, driver: RawDriver, device: DeviceHandle
    ) raises HALError -> IntelDeviceFeatures:
        var features = IntelDeviceFeatures.request()
        var status = self.query.get()(device, MutPointer(to=features))
        if status != STATUS_SUCCESS:
            var err = driver.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to query Intel device features: {err.message}"
                ),
            )
        return features


struct IntelHALExtensions(HALExtensions, SupportsIntelDeviceFeaturesV1):
    """`HALExtensions` implementation for Intel GPUs on Level Zero."""

    comptime IntelDeviceFeaturesTy = RawPluginIntelDeviceFeatures
    var _features: OptionalReg[RawPluginIntelDeviceFeatures]

    def __init__(
        out self, lib: _DLHandle, so_path: StringSlice
    ) raises HALError:
        self._features = RawPluginIntelDeviceFeatures.load(lib, so_path)

    def intel_device_features(
        self,
    ) -> OptionalReg[Self.IntelDeviceFeaturesTy]:
        return self._features
