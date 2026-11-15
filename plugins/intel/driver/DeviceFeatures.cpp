//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Sean Avery. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// M_driver_intel_device_features (IntelDeviceFeatures.h): queried from Level
// Zero once per device, then copied from the device's cache.
//
// DPAS, DP4A and 2D block I/O come from Intel's experimental properties
// structs, which compute-runtime declares in its own level_zero/ze_intel_gpu.h
// and ze_stypes.h (MIT), not in the loader's headers this plugin builds
// against; their layouts and structure types are restated below. The B70's
// driver doesn't offer the core ZE_extension_device_compute_dotproduct_ext
// (its dp_caps read 0), so it isn't used. A feature whose extension the
// driver doesn't list is reported unknown, not absent.
//
//===----------------------------------------------------------------------===//

#include "IntelDeviceFeatures.h"
#include "Plugin.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// MAX's IntelDeviceFeatures (hal/extensions/intel.mojo) mirrors this layout.
static_assert(sizeof(M_driver_intel_device_features_t) == 80,
              "update the Mojo mirror with the C struct");

namespace {

// level_zero/ze_intel_gpu.h, ze_stypes.h (compute-runtime 26.35).
constexpr const char *kModuleDpExtension =
    "ZE_intel_experimental_device_module_dp_properties";
constexpr const char *kBlockArrayExtension =
    "ZE_intel_experimental_device_block_array_properties";
/// ZE_STRUCTURE_INTEL_DEVICE_MODULE_DP_EXP_PROPERTIES; chained to
/// ze_device_module_properties_t.
constexpr uint32_t kModuleDpStype = 0x00030013;
constexpr uint32_t kModuleDpFlagDp4a = 1u << 0;
constexpr uint32_t kModuleDpFlagDpas = 1u << 1;
/// ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_PROPERTIES; chained to
/// ze_device_properties_t.
constexpr uint32_t kBlockArrayStype = 0x00030007;
constexpr uint32_t kBlockArrayFlagStore = 1u << 0;
constexpr uint32_t kBlockArrayFlagLoad = 1u << 1;

/// ze_intel_device_module_dp_exp_properties_t and
/// ze_intel_device_block_array_exp_properties_t share this layout.
struct IntelFlagsProperties {
  uint32_t stype;
  void *pNext;
  uint32_t flags;
};

int32_t feature(bool known, bool present) {
  if (!known)
    return M_DRIVER_INTEL_FEATURE_UNKNOWN;
  return present ? M_DRIVER_INTEL_FEATURE_YES : M_DRIVER_INTEL_FEATURE_NO;
}

std::vector<std::string> driverExtensions(const M::Intel::ZeApi &ze,
                                          ze_driver_handle_t driver) {
  std::vector<std::string> names;
  if (ze.zeDriverGetExtensionProperties == nullptr)
    return names;
  uint32_t count = 0;
  if (ze.zeDriverGetExtensionProperties(driver, &count, nullptr) !=
      ZE_RESULT_SUCCESS)
    return names;
  std::vector<ze_driver_extension_properties_t> props(count);
  if (ze.zeDriverGetExtensionProperties(driver, &count, props.data()) !=
      ZE_RESULT_SUCCESS)
    return names;
  for (const ze_driver_extension_properties_t &p : props)
    names.emplace_back(p.name);
  return names;
}

M_driver_intel_device_features_t query(const M_driver_device &device) {
  M_driver_intel_device_features_t f = {};
  f.version = M_DRIVER_INTEL_DEVICE_FEATURES_VERSION_1;
  f.size = sizeof(f);
  const M::Intel::ZeApi &ze = *device.driver->ze;

  const ze_device_compute_properties_t &compute = device.compute;
  for (uint32_t i = 0;
       i < compute.numSubGroupSizes && i < ZE_SUBGROUPSIZE_COUNT; ++i)
    if (compute.subGroupSizes[i] < 64)
      f.sub_group_sizes |= uint64_t(1) << compute.subGroupSizes[i];
  f.max_shared_local_memory = compute.maxSharedLocalMemory;
  const ze_device_properties_t &props = device.props;
  f.num_slices = props.numSlices;
  f.num_subslices_per_slice = props.numSubslicesPerSlice;
  f.num_eus_per_subslice = props.numEUsPerSubslice;
  f.num_threads_per_eu = props.numThreadsPerEU;
  f.physical_simd_width = props.physicalEUSimdWidth;
  f.max_clock_mhz = props.coreClockRate;

  std::vector<std::string> extensions = driverExtensions(ze, device.zeDriver);
  auto offers = [&](const char *name) {
    return std::find(extensions.begin(), extensions.end(), name) !=
           extensions.end();
  };

  // Without Intel's extension, DP4A still has the core flag.
  bool dpKnown = false;
  IntelFlagsProperties dp = {kModuleDpStype, nullptr, 0};
  ze_device_module_properties_t module = {};
  module.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
  bool moduleKnown = false;
  if (ze.zeDeviceGetModuleProperties != nullptr) {
    dpKnown = offers(kModuleDpExtension);
    module.pNext = dpKnown ? &dp : nullptr;
    moduleKnown = ze.zeDeviceGetModuleProperties(device.zeDevice, &module) ==
                  ZE_RESULT_SUCCESS;
    dpKnown = dpKnown && moduleKnown;
  }
  f.dpas = feature(dpKnown, (dp.flags & kModuleDpFlagDpas) != 0);
  if (dpKnown)
    f.dp4a = feature(true, (dp.flags & kModuleDpFlagDp4a) != 0);
  else
    f.dp4a =
        feature(moduleKnown, (module.flags & ZE_DEVICE_MODULE_FLAG_DP4A) != 0);

  bool blockKnown = offers(kBlockArrayExtension);
  IntelFlagsProperties block = {kBlockArrayStype, nullptr, 0};
  if (blockKnown) {
    ze_device_properties_t chained = {};
    chained.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    chained.pNext = &block;
    blockKnown = ze.zeDeviceGetProperties(device.zeDevice, &chained) ==
                 ZE_RESULT_SUCCESS;
  }
  f.block_2d_load =
      feature(blockKnown, (block.flags & kBlockArrayFlagLoad) != 0);
  f.block_2d_store =
      feature(blockKnown, (block.flags & kBlockArrayFlagStore) != 0);

  // Device memory: the widest of the modules Level Zero lists.
  uint32_t count = 0;
  if (ze.zeDeviceGetMemoryProperties(device.zeDevice, &count, nullptr) ==
          ZE_RESULT_SUCCESS &&
      count > 0) {
    std::vector<ze_device_memory_properties_t> memory(count);
    for (ze_device_memory_properties_t &m : memory)
      m.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    if (ze.zeDeviceGetMemoryProperties(device.zeDevice, &count,
                                       memory.data()) == ZE_RESULT_SUCCESS)
      for (uint32_t i = 0; i < count; ++i)
        if (memory[i].maxBusWidth >= f.memory_bus_width) {
          f.memory_bus_width = memory[i].maxBusWidth;
          f.memory_clock_mhz = memory[i].maxClockRate;
        }
  }

  // Caches: Intel GPUs report one, the L2.
  count = 0;
  if (ze.zeDeviceGetCacheProperties != nullptr &&
      ze.zeDeviceGetCacheProperties(device.zeDevice, &count, nullptr) ==
          ZE_RESULT_SUCCESS &&
      count > 0) {
    std::vector<ze_device_cache_properties_t> caches(count);
    for (ze_device_cache_properties_t &c : caches)
      c.stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
    if (ze.zeDeviceGetCacheProperties(device.zeDevice, &count, caches.data()) ==
        ZE_RESULT_SUCCESS)
      for (uint32_t i = 0; i < count; ++i)
        f.l2_cache_bytes =
            std::max<uint64_t>(f.l2_cache_bytes, caches[i].cacheSize);
  }
  return f;
}

} // namespace

M_driver_status_t
M_driver_intel_device_features(M_driver_device_t device,
                               M_driver_intel_device_features_t *out) {
  if (device == nullptr)
    return M_DRIVER_STATUS_BAD_DEVICE;
  // Version 1 is the whole struct.
  if (out == nullptr || out->version == 0 ||
      out->size < sizeof(M_driver_intel_device_features_t))
    return M_DRIVER_STATUS_INVALID_ARG;
  std::call_once(device->featuresOnce,
                 [device] { device->features = query(*device); });
  // Version 1 is the only version: copy it, keeping the caller's size.
  uint32_t size = out->size;
  std::memcpy(out, &device->features, sizeof(device->features));
  out->size = size;
  return M_DRIVER_STATUS_SUCCESS;
}
