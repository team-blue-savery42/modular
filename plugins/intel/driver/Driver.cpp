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
// HAL driver and device entry points on Level Zero. A driver owns every GPU
// device of every Level Zero GPU driver, created once in M_driver_create; a
// device handle stays valid until M_driver_destroy.
//
//===----------------------------------------------------------------------===//

#include "Plugin.h"

#include <dlpack/dlpack.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using M::Intel::fail;
using M::Intel::resultName;
using M::Intel::ZeApi;

M_driver_status_t M::Intel::fail(M_driver_driver &driver,
                                 const std::string &message) {
  return driver.status.create(message);
}

M_driver_status_t M::Intel::failZe(M_driver_driver &driver, const char *call,
                                   ze_result_t result) {
  return fail(driver, std::string(call) + ": " + resultName(result));
}

namespace {

/// The `ocloc -device` name for an IP version, or the dotted IP version, which
/// `ocloc` also accepts, for devices not in the table.
std::string archName(uint32_t ipVersion) {
  uint32_t architecture = ipVersion >> 22;
  uint32_t release = (ipVersion >> 14) & 0xff;
  uint32_t revision = ipVersion & 0x3f;
  struct Known {
    uint32_t architecture, release;
    const char *name;
  };
  // From `ocloc ids <name>`; revision-independent.
  static constexpr Known kKnown[] = {
      {12, 3, "rpl-p"},  // Raptor Lake-P integrated GPU, Xe-LP
      {12, 70, "arl-s"}, // Arrow Lake-S integrated GPU, Xe-LPG
      {20, 1, "bmg-g21"},
      {20, 2, "bmg-g31"},
  };
  for (const Known &known : kKnown)
    if (known.architecture == architecture && known.release == release)
      return known.name;
  char dotted[32];
  std::snprintf(dotted, sizeof(dotted), "%u.%u.%u", architecture, release,
                revision);
  return dotted;
}

M_driver_status_t initialize(M_driver_driver &driver) {
  const ZeApi *ze = ZeApi::get();
  if (ze == nullptr)
    return fail(driver, ZeApi::error());
  driver.ze = ze;

  ze_init_driver_type_desc_t desc = {ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC,
                                     nullptr, ZE_INIT_DRIVER_TYPE_FLAG_GPU};
  uint32_t driverCount = 0;
  ze_result_t result = ze->zeInitDrivers(&driverCount, nullptr, &desc);
  if (result != ZE_RESULT_SUCCESS)
    return fail(driver, std::string("zeInitDrivers: ") + resultName(result));
  std::vector<ze_driver_handle_t> zeDrivers(driverCount);
  if (driverCount > 0) {
    result = ze->zeInitDrivers(&driverCount, zeDrivers.data(), &desc);
    if (result != ZE_RESULT_SUCCESS)
      return fail(driver, std::string("zeInitDrivers: ") + resultName(result));
  }

  for (ze_driver_handle_t zeDriver : zeDrivers) {
    uint32_t deviceCount = 0;
    result = ze->zeDeviceGet(zeDriver, &deviceCount, nullptr);
    if (result != ZE_RESULT_SUCCESS)
      return fail(driver, std::string("zeDeviceGet: ") + resultName(result));
    std::vector<ze_device_handle_t> zeDevices(deviceCount);
    if (deviceCount > 0) {
      result = ze->zeDeviceGet(zeDriver, &deviceCount, zeDevices.data());
      if (result != ZE_RESULT_SUCCESS)
        return fail(driver, std::string("zeDeviceGet: ") + resultName(result));
    }
    for (ze_device_handle_t zeDevice : zeDevices) {
      auto device = std::make_unique<M_driver_device>();
      device->driver = &driver;
      device->id = static_cast<int64_t>(driver.devices.size());
      device->zeDriver = zeDriver;
      device->zeDevice = zeDevice;
      ze_device_ip_version_ext_t ip = {ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT,
                                       nullptr, 0};
      device->props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
      device->props.pNext = &ip;
      result = ze->zeDeviceGetProperties(zeDevice, &device->props);
      if (result != ZE_RESULT_SUCCESS)
        return fail(driver, std::string("zeDeviceGetProperties: ") +
                                resultName(result));
      device->props.pNext = nullptr;
      device->compute.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
      result = ze->zeDeviceGetComputeProperties(zeDevice, &device->compute);
      if (result != ZE_RESULT_SUCCESS)
        return fail(driver, std::string("zeDeviceGetComputeProperties: ") +
                                resultName(result));
      device->ipVersion = ip.ipVersion;
      device->arch = archName(ip.ipVersion);
      driver.devices.push_back(std::move(device));
    }
  }
  return M_DRIVER_STATUS_SUCCESS;
}

/// Copies `text` into a `M_DRIVER_DEVICE_PROPERTY_MAX_LEN` buffer.
void copyProperty(void *value, const std::string &text) {
  M::Target::copyStatusMessage(static_cast<char *>(value),
                               M_DRIVER_DEVICE_PROPERTY_MAX_LEN, text);
}

} // namespace

M_driver_status_t M_driver_status_message(M_driver_driver_t driver,
                                          M_driver_status_t status,
                                          char *message_buffer,
                                          int64_t message_buffer_size) {
  if (driver == nullptr)
    return M_DRIVER_STATUS_BAD_DRIVER;
  return driver->status.getMessage(status, message_buffer, message_buffer_size);
}

M_driver_status_t M_driver_create(M_driver_version *version,
                                  M_driver_driver_t *driver) {
  if (version == nullptr || driver == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (version->major != M_DRIVER_INTERFACE_VERSION_MAJOR)
    return M_DRIVER_STATUS_INCOMPATIBLE_VERSION;
  version->major = M_DRIVER_INTERFACE_VERSION_MAJOR;
  version->minor = M_DRIVER_INTERFACE_VERSION_MINOR;
  version->patch = M_DRIVER_INTERFACE_VERSION_PATCH;
  auto created = std::make_unique<M_driver_driver>();
  created->initStatus = initialize(*created);
  *driver = created.release();
  return (*driver)->initStatus;
}

M_driver_status_t M_driver_destroy(M_driver_driver_t driver) {
  if (driver == nullptr)
    return M_DRIVER_STATUS_BAD_DRIVER;
  delete driver;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_property(M_driver_driver_t driver,
                                    const char *property_name, void *value) {
  if (driver == nullptr)
    return M_DRIVER_STATUS_BAD_DRIVER;
  if (property_name == nullptr || value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (std::strcmp(property_name, M_DRIVER_PROPERTY_API) == 0) {
    *static_cast<const char **>(value) = "level_zero";
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (std::strcmp(property_name, M_DRIVER_PROPERTY_VERSION) == 0) {
    auto *version = static_cast<M_driver_version *>(value);
    version->major = M_DRIVER_INTERFACE_VERSION_MAJOR;
    version->minor = M_DRIVER_INTERFACE_VERSION_MINOR;
    version->patch = M_DRIVER_INTERFACE_VERSION_PATCH;
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_driver_status_t M_driver_device_count(M_driver_driver_t driver,
                                        int64_t *count) {
  if (driver == nullptr)
    return M_DRIVER_STATUS_BAD_DRIVER;
  if (count == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (driver->initStatus != M_DRIVER_STATUS_SUCCESS)
    return driver->initStatus;
  *count = static_cast<int64_t>(driver->devices.size());
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device_get(M_driver_driver_t driver,
                                      int64_t device_id,
                                      M_driver_device_t *device) {
  if (driver == nullptr)
    return M_DRIVER_STATUS_BAD_DRIVER;
  if (device == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (device_id < 0 ||
      device_id >= static_cast<int64_t>(driver->devices.size()))
    return M_DRIVER_STATUS_BAD_DEVICE;
  *device = driver->devices[static_cast<size_t>(device_id)].get();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device_property(M_driver_device_t device,
                                           const char *property_name,
                                           void *value) {
  if (device == nullptr)
    return M_DRIVER_STATUS_BAD_DEVICE;
  if (property_name == nullptr || value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  std::string name = property_name;
  if (name == M_DRIVER_DEVICE_PROPERTY_NAME) {
    copyProperty(value, device->props.name);
  } else if (name == M_DRIVER_DEVICE_PROPERTY_ARCH) {
    copyProperty(value, device->arch);
  } else if (name == M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE) {
    auto *dl = static_cast<M_driver_dlpack_device *>(value);
    dl->device_type = kDLOneAPI;
    dl->device_id = static_cast<int32_t>(device->id);
  } else if (name == M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE_PINNED) {
    // USM host allocations are ordinary host memory to DLPack consumers.
    auto *dl = static_cast<M_driver_dlpack_device *>(value);
    dl->device_type = kDLCPU;
    dl->device_id = 0;
  } else if (name == "host_unified") {
    *static_cast<int64_t *>(value) =
        (device->props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) != 0;
  } else if (name == M_DRIVER_DEVICE_PROPERTY_MAX_ALLOCATION_SIZE) {
    *static_cast<uint64_t *>(value) = device->props.maxMemAllocSize;
  } else {
    return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
  }
  return M_DRIVER_STATUS_SUCCESS;
}
