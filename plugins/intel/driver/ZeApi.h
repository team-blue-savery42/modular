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
// The Level Zero entry points the plugin uses, resolved from the system's
// libze_loader.so.1 with dlopen, so the plugin has no link-time dependency on
// the loader (the pattern of the HIP plugin's HIPApi.h).
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_DRIVER_ZEAPI_H
#define MOJO_INTEL_DRIVER_ZEAPI_H

#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>

#include <string>

/// Core entry points the plugin can't work without.
#define MOJO_INTEL_ZE_REQUIRED(X)                                              \
  X(zeInitDrivers)                                                             \
  X(zeDeviceGet)                                                               \
  X(zeDeviceGetProperties)                                                     \
  X(zeDeviceGetMemoryProperties)                                               \
  X(zeDeviceGetComputeProperties)                                              \
  X(zeDeviceGetStatus)                                                         \
  X(zeKernelGetProperties)                                                     \
  X(zeContextCreate)                                                           \
  X(zeContextDestroy)                                                          \
  X(zeContextMakeMemoryResident)                                               \
  X(zeCommandListCreateImmediate)                                              \
  X(zeCommandListDestroy)                                                      \
  X(zeCommandListAppendMemoryCopy)                                             \
  X(zeCommandListAppendMemoryFill)                                             \
  X(zeCommandListHostSynchronize)                                              \
  X(zeCommandListAppendBarrier)                                                \
  X(zeCommandListAppendWaitOnEvents)                                           \
  X(zeEventPoolCreate)                                                         \
  X(zeEventPoolDestroy)                                                        \
  X(zeEventCreate)                                                             \
  X(zeEventDestroy)                                                            \
  X(zeEventHostReset)                                                          \
  X(zeEventHostSynchronize)                                                    \
  X(zeEventQueryStatus)                                                        \
  X(zeModuleCreate)                                                            \
  X(zeModuleDestroy)                                                           \
  X(zeModuleBuildLogGetString)                                                 \
  X(zeModuleBuildLogDestroy)                                                   \
  X(zeModuleGetGlobalPointer)                                                  \
  X(zeKernelCreate)                                                            \
  X(zeKernelDestroy)                                                           \
  X(zeKernelSetGroupSize)                                                      \
  X(zeKernelSetArgumentValue)                                                  \
  X(zeCommandListAppendLaunchKernel)                                           \
  X(zeMemGetAllocProperties)                                                   \
  X(zeMemAllocDevice)                                                          \
  X(zeMemAllocHost)                                                            \
  X(zeMemFree)

/// Counter-based events (Level Zero 1.15+); pooled events are the fallback.
/// Linkage inspection names a module's unresolved imports; without it, the
/// first zeKernelCreate fails with no names. The device feature queries
/// (DeviceFeatures.cpp) report "unknown" without theirs. Launch profiling
/// (Profile.h) needs the timestamp queries.
#define MOJO_INTEL_ZE_OPTIONAL(X)                                              \
  X(zeEventCounterBasedCreate)                                                 \
  X(zeModuleInspectLinkageExt)                                                 \
  X(zeDriverGetExtensionProperties)                                            \
  X(zeDeviceGetModuleProperties)                                               \
  X(zeDeviceGetCacheProperties)                                                \
  X(zeEventQueryKernelTimestamp)                                               \
  X(zeDeviceGetGlobalTimestamps)

/// Sysman entry points, used only for free-memory queries; may be missing.
#define MOJO_INTEL_ZES_OPTIONAL(X)                                             \
  X(zesInit)                                                                   \
  X(zesDriverGet)                                                              \
  X(zesDriverGetDeviceByUuidExp)                                               \
  X(zesDeviceEnumMemoryModules)                                                \
  X(zesMemoryGetProperties)                                                    \
  X(zesMemoryGetState)

namespace M::Intel {

struct ZeApi {
  /// The loaded API, or null with `error()` set when the loader or a required
  /// entry point is missing. Loaded once per process.
  static const ZeApi *get();
  static const std::string &error();

#define MOJO_INTEL_ZE_MEMBER(name) decltype(&::name) name = nullptr;
  MOJO_INTEL_ZE_REQUIRED(MOJO_INTEL_ZE_MEMBER)
  MOJO_INTEL_ZE_OPTIONAL(MOJO_INTEL_ZE_MEMBER)
  MOJO_INTEL_ZES_OPTIONAL(MOJO_INTEL_ZE_MEMBER)
#undef MOJO_INTEL_ZE_MEMBER

  /// Whether every Sysman entry point resolved.
  bool hasSysman = false;

  /// The dlopen handle; kept open for the life of the process.
  void *handle = nullptr;
};

/// A name for a Level Zero result, e.g. "ZE_RESULT_ERROR_DEVICE_LOST", or its
/// hex value for results without one here.
std::string resultName(ze_result_t result);

} // namespace M::Intel

#endif // MOJO_INTEL_DRIVER_ZEAPI_H
