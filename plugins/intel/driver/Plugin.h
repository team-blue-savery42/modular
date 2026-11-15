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
// The objects behind the HAL's opaque handles, shared by the entry points.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_DRIVER_PLUGIN_H
#define MOJO_INTEL_DRIVER_PLUGIN_H

#include "IntelDeviceFeatures.h"
#include "Profile.h"
#include "ZeApi.h"

#include "HAL/M_driver_api.h"
#include "Target/lib/Support/Utils.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

struct M_driver_device {
  M_driver_driver *driver = nullptr;
  int64_t id = 0;
  ze_driver_handle_t zeDriver = nullptr;
  ze_device_handle_t zeDevice = nullptr;
  ze_device_properties_t props = {};
  ze_device_compute_properties_t compute = {};
  /// IP version (GMD ID): architecture [31:22], release [21:14],
  /// revision [5:0].
  uint32_t ipVersion = 0;
  std::string arch;
  /// Bytes this plugin holds in device allocations on this device; the
  /// free-memory estimate when Sysman is unavailable.
  std::atomic<uint64_t> allocatedBytes{0};
  /// Sysman handles of the device-local memory modules, found once.
  std::once_flag sysmanOnce;
  std::vector<zes_mem_handle_t> sysmanMemory;
  /// When a wait last asked whether the device is lost (steady clock ticks),
  /// and whether it was (Context.cpp `waited`).
  std::atomic<int64_t> statusCheckedAt{0};
  std::atomic<bool> lost{false};
  /// M_driver_intel_device_features, queried once (DeviceFeatures.cpp).
  std::once_flag featuresOnce;
  M_driver_intel_device_features_t features = {};
};

struct M_driver_driver {
  M::Target::CustomStatus status{16};
  /// The status `M_driver_create` returns; the driver is created even when
  /// Level Zero is unusable, so its status message can explain why.
  M_driver_status_t initStatus = M_DRIVER_STATUS_SUCCESS;
  std::vector<std::unique_ptr<M_driver_device>> devices;
  const M::Intel::ZeApi *ze = nullptr;
};

struct M_driver_queue;

/// A Level Zero context on one device, with an in-order immediate command list
/// for the HAL's synchronous copies and fills.
struct M_driver_context {
  M_driver_device *device = nullptr;
  ze_context_handle_t zeContext = nullptr;
  ze_command_list_handle_t syncList = nullptr;
  /// Serializes use of `syncList` across threads.
  std::mutex syncMutex;
  /// Host memory a kernel in this context sets to nonzero when it calls
  /// `abort()`: IGC has no trap, so the compiler plugin's `__mojo_intel_abort`
  /// stores here, through the module global `__mojo_intel_abort_flag` that
  /// bundle_load points at it. A synchronize that finds it set fails, and
  /// keeps failing, as a CUDA context does after a trap.
  int32_t *abortFlag = nullptr;
  /// The device heap kernels' malloc allocates from: device memory,
  /// resident, created when the first module that allocates loads, and
  /// freed at context_destroy once the context's queues are idle. A module
  /// that allocates reaches it through its global `__mojo_intel_heap`, which
  /// bundle_load points here. Guarded by `heapMutex`.
  std::mutex heapMutex;
  void *heap = nullptr;
  uint64_t heapBytes = 0;
  /// Released USM blocks kept for reuse, by capacity: MAX's graph
  /// runtime on the HAL allocates and frees its intermediate buffers on every
  /// step, and a fresh Level Zero allocation made resident costs from
  /// microseconds to over a millisecond. Device blocks stay counted in the
  /// device's `allocatedBytes`. Guarded by `cacheMutex`.
  std::mutex cacheMutex;
  std::multimap<uint64_t, void *> cachedDevice, cachedHost;
  uint64_t cachedBytes = 0;
  /// The USM blocks this plugin allocated and made resident, by address, to
  /// their size: `memory_wrap` of a region inside one skips a second
  /// `zeContextMakeMemoryResident`, which on the integrated GPUs costs a
  /// kernel call per wrap (MAX's graph runtime wraps buffer views on every
  /// step). Guarded by `blocksMutex`.
  std::mutex blocksMutex;
  std::map<uintptr_t, uint64_t> blocks;
  /// The context's queues, after whose work so far a synchronous free
  /// releases its memory. Guarded by `queuesMutex`; a free copies the list
  /// and appends to each queue without the lock.
  std::mutex queuesMutex;
  std::vector<M_driver_queue *> queues;
  /// Notified when a free stops draining queues (`M_driver_queue::drainers`).
  std::condition_variable queuesIdle;
};

struct M_driver_memory;

/// A HAL queue: its own in-order immediate command list on the compute engine
/// (copies too, unless MOJO_INTEL_COPY_OFFLOAD=1). Work is
/// appended without waiting; the list executes in submission order.
struct M_driver_queue {
  M_driver_context *context = nullptr;
  ze_command_list_handle_t list = nullptr;
  /// Serializes appends from several threads.
  std::mutex mutex;
  /// Memory freed with `memory_free_async`: released once the commands
  /// appended before the free (`after` of them) have completed. Guarded by
  /// `mutex`. A synchronous free (`memory_free_sync`, `memory_free_pinned`)
  /// puts one entry on every queue of the context with work it may still
  /// use, sharing `sharers`: the memory is released by the last.
  struct PendingFree {
    M_driver_memory *memory;
    uint64_t after;
    std::shared_ptr<std::atomic<int>> sharers = nullptr;
  };
  std::vector<PendingFree> pendingFrees;
  /// Free marks: commands appended so far; how many of them the
  /// latest free mark follows; how many are known complete. A free appends a
  /// mark (a barrier signalling one of `freeMarks`, counter-based events made
  /// on first use) only when commands came since the last one, and status
  /// queries of the marks, oldest first, retire every entry the completed
  /// ones cover, instead of an event and a barrier per free and per queue.
  /// The marks are a ring: with a single event, only the
  /// newest mark could be queried, which completes only once the device
  /// catches up with the host, so while the host ran ahead nothing was
  /// retired. Reusing a ring slot whose mark is still pending only makes its
  /// completion arrive later (a counter-based event reports its newest
  /// signal). Guarded by `mutex`.
  static constexpr size_t kFreeMarks = 8;
  struct FreeMark {
    ze_event_handle_t event = nullptr;
    /// `appended` when the mark was appended; 0: never used.
    uint64_t at = 0;
  };
  uint64_t appended = 0;
  uint64_t markedAt = 0;
  uint64_t doneAt = 0;
  std::array<FreeMark, kFreeMarks> freeMarks;
  /// The ring slot the next mark uses.
  size_t nextFreeMark = 0;
  /// Counter-based event `queue_synchronize` records and waits on, so the
  /// wait doesn't hold `mutex`; null without counter-based events.
  ze_event_handle_t syncEvent = nullptr;
  /// Launch profiling (MOJO_INTEL_PROFILE, Profile.h); null when off.
  /// Guarded by `mutex`.
  std::unique_ptr<M::Intel::Profiling::QueueProfile> profile;
  /// Synchronous frees appending to this queue; `queue_destroy` waits for
  /// none.
  /// Guarded by the context's `queuesMutex`.
  int drainers = 0;
};

/// A HAL event: a host-visible counter-based Level Zero event, or, where the
/// driver lacks those, an ordinary event in a pool of its own.
struct M_driver_event {
  /// Null for a counter-based event.
  ze_event_pool_handle_t pool = nullptr;
  ze_event_handle_t event = nullptr;
  /// A never-recorded event reports complete, like a fresh CUDA event; the
  /// Level Zero event itself starts unsignaled and a host wait would block.
  std::atomic<bool> recorded{false};
};

/// A Level Zero module, shared by its bundle and the functions loaded from
/// it and destroyed when the last of them lets go. Level Zero requires a
/// module's kernels destroyed before the module, but MAX's client may unload
/// a bundle first: Mojo destroys a `RuntimeBundle` once it's no longer used,
/// while its functions stay loaded (CUDA allows that order).
struct LoadedModule {
  const M::Intel::ZeApi *api = nullptr;
  ze_module_handle_t handle = nullptr;
  /// The names the SPIR-V module exports. Level Zero can't tell a module
  /// without a global from a failed lookup (both may return
  /// ZE_RESULT_ERROR_UNSUPPORTED_FEATURE), so callers check this first.
  std::vector<std::string> exported;
  LoadedModule(const M::Intel::ZeApi &api, ze_module_handle_t handle)
      : api(&api), handle(handle) {}
  LoadedModule(const LoadedModule &) = delete;
  LoadedModule &operator=(const LoadedModule &) = delete;
  ~LoadedModule() {
    if (handle != nullptr)
      api->zeModuleDestroy(handle);
  }
};

/// A loaded SPIR-V module (built for the device by IGC at load time).
struct M_driver_runtime_bundle {
  std::shared_ptr<LoadedModule> module;
  /// Kernels whose last parameter is a `Workgroup` pointer: the dynamic shared
  /// memory the compiler plugin appends, which a launch sizes rather than
  /// passes.
  std::set<std::string> dynamicSharedKernels;
};

/// A kernel in a loaded module. Level Zero keeps arguments and group size on
/// the kernel object and captures them when a launch is appended, so a launch
/// holds `mutex` from setting them to appending.
struct M_driver_function {
  /// The context the module was loaded into; launches must use its queues.
  M_driver_context *context = nullptr;
  ze_kernel_handle_t kernel = nullptr;
  /// Keeps the module alive until the kernel is destroyed.
  std::shared_ptr<LoadedModule> module;
  std::mutex mutex;
  /// The group size last set, to skip redundant zeKernelSetGroupSize calls.
  uint32_t groupX = 0, groupY = 0, groupZ = 0;
  /// The kernel's parameter count, including dynamic shared memory.
  uint32_t numArgs = 0;
  /// Whether the last parameter is dynamic shared memory, sized per launch
  /// from `shared_mem_bytes` rather than passed.
  bool dynamicShared = false;
  /// The function's id in the launch profile (Profile.h).
  uint32_t profileId = 0;
};

/// A USM allocation (device or host), or a wrapped device address.
struct M_driver_memory {
  void *ptr = nullptr;
  uint64_t size = 0;
  /// The bytes behind `ptr` when allocated here (`size` rounded up to its
  /// cache class), else 0.
  uint64_t capacity = 0;
  bool host = false;
  /// Freeing the handle frees `ptr` only when owning.
  bool owning = true;
  /// Whether `capacity` is counted in the device's `allocatedBytes`.
  bool counted = false;
  /// A block handed out again while its last free waited for `reuseQueue`'s
  /// first `reuseAfter` commands: work appended to that
  /// queue later runs after them, but the context's synchronous copies and
  /// fills must wait for them first. Null once they are known done.
  M_driver_queue *reuseQueue = nullptr;
  uint64_t reuseAfter = 0;
};

namespace M::Intel {

/// Records `message` as an ephemeral status on `driver` and returns its code.
M_driver_status_t fail(M_driver_driver &driver, const std::string &message);

/// `fail` for a Level Zero call that returned `result`.
M_driver_status_t failZe(M_driver_driver &driver, const char *call,
                         ze_result_t result);

} // namespace M::Intel

#endif // MOJO_INTEL_DRIVER_PLUGIN_H
