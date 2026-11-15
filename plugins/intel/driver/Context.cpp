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
// HAL contexts, USM memory and synchronous copies and fills on Level Zero.
// A context is a ze_context on one device plus an in-order immediate command
// list on the device's first command queue group, used for the synchronous
// operations (append, then zeCommandListHostSynchronize). Device memory is
// zeMemAllocDevice, pinned host memory zeMemAllocHost. Level Zero has no
// per-thread current context, so "current" is plugin bookkeeping.
//
// Queues are in-order immediate command lists; events are host-visible Level
// Zero events, recorded with a barrier that signals after all prior work.
//
//===----------------------------------------------------------------------===//

#include "IntelGraphs.h"
#include "Plugin.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using M::Intel::fail;
using M::Intel::failZe;
using M::Intel::ZeApi;
namespace Profiling = M::Intel::Profiling;

namespace {

/// Alignment for device and host allocations, in bytes.
constexpr size_t kAllocAlignment = 256;

/// Pageable host-to-device copies below this size are staged through pinned
/// memory; larger ones copy directly and wait.
constexpr uint64_t kStageLimit = 4u << 20;

/// The thread's "current" driver-level context (a ze_context_handle_t).
thread_local M_driver_native_handle_t currentDriverContext = 0;

const ZeApi &ze(const M_driver_context &context) {
  return *context.device->driver->ze;
}

M_driver_driver &driverOf(const M_driver_context &context) {
  return *context.device->driver;
}

/// Checks `view` lies inside its allocation; null views are bad memory. A
/// zero-byte allocation has no memory handle (allocate returns null), so an
/// empty view of none is an empty view (MAX copies a batch of zero rows).
M_driver_status_t checkView(M_driver_memory_view_t view) {
  if (view == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (view->memory == nullptr)
    return view->size == 0 && view->offset == 0 ? M_DRIVER_STATUS_SUCCESS
                                                : M_DRIVER_STATUS_BAD_MEMORY;
  if (view->offset > view->memory->size ||
      view->size > view->memory->size - view->offset)
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;
  return M_DRIVER_STATUS_SUCCESS;
}

char *address(M_driver_memory_view_t view) {
  if (view->memory == nullptr)
    return nullptr;
  return static_cast<char *>(view->memory->ptr) + view->offset;
}

/// How often waits ask whether the device is lost. The query is an ioctl of
/// about 4.5 us on the B70; asked at every wait, it took a launch + sync from
/// 6.6 to 11 us.
constexpr std::chrono::milliseconds kDeviceStatusInterval{1};

/// The outcome of `call`, a finished Level Zero wait, which also fails once
/// the device is lost. After a kernel page-faults, the xe driver resets the
/// engine and compute-runtime may report the interrupted work complete; only
/// the device's status tells (seen on the B70). It is asked when a wait fails
/// and at most every `kDeviceStatusInterval` otherwise, so a fault is reported
/// by a later wait, as CUDA reports an illegal address; from then on every
/// wait fails.
M_driver_status_t waited(M_driver_context &context, const char *call,
                         ze_result_t result) {
  M_driver_device &device = *context.device;
  int64_t now = std::chrono::steady_clock::now().time_since_epoch().count();
  int64_t interval =
      std::chrono::steady_clock::duration(kDeviceStatusInterval).count();
  if (!device.lost && (result != ZE_RESULT_SUCCESS ||
                       now - device.statusCheckedAt.load() >= interval)) {
    device.statusCheckedAt = now;
    if (ze(context).zeDeviceGetStatus(device.zeDevice) != ZE_RESULT_SUCCESS)
      device.lost = true;
  }
  if (device.lost)
    return fail(driverOf(context),
                "the GPU was lost: a kernel faulted or the GPU was reset "
                "(`journalctl -k` shows why)");
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(context), call, result);
  return M_DRIVER_STATUS_SUCCESS;
}

/// Waits for the work on the context's list.
M_driver_status_t waitSyncList(M_driver_context &context) {
  return waited(
      context, "zeCommandListHostSynchronize",
      ze(context).zeCommandListHostSynchronize(context.syncList, UINT64_MAX));
}

/// Appends a copy on the context's list and waits for it.
M_driver_status_t copySync(M_driver_context &context, void *dst,
                           const void *src, uint64_t size) {
  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS;
  std::lock_guard<std::mutex> lock(context.syncMutex);
  ze_result_t result = ze(context).zeCommandListAppendMemoryCopy(
      context.syncList, dst, src, size, nullptr, 0, nullptr);
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(context), "zeCommandListAppendMemoryCopy", result);
  return waitSyncList(context);
}

/// Appends a pattern fill on the context's list and waits for it.
M_driver_status_t fillSync(M_driver_context &context,
                           M_driver_memory_view_t dst, const void *pattern,
                           size_t patternSize) {
  if (dst->size == 0)
    return M_DRIVER_STATUS_SUCCESS;
  std::lock_guard<std::mutex> lock(context.syncMutex);
  ze_result_t result = ze(context).zeCommandListAppendMemoryFill(
      context.syncList, address(dst), pattern, patternSize, dst->size, nullptr,
      0, nullptr);
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(context), "zeCommandListAppendMemoryFill", result);
  return waitSyncList(context);
}

/// The Sysman handles of `device`'s local memory modules (what xpu-smi
/// reports), or none when Sysman can't see the device.
std::vector<zes_mem_handle_t> findSysmanMemory(const ZeApi &api,
                                               const M_driver_device &device) {
  std::vector<zes_mem_handle_t> local;
  if (api.zesInit(0) != ZE_RESULT_SUCCESS)
    return local;
  uint32_t driverCount = 0;
  if (api.zesDriverGet(&driverCount, nullptr) != ZE_RESULT_SUCCESS ||
      driverCount == 0)
    return local;
  std::vector<zes_driver_handle_t> drivers(driverCount);
  if (api.zesDriverGet(&driverCount, drivers.data()) != ZE_RESULT_SUCCESS)
    return local;
  zes_uuid_t uuid = {};
  std::memcpy(uuid.id, device.props.uuid.id, sizeof(uuid.id));
  for (zes_driver_handle_t driver : drivers) {
    zes_device_handle_t sysmanDevice = nullptr;
    ze_bool_t onSubdevice = false;
    uint32_t subdeviceId = 0;
    if (api.zesDriverGetDeviceByUuidExp(driver, uuid, &sysmanDevice,
                                        &onSubdevice,
                                        &subdeviceId) != ZE_RESULT_SUCCESS)
      continue;
    uint32_t moduleCount = 0;
    if (api.zesDeviceEnumMemoryModules(sysmanDevice, &moduleCount, nullptr) !=
            ZE_RESULT_SUCCESS ||
        moduleCount == 0)
      return local;
    std::vector<zes_mem_handle_t> modules(moduleCount);
    if (api.zesDeviceEnumMemoryModules(sysmanDevice, &moduleCount,
                                       modules.data()) != ZE_RESULT_SUCCESS)
      return local;
    for (zes_mem_handle_t module : modules) {
      zes_mem_properties_t props = {};
      props.stype = ZES_STRUCTURE_TYPE_MEM_PROPERTIES;
      if (api.zesMemoryGetProperties(module, &props) == ZE_RESULT_SUCCESS &&
          props.location == ZES_MEM_LOC_DEVICE)
        local.push_back(module);
    }
    return local;
  }
  return local;
}

/// Free device-local memory from Sysman, or nullopt. The memory modules are
/// found once per device; each query reads only their state.
std::optional<uint64_t> sysmanFreeBytes(const ZeApi &api,
                                        M_driver_device &device) {
  if (!api.hasSysman)
    return std::nullopt;
  std::call_once(device.sysmanOnce,
                 [&] { device.sysmanMemory = findSysmanMemory(api, device); });
  if (device.sysmanMemory.empty())
    return std::nullopt;
  uint64_t freeBytes = 0;
  for (zes_mem_handle_t module : device.sysmanMemory) {
    zes_mem_state_t state = {};
    state.stype = ZES_STRUCTURE_TYPE_MEM_STATE;
    if (api.zesMemoryGetState(module, &state) != ZE_RESULT_SUCCESS)
      return std::nullopt;
    freeBytes += state.free;
  }
  return freeBytes;
}

/// Whether `ptr` is USM memory Level Zero knows (host, device or shared), as
/// opposed to ordinary pageable host memory.
bool isUsm(const M_driver_context &context, const void *ptr) {
  ze_memory_allocation_properties_t props = {};
  props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
  return ze(context).zeMemGetAllocProperties(context.zeContext, ptr, &props,
                                             nullptr) == ZE_RESULT_SUCCESS &&
         props.type != ZE_MEMORY_TYPE_UNKNOWN;
}

/// Whether `ptr` is aligned to a `patternSize`-byte fill pattern, as
/// zeCommandListAppendMemoryFill requires.
bool alignedForFill(const void *ptr, uint64_t patternSize) {
  return reinterpret_cast<uintptr_t>(ptr) % patternSize == 0;
}

/// Debugging aid: with MOJO_INTEL_DEBUG_ALLOC_FILL=<byte> every new
/// allocation is filled with that byte before it's returned, so a read of
/// memory nothing wrote shows up (0xff makes NaNs in floating point). -1 when
/// unset.
int debugAllocFill() {
  static const int fill = [] {
    const char *text = std::getenv("MOJO_INTEL_DEBUG_ALLOC_FILL");
    if (text == nullptr || *text == '\0')
      return -1;
    return static_cast<int>(std::strtol(text, nullptr, 0) & 0xff);
  }();
  return fill;
}

/// Debugging aid: MOJO_INTEL_DEBUG_DRAIN_FREES=1 makes a synchronous free
/// drain every queue of the context first.
bool debugDrainFrees() {
  static const bool drain = [] {
    const char *text = std::getenv("MOJO_INTEL_DEBUG_DRAIN_FREES");
    return text != nullptr && std::strcmp(text, "1") == 0;
  }();
  return drain;
}

/// The most bytes a context's allocation cache holds:
/// MOJO_INTEL_ALLOC_CACHE_MB, default 4 GiB; 0 turns the cache off.
uint64_t cacheLimit() {
  static const uint64_t limit = [] {
    const char *text = std::getenv("MOJO_INTEL_ALLOC_CACHE_MB");
    if (text == nullptr || *text == '\0')
      return uint64_t(4) << 30;
    return uint64_t(std::strtoull(text, nullptr, 0)) << 20;
  }();
  return limit;
}

/// The capacity an allocation of `size` bytes gets, so that freed blocks
/// fit later requests of similar size: 256-byte multiples up to 4 KiB, then
/// eighths of the next power of two (up to 25% more than asked, just above
/// a power of two). A block the cache could never hold (larger than its
/// limit, or the cache off) is not rounded beyond 256 bytes.
uint64_t capacityFor(uint64_t size) {
  uint64_t exact = (size + 255) & ~uint64_t(255);
  if (size <= 4096 || exact > cacheLimit())
    return exact;
  int top = 63 - __builtin_clzll(size - 1);
  uint64_t step = uint64_t(1) << (top - 2);
  return (size + step - 1) & ~(step - 1);
}

/// A cached block of exactly `capacity` bytes, or null.
void *takeCached(M_driver_context &context, uint64_t capacity, bool host) {
  std::lock_guard<std::mutex> lock(context.cacheMutex);
  auto &cache = host ? context.cachedHost : context.cachedDevice;
  auto it = cache.find(capacity);
  if (it == cache.end())
    return nullptr;
  void *ptr = it->second;
  cache.erase(it);
  context.cachedBytes -= capacity;
  return ptr;
}

/// Keeps a released block for reuse; false when the cache is full or off.
bool cacheBlock(M_driver_context &context, void *ptr, uint64_t capacity,
                bool host) {
  std::lock_guard<std::mutex> lock(context.cacheMutex);
  if (context.cachedBytes + capacity > cacheLimit())
    return false;
  (host ? context.cachedHost : context.cachedDevice).emplace(capacity, ptr);
  context.cachedBytes += capacity;
  return true;
}

/// MOJO_INTEL_DEBUG_MEMLOG=<file>: one line per USM allocation and free
/// (`+`/`-`, address, bytes, h|d, where), to match a GPU fault's address
/// with the memory that was there. Off by default.
void memLog(char op, const void *ptr, uint64_t bytes, bool host,
            const char *where) {
  static FILE *file = [] {
    const char *path = std::getenv("MOJO_INTEL_DEBUG_MEMLOG");
    return path != nullptr && *path != 0 ? std::fopen(path, "w") : nullptr;
  }();
  if (file == nullptr)
    return;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  std::fprintf(file, "%c %p %llu %c %s\n", op, ptr, (unsigned long long)bytes,
               host ? 'h' : 'd', where);
  std::fflush(file);
}

/// Records or forgets a block in `context.blocks`.
void trackBlock(M_driver_context &context, const void *ptr, uint64_t bytes) {
  std::lock_guard<std::mutex> lock(context.blocksMutex);
  context.blocks[reinterpret_cast<uintptr_t>(ptr)] = bytes;
}
void untrackBlock(M_driver_context &context, const void *ptr) {
  std::lock_guard<std::mutex> lock(context.blocksMutex);
  context.blocks.erase(reinterpret_cast<uintptr_t>(ptr));
}

/// Whether [address, address + size) lies inside one tracked block.
bool insideBlock(M_driver_context &context, uint64_t address, uint64_t size) {
  std::lock_guard<std::mutex> lock(context.blocksMutex);
  auto it = context.blocks.upper_bound(address);
  if (it == context.blocks.begin())
    return false;
  --it;
  return address + size <= it->first + it->second;
}

/// Frees every cached block (out of memory, or the context goes away).
void flushCache(M_driver_context &context) {
  std::lock_guard<std::mutex> lock(context.cacheMutex);
  for (auto &[capacity, ptr] : context.cachedDevice) {
    memLog('-', ptr, capacity, false, "flushCache");
    untrackBlock(context, ptr);
    ze(context).zeMemFree(context.zeContext, ptr);
    context.device->allocatedBytes -= capacity;
  }
  for (auto &[capacity, ptr] : context.cachedHost) {
    memLog('-', ptr, capacity, true, "flushCache");
    untrackBlock(context, ptr);
    ze(context).zeMemFree(context.zeContext, ptr);
  }
  context.cachedDevice.clear();
  context.cachedHost.clear();
  context.cachedBytes = 0;
}

/// A new resident USM block of `capacity` bytes. A block over the device's
/// maxMemAllocSize (4 GiB on the integrated GPUs, where MAX's KV cache and
/// logits buffers pass it) asks for the relaxed allocation limits
/// (ZE_experimental_relaxed_allocation_limits); without the request Level
/// Zero answers ZE_RESULT_ERROR_UNSUPPORTED_SIZE.
ze_result_t allocateFresh(M_driver_context &context, uint64_t capacity,
                          bool host, void **ptr) {
  ze_result_t result;
  ze_relaxed_allocation_limits_ext_desc_t relaxed = {
      ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXT_DESC, nullptr,
      ZE_RELAXED_ALLOCATION_LIMITS_EXT_FLAG_MAX_SIZE};
  const void *next =
      capacity > context.device->props.maxMemAllocSize ? &relaxed : nullptr;
  if (host) {
    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                     next, 0};
    result = ze(context).zeMemAllocHost(context.zeContext, &desc, capacity,
                                        kAllocAlignment, ptr);
  } else {
    ze_device_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                       next, 0, 0};
    result = ze(context).zeMemAllocDevice(context.zeContext, &desc, capacity,
                                          kAllocAlignment,
                                          context.device->zeDevice, ptr);
  }
  if (result != ZE_RESULT_SUCCESS)
    return result;
  // Resident from the start, as CUDA and HIP memory is: otherwise Level Zero
  // maps an allocation only once a command names it, and a kernel that
  // reaches it first, through a pointer inside a struct (MAX's closures
  // capture buffers that way), loses its writes or page-faults.
  result = ze(context).zeContextMakeMemoryResident(
      context.zeContext, context.device->zeDevice, *ptr, capacity);
  if (result != ZE_RESULT_SUCCESS)
    ze(context).zeMemFree(context.zeContext, *ptr);
  else {
    trackBlock(context, *ptr, capacity);
    memLog('+', *ptr, capacity, host, "allocateFresh");
  }
  return result;
}

bool outOfMemory(ze_result_t result) {
  return result == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY ||
         result == ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY;
}

M_driver_status_t allocateUntimed(M_driver_context_t context, uint64_t size,
                                  bool host, M_driver_memory_t *memory,
                                  bool *cached);

/// Retires the pending frees whose queue work has completed on every queue
/// of `context` (one event query per queue with frees pending), so their
/// blocks join the cache. Called on a cache miss: frees are otherwise
/// retired only by later frees and synchronizes, and a graph execution
/// allocates its intermediates before it frees any, so a block whose work
/// finished meanwhile would be passed over for a fresh allocation.
void reclaimCompletedFrees(M_driver_context &context);

M_driver_memory *takePendingFree(M_driver_context &context, uint64_t capacity);
M_driver_memory *takeGraphPool(M_driver_context &context, uint64_t capacity,
                               bool host);
M_driver_status_t releaseGraphPoolIfUnused(M_driver_context &context);
bool capturing(M_driver_context &context);

M_driver_status_t allocate(M_driver_context_t context, uint64_t size, bool host,
                           M_driver_memory_t *memory) {
  bool cached = false;
  if (!Profiling::enabled())
    return allocateUntimed(context, size, host, memory, &cached);
  uint64_t enter = Profiling::now();
  M_driver_status_t status =
      allocateUntimed(context, size, host, memory, &cached);
  Profiling::allocation(cached ? 'a' : 'A', size, host, enter,
                        Profiling::now());
  return status;
}

M_driver_status_t allocateUntimed(M_driver_context_t context, uint64_t size,
                                  bool host, M_driver_memory_t *memory,
                                  bool *cached) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (memory == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (size == 0) {
    // Nothing to allocate, so nothing fresh either.
    *memory = nullptr;
    *cached = true;
    return M_DRIVER_STATUS_SUCCESS;
  }
  uint64_t capacity = capacityFor(size);
  if (M_driver_memory *pooled = takeGraphPool(*context, capacity, host)) {
    pooled->size = size;
    pooled->graphOwned = true;
    *memory = pooled;
    *cached = true;
    return M_DRIVER_STATUS_SUCCESS;
  }
  void *ptr = takeCached(*context, capacity, host);
  if (ptr == nullptr && cacheLimit() != 0) {
    reclaimCompletedFrees(*context);
    ptr = takeCached(*context, capacity, host);
  }
  if (ptr == nullptr && !host) {
    if (M_driver_memory *reused = takePendingFree(*context, capacity)) {
      reused->size = size;
      *memory = reused;
      *cached = true;
      return M_DRIVER_STATUS_SUCCESS;
    }
  }
  *cached = ptr != nullptr;
  if (ptr == nullptr) {
    ze_result_t result = allocateFresh(*context, capacity, host, &ptr);
    if (outOfMemory(result)) {
      // Cached blocks may be what fills the memory.
      flushCache(*context);
      result = allocateFresh(*context, capacity, host, &ptr);
    }
    if (outOfMemory(result))
      return M_DRIVER_STATUS_OUT_OF_MEMORY;
    if (result != ZE_RESULT_SUCCESS)
      return failZe(driverOf(*context),
                    host ? "zeMemAllocHost" : "zeMemAllocDevice", result);
    if (!host)
      context->device->allocatedBytes += capacity;
  }
  ze_result_t result = ZE_RESULT_SUCCESS;
  if (int fill = debugAllocFill(); fill >= 0) {
    if (host) {
      std::memset(ptr, fill, size);
    } else {
      unsigned char pattern = static_cast<unsigned char>(fill);
      std::lock_guard<std::mutex> lock(context->syncMutex);
      result = ze(*context).zeCommandListAppendMemoryFill(
          context->syncList, ptr, &pattern, 1, size, nullptr, 0, nullptr);
      if (result == ZE_RESULT_SUCCESS)
        result = ze(*context).zeCommandListHostSynchronize(context->syncList,
                                                           UINT64_MAX);
      if (result != ZE_RESULT_SUCCESS) {
        untrackBlock(*context, ptr);
        ze(*context).zeMemFree(context->zeContext, ptr);
        context->device->allocatedBytes -= capacity;
        return failZe(driverOf(*context), "zeCommandListAppendMemoryFill",
                      result);
      }
    }
  }
  auto handle = std::make_unique<M_driver_memory>();
  handle->ptr = ptr;
  handle->size = size;
  handle->capacity = capacity;
  handle->host = host;
  handle->counted = !host;
  handle->graphOwned = capturing(*context);
  *memory = handle.release();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t release(M_driver_context_t context,
                          M_driver_memory_t memory) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;
  std::unique_ptr<M_driver_memory> handle(memory);
  if (!handle->owning)
    return M_DRIVER_STATUS_SUCCESS;
  uint64_t enter = Profiling::enabled() ? Profiling::now() : 0;
  // Released only once no queued work uses it (the callers' contract), so
  // the block can serve the next allocation of its class at once.
  if (handle->capacity != 0 &&
      cacheBlock(*context, handle->ptr, handle->capacity, handle->host)) {
    if (enter != 0)
      Profiling::allocation('r', handle->size, handle->host, enter,
                            Profiling::now());
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (handle->counted)
    context->device->allocatedBytes -= handle->capacity;
  memLog('-', handle->ptr, handle->capacity, handle->host, "release");
  untrackBlock(*context, handle->ptr);
  ze_result_t result = ze(*context).zeMemFree(context->zeContext, handle->ptr);
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(*context), "zeMemFree", result);
  if (enter != 0)
    Profiling::allocation('R', handle->size, handle->host, enter,
                          Profiling::now());
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t drainQueues(M_driver_context &context);

} // namespace

M_driver_status_t M_driver_context_create(M_driver_device_t device,
                                          M_driver_context_t *context) {
  if (device == nullptr)
    return M_DRIVER_STATUS_BAD_DEVICE;
  if (context == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  M_driver_driver &driver = *device->driver;
  const ZeApi &api = *driver.ze;
  auto created = std::make_unique<M_driver_context>();
  created->device = device;
  ze_context_desc_t desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  ze_result_t result =
      api.zeContextCreate(device->zeDriver, &desc, &created->zeContext);
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driver, "zeContextCreate", result);
  ze_command_queue_desc_t queue = {};
  queue.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  queue.ordinal = 0; // the compute group on current Intel GPUs
  // The hint sends host transfers to the copy engine: about 35% faster for
  // small copies on the B70, while device-to-device copies stay on compute.
  queue.flags =
      ZE_COMMAND_QUEUE_FLAG_IN_ORDER | ZE_COMMAND_QUEUE_FLAG_COPY_OFFLOAD_HINT;
  queue.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  queue.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  result = api.zeCommandListCreateImmediate(
      created->zeContext, device->zeDevice, &queue, &created->syncList);
  if (result != ZE_RESULT_SUCCESS) {
    api.zeContextDestroy(created->zeContext);
    return failZe(driver, "zeCommandListCreateImmediate", result);
  }
  ze_host_mem_alloc_desc_t host = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                   nullptr, 0};
  void *flag = nullptr;
  result = api.zeMemAllocHost(created->zeContext, &host, sizeof(int32_t),
                              sizeof(int32_t), &flag);
  if (result != ZE_RESULT_SUCCESS) {
    api.zeCommandListDestroy(created->syncList);
    api.zeContextDestroy(created->zeContext);
    return failZe(driver, "zeMemAllocHost", result);
  }
  created->abortFlag = static_cast<int32_t *>(flag);
  *created->abortFlag = 0;
  // Kernels reach the flag through a module global, not an argument, so
  // Level Zero doesn't make it resident for a launch: the store faults
  // (Engine memory CAT error) and the job times out. It is made resident
  // once, for the context's lifetime, like every allocation (`allocate`).
  result = api.zeContextMakeMemoryResident(created->zeContext, device->zeDevice,
                                           flag, sizeof(int32_t));
  if (result != ZE_RESULT_SUCCESS) {
    api.zeMemFree(created->zeContext, flag);
    api.zeCommandListDestroy(created->syncList);
    api.zeContextDestroy(created->zeContext);
    return failZe(driver, "zeContextMakeMemoryResident", result);
  }
  *context = created.release();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context_destroy(M_driver_context_t context) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  std::unique_ptr<M_driver_context> owned(context);
  const ZeApi &api = ze(*owned);
  if (currentDriverContext ==
      reinterpret_cast<M_driver_native_handle_t>(owned->zeContext))
    currentDriverContext = 0;
  // Kernels may still be running on a queue the caller didn't destroy
  // (queue_destroy has drained the others), and they reach the
  // abort flag and the heap through module globals: both are freed only
  // once those queues are idle.
  drainQueues(*owned);
  {
    // Graphs the caller didn't destroy: their pool goes with the context.
    std::lock_guard<std::mutex> lock(owned->captureMutex);
    owned->liveGraphs = 0;
    owned->capturing = 0;
  }
  releaseGraphPoolIfUnused(*owned);
  flushCache(*owned);
  api.zeMemFree(owned->zeContext, owned->abortFlag);
  if (owned->heap != nullptr) {
    api.zeMemFree(owned->zeContext, owned->heap);
    owned->device->allocatedBytes -= owned->heapBytes;
  }
  ze_result_t listResult = api.zeCommandListDestroy(owned->syncList);
  ze_result_t contextResult = api.zeContextDestroy(owned->zeContext);
  if (listResult != ZE_RESULT_SUCCESS)
    return failZe(driverOf(*owned), "zeCommandListDestroy", listResult);
  if (contextResult != ZE_RESULT_SUCCESS)
    return failZe(driverOf(*owned), "zeContextDestroy", contextResult);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context_set_current(M_driver_context_t context) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  currentDriverContext =
      reinterpret_cast<M_driver_native_handle_t>(context->zeContext);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context_get_current_driver_context(
    M_driver_context_t context, M_driver_native_handle_t *driver_context) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (driver_context == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  *driver_context = currentDriverContext;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context_set_current_driver_context(
    M_driver_context_t context, M_driver_native_handle_t driver_context) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  currentDriverContext = driver_context;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context_memory_info(M_driver_context_t context,
                                               uint64_t *free,
                                               uint64_t *total) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (free == nullptr || total == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  const ZeApi &api = ze(*context);
  M_driver_device &device = *context->device;
  uint32_t count = 0;
  ze_result_t result =
      api.zeDeviceGetMemoryProperties(device.zeDevice, &count, nullptr);
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(*context), "zeDeviceGetMemoryProperties", result);
  std::vector<ze_device_memory_properties_t> props(count);
  for (ze_device_memory_properties_t &prop : props)
    prop.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
  result =
      api.zeDeviceGetMemoryProperties(device.zeDevice, &count, props.data());
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(*context), "zeDeviceGetMemoryProperties", result);
  uint64_t totalBytes = 0;
  for (const ze_device_memory_properties_t &prop : props)
    totalBytes += prop.totalSize;
  // Sysman sees every process's use; without it, count only this plugin's.
  // Blocks in the allocation cache are free to this context's callers (an
  // allocation that doesn't fit flushes the cache), so they count as free.
  uint64_t cached = 0;
  {
    std::lock_guard<std::mutex> lock(context->cacheMutex);
    for (const auto &[capacity, ptr] : context->cachedDevice)
      cached += capacity;
  }
  uint64_t used = device.allocatedBytes.load();
  used = used > cached ? used - cached : 0;
  // Sysman's figure doesn't always include this process's own allocations
  // (in-process it reported the whole memory free with 300 MiB held):
  // bound it by this plugin's own count, so the caller sees at
  // least its own use and, through Sysman, other processes' too.
  uint64_t ownFree = used < totalBytes ? totalBytes - used : 0;
  std::optional<uint64_t> sysman = sysmanFreeBytes(api, device);
  uint64_t freeBytes =
      sysman.has_value() ? std::min(*sysman + cached, ownFree) : ownFree;
  *total = totalBytes;
  *free = freeBytes < totalBytes ? freeBytes : totalBytes;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_memory_alloc_sync(M_driver_context_t context,
                                             uint64_t size,
                                             M_driver_memory_t *memory) {
  return allocate(context, size, /*host=*/false, memory);
}

namespace {
M_driver_status_t releaseAfterQueues(M_driver_context_t context,
                                     M_driver_memory_t memory);
void waitForReuse(M_driver_memory *memory);
} // namespace

M_driver_status_t M_driver_memory_free_sync(M_driver_context_t context,
                                            M_driver_memory_t memory) {
  return releaseAfterQueues(context, memory);
}

M_driver_status_t M_driver_memory_alloc_pinned(M_driver_context_t context,
                                               uint64_t size,
                                               M_driver_memory_t *memory) {
  return allocate(context, size, /*host=*/true, memory);
}

M_driver_status_t M_driver_memory_free_pinned(M_driver_context_t context,
                                              M_driver_memory_t memory) {
  return releaseAfterQueues(context, memory);
}

M_driver_status_t M_driver_memory_wrap(M_driver_context_t context,
                                       uint64_t address, uint64_t size,
                                       bool owning, M_driver_memory_t *memory) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (memory == nullptr || address == 0)
    return M_DRIVER_STATUS_INVALID_ARG;
  // Resident, as `allocate` makes its memory: the region may be another
  // library's allocation in this context (whose handle
  // context_get_current_driver_context gives out), which a kernel reaching
  // it through a struct would otherwise fault on. A region inside a block
  // this plugin allocated is resident already.
  if (size > 0 && !insideBlock(*context, address, size)) {
    ze_result_t result = ze(*context).zeContextMakeMemoryResident(
        context->zeContext, context->device->zeDevice,
        reinterpret_cast<void *>(address), size);
    if (result != ZE_RESULT_SUCCESS)
      return failZe(driverOf(*context), "zeContextMakeMemoryResident", result);
  }
  auto handle = std::make_unique<M_driver_memory>();
  handle->ptr = reinterpret_cast<void *>(address);
  handle->size = size;
  handle->owning = owning;
  *memory = handle.release();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_memory_unwrap(M_driver_context_t context,
                                         M_driver_memory_t memory,
                                         uint64_t *address) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (address == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (memory->counted) {
    context->device->allocatedBytes -= memory->capacity;
    memory->counted = false;
  }
  memory->owning = false;
  *address = reinterpret_cast<uint64_t>(memory->ptr);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_memory_property(M_driver_memory_t memory,
                                           const char *property_name,
                                           void *value) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (property_name == nullptr || value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (std::strcmp(property_name, M_DRIVER_MEMORY_PROPERTY_ADDRESS) == 0) {
    *static_cast<void **>(value) = memory->ptr;
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (std::strcmp(property_name, M_DRIVER_MEMORY_PROPERTY_HOST_ADDRESS) == 0) {
    if (!memory->host)
      return M_DRIVER_STATUS_INVALID_ARG;
    *static_cast<void **>(value) = memory->ptr;
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_driver_status_t M_driver_copy_to_device_sync(M_driver_context_t context,
                                               M_driver_memory_view_t dst,
                                               const void *src) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (M_driver_status_t status = checkView(dst))
    return status;
  if (src == nullptr && dst->size != 0)
    return M_DRIVER_STATUS_INVALID_ARG;
  waitForReuse(dst->memory);
  return copySync(*context, address(dst), src, dst->size);
}

M_driver_status_t M_driver_copy_from_device_sync(M_driver_context_t context,
                                                 void *dst,
                                                 M_driver_memory_view_t src) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (M_driver_status_t status = checkView(src))
    return status;
  if (dst == nullptr && src->size != 0)
    return M_DRIVER_STATUS_INVALID_ARG;
  waitForReuse(src->memory);
  return copySync(*context, dst, address(src), src->size);
}

M_driver_status_t M_driver_copy_intra_device_sync(M_driver_context_t context,
                                                  M_driver_memory_view_t dst,
                                                  M_driver_memory_view_t src) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (M_driver_status_t status = checkView(dst))
    return status;
  if (M_driver_status_t status = checkView(src))
    return status;
  if (src->size < dst->size)
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;
  waitForReuse(dst->memory);
  waitForReuse(src->memory);
  return copySync(*context, address(dst), address(src), dst->size);
}

M_driver_status_t M_driver_set_memory_sync(M_driver_context_t context,
                                           M_driver_memory_view_t dst,
                                           uint8_t value) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (M_driver_status_t status = checkView(dst))
    return status;
  waitForReuse(dst->memory);
  return fillSync(*context, dst, &value, sizeof(value));
}

M_driver_status_t M_driver_fill_sync(M_driver_context_t context,
                                     M_driver_memory_view_t dst, uint64_t value,
                                     uint64_t value_size) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (M_driver_status_t status = checkView(dst))
    return status;
  if ((value_size != 1 && value_size != 2 && value_size != 4 &&
       value_size != 8) ||
      dst->size % value_size != 0 || !alignedForFill(address(dst), value_size))
    return M_DRIVER_STATUS_INVALID_ARG;
  // The low `value_size` bytes of `value`, in host (little-endian) order.
  unsigned char pattern[8];
  std::memcpy(pattern, &value, sizeof(pattern));
  waitForReuse(dst->memory);
  return fillSync(*context, dst, pattern, value_size);
}

//===----------------------------------------------------------------------===//
// Queues and events
//===----------------------------------------------------------------------===//

namespace {

const ZeApi &ze(const M_driver_queue &queue) { return ze(*queue.context); }

M_driver_driver &driverOf(const M_driver_queue &queue) {
  return driverOf(*queue.context);
}

/// The timestamp event for the next command on `queue` when profiling, else
/// null. Caller holds `queue.mutex`.
ze_event_handle_t profileEvent(M_driver_queue &queue) {
  // A captured command can't signal an event the host harvests.
  return queue.profile != nullptr && !queue.capturing
             ? queue.profile->nextEvent()
             : nullptr;
}

/// Whether a queue of `context` captures.
bool capturing(M_driver_context &context) {
  std::lock_guard<std::mutex> lock(context.captureMutex);
  return context.capturing != 0;
}

/// While a queue of `context` captures, or while a graph lives for memory
/// allocated during a capture, puts `memory` in the graph pool
/// (IntelGraphs.h) and returns true: the graph's launches reuse the
/// addresses the captured work saw, its outputs among them, which MAX
/// frees after the capture, so only later captures may reuse it.
bool keepForCapture(M_driver_context &context, M_driver_memory *memory) {
  std::lock_guard<std::mutex> lock(context.captureMutex);
  if (context.capturing == 0 &&
      !(memory->graphOwned && context.liveGraphs != 0))
    return false;
  context.graphPool.emplace(memory->capacity, memory);
  return true;
}

/// A pooled block of `capacity` for an allocation made while a queue
/// captures, or null.
M_driver_memory *takeGraphPool(M_driver_context &context, uint64_t capacity,
                               bool host) {
  std::lock_guard<std::mutex> lock(context.captureMutex);
  if (context.capturing == 0)
    return nullptr;
  auto [begin, end] = context.graphPool.equal_range(capacity);
  for (auto it = begin; it != end; ++it) {
    if (it->second->host != host)
      continue;
    M_driver_memory *memory = it->second;
    context.graphPool.erase(it);
    return memory;
  }
  return nullptr;
}

/// Releases the graph pool once no graph lives and no queue captures.
M_driver_status_t releaseGraphPoolIfUnused(M_driver_context &context) {
  std::multimap<uint64_t, M_driver_memory *> pool;
  {
    std::lock_guard<std::mutex> lock(context.captureMutex);
    if (context.liveGraphs != 0 || context.capturing != 0)
      return M_DRIVER_STATUS_SUCCESS;
    pool.swap(context.graphPool);
  }
  M_driver_status_t first = M_DRIVER_STATUS_SUCCESS;
  for (auto &[capacity, memory] : pool)
    if (M_driver_status_t status = release(&context, memory);
        status != M_DRIVER_STATUS_SUCCESS && first == M_DRIVER_STATUS_SUCCESS)
      first = status;
  return first;
}

/// Gives back the timestamp event `profileEvent` handed out when the command
/// could not be appended. Caller holds `queue.mutex`.
void abandonProfileEvent(M_driver_queue &queue) {
  if (queue.profile != nullptr)
    queue.profile->abandonEvent();
}

/// Appends a copy to `queue` without waiting.
M_driver_status_t appendCopy(M_driver_queue &queue, void *dst, const void *src,
                             uint64_t size) {
  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS;
  uint64_t enter = queue.profile != nullptr ? Profiling::now() : 0;
  std::lock_guard<std::mutex> lock(queue.mutex);
  ze_event_handle_t signal = profileEvent(queue);
  ze_result_t result = ze(queue).zeCommandListAppendMemoryCopy(
      queue.list, dst, src, size, signal, 0, nullptr);
  if (result != ZE_RESULT_SUCCESS) {
    abandonProfileEvent(queue);
    return failZe(driverOf(queue), "zeCommandListAppendMemoryCopy", result);
  }
  ++queue.appended;
  if (queue.profile != nullptr)
    queue.profile->transfer('c', size, enter, Profiling::now());
  return M_DRIVER_STATUS_SUCCESS;
}

/// Appends a pattern fill to `queue` without waiting.
M_driver_status_t appendFill(M_driver_queue &queue, M_driver_memory_view_t dst,
                             const void *pattern, size_t patternSize) {
  if (dst->size == 0)
    return M_DRIVER_STATUS_SUCCESS;
  uint64_t enter = queue.profile != nullptr ? Profiling::now() : 0;
  std::lock_guard<std::mutex> lock(queue.mutex);
  ze_event_handle_t signal = profileEvent(queue);
  ze_result_t result = ze(queue).zeCommandListAppendMemoryFill(
      queue.list, address(dst), pattern, patternSize, dst->size, signal, 0,
      nullptr);
  if (result != ZE_RESULT_SUCCESS) {
    abandonProfileEvent(queue);
    return failZe(driverOf(queue), "zeCommandListAppendMemoryFill", result);
  }
  ++queue.appended;
  if (queue.profile != nullptr)
    queue.profile->transfer('f', dst->size, enter, Profiling::now());
  return M_DRIVER_STATUS_SUCCESS;
}

/// A host-visible counter-based event, or null when the driver has none.
ze_event_handle_t createCounterEvent(const M_driver_context &context) {
  const ZeApi &api = ze(context);
  if (api.zeEventCounterBasedCreate == nullptr)
    return nullptr;
  ze_event_counter_based_desc_t desc = {};
  desc.stype = ZE_STRUCTURE_TYPE_EVENT_COUNTER_BASED_DESC;
  desc.flags = ZE_EVENT_COUNTER_BASED_FLAG_IMMEDIATE |
               ZE_EVENT_COUNTER_BASED_FLAG_HOST_VISIBLE;
  desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
  ze_event_handle_t event = nullptr;
  if (api.zeEventCounterBasedCreate(context.zeContext, context.device->zeDevice,
                                    &desc, &event) != ZE_RESULT_SUCCESS)
    return nullptr;
  return event;
}

/// Advances `queue.doneAt` past the free marks that have completed. Caller
/// holds `queue.mutex`.
void advanceDoneLocked(M_driver_queue &queue) {
  // The pending marks in the order they were appended: the queue runs in
  // order, so the first one not complete ends the scan.
  for (size_t i = 1; i <= M_driver_queue::kFreeMarks; ++i) {
    const M_driver_queue::FreeMark &mark =
        queue.freeMarks[(queue.nextFreeMark + i - 1) %
                        M_driver_queue::kFreeMarks];
    if (mark.event == nullptr || mark.at <= queue.doneAt)
      continue;
    if (ze(queue).zeEventQueryStatus(mark.event) != ZE_RESULT_SUCCESS)
      break;
    queue.doneAt = mark.at;
  }
}

/// Releases pending frees whose work has completed, or all of them when
/// `all` (the caller has drained the queue). Caller holds `queue.mutex`.
M_driver_status_t reclaimFreesLocked(M_driver_queue &queue, bool all) {
  M_driver_status_t firstError = M_DRIVER_STATUS_SUCCESS;
  auto &pending = queue.pendingFrees;
  if (pending.empty())
    return firstError;
  if (all)
    queue.doneAt = queue.appended;
  else
    advanceDoneLocked(queue);
  size_t kept = 0;
  for (M_driver_queue::PendingFree &entry : pending) {
    if (entry.after > queue.doneAt) {
      pending[kept++] = entry;
      continue;
    }
    // A synchronous free's memory waits for its entries on every queue.
    if (entry.sharers != nullptr && entry.sharers->fetch_sub(1) != 1)
      continue;
    M_driver_status_t status = release(queue.context, entry.memory);
    if (status != M_DRIVER_STATUS_SUCCESS &&
        firstError == M_DRIVER_STATUS_SUCCESS)
      firstError = status;
  }
  pending.resize(kept);
  return firstError;
}

void reclaimCompletedFrees(M_driver_context &context) {
  std::vector<M_driver_queue *> queues;
  {
    std::lock_guard<std::mutex> lock(context.queuesMutex);
    queues = context.queues;
    for (M_driver_queue *queue : queues)
      ++queue->drainers;
  }
  for (M_driver_queue *queue : queues) {
    std::lock_guard<std::mutex> lock(queue->mutex);
    // An allocation goes on whatever this finds: a release that fails here
    // (zeMemFree, for a block the full cache doesn't take) is not reported.
    (void)reclaimFreesLocked(*queue, /*all=*/false);
  }
  {
    std::lock_guard<std::mutex> lock(context.queuesMutex);
    for (M_driver_queue *queue : queues)
      --queue->drainers;
  }
  context.queuesIdle.notify_all();
}

/// A device block of `capacity` bytes whose free still waits for work on
/// the context's only queue, taken out of that free: the
/// queue is in order, so whatever uses the block next on it runs after the
/// work the free waited for, as with CUDA's stream-ordered reuse. MAX frees
/// a step's buffers while the host runs layers ahead of the device, and
/// without this a long prompt step holds every one of them at once. Only
/// with one queue (another queue's work wouldn't be ordered after it) and
/// only for blocks no other queue shares; the synchronous copies and fills
/// wait for `reuseAfter` (`waitForReuse`).
M_driver_memory *takePendingFree(M_driver_context &context, uint64_t capacity) {
  static const bool off = [] {
    const char *text = std::getenv("MOJO_INTEL_DEBUG_NO_FREE_REUSE");
    return text != nullptr && std::strcmp(text, "1") == 0;
  }();
  if (off)
    return nullptr;
  M_driver_queue *queue = nullptr;
  {
    std::lock_guard<std::mutex> lock(context.queuesMutex);
    if (context.queues.size() != 1)
      return nullptr;
    queue = context.queues.front();
    ++queue->drainers;
  }
  M_driver_memory *found = nullptr;
  {
    std::lock_guard<std::mutex> lock(queue->mutex);
    auto &pending = queue->pendingFrees;
    for (size_t i = 0; i < pending.size(); ++i) {
      M_driver_queue::PendingFree &entry = pending[i];
      M_driver_memory *memory = entry.memory;
      if (memory->capacity != capacity || memory->host || !memory->owning)
        continue;
      if (entry.sharers != nullptr && entry.sharers->load() != 1)
        continue;
      found = memory;
      found->reuseQueue = queue;
      found->reuseAfter = std::max(found->reuseAfter, entry.after);
      pending.erase(pending.begin() + i);
      break;
    }
  }
  {
    std::lock_guard<std::mutex> lock(context.queuesMutex);
    --queue->drainers;
  }
  context.queuesIdle.notify_all();
  return found;
}

/// Before a synchronous copy or fill touches `memory`: waits until the work
/// its reused block's last free waited for has run. It only polls the free
/// marks already on the queue, which follow that work, so it appends
/// nothing and waits for nothing the host does later.
void waitForReuse(M_driver_memory *memory) {
  if (memory == nullptr || memory->reuseQueue == nullptr)
    return;
  M_driver_queue &queue = *memory->reuseQueue;
  for (;;) {
    {
      std::lock_guard<std::mutex> lock(queue.mutex);
      if (queue.doneAt >= memory->reuseAfter) {
        memory->reuseQueue = nullptr;
        return;
      }
      advanceDoneLocked(queue);
      if (queue.doneAt >= memory->reuseAfter) {
        memory->reuseQueue = nullptr;
        return;
      }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

/// Waits for the work on `queue`'s list.
M_driver_status_t waitList(M_driver_queue &queue) {
  return waited(*queue.context, "zeCommandListHostSynchronize",
                ze(queue).zeCommandListHostSynchronize(queue.list, UINT64_MAX));
}

/// Ensures a free mark follows every command appended to `queue` so far, and
/// sets `wait` when some of them may still be running. Without counter-based
/// events, drains the queue instead (`wait` stays false). Caller holds
/// `queue.mutex`.
M_driver_status_t markLocked(M_driver_queue &queue, bool *wait) {
  *wait = false;
  if (queue.appended <= queue.doneAt)
    return M_DRIVER_STATUS_SUCCESS;
  if (queue.markedAt < queue.appended) {
    M_driver_queue::FreeMark &mark = queue.freeMarks[queue.nextFreeMark];
    if (mark.event == nullptr)
      mark.event = createCounterEvent(*queue.context);
    ze_result_t result = mark.event != nullptr
                             ? ze(queue).zeCommandListAppendBarrier(
                                   queue.list, mark.event, 0, nullptr)
                             : ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;
    if (result != ZE_RESULT_SUCCESS) {
      M_driver_status_t drained = waitList(queue);
      if (drained == M_DRIVER_STATUS_SUCCESS)
        queue.doneAt = queue.appended;
      return drained;
    }
    queue.markedAt = queue.appended;
    mark.at = queue.appended;
    queue.nextFreeMark = (queue.nextFreeMark + 1) % M_driver_queue::kFreeMarks;
  }
  *wait = true;
  return M_DRIVER_STATUS_SUCCESS;
}

/// Releases `memory` once the work already on `queue` has completed: a free
/// mark follows that work and the memory joins `pendingFrees`.
/// Without an event to order it by, drains the queue and releases at once,
/// so the memory is never leaked. Caller holds `queue.mutex`.
M_driver_status_t releaseWhenDoneLocked(M_driver_queue &queue,
                                        M_driver_memory *memory) {
  if (memory != nullptr && memory->owning &&
      keepForCapture(*queue.context, memory))
    return M_DRIVER_STATUS_SUCCESS;
  bool wait = false;
  if (M_driver_status_t status = markLocked(queue, &wait)) {
    release(queue.context, memory);
    return status;
  }
  if (wait) {
    queue.pendingFrees.push_back({memory, queue.appended});
    return reclaimFreesLocked(queue, /*all=*/false);
  }
  return release(queue.context, memory);
}

/// Fails once a kernel in `context` has called `abort()`. One load of host
/// memory, written by the device before the wait that precedes this check.
M_driver_status_t checkAbort(M_driver_context &context) {
  if (__atomic_load_n(context.abortFlag, __ATOMIC_ACQUIRE) == 0)
    return M_DRIVER_STATUS_SUCCESS;
  return fail(driverOf(context),
              "a kernel called abort(); its message went to stdout, and the "
              "context can't be used any more");
}

/// Waits for everything enqueued on `queue` so far. With a sync event the wait
/// happens outside `queue.mutex`, so other threads keep appending meanwhile.
M_driver_status_t hostSynchronize(M_driver_queue &queue) {
  std::unique_lock<std::mutex> lock(queue.mutex);
  if (queue.capturing)
    return fail(driverOf(queue), "queue_synchronize on a queue that captures "
                                 "a graph: its work hasn't run");
  if (queue.syncEvent == nullptr) {
    if (M_driver_status_t status = waitList(queue))
      return status;
    return reclaimFreesLocked(queue, /*all=*/true);
  }
  ze_result_t result = ze(queue).zeCommandListAppendBarrier(
      queue.list, queue.syncEvent, 0, nullptr);
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(queue), "zeCommandListAppendBarrier", result);
  uint64_t covered = queue.appended;
  lock.unlock();
  // A counter-based event completes with its latest signal: at least the
  // barrier just appended, so all work enqueued before it is done.
  M_driver_status_t status =
      waited(*queue.context, "zeEventHostSynchronize",
             ze(queue).zeEventHostSynchronize(queue.syncEvent, UINT64_MAX));
  lock.lock();
  if (status != M_DRIVER_STATUS_SUCCESS)
    return status;
  queue.doneAt = std::max(queue.doneAt, covered);
  if (queue.profile != nullptr)
    queue.profile->synchronized();
  return reclaimFreesLocked(queue, /*all=*/false);
}

/// Waits until every queue of `context` is idle; returns the first failure.
M_driver_status_t drainQueues(M_driver_context &context) {
  M_driver_status_t drained = M_DRIVER_STATUS_SUCCESS;
  // Drains a copy of the list without `queuesMutex`, so other threads can
  // create and destroy queues meanwhile; `drainers` keeps each copied queue
  // alive until its wait is over.
  std::vector<M_driver_queue *> queues;
  {
    std::lock_guard<std::mutex> lock(context.queuesMutex);
    queues = context.queues;
    for (M_driver_queue *queue : queues)
      ++queue->drainers;
  }
  for (M_driver_queue *queue : queues)
    if (M_driver_status_t status = hostSynchronize(*queue);
        status != M_DRIVER_STATUS_SUCCESS && drained == M_DRIVER_STATUS_SUCCESS)
      drained = status;
  {
    std::lock_guard<std::mutex> lock(context.queuesMutex);
    for (M_driver_queue *queue : queues)
      --queue->drainers;
  }
  context.queuesIdle.notify_all();
  return drained;
}

/// Releases `memory` once the work enqueued so far on every queue of
/// `context` has finished, as cuMemFree waits for the device: MAX frees a
/// buffer with `memory_free_sync` as soon as it's unused, often while kernels
/// that use it are still queued, and zeMemFree doesn't wait (they
/// page-faulted). The free itself doesn't wait: each queue gets a
/// barrier signalling an event and an entry in its `pendingFrees`, and the
/// last entry to complete releases the memory (reclaimed by later frees and
/// synchronizes). Draining every queue here would make each execution of a
/// MAX graph, which frees earlier outputs, a full host round trip. A queue
/// without counter-based events is drained instead.
M_driver_status_t releaseAfterQueues(M_driver_context_t context,
                                     M_driver_memory_t memory) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (memory == nullptr || !memory->owning)
    return release(context, memory);
  if (keepForCapture(*context, memory))
    return M_DRIVER_STATUS_SUCCESS;
  if (debugDrainFrees()) {
    M_driver_status_t drained = drainQueues(*context);
    M_driver_status_t status = release(context, memory);
    return drained != M_DRIVER_STATUS_SUCCESS ? drained : status;
  }
  std::vector<M_driver_queue *> queues;
  {
    std::lock_guard<std::mutex> lock(context->queuesMutex);
    queues = context->queues;
    for (M_driver_queue *queue : queues)
      ++queue->drainers;
  }
  // One share per queue that may still use the memory, plus this thread's
  // until every queue has been visited.
  auto sharers = std::make_shared<std::atomic<int>>(1);
  M_driver_status_t first = M_DRIVER_STATUS_SUCCESS;
  auto note = [&](M_driver_status_t status) {
    if (status != M_DRIVER_STATUS_SUCCESS && first == M_DRIVER_STATUS_SUCCESS)
      first = status;
  };
  for (M_driver_queue *queue : queues) {
    std::lock_guard<std::mutex> lock(queue->mutex);
    bool wait = false;
    note(markLocked(*queue, &wait));
    if (!wait)
      continue;
    sharers->fetch_add(1);
    queue->pendingFrees.push_back({memory, queue->appended, sharers});
    note(reclaimFreesLocked(*queue, /*all=*/false));
  }
  if (sharers->fetch_sub(1) == 1)
    note(release(context, memory));
  {
    std::lock_guard<std::mutex> lock(context->queuesMutex);
    for (M_driver_queue *queue : queues)
      --queue->drainers;
  }
  context->queuesIdle.notify_all();
  return first;
}

/// `hostSynchronize`, then `checkAbort`: the waits callers see. Destroying a
/// queue waits without it, so an aborted context can still be torn down.
M_driver_status_t synchronizeAndCheck(M_driver_queue &queue) {
  if (M_driver_status_t status = hostSynchronize(queue))
    return status;
  return checkAbort(*queue.context);
}

/// Copies pageable host memory to the device with CUDA's semantics: the
/// source is copied into a pinned staging buffer before returning, so the
/// caller may reuse it at once (Level Zero would read it at execution time).
M_driver_status_t stagedCopyToDevice(M_driver_queue &queue, void *dst,
                                     const void *src, uint64_t size) {
  M_driver_memory_t staging = nullptr;
  if (M_driver_status_t status =
          allocate(queue.context, size, /*host=*/true, &staging))
    return status;
  uint64_t enter = queue.profile != nullptr ? Profiling::now() : 0;
  std::memcpy(staging->ptr, src, size);
  std::lock_guard<std::mutex> lock(queue.mutex);
  ze_event_handle_t signal = profileEvent(queue);
  ze_result_t result = ze(queue).zeCommandListAppendMemoryCopy(
      queue.list, dst, staging->ptr, size, signal, 0, nullptr);
  if (result != ZE_RESULT_SUCCESS) {
    abandonProfileEvent(queue);
    release(queue.context, staging);
    return failZe(driverOf(queue), "zeCommandListAppendMemoryCopy", result);
  }
  ++queue.appended;
  if (queue.profile != nullptr)
    queue.profile->transfer('s', size, enter, Profiling::now());
  return releaseWhenDoneLocked(queue, staging);
}

} // namespace

M_driver_status_t M_driver_queue_create(M_driver_context_t context,
                                        M_driver_queue_t *queue) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (queue == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  auto created = std::make_unique<M_driver_queue>();
  created->context = context;
  ze_command_queue_desc_t desc = {};
  desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  desc.ordinal = 0; // the compute group on current Intel GPUs
  // Copies stay on the compute engine: with the copy-offload hint
  // decode was 5% slower on Qwen3.5-0.8B (host-to-device copies of a few
  // bytes each step waited on the copy engine), and compute-runtime 26.35's
  // copy engine faulted when device memory ran out. MOJO_INTEL_COPY_OFFLOAD=1
  // sets the hint again.
  static const bool copyOffload = [] {
    const char *text = std::getenv("MOJO_INTEL_COPY_OFFLOAD");
    return text != nullptr && std::strcmp(text, "1") == 0;
  }();
  desc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER |
               (copyOffload ? ZE_COMMAND_QUEUE_FLAG_COPY_OFFLOAD_HINT : 0);
  desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  ze_result_t result = ze(*context).zeCommandListCreateImmediate(
      context->zeContext, context->device->zeDevice, &desc, &created->list);
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(*context), "zeCommandListCreateImmediate", result);
  created->syncEvent = createCounterEvent(*context);
  created->profile.reset(Profiling::QueueProfile::create(*created));
  {
    std::lock_guard<std::mutex> lock(context->queuesMutex);
    context->queues.push_back(created.get());
  }
  *queue = created.release();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue_destroy(M_driver_context_t context,
                                         M_driver_queue_t queue) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  std::unique_ptr<M_driver_queue> owned(queue);
  // Let enqueued work finish before the list goes away, as a stream would,
  // and while the queue is still listed, so a synchronous free on another
  // thread waits for that work too. Then wait out frees still draining it.
  M_driver_status_t status = hostSynchronize(*owned);
  {
    std::unique_lock<std::mutex> lock(context->queuesMutex);
    std::erase(context->queues, queue);
    context->queuesIdle.wait(lock, [&] { return owned->drainers == 0; });
  }
  // A free that listed the queue before it was unlisted may have appended a
  // mark and a pending entry after the synchronize above: drain and retire
  // them, so the memory is released (no one can list the queue now).
  {
    std::lock_guard<std::mutex> lock(owned->mutex);
    if (!owned->pendingFrees.empty()) {
      M_driver_status_t drained = waitList(*owned);
      M_driver_status_t reclaimed = reclaimFreesLocked(*owned, /*all=*/true);
      if (status == M_DRIVER_STATUS_SUCCESS)
        status = drained != M_DRIVER_STATUS_SUCCESS ? drained : reclaimed;
    }
  }
  if (owned->syncEvent != nullptr)
    ze(*context).zeEventDestroy(owned->syncEvent);
  for (M_driver_queue::FreeMark &mark : owned->freeMarks)
    if (mark.event != nullptr)
      ze(*context).zeEventDestroy(mark.event);
  owned->profile.reset();
  if (status != M_DRIVER_STATUS_SUCCESS) {
    ze(*context).zeCommandListDestroy(owned->list);
    return status;
  }
  ze_result_t result = ze(*context).zeCommandListDestroy(owned->list);
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(*context), "zeCommandListDestroy", result);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue_copy_to_device(M_driver_queue_t queue,
                                                M_driver_memory_view_t dst,
                                                const void *src) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (M_driver_status_t status = checkView(dst))
    return status;
  if (src == nullptr && dst->size != 0)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (dst->size != 0 && !isUsm(*queue->context, src)) {
    // Small copies stage through pinned memory and stay asynchronous; large
    // ones copy directly and wait, as CUDA blocks for large pageable copies:
    // a serial memcpy into staging would double their time (64 MiB on the
    // B70: 4.97 ms staged, 2.74 ms direct).
    if (dst->size < kStageLimit)
      return stagedCopyToDevice(*queue, address(dst), src, dst->size);
    if (M_driver_status_t status =
            appendCopy(*queue, address(dst), src, dst->size))
      return status;
    return synchronizeAndCheck(*queue);
  }
  return appendCopy(*queue, address(dst), src, dst->size);
}

M_driver_status_t M_driver_queue_copy_from_device(M_driver_queue_t queue,
                                                  void *dst,
                                                  M_driver_memory_view_t src) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (M_driver_status_t status = checkView(src))
    return status;
  if (dst == nullptr && src->size != 0)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (M_driver_status_t status =
          appendCopy(*queue, dst, address(src), src->size))
    return status;
  // Into pageable memory, CUDA returns once the data has arrived; callers may
  // read or free `dst` right away.
  if (src->size != 0 && !isUsm(*queue->context, dst))
    return synchronizeAndCheck(*queue);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue_copy_intra_device(M_driver_queue_t queue,
                                                   M_driver_memory_view_t dst,
                                                   M_driver_memory_view_t src) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (M_driver_status_t status = checkView(dst))
    return status;
  if (M_driver_status_t status = checkView(src))
    return status;
  if (src->size < dst->size)
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;
  return appendCopy(*queue, address(dst), address(src), dst->size);
}

M_driver_status_t M_driver_queue_set_memory(M_driver_queue_t queue,
                                            M_driver_memory_view_t dst,
                                            uint8_t value) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (M_driver_status_t status = checkView(dst))
    return status;
  // Level Zero copies the pattern when the fill is appended.
  return appendFill(*queue, dst, &value, sizeof(value));
}

M_driver_status_t M_driver_queue_fill(M_driver_queue_t queue,
                                      M_driver_memory_view_t dst,
                                      uint64_t value, uint64_t value_size) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (M_driver_status_t status = checkView(dst))
    return status;
  if ((value_size != 1 && value_size != 2 && value_size != 4 &&
       value_size != 8) ||
      dst->size % value_size != 0 || !alignedForFill(address(dst), value_size))
    return M_DRIVER_STATUS_INVALID_ARG;
  unsigned char pattern[8];
  std::memcpy(pattern, &value, sizeof(pattern));
  return appendFill(*queue, dst, pattern, value_size);
}

M_driver_status_t M_driver_queue_synchronize(M_driver_queue_t queue) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (queue->profile == nullptr)
    return synchronizeAndCheck(*queue);
  uint64_t enter = Profiling::now();
  M_driver_status_t status = synchronizeAndCheck(*queue);
  queue->profile->host('S', enter, Profiling::now());
  return status;
}

M_driver_status_t M_driver_queue_is_stream(M_driver_queue_t queue,
                                           bool *is_stream) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (is_stream == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  *is_stream = true;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue_property(M_driver_queue_t queue,
                                          const char *property_name,
                                          void *value) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (property_name == nullptr || value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  // The device's DLPack type is kDLOneAPI, whose consumers read a stream
  // handle as a sycl::queue*; a Level Zero command list would crash them.
  // No native handle until there is a SYCL interop path.
  if (std::strcmp(property_name, M_DRIVER_QUEUE_PROPERTY_NATIVE_HANDLE) == 0)
    return M_DRIVER_STATUS_UNIMPLEMENTED;
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_driver_status_t M_driver_queue_record_event(M_driver_queue_t queue,
                                              M_driver_event_t event) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (event == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  std::lock_guard<std::mutex> lock(queue->mutex);
  // A counter-based event takes the state of its latest signal; a pooled one
  // must be reset first.
  if (event->pool != nullptr) {
    ze_result_t reset = ze(*queue).zeEventHostReset(event->event);
    if (reset != ZE_RESULT_SUCCESS)
      return failZe(driverOf(*queue), "zeEventHostReset", reset);
  }
  // A barrier signals its event once every earlier command has completed.
  uint64_t enter = queue->profile != nullptr ? Profiling::now() : 0;
  ze_result_t result = ze(*queue).zeCommandListAppendBarrier(
      queue->list, event->event, 0, nullptr);
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(*queue), "zeCommandListAppendBarrier", result);
  event->recorded = true;
  ++queue->appended;
  if (queue->profile != nullptr)
    queue->profile->host('E', enter, Profiling::now());
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue_wait_for_events(M_driver_queue_t queue,
                                                 M_driver_event_t *events,
                                                 uint32_t num_events) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (num_events == 0)
    return M_DRIVER_STATUS_SUCCESS;
  if (events == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  std::vector<ze_event_handle_t> handles;
  for (uint32_t i = 0; i < num_events; ++i) {
    if (events[i] == nullptr)
      return M_DRIVER_STATUS_BAD_EVENT;
    // Waiting on a never-recorded event is a no-op, as it is complete.
    if (events[i]->recorded)
      handles.push_back(events[i]->event);
  }
  if (handles.empty())
    return M_DRIVER_STATUS_SUCCESS;
  std::lock_guard<std::mutex> lock(queue->mutex);
  uint64_t enter = queue->profile != nullptr ? Profiling::now() : 0;
  ze_result_t result = ze(*queue).zeCommandListAppendWaitOnEvents(
      queue->list, static_cast<uint32_t>(handles.size()), handles.data());
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(*queue), "zeCommandListAppendWaitOnEvents", result);
  ++queue->appended;
  if (queue->profile != nullptr)
    queue->profile->host('W', enter, Profiling::now());
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_memory_alloc_async(M_driver_queue_t queue,
                                              uint64_t size,
                                              M_driver_memory_t *memory) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  // Level Zero has no stream-ordered allocator; the allocation is usable at
  // once, which satisfies any queue order.
  return allocate(queue->context, size, /*host=*/false, memory);
}

M_driver_status_t M_driver_memory_free_async(M_driver_queue_t queue,
                                             M_driver_memory_t memory) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;
  // Work already on the queue may still use the memory: the free is
  // stream-ordered.
  std::lock_guard<std::mutex> lock(queue->mutex);
  return releaseWhenDoneLocked(*queue, memory);
}

M_driver_status_t M_driver_event_create(M_driver_context_t context,
                                        M_driver_event_flags_t flags,
                                        M_driver_event_t *event) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (event == nullptr || (flags & ~M_DRIVER_EVENT_FLAGS_ALL) != 0)
    return M_DRIVER_STATUS_INVALID_ARG;
  const ZeApi &api = ze(*context);
  auto created = std::make_unique<M_driver_event>();
  // Counter-based events cost about 0.1 us to create against about 60 us for
  // an event pool, and need no reset. Always host-visible: the HAL lets the
  // host query and wait on any event.
  created->event = createCounterEvent(*context);
  if (created->event != nullptr) {
    *event = created.release();
    return M_DRIVER_STATUS_SUCCESS;
  }
  ze_event_pool_desc_t poolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
                                   ZE_EVENT_POOL_FLAG_HOST_VISIBLE, 1};
  ze_device_handle_t device = context->device->zeDevice;
  ze_result_t result = api.zeEventPoolCreate(context->zeContext, &poolDesc, 1,
                                             &device, &created->pool);
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(*context), "zeEventPoolCreate", result);
  ze_event_desc_t desc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, 0,
                          ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST};
  result = api.zeEventCreate(created->pool, &desc, &created->event);
  if (result != ZE_RESULT_SUCCESS) {
    api.zeEventPoolDestroy(created->pool);
    return failZe(driverOf(*context), "zeEventCreate", result);
  }
  *event = created.release();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event_destroy(M_driver_context_t context,
                                         M_driver_event_t event) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (event == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  std::unique_ptr<M_driver_event> owned(event);
  const ZeApi &api = ze(*context);
  ze_result_t eventResult = api.zeEventDestroy(owned->event);
  ze_result_t poolResult = owned->pool != nullptr
                               ? api.zeEventPoolDestroy(owned->pool)
                               : ZE_RESULT_SUCCESS;
  if (eventResult != ZE_RESULT_SUCCESS)
    return failZe(driverOf(*context), "zeEventDestroy", eventResult);
  if (poolResult != ZE_RESULT_SUCCESS)
    return failZe(driverOf(*context), "zeEventPoolDestroy", poolResult);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event_synchronize(M_driver_context_t context,
                                             M_driver_event_t event) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (event == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  if (!event->recorded)
    return M_DRIVER_STATUS_SUCCESS;
  if (M_driver_status_t status =
          waited(*context, "zeEventHostSynchronize",
                 ze(*context).zeEventHostSynchronize(event->event, UINT64_MAX)))
    return status;
  return checkAbort(*context);
}

M_driver_status_t M_driver_is_event_ready(M_driver_context_t context,
                                          M_driver_event_t event,
                                          bool *is_ready) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (event == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  if (is_ready == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (!event->recorded) {
    *is_ready = true;
    return M_DRIVER_STATUS_SUCCESS;
  }
  ze_result_t result = ze(*context).zeEventQueryStatus(event->event);
  if (result == ZE_RESULT_SUCCESS || result == ZE_RESULT_NOT_READY) {
    *is_ready = result == ZE_RESULT_SUCCESS;
    return M_DRIVER_STATUS_SUCCESS;
  }
  return failZe(driverOf(*context), "zeEventQueryStatus", result);
}

// ===-- Graphs (IntelGraphs.h) --------------------------------------------===

struct M_driver_intel_graph_s {
  M_driver_context *context = nullptr;
  ze_graph_handle_t graph = nullptr;
  ze_executable_graph_handle_t executable = nullptr;
  /// The graph's id in the launch profile (MOJO_INTEL_PROFILE); 0 when off.
  uint32_t profileId = 0;
};

namespace {

bool hasGraphCapture(const M_driver_device &device) {
  const ZeApi &api = *device.driver->ze;
  if (api.zeCommandListBeginGraphCaptureExt == nullptr ||
      api.zeCommandListEndGraphCaptureExt == nullptr ||
      api.zeGraphInstantiateExt == nullptr ||
      api.zeCommandListAppendGraphExt == nullptr ||
      api.zeExecutableGraphDestroyExt == nullptr ||
      api.zeGraphDestroyExt == nullptr ||
      api.zeDriverGetExtensionProperties == nullptr)
    return false;
  uint32_t count = 0;
  if (api.zeDriverGetExtensionProperties(device.zeDriver, &count, nullptr) !=
      ZE_RESULT_SUCCESS)
    return false;
  std::vector<ze_driver_extension_properties_t> props(count);
  if (api.zeDriverGetExtensionProperties(device.zeDriver, &count,
                                         props.data()) != ZE_RESULT_SUCCESS)
    return false;
  for (const ze_driver_extension_properties_t &p : props)
    if (std::strcmp(p.name, ZE_RECORD_REPLAY_GRAPH_EXT_NAME) == 0)
      return p.version >= ZE_RECORD_REPLAY_GRAPH_EXT_VERSION_1_0;
  return false;
}

/// Ends the context's capture count for one queue, whose graph now lives.
void endCaptureCount(M_driver_context &context) {
  std::lock_guard<std::mutex> lock(context.captureMutex);
  --context.capturing;
  ++context.liveGraphs;
}

} // namespace

M_driver_status_t M_driver_intel_graphs_version(M_driver_device_t device,
                                                uint32_t *version) {
  if (device == nullptr)
    return M_DRIVER_STATUS_BAD_DEVICE;
  if (version == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  *version = hasGraphCapture(*device) ? M_DRIVER_INTEL_GRAPHS_VERSION_1 : 0;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_intel_graph_capture_begin(M_driver_queue_t queue) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (!hasGraphCapture(*queue->context->device))
    return M_DRIVER_STATUS_UNSUPPORTED;
  std::lock_guard<std::mutex> lock(queue->mutex);
  if (queue->capturing)
    return M_DRIVER_STATUS_INVALID_ARG;
  ze_result_t result =
      ze(*queue).zeCommandListBeginGraphCaptureExt(queue->list, nullptr);
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(*queue), "zeCommandListBeginGraphCaptureExt",
                  result);
  queue->capturing = true;
  if (queue->profile != nullptr)
    queue->profile->beginCapture(Profiling::QueueProfile::nextGraphId());
  std::lock_guard<std::mutex> captureLock(queue->context->captureMutex);
  ++queue->context->capturing;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_intel_graph_capture_end(M_driver_queue_t queue,
                                 M_driver_intel_graph_t *graph) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (graph == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  *graph = nullptr;
  auto made = std::make_unique<M_driver_intel_graph_s>();
  made->context = queue->context;
  const ZeApi &api = ze(*queue);
  {
    std::lock_guard<std::mutex> lock(queue->mutex);
    if (!queue->capturing)
      return M_DRIVER_STATUS_INVALID_ARG;
    ze_result_t result =
        api.zeCommandListEndGraphCaptureExt(queue->list, nullptr, &made->graph);
    queue->capturing = false;
    if (queue->profile != nullptr) {
      made->profileId = queue->profile->capturingGraph();
      queue->profile->endCapture();
    }
    endCaptureCount(*queue->context);
    if (result != ZE_RESULT_SUCCESS) {
      // The kept memory may still be used by work recorded before the
      // failure; leave it to the destroy, which waits first.
      M_driver_status_t status =
          failZe(driverOf(*queue), "zeCommandListEndGraphCaptureExt", result);
      M_driver_intel_graph_destroy(made.release());
      return status;
    }
  }
  ze_result_t result =
      api.zeGraphInstantiateExt(made->graph, nullptr, &made->executable);
  if (result != ZE_RESULT_SUCCESS) {
    M_driver_status_t status =
        failZe(driverOf(*queue), "zeGraphInstantiateExt", result);
    M_driver_intel_graph_destroy(made.release());
    return status;
  }
  *graph = made.release();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_intel_graph_launch(M_driver_queue_t queue,
                                              M_driver_intel_graph_t graph) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (graph == nullptr || graph->executable == nullptr ||
      graph->context != queue->context)
    return M_DRIVER_STATUS_INVALID_ARG;
  uint64_t enter = queue->profile != nullptr ? Profiling::now() : 0;
  std::lock_guard<std::mutex> lock(queue->mutex);
  // The whole graph signals one timestamp event: its device time.
  ze_event_handle_t signal = profileEvent(*queue);
  ze_result_t result = ze(*queue).zeCommandListAppendGraphExt(
      queue->list, graph->executable, nullptr, signal, 0, nullptr);
  if (result != ZE_RESULT_SUCCESS) {
    abandonProfileEvent(*queue);
    return failZe(driverOf(*queue), "zeCommandListAppendGraphExt", result);
  }
  ++queue->appended;
  if (queue->profile != nullptr)
    queue->profile->graphLaunch(graph->profileId, enter, Profiling::now());
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_intel_graph_destroy(M_driver_intel_graph_t graph) {
  if (graph == nullptr)
    return M_DRIVER_STATUS_SUCCESS;
  std::unique_ptr<M_driver_intel_graph_s> owned(graph);
  M_driver_context &context = *graph->context;
  const ZeApi &api = ze(context);
  // A launch of the graph may still be running on any queue.
  M_driver_status_t first = drainQueues(context);
  auto note = [&](M_driver_status_t status) {
    if (status != M_DRIVER_STATUS_SUCCESS && first == M_DRIVER_STATUS_SUCCESS)
      first = status;
  };
  if (graph->executable != nullptr)
    if (ze_result_t r = api.zeExecutableGraphDestroyExt(graph->executable))
      note(failZe(driverOf(context), "zeExecutableGraphDestroyExt", r));
  if (graph->graph != nullptr)
    if (ze_result_t r = api.zeGraphDestroyExt(graph->graph))
      note(failZe(driverOf(context), "zeGraphDestroyExt", r));
  {
    std::lock_guard<std::mutex> lock(context.captureMutex);
    --context.liveGraphs;
  }
  note(releaseGraphPoolIfUnused(context));
  return first;
}
