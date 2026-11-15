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
// Optional launch profiling; see Profile.h.
//
//===----------------------------------------------------------------------===//

#include "Profile.h"

#include "Plugin.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <set>
#include <string>
#include <unistd.h>
#include <unordered_map>

namespace M::Intel::Profiling {
namespace {

/// Timestamp events per queue; a full ring waits for the queue and harvests.
constexpr uint32_t kRingSize = 16384;

/// A synchronize harvests when this many records wait, or when the last
/// harvest was this long ago.
constexpr size_t kHarvestRecords = kRingSize / 2;
constexpr uint64_t kHarvestNs = 2'000'000'000;

struct Output {
  std::mutex mutex;
  FILE *file = nullptr;
  std::unordered_map<std::string, uint32_t> functions;
  uint32_t queues = 0;
  bool header = false;
};

Output &output() {
  static Output *out = [] {
    auto *result = new Output();
    const char *prefix = std::getenv("MOJO_INTEL_PROFILE");
    if (prefix != nullptr && *prefix != '\0') {
      std::string path = std::string(prefix) + "." + std::to_string(::getpid());
      result->file = std::fopen(path.c_str(), "w");
      if (result->file != nullptr)
        std::setvbuf(result->file, nullptr, _IOFBF, 1 << 20);
    }
    return result;
  }();
  return *out;
}

/// Writes one line under the output lock.
template <typename... Args>
void line(const char *format, Args... args) {
  Output &out = output();
  std::lock_guard<std::mutex> lock(out.mutex);
  if (out.file != nullptr)
    std::fprintf(out.file, format, args...);
}

const ZeApi &ze(const M_driver_queue &queue) {
  return *queue.context->device->driver->ze;
}

} // namespace

bool enabled() {
  static const bool on = output().file != nullptr;
  return on;
}

/// MOJO_INTEL_PROFILE_DEVICE=0 records host times only: no timestamp event
/// per command, whose cost (several microseconds a launch) stretches the
/// host timeline the profile measures.
static bool deviceTimes() {
  static const bool on = [] {
    const char *text = std::getenv("MOJO_INTEL_PROFILE_DEVICE");
    return text == nullptr || std::strcmp(text, "0") != 0;
  }();
  return on;
}

uint64_t now() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1'000'000'000u + uint64_t(ts.tv_nsec);
}

uint32_t functionId(const char *name, uint64_t length) {
  if (!enabled())
    return 0;
  Output &out = output();
  std::lock_guard<std::mutex> lock(out.mutex);
  std::string key(name, length);
  auto [it, inserted] =
      out.functions.emplace(key, static_cast<uint32_t>(out.functions.size()));
  if (inserted) {
    // Names hold spaces and newlines rarely; keep the line parseable.
    for (char &c : key)
      if (c == '\n' || c == '\r')
        c = ' ';
    std::fprintf(out.file, "F %u %s\n", it->second, key.c_str());
  }
  return it->second;
}

QueueProfile *QueueProfile::create(M_driver_queue &queue) {
  if (!enabled())
    return nullptr;
  const ZeApi &api = ze(queue);
  if (api.zeEventQueryKernelTimestamp == nullptr)
    return nullptr;
  M_driver_device &device = *queue.context->device;
  {
    Output &out = output();
    std::lock_guard<std::mutex> lock(out.mutex);
    if (!out.header) {
      out.header = true;
      // Device properties were read as DEVICE_PROPERTIES_1_2 (Driver.cpp):
      // timerResolution is in cycles per second.
      std::fprintf(out.file, "P %llu %u %u\n",
                   (unsigned long long)device.props.timerResolution,
                   device.props.timestampValidBits,
                   device.props.kernelTimestampValidBits);
    }
  }
  ze_event_pool_desc_t poolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
                                   ZE_EVENT_POOL_FLAG_HOST_VISIBLE |
                                       ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP,
                                   kRingSize};
  ze_event_pool_handle_t pool = nullptr;
  ze_device_handle_t zeDevice = device.zeDevice;
  if (api.zeEventPoolCreate(queue.context->zeContext, &poolDesc, 1, &zeDevice,
                            &pool) != ZE_RESULT_SUCCESS)
    return nullptr;
  std::vector<ze_event_handle_t> events(kRingSize, nullptr);
  for (uint32_t i = 0; i < kRingSize; ++i) {
    ze_event_desc_t desc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, i,
                            ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST};
    if (api.zeEventCreate(pool, &desc, &events[i]) != ZE_RESULT_SUCCESS) {
      for (uint32_t j = 0; j < i; ++j)
        api.zeEventDestroy(events[j]);
      api.zeEventPoolDestroy(pool);
      return nullptr;
    }
  }
  return new QueueProfile(queue, pool, std::move(events));
}

namespace {

/// The live queue profiles, so the records still pending when the process
/// exits get written: MAX's processes exit without destroying their queues,
/// and a short run (a long prompt step, a few graph launches) may never
/// fill the event ring that triggers a harvest.
std::mutex &liveMutex() {
  static std::mutex m;
  return m;
}
std::set<QueueProfile *> &live() {
  static std::set<QueueProfile *> s;
  return s;
}

void harvestAtExit() {
  std::lock_guard<std::mutex> lock(liveMutex());
  for (QueueProfile *profile : live())
    profile->harvestUnlocked();
}

} // namespace

void QueueProfile::harvestUnlocked() {
  // At exit no other thread should append; one that died holding the
  // queue's lock leaves its records unwritten rather than deadlocking.
  std::unique_lock<std::mutex> lock(queue.mutex, std::try_to_lock);
  if (lock.owns_lock())
    harvest(/*wait=*/false);
}

QueueProfile::QueueProfile(M_driver_queue &queue, ze_event_pool_handle_t pool,
                           std::vector<ze_event_handle_t> events)
    : queue(queue), pool(pool), events(std::move(events)),
      busy(this->events.size(), false), lastHarvest(now()) {
  {
    Output &out = output();
    std::lock_guard<std::mutex> lock(out.mutex);
    id = out.queues++;
  }
  static const bool registered = [] {
    // Construct what the exit handler uses first: statics are destroyed in
    // reverse order of construction, and handlers run in reverse of their
    // registration, so these must outlive it.
    (void)liveMutex();
    (void)live();
    std::atexit(harvestAtExit);
    return true;
  }();
  (void)registered;
  std::lock_guard<std::mutex> lock(liveMutex());
  live().insert(this);
}

QueueProfile::~QueueProfile() {
  {
    std::lock_guard<std::mutex> lock(liveMutex());
    live().erase(this);
  }
  harvest(/*wait=*/true);
  const ZeApi &api = ze(queue);
  for (ze_event_handle_t event : events)
    api.zeEventDestroy(event);
  api.zeEventPoolDestroy(pool);
  Output &out = output();
  std::lock_guard<std::mutex> lock(out.mutex);
  std::fflush(out.file);
}

ze_event_handle_t QueueProfile::nextEvent() {
  if (!deviceTimes()) {
    pendingSlot = -1;
    return nullptr;
  }
  if (busy[next])
    harvest(/*wait=*/true);
  if (busy[next]) {
    pendingSlot = -1;
    return nullptr;
  }
  pendingSlot = static_cast<int32_t>(next);
  busy[next] = true;
  next = (next + 1) % events.size();
  return events[pendingSlot];
}

void QueueProfile::abandonEvent() {
  if (pendingSlot >= 0)
    busy[pendingSlot] = false;
  pendingSlot = -1;
}

void QueueProfile::push(Record record) {
  record.slot = pendingSlot;
  pendingSlot = -1;
  pending.push_back(record);
  // Host-only records need no device query: write them out in batches (the
  // queue may never be synchronized directly, MAX waits on events).
  if (!deviceTimes() && pending.size() >= 1024)
    harvest(/*wait=*/false);
}

void QueueProfile::kernel(uint32_t function, const uint32_t grid[3],
                          const uint32_t block[3], uint64_t enter,
                          uint64_t exit) {
  Record record = {};
  record.kind = capturing != 0 ? 'k' : 'K';
  record.graph = capturing;
  record.function = function;
  for (int i = 0; i < 3; ++i) {
    record.grid[i] = grid[i];
    record.block[i] = block[i];
  }
  record.enter = enter;
  record.exit = exit;
  push(record);
}

void QueueProfile::transfer(char kind, uint64_t bytes, uint64_t enter,
                            uint64_t exit) {
  Record record = {};
  record.kind = capturing != 0 ? 'c' : 'C';
  record.graph = capturing;
  record.sub = kind;
  record.bytes = bytes;
  record.enter = enter;
  record.exit = exit;
  push(record);
}

void QueueProfile::graphLaunch(uint32_t graph, uint64_t enter, uint64_t exit) {
  Record record = {};
  record.kind = 'G';
  record.graph = graph;
  record.enter = enter;
  record.exit = exit;
  push(record);
  // A step that replays a graph makes one record, so the event ring may
  // never fill: write the completed ones as they come.
  if (pending.size() >= 16)
    harvest(/*wait=*/false);
}

uint32_t QueueProfile::nextGraphId() {
  static std::atomic<uint32_t> last{0};
  return ++last;
}

void QueueProfile::host(char kind, uint64_t enter, uint64_t exit) {
  line("%c %u %llu %llu\n", kind, id, (unsigned long long)enter,
       (unsigned long long)exit);
}

void QueueProfile::synchronized() {
  if (pending.size() >= kHarvestRecords || now() - lastHarvest >= kHarvestNs)
    harvest(/*wait=*/false);
}

void QueueProfile::harvest(bool wait) {
  uint64_t enter = now();
  const ZeApi &api = ze(queue);
  size_t done = 0;
  Output &out = output();
  std::lock_guard<std::mutex> lock(out.mutex);
  for (; done < pending.size(); ++done) {
    const Record &r = pending[done];
    long long start = -1, end = -1;
    if (r.slot >= 0) {
      ze_event_handle_t event = events[r.slot];
      ze_result_t status = wait ? api.zeEventHostSynchronize(event, UINT64_MAX)
                                : api.zeEventQueryStatus(event);
      if (status == ZE_RESULT_NOT_READY)
        break;
      ze_kernel_timestamp_result_t stamp = {};
      if (status == ZE_RESULT_SUCCESS &&
          api.zeEventQueryKernelTimestamp(event, &stamp) == ZE_RESULT_SUCCESS) {
        start = static_cast<long long>(stamp.global.kernelStart);
        end = static_cast<long long>(stamp.global.kernelEnd);
      }
      api.zeEventHostReset(event);
      busy[r.slot] = false;
    }
    if (r.kind == 'K')
      std::fprintf(out.file, "K %u %u %u %u %u %u %u %u %llu %llu %lld %lld\n",
                   id, r.function, r.grid[0], r.grid[1], r.grid[2], r.block[0],
                   r.block[1], r.block[2], (unsigned long long)r.enter,
                   (unsigned long long)r.exit, start, end);
    else if (r.kind == 'k')
      std::fprintf(out.file, "k %u %u %u %u %u %u %u %u %u %llu %llu\n", id,
                   r.graph, r.function, r.grid[0], r.grid[1], r.grid[2],
                   r.block[0], r.block[1], r.block[2],
                   (unsigned long long)r.enter, (unsigned long long)r.exit);
    else if (r.kind == 'c')
      std::fprintf(out.file, "c %u %u %c %llu %llu %llu\n", id, r.graph, r.sub,
                   (unsigned long long)r.bytes, (unsigned long long)r.enter,
                   (unsigned long long)r.exit);
    else if (r.kind == 'G')
      std::fprintf(out.file, "G %u %u %llu %llu %lld %lld\n", id, r.graph,
                   (unsigned long long)r.enter, (unsigned long long)r.exit,
                   start, end);
    else
      std::fprintf(out.file, "C %u %c %llu %llu %llu %lld %lld\n", id, r.sub,
                   (unsigned long long)r.bytes, (unsigned long long)r.enter,
                   (unsigned long long)r.exit, start, end);
  }
  pending.erase(pending.begin(), pending.begin() + done);
  if (api.zeDeviceGetGlobalTimestamps != nullptr) {
    uint64_t host = 0, device = 0;
    uint64_t before = now();
    if (api.zeDeviceGetGlobalTimestamps(queue.context->device->zeDevice, &host,
                                        &device) == ZE_RESULT_SUCCESS)
      std::fprintf(out.file, "T %llu %llu\n",
                   (unsigned long long)((before + now()) / 2),
                   (unsigned long long)device);
  }
  lastHarvest = now();
  std::fprintf(out.file, "H %llu %llu\n", (unsigned long long)enter,
               (unsigned long long)lastHarvest);
  std::fflush(out.file);
}

void allocation(char kind, uint64_t bytes, bool host, uint64_t enter,
                uint64_t exit) {
  line("%c %llu %c %llu %llu\n", kind, (unsigned long long)bytes,
       host ? 'h' : 'd', (unsigned long long)enter, (unsigned long long)exit);
}

} // namespace M::Intel::Profiling
