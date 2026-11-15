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
// Optional launch profiling (default off). With MOJO_INTEL_PROFILE=<prefix>
// the plugin writes <prefix>.<pid> with one line per kernel launch, copy,
// fill, synchronize, event record or wait, and allocation: the host time
// spent in the call (CLOCK_MONOTONIC ns) and, for work on a queue, its device
// start and end (Level Zero kernel timestamps, raw ticks). Profilers that
// trace the Level Zero loader (unitrace) don't see the kernels MAX's graph
// runtime launches through this plugin; this does, from inside it.
//
// The device timestamps are read lazily: each queue keeps a ring of
// timestamp events and harvests the completed ones when the ring fills or
// at a synchronize after a quiet period, so most steps run without the
// extra queries (each harvest is logged with its host time, so analysis can
// skip the steps it delayed). Format (space-separated, one record per line):
//   P <timer Hz> <timestamp valid bits> <kernel timestamp valid bits>
//   F <function id> <name>
//   T <host ns> <device ticks>                 (zeDeviceGetGlobalTimestamps)
//   K <queue> <function id> <grid x y z> <block x y z> <host enter> <host
//     exit> <device start> <device end>
//   C <queue> <c|f|s> <bytes> <host enter> <host exit> <device start> <end>
//   S|E|W <queue> <host enter> <host exit>    (synchronize, record, wait)
//   A|R <bytes> <h|d> <host enter> <host exit> (allocate, release; a|r
//     when served from or returned to the allocation cache)
//   H <host enter> <host exit>                (a harvest)
//   k <queue> <graph> <function id> <grid x y z> <block x y z> <host enter>
//     <host exit>                              (a kernel captured into a
//                                               device graph, IntelGraphs.h;
//                                               it runs in that graph's G)
//   c <queue> <graph> <c|f|s> <bytes> <host enter> <host exit> (a captured
//     copy or fill)
//   G <queue> <graph> <host enter> <host exit> <device start> <device end>
//     (one launch of a device graph: the device time of all its commands)
// Device fields are -1 where no timestamp was taken; with
// MOJO_INTEL_PROFILE_DEVICE=0 none is (host times only, a cheaper record).
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_DRIVER_PROFILE_H
#define MOJO_INTEL_DRIVER_PROFILE_H

#include "ZeApi.h"

#include <cstdint>
#include <vector>

struct M_driver_context;
struct M_driver_queue;

namespace M::Intel::Profiling {

/// Whether MOJO_INTEL_PROFILE is set (read once).
bool enabled();

/// CLOCK_MONOTONIC in nanoseconds (Python's time.monotonic_ns()).
uint64_t now();

/// A function's id in the profile, writing its name the first time.
uint32_t functionId(const char *name, uint64_t length);

/// Per-queue state: a ring of timestamp events and the records waiting for
/// their device times. All calls hold the queue's mutex.
class QueueProfile {
public:
  /// Null unless profiling is enabled and the device supports timestamps.
  static QueueProfile *create(M_driver_queue &queue);
  ~QueueProfile();

  /// The event to signal from the next command, or null when none is free.
  ze_event_handle_t nextEvent();
  /// Gives back the event `nextEvent` returned when its command could not
  /// be appended (no record follows for it).
  void abandonEvent();
  /// Records the command `nextEvent` was taken for.
  void kernel(uint32_t function, const uint32_t grid[3],
              const uint32_t block[3], uint64_t enter, uint64_t exit);
  void transfer(char kind, uint64_t bytes, uint64_t enter, uint64_t exit);
  /// Records made between these two are a graph's captured commands
  /// (written as `k` and `c`, without device times).
  void beginCapture(uint32_t graph) { capturing = graph; }
  /// Ends the capture and writes its records (they have no device times).
  void endCapture() {
    capturing = 0;
    harvest(/*wait=*/false);
  }
  uint32_t capturingGraph() const { return capturing; }
  /// Records the launch of `graph` that `nextEvent` was taken for.
  void graphLaunch(uint32_t graph, uint64_t enter, uint64_t exit);
  /// A new graph's id for the profile (1, 2, ...).
  static uint32_t nextGraphId();
  /// A host-only record (synchronize, record, wait).
  void host(char kind, uint64_t enter, uint64_t exit);
  /// After a completed synchronize: harvests when the ring is half full or
  /// a while has passed since the last harvest.
  void synchronized();
  /// Harvests every completed record.
  void harvest(bool wait);
  /// `harvest(false)` taking the queue's lock (if free): the exit-time flush.
  void harvestUnlocked();

private:
  struct Record {
    char kind;
    char sub;
    uint32_t function;
    uint32_t grid[3];
    uint32_t block[3];
    uint64_t bytes;
    uint64_t enter, exit;
    int32_t slot;
    /// The device graph a captured record belongs to, or launched; 0: none.
    uint32_t graph;
  };
  QueueProfile(M_driver_queue &queue, ze_event_pool_handle_t pool,
               std::vector<ze_event_handle_t> events);
  void push(Record record);

  M_driver_queue &queue;
  uint32_t id;
  ze_event_pool_handle_t pool;
  std::vector<ze_event_handle_t> events;
  std::vector<bool> busy;
  uint32_t next = 0;
  /// Records in submission order, waiting for their device times.
  std::vector<Record> pending;
  int32_t pendingSlot = -1;
  uint64_t lastHarvest = 0;
  /// The graph being captured on the queue; 0 when not capturing.
  uint32_t capturing = 0;
};

/// Allocation and release records (no queue).
void allocation(char kind, uint64_t bytes, bool host, uint64_t enter,
                uint64_t exit);

} // namespace M::Intel::Profiling

#endif // MOJO_INTEL_DRIVER_PROFILE_H
