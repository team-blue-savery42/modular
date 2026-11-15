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
// Intel device graphs: entry points of the Level Zero driver plugin outside
// the HAL (not in M_driver_hal_symbols.inc), resolved by name by MAX's Intel
// HAL extension set (`hal/extensions/intel.mojo`), as the device features
// are (IntelDeviceFeatures.h). A queue records the work appended to
// it between `capture_begin` and `capture_end` into a graph instead of
// running it; `launch` appends the whole graph to a queue in one call, as
// often as wanted. Built on Level Zero's `ZE_extension_record_replay_graph`
// (graph capture on the queue's immediate command list).
//
// While a queue captures:
//  - memory freed in its context (`memory_free_async` on it,
//    `memory_free_sync`, `memory_free_pinned`) is not released or reused:
//    the graph takes it and releases it when destroyed, since every launch
//    of the graph uses those addresses again;
//  - `queue_synchronize` on it fails (nothing it captured has run), and the
//    launch profile (MOJO_INTEL_PROFILE) records its appends without device
//    times;
//  - other queues of the context keep running work as usual.
// Memory allocated while a queue captures stays the graphs': freed after
// the capture too (a graph's outputs, say), it goes to the same pool while
// any graph of the context lives, since every launch writes it again.
// Versioned: `M_driver_intel_graphs_version` reports the version the
// plugin implements for a device, or 0 when its driver has no graph capture.
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_DRIVER_INTELGRAPHS_H
#define MOJO_INTEL_DRIVER_INTELGRAPHS_H

#include "HAL/M_driver_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// The first version of the Intel graph entry points.
#define M_DRIVER_INTEL_GRAPHS_VERSION_1 1u

/// A captured, instantiated graph of one context.
typedef struct M_driver_intel_graph_s *M_driver_intel_graph_t;

/// Sets `*version` to the graph version the plugin supports on `device`:
/// M_DRIVER_INTEL_GRAPHS_VERSION_1, or 0 when the driver can't capture.
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_intel_graphs_version(M_driver_device_t device,
                                                uint32_t *version);

/// Starts recording the work appended to `queue`.
///
/// @return M_DRIVER_STATUS_UNSUPPORTED without graph capture,
///         M_DRIVER_STATUS_INVALID_ARG when `queue` already captures.
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_intel_graph_capture_begin(M_driver_queue_t queue);

/// Stops recording on `queue` and instantiates what it recorded as
/// `*graph`, which owns the memory freed in the context during the capture.
/// The queue is out of capture mode afterwards even when this fails.
M_DRIVER_API_EXPORT
M_driver_status_t
M_driver_intel_graph_capture_end(M_driver_queue_t queue,
                                 M_driver_intel_graph_t *graph);

/// Appends one run of `graph` to `queue` (a queue of the graph's context).
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_intel_graph_launch(M_driver_queue_t queue,
                                              M_driver_intel_graph_t graph);

/// Waits for the context's queues, then destroys `graph` and releases the
/// memory it owns. A null graph is a no-op.
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_intel_graph_destroy(M_driver_intel_graph_t graph);

typedef M_driver_status_t (*M_driver_intel_graphs_version_func_ptr_t)(
    M_driver_device_t device, uint32_t *version);
typedef M_driver_status_t (*M_driver_intel_graph_capture_begin_func_ptr_t)(
    M_driver_queue_t queue);
typedef M_driver_status_t (*M_driver_intel_graph_capture_end_func_ptr_t)(
    M_driver_queue_t queue, M_driver_intel_graph_t *graph);
typedef M_driver_status_t (*M_driver_intel_graph_launch_func_ptr_t)(
    M_driver_queue_t queue, M_driver_intel_graph_t graph);
typedef M_driver_status_t (*M_driver_intel_graph_destroy_func_ptr_t)(
    M_driver_intel_graph_t graph);

#ifdef __cplusplus
}
#endif

#endif // MOJO_INTEL_DRIVER_INTELGRAPHS_H
