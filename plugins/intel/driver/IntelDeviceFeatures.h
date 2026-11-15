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
// Intel device features: an entry point of the Level Zero driver plugin
// outside the HAL (not in M_driver_hal_symbols.inc), resolved by name by
// MAX's Intel HAL extension set (`hal/extensions/intel.mojo`,
// `SupportsIntelDeviceFeaturesV1`). Run-time facts about the device that
// Intel-tuned kernels and their dispatch choose by: sub-group sizes, matrix
// (DPAS) and dot-product (DP4A) units, 2D block loads and stores, and the
// sizes of the machine.
//
// Versioned: the caller sets `version` and `size`, the plugin fills the
// fields of every version up to the lower of the caller's and its own, and
// sets `version` to the version it filled. Later versions only append fields.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_DRIVER_INTELDEVICEFEATURES_H
#define MOJO_INTEL_DRIVER_INTELDEVICEFEATURES_H

#include "HAL/M_driver_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// The first version of `M_driver_intel_device_features`.
#define M_DRIVER_INTEL_DEVICE_FEATURES_VERSION_1 1u

/// A yes/no feature the driver doesn't report (its Level Zero extension is
/// missing, or the query failed). Numeric fields the driver doesn't report
/// are 0.
#define M_DRIVER_INTEL_FEATURE_UNKNOWN (-1)
#define M_DRIVER_INTEL_FEATURE_NO 0
#define M_DRIVER_INTEL_FEATURE_YES 1

typedef struct M_driver_intel_device_features_s {
  /// [in] The version the caller knows; [out] the version filled.
  uint32_t version;
  /// [in] sizeof the caller's struct; at least the size of `version`'s.
  uint32_t size;

  // Version 1.
  /// Supported sub-group sizes: bit s is set when size s is supported.
  uint64_t sub_group_sizes;
  /// DPAS (Xe matrix engines), from Intel's experimental module
  /// dot-product properties: M_DRIVER_INTEL_FEATURE_{YES,NO,UNKNOWN}.
  int32_t dpas;
  /// DP4A (four-way integer dot product and accumulate).
  int32_t dp4a;
  /// 2D block array loads and stores (`SPV_INTEL_2d_block_io`), from
  /// Intel's experimental block-array properties.
  int32_t block_2d_load;
  int32_t block_2d_store;
  /// Shared local memory per work-group, in bytes.
  uint32_t max_shared_local_memory;
  /// EUs = slices x sub-slices per slice x EUs per sub-slice; an Xe-core is
  /// a sub-slice.
  uint32_t num_slices;
  uint32_t num_subslices_per_slice;
  uint32_t num_eus_per_subslice;
  uint32_t num_threads_per_eu;
  /// The EU's physical SIMD width.
  uint32_t physical_simd_width;
  /// Core clock, in MHz.
  uint32_t max_clock_mhz;
  /// Device memory as Level Zero reports it (the B70's bus width reads 64,
  /// its clock 19,000 MHz).
  uint32_t memory_bus_width;
  uint32_t memory_clock_mhz;
  uint32_t reserved;
  /// The largest device cache Level Zero reports (the L2), in bytes.
  uint64_t l2_cache_bytes;
} M_driver_intel_device_features_t;

/// Fills `out` with `device`'s features. The plugin queries the device once
/// and answers later calls from a cache.
///
/// @return M_DRIVER_STATUS_BAD_DEVICE for a null device,
///         M_DRIVER_STATUS_INVALID_ARG for a null `out`, a version of 0 or a
///         size smaller than version 1's.
M_DRIVER_API_EXPORT
M_driver_status_t
M_driver_intel_device_features(M_driver_device_t device,
                               M_driver_intel_device_features_t *out);

typedef M_driver_status_t (*M_driver_intel_device_features_func_ptr_t)(
    M_driver_device_t device, M_driver_intel_device_features_t *out);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MOJO_INTEL_DRIVER_INTELDEVICEFEATURES_H
