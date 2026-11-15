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
// Device attributes, function attributes and occupancy, which the HAL defines
// in CUDA's terms, answered from Level Zero's device and kernel properties;
// and the entry points Intel GPUs or this plugin don't support.
// CUDA's multiprocessor maps to an Xe-core (Level Zero sub-slice) and its warp
// to the smallest subgroup size.
//
//===----------------------------------------------------------------------===//

#include "ComputeCapability.h"
#include "Plugin.h"

#include <algorithm>
#include <climits>
#include <cstdint>

namespace {

// CUDA attribute IDs, as in MAX's DeviceAttribute and FuncAttribute.
enum DeviceAttribute : int32_t {
  kMaxThreadsPerBlock = 1,
  kMaxBlockDimX = 2,
  kMaxBlockDimY = 3,
  kMaxBlockDimZ = 4,
  kMaxGridDimX = 5,
  kMaxGridDimY = 6,
  kMaxGridDimZ = 7,
  kMaxSharedMemoryPerBlock = 8,
  kWarpSize = 10,
  kClockRate = 13,
  kMultiprocessorCount = 16,
  kMaxThreadsPerMultiprocessor = 39,
  kComputeCapabilityMajor = 75,
  kComputeCapabilityMinor = 76,
  kMaxSharedMemoryPerMultiprocessor = 81,
  kCooperativeLaunch = 95,
  kMaxSharedMemoryPerBlockOptin = 97,
  kMaxBlocksPerMultiprocessor = 106,
};
constexpr int32_t kFuncMaxDynamicSharedSizeBytes = 8;

int32_t clampToInt32(uint64_t value) {
  return static_cast<int32_t>(std::min<uint64_t>(value, INT32_MAX));
}

/// Hardware threads in one Xe-core (EUs times threads per EU).
uint32_t threadsPerXeCore(const M_driver_device &device) {
  return device.props.numEUsPerSubslice * device.props.numThreadsPerEU;
}

/// The smallest subgroup size: what MAX's warp maps to.
uint32_t smallestSubgroup(const M_driver_device &device) {
  const ze_device_compute_properties_t &c = device.compute;
  uint32_t smallest = 0;
  for (uint32_t i = 0; i < c.numSubGroupSizes; ++i)
    if (smallest == 0 || c.subGroupSizes[i] < smallest)
      smallest = c.subGroupSizes[i];
  return smallest != 0 ? smallest : device.props.physicalEUSimdWidth;
}

} // namespace

M_driver_status_t M_driver_device_attribute(M_driver_device_t device,
                                            int32_t attribute, int32_t *value) {
  if (device == nullptr)
    return M_DRIVER_STATUS_BAD_DEVICE;
  if (value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  const ze_device_properties_t &p = device->props;
  const ze_device_compute_properties_t &c = device->compute;
  switch (attribute) {
  case kMaxThreadsPerBlock:
    *value = clampToInt32(c.maxTotalGroupSize);
    break;
  case kMaxBlockDimX:
    *value = clampToInt32(c.maxGroupSizeX);
    break;
  case kMaxBlockDimY:
    *value = clampToInt32(c.maxGroupSizeY);
    break;
  case kMaxBlockDimZ:
    *value = clampToInt32(c.maxGroupSizeZ);
    break;
  case kMaxGridDimX:
    *value = clampToInt32(c.maxGroupCountX);
    break;
  case kMaxGridDimY:
    *value = clampToInt32(c.maxGroupCountY);
    break;
  case kMaxGridDimZ:
    *value = clampToInt32(c.maxGroupCountZ);
    break;
  case kMaxSharedMemoryPerBlock:
  case kMaxSharedMemoryPerBlockOptin:
  case kMaxSharedMemoryPerMultiprocessor:
    // Xe2's shared local memory per work-group equals the Xe-core's.
    *value = clampToInt32(c.maxSharedLocalMemory);
    break;
  case kWarpSize:
    *value = clampToInt32(smallestSubgroup(*device));
    break;
  case kClockRate: // CUDA reports kHz
    *value = clampToInt32(uint64_t(p.coreClockRate) * 1000);
    break;
  case kMultiprocessorCount:
    *value = clampToInt32(uint64_t(p.numSlices) * p.numSubslicesPerSlice);
    break;
  case kMaxThreadsPerMultiprocessor:
    *value = clampToInt32(uint64_t(threadsPerXeCore(*device)) *
                          p.physicalEUSimdWidth);
    break;
  case kMaxBlocksPerMultiprocessor:
    // Each work-group needs at least one hardware thread: an upper bound.
    *value = clampToInt32(threadsPerXeCore(*device));
    break;
  case kComputeCapabilityMajor:
  case kComputeCapabilityMinor:
    return M::Intel::computeCapabilityAttribute(
        device->ipVersion, attribute == kComputeCapabilityMajor, value);
  case kCooperativeLaunch:
    *value = 0;
    break;
  default:
    // Registers, access policy windows and the rest have no Intel equivalent.
    return M_DRIVER_STATUS_UNSUPPORTED;
  }
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_function_set_attribute(M_driver_context_t context,
                                                  M_driver_function_t function,
                                                  int32_t attribute,
                                                  int32_t value) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (function == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  // Level Zero has no per-kernel shared-memory cap to raise: anything up to
  // the device's shared local memory is accepted as is.
  if (attribute == kFuncMaxDynamicSharedSizeBytes && value >= 0 &&
      uint64_t(value) <= context->device->compute.maxSharedLocalMemory)
    return M_DRIVER_STATUS_SUCCESS;
  return M_DRIVER_STATUS_UNSUPPORTED;
}

M_driver_status_t M_driver_function_occupancy_max_active_blocks(
    M_driver_context_t context, M_driver_function_t function,
    int32_t block_size, uint64_t dynamic_shared_memory_bytes,
    int32_t *num_blocks) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (function == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  if (num_blocks == nullptr || block_size <= 0)
    return M_DRIVER_STATUS_INVALID_ARG;
  const M_driver_device &device = *context->device;
  ze_kernel_properties_t kernel = {};
  kernel.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  ze_result_t result =
      device.driver->ze->zeKernelGetProperties(function->kernel, &kernel);
  if (result != ZE_RESULT_SUCCESS)
    return M::Intel::failZe(*device.driver, "zeKernelGetProperties", result);
  // An estimate: work-groups per Xe-core bounded by hardware threads
  // (one per subgroup) and by shared local memory. It ignores large-GRF
  // compilation, which halves the threads per EU.
  uint32_t simd = kernel.maxSubgroupSize != 0 ? kernel.maxSubgroupSize
                                              : smallestSubgroup(device);
  uint64_t threadsPerGroup = (uint64_t(block_size) + simd - 1) / simd;
  uint64_t byThreads = threadsPerXeCore(device) / threadsPerGroup;
  uint64_t sharedPerGroup = kernel.localMemSize + dynamic_shared_memory_bytes;
  uint64_t bySharedMemory =
      sharedPerGroup == 0
          ? UINT64_MAX
          : device.compute.maxSharedLocalMemory / sharedPerGroup;
  *num_blocks = clampToInt32(std::min(byThreads, bySharedMemory));
  return M_DRIVER_STATUS_SUCCESS;
}

// Not supported on Intel GPUs through this plugin.

M_driver_status_t
M_driver_queue_launch_host_func([[maybe_unused]] M_driver_queue_t queue,
                                [[maybe_unused]] void (*fn)(void *),
                                [[maybe_unused]] void *user_data) {
  return M_DRIVER_STATUS_UNSUPPORTED;
}

M_driver_status_t
M_driver_queue_wait_value64([[maybe_unused]] M_driver_queue_t queue,
                            [[maybe_unused]] uint64_t device_address,
                            [[maybe_unused]] uint64_t value) {
  return M_DRIVER_STATUS_UNSUPPORTED;
}

M_driver_status_t M_driver_copy_inter_device_sync(
    [[maybe_unused]] M_driver_context_t context,
    [[maybe_unused]] M_driver_memory_view_t dst,
    [[maybe_unused]] M_driver_memory_view_t src,
    [[maybe_unused]] M_driver_context_t src_context) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t M_driver_queue_copy_inter_device(
    [[maybe_unused]] M_driver_queue_t queue,
    [[maybe_unused]] M_driver_memory_view_t dst,
    [[maybe_unused]] M_driver_memory_view_t src,
    [[maybe_unused]] M_driver_context_t src_context) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t M_driver_tensor_map_encode_tiled(
    [[maybe_unused]] M_driver_context_t context,
    [[maybe_unused]] void *tensor_map, [[maybe_unused]] int32_t data_type,
    [[maybe_unused]] uint32_t rank,
    [[maybe_unused]] uint64_t global_device_address,
    [[maybe_unused]] const uint64_t *global_dim,
    [[maybe_unused]] const uint64_t *global_strides,
    [[maybe_unused]] const uint32_t *box_dim,
    [[maybe_unused]] const uint32_t *element_strides,
    [[maybe_unused]] int32_t interleave, [[maybe_unused]] int32_t swizzle,
    [[maybe_unused]] int32_t l2_promotion, [[maybe_unused]] int32_t oob_fill) {
  return M_DRIVER_STATUS_UNSUPPORTED;
}

M_driver_status_t M_driver_tensor_map_encode_im2col(
    [[maybe_unused]] M_driver_context_t context,
    [[maybe_unused]] void *tensor_map, [[maybe_unused]] int32_t data_type,
    [[maybe_unused]] uint32_t rank,
    [[maybe_unused]] uint64_t global_device_address,
    [[maybe_unused]] const uint64_t *global_dim,
    [[maybe_unused]] const uint64_t *global_strides,
    [[maybe_unused]] const int32_t *pixel_box_lower_corner,
    [[maybe_unused]] const int32_t *pixel_box_upper_corner,
    [[maybe_unused]] int32_t channels_per_pixel,
    [[maybe_unused]] int32_t pixels_per_column,
    [[maybe_unused]] const uint32_t *element_strides,
    [[maybe_unused]] int32_t interleave, [[maybe_unused]] int32_t swizzle,
    [[maybe_unused]] int32_t l2_promotion, [[maybe_unused]] int32_t oob_fill) {
  return M_DRIVER_STATUS_UNSUPPORTED;
}
