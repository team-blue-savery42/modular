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
// A GPU page fault through the HAL. The fault loses the device for the whole
// process, so this is a test of its own. It is manual: on the B70 (xe driver,
// compute-runtime 26.35) the next process to use the GPU after a faulting one
// hangs at its first wait, whatever the delay, and the one after it runs.
//
//===----------------------------------------------------------------------===//

#include "Target/test/Fixture.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

// `boxed_pointer(p, s)` (plugins/intel/tests/codegen/kernels.mojo) writes
// through the address in s = { i64 address @0, i32 v @8 }.
constexpr const char *kBoxedPointer =
    "kernels::boxed_pointer(::Pointer[True, MutUnsafeAnyOrigin, "
    "::SIMD[DType.int32, 1], ::Origin[True, MutUnsafeAnyOrigin](), "
    "AddressSpace.GENERIC],kernels::Boxed)";

// A kernel that writes unmapped memory page-faults, the xe driver resets the
// engine, and compute-runtime may report the interrupted work complete. Waits
// must not keep passing: the free returns, and the next synchronize fails, as
// CUDA reports an illegal address at a later call. The free runs in a thread
// so a hang fails the test instead of stalling it.
TEST_F(DriverPluginQueueTest, WaitsFailAfterAGpuFault) {
  std::ifstream in("plugins/intel/driver/test/data/boxed_pointer.spv",
                   std::ios::binary);
  std::vector<char> spirv((std::istreambuf_iterator<char>(in)), {});
  ASSERT_FALSE(spirv.empty()) << "missing test/data/boxed_pointer.spv";
  M_driver_context_t context = contexts[0];
  M_driver_queue_t queue = queues[0];
  M_driver_static_bundle bundle = {};
  bundle.mapped_data = {spirv.data(), spirv.size()};
  bundle.file_type = "object";
  bundle.file_type_len = 6;
  M_driver_runtime_bundle_t runtime = nullptr;
  ASSERT_SUCCESS(bundle_load(context, &bundle, nullptr, &runtime));
  M_driver_function_t kernel = nullptr;
  ASSERT_SUCCESS(function_load(context, runtime, kBoxedPointer,
                               std::strlen(kBoxedPointer), &kernel));
  M_driver_memory_t memory = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(context, sizeof(int32_t), &memory));
  void *address = nullptr;
  ASSERT_SUCCESS(
      memory_property(memory, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &address));
  uint64_t p = reinterpret_cast<uint64_t>(address);
  unsigned char bytes[16] = {};
  uint64_t unmapped = 0x0000100000000000;
  std::memcpy(bytes, &unmapped, sizeof(unmapped));
  void *args[] = {&p, bytes};
  uint64_t sizes[] = {8, 16};
  M_driver_queue_execute_config_t config = {};
  config.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
  config.config.gpu.grid = {1, 1, 1};
  config.config.gpu.block = {1, 1, 1};
  ASSERT_SUCCESS(queue_execute(queue, kernel, &config, args, sizes, 2));

  auto freed = std::make_shared<std::promise<M_driver_status_t>>();
  std::future<M_driver_status_t> done = freed->get_future();
  std::thread([this, context, memory, freed] {
    freed->set_value(memory_free_sync(context, memory));
  }).detach();
  if (done.wait_for(std::chrono::seconds(60)) != std::future_status::ready) {
    ADD_FAILURE() << "memory_free_sync still waiting 60 s after a GPU fault";
    std::fflush(nullptr);
    std::_Exit(1);
  }
  // Waits ask whether the device is lost at most every millisecond.
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  EXPECT_NE(queue_synchronize(queue), M_DRIVER_STATUS_SUCCESS);
  EXPECT_NE(queue_synchronize(queue), M_DRIVER_STATUS_SUCCESS);

  // The queue's final wait fails too; it is destroyed all the same.
  EXPECT_NE(queue_destroy(context, queue), M_DRIVER_STATUS_SUCCESS);
  queues.clear();
  EXPECT_SUCCESS(function_unload(context, kernel));
  EXPECT_SUCCESS(bundle_unload(context, runtime));
}

} // namespace
