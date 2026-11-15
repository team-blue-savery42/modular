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
// Latency of the HAL operations MAX leans on, through the Level Zero plugin:
// queue copies, events, cross-queue waits, allocation and async free. Prints
// medians in microseconds; asserts nothing about speed. Run with
//   bazel test //plugins/intel/driver:LevelZeroPluginBench --test_output=all
//
//===----------------------------------------------------------------------===//

#include "Target/test/Fixture.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>

namespace {

double median(int reps, const std::function<void()> &body) {
  body(); // warm up
  std::vector<double> us;
  for (int i = 0; i < reps; ++i) {
    auto start = std::chrono::steady_clock::now();
    body();
    us.push_back(std::chrono::duration<double, std::micro>(
                     std::chrono::steady_clock::now() - start)
                     .count());
  }
  std::sort(us.begin(), us.end());
  return us[us.size() / 2];
}

TEST_F(DriverPluginQueueTest, Bench) {
  M_driver_context_t context = contexts[0];
  M_driver_queue_t queue = queues[0];
  constexpr uint64_t kBig = 64u << 20;
  M_driver_memory_t device = nullptr, device2 = nullptr, pinned = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(context, kBig, &device));
  ASSERT_SUCCESS(memory_alloc_sync(context, kBig, &device2));
  ASSERT_SUCCESS(memory_alloc_pinned(context, kBig, &pinned));
  void *host = nullptr;
  ASSERT_SUCCESS(
      memory_property(pinned, M_DRIVER_MEMORY_PROPERTY_HOST_ADDRESS, &host));
  std::vector<char> pageable(kBig, 1);

  std::printf("== queue copy + queue_synchronize (us)\n");
  for (uint64_t bytes : {uint64_t(4096), uint64_t(1) << 20, kBig}) {
    M_driver_memory_view view{device, 0, bytes};
    M_driver_memory_view view2{device2, 0, bytes};
    int reps = bytes == kBig ? 20 : 300;
    double h2d = median(reps, [&] {
      EXPECT_SUCCESS(queue_copy_to_device(queue, &view, host));
      EXPECT_SUCCESS(queue_synchronize(queue));
    });
    double d2h = median(reps, [&] {
      EXPECT_SUCCESS(queue_copy_from_device(queue, host, &view));
      EXPECT_SUCCESS(queue_synchronize(queue));
    });
    double h2dPage = median(reps, [&] {
      EXPECT_SUCCESS(queue_copy_to_device(queue, &view, pageable.data()));
      EXPECT_SUCCESS(queue_synchronize(queue));
    });
    double d2d = median(reps, [&] {
      EXPECT_SUCCESS(queue_copy_intra_device(queue, &view2, &view));
      EXPECT_SUCCESS(queue_synchronize(queue));
    });
    std::printf("%8llu KiB: H2D pinned %.1f  D2H pinned %.1f  H2D pageable "
                "%.1f  D2D %.1f\n",
                static_cast<unsigned long long>(bytes >> 10), h2d, d2h, h2dPage,
                d2d);
  }

  std::printf("== events (us)\n");
  double create = median(300, [&] {
    M_driver_event_t event = nullptr;
    EXPECT_SUCCESS(event_create(context, M_DRIVER_EVENT_FLAG_NONE, &event));
    EXPECT_SUCCESS(event_destroy(context, event));
  });
  M_driver_event_t event = nullptr;
  ASSERT_SUCCESS(
      event_create(context, M_DRIVER_EVENT_FLAG_CPU_VISIBLE, &event));
  double record = median(500, [&] {
    EXPECT_SUCCESS(queue_record_event(queue, event));
    EXPECT_SUCCESS(event_synchronize(context, event));
  });
  M_driver_queue_t other = nullptr;
  ASSERT_SUCCESS(queue_create(context, &other));
  M_driver_memory_view small{device, 0, 4096};
  double cross = median(300, [&] {
    EXPECT_SUCCESS(queue_copy_to_device(queue, &small, host));
    EXPECT_SUCCESS(queue_record_event(queue, event));
    EXPECT_SUCCESS(queue_wait_for_events(other, &event, 1));
    EXPECT_SUCCESS(queue_copy_from_device(other, host, &small));
    EXPECT_SUCCESS(queue_synchronize(other));
  });
  std::printf("create+destroy %.1f  record+synchronize %.1f  cross-queue "
              "4 KiB copy -> record -> wait -> copy -> sync %.1f\n",
              create, record, cross);

  std::printf("== allocation (us)\n");
  // Allocations are made resident when created, so the cost grows with size.
  auto allocFree = [&](uint64_t bytes) {
    return median(100, [&] {
      M_driver_memory_t m = nullptr;
      EXPECT_SUCCESS(memory_alloc_sync(context, bytes, &m));
      EXPECT_SUCCESS(memory_free_sync(context, m));
    });
  };
  double alloc4K = allocFree(4 << 10);
  double allocSync = allocFree(1 << 20);
  double alloc64M = allocFree(64 << 20);
  M_driver_memory_view big{device, 0, kBig};
  // Async free while a 64 MiB pinned copy (about 2.4 ms) is still running: a
  // stream-ordered free returns at once, a draining one waits for the copy.
  std::vector<double> freeCall;
  for (int i = 0; i < 20; ++i) {
    M_driver_memory_t m = nullptr;
    EXPECT_SUCCESS(memory_alloc_async(queue, 1 << 20, &m));
    EXPECT_SUCCESS(queue_copy_to_device(queue, &big, host));
    auto start = std::chrono::steady_clock::now();
    EXPECT_SUCCESS(memory_free_async(queue, m));
    freeCall.push_back(std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - start)
                           .count());
    EXPECT_SUCCESS(queue_synchronize(queue));
  }
  std::sort(freeCall.begin(), freeCall.end());
  std::printf("alloc+free 4 KiB %.1f  64 MiB %.1f\n", alloc4K, alloc64M);
  std::printf("alloc+free 1 MiB %.1f  free_async with a 64 MiB copy in "
              "flight returns after %.1f\n",
              allocSync, freeCall[freeCall.size() / 2]);

  ASSERT_SUCCESS(queue_destroy(context, other));
  ASSERT_SUCCESS(event_destroy(context, event));
  ASSERT_SUCCESS(memory_free_pinned(context, pinned));
  ASSERT_SUCCESS(memory_free_sync(context, device2));
  ASSERT_SUCCESS(memory_free_sync(context, device));
}

} // namespace
