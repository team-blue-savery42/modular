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
// Level Zero specifics on top of Modular's HAL conformance suite
// (Target/test/PluginTest.cpp, linked in through :DriverPluginTests).
//
//===----------------------------------------------------------------------===//

#include "Target/test/Fixture.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

TEST_F(DriverPluginTest, LevelZeroApiName) {
  const char *apiName = nullptr;
  ASSERT_SUCCESS(property(driver, M_DRIVER_PROPERTY_API, &apiName));
  EXPECT_STREQ(apiName, "level_zero");
}

TEST_F(DriverPluginDeviceTest, ArchIsAnOclocDeviceName) {
  for (M_driver_device_t device : devices) {
    char arch[M_DRIVER_DEVICE_PROPERTY_MAX_LEN] = {};
    ASSERT_SUCCESS(
        device_property(device, M_DRIVER_DEVICE_PROPERTY_ARCH, arch));
    char name[M_DRIVER_DEVICE_PROPERTY_MAX_LEN] = {};
    ASSERT_SUCCESS(
        device_property(device, M_DRIVER_DEVICE_PROPERTY_NAME, name));
    std::cerr << name << ": " << arch << "\n";
    // A table name such as "bmg-g31", or a dotted IP version such as "20.2.0";
    // `ocloc -device` accepts both.
    ASSERT_NE(std::strlen(arch), 0u);
    EXPECT_EQ(std::string(arch).find(' '), std::string::npos);
  }
}

TEST_F(DriverPluginDeviceTest, DlpackDeviceIsOneApi) {
  for (size_t i = 0; i < devices.size(); ++i) {
    M_driver_dlpack_device dl = {};
    ASSERT_SUCCESS(device_property(
        devices[i], M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE, &dl));
    EXPECT_EQ(dl.device_type, 14); // kDLOneAPI
    EXPECT_EQ(dl.device_id, static_cast<int32_t>(i));
  }
}

// Reports memory info and median 1 MiB copy times through the synchronous
// context copies (no timing assertion: timings depend on what else the
// machine runs).
TEST_F(DriverPluginContextTest, MemoryInfoAndCopyTimes) {
  constexpr uint64_t kSize = 1 << 20;
  constexpr int kReps = 200;
  for (M_driver_context_t context : contexts) {
    uint64_t freeBytes = 0, totalBytes = 0;
    ASSERT_SUCCESS(context_memory_info(context, &freeBytes, &totalBytes));
    // std::dec: an earlier suite test leaves std::cerr in hex.
    std::cerr << std::dec << "memory: " << (freeBytes >> 20) << " MiB free of "
              << (totalBytes >> 20) << " MiB\n";
    EXPECT_GT(freeBytes, 0u);

    M_driver_memory_t device = nullptr, pinned = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, kSize, &device));
    ASSERT_SUCCESS(memory_alloc_pinned(context, kSize, &pinned));
    void *pinnedPtr = nullptr;
    ASSERT_SUCCESS(memory_property(
        pinned, M_DRIVER_MEMORY_PROPERTY_HOST_ADDRESS, &pinnedPtr));
    std::vector<uint8_t> pageable(kSize, 7);
    M_driver_memory_view view{device, 0, kSize};

    auto median = [&](auto &&copy) {
      std::vector<double> us;
      for (int i = 0; i < kReps; ++i) {
        auto start = std::chrono::steady_clock::now();
        EXPECT_SUCCESS(copy());
        us.push_back(std::chrono::duration<double, std::micro>(
                         std::chrono::steady_clock::now() - start)
                         .count());
      }
      std::sort(us.begin(), us.end());
      return us[us.size() / 2];
    };
    double h2dPageable = median(
        [&] { return copy_to_device_sync(context, &view, pageable.data()); });
    double d2hPageable = median(
        [&] { return copy_from_device_sync(context, pageable.data(), &view); });
    double h2dPinned =
        median([&] { return copy_to_device_sync(context, &view, pinnedPtr); });
    double d2hPinned = median(
        [&] { return copy_from_device_sync(context, pinnedPtr, &view); });
    std::cerr << "1 MiB H2D pageable/pinned us: " << h2dPageable << " / "
              << h2dPinned << "\n1 MiB D2H pageable/pinned us: " << d2hPageable
              << " / " << d2hPinned << "\n";

    ASSERT_SUCCESS(memory_free_pinned(context, pinned));
    ASSERT_SUCCESS(memory_free_sync(context, device));
  }
}

TEST_F(DriverPluginDeviceTest, CudaStyleAttributes) {
  struct Expect {
    int32_t id;
    const char *name;
  };
  // MAX's DeviceAttribute IDs that have Intel equivalents.
  constexpr Expect kSupported[] = {{1, "max threads per block"},
                                   {2, "max block dim x"},
                                   {5, "max grid dim x"},
                                   {8, "shared memory per block"},
                                   {10, "warp size"},
                                   {13, "clock rate (kHz)"},
                                   {16, "multiprocessor count"},
                                   {39, "max threads per multiprocessor"},
                                   {81, "shared memory per multiprocessor"},
                                   {106, "max blocks per multiprocessor"}};
  for (M_driver_device_t device : devices) {
    for (const Expect &e : kSupported) {
      SCOPED_TRACE(e.name);
      int32_t value = 0;
      ASSERT_SUCCESS(device_attribute(device, e.id, &value));
      std::cerr << std::dec << e.name << ": " << value << "\n";
      EXPECT_GT(value, 0) << e.name;
    }
    int32_t value = 0;
    EXPECT_SUCCESS(device_attribute(device, 95, &value)); // cooperative launch
    EXPECT_EQ(value, 0);
    // Compute capability: the IP version's architecture and
    // release, so MAX's major * 10 + minor is the target's `compute` * 10.
    // The arch property is a table name (Battlemage, Raptor Lake-P, Arrow
    // Lake-S) or the dotted IP version; a two-digit release (dg2-g10 is
    // 12.55) keeps its first digit; no IP version (0.0.0) leaves both
    // attributes unsupported; any other architecture, pre-Xe ones (below
    // 12) included, is answered as is.
    char arch[M_DRIVER_DEVICE_PROPERTY_MAX_LEN] = {};
    ASSERT_SUCCESS(
        device_property(device, M_DRIVER_DEVICE_PROPERTY_ARCH, arch));
    unsigned ipArchitecture = 0, ipRelease = 0, ipRevision = 0;
    bool dotted = std::sscanf(arch, "%u.%u.%u", &ipArchitecture, &ipRelease,
                              &ipRevision) == 3;
    if (dotted && ipArchitecture == 0) {
      std::cerr << arch << ": no compute capability\n";
      EXPECT_EQ(device_attribute(device, 75, &value),
                M_DRIVER_STATUS_UNSUPPORTED);
      EXPECT_EQ(device_attribute(device, 76, &value),
                M_DRIVER_STATUS_UNSUPPORTED);
      continue;
    }
    int32_t major = 0, minor = -1;
    ASSERT_SUCCESS(device_attribute(device, 75, &major));
    ASSERT_SUCCESS(device_attribute(device, 76, &minor));
    std::cerr << "compute capability: " << major << "." << minor << "\n";
    EXPECT_GE(major, 1);
    EXPECT_GE(minor, 0);
    EXPECT_LE(minor, 9);
    if (dotted) {
      EXPECT_EQ(major, static_cast<int32_t>(ipArchitecture));
      EXPECT_EQ(minor, static_cast<int32_t>(ipRelease > 9 ? ipRelease / 10
                                                          : ipRelease));
    } else if (std::string(arch) == "rpl-p") { // compute=12.3
      EXPECT_EQ(major, 12);
      EXPECT_EQ(minor, 3);
    } else if (std::string(arch) == "arl-s") { // compute=12.7
      EXPECT_EQ(major, 12);
      EXPECT_EQ(minor, 7);
    } else if (std::string(arch) == "bmg-g31") { // compute=20.2
      EXPECT_EQ(major, 20);
      EXPECT_EQ(minor, 2);
    } else if (std::string(arch) == "bmg-g21") { // compute=20.1
      EXPECT_EQ(major, 20);
      EXPECT_EQ(minor, 1);
    } else {
      ADD_FAILURE() << "arch " << arch << " is neither a known name nor dotted";
    }
  }
}

// CUDA semantics for pageable host memory: the source of an async H2D copy is
// reusable as soon as the call returns, and an async D2H into pageable memory
// has landed when it returns.
TEST_F(DriverPluginQueueTest, PageableCopiesHaveCudaSemantics) {
  // 64 KiB goes through the pinned staging path, 64 MiB the direct one.
  for (uint64_t kSize : {uint64_t(64) << 10, uint64_t(64) << 20}) {
    SCOPED_TRACE(kSize);
    M_driver_memory_t device = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(contexts[0], kSize, &device));
    M_driver_memory_view view{device, 0, kSize};
    std::vector<uint8_t> source(kSize);
    for (uint64_t i = 0; i < kSize; ++i)
      source[i] = static_cast<uint8_t>(i * 7);
    std::vector<uint8_t> expected = source;
    ASSERT_SUCCESS(queue_copy_to_device(queues[0], &view, source.data()));
    std::fill(source.begin(), source.end(), 0); // reuse at once
    std::vector<uint8_t> back(kSize, 1);
    ASSERT_SUCCESS(queue_copy_from_device(queues[0], back.data(), &view));
    // No queue_synchronize: the pageable destination must already hold it.
    EXPECT_TRUE(back == expected);
    ASSERT_SUCCESS(queue_synchronize(queues[0]));
    ASSERT_SUCCESS(memory_free_sync(contexts[0], device));
  }
}

TEST_F(DriverPluginQueueTest, FillRejectsMisalignedDestination) {
  M_driver_memory_t device = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], 64, &device));
  M_driver_memory_view misaligned{device, 2, 8};
  EXPECT_EQ(queue_fill(queues[0], &misaligned, 0xABCD1234u, 4),
            M_DRIVER_STATUS_INVALID_ARG);
  EXPECT_EQ(fill_sync(contexts[0], &misaligned, 0xABCD1234u, 4),
            M_DRIVER_STATUS_INVALID_ARG);
  M_driver_memory_view aligned{device, 4, 8};
  EXPECT_SUCCESS(queue_fill(queues[0], &aligned, 0xABCD1234u, 4));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], device));
}

TEST_F(DriverPluginQueueTest, NoNativeQueueHandle) {
  uint64_t handle = 0;
  EXPECT_EQ(
      queue_property(queues[0], M_DRIVER_QUEUE_PROPERTY_NATIVE_HANDLE, &handle),
      M_DRIVER_STATUS_UNIMPLEMENTED);
}

TEST_F(DriverPluginTest, DeviceGetOutOfRange) {
  int64_t count = 0;
  ASSERT_SUCCESS(device_count(driver, &count));
  M_driver_device_t device = nullptr;
  EXPECT_EQ(device_get(driver, count, &device), M_DRIVER_STATUS_BAD_DEVICE);
  EXPECT_EQ(device_get(driver, -1, &device), M_DRIVER_STATUS_BAD_DEVICE);
}

} // namespace
