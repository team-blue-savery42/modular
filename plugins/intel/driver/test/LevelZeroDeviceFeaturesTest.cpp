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
// M_driver_intel_device_features (IntelDeviceFeatures.h), the plugin's entry
// point outside the HAL: resolved by name as MAX's Intel extension set
// resolves it, checked against the HAL's device attributes, its argument
// checks, versioning, its cache, and the cost of a cached call.
//
//===----------------------------------------------------------------------===//

#include "Target/test/Fixture.h"

#include "plugins/intel/driver/IntelDeviceFeatures.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <vector>

namespace {

struct DeviceFeaturesTest : DriverPluginDeviceTest {
  void SetUp() override {
    DriverPluginDeviceTest::SetUp();
    if (HasFatalFailure() || IsSkipped())
      return;
    features = reinterpret_cast<M_driver_intel_device_features_func_ptr_t>(
        dlsym(soHandle, "M_driver_intel_device_features"));
    ASSERT_NE(features, nullptr) << "M_driver_intel_device_features missing";
  }

  static M_driver_intel_device_features_t request() {
    M_driver_intel_device_features_t f = {};
    f.version = M_DRIVER_INTEL_DEVICE_FEATURES_VERSION_1;
    f.size = sizeof(f);
    return f;
  }

  M_driver_intel_device_features_func_ptr_t features = nullptr;
};

bool isFeature(int32_t value) {
  return value == M_DRIVER_INTEL_FEATURE_UNKNOWN ||
         value == M_DRIVER_INTEL_FEATURE_NO ||
         value == M_DRIVER_INTEL_FEATURE_YES;
}

TEST_F(DeviceFeaturesTest, AgreesWithTheDeviceAttributes) {
  for (M_driver_device_t device : devices) {
    M_driver_intel_device_features_t f = request();
    ASSERT_SUCCESS(features(device, &f));
    EXPECT_EQ(f.version, M_DRIVER_INTEL_DEVICE_FEATURES_VERSION_1);
    EXPECT_EQ(f.size, sizeof(f));
    for (int32_t value : {f.dpas, f.dp4a, f.block_2d_load, f.block_2d_store})
      EXPECT_TRUE(isFeature(value)) << value;
    ASSERT_NE(f.sub_group_sizes, 0u);
    // The HAL's warp size is the smallest sub-group size (Limits.cpp).
    int32_t warp = 0, sharedPerBlock = 0, xeCores = 0, clockKHz = 0;
    ASSERT_SUCCESS(device_attribute(device, 10, &warp));
    ASSERT_SUCCESS(device_attribute(device, 8, &sharedPerBlock));
    ASSERT_SUCCESS(device_attribute(device, 16, &xeCores));
    ASSERT_SUCCESS(device_attribute(device, 13, &clockKHz));
    EXPECT_EQ(__builtin_ctzll(f.sub_group_sizes), warp);
    EXPECT_EQ(int64_t(f.max_shared_local_memory), sharedPerBlock);
    EXPECT_EQ(int64_t(f.num_slices) * f.num_subslices_per_slice, xeCores);
    EXPECT_EQ(int64_t(f.max_clock_mhz) * 1000, clockKHz);
    EXPECT_GT(f.num_eus_per_subslice, 0u);
    EXPECT_GT(f.num_threads_per_eu, 0u);
    EXPECT_GT(f.l2_cache_bytes, 0u);
    std::printf("sub-group sizes 0x%llx dpas %d dp4a %d block load %d store %d "
                "slm %u eus %u x %u x %u threads/eu %u simd %u clock %u MHz "
                "memory %u bits %u MHz l2 %llu\n",
                static_cast<unsigned long long>(f.sub_group_sizes), f.dpas,
                f.dp4a, f.block_2d_load, f.block_2d_store,
                f.max_shared_local_memory, f.num_eus_per_subslice,
                f.num_subslices_per_slice, f.num_slices, f.num_threads_per_eu,
                f.physical_simd_width, f.max_clock_mhz, f.memory_bus_width,
                f.memory_clock_mhz,
                static_cast<unsigned long long>(f.l2_cache_bytes));
  }
}

TEST_F(DeviceFeaturesTest, RejectsBadArguments) {
  M_driver_intel_device_features_t f = request();
  EXPECT_EQ(features(nullptr, &f), M_DRIVER_STATUS_BAD_DEVICE);
  for (M_driver_device_t device : devices) {
    EXPECT_EQ(features(device, nullptr), M_DRIVER_STATUS_INVALID_ARG);
    f = request();
    f.version = 0;
    EXPECT_EQ(features(device, &f), M_DRIVER_STATUS_INVALID_ARG);
    f = request();
    f.size = sizeof(f) - 8;
    EXPECT_EQ(features(device, &f), M_DRIVER_STATUS_INVALID_ARG);
  }
}

// A caller that knows a later version gets version 1 back, the rest of its
// struct untouched.
TEST_F(DeviceFeaturesTest, AnswersALaterVersionWithItsOwn) {
  struct Later {
    M_driver_intel_device_features_t v1;
    uint64_t v2Field;
  };
  for (M_driver_device_t device : devices) {
    Later later = {request(), 0x5a5a5a5a5a5a5a5aull};
    later.v1.version = 2;
    later.v1.size = sizeof(later);
    ASSERT_SUCCESS(features(device, &later.v1));
    EXPECT_EQ(later.v1.version, M_DRIVER_INTEL_DEVICE_FEATURES_VERSION_1);
    EXPECT_EQ(later.v1.size, sizeof(later));
    EXPECT_EQ(later.v2Field, 0x5a5a5a5a5a5a5a5aull);
  }
}

// Every call after the first answers from the cache: the same bytes, and
// (printed, not asserted) well under a microsecond.
TEST_F(DeviceFeaturesTest, CachedCallsAreCheap) {
  for (M_driver_device_t device : devices) {
    M_driver_intel_device_features_t first = request();
    auto start = std::chrono::steady_clock::now();
    ASSERT_SUCCESS(features(device, &first));
    double firstUs = std::chrono::duration<double, std::micro>(
                         std::chrono::steady_clock::now() - start)
                         .count();
    constexpr int kCalls = 1000000;
    std::vector<double> perCall;
    for (int r = 0; r < 20; ++r) {
      M_driver_intel_device_features_t again = request();
      start = std::chrono::steady_clock::now();
      for (int i = 0; i < kCalls / 20; ++i) {
        again.version = M_DRIVER_INTEL_DEVICE_FEATURES_VERSION_1;
        ASSERT_SUCCESS(features(device, &again));
      }
      perCall.push_back(std::chrono::duration<double, std::nano>(
                            std::chrono::steady_clock::now() - start)
                            .count() /
                        (kCalls / 20));
      EXPECT_EQ(std::memcmp(&first, &again, sizeof(first)), 0);
    }
    std::sort(perCall.begin(), perCall.end());
    std::printf("first call %.1f us (this test's driver); "
                "cached call median %.1f ns\n",
                firstUs, perCall[perCall.size() / 2]);
  }
}

} // namespace
