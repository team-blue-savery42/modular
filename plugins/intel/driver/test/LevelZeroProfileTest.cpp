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
// Launch profiling (MOJO_INTEL_PROFILE, Profile.h): the plugin writes a line
// per launch, copy and synchronize with host and device times. Its own
// binary, since the plugin reads the variable once per process.
//
//===----------------------------------------------------------------------===//

#include "Target/test/Fixture.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

constexpr const char *kVectorAdd =
    "kernels::vector_add(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType."
    "float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],"
    "::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], "
    "::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer["
    "True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, "
    "MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::SIMD[DType.int, 1])";

/// The profile's path prefix, set before the plugin first reads it.
std::string prefix() {
  const char *dir = std::getenv("TEST_TMPDIR");
  return std::string(dir != nullptr ? dir : "/tmp") + "/intel-plugin-profile";
}

const bool kProfileSet = [] {
  setenv("MOJO_INTEL_PROFILE", prefix().c_str(), 1);
  return true;
}();

struct ProfileTest : DriverPluginQueueTest {};

TEST_F(ProfileTest, LaunchesCopiesAndSynchronizesAreRecorded) {
  ASSERT_TRUE(kProfileSet);
  std::ifstream in("plugins/intel/driver/test/data/vector_add.spv",
                   std::ios::binary);
  std::vector<char> spirv((std::istreambuf_iterator<char>(in)), {});
  ASSERT_FALSE(spirv.empty());
  M_driver_static_bundle bundle = {};
  bundle.mapped_data = {spirv.data(), spirv.size()};
  bundle.file_type = "object";
  bundle.file_type_len = 6;
  M_driver_runtime_bundle_t runtime = nullptr;
  M_driver_function_t function = nullptr;
  ASSERT_SUCCESS(bundle_load(contexts[0], &bundle, nullptr, &runtime));
  ASSERT_SUCCESS(function_load(contexts[0], runtime, kVectorAdd,
                               std::strlen(kVectorAdd), &function));
  // Its own queue, so destroying it harvests and flushes the profile here.
  M_driver_queue_t queue = nullptr;
  ASSERT_SUCCESS(queue_create(contexts[0], &queue));
  constexpr int64_t n = 1 << 16;
  constexpr uint64_t bytes = n * sizeof(float);
  std::vector<float> host(n, 1.0f);
  M_driver_memory_t memory = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &memory));
  void *address = nullptr;
  ASSERT_SUCCESS(
      memory_property(memory, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &address));
  M_driver_memory_view view{memory, 0, bytes};
  ASSERT_SUCCESS(queue_copy_to_device(queue, &view, host.data()));
  uint64_t p = reinterpret_cast<uint64_t>(address);
  int64_t count = n;
  void *args[] = {&p, &p, &p, &count};
  uint64_t sizes[] = {8, 8, 8, 8};
  M_driver_queue_execute_config_t config = {};
  config.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
  config.config.gpu.grid = {n / 256, 1, 1};
  config.config.gpu.block = {256, 1, 1};
  constexpr int kLaunches = 5;
  for (int i = 0; i < kLaunches; ++i)
    ASSERT_SUCCESS(queue_execute(queue, function, &config, args, sizes, 4));
  ASSERT_SUCCESS(queue_copy_from_device(queue, host.data(), &view));
  ASSERT_SUCCESS(queue_synchronize(queue));
  EXPECT_EQ(host[0], 32.0f); // doubled five times

  ASSERT_SUCCESS(queue_destroy(contexts[0], queue));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], memory));
  ASSERT_SUCCESS(function_unload(contexts[0], function));
  ASSERT_SUCCESS(bundle_unload(contexts[0], runtime));

  std::ifstream profile(prefix() + "." + std::to_string(getpid()));
  ASSERT_TRUE(profile.good());
  int kernels = 0, copies = 0, syncs = 0, names = 0, header = 0, allocs = 0;
  std::string line;
  while (std::getline(profile, line)) {
    std::istringstream fields(line);
    std::string kind;
    fields >> kind;
    if (kind == "P") {
      unsigned long long hz = 0;
      fields >> hz;
      EXPECT_GT(hz, 0u);
      ++header;
    } else if (kind == "F") {
      names += line.find("vector_add") != std::string::npos;
    } else if (kind == "K") {
      unsigned q, f, gx, gy, gz, bx, by, bz;
      unsigned long long enter, exit;
      long long start, end;
      fields >> q >> f >> gx >> gy >> gz >> bx >> by >> bz >> enter >> exit >>
          start >> end;
      EXPECT_EQ(gx, n / 256);
      EXPECT_EQ(bx, 256u);
      EXPECT_LE(enter, exit);
      EXPECT_GE(start, 0);
      EXPECT_GT(end, start);
      ++kernels;
    } else if (kind == "C") {
      ++copies;
    } else if (kind == "S") {
      ++syncs;
    } else if (kind == "A") {
      ++allocs;
    }
  }
  EXPECT_EQ(header, 1);
  EXPECT_EQ(names, 1);
  EXPECT_EQ(kernels, kLaunches);
  EXPECT_EQ(copies, 2);
  EXPECT_GE(syncs, 1);
  EXPECT_GE(allocs, 1);
}

} // namespace
