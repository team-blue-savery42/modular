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
// Intel device graphs (IntelGraphs.h), entry points outside the HAL resolved
// by name as MAX's Intel extension set resolves them: capture a queue's
// kernels and copies, launch the graph many times, memory freed during a
// capture, memory a capture allocated and freed after it, what a capture
// refuses, and a graph launch's cost against the same kernels launched one
// by one.
//
//===----------------------------------------------------------------------===//

#include "Target/test/Fixture.h"

#include "plugins/intel/driver/IntelGraphs.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

constexpr const char *kVectorAdd =
    "kernels::vector_add(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType."
    "float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],"
    "::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], "
    "::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer["
    "True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, "
    "MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::SIMD[DType.int, 1])";

struct GraphTest : DriverPluginQueueTest {
  void SetUp() override {
    DriverPluginQueueTest::SetUp();
    if (HasFatalFailure() || IsSkipped())
      return;
#define RESOLVE(name)                                                          \
  name = reinterpret_cast<M_driver_##name##_func_ptr_t>(                       \
      dlsym(soHandle, "M_driver_" #name));                                     \
  ASSERT_NE(name, nullptr) << "M_driver_" #name " missing";
    RESOLVE(intel_graphs_version)
    RESOLVE(intel_graph_capture_begin)
    RESOLVE(intel_graph_capture_end)
    RESOLVE(intel_graph_launch)
    RESOLVE(intel_graph_destroy)
#undef RESOLVE
    uint32_t version = 0;
    ASSERT_SUCCESS(intel_graphs_version(devices[0], &version));
    if (version < M_DRIVER_INTEL_GRAPHS_VERSION_1)
      GTEST_SKIP() << "the driver has no graph capture";
    std::ifstream in("plugins/intel/driver/test/data/vector_add.spv",
                     std::ios::binary);
    spirv.assign(std::istreambuf_iterator<char>(in), {});
    ASSERT_FALSE(spirv.empty()) << "missing test/data/vector_add.spv";
    M_driver_static_bundle bundle = {};
    bundle.mapped_data = {spirv.data(), spirv.size()};
    bundle.file_type = "object";
    bundle.file_type_len = 6;
    ASSERT_SUCCESS(bundle_load(contexts[0], &bundle, nullptr, &runtime));
    ASSERT_SUCCESS(function_load(contexts[0], runtime, kVectorAdd,
                                 std::strlen(kVectorAdd), &function));
  }

  void TearDown() override {
    if (function != nullptr)
      EXPECT_SUCCESS(function_unload(contexts[0], function));
    if (runtime != nullptr)
      EXPECT_SUCCESS(bundle_unload(contexts[0], runtime));
    DriverPluginQueueTest::TearDown();
  }

  M_driver_status_t launch(uint64_t a, uint64_t b, uint64_t c, int64_t n) {
    void *args[] = {&a, &b, &c, &n};
    uint64_t sizes[] = {8, 8, 8, 8};
    M_driver_queue_execute_config_t config = {};
    config.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
    config.config.gpu.grid = {static_cast<uint32_t>((n + 255) / 256), 1, 1};
    config.config.gpu.block = {256, 1, 1};
    return queue_execute(queues[0], function, &config, args, sizes, 4);
  }

  uint64_t addressOf(M_driver_memory_t memory) {
    void *address = nullptr;
    EXPECT_SUCCESS(
        memory_property(memory, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &address));
    return reinterpret_cast<uint64_t>(address);
  }

  M_driver_intel_graphs_version_func_ptr_t intel_graphs_version = nullptr;
  M_driver_intel_graph_capture_begin_func_ptr_t intel_graph_capture_begin =
      nullptr;
  M_driver_intel_graph_capture_end_func_ptr_t intel_graph_capture_end = nullptr;
  M_driver_intel_graph_launch_func_ptr_t intel_graph_launch = nullptr;
  M_driver_intel_graph_destroy_func_ptr_t intel_graph_destroy = nullptr;
  std::vector<char> spirv;
  M_driver_runtime_bundle_t runtime = nullptr;
  M_driver_function_t function = nullptr;
};

// c = a + b, then a = c (a copy), captured once and launched 5 times:
// a holds a0 + k b after the k-th launch. The graph also owns a scratch
// buffer allocated and freed during the capture.
TEST_F(GraphTest, LaunchesRepeatTheCapturedWork) {
  constexpr int64_t n = 1 << 16;
  constexpr uint64_t bytes = n * sizeof(float);
  std::vector<float> a(n), b(n), out(n);
  for (int64_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i % 97);
    b[i] = 0.25f * static_cast<float>(i % 13);
  }
  M_driver_memory_t da = nullptr, db = nullptr, dc = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &da));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &db));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &dc));
  M_driver_memory_view va{da, 0, bytes}, vb{db, 0, bytes}, vc{dc, 0, bytes};
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &va, a.data()));
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &vb, b.data()));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));

  ASSERT_SUCCESS(intel_graph_capture_begin(queues[0]));
  EXPECT_EQ(intel_graph_capture_begin(queues[0]), M_DRIVER_STATUS_INVALID_ARG);
  M_driver_memory_t scratch = nullptr;
  ASSERT_SUCCESS(memory_alloc_async(queues[0], bytes, &scratch));
  M_driver_memory_view vs{scratch, 0, bytes};
  ASSERT_SUCCESS(launch(addressOf(da), addressOf(db), addressOf(scratch), n));
  ASSERT_SUCCESS(queue_copy_intra_device(queues[0], &vc, &vs));
  ASSERT_SUCCESS(queue_copy_intra_device(queues[0], &va, &vc));
  // Freed while capturing: the graph keeps it, so its address stays valid.
  ASSERT_SUCCESS(memory_free_async(queues[0], scratch));
  // Nothing captured has run: a host wait on the queue is refused.
  EXPECT_NE(queue_synchronize(queues[0]), M_DRIVER_STATUS_SUCCESS);
  M_driver_intel_graph_t graph = nullptr;
  ASSERT_SUCCESS(intel_graph_capture_end(queues[0], &graph));
  ASSERT_NE(graph, nullptr);
  M_driver_intel_graph_t none = nullptr;
  EXPECT_EQ(intel_graph_capture_end(queues[0], &none),
            M_DRIVER_STATUS_INVALID_ARG);

  // The capture ran nothing.
  ASSERT_SUCCESS(queue_copy_from_device(queues[0], out.data(), &va));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  EXPECT_EQ(std::memcmp(out.data(), a.data(), bytes), 0);
  // A buffer allocated after the capture doesn't reuse the graph's scratch.
  M_driver_memory_t other = nullptr;
  ASSERT_SUCCESS(memory_alloc_async(queues[0], bytes, &other));
  EXPECT_NE(addressOf(other), 0u);
  M_driver_memory_view vo{other, 0, bytes};
  ASSERT_SUCCESS(queue_set_memory(queues[0], &vo, 0x7f));

  for (int k = 1; k <= 5; ++k) {
    ASSERT_SUCCESS(intel_graph_launch(queues[0], graph));
    ASSERT_SUCCESS(queue_copy_from_device(queues[0], out.data(), &va));
    ASSERT_SUCCESS(queue_synchronize(queues[0]));
    int64_t wrong = 0;
    for (int64_t i = 0; i < n; ++i)
      wrong += out[i] != a[i] + static_cast<float>(k) * b[i];
    EXPECT_EQ(wrong, 0) << "after launch " << k;
  }
  ASSERT_SUCCESS(memory_free_async(queues[0], other));
  EXPECT_SUCCESS(intel_graph_destroy(graph));
  EXPECT_SUCCESS(intel_graph_destroy(nullptr));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], da));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], db));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], dc));
}

// Two graphs captured one after the other share their scratch through the
// context's graph pool: the second capture's scratch
// allocation gets the first one's freed block, and each graph still
// computes its own result when they launch in turn.
TEST_F(GraphTest, CapturesShareTheirScratch) {
  constexpr int64_t n = 1 << 14;
  constexpr uint64_t bytes = n * sizeof(float);
  std::vector<float> a(n, 1.0f), b(n, 2.0f), out(n);
  M_driver_memory_t da = nullptr, db = nullptr, dc1 = nullptr, dc2 = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &da));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &db));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &dc1));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &dc2));
  M_driver_memory_view va{da, 0, bytes}, vb{db, 0, bytes};
  M_driver_memory_view vc1{dc1, 0, bytes}, vc2{dc2, 0, bytes};
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &va, a.data()));
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &vb, b.data()));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  M_driver_intel_graph_t graphs[2] = {nullptr, nullptr};
  uint64_t scratchAt[2] = {0, 0};
  M_driver_memory_view *outs[2] = {&vc1, &vc2};
  for (int g = 0; g < 2; ++g) {
    ASSERT_SUCCESS(intel_graph_capture_begin(queues[0]));
    M_driver_memory_t scratch = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &scratch));
    scratchAt[g] = addressOf(scratch);
    M_driver_memory_view vs{scratch, 0, bytes};
    // Graph 0: c1 = a + b; graph 1: c2 = b + b.
    uint64_t x = g == 0 ? addressOf(da) : addressOf(db);
    ASSERT_SUCCESS(launch(x, addressOf(db), addressOf(scratch), n));
    ASSERT_SUCCESS(queue_copy_intra_device(queues[0], outs[g], &vs));
    ASSERT_SUCCESS(memory_free_sync(contexts[0], scratch));
    ASSERT_SUCCESS(intel_graph_capture_end(queues[0], &graphs[g]));
  }
  EXPECT_EQ(scratchAt[0], scratchAt[1]);
  for (int rep = 0; rep < 3; ++rep) {
    for (int g = 0; g < 2; ++g)
      ASSERT_SUCCESS(intel_graph_launch(queues[0], graphs[g]));
  }
  ASSERT_SUCCESS(queue_copy_from_device(queues[0], out.data(), &vc1));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  EXPECT_EQ(out[0], 3.0f);
  EXPECT_EQ(out[n - 1], 3.0f);
  ASSERT_SUCCESS(queue_copy_from_device(queues[0], out.data(), &vc2));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  EXPECT_EQ(out[0], 4.0f);
  EXPECT_EQ(out[n - 1], 4.0f);
  for (M_driver_intel_graph_t graph : graphs)
    EXPECT_SUCCESS(intel_graph_destroy(graph));
  for (M_driver_memory_t m : {da, db, dc1, dc2})
    ASSERT_SUCCESS(memory_free_sync(contexts[0], m));
}

// A graph's output allocated during its capture and freed after it, as MAX
// frees a captured graph's outputs: every launch writes it again, so it
// stays the graph's. An allocation of its size outside a capture gets
// another block, and launches leave that block alone. When the output block
// went back to the cache, the allocation got it and each launch overwrote
// it; once the cache released it, launches faulted the GPU.
TEST_F(GraphTest, KeepsWhatItsCaptureAllocated) {
  constexpr int64_t n = 1 << 14;
  constexpr uint64_t bytes = n * sizeof(float);
  std::vector<float> a(n, 1.0f), b(n, 2.0f), out(n);
  M_driver_memory_t da = nullptr, db = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &da));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &db));
  M_driver_memory_view va{da, 0, bytes}, vb{db, 0, bytes};
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &va, a.data()));
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &vb, b.data()));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));

  ASSERT_SUCCESS(intel_graph_capture_begin(queues[0]));
  M_driver_memory_t output = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &output));
  uint64_t outputAt = addressOf(output);
  ASSERT_SUCCESS(launch(addressOf(da), addressOf(db), outputAt, n));
  M_driver_intel_graph_t graph = nullptr;
  ASSERT_SUCCESS(intel_graph_capture_end(queues[0], &graph));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], output));

  M_driver_memory_t other = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &other));
  EXPECT_NE(addressOf(other), outputAt);
  M_driver_memory_view vo{other, 0, bytes};
  ASSERT_SUCCESS(queue_set_memory(queues[0], &vo, 0x7f));
  for (int k = 0; k < 3; ++k)
    ASSERT_SUCCESS(intel_graph_launch(queues[0], graph));
  ASSERT_SUCCESS(queue_copy_from_device(queues[0], out.data(), &vo));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  std::vector<float> filled(n);
  std::memset(filled.data(), 0x7f, bytes);
  EXPECT_EQ(std::memcmp(out.data(), filled.data(), bytes), 0)
      << "a graph launch wrote a block allocated after its capture";

  EXPECT_SUCCESS(intel_graph_destroy(graph));
  for (M_driver_memory_t m : {other, da, db})
    ASSERT_SUCCESS(memory_free_sync(contexts[0], m));
}

TEST_F(GraphTest, RefusesWhatItCannotDo) {
  EXPECT_EQ(intel_graph_capture_begin(nullptr), M_DRIVER_STATUS_BAD_QUEUE);
  M_driver_intel_graph_t graph = nullptr;
  // Not capturing.
  EXPECT_EQ(intel_graph_capture_end(queues[0], &graph),
            M_DRIVER_STATUS_INVALID_ARG);
  EXPECT_EQ(intel_graph_capture_end(queues[0], nullptr),
            M_DRIVER_STATUS_INVALID_ARG);
  EXPECT_EQ(intel_graph_launch(queues[0], nullptr),
            M_DRIVER_STATUS_INVALID_ARG);
  uint32_t version = 0;
  EXPECT_EQ(intel_graphs_version(nullptr, &version),
            M_DRIVER_STATUS_BAD_DEVICE);
  EXPECT_EQ(intel_graphs_version(devices[0], nullptr),
            M_DRIVER_STATUS_INVALID_ARG);
}

// One graph of 20 small launches against the same 20 launched one by one:
// host time to enqueue them, and to enqueue and synchronize (median of 200).
TEST_F(GraphTest, OneGraphLaunchCostsLessThanItsKernels) {
  constexpr int64_t n = 256;
  constexpr int kKernels = 20;
  M_driver_memory_t da = nullptr, db = nullptr, dc = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], n * 4, &da));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], n * 4, &db));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], n * 4, &dc));
  uint64_t pa = addressOf(da), pb = addressOf(db), pc = addressOf(dc);
  auto now = [] { return std::chrono::steady_clock::now(); };
  auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  for (int i = 0; i < kKernels; ++i)
    ASSERT_SUCCESS(launch(pa, pb, pc, n));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  std::vector<double> eager, graphed, eagerHost, graphedHost;
  auto us = [](auto d) {
    return std::chrono::duration<double, std::micro>(d).count();
  };
  for (int r = 0; r < 200; ++r) {
    auto t0 = now();
    for (int i = 0; i < kKernels; ++i)
      ASSERT_SUCCESS(launch(pa, pb, pc, n));
    auto t1 = now();
    ASSERT_SUCCESS(queue_synchronize(queues[0]));
    eagerHost.push_back(us(t1 - t0));
    eager.push_back(us(now() - t0));
  }
  ASSERT_SUCCESS(intel_graph_capture_begin(queues[0]));
  for (int i = 0; i < kKernels; ++i)
    ASSERT_SUCCESS(launch(pa, pb, pc, n));
  M_driver_intel_graph_t graph = nullptr;
  ASSERT_SUCCESS(intel_graph_capture_end(queues[0], &graph));
  ASSERT_SUCCESS(intel_graph_launch(queues[0], graph));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  for (int r = 0; r < 200; ++r) {
    auto t0 = now();
    ASSERT_SUCCESS(intel_graph_launch(queues[0], graph));
    auto t1 = now();
    ASSERT_SUCCESS(queue_synchronize(queues[0]));
    graphedHost.push_back(us(t1 - t0));
    graphed.push_back(us(now() - t0));
  }
  double e = median(eager), g = median(graphed);
  double eh = median(eagerHost), gh = median(graphedHost);
  std::printf("%d launches: host %.1f us, + sync %.1f us; one graph launch: "
              "host %.1f us, + sync %.1f us\n",
              kKernels, eh, e, gh, g);
  // The host's cost is what a graph saves; the device runs the same kernels.
  EXPECT_LT(gh, eh);
  EXPECT_SUCCESS(intel_graph_destroy(graph));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], da));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], db));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], dc));
}

} // namespace
