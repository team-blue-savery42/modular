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
// Kernel launch through the HAL with a kernel the Mojo compiler built for
// Intel: test/data/*.spv are kernels from
// plugins/intel/tests/codegen/kernels.mojo, compiled by IntelCompilerPlugin.so
// (`compile_info[kernel, emission_kind="object"]`). Modular's conformance
// suite has no launch cases.
//
//===----------------------------------------------------------------------===//

#include "Target/test/Fixture.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

// The entry point Mojo emits: the function's full mangled name.
constexpr const char *kVectorAdd =
    "kernels::vector_add(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType."
    "float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],"
    "::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], "
    "::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer["
    "True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, "
    "MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::SIMD[DType.int, 1])";

std::vector<char> readSpirv(const std::string &name = "vector_add") {
  std::ifstream in("plugins/intel/driver/test/data/" + name + ".spv",
                   std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(in)), {});
}

/// The entry point of a corpus kernel taking one `Int32` pointer.
std::string int32Kernel(const std::string &name) {
  return "kernels::" + name +
         "(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.int32, 1], "
         "::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC])";
}

struct LaunchTest : DriverPluginQueueTest {
  void SetUp() override {
    DriverPluginQueueTest::SetUp();
    if (HasFatalFailure() || IsSkipped())
      return;
    spirv = readSpirv();
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

  /// Launches vector_add(a, b, c, n) with `block`-wide groups.
  M_driver_status_t launch(uint64_t a, uint64_t b, uint64_t c, int64_t n,
                           uint32_t block, M_driver_queue_t queue = nullptr) {
    void *args[] = {&a, &b, &c, &n};
    uint64_t sizes[] = {8, 8, 8, 8};
    M_driver_queue_execute_config_t config = {};
    config.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
    config.config.gpu.grid = {static_cast<uint32_t>((n + block - 1) / block), 1,
                              1};
    config.config.gpu.block = {block, 1, 1};
    return queue_execute(queue != nullptr ? queue : queues[0], function,
                         &config, args, sizes, 4);
  }

  uint64_t addressOf(M_driver_memory_t memory) {
    void *address = nullptr;
    EXPECT_SUCCESS(
        memory_property(memory, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &address));
    return reinterpret_cast<uint64_t>(address);
  }

  std::vector<char> spirv;
  M_driver_runtime_bundle_t runtime = nullptr;
  M_driver_function_t function = nullptr;
};

// A device block freed while a kernel that writes it is still queued comes
// back from the next allocation of its size on a context with one queue,
// and a synchronous copy into it waits for that kernel: the
// copy's bytes, not the kernel's, are what remain.
TEST_F(LaunchTest, PendingFreeIsReusedInQueueOrder) {
  constexpr int64_t n = 1 << 24;
  constexpr uint64_t bytes = n * sizeof(float);
  M_driver_memory_t da = nullptr, db = nullptr, dc = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &da));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &db));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &dc));
  M_driver_memory_view va{da, 0, bytes}, vb{db, 0, bytes};
  ASSERT_SUCCESS(queue_set_memory(queues[0], &va, 0));
  ASSERT_SUCCESS(queue_set_memory(queues[0], &vb, 0));
  void *oldAddress = reinterpret_cast<void *>(addressOf(dc));
  // Several launches writing c, so the free comes while they run.
  for (int i = 0; i < 20; ++i)
    ASSERT_SUCCESS(launch(addressOf(da), addressOf(db), addressOf(dc), n, 256));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], dc));
  M_driver_memory_t again = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &again));
  EXPECT_EQ(reinterpret_cast<void *>(addressOf(again)), oldAddress);
  std::vector<float> ones(n, 1.0f), out(n, -1.0f);
  M_driver_memory_view vagain{again, 0, bytes};
  ASSERT_SUCCESS(copy_to_device_sync(contexts[0], &vagain, ones.data()));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  ASSERT_SUCCESS(copy_from_device_sync(contexts[0], out.data(), &vagain));
  int64_t wrong = 0;
  for (int64_t i = 0; i < n; ++i)
    wrong += out[i] != 1.0f;
  EXPECT_EQ(wrong, 0);
  ASSERT_SUCCESS(memory_free_sync(contexts[0], again));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], da));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], db));
}

TEST_F(LaunchTest, MojoVectorAddMatchesTheCpu) {
  constexpr int64_t n = 1 << 20;
  constexpr uint64_t bytes = n * sizeof(float);
  std::vector<float> a(n), b(n), c(n, -1.0f);
  for (int64_t i = 0; i < n; ++i) {
    a[i] = 0.5f * static_cast<float>(i);
    b[i] = 1.0f - static_cast<float>(i % 1000);
  }
  M_driver_memory_t da = nullptr, db = nullptr, dc = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &da));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &db));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &dc));
  M_driver_memory_view va{da, 0, bytes}, vb{db, 0, bytes}, vc{dc, 0, bytes};
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &va, a.data()));
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &vb, b.data()));
  ASSERT_SUCCESS(launch(addressOf(da), addressOf(db), addressOf(dc), n, 256));
  ASSERT_SUCCESS(queue_copy_from_device(queues[0], c.data(), &vc));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  int64_t wrong = 0;
  for (int64_t i = 0; i < n; ++i)
    wrong += c[i] != a[i] + b[i];
  EXPECT_EQ(wrong, 0);

  // Launch latency: one work-item, then sync; and 1,000 launches back to
  // back, then sync.
  auto now = [] { return std::chrono::steady_clock::now(); };
  uint64_t pa = addressOf(da), pb = addressOf(db), pc = addressOf(dc);
  std::vector<double> single, execute, wait;
  for (int i = 0; i < 1000; ++i) {
    auto start = now();
    ASSERT_SUCCESS(launch(pa, pb, pc, 1, 1));
    auto launched = now();
    ASSERT_SUCCESS(queue_synchronize(queues[0]));
    auto done = now();
    single.push_back(
        std::chrono::duration<double, std::micro>(done - start).count());
    execute.push_back(
        std::chrono::duration<double, std::micro>(launched - start).count());
    wait.push_back(
        std::chrono::duration<double, std::micro>(done - launched).count());
  }
  std::sort(single.begin(), single.end());
  std::sort(execute.begin(), execute.end());
  std::sort(wait.begin(), wait.end());
  std::printf("single launch breakdown: queue_execute %.1f us, "
              "queue_synchronize %.1f us\n",
              execute[execute.size() / 2], wait[wait.size() / 2]);
  std::vector<double> burst;
  for (int r = 0; r < 20; ++r) {
    auto start = now();
    for (int i = 0; i < 1000; ++i)
      ASSERT_SUCCESS(launch(addressOf(da), addressOf(db), addressOf(dc), 1, 1));
    ASSERT_SUCCESS(queue_synchronize(queues[0]));
    burst.push_back(
        std::chrono::duration<double, std::micro>(now() - start).count() /
        1000);
  }
  std::sort(burst.begin(), burst.end());
  std::printf("vector_add launch + sync %.1f us; back to back %.2f us per "
              "launch\n",
              single[single.size() / 2], burst[burst.size() / 2]);

  ASSERT_SUCCESS(memory_free_sync(contexts[0], dc));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], db));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], da));
}

// A zero-byte allocation has no memory handle, and copies and fills of an
// empty view of it do nothing (MAX's sampler copies a batch of zero rows,
// test_topk_gpu); a view past its end is still refused.
TEST_F(LaunchTest, EmptyViewsOfZeroByteAllocationsAreEmpty) {
  M_driver_memory_t empty = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], 0, &empty));
  M_driver_memory_view view{empty, 0, 0};
  int32_t host = 7;
  EXPECT_SUCCESS(queue_copy_to_device(queues[0], &view, &host));
  EXPECT_SUCCESS(queue_copy_from_device(queues[0], &host, &view));
  EXPECT_SUCCESS(queue_copy_intra_device(queues[0], &view, &view));
  EXPECT_SUCCESS(queue_set_memory(queues[0], &view, 0));
  EXPECT_SUCCESS(copy_to_device_sync(contexts[0], &view, &host));
  EXPECT_SUCCESS(queue_synchronize(queues[0]));
  EXPECT_EQ(host, 7);
  M_driver_memory_view past{empty, 0, 4};
  EXPECT_EQ(queue_copy_to_device(queues[0], &past, &host),
            M_DRIVER_STATUS_BAD_MEMORY);
  EXPECT_SUCCESS(memory_free_sync(contexts[0], empty));
}

// MAX's DeviceBuffer frees with memory_free_sync as soon as it is unused,
// often while kernels that write it are still queued; cuMemFree waits for
// them, and zeMemFree alone doesn't (the kernels page-faulted).
TEST_F(LaunchTest, FreeWaitsForTheKernelsUsingTheMemory) {
  constexpr int64_t n = 1 << 22;
  constexpr uint64_t bytes = n * sizeof(float);
  std::vector<float> a(n, 1.0f);
  M_driver_memory_t da = nullptr, dc = nullptr, again = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &da));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &dc));
  M_driver_memory_view va{da, 0, bytes};
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &va, a.data()));
  for (int i = 0; i < 100; ++i)
    ASSERT_SUCCESS(launch(addressOf(da), addressOf(da), addressOf(dc), n, 256));
  EXPECT_SUCCESS(memory_free_sync(contexts[0], dc));
  EXPECT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &again));
  EXPECT_SUCCESS(queue_synchronize(queues[0]));
  EXPECT_SUCCESS(memory_free_sync(contexts[0], again));
  EXPECT_SUCCESS(memory_free_sync(contexts[0], da));
}

// The free doesn't wait for the queue: MAX's graph runtime frees an
// execution's earlier outputs on every execution, and a drain there made each
// one a host round trip. The memory is released once the work queued before
// the free is done; the kernels using it neither fault nor see it reused.
TEST_F(LaunchTest, FreeReturnsBeforeTheQueueIsIdle) {
  constexpr int64_t n = 1 << 22;
  constexpr uint64_t bytes = n * sizeof(float);
  std::vector<float> a(n), c(n, -1.0f);
  for (int64_t i = 0; i < n; ++i)
    a[i] = static_cast<float>(i % 1000);
  M_driver_memory_t da = nullptr, dc = nullptr, dd = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &da));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &dc));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &dd));
  M_driver_memory_view va{da, 0, bytes};
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &va, a.data()));
  // About 0.2 s of work on the B70, reading da and writing dc, then dd.
  for (int i = 0; i < 2000; ++i)
    ASSERT_SUCCESS(launch(addressOf(da), addressOf(da), addressOf(dc), n, 256));
  ASSERT_SUCCESS(launch(addressOf(da), addressOf(da), addressOf(dd), n, 256));
  auto start = std::chrono::steady_clock::now();
  EXPECT_SUCCESS(memory_free_sync(contexts[0], dc));
  EXPECT_SUCCESS(memory_free_sync(contexts[0], da));
  auto took = std::chrono::steady_clock::now() - start;
  EXPECT_LT(took, std::chrono::milliseconds(50))
      << "memory_free_sync waited for the queue";
  // Fresh allocations meanwhile don't get the freed memory's contents mixed
  // into the queued kernels' results.
  M_driver_memory_t fresh = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &fresh));
  M_driver_memory_view vf{fresh, 0, bytes};
  std::vector<float> junk(n, 7.0f);
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &vf, junk.data()));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  M_driver_memory_view vd{dd, 0, bytes};
  ASSERT_SUCCESS(queue_copy_from_device(queues[0], c.data(), &vd));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  int64_t wrong = 0;
  for (int64_t i = 0; i < n; ++i)
    wrong += c[i] != 2.0f * a[i];
  EXPECT_EQ(wrong, 0);
  EXPECT_SUCCESS(memory_free_sync(contexts[0], fresh));
  EXPECT_SUCCESS(memory_free_sync(contexts[0], dd));
}

// Allocation churn with immediate reuse across two queues: each round fills a
// fresh buffer, queues kernels on one queue that read it into the round's own
// output, frees it while they're queued, then allocates the same size again and
// has the other queue overwrite that allocation. A free released before the
// kernels are done would hand them the overwritten memory.
TEST_F(LaunchTest, FreedMemoryIsNotReusedUnderQueuedKernels) {
  constexpr int64_t n = 1 << 18; // 1 MiB: staged, asynchronous copies
  constexpr uint64_t bytes = n * sizeof(float);
  constexpr int rounds = 200;
  M_driver_queue_t other = nullptr;
  ASSERT_SUCCESS(queue_create(contexts[0], &other));
  M_driver_memory_t out = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes * rounds, &out));
  std::vector<float> junk(n, -7.0f);
  std::vector<std::vector<float>> inputs(rounds);
  for (int r = 0; r < rounds; ++r) {
    inputs[r].assign(n, static_cast<float>(r + 1));
    M_driver_memory_t in = nullptr, again = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &in));
    M_driver_memory_view vin{in, 0, bytes};
    ASSERT_SUCCESS(queue_copy_to_device(queues[0], &vin, inputs[r].data()));
    uint64_t dst = addressOf(out) + r * bytes;
    // Enough queued work that the other queue's copies run before it ends.
    for (int k = 0; k < 20; ++k)
      ASSERT_SUCCESS(launch(addressOf(in), addressOf(in), dst, n, 256));
    ASSERT_SUCCESS(memory_free_sync(contexts[0], in));
    ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &again));
    M_driver_memory_view vagain{again, 0, bytes};
    ASSERT_SUCCESS(queue_copy_to_device(other, &vagain, junk.data()));
    ASSERT_SUCCESS(memory_free_sync(contexts[0], again));
  }
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  ASSERT_SUCCESS(queue_synchronize(other));
  std::vector<float> got(n * rounds);
  M_driver_memory_view vout{out, 0, bytes * rounds};
  ASSERT_SUCCESS(queue_copy_from_device(queues[0], got.data(), &vout));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  int64_t wrong = 0;
  for (int r = 0; r < rounds; ++r)
    for (int64_t i = 0; i < n; ++i)
      wrong += got[r * n + i] != 2.0f * static_cast<float>(r + 1);
  EXPECT_EQ(wrong, 0);
  EXPECT_SUCCESS(memory_free_sync(contexts[0], out));
  EXPECT_SUCCESS(queue_destroy(contexts[0], other));
}

// Released blocks are kept and handed to the next allocation of their size
// class: MAX's graph runtime allocates and frees its intermediate
// buffers on every step. Device memory stays counted while cached, and a
// block comes back only after the work queued before its free is done (the
// test above).
TEST_F(LaunchTest, ReleasedMemoryIsReusedBySizeClass) {
  constexpr uint64_t bytes = 3u << 20;
  M_driver_memory_t first = nullptr, second = nullptr, third = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &first));
  uint64_t address = addressOf(first);
  uint64_t freeBefore = 0, total = 0;
  ASSERT_SUCCESS(memory_free_sync(contexts[0], first));
  // Within an eighth of the class: the same block.
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes - 4096, &second));
  EXPECT_EQ(addressOf(second), address);
  // A different class gets other memory.
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes * 2, &third));
  EXPECT_NE(addressOf(third), address);
  ASSERT_SUCCESS(context_memory_info(contexts[0], &freeBefore, &total));
  EXPECT_GT(total, 0u);
  // The reused block holds what the kernels write.
  constexpr int64_t n = (bytes - 4096) / sizeof(float);
  std::vector<float> a(n, 3.0f), c(n, 0.0f);
  M_driver_memory_view v{second, 0, bytes - 4096};
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &v, a.data()));
  ASSERT_SUCCESS(launch(address, address, address, n, 256));
  ASSERT_SUCCESS(queue_copy_from_device(queues[0], c.data(), &v));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  EXPECT_EQ(c[0], 6.0f);
  EXPECT_EQ(c[n - 1], 6.0f);
  EXPECT_SUCCESS(memory_free_sync(contexts[0], second));
  EXPECT_SUCCESS(memory_free_sync(contexts[0], third));
  // Pinned host memory is reused the same way.
  M_driver_memory_t pinned = nullptr, again = nullptr;
  ASSERT_SUCCESS(memory_alloc_pinned(contexts[0], 1000, &pinned));
  void *host = nullptr;
  ASSERT_SUCCESS(
      memory_property(pinned, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &host));
  ASSERT_SUCCESS(memory_free_pinned(contexts[0], pinned));
  ASSERT_SUCCESS(memory_alloc_pinned(contexts[0], 1024, &again));
  void *hostAgain = nullptr;
  ASSERT_SUCCESS(
      memory_property(again, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &hostAgain));
  EXPECT_EQ(hostAgain, host);
  EXPECT_SUCCESS(memory_free_pinned(contexts[0], again));
}

// Blocks the allocation cache keeps are free to the context's callers (an
// allocation that doesn't fit flushes the cache), so memory_info reports
// them as free: MAX sizes its KV cache from free memory, after
// graph runs have filled the cache.
TEST_F(LaunchTest, CachedBlocksCountAsFree) {
  // An odd size, so no block cached by an earlier test serves it.
  constexpr uint64_t bytes = (300u << 20) + 12345;
  uint64_t freeHeld = 0, freeCached = 0, total = 0;
  M_driver_memory_t block = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &block));
  // Touched, so the device memory is committed.
  M_driver_memory_view all{block, 0, bytes};
  ASSERT_SUCCESS(queue_set_memory(queues[0], &all, 1));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  ASSERT_SUCCESS(context_memory_info(contexts[0], &freeHeld, &total));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], block));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  ASSERT_SUCCESS(context_memory_info(contexts[0], &freeCached, &total));
  // The held block shows (Sysman alone didn't, in-process).
  EXPECT_LE(freeHeld + bytes - (bytes >> 4), total)
      << "memory_info shows the held block";
  EXPECT_GE(freeCached, freeHeld + bytes - (bytes >> 4))
      << "the cached block counts as free";
}

// A block freed while a kernel that uses it is queued is reused once that
// kernel has finished, even when nothing retired the free since: the cache
// miss checks the queues' free marks. Before, the next
// allocation of the class took a fresh block until a later free or
// synchronize retired it (MAX's graph runtime allocates a step's buffers
// before it frees any). The kernel's completion is awaited through an event,
// which retires nothing.
TEST_F(LaunchTest, FinishedFreesAreReusedWithoutASynchronize) {
  constexpr int64_t n = 1 << 20;
  M_driver_memory_t first = nullptr, second = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], n * sizeof(float), &first));
  uint64_t address = addressOf(first);
  ASSERT_SUCCESS(launch(address, address, address, n, 256));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], first));
  M_driver_event_t done = nullptr;
  ASSERT_SUCCESS(
      event_create(contexts[0], M_DRIVER_EVENT_FLAG_CPU_VISIBLE, &done));
  ASSERT_SUCCESS(queue_record_event(queues[0], done));
  ASSERT_SUCCESS(event_synchronize(contexts[0], done));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], n * sizeof(float), &second));
  EXPECT_EQ(addressOf(second), address);
  EXPECT_SUCCESS(memory_free_sync(contexts[0], second));
  EXPECT_SUCCESS(event_destroy(contexts[0], done));
}

// A free is retired once the work queued before it completes, even while
// later work (with a later free's mark behind it) still runs: the queue's
// free marks are a ring, queried oldest first. With one mark
// event, only the newest mark could be queried, and it completes only when
// the device catches up with the host. A short kernel then a long one
// (~0.5 ms over 192 MiB), each followed by a free; once the short one is
// done, its block is reused while the long one runs.
TEST_F(LaunchTest, FreesAreRetiredWhileLaterWorkRuns) {
  constexpr int64_t small = 1 << 16, large = 1 << 24;
  M_driver_memory_t first = nullptr, busy = nullptr, again = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], small * sizeof(float), &first));
  ASSERT_SUCCESS(
      memory_alloc_sync(contexts[0], 3 * large * sizeof(float), &busy));
  uint64_t address = addressOf(first), big = addressOf(busy);
  ASSERT_SUCCESS(launch(address, address, address, small, 256));
  M_driver_event_t done = nullptr;
  ASSERT_SUCCESS(
      event_create(contexts[0], M_DRIVER_EVENT_FLAG_CPU_VISIBLE, &done));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], first));
  ASSERT_SUCCESS(queue_record_event(queues[0], done));
  ASSERT_SUCCESS(launch(big, big + large * sizeof(float),
                        big + 2 * large * sizeof(float), large, 256));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], busy));
  ASSERT_SUCCESS(event_synchronize(contexts[0], done));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], small * sizeof(float), &again));
  EXPECT_EQ(addressOf(again), address);
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  EXPECT_SUCCESS(memory_free_sync(contexts[0], again));
  EXPECT_SUCCESS(event_destroy(contexts[0], done));
}

// A synchronous free appends to the context's queues without holding the
// queue list's lock, so other threads create and destroy queues meanwhile; a
// queue destroyed with the free still pending on it completes the free's
// entry there, and the memory is released once every queue is done.
TEST_F(LaunchTest, QueuesComeAndGoWhileAFreeIsPending) {
  constexpr int64_t n = 1 << 22;
  constexpr uint64_t bytes = n * sizeof(float);
  M_driver_memory_t da = nullptr, dc = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &da));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &dc));
  M_driver_queue_t listed = nullptr;
  ASSERT_SUCCESS(queue_create(contexts[0], &listed));
  // About 0.2 s of work on the B70.
  for (int i = 0; i < 2000; ++i)
    ASSERT_SUCCESS(launch(addressOf(da), addressOf(da), addressOf(dc), n, 256));
  auto freed = std::make_shared<std::promise<M_driver_status_t>>();
  std::future<M_driver_status_t> done = freed->get_future();
  M_driver_context_t context = contexts[0];
  std::thread([this, context, dc, freed] {
    freed->set_value(memory_free_sync(context, dc));
  }).detach();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  M_driver_queue_t fresh = nullptr;
  EXPECT_SUCCESS(queue_create(contexts[0], &fresh));
  EXPECT_SUCCESS(queue_destroy(contexts[0], fresh));
  // The free returned without waiting for the 0.2 s of work.
  EXPECT_EQ(done.wait_for(std::chrono::seconds(0)), std::future_status::ready)
      << "memory_free_sync waited for the queues";
  EXPECT_SUCCESS(queue_destroy(contexts[0], listed));
  EXPECT_SUCCESS(queue_synchronize(queues[0]));
  ASSERT_EQ(done.wait_for(std::chrono::seconds(60)), std::future_status::ready);
  EXPECT_SUCCESS(done.get());
  EXPECT_SUCCESS(memory_free_sync(contexts[0], da));
}

// MAX's client may unload a bundle before the functions loaded from it
// (Mojo destroys the bundle once it's unused); they keep working until
// they're unloaded, which Level Zero alone doesn't allow.
TEST_F(LaunchTest, FunctionsOutliveTheirBundle) {
  ASSERT_SUCCESS(bundle_unload(contexts[0], runtime));
  runtime = nullptr;
  constexpr int64_t n = 4096;
  constexpr uint64_t bytes = n * sizeof(float);
  std::vector<float> a(n), b(n), c(n, -1.0f);
  for (int64_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(3 * i);
  }
  M_driver_memory_t da = nullptr, db = nullptr, dc = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &da));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &db));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &dc));
  M_driver_memory_view va{da, 0, bytes}, vb{db, 0, bytes}, vc{dc, 0, bytes};
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &va, a.data()));
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &vb, b.data()));
  ASSERT_SUCCESS(launch(addressOf(da), addressOf(db), addressOf(dc), n, 256));
  ASSERT_SUCCESS(queue_copy_from_device(queues[0], c.data(), &vc));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  for (int64_t i = 0; i < n; ++i)
    ASSERT_EQ(c[i], a[i] + b[i]) << i;
  ASSERT_SUCCESS(memory_free_sync(contexts[0], dc));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], db));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], da));
  // TearDown unloads the function, and with it the module.
}

TEST_F(LaunchTest, OccupancyAndFunctionAttributes) {
  int32_t blocks = 0;
  ASSERT_SUCCESS(function_occupancy_max_active_blocks(contexts[0], function,
                                                      256, 0, &blocks));
  std::printf("vector_add, 256-wide groups: %d per Xe-core\n", blocks);
  EXPECT_GT(blocks, 0);
  // More shared memory per group than an Xe-core has: none fit.
  ASSERT_SUCCESS(function_occupancy_max_active_blocks(
      contexts[0], function, 256, 1ull << 30, &blocks));
  EXPECT_EQ(blocks, 0);
  // The dynamic shared-memory cap (CUDA attribute 8) within the device limit.
  EXPECT_SUCCESS(function_set_attribute(contexts[0], function, 8, 64 * 1024));
  EXPECT_EQ(function_set_attribute(contexts[0], function, 8, 1 << 30),
            M_DRIVER_STATUS_UNSUPPORTED);
}

TEST_F(LaunchTest, FunctionFromAnotherContextIsRejected) {
  M_driver_device_t device = nullptr;
  ASSERT_SUCCESS(device_get(driver, 0, &device));
  M_driver_context_t other = nullptr;
  ASSERT_SUCCESS(context_create(device, &other));
  M_driver_queue_t otherQueue = nullptr;
  ASSERT_SUCCESS(queue_create(other, &otherQueue));
  int64_t n = 0;
  uint64_t null = 0;
  void *args[] = {&null, &null, &null, &n};
  uint64_t sizes[] = {8, 8, 8, 8};
  M_driver_queue_execute_config_t config = {};
  config.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
  config.config.gpu.grid = {1, 1, 1};
  config.config.gpu.block = {1, 1, 1};
  EXPECT_EQ(queue_execute(otherQueue, function, &config, args, sizes, 4),
            M_DRIVER_STATUS_BAD_FUNCTION);
  ASSERT_SUCCESS(queue_destroy(other, otherQueue));
  ASSERT_SUCCESS(context_destroy(other));
}

TEST_F(LaunchTest, LaunchAttributesAreUnsupported) {
  // CUDA launch attributes (MAX passes programmatic dependent launch's) have
  // no Level Zero counterpart: the device can't do them, which isn't a
  // missing implementation.
  int64_t n = 0;
  uint64_t null = 0;
  void *args[] = {&null, &null, &null, &n};
  uint64_t sizes[] = {8, 8, 8, 8};
  uint64_t attribute[9] = {};
  M_driver_queue_execute_config_t config = {};
  config.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
  config.config.gpu.grid = {1, 1, 1};
  config.config.gpu.block = {1, 1, 1};
  config.config.gpu.attributes = attribute;
  config.config.gpu.num_attributes = 1;
  EXPECT_EQ(queue_execute(queues[0], function, &config, args, sizes, 4),
            M_DRIVER_STATUS_UNSUPPORTED);
}

TEST_F(LaunchTest, UnknownFunctionAndWrongFileType) {
  M_driver_function_t missing = nullptr;
  EXPECT_EQ(function_load(contexts[0], runtime, "no_such_kernel", 14, &missing),
            M_DRIVER_STATUS_UNKNOWN_SYMBOL);
  M_driver_static_bundle bundle = {};
  bundle.mapped_data = {spirv.data(), spirv.size()};
  bundle.file_type = "cubin";
  bundle.file_type_len = 5;
  M_driver_runtime_bundle_t other = nullptr;
  EXPECT_EQ(bundle_load(contexts[0], &bundle, nullptr, &other),
            M_DRIVER_STATUS_INCOMPATIBLE);
}

// A kernel calling a function nothing links: loading fails and names it,
// rather than zeKernelCreate failing later without names. (A kernel holding
// a List was this case until the device heap.)
TEST_F(LaunchTest, UnresolvedImportsAreNamed) {
  std::vector<char> spirv = readSpirv("unresolved_import");
  ASSERT_FALSE(spirv.empty());
  M_driver_static_bundle bundle = {};
  bundle.mapped_data = {spirv.data(), spirv.size()};
  bundle.file_type = "object";
  bundle.file_type_len = 6;
  M_driver_runtime_bundle_t runtime = nullptr;
  M_driver_status_t status =
      bundle_load(contexts[0], &bundle, nullptr, &runtime);
  ASSERT_NE(status, M_DRIVER_STATUS_SUCCESS);
  std::string message = getStatusMessage(status);
  EXPECT_NE(message.find("mojo_intel_undefined"), std::string::npos) << message;
  EXPECT_EQ(message.find("__spirv_"), std::string::npos) << message;
}

// A module IGC rejects reports why within the 1 KiB MAX's client reads,
// though the kernel's name alone runs past it: an older Mojo reduction kernel
// taking an array argument, which the compiler plugin no longer emits.
TEST_F(LaunchTest, BuildFailureNamesItsReason) {
  std::vector<char> rejected = readSpirv("rejected_array_argument");
  ASSERT_FALSE(rejected.empty());
  M_driver_static_bundle bundle = {};
  bundle.mapped_data = {rejected.data(), rejected.size()};
  bundle.file_type = "object";
  bundle.file_type_len = 6;
  M_driver_runtime_bundle_t other = nullptr;
  M_driver_status_t status = bundle_load(contexts[0], &bundle, nullptr, &other);
  ASSERT_NE(status, M_DRIVER_STATUS_SUCCESS);
  std::string message = getStatusMessage(status);
  EXPECT_NE(message.find("Array type is not allowed as a kernel argument"),
            std::string::npos)
      << message;
}

// Shared memory: `shared_reduce` sums each 64-wide block in static shared
// memory; `reverse_shared` reverses each block through dynamic shared memory,
// which the compiler passes as a trailing local-memory argument the driver
// sizes from `shared_mem_bytes`.
struct SharedMemoryTest : DriverPluginQueueTest {
  void TearDown() override {
    for (M_driver_function_t function : functions)
      EXPECT_SUCCESS(function_unload(contexts[0], function));
    for (M_driver_runtime_bundle_t bundle : bundles)
      EXPECT_SUCCESS(bundle_unload(contexts[0], bundle));
    DriverPluginQueueTest::TearDown();
  }

  M_driver_function_t load(const std::string &name,
                           std::string entry = std::string()) {
    spirvs.push_back(readSpirv(name));
    std::vector<char> &spirv = spirvs.back();
    EXPECT_FALSE(spirv.empty()) << "missing test/data/" << name << ".spv";
    M_driver_static_bundle bundle = {};
    bundle.mapped_data = {spirv.data(), spirv.size()};
    bundle.file_type = "object";
    bundle.file_type_len = 6;
    M_driver_runtime_bundle_t runtime = nullptr;
    EXPECT_SUCCESS(bundle_load(contexts[0], &bundle, nullptr, &runtime));
    bundles.push_back(runtime);
    if (entry.empty())
      entry = int32Kernel(name);
    M_driver_function_t function = nullptr;
    EXPECT_SUCCESS(function_load(contexts[0], runtime, entry.c_str(),
                                 entry.size(), &function));
    functions.push_back(function);
    return function;
  }

  /// Launches `function(p)` on `blocks` blocks of 64.
  M_driver_status_t launch(M_driver_function_t function, uint64_t p,
                           uint32_t blocks, uint32_t sharedBytes,
                           uint32_t numArgs = 1) {
    void *args[] = {&p};
    uint64_t sizes[] = {8};
    M_driver_queue_execute_config_t config = {};
    config.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
    config.config.gpu.grid = {blocks, 1, 1};
    config.config.gpu.block = {64, 1, 1};
    config.config.gpu.shared_mem_bytes = sharedBytes;
    return queue_execute(queues[0], function, &config, args, sizes, numArgs);
  }

  /// Runs `function` over 0, 1, ..., 64 * blocks - 1 and returns the result.
  std::vector<int32_t> run(M_driver_function_t function, uint32_t blocks,
                           uint32_t sharedBytes) {
    std::vector<int32_t> data(64 * blocks);
    for (size_t i = 0; i < data.size(); ++i)
      data[i] = static_cast<int32_t>(i);
    return runOn(function, std::move(data), sharedBytes);
  }

  /// Runs `function` over `data` (64 values a block) and returns the result.
  std::vector<int32_t> runOn(M_driver_function_t function,
                             std::vector<int32_t> data, uint32_t sharedBytes) {
    uint32_t blocks = static_cast<uint32_t>(data.size() / 64);
    uint64_t bytes = data.size() * sizeof(int32_t);
    M_driver_memory_t memory = nullptr;
    EXPECT_SUCCESS(memory_alloc_sync(contexts[0], bytes, &memory));
    M_driver_memory_view view{memory, 0, bytes};
    void *address = nullptr;
    EXPECT_SUCCESS(
        memory_property(memory, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &address));
    EXPECT_SUCCESS(queue_copy_to_device(queues[0], &view, data.data()));
    EXPECT_SUCCESS(launch(function, reinterpret_cast<uint64_t>(address), blocks,
                          sharedBytes));
    EXPECT_SUCCESS(queue_copy_from_device(queues[0], data.data(), &view));
    EXPECT_SUCCESS(queue_synchronize(queues[0]));
    EXPECT_SUCCESS(memory_free_sync(contexts[0], memory));
    return data;
  }

  // Launches `name(p, s)` where s is the 16 bytes { i32 1234 @0, i64 56789
  // @8 } and the kernel writes p[0] = s.a and p[1] = s.b; returns p[0], p[1].
  std::pair<int32_t, int32_t> launchWithStruct(const std::string &name,
                                               const std::string &type) {
    M_driver_function_t kernel = load(
        name,
        "kernels::" + name +
            "(::Pointer[True, MutUnsafeAnyOrigin, "
            "::SIMD[DType.int32, 1], ::Origin[True, MutUnsafeAnyOrigin](), "
            "AddressSpace.GENERIC],kernels::" +
            type + ")");
    EXPECT_NE(kernel, nullptr);
    if (kernel == nullptr)
      return {0, 0};
    M_driver_context_t context = contexts[0];
    M_driver_queue_t queue = queues[0];
    M_driver_memory_t memory = nullptr;
    EXPECT_SUCCESS(memory_alloc_sync(context, 2 * sizeof(int32_t), &memory));
    void *address = nullptr;
    EXPECT_SUCCESS(
        memory_property(memory, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &address));
    uint64_t p = reinterpret_cast<uint64_t>(address);
    int32_t out[2] = {0, 0};
    M_driver_memory_view view{memory, 0, sizeof(out)};
    EXPECT_SUCCESS(queue_copy_to_device(queue, &view, out));
    unsigned char bytes[16] = {};
    int32_t a = 1234;
    int64_t b = 56789;
    std::memcpy(bytes, &a, sizeof(a));
    std::memcpy(bytes + 8, &b, sizeof(b));
    void *args[] = {&p, bytes};
    uint64_t sizes[] = {8, 16};
    M_driver_queue_execute_config_t config = {};
    config.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
    config.config.gpu.grid = {1, 1, 1};
    config.config.gpu.block = {1, 1, 1};
    EXPECT_SUCCESS(queue_execute(queue, kernel, &config, args, sizes, 2));
    EXPECT_SUCCESS(queue_copy_from_device(queue, out, &view));
    EXPECT_SUCCESS(queue_synchronize(queue));
    EXPECT_SUCCESS(memory_free_sync(context, memory));
    return {out[0], out[1]};
  }

  std::vector<std::vector<char>> spirvs;
  std::vector<M_driver_runtime_bundle_t> bundles;
  std::vector<M_driver_function_t> functions;
};

TEST_F(SharedMemoryTest, StaticSharedMemoryReduces) {
  M_driver_function_t reduce = load("shared_reduce");
  ASSERT_NE(reduce, nullptr);
  constexpr uint32_t blocks = 8;
  std::vector<int32_t> result = run(reduce, blocks, 0);
  for (uint32_t b = 0; b < blocks; ++b) {
    // Sum of 64b, ..., 64b + 63.
    int32_t expected = 64 * 64 * static_cast<int32_t>(b) + 63 * 64 / 2;
    EXPECT_EQ(result[b], expected) << "block " << b;
  }
}

TEST_F(SharedMemoryTest, DynamicSharedMemoryIsSizedAtLaunch) {
  M_driver_function_t reverse = load("reverse_shared");
  ASSERT_NE(reverse, nullptr);
  constexpr uint32_t blocks = 8;
  std::vector<int32_t> result = run(reverse, blocks, 64 * sizeof(int32_t));
  int64_t wrong = 0;
  for (uint32_t i = 0; i < 64 * blocks; ++i)
    wrong += result[i] != static_cast<int32_t>((i / 64) * 64 + 63 - i % 64);
  EXPECT_EQ(wrong, 0);
}

TEST_F(SharedMemoryTest, ArgumentCountMustMatchTheKernel) {
  M_driver_function_t reverse = load("reverse_shared");
  M_driver_function_t reduce = load("shared_reduce");
  ASSERT_NE(reverse, nullptr);
  ASSERT_NE(reduce, nullptr);
  // Neither kernel takes zero arguments, with or without dynamic shared
  // memory to fill the last one.
  EXPECT_EQ(launch(reverse, 0, 1, 256, /*numArgs=*/0),
            M_DRIVER_STATUS_INVALID_ARG);
  EXPECT_EQ(launch(reduce, 0, 1, 0, /*numArgs=*/0),
            M_DRIVER_STATUS_INVALID_ARG);
}

// abort(): IGC can't trap, so the aborting work-item raises the context's
// abort flag and the next synchronize fails, then keeps failing.
TEST_F(SharedMemoryTest, AbortFailsTheNextSynchronize) {
  M_driver_function_t aborts = load("aborts");
  ASSERT_NE(aborts, nullptr);
  M_driver_memory_t memory = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], 64 * sizeof(int32_t), &memory));
  M_driver_memory_view view{memory, 0, 64 * sizeof(int32_t)};
  void *address = nullptr;
  ASSERT_SUCCESS(
      memory_property(memory, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &address));
  uint64_t p = reinterpret_cast<uint64_t>(address);

  // Without p[0] == 7 nothing aborts, and the kernel writes 1.
  std::vector<int32_t> data(64, 0);
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &view, data.data()));
  ASSERT_SUCCESS(launch(aborts, p, 1, 0));
  ASSERT_SUCCESS(queue_copy_from_device(queues[0], data.data(), &view));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  EXPECT_EQ(data[0], 1);

  data[0] = 7;
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &view, data.data()));
  ASSERT_SUCCESS(launch(aborts, p, 1, 0));
  EXPECT_NE(queue_synchronize(queues[0]), M_DRIVER_STATUS_SUCCESS);
  EXPECT_NE(queue_synchronize(queues[0]), M_DRIVER_STATUS_SUCCESS);
  EXPECT_SUCCESS(memory_free_sync(contexts[0], memory));
}

// The flag is reached through a module global, not an argument, so nothing
// else makes it resident for the launch. Here no host memory is touched (a
// fill, not a staged copy), which is how a Mojo program through the HAL
// client runs: the store faulted and the abort went unreported.
TEST_F(SharedMemoryTest, AbortIsReportedWithoutOtherHostMemory) {
  M_driver_function_t aborts = load("aborts");
  ASSERT_NE(aborts, nullptr);
  M_driver_memory_t memory = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], 64 * sizeof(int32_t), &memory));
  M_driver_memory_view view{memory, 0, 64 * sizeof(int32_t)};
  void *address = nullptr;
  ASSERT_SUCCESS(
      memory_property(memory, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &address));
  ASSERT_SUCCESS(queue_fill(queues[0], &view, 7, 4));
  ASSERT_SUCCESS(launch(aborts, reinterpret_cast<uint64_t>(address), 1, 0));
  EXPECT_NE(queue_synchronize(queues[0]), M_DRIVER_STATUS_SUCCESS);
  EXPECT_SUCCESS(memory_free_sync(contexts[0], memory));
}

TEST_F(SharedMemoryTest, StructPassedByValue) {
  auto [a, b] = launchWithStruct("plain_struct", "Plain");
  EXPECT_EQ(a, 1234);
  EXPECT_EQ(b, 56789);
}

// A struct holding a zero-length array, { i32 a @0, [0 x i64] @8, i64 b @8 }:
// the compiler plugin passes it as its 16 bytes, and the kernel must read a
// and b at their offsets.
TEST_F(SharedMemoryTest, StructWithAnEmptyTupleKeepsItsLayout) {
  auto [a, b] = launchWithStruct("zero_tuple", "WithEmpty");
  EXPECT_EQ(a, 1234);
  EXPECT_EQ(b, 56789);
}

// `boxed_pointer(p, s)` writes s.v through the address in s = { i64 address
// @0, i32 v @8 } and s.v + 1 to p. The driver can't see the first allocation
// in the kernel's arguments (MAX passes a closure's captured buffers this
// way), and nothing else has used it: the kernel must still reach it.
TEST_F(SharedMemoryTest, KernelsReachAllocationsHeldInStructs) {
  M_driver_function_t kernel =
      load("boxed_pointer",
           "kernels::boxed_pointer(::Pointer[True, MutUnsafeAnyOrigin, "
           "::SIMD[DType.int32, 1], ::Origin[True, MutUnsafeAnyOrigin](), "
           "AddressSpace.GENERIC],kernels::Boxed)");
  ASSERT_NE(kernel, nullptr);
  M_driver_context_t context = contexts[0];
  M_driver_queue_t queue = queues[0];
  M_driver_memory_t direct = nullptr, indirect = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(context, sizeof(int32_t), &direct));
  ASSERT_SUCCESS(memory_alloc_sync(context, sizeof(int32_t), &indirect));
  void *directAddress = nullptr, *indirectAddress = nullptr;
  ASSERT_SUCCESS(memory_property(direct, M_DRIVER_MEMORY_PROPERTY_ADDRESS,
                                 &directAddress));
  ASSERT_SUCCESS(memory_property(indirect, M_DRIVER_MEMORY_PROPERTY_ADDRESS,
                                 &indirectAddress));
  uint64_t p = reinterpret_cast<uint64_t>(directAddress);
  unsigned char bytes[16] = {};
  uint64_t address = reinterpret_cast<uint64_t>(indirectAddress);
  int32_t v = 41;
  std::memcpy(bytes, &address, sizeof(address));
  std::memcpy(bytes + 8, &v, sizeof(v));
  void *args[] = {&p, bytes};
  uint64_t sizes[] = {8, 16};
  M_driver_queue_execute_config_t config = {};
  config.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
  config.config.gpu.grid = {1, 1, 1};
  config.config.gpu.block = {1, 1, 1};
  ASSERT_SUCCESS(queue_execute(queue, kernel, &config, args, sizes, 2));
  int32_t viaStruct = 0, viaArgument = 0;
  M_driver_memory_view indirectView{indirect, 0, sizeof(int32_t)};
  M_driver_memory_view directView{direct, 0, sizeof(int32_t)};
  EXPECT_SUCCESS(queue_copy_from_device(queue, &viaStruct, &indirectView));
  EXPECT_SUCCESS(queue_copy_from_device(queue, &viaArgument, &directView));
  EXPECT_SUCCESS(queue_synchronize(queue));
  EXPECT_EQ(viaStruct, 41);
  EXPECT_EQ(viaArgument, 42);
  EXPECT_SUCCESS(memory_free_sync(context, direct));
  EXPECT_SUCCESS(memory_free_sync(context, indirect));
}

// IGC links OpenCL built-ins called by their mangled names, so a module that
// imports one isn't refused as unresolved: `opencl_import(p)` sets p[0] to
// _Z3expf(p[1]) (plugins/intel/tests/codegen/opencl_import.spvasm).
TEST_F(SharedMemoryTest, MangledBuiltinImportsLoad) {
  M_driver_function_t kernel = load("opencl_import", "opencl_import");
  ASSERT_NE(kernel, nullptr);
  float data[2] = {0.0f, 1.0f};
  M_driver_memory_t memory = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], sizeof(data), &memory));
  M_driver_memory_view view{memory, 0, sizeof(data)};
  void *address = nullptr;
  ASSERT_SUCCESS(
      memory_property(memory, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &address));
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &view, data));
  ASSERT_SUCCESS(launch(kernel, reinterpret_cast<uint64_t>(address), 1, 0));
  ASSERT_SUCCESS(queue_copy_from_device(queues[0], data, &view));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  EXPECT_NEAR(data[0], 2.7182817f, 1e-6f);
  EXPECT_SUCCESS(memory_free_sync(contexts[0], memory));
}

// Constant memory the host fills: `constant_import(p)` sets p[i] to
// constant_table[i], a named constant-space array the compiler plugin
// exports. Its address and size come from the function's "global_address:"
// and "global_size:" properties, and memory_wrap and a copy fill it, as
// cuModuleGetGlobal and a copy do on CUDA; each launch sees the last copy.
TEST_F(SharedMemoryTest, ConstantMemoryIsFilledThroughItsGlobal) {
  M_driver_function_t kernel = load("constant_import");
  ASSERT_NE(kernel, nullptr);
  M_driver_context_t context = contexts[0];
  uint64_t address = 0, size = 0;
  ASSERT_SUCCESS(function_property(context, kernel,
                                   "global_address:constant_table", &address));
  ASSERT_SUCCESS(
      function_property(context, kernel, "global_size:constant_table", &size));
  EXPECT_NE(address, 0u);
  ASSERT_EQ(size, 16 * sizeof(int32_t));
  EXPECT_EQ(
      function_property(context, kernel, "global_address:missing", &address),
      M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME);
  M_driver_memory_t table = nullptr, out = nullptr;
  ASSERT_SUCCESS(memory_wrap(context, address, size, /*owning=*/false, &table));
  ASSERT_SUCCESS(memory_alloc_sync(context, size, &out));
  void *outAddress = nullptr;
  ASSERT_SUCCESS(
      memory_property(out, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &outAddress));
  M_driver_memory_view tableView{table, 0, size}, outView{out, 0, size};
  for (int32_t round = 0; round < 2; ++round) {
    int32_t values[16], results[16] = {};
    for (int32_t i = 0; i < 16; ++i)
      values[i] = 100 * round + i * i;
    ASSERT_SUCCESS(queue_copy_to_device(queues[0], &tableView, values));
    uint64_t p = reinterpret_cast<uint64_t>(outAddress);
    void *args[] = {&p};
    uint64_t sizes[] = {8};
    M_driver_queue_execute_config_t config = {};
    config.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
    config.config.gpu.grid = {1, 1, 1};
    config.config.gpu.block = {16, 1, 1};
    ASSERT_SUCCESS(queue_execute(queues[0], kernel, &config, args, sizes, 1));
    ASSERT_SUCCESS(queue_copy_from_device(queues[0], results, &outView));
    ASSERT_SUCCESS(queue_synchronize(queues[0]));
    for (int i = 0; i < 16; ++i)
      EXPECT_EQ(results[i], values[i]) << "round " << round << " index " << i;
  }
  EXPECT_SUCCESS(memory_free_sync(context, table));
  EXPECT_SUCCESS(memory_free_sync(context, out));
}

// "global_address:" says UNKNOWN_PROPERTY_NAME only for a name the module
// doesn't export. Level Zero returns the same result for a missing global as
// for a failed lookup, so a host that skips a missing global would skip one a
// lost device hid. `exported_function` exports a function, not a global
// (plugins/intel/tests/codegen/exported_function.spvasm): Level Zero has no
// global by that name, and the plugin reports that as an error.
TEST_F(SharedMemoryTest, GlobalLookupFailuresAreReported) {
  M_driver_function_t kernel = load("exported_function", "exported_function");
  ASSERT_NE(kernel, nullptr);
  M_driver_context_t context = contexts[0];
  uint64_t value = 0;
  EXPECT_EQ(
      function_property(context, kernel, "global_address:missing", &value),
      M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME);
  for (const char *property : {"global_address:mojo_intel_exported",
                               "global_size:mojo_intel_exported"}) {
    M_driver_status_t status =
        function_property(context, kernel, property, &value);
    EXPECT_NE(status, M_DRIVER_STATUS_SUCCESS) << property;
    EXPECT_NE(status, M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME) << property;
  }
  // The kernel still runs: p[0] = p[1] + p[1].
  float data[2] = {0.0f, 1.5f};
  M_driver_memory_t memory = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(context, sizeof(data), &memory));
  M_driver_memory_view view{memory, 0, sizeof(data)};
  void *address = nullptr;
  ASSERT_SUCCESS(
      memory_property(memory, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &address));
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &view, data));
  ASSERT_SUCCESS(launch(kernel, reinterpret_cast<uint64_t>(address), 1, 0));
  ASSERT_SUCCESS(queue_copy_from_device(queues[0], data, &view));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  EXPECT_EQ(data[0], 3.0f);
  EXPECT_SUCCESS(memory_free_sync(context, memory));
}

// memory_wrap takes memory another library allocated in the plugin's context
// (which context_get_current_driver_context hands out); a kernel that reaches
// it through a struct, as in KernelsReachAllocationsHeldInStructs, must
// still find it mapped. Here the library is Level Zero itself.
TEST_F(SharedMemoryTest, KernelsReachWrappedMemory) {
  M_driver_function_t kernel =
      load("boxed_pointer",
           "kernels::boxed_pointer(::Pointer[True, MutUnsafeAnyOrigin, "
           "::SIMD[DType.int32, 1], ::Origin[True, MutUnsafeAnyOrigin](), "
           "AddressSpace.GENERIC],kernels::Boxed)");
  ASSERT_NE(kernel, nullptr);
  void *loader = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(loader, nullptr);
  using AllocDevice =
      int (*)(void *, const void *, size_t, size_t, void *, void **);
  using AllocProperties = int (*)(void *, const void *, void *, void **);
  using Free = int (*)(void *, void *);
  auto allocDevice =
      reinterpret_cast<AllocDevice>(dlsym(loader, "zeMemAllocDevice"));
  auto allocProperties = reinterpret_cast<AllocProperties>(
      dlsym(loader, "zeMemGetAllocProperties"));
  auto memFree = reinterpret_cast<Free>(dlsym(loader, "zeMemFree"));
  ASSERT_NE(allocDevice, nullptr);
  ASSERT_NE(allocProperties, nullptr);
  ASSERT_NE(memFree, nullptr);
  M_driver_context_t context = contexts[0];
  ASSERT_SUCCESS(context_set_current(context));
  M_driver_native_handle_t native = 0;
  ASSERT_SUCCESS(context_get_current_driver_context(context, &native));
  void *zeContext = reinterpret_cast<void *>(native);
  M_driver_memory_t direct = nullptr, wrapped = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(context, sizeof(int32_t), &direct));
  void *directAddress = nullptr;
  ASSERT_SUCCESS(memory_property(direct, M_DRIVER_MEMORY_PROPERTY_ADDRESS,
                                 &directAddress));
  // The device, from ze_memory_allocation_properties_t (stype 0x17).
  struct {
    uint32_t stype = 0x17;
    void *next = nullptr;
    unsigned char rest[64] = {};
  } properties;
  void *device = nullptr;
  ASSERT_EQ(allocProperties(zeContext, directAddress, &properties, &device), 0);
  // ze_device_mem_alloc_desc_t: stype 0x15, pNext, flags, ordinal.
  struct {
    uint32_t stype = 0x15;
    const void *next = nullptr;
    uint32_t flags = 0;
    uint32_t ordinal = 0;
  } desc;
  // Large enough not to come from compute-runtime's pool of small
  // allocations, whose backing memory the plugin's own allocations share.
  constexpr size_t kExternalBytes = 8u << 20;
  void *external = nullptr;
  ASSERT_EQ(
      allocDevice(zeContext, &desc, kExternalBytes, 4096, device, &external),
      0);
  ASSERT_SUCCESS(memory_wrap(context, reinterpret_cast<uint64_t>(external),
                             kExternalBytes, /*owning=*/false, &wrapped));
  uint64_t p = reinterpret_cast<uint64_t>(directAddress);
  unsigned char bytes[16] = {};
  uint64_t address = reinterpret_cast<uint64_t>(external);
  int32_t v = 41;
  std::memcpy(bytes, &address, sizeof(address));
  std::memcpy(bytes + 8, &v, sizeof(v));
  void *args[] = {&p, bytes};
  uint64_t sizes[] = {8, 16};
  M_driver_queue_execute_config_t config = {};
  config.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
  config.config.gpu.grid = {1, 1, 1};
  config.config.gpu.block = {1, 1, 1};
  ASSERT_SUCCESS(queue_execute(queues[0], kernel, &config, args, sizes, 2));
  int32_t viaStruct = 0;
  M_driver_memory_view view{wrapped, 0, sizeof(int32_t)};
  EXPECT_SUCCESS(queue_copy_from_device(queues[0], &viaStruct, &view));
  EXPECT_SUCCESS(queue_synchronize(queues[0]));
  EXPECT_EQ(viaStruct, 41);
  EXPECT_SUCCESS(memory_free_sync(context, wrapped));
  EXPECT_SUCCESS(memory_free_sync(context, direct));
  EXPECT_EQ(memFree(zeContext, external), 0);
  dlclose(loader);
}

// LLVM's fdiv is correctly rounded; IGC's default for SPIR-V is OpenCL's
// 2.5 ulp, which got 10/3, 1/7 and 0.1/0.3 one ulp off on the B70.
TEST_F(SharedMemoryTest, Float32DivisionIsCorrectlyRounded) {
  M_driver_function_t kernel =
      load("divide",
           "kernels::divide(::Pointer[True, MutUnsafeAnyOrigin, "
           "::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), "
           "AddressSpace.GENERIC])");
  ASSERT_NE(kernel, nullptr);
  volatile float dividends[4] = {10.0f, 1.0f, 0.1f, 3.0e38f};
  volatile float divisors[4] = {3.0f, 7.0f, 0.3f, 1.5e38f};
  float data[12] = {};
  for (int i = 0; i < 4; ++i) {
    data[4 + i] = dividends[i];
    data[8 + i] = divisors[i];
  }
  M_driver_memory_t memory = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], sizeof(data), &memory));
  M_driver_memory_view view{memory, 0, sizeof(data)};
  void *address = nullptr;
  ASSERT_SUCCESS(
      memory_property(memory, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &address));
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &view, data));
  ASSERT_SUCCESS(launch(kernel, reinterpret_cast<uint64_t>(address), 1, 0));
  ASSERT_SUCCESS(queue_copy_from_device(queues[0], data, &view));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  for (int i = 0; i < 4; ++i) {
    float expected = dividends[i] / divisors[i];
    uint32_t got = 0, want = 0;
    std::memcpy(&got, &data[i], sizeof(got));
    std::memcpy(&want, &expected, sizeof(want));
    EXPECT_EQ(got, want) << dividends[i] << " / " << divisors[i] << ": "
                         << data[i] << ", expected " << expected;
  }
  EXPECT_SUCCESS(memory_free_sync(contexts[0], memory));
}

// A List in a kernel: 4,096 work-items at once each grow a list
// through malloc and free from the context's heap, and every sum is right,
// so no two got the same memory. The kernel is `uses_heap` in
// plugins/intel/tests/codegen/kernels.mojo:
// p[i] = sum of i * 8 + k for k < i % 7 + 1.
TEST_F(SharedMemoryTest, KernelsAllocateFromTheDeviceHeap) {
  M_driver_function_t heap = load("uses_heap");
  ASSERT_NE(heap, nullptr);
  constexpr uint32_t blocks = 64;
  for (int round = 0; round < 2; ++round) {
    std::vector<int32_t> result = run(heap, blocks, 0);
    ASSERT_EQ(result.size(), 64u * blocks);
    size_t wrong = 0;
    for (int32_t i = 0; i < static_cast<int32_t>(result.size()); ++i) {
      int32_t count = i % 7 + 1;
      int32_t expected = count * i * 8 + count * (count - 1) / 2;
      if (result[i] != expected && wrong++ < 8)
        ADD_FAILURE() << "round " << round << ", p[" << i << "] = " << result[i]
                      << ", expected " << expected;
    }
    EXPECT_EQ(wrong, 0u) << "round " << round;
  }
}

// malloc returns null once the heap is used up, as CUDA's does (blocks are
// held, never freed, here). With a 1 MiB heap (MOJO_INTEL_HEAP_SIZE): a
// request bigger than the heap fails without using any of it; of 63
// work-items asking for 65,520 bytes at once (64 KiB blocks with their
// 16-byte headers) exactly 16 get memory; after that even 16 bytes fail. The
// kernel is `heap_exhaust`: p[i] = malloc(p[i]) != null.
TEST_F(SharedMemoryTest, MallocReturnsNullWhenTheHeapIsUsedUp) {
  // The context's heap is created, at this size, when the first module that
  // allocates loads.
  ASSERT_EQ(setenv("MOJO_INTEL_HEAP_SIZE", "1048576", 1), 0);
  M_driver_function_t exhaust = load("heap_exhaust");
  unsetenv("MOJO_INTEL_HEAP_SIZE");
  ASSERT_NE(exhaust, nullptr);
  std::vector<int32_t> requests(64, 65520);
  requests[0] = 2 << 20;
  std::vector<int32_t> got = runOn(exhaust, requests, 0);
  ASSERT_EQ(got.size(), 64u);
  EXPECT_EQ(got[0], 0) << "a request bigger than the heap succeeded";
  int successes = 0;
  for (size_t i = 1; i < got.size(); ++i) {
    EXPECT_TRUE(got[i] == 0 || got[i] == 1) << "p[" << i << "] = " << got[i];
    successes += got[i] == 1;
  }
  EXPECT_EQ(successes, 16);
  got = runOn(exhaust, std::vector<int32_t>(64, 16), 0);
  for (size_t i = 0; i < got.size(); ++i)
    EXPECT_EQ(got[i], 0) << "p[" << i << "]: the used-up heap gave memory";
}

// A request that doesn't fit in what is left fails without using any of
// it, so smaller requests still get the rest, as with CUDA's malloc (an
// atomic add that kept the bytes of every failed request made all later
// requests fail). With a 1 MiB heap: 8 blocks of
// 64 KiB and 56 of 32 bytes (16 asked for; 526,080 bytes used); then 64
// requests of 600,000 bytes (1 MiB blocks, the heap's size classes), each
// fitting the heap but not what is left, all fail; then 64 of 4,080 bytes
// (4 KiB blocks) all succeed; then of 64 more exactly 63 do, which is all
// the 260,352 bytes left hold.
TEST_F(SharedMemoryTest, MallocAfterARequestThatDidntFitGetsTheRest) {
  ASSERT_EQ(setenv("MOJO_INTEL_HEAP_SIZE", "1048576", 1), 0);
  M_driver_function_t exhaust = load("heap_exhaust");
  unsetenv("MOJO_INTEL_HEAP_SIZE");
  ASSERT_NE(exhaust, nullptr);
  auto count = [](const std::vector<int32_t> &got) {
    int successes = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      EXPECT_TRUE(got[i] == 0 || got[i] == 1) << "p[" << i << "] = " << got[i];
      successes += got[i] == 1;
    }
    return successes;
  };
  std::vector<int32_t> requests(64, 16);
  std::fill(requests.begin(), requests.begin() + 8, 65520);
  // (The 600,000-byte requests round up to 1 MiB blocks, which fit the
  // heap's capacity but not what is left.)
  EXPECT_EQ(count(runOn(exhaust, requests, 0)), 64);
  EXPECT_EQ(count(runOn(exhaust, std::vector<int32_t>(64, 600000), 0)), 0)
      << "a request larger than what is left got memory";
  EXPECT_EQ(count(runOn(exhaust, std::vector<int32_t>(64, 4080), 0)), 64)
      << "the requests that didn't fit used up the heap";
  EXPECT_EQ(count(runOn(exhaust, std::vector<int32_t>(64, 4080), 0)), 63);
}

// Freed blocks are reused (a heap that only bumps runs out in a long-lived
// context whose kernels allocate on every launch). With a
// 1 MiB heap, 64 work-items each allocate, write, check and free a block 256
// times per launch, over 8 launches: 56 work-items take 4 KiB blocks and 8
// take 64 KiB blocks (736 KiB at most at once), 140 MiB in all, all served.
// The kernel is `heap_reuse`: p[i] = the rounds whose block held its values.
TEST_F(SharedMemoryTest, FreedBlocksAreReused) {
  ASSERT_EQ(setenv("MOJO_INTEL_HEAP_SIZE", "1048576", 1), 0);
  M_driver_function_t reuse = load("heap_reuse");
  unsetenv("MOJO_INTEL_HEAP_SIZE");
  ASSERT_NE(reuse, nullptr);
  std::vector<int32_t> requests(64, 4080);
  std::fill(requests.begin() + 56, requests.end(), 65520);
  for (int launch = 0; launch < 8; ++launch) {
    std::vector<int32_t> got = runOn(reuse, requests, 0);
    ASSERT_EQ(got.size(), 64u);
    for (size_t i = 0; i < got.size(); ++i)
      EXPECT_EQ(got[i], 256) << "launch " << launch << ", p[" << i << "]";
  }
}

// MOJO_INTEL_HEAP_SIZE must be a number of bytes from 1 to 2^62: strtoull
// alone took "-1" for 2^64 - 1, the heap's size wrapped to a few bytes and
// malloc's capacity didn't, so kernels wrote past the heap. A module that
// allocates fails to load with such a value (set but empty too); then the
// default heap works.
TEST_F(SharedMemoryTest, HeapSizeMustBeAPositiveNumberOfBytes) {
  std::vector<char> spirv = readSpirv("heap_exhaust");
  ASSERT_FALSE(spirv.empty()) << "missing test/data/heap_exhaust.spv";
  M_driver_static_bundle bundle = {};
  bundle.mapped_data = {spirv.data(), spirv.size()};
  bundle.file_type = "object";
  bundle.file_type_len = 6;
  int taken = 0;
  for (const char *bad :
       {"", "-1", "18446744073709551615", "18446744073709551616",
        "99999999999999999999999", "4611686018427387905", "0", "+16", " 16",
        "16 KiB", "0x100"}) {
    if (setenv("MOJO_INTEL_HEAP_SIZE", bad, 1) != 0) {
      ADD_FAILURE() << "setenv failed";
      break;
    }
    M_driver_runtime_bundle_t runtime = nullptr;
    M_driver_status_t status =
        bundle_load(contexts[0], &bundle, nullptr, &runtime);
    if (status == M_DRIVER_STATUS_SUCCESS) {
      ADD_FAILURE() << "MOJO_INTEL_HEAP_SIZE=\"" << bad << "\" was taken";
      ++taken;
    }
    if (runtime != nullptr)
      bundles.push_back(runtime);
  }
  unsetenv("MOJO_INTEL_HEAP_SIZE");
  // A value taken may have made a heap whose capacity is past its memory,
  // and this context keeps the first heap made: running a kernel on it can
  // fault the shared B70 and hang the next process to use it. Fail
  // here instead.
  ASSERT_EQ(taken, 0) << "not running a kernel on a heap a bad value made";
  M_driver_function_t exhaust = load("heap_exhaust");
  ASSERT_NE(exhaust, nullptr);
  std::vector<int32_t> got = runOn(exhaust, std::vector<int32_t>(64, 16), 0);
  for (size_t i = 0; i < got.size(); ++i)
    EXPECT_EQ(got[i], 1) << "p[" << i << "]: the default heap gave no memory";
}

TEST_F(LaunchTest, DynamicSharedMemoryIsIgnoredByAKernelWithout) {
  // As in CUDA, a kernel that declares none just doesn't see it.
  constexpr int64_t n = 256;
  M_driver_memory_t memory = nullptr;
  ASSERT_SUCCESS(
      memory_alloc_sync(contexts[0], 3 * n * sizeof(float), &memory));
  uint64_t base = addressOf(memory);
  void *args[] = {&base, &base, &base, const_cast<int64_t *>(&n)};
  uint64_t sizes[] = {8, 8, 8, 8};
  M_driver_queue_execute_config_t config = {};
  config.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
  config.config.gpu.grid = {1, 1, 1};
  config.config.gpu.block = {256, 1, 1};
  config.config.gpu.shared_mem_bytes = 1024;
  EXPECT_SUCCESS(queue_execute(queues[0], function, &config, args, sizes, 4));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  ASSERT_SUCCESS(memory_free_sync(contexts[0], memory));
}

} // namespace
