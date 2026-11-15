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

// After a process whose kernel page-faults, the next process to use the
// GPU hangs at its first wait. Plain Level Zero, no Mojo.
//
//   repro add     runs `add` on 1024 floats, waits up to 10 s, checks it
//   repro fault   runs `store_to` at an unmapped address, then waits
//
// Each wait is bounded; on timeout the program prints what Level Zero says
// (event, command list, device status) and exits 2. utils/check-max-gpu
// builds it and runs `repro add` after a faulting test, so that the hang
// lands on this throwaway process rather than on the next test.

#include <level_zero/ze_api.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

void check(ze_result_t r, const char *what) {
  if (r != ZE_RESULT_SUCCESS) {
    std::printf("%s failed: 0x%x\n", what, r);
    std::exit(1);
  }
}

constexpr uint64_t kTimeoutNs = 10'000'000'000;

struct Gpu {
  ze_driver_handle_t driver = nullptr;
  ze_device_handle_t device = nullptr;
  ze_context_handle_t context = nullptr;
  ze_command_list_handle_t list = nullptr;
  ze_module_handle_t module = nullptr;
};

Gpu open(const char *spvPath) {
  Gpu gpu;
  ze_init_driver_type_desc_t types = {ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC,
                                      nullptr, ZE_INIT_DRIVER_TYPE_FLAG_GPU};
  uint32_t count = 1;
  check(zeInitDrivers(&count, &gpu.driver, &types), "zeInitDrivers");
  count = 1;
  check(zeDeviceGet(gpu.driver, &count, &gpu.device), "zeDeviceGet");
  ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  check(zeContextCreate(gpu.driver, &cdesc, &gpu.context), "zeContextCreate");
  // In-order immediate list on the compute group, as the Mojo driver plugin
  // uses.
  ze_command_queue_desc_t qdesc = {};
  qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  qdesc.ordinal = 0;
  qdesc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  qdesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  check(
      zeCommandListCreateImmediate(gpu.context, gpu.device, &qdesc, &gpu.list),
      "zeCommandListCreateImmediate");
  std::ifstream in(spvPath, std::ios::binary);
  std::vector<char> spirv((std::istreambuf_iterator<char>(in)), {});
  if (spirv.empty()) {
    std::printf("cannot read %s\n", spvPath);
    std::exit(1);
  }
  ze_module_desc_t mdesc = {};
  mdesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  mdesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  mdesc.inputSize = spirv.size();
  mdesc.pInputModule = reinterpret_cast<const uint8_t *>(spirv.data());
  check(zeModuleCreate(gpu.context, gpu.device, &mdesc, &gpu.module, nullptr),
        "zeModuleCreate");
  return gpu;
}

ze_kernel_handle_t kernel(Gpu &gpu, const char *name) {
  ze_kernel_desc_t desc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, name};
  ze_kernel_handle_t k = nullptr;
  check(zeKernelCreate(gpu.module, &desc, &k), "zeKernelCreate");
  return k;
}

void *deviceAlloc(Gpu &gpu, size_t bytes) {
  ze_device_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                     nullptr, 0, 0};
  void *p = nullptr;
  check(zeMemAllocDevice(gpu.context, &desc, bytes, 64, gpu.device, &p),
        "zeMemAllocDevice");
  return p;
}

/// Waits up to kTimeoutNs for the list; on timeout reports and exits 2.
void wait(Gpu &gpu, const char *what) {
  auto start = std::chrono::steady_clock::now();
  ze_result_t r = zeCommandListHostSynchronize(gpu.list, kTimeoutNs);
  double s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  ze_result_t device = zeDeviceGetStatus(gpu.device);
  std::printf("%s: zeCommandListHostSynchronize 0x%x after %.3f s, "
              "zeDeviceGetStatus 0x%x\n",
              what, r, s, device);
  if (r == ZE_RESULT_NOT_READY) {
    std::printf("HANG: %s not done after %.0f s\n", what, kTimeoutNs / 1e9);
    std::fflush(stdout);
    std::_Exit(2);
  }
}

int add(Gpu &gpu) {
  constexpr size_t n = 1024;
  std::vector<float> a(n), b(n), out(n, -1.0f);
  for (size_t i = 0; i < n; ++i) {
    a[i] = float(i);
    b[i] = 2.0f * float(i);
  }
  void *da = deviceAlloc(gpu, n * 4), *db = deviceAlloc(gpu, n * 4),
       *dout = deviceAlloc(gpu, n * 4);
  check(zeCommandListAppendMemoryCopy(gpu.list, da, a.data(), n * 4, nullptr, 0,
                                      nullptr),
        "copy a");
  check(zeCommandListAppendMemoryCopy(gpu.list, db, b.data(), n * 4, nullptr, 0,
                                      nullptr),
        "copy b");
  wait(gpu, "copies in");
  ze_kernel_handle_t k = kernel(gpu, "add");
  check(zeKernelSetGroupSize(k, 64, 1, 1), "zeKernelSetGroupSize");
  check(zeKernelSetArgumentValue(k, 0, sizeof(da), &da), "arg 0");
  check(zeKernelSetArgumentValue(k, 1, sizeof(db), &db), "arg 1");
  check(zeKernelSetArgumentValue(k, 2, sizeof(dout), &dout), "arg 2");
  ze_group_count_t groups = {n / 64, 1, 1};
  check(zeCommandListAppendLaunchKernel(gpu.list, k, &groups, nullptr, 0,
                                        nullptr),
        "launch add");
  check(zeCommandListAppendMemoryCopy(gpu.list, out.data(), dout, n * 4,
                                      nullptr, 0, nullptr),
        "copy out");
  wait(gpu, "add");
  int wrong = 0;
  for (size_t i = 0; i < n; ++i)
    wrong += out[i] != a[i] + b[i];
  std::printf("add: %d of %zu wrong\n", wrong, n);
  return wrong == 0 ? 0 : 1;
}

int fault(Gpu &gpu) {
  ze_kernel_handle_t k = kernel(gpu, "store_to");
  check(zeKernelSetGroupSize(k, 1, 1, 1), "zeKernelSetGroupSize");
  uint64_t unmapped = 0x0000100000000000;
  int value = 41;
  check(zeKernelSetArgumentValue(k, 0, sizeof(unmapped), &unmapped), "arg 0");
  check(zeKernelSetArgumentValue(k, 1, sizeof(value), &value), "arg 1");
  ze_group_count_t groups = {1, 1, 1};
  check(zeCommandListAppendLaunchKernel(gpu.list, k, &groups, nullptr, 0,
                                        nullptr),
        "launch store_to");
  wait(gpu, "fault");
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3 || (std::strcmp(argv[1], "add") != 0 &&
                    std::strcmp(argv[1], "fault") != 0)) {
    std::printf("usage: repro add|fault kernels.spv\n");
    return 1;
  }
  Gpu gpu = open(argv[2]);
  return std::strcmp(argv[1], "add") == 0 ? add(gpu) : fault(gpu);
}
