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

// Run one kernel from a SPIR-V module on the first Level Zero GPU with one
// argument: a zeroed device buffer of 1024 int32 whose first word is 7. The
// kernel's printf output reaches stdout through the driver; the buffer's first
// `count` words are printed afterwards. By default the kernel runs as one
// work-item. For checking device-side features without the Mojo runtime.
//
// Usage: l0-run <module.spv> <substring of the kernel name>
//               [<global size> <local size> [<count>]]
//
// With L0RUN_STATUS=1 it also reports, after the kernel, what Level Zero says
// about the device, the context and the command list, and how long
// zeDeviceGetStatus takes: whether a hung kernel's reset is visible.

#include <level_zero/ze_api.h>

#include <algorithm>
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
    std::fprintf(stderr, "FAIL: %s returned 0x%x\n", what,
                 static_cast<unsigned>(r));
    std::exit(1);
  }
}
#define CHECK(call) check((call), #call)

} // namespace

int main(int argc, char **argv) {
  if (argc != 3 && argc != 5 && argc != 6) {
    std::fprintf(stderr,
                 "usage: %s <module.spv> <kernel-name-substring> "
                 "[<global size> <local size> [<count>]]\n",
                 argv[0]);
    return 2;
  }
  uint32_t global = argc >= 5 ? std::strtoul(argv[3], nullptr, 10) : 1;
  uint32_t local = argc >= 5 ? std::strtoul(argv[4], nullptr, 10) : 1;
  uint32_t printCount = argc == 6 ? std::strtoul(argv[5], nullptr, 10) : 1;
  constexpr uint32_t kWords = 1024;
  if (local == 0 || global % local != 0 || printCount > kWords) {
    std::fprintf(stderr,
                 "FAIL: the global size must be a multiple of the "
                 "local size, and count at most %u\n",
                 kWords);
    return 2;
  }
  std::ifstream in(argv[1], std::ios::binary);
  std::vector<uint8_t> spirv((std::istreambuf_iterator<char>(in)), {});
  if (spirv.empty()) {
    std::fprintf(stderr, "FAIL: cannot read %s\n", argv[1]);
    return 1;
  }

  ze_driver_handle_t driver = nullptr;
  ze_device_handle_t device = nullptr;
  ze_init_driver_type_desc_t init = {ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC,
                                     nullptr, ZE_INIT_DRIVER_TYPE_FLAG_GPU};
  uint32_t count = 1;
  CHECK(zeInitDrivers(&count, &driver, &init));
  if (count == 0) {
    std::fprintf(stderr, "FAIL: no Level Zero GPU driver\n");
    return 1;
  }
  count = 1;
  CHECK(zeDeviceGet(driver, &count, &device));
  ze_context_handle_t context = nullptr;
  ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  CHECK(zeContextCreate(driver, &cdesc, &context));
  ze_command_queue_desc_t qdesc = {};
  qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  qdesc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  ze_command_list_handle_t list = nullptr;
  CHECK(zeCommandListCreateImmediate(context, device, &qdesc, &list));

  ze_module_desc_t mdesc = {};
  mdesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  mdesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  mdesc.inputSize = spirv.size();
  mdesc.pInputModule = spirv.data();
  mdesc.pBuildFlags = "";
  ze_module_handle_t module = nullptr;
  ze_module_build_log_handle_t log = nullptr;
  ze_result_t r = zeModuleCreate(context, device, &mdesc, &module, &log);
  if (r != ZE_RESULT_SUCCESS) {
    size_t size = 0;
    zeModuleBuildLogGetString(log, &size, nullptr);
    std::string text(size, '\0');
    zeModuleBuildLogGetString(log, &size, text.data());
    std::fprintf(stderr, "build log:\n%s\n", text.c_str());
  }
  if (log)
    zeModuleBuildLogDestroy(log);
  check(r, "zeModuleCreate");

  uint32_t names = 0;
  CHECK(zeModuleGetKernelNames(module, &names, nullptr));
  std::vector<const char *> list_names(names);
  CHECK(zeModuleGetKernelNames(module, &names, list_names.data()));
  const char *name = nullptr;
  for (const char *candidate : list_names)
    if (std::strstr(candidate, argv[2]))
      name = candidate;
  if (!name) {
    std::fprintf(stderr, "FAIL: no kernel name contains '%s'\n", argv[2]);
    return 1;
  }
  ze_kernel_desc_t kdesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, name};
  ze_kernel_handle_t kernel = nullptr;
  r = zeKernelCreate(module, &kdesc, &kernel);
  if (r != ZE_RESULT_SUCCESS)
    std::fprintf(stderr, "kernel: %s\n", name);
  check(r, "zeKernelCreate");

  ze_host_mem_alloc_desc_t hdesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                    nullptr, 0};
  ze_device_mem_alloc_desc_t ddesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                      nullptr, 0, 0};
  int32_t *buffer = nullptr;
  CHECK(zeMemAllocShared(context, &ddesc, &hdesc, kWords * sizeof(int32_t), 64,
                         device, reinterpret_cast<void **>(&buffer)));
  std::memset(buffer, 0, kWords * sizeof(int32_t));
  buffer[0] = 7;
  CHECK(zeKernelSetGroupSize(kernel, local, 1, 1));
  CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(buffer), &buffer));
  ze_group_count_t groups = {global / local, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(list, kernel, &groups, nullptr, 0,
                                        nullptr));
  ze_result_t synced = zeCommandListHostSynchronize(list, UINT64_MAX);
  if (std::getenv("L0RUN_STATUS") != nullptr) {
    std::printf("status: synchronize 0x%x\n", static_cast<unsigned>(synced));
    std::printf("status: device 0x%x\n",
                static_cast<unsigned>(zeDeviceGetStatus(device)));
    std::printf("status: context 0x%x\n",
                static_cast<unsigned>(zeContextGetStatus(context)));
    std::printf(
        "status: synchronize again 0x%x\n",
        static_cast<unsigned>(zeCommandListHostSynchronize(list, 1000000000)));
    int32_t probe = -1;
    ze_result_t copied = zeCommandListAppendMemoryCopy(
        list, &probe, buffer, sizeof(probe), nullptr, 0, nullptr);
    ze_result_t after = zeCommandListHostSynchronize(list, 5000000000);
    std::printf("status: copy 0x%x, synchronize 0x%x, read %d\n",
                static_cast<unsigned>(copied), static_cast<unsigned>(after),
                probe);
    std::vector<double> us;
    for (int i = 0; i < 1000; ++i) {
      auto t = std::chrono::steady_clock::now();
      (void)zeDeviceGetStatus(device);
      us.push_back(std::chrono::duration<double, std::micro>(
                       std::chrono::steady_clock::now() - t)
                       .count());
    }
    std::sort(us.begin(), us.end());
    std::printf("status: zeDeviceGetStatus median %.2f us, p99 %.2f us\n",
                us[500], us[990]);
  }
  check(synced, "zeCommandListHostSynchronize");
  std::fflush(stdout);
  for (uint32_t i = 0; i < printCount; ++i)
    std::printf("buffer[%u] = %d\n", i, buffer[i]);

  zeMemFree(context, buffer);
  zeKernelDestroy(kernel);
  zeModuleDestroy(module);
  zeCommandListDestroy(list);
  zeContextDestroy(context);
  return 0;
}
