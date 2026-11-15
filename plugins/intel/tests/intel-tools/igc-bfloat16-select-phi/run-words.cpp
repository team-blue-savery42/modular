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

// Runs a SPIR-V module's kernel named on the command line, with one
// work-item and one argument: a buffer of 64 16-bit words. Words 0..15 are
// the inputs given on the command line (hex), the rest start as 0xdead. The
// kernel is built twice, with the Mojo driver plugin's options and with
// -cl-opt-disable added; both buffers are printed in hex.
//
// Usage: run-words <module.spv> <kernel> [hex-word...]
#include <level_zero/ze_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

constexpr int kWords = 64;
constexpr const char *kOptions =
    "-ze-intel-enable-auto-large-GRF-mode "
    "-cl-fp32-correctly-rounded-divide-sqrt";

std::vector<uint16_t> run(ze_context_handle_t context,
                          ze_device_handle_t device,
                          const std::vector<char> &spirv, const char *name,
                          const std::string &options,
                          const std::vector<uint16_t> &input) {
  ze_module_desc_t mdesc = {};
  mdesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  mdesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  mdesc.inputSize = spirv.size();
  mdesc.pInputModule = reinterpret_cast<const uint8_t *>(spirv.data());
  mdesc.pBuildFlags = options.c_str();
  ze_module_handle_t module = nullptr;
  check(zeModuleCreate(context, device, &mdesc, &module, nullptr),
        "zeModuleCreate");
  ze_kernel_desc_t kdesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, name};
  ze_kernel_handle_t kernel = nullptr;
  check(zeKernelCreate(module, &kdesc, &kernel), "zeKernelCreate");
  check(zeKernelSetGroupSize(kernel, 1, 1, 1), "zeKernelSetGroupSize");
  std::vector<uint16_t> data = input;
  size_t bytes = data.size() * sizeof(uint16_t);
  ze_device_mem_alloc_desc_t adesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                      nullptr, 0, 0};
  void *p = nullptr;
  check(zeMemAllocDevice(context, &adesc, bytes, 256, device, &p),
        "zeMemAllocDevice");
  check(zeKernelSetArgumentValue(kernel, 0, sizeof(p), &p), "arg 0");
  ze_command_queue_desc_t qdesc = {};
  qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  qdesc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  ze_command_list_handle_t list = nullptr;
  check(zeCommandListCreateImmediate(context, device, &qdesc, &list),
        "zeCommandListCreateImmediate");
  check(zeCommandListAppendMemoryCopy(list, p, data.data(), bytes, nullptr, 0,
                                      nullptr),
        "copy in");
  ze_group_count_t groups = {1, 1, 1};
  check(zeCommandListAppendLaunchKernel(list, kernel, &groups, nullptr, 0,
                                        nullptr),
        "launch");
  check(zeCommandListAppendMemoryCopy(list, data.data(), p, bytes, nullptr, 0,
                                      nullptr),
        "copy out");
  check(zeCommandListHostSynchronize(list, UINT64_MAX), "synchronize");
  zeCommandListDestroy(list);
  zeMemFree(context, p);
  zeKernelDestroy(kernel);
  zeModuleDestroy(module);
  return data;
}

void print(const char *label, const std::vector<uint16_t> &data) {
  std::printf("%-16s", label);
  for (int i = 0; i < 32; ++i)
    std::printf(" %04x", data[i]);
  std::printf("\n");
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::printf("usage: run-words <module.spv> <kernel> [hex-word...]\n");
    return 1;
  }
  std::ifstream in(argv[1], std::ios::binary);
  std::vector<char> spirv((std::istreambuf_iterator<char>(in)), {});
  std::vector<uint16_t> input(kWords, 0xdead);
  for (int i = 3; i < argc && i - 3 < 16; ++i)
    input[i - 3] = uint16_t(std::strtoul(argv[i], nullptr, 16));
  ze_init_driver_type_desc_t types = {ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC,
                                      nullptr, ZE_INIT_DRIVER_TYPE_FLAG_GPU};
  uint32_t count = 1;
  ze_driver_handle_t driver = nullptr;
  check(zeInitDrivers(&count, &driver, &types), "zeInitDrivers");
  count = 1;
  ze_device_handle_t device = nullptr;
  check(zeDeviceGet(driver, &count, &device), "zeDeviceGet");
  ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  ze_context_handle_t context = nullptr;
  check(zeContextCreate(driver, &cdesc, &context), "zeContextCreate");
  print("input", input);
  print("default", run(context, device, spirv, argv[2], kOptions, input));
  print("-cl-opt-disable", run(context, device, spirv, argv[2],
                               std::string(kOptions) + " -cl-opt-disable",
                               input));
  zeContextDestroy(context);
  return 0;
}
