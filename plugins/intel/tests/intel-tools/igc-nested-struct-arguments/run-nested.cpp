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

// Runs a module's kernel k(out, name) once with the Mojo driver
// plugin's build options and once with -cl-opt-disable added, and checks
// out[0..3] = { 147, 81, address of name, 3 }. nested.spv's kernel passes
// { 147, 81, { (generic) name, 3 } } by value to a noinline function that
// stores its four leaves; nested_ret.spv's noinline function returns that
// struct and the kernel stores its leaves. With "kernarg", the kernel is
// k(out, loc) and the runner passes loc = { 147, 81, { name, 3 } } by value
// (kernarg.spv).
//
// Usage: run-nested <module.spv> [kernarg]

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

constexpr const char *kOptions =
    "-ze-intel-enable-auto-large-GRF-mode "
    "-cl-fp32-correctly-rounded-divide-sqrt";

/// Runs kernel `k` and returns out[0..3] and the address of `name`.
std::vector<uint64_t> run(ze_context_handle_t context, ze_device_handle_t device,
                          const std::vector<char> &spirv,
                          const std::string &options, uint64_t &name,
                          bool kernarg) {
  ze_module_desc_t mdesc = {};
  mdesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  mdesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  mdesc.inputSize = spirv.size();
  mdesc.pInputModule = reinterpret_cast<const uint8_t *>(spirv.data());
  mdesc.pBuildFlags = options.c_str();
  ze_module_handle_t module = nullptr;
  check(zeModuleCreate(context, device, &mdesc, &module, nullptr),
        "zeModuleCreate");
  ze_kernel_desc_t kdesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "k"};
  ze_kernel_handle_t kernel = nullptr;
  check(zeKernelCreate(module, &kdesc, &kernel), "zeKernelCreate");
  check(zeKernelSetGroupSize(kernel, 1, 1, 1), "zeKernelSetGroupSize");

  ze_device_mem_alloc_desc_t adesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                      nullptr, 0, 0};
  void *out = nullptr;
  void *text = nullptr;
  check(zeMemAllocDevice(context, &adesc, 64, 64, device, &out), "alloc out");
  check(zeMemAllocDevice(context, &adesc, 64, 64, device, &text), "alloc name");
  name = reinterpret_cast<uint64_t>(text);
  check(zeKernelSetArgumentValue(kernel, 0, sizeof(out), &out), "arg 0");
  struct {
    uint64_t line, column;
    void *name;
    uint64_t length;
  } loc = {147, 81, text, 3};
  if (kernarg)
    check(zeKernelSetArgumentValue(kernel, 1, sizeof(loc), &loc), "arg 1");
  else
    check(zeKernelSetArgumentValue(kernel, 1, sizeof(text), &text), "arg 1");
  ze_command_queue_desc_t qdesc = {};
  qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  qdesc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  ze_command_list_handle_t list = nullptr;
  check(zeCommandListCreateImmediate(context, device, &qdesc, &list),
        "zeCommandListCreateImmediate");
  std::vector<uint64_t> result(4, 0xdeadbeef);
  check(zeCommandListAppendMemoryCopy(list, out, result.data(), 32, nullptr, 0,
                                      nullptr),
        "copy in");
  ze_group_count_t groups = {1, 1, 1};
  check(zeCommandListAppendLaunchKernel(list, kernel, &groups, nullptr, 0,
                                        nullptr),
        "launch");
  check(zeCommandListAppendMemoryCopy(list, result.data(), out, 32, nullptr, 0,
                                      nullptr),
        "copy out");
  check(zeCommandListHostSynchronize(list, 10'000'000'000ull), "synchronize");
  zeCommandListDestroy(list);
  zeMemFree(context, out);
  zeMemFree(context, text);
  zeKernelDestroy(kernel);
  zeModuleDestroy(module);
  return result;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::printf("usage: run-nested <module.spv> [kernarg]\n");
    return 1;
  }
  std::ifstream in(argv[1], std::ios::binary);
  std::vector<char> spirv((std::istreambuf_iterator<char>(in)), {});
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

  int wrong = 0;
  for (std::string extra : {"", " -cl-opt-disable"}) {
    uint64_t name = 0;
    std::vector<uint64_t> got =
        run(context, device, spirv, kOptions + extra, name,
            argc == 3 && std::string(argv[2]) == "kernarg");
    uint64_t want[4] = {147, 81, name, 3};
    std::printf("options \"%s%s\":\n", kOptions, extra.c_str());
    const char *fields[4] = {"line", "column", "name", "length"};
    for (int i = 0; i < 4; ++i) {
      bool bad = got[i] != want[i];
      wrong += bad;
      std::printf("  %-6s 0x%016llx  expected 0x%016llx%s\n", fields[i],
                  (unsigned long long)got[i], (unsigned long long)want[i],
                  bad ? "  <- wrong" : "");
    }
  }
  zeContextDestroy(context);
  return wrong == 0 ? 0 : 1;
}
