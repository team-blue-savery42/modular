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

// Runs the one kernel of a bounds_loop module (built by the Mojo Intel
// compiler plugin from tests/e2e/bounds_loop.mojo) on one work-item, as the
// Mojo program does, once with the Mojo driver plugin's build options and
// once with -cl-opt-disable added. The kernel takes one int32 buffer p, with
// p[0] = 0 and the rest -1; it writes the enumerate loop's index of each
// iteration to p[4 + n], the number of bytes that differ to p[1] and the
// number of iterations to p[2].
//
// Usage: run-bounds <module.spv>

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

/// Runs the module's only kernel and returns p[0..15].
std::vector<int32_t> run(ze_context_handle_t context, ze_device_handle_t device,
                         const std::vector<char> &spirv,
                         const std::string &options) {
  ze_module_desc_t mdesc = {};
  mdesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  mdesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  mdesc.inputSize = spirv.size();
  mdesc.pInputModule = reinterpret_cast<const uint8_t *>(spirv.data());
  mdesc.pBuildFlags = options.c_str();
  ze_module_handle_t module = nullptr;
  check(zeModuleCreate(context, device, &mdesc, &module, nullptr),
        "zeModuleCreate");
  // The module's kernel; IGC also lists its own symbol table kernel.
  uint32_t count = 0;
  check(zeModuleGetKernelNames(module, &count, nullptr), "kernel names");
  std::vector<const char *> names(count);
  check(zeModuleGetKernelNames(module, &count, names.data()), "kernel names");
  const char *name = nullptr;
  for (const char *n : names)
    if (std::string(n).rfind("Intel_Symbol_Table", 0) != 0) {
      if (name) {
        std::printf("expected one kernel, found %s and %s\n", name, n);
        std::exit(1);
      }
      name = n;
    }
  if (!name) {
    std::printf("no kernel in the module\n");
    std::exit(1);
  }
  ze_kernel_desc_t kdesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, name};
  ze_kernel_handle_t kernel = nullptr;
  check(zeKernelCreate(module, &kdesc, &kernel), "zeKernelCreate");
  check(zeKernelSetGroupSize(kernel, 1, 1, 1), "zeKernelSetGroupSize");

  ze_device_mem_alloc_desc_t adesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                      nullptr, 0, 0};
  void *p = nullptr;
  check(zeMemAllocDevice(context, &adesc, 64, 64, device, &p), "alloc p");
  check(zeKernelSetArgumentValue(kernel, 0, sizeof(p), &p), "arg 0");
  ze_command_queue_desc_t qdesc = {};
  qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  qdesc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  ze_command_list_handle_t list = nullptr;
  check(zeCommandListCreateImmediate(context, device, &qdesc, &list),
        "zeCommandListCreateImmediate");
  std::vector<int32_t> h(16, -1);
  h[0] = 0;
  check(zeCommandListAppendMemoryCopy(list, p, h.data(), 64, nullptr, 0,
                                      nullptr),
        "copy in");
  ze_group_count_t groups = {1, 1, 1};
  check(zeCommandListAppendLaunchKernel(list, kernel, &groups, nullptr, 0,
                                        nullptr),
        "launch");
  check(zeCommandListAppendMemoryCopy(list, h.data(), p, 64, nullptr, 0,
                                      nullptr),
        "copy out");
  check(zeCommandListHostSynchronize(list, 10'000'000'000ull), "synchronize");
  zeCommandListDestroy(list);
  zeMemFree(context, p);
  zeKernelDestroy(kernel);
  zeModuleDestroy(module);
  return h;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::printf("usage: run-bounds <module.spv>\n");
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
    std::vector<int32_t> h = run(context, device, spirv, kOptions + extra);
    bool right = h[2] == 3 && h[4] == 0 && h[5] == 1 && h[6] == 2;
    wrong += !right;
    std::printf("options \"%s%s\":\n", kOptions, extra.c_str());
    std::printf("  iterations %d (expected 3)\n", h[2]);
    std::printf("  indices %d %d %d (expected 0 1 2)%s\n", h[4], h[5], h[6],
                right ? "" : "  <- wrong");
    std::printf("  bytes that differ %d (expected 3)\n", h[1]);
  }
  zeContextDestroy(context);
  return wrong == 0 ? 0 : 1;
}
