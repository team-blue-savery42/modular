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

// float16 log2: runs a Mojo-compiled kernel (log2_f16_w*.spv) that loads
// `width` halves from p[0..], stores log2 to p[64..] and exp2 to p[128..].
// Builds the module twice, with the Mojo driver plugin's options and with
// -cl-opt-disable added, and prints both results next to the host's. The
// buffer sits between 32 KiB guard zones; stores that land in them are
// reported by offset from p.
//
// Usage: run-log2 <module.spv> <width>
// RUN_LOG2_OPTIONS replaces the base build options.

#include <level_zero/ze_api.h>

#include <algorithm>
#include <cmath>
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

constexpr int kSection = 64;
/// Halves of guard zone on each side of the buffer, filled with kGuardBits.
constexpr int kGuard = 16384;
constexpr uint16_t kGuardBits = 0x7e7e;
constexpr const char *kOptions =
    "-ze-intel-enable-auto-large-GRF-mode "
    "-cl-fp32-correctly-rounded-divide-sqrt";

/// Runs the module's one kernel with the given build options; returns the
/// buffer of 3 * kSection halves.
std::vector<_Float16> run(ze_context_handle_t context, ze_device_handle_t device,
                          const std::vector<char> &spirv,
                          const std::string &options, int width) {
  ze_module_desc_t mdesc = {};
  mdesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  mdesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  mdesc.inputSize = spirv.size();
  mdesc.pInputModule = reinterpret_cast<const uint8_t *>(spirv.data());
  mdesc.pBuildFlags = options.c_str();
  ze_module_handle_t module = nullptr;
  check(zeModuleCreate(context, device, &mdesc, &module, nullptr),
        "zeModuleCreate");
  uint32_t count = 0;
  check(zeModuleGetKernelNames(module, &count, nullptr), "kernel names");
  std::vector<const char *> names(count);
  check(zeModuleGetKernelNames(module, &count, names.data()), "kernel names");
  ze_kernel_desc_t kdesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                            names[0]};
  ze_kernel_handle_t kernel = nullptr;
  check(zeKernelCreate(module, &kdesc, &kernel), "zeKernelCreate");
  check(zeKernelSetGroupSize(kernel, 1, 1, 1), "zeKernelSetGroupSize");

  std::vector<_Float16> data(3 * kSection, _Float16(0));
  for (int i = 0; i < kSection; ++i)
    data[i] = _Float16(float(i % 28) * 0.25f + 0.5f);
  std::vector<_Float16> whole(kGuard + data.size() + kGuard);
  for (_Float16 &h : whole)
    __builtin_memcpy(&h, &kGuardBits, 2);
  std::copy(data.begin(), data.end(), whole.begin() + kGuard);
  size_t bytes = whole.size() * sizeof(_Float16);
  ze_device_mem_alloc_desc_t adesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                      nullptr, 0, 0};
  void *base = nullptr;
  check(zeMemAllocDevice(context, &adesc, bytes, 256, device, &base),
        "zeMemAllocDevice");
  void *p = static_cast<_Float16 *>(base) + kGuard;
  check(zeKernelSetArgumentValue(kernel, 0, sizeof(p), &p), "arg 0");
  ze_command_queue_desc_t qdesc = {};
  qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  qdesc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  ze_command_list_handle_t list = nullptr;
  check(zeCommandListCreateImmediate(context, device, &qdesc, &list),
        "zeCommandListCreateImmediate");
  check(zeCommandListAppendMemoryCopy(list, base, whole.data(), bytes, nullptr,
                                      0, nullptr),
        "copy in");
  ze_group_count_t groups = {1, 1, 1};
  check(zeCommandListAppendLaunchKernel(list, kernel, &groups, nullptr, 0,
                                        nullptr),
        "launch");
  check(zeCommandListAppendMemoryCopy(list, whole.data(), base, bytes, nullptr,
                                      0, nullptr),
        "copy out");
  check(zeCommandListHostSynchronize(list, UINT64_MAX), "synchronize");
  int stray = 0;
  for (size_t i = 0; i < whole.size(); ++i) {
    if (i >= kGuard && i < kGuard + data.size())
      continue;
    uint16_t bits;
    __builtin_memcpy(&bits, &whole[i], 2);
    if (bits != kGuardBits && stray++ < 16)
      std::printf("  [%s] stray store at p[%ld]: %g\n", options.c_str(),
                  long(i) - kGuard, float(whole[i]));
  }
  if (stray > 0)
    std::printf("  [%s] %d halves written outside p[0..%zu)\n",
                options.c_str(), stray, data.size());
  std::copy(whole.begin() + kGuard, whole.begin() + kGuard + data.size(),
            data.begin());
  zeCommandListDestroy(list);
  zeMemFree(context, base);
  zeKernelDestroy(kernel);
  zeModuleDestroy(module);
  return data;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::printf("usage: run-log2 <module.spv> <width>\n");
    return 1;
  }
  std::ifstream in(argv[1], std::ios::binary);
  std::vector<char> spirv((std::istreambuf_iterator<char>(in)), {});
  int width = std::atoi(argv[2]);
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

  const char *base = std::getenv("RUN_LOG2_OPTIONS");
  std::string options = base != nullptr ? base : kOptions;
  std::vector<_Float16> optimized = run(context, device, spirv, options, width);
  std::vector<_Float16> unoptimized =
      run(context, device, spirv, options + " -cl-opt-disable", width);
  int wrong = 0;
  std::printf("lane  x       log2: host      default    -cl-opt-disable\n");
  for (int i = 0; i < width; ++i) {
    float x = float(optimized[i]);
    float host = float(_Float16(std::log2(x)));
    float a = float(optimized[kSection + i]);
    float b = float(unoptimized[kSection + i]);
    bool bad = std::fabs(a - host) > 0.01f;
    wrong += bad;
    std::printf("%4d  %-6g  %12g  %10g  %10g%s\n", i, x, host, a, b,
                bad ? "  <- wrong" : "");
  }
  std::printf("default build: %d of %d lanes wrong\n", wrong, width);
  zeContextDestroy(context);
  return wrong == 0 ? 0 : 1;
}
