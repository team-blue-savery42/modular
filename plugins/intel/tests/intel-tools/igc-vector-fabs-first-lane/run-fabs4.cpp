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

// Runs kernel k of fabs4.spv (4-wide float vectors) on x = 2.5 in every lane
// and counts lanes whose result isn't 1/sqrt(2.5), the right answer:
//   alone      k is the first kernel of the process
//   after-fill fill.spv's kernel (it leaves +inf in registers), then k at once
//   after-idle fill, then 100 ms with the GPU idle, then k
// Usage: run-fabs4 fabs4.spv fill.spv alone|after-fill|after-idle [OPTIONS]
// OPTIONS (such as -cl-opt-disable) are added to k's IGC options.
#include <level_zero/ze_api.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
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
ze_context_handle_t ctx;
ze_device_handle_t dev;
ze_command_list_handle_t list;

std::string extra;

ze_kernel_handle_t load(const char *path, const char *name) {
  std::ifstream in(path, std::ios::binary);
  std::vector<char> spv((std::istreambuf_iterator<char>(in)), {});
  if (spv.empty()) {
    std::printf("cannot read %s\n", path);
    std::exit(1);
  }
  ze_module_desc_t md = {ZE_STRUCTURE_TYPE_MODULE_DESC};
  md.format = ZE_MODULE_FORMAT_IL_SPIRV;
  md.inputSize = spv.size();
  md.pInputModule = reinterpret_cast<const uint8_t *>(spv.data());
  std::string options = kOptions;
  if (std::strcmp(name, "k") == 0 && !extra.empty())
    options += " " + extra;
  md.pBuildFlags = options.c_str();
  ze_module_handle_t m;
  check(zeModuleCreate(ctx, dev, &md, &m, nullptr), "zeModuleCreate");
  ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, name};
  ze_kernel_handle_t k;
  check(zeKernelCreate(m, &kd, &k), "zeKernelCreate");
  return k;
}

void *alloc(size_t bytes) {
  ze_device_mem_alloc_desc_t d = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  void *p;
  check(zeMemAllocDevice(ctx, &d, bytes, 64, dev, &p), "zeMemAllocDevice");
  return p;
}

void launch(ze_kernel_handle_t k, void *a, void *b, uint32_t items) {
  uint32_t gx = 0, gy = 0, gz = 0;
  check(zeKernelSuggestGroupSize(k, items, 1, 1, &gx, &gy, &gz), "suggest");
  check(zeKernelSetGroupSize(k, gx, 1, 1), "group size");
  check(zeKernelSetArgumentValue(k, 0, sizeof(a), &a), "arg 0");
  check(zeKernelSetArgumentValue(k, 1, sizeof(b), &b), "arg 1");
  ze_group_count_t g = {items / gx, 1, 1};
  check(zeCommandListAppendLaunchKernel(list, k, &g, nullptr, 0, nullptr),
        "launch");
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 4 && argc != 5) {
    std::printf("usage: run-fabs4 fabs4.spv fill.spv alone|after-fill|after-idle\n");
    return 2;
  }
  std::string mode = argv[3];
  if (argc == 5)
    extra = argv[4];
  ze_driver_handle_t drv;
  ze_init_driver_type_desc_t t = {ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC,
                                  nullptr, ZE_INIT_DRIVER_TYPE_FLAG_GPU};
  uint32_t n = 1;
  check(zeInitDrivers(&n, &drv, &t), "zeInitDrivers");
  n = 1;
  check(zeDeviceGet(drv, &n, &dev), "zeDeviceGet");
  ze_context_desc_t cd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  check(zeContextCreate(drv, &cd, &ctx), "zeContextCreate");
  // As the Mojo driver plugin creates its queues.
  ze_command_queue_desc_t q = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  q.ordinal = 0;
  q.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  q.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  check(zeCommandListCreateImmediate(ctx, dev, &q, &list),
        "zeCommandListCreateImmediate");
  ze_kernel_handle_t k = load(argv[1], "k");
  ze_kernel_handle_t fill = load(argv[2], "fill");

  constexpr uint32_t items = 1 << 16; // work-items, 4 floats each
  std::vector<float> x(4 * items, 2.5f), y(4 * items, -1.0f);
  std::vector<float> infs(64 * items, INFINITY);
  void *dx = alloc(x.size() * 4), *dy = alloc(y.size() * 4);
  void *fin = alloc(infs.size() * 4), *fout = alloc(16 * items * 4);
  check(zeCommandListAppendMemoryCopy(list, dx, x.data(), x.size() * 4,
                                      nullptr, 0, nullptr), "copy x");
  check(zeCommandListAppendMemoryCopy(list, fin, infs.data(), infs.size() * 4,
                                      nullptr, 0, nullptr), "copy infs");
  check(zeCommandListHostSynchronize(list, UINT64_MAX), "sync");
  if (mode != "alone") {
    launch(fill, fin, fout, items);
    if (mode == "after-idle") {
      check(zeCommandListHostSynchronize(list, UINT64_MAX), "sync");
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  launch(k, dx, dy, items);
  check(zeCommandListAppendMemoryCopy(list, y.data(), dy, y.size() * 4,
                                      nullptr, 0, nullptr), "copy y");
  check(zeCommandListHostSynchronize(list, UINT64_MAX), "sync");
  const float want = 1.0f / std::sqrt(2.5f);
  long wrong[4] = {0, 0, 0, 0};
  float sample[4] = {0, 0, 0, 0};
  for (uint32_t i = 0; i < items; ++i)
    for (int l = 0; l < 4; ++l)
      if (std::fabs(y[4 * i + l] - want) > 1e-6f) {
        if (wrong[l] == 0)
          sample[l] = y[4 * i + l];
        ++wrong[l];
      }
  std::printf("%-10s %-16s want %.6f; wrong of %u per lane:", mode.c_str(),
              extra.c_str(), want, items);
  for (int l = 0; l < 4; ++l) {
    std::printf(" lane %d: %ld", l, wrong[l]);
    if (wrong[l])
      std::printf(" (the first is %g)", sample[l]);
  }
  std::printf("\n");
  return wrong[0] + wrong[1] + wrong[2] + wrong[3] ? 1 : 0;
}
