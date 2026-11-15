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

// Runs kernel k(in, out) of mul128.spv (3 results per work-item),
// ops.spv (10) or vars.spv (8) on 4 work-items, once with the Mojo driver plugin's build
// options and once with -cl-opt-disable added, and compares each 128-bit
// result with the host's.
//
// Usage: run-mul128 <mul128.spv | ops.spv | vars.spv>

#include <level_zero/ze_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using u128 = unsigned __int128;

void check(ze_result_t r, const char *what) {
  if (r != ZE_RESULT_SUCCESS) {
    std::printf("%s failed: 0x%x\n", what, r);
    std::exit(1);
  }
}

constexpr const char *kOptions =
    "-ze-intel-enable-auto-large-GRF-mode "
    "-cl-fp32-correctly-rounded-divide-sqrt";
constexpr int kItems = 4;
int results = 3;

u128 make(uint64_t hi, uint64_t lo) { return (u128(hi) << 64) | lo; }

void print128(const char *label, u128 v) {
  std::printf("%s 0x%016llx%016llx", label, (unsigned long long)(v >> 64),
              (unsigned long long)v);
}

std::vector<u128> run(ze_context_handle_t context, ze_device_handle_t device,
                      const std::vector<char> &spirv,
                      const std::string &options,
                      const std::vector<u128> &in) {
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
  size_t in_bytes = in.size() * sizeof(u128);
  size_t out_bytes = results * kItems * sizeof(u128);
  void *din = nullptr;
  void *dout = nullptr;
  check(zeMemAllocDevice(context, &adesc, in_bytes, 64, device, &din), "alloc in");
  check(zeMemAllocDevice(context, &adesc, out_bytes, 64, device, &dout), "alloc out");
  check(zeKernelSetArgumentValue(kernel, 0, sizeof(din), &din), "arg 0");
  check(zeKernelSetArgumentValue(kernel, 1, sizeof(dout), &dout), "arg 1");
  ze_command_queue_desc_t qdesc = {};
  qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  qdesc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  ze_command_list_handle_t list = nullptr;
  check(zeCommandListCreateImmediate(context, device, &qdesc, &list),
        "zeCommandListCreateImmediate");
  std::vector<u128> result(results * kItems, 0);
  check(zeCommandListAppendMemoryCopy(list, din, in.data(), in_bytes, nullptr,
                                      0, nullptr),
        "copy in");
  ze_group_count_t groups = {kItems, 1, 1};
  check(zeCommandListAppendLaunchKernel(list, kernel, &groups, nullptr, 0,
                                        nullptr),
        "launch");
  check(zeCommandListAppendMemoryCopy(list, result.data(), dout, out_bytes,
                                      nullptr, 0, nullptr),
        "copy out");
  check(zeCommandListHostSynchronize(list, 10'000'000'000ull), "synchronize");
  zeCommandListDestroy(list);
  zeMemFree(context, din);
  zeMemFree(context, dout);
  zeKernelDestroy(kernel);
  zeModuleDestroy(module);
  return result;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::printf("usage: run-mul128 <mul128.spv | ops.spv | vars.spv>\n");
    return 1;
  }
  std::string name = argv[1];
  bool ops = name.find("ops") != std::string::npos;
  bool vars = name.find("vars") != std::string::npos;
  results = ops ? 10 : vars ? 8 : 3;
  std::ifstream file(argv[1], std::ios::binary);
  std::vector<char> spirv((std::istreambuf_iterator<char>(file)), {});
  // Pairs (a, b); the second b has a zero low half.
  const u128 pairs[2 * kItems] = {
      make(0x68a7c603afd47bf5, 0x3c89ed7106ddc391), make(0, 0x8fe78ab543e1de28),
      make(0x4e1596a4aab10aea, 0x158665da61c66bce), make(0x536f853a1686b174, 0),
      make(0, 3), make(0, 5),
      make(0, 0xffffffffffffffff), make(0, 0xffffffffffffffff),
  };
  const u128 mask = ~uint64_t(0);
  const int stride = vars ? 3 : 2;
  std::vector<u128> in;
  for (int i = 0; i < kItems; ++i) {
    in.push_back(pairs[2 * i]);
    in.push_back(pairs[2 * i + 1]);
    // vars.ll reads the mask 2^64 - 1 as a third input.
    if (vars)
      in.push_back(mask);
  }
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

  const char *mulForms[3] = {"a * b", "zext(lo a) * zext(lo b)",
                             "(a & 2^64-1) * zext(lo b)"};
  const char *opForms[10] = {"a & (2^64-1)", "a >> 64", "a << 64",
                             "zext(lo a)", "(a & (2^64-1)) >> 64", "a * b",
                             "zext(lo a) * zext(lo b)", "zext(lo(a * b))",
                             "a ^ (2^64-1)", "a ^ 0x0123..3210"};
  const char *varForms[8] = {"a & m", "(a & m) >> 64", "(a & m) * (b & m)",
                             "a * b", "zext(lo a)", "a + zext(lo a)",
                             "zext(lo a) << 64", "a + b"};
  const u128 c = make(0x0123456789abcdef, 0xfedcba9876543210);
  const char **forms = ops ? opForms : vars ? varForms : mulForms;
  int wrong = 0;
  for (std::string extra : {"", " -cl-opt-disable"}) {
    std::vector<u128> got = run(context, device, spirv, kOptions + extra, in);
    std::printf("options \"%s%s\":\n", kOptions, extra.c_str());
    for (int i = 0; i < kItems; ++i) {
      u128 a = in[stride * i];
      u128 b = in[stride * i + 1];
      u128 lo = a & mask;
      std::vector<u128> want =
          ops    ? std::vector<u128>{lo, a >> 64, a << 64, lo, lo >> 64, a * b,
                                     lo * (b & mask), (a * b) & mask, a ^ mask,
                                     a ^ c}
          : vars ? std::vector<u128>{lo, lo >> 64, lo * (b & mask), a * b, lo,
                                     a + lo, lo << 64, a + b}
                 : std::vector<u128>{a * b, lo * (b & mask), lo * (b & mask)};
      for (int f = 0; f < results; ++f) {
        bool bad = got[results * i + f] != want[f];
        wrong += bad;
        std::printf("  item %d %-26s", i, forms[f]);
        print128(" got", got[results * i + f]);
        print128(" expected", want[f]);
        std::printf("%s\n", bad ? "  <- wrong" : "");
      }
    }
  }
  zeContextDestroy(context);
  std::printf("wrong %d of %d\n", wrong, 2 * results * kItems);
  return wrong == 0 ? 0 : 1;
}
