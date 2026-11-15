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

#include "ZeApi.h"

#include <dlfcn.h>

#include <cstdio>

namespace M::Intel {
namespace {

struct Loaded {
  ZeApi api;
  std::string error;
  bool ok = false;
};

const Loaded &loaded() {
  static const Loaded state = [] {
    Loaded result;
    void *handle = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      const char *why = dlerror();
      result.error = std::string("cannot load libze_loader.so.1: ") +
                     (why != nullptr ? why : "unknown error");
      return result;
    }
    ZeApi &api = result.api;
    api.handle = handle;
#define MOJO_INTEL_ZE_RESOLVE(name)                                            \
  api.name = reinterpret_cast<decltype(api.name)>(dlsym(handle, #name));       \
  if (api.name == nullptr) {                                                   \
    result.error = "libze_loader.so.1 has no " #name;                          \
    return result;                                                             \
  }
    MOJO_INTEL_ZE_REQUIRED(MOJO_INTEL_ZE_RESOLVE)
#undef MOJO_INTEL_ZE_RESOLVE
#define MOJO_INTEL_ZE_RESOLVE_OPTIONAL(name)                                   \
  api.name = reinterpret_cast<decltype(api.name)>(dlsym(handle, #name));
    MOJO_INTEL_ZE_OPTIONAL(MOJO_INTEL_ZE_RESOLVE_OPTIONAL)
#undef MOJO_INTEL_ZE_RESOLVE_OPTIONAL
    api.hasSysman = true;
#define MOJO_INTEL_ZES_RESOLVE(name)                                           \
  api.name = reinterpret_cast<decltype(api.name)>(dlsym(handle, #name));       \
  api.hasSysman = api.hasSysman && api.name != nullptr;
    MOJO_INTEL_ZES_OPTIONAL(MOJO_INTEL_ZES_RESOLVE)
#undef MOJO_INTEL_ZES_RESOLVE
    result.ok = true;
    return result;
  }();
  return state;
}

} // namespace

const ZeApi *ZeApi::get() { return loaded().ok ? &loaded().api : nullptr; }

const std::string &ZeApi::error() { return loaded().error; }

std::string resultName(ze_result_t result) {
  switch (result) {
  case ZE_RESULT_SUCCESS:
    return "ZE_RESULT_SUCCESS";
  case ZE_RESULT_ERROR_UNINITIALIZED:
    return "ZE_RESULT_ERROR_UNINITIALIZED";
  case ZE_RESULT_ERROR_DEVICE_LOST:
    return "ZE_RESULT_ERROR_DEVICE_LOST";
  case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
    return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
  case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
    return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
  case ZE_RESULT_ERROR_INVALID_ARGUMENT:
    return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
  case ZE_RESULT_ERROR_INVALID_SIZE:
    return "ZE_RESULT_ERROR_INVALID_SIZE";
  case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
    return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
  default: {
    char hex[32];
    std::snprintf(hex, sizeof(hex), "ze_result_t 0x%x",
                  static_cast<unsigned>(result));
    return hex;
  }
  }
}

} // namespace M::Intel
