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
// HAL bundles, functions and kernel launch on Level Zero. A bundle is a SPIR-V
// module (zeModuleCreate; IGC compiles it for the device at load), a function
// a kernel in it, and a launch appends to the queue's in-order list with no
// event of its own: ordering comes from the list, and the HAL
// records events only when asked.
//
//===----------------------------------------------------------------------===//

#include "Plugin.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using M::Intel::fail;
using M::Intel::failZe;
using M::Intel::ZeApi;

namespace {

/// IGC options for every module. Auto GRF lets IGC give a kernel with high
/// register pressure the large register file (256 registers, half the
/// threads per EU) and keep 128 elsewhere. Without it, a kernel that spills
/// heavily can take IGC more than 30 minutes (256-wide float32 conversions:
/// 3 s with it), and spills cost at run time. ../utils/check-spirv
/// compiles with the same options. Float32 division and square root are
/// correctly rounded, as LLVM defines them (OpenCL, IGC's default, allows 2.5
/// and 3 ulp); MAX's libm and its tests rely on it.
constexpr const char *kBuildFlags = "-ze-intel-enable-auto-large-GRF-mode "
                                    "-cl-fp32-correctly-rounded-divide-sqrt";

/// `kBuildFlags`, then MOJO_INTEL_IGC_OPTIONS when it is set: extra IGC
/// options for debugging, such as `-cl-opt-disable` or `-ze-opt-level=1`.
std::string buildFlags() {
  std::string flags = kBuildFlags;
  if (const char *extra = std::getenv("MOJO_INTEL_IGC_OPTIONS");
      extra != nullptr && *extra != '\0')
    flags.append(" ").append(extra);
  return flags;
}

/// SPIR-V binaries start with this word (little endian on disk).
constexpr uint32_t kSPIRVMagic = 0x07230203;

const ZeApi &ze(const M_driver_context &context) {
  return *context.device->driver->ze;
}

M_driver_driver &driverOf(const M_driver_context &context) {
  return *context.device->driver;
}

/// The entry points in a SPIR-V module whose last parameter is a `Workgroup`
/// pointer. Host code can't pass one, so it's the dynamic shared memory the
/// Intel compiler plugin appends (DynamicSharedMemory.h). Level Zero has no
/// query for argument types, so the module is read directly.
std::set<std::string> dynamicSharedKernels(const char *data, size_t size) {
  constexpr uint32_t kOpEntryPoint = 15, kOpTypePointer = 32, kOpFunction = 54,
                     kOpFunctionParameter = 55;
  constexpr uint32_t kWorkgroup = 4;
  std::vector<uint32_t> words(size / 4);
  std::memcpy(words.data(), data, words.size() * 4);
  std::unordered_set<uint32_t> workgroupPointers;
  std::unordered_map<uint32_t, std::string> entryPoints;
  std::unordered_map<uint32_t, uint32_t> lastParameterType;
  uint32_t function = 0;
  // Instructions follow the five-word header; each starts with its word
  // count (high half) and opcode (low half).
  for (size_t i = 5; i < words.size();) {
    uint32_t count = words[i] >> 16, opcode = words[i] & 0xFFFF;
    if (count == 0 || i + count > words.size())
      break;
    const uint32_t *op = &words[i];
    if (opcode == kOpEntryPoint && count >= 4) {
      // Execution model, function, then the nul-terminated name.
      const char *name = reinterpret_cast<const char *>(op + 3);
      entryPoints[op[2]] = std::string(name, strnlen(name, (count - 3) * 4));
    } else if (opcode == kOpTypePointer && count >= 4 && op[2] == kWorkgroup) {
      workgroupPointers.insert(op[1]);
    } else if (opcode == kOpFunction && count >= 3) {
      function = op[2];
    } else if (opcode == kOpFunctionParameter && count >= 3) {
      lastParameterType[function] = op[1];
    }
    i += count;
  }
  std::set<std::string> kernels;
  for (const auto &[id, name] : entryPoints) {
    auto last = lastParameterType.find(id);
    if (last != lastParameterType.end() &&
        workgroupPointers.count(last->second))
      kernels.insert(name);
  }
  return kernels;
}

/// The global the Intel compiler plugin exports from a module that can abort
/// (its Abort.h).
constexpr const char *kAbortFlagGlobal = "__mojo_intel_abort_flag";

/// The names the SPIR-V module exports.
std::vector<std::string> exportedNames(const char *data, size_t size) {
  constexpr uint32_t kOpDecorate = 71, kLinkageAttributes = 41, kExport = 0;
  std::vector<std::string> names;
  std::vector<uint32_t> words(size / 4);
  std::memcpy(words.data(), data, words.size() * 4);
  for (size_t i = 5; i < words.size();) {
    uint32_t count = words[i] >> 16, opcode = words[i] & 0xFFFF;
    if (count == 0 || i + count > words.size())
      break;
    const uint32_t *op = &words[i];
    // OpDecorate target LinkageAttributes "name" linkage-type.
    if (opcode == kOpDecorate && count >= 5 && op[2] == kLinkageAttributes &&
        op[count - 1] == kExport) {
      const char *name = reinterpret_cast<const char *>(op + 3);
      names.emplace_back(name, strnlen(name, (count - 4) * 4));
    }
    i += count;
  }
  return names;
}

/// The global the Intel compiler plugin exports from a module that allocates
/// (its Heap.h): a pointer to the context's heap.
constexpr const char *kHeapGlobal = "__mojo_intel_heap";
/// The heap's header, before its first block (Heap.h `kHeapHeaderBytes`):
/// the bytes handed out so far (i64), then the bytes after the header (i64),
/// then the free-list heads, all zero to start with.
constexpr uint64_t kHeapHeaderBytes = 512;
/// The heap's size unless MOJO_INTEL_HEAP_SIZE says otherwise: CUDA's
/// default (`cudaLimitMallocHeapSize`, 8 MiB).
constexpr uint64_t kDefaultHeapBytes = 8 << 20;

/// Whether the SPIR-V module exports `name`. Level Zero can't say: for a
/// module without the global, zeModuleGetGlobalPointer returns
/// ZE_RESULT_ERROR_UNSUPPORTED_FEATURE, which a real failure could return too.
bool exports(const std::vector<std::string> &exported, const char *name) {
  return std::find(exported.begin(), exported.end(), name) != exported.end();
}

/// The largest heap MOJO_INTEL_HEAP_SIZE may ask for. The compiler plugin's
/// malloc adds a block's size to an offset into the heap (Heap.cpp): both
/// stay under 2^62 here, so the sum can't wrap. No device has this much.
constexpr uint64_t kMaxHeapBytes = uint64_t(1) << 62;

/// The heap's size in bytes: MOJO_INTEL_HEAP_SIZE (bytes, decimal digits
/// only) when it is set, the default otherwise: what a context's kernels can
/// hold at once (freed blocks are reused by size class). False, with
/// `error` set, for a value that isn't a number from 1 to 2^62, set but
/// empty included: strtoull alone would take "-1" for 2^64 - 1 and saturate
/// on overflow, and a size near 2^64 wrapped the allocation's size, leaving
/// malloc a capacity past the memory it had.
bool heapBytes(uint64_t &bytes, std::string &error) {
  const char *text = std::getenv("MOJO_INTEL_HEAP_SIZE");
  if (text == nullptr) {
    bytes = kDefaultHeapBytes;
    return true;
  }
  bool digits = true;
  for (const char *c = text; *c != '\0'; ++c)
    digits &= *c >= '0' && *c <= '9';
  errno = 0;
  char *end = nullptr;
  unsigned long long value = digits ? std::strtoull(text, &end, 10) : 0;
  if (!digits || errno == ERANGE || end == nullptr || *end != '\0' ||
      value == 0 || value > kMaxHeapBytes) {
    error = std::string("MOJO_INTEL_HEAP_SIZE is \"") + text +
            "\": it must be a number of bytes from 1 to 2^62";
    return false;
  }
  bytes = value;
  return true;
}

/// The functions and globals `module` uses that nothing defines, as IGC
/// found after linking its built-ins (which include OpenCL's, by their
/// mangled names, such as `_Z3expf`); nothing else links a bundle's single
/// module. Empty if the driver can't inspect linkage. Its log repeats lines
/// and lists exports too, as `Module <0x...>:  Unresolved Imported Symbol
/// <malloc>`.
std::vector<std::string> unresolvedImports(const ZeApi &api,
                                           ze_module_handle_t module) {
  std::vector<std::string> names;
  if (api.zeModuleInspectLinkageExt == nullptr)
    return names;
  ze_linkage_inspection_ext_desc_t desc = {
      ZE_STRUCTURE_TYPE_LINKAGE_INSPECTION_EXT_DESC, nullptr,
      ZE_LINKAGE_INSPECTION_EXT_FLAG_UNRESOLVABLE_IMPORTS};
  ze_module_build_log_handle_t log = nullptr;
  if (api.zeModuleInspectLinkageExt(&desc, 1, &module, &log) !=
      ZE_RESULT_SUCCESS)
    return names;
  size_t size = 0;
  std::string text;
  if (api.zeModuleBuildLogGetString(log, &size, nullptr) == ZE_RESULT_SUCCESS) {
    text.resize(size);
    if (api.zeModuleBuildLogGetString(log, &size, text.data()) !=
        ZE_RESULT_SUCCESS)
      text.clear();
  }
  api.zeModuleBuildLogDestroy(log);
  constexpr std::string_view kMarker = "Unresolved Imported Symbol <";
  for (size_t at = text.find(kMarker); at != std::string::npos;
       at = text.find(kMarker, at)) {
    at += kMarker.size();
    size_t end = text.find('>', at);
    if (end == std::string::npos)
      break;
    std::string name = text.substr(at, end - at);
    if (std::find(names.begin(), names.end(), name) == names.end())
      names.push_back(std::move(name));
  }
  return names;
}

/// Copies `size` bytes from the host to `dst` in device memory and waits.
M_driver_status_t copyToDevice(M_driver_context &context, void *dst,
                               const void *src, size_t size,
                               const std::string &what) {
  const ZeApi &api = ze(context);
  std::lock_guard<std::mutex> lock(context.syncMutex);
  ze_result_t result = api.zeCommandListAppendMemoryCopy(
      context.syncList, dst, src, size, nullptr, 0, nullptr);
  if (result == ZE_RESULT_SUCCESS)
    result = api.zeCommandListHostSynchronize(context.syncList, UINT64_MAX);
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(context), what.c_str(), result);
  return M_DRIVER_STATUS_SUCCESS;
}

/// Sets the pointer-sized global `name` that `module` exports to `value`:
/// the abort flag or the heap, which kernels reach through module globals.
M_driver_status_t pointGlobalAt(M_driver_context &context,
                                ze_module_handle_t module, const char *name,
                                void *value) {
  size_t size = 0;
  void *global = nullptr;
  ze_result_t result =
      ze(context).zeModuleGetGlobalPointer(module, name, &size, &global);
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(context), "zeModuleGetGlobalPointer", result);
  if (size != sizeof(void *))
    return fail(driverOf(context),
                std::string(name) + " is not a pointer-sized global");
  return copyToDevice(context, global, &value, sizeof(value),
                      std::string("setting ") + name);
}

/// The context's heap, created the first time a module that allocates
/// loads: device memory with a header saying nothing is handed out
/// yet, resident for the context's lifetime, since kernels reach it through a
/// module global that Level Zero doesn't see (as the abort flag).
M_driver_status_t heapOf(M_driver_context &context, void **heap) {
  std::lock_guard<std::mutex> lock(context.heapMutex);
  if (context.heap == nullptr) {
    const ZeApi &api = ze(context);
    uint64_t bytes = 0;
    std::string error;
    if (!heapBytes(bytes, error))
      return fail(driverOf(context), error);
    uint64_t total = kHeapHeaderBytes + bytes;
    ze_device_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                       nullptr, 0, 0};
    void *memory = nullptr;
    ze_result_t result =
        api.zeMemAllocDevice(context.zeContext, &desc, total, kHeapHeaderBytes,
                             context.device->zeDevice, &memory);
    if (result == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
      return fail(driverOf(context),
                  "no device memory for the " + std::to_string(bytes) +
                      "-byte device heap (MOJO_INTEL_HEAP_SIZE)");
    if (result != ZE_RESULT_SUCCESS)
      return failZe(driverOf(context), "zeMemAllocDevice (device heap)",
                    result);
    result = api.zeContextMakeMemoryResident(
        context.zeContext, context.device->zeDevice, memory, total);
    if (result != ZE_RESULT_SUCCESS) {
      api.zeMemFree(context.zeContext, memory);
      return failZe(driverOf(context),
                    "zeContextMakeMemoryResident (device heap)", result);
    }
    uint64_t header[kHeapHeaderBytes / sizeof(uint64_t)] = {0, bytes};
    if (M_driver_status_t status = copyToDevice(
            context, memory, header, sizeof(header), "initializing the heap")) {
      api.zeMemFree(context.zeContext, memory);
      return status;
    }
    context.device->allocatedBytes += total;
    context.heap = memory;
    context.heapBytes = total;
  }
  *heap = context.heap;
  return M_DRIVER_STATUS_SUCCESS;
}

/// `log` with each kernel name IGC quotes (`in kernel 'NAME': reason`)
/// shortened to its first and last characters. Mojo's kernel names run to
/// thousands of characters, and MAX's client reads a status message into
/// 1 KiB, so the reason after a full name never reached the user.
std::string compactBuildLog(const std::string &log) {
  constexpr std::string_view kStart = "in kernel '", kEnd = "': ";
  constexpr size_t kKeep = 60;
  std::string out;
  size_t at = 0;
  while (true) {
    size_t start = log.find(kStart, at);
    if (start == std::string::npos)
      break;
    start += kStart.size();
    size_t end = log.find(kEnd, start);
    if (end == std::string::npos)
      break;
    out.append(log, at, start - at);
    if (end - start > 2 * kKeep + 3)
      out.append(log, start, kKeep)
          .append("...")
          .append(log, end - kKeep, kKeep);
    else
      out.append(log, start, end - start);
    at = end;
  }
  return out.append(log, at, std::string::npos);
}

/// The file types the plugin loads: MAX's HAL compat layer sends "object" for
/// targets other than CUDA and HIP.
bool acceptedFileType(const char *data, uint64_t length) {
  std::string type(data != nullptr ? data : "", data != nullptr ? length : 0);
  return type == "object" || type == "spv" || type == "spirv";
}

} // namespace

// `opts` (debug level, optimization level) is not mapped to IGC build flags
// yet; the module builds with the driver's defaults.
M_driver_status_t M_driver_bundle_load(
    M_driver_context_t context, M_driver_static_bundle_t bundle,
    [[maybe_unused]] M_driver_bundle_compilation_options_t opts,
    M_driver_runtime_bundle_t *runtime_bundle) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (bundle == nullptr || bundle->mapped_data.data == nullptr)
    return M_DRIVER_STATUS_BAD_BUNDLE;
  if (runtime_bundle == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (!acceptedFileType(bundle->file_type, bundle->file_type_len))
    return M_DRIVER_STATUS_INCOMPATIBLE;
  const M_driver_slice &data = bundle->mapped_data;
  uint32_t magic = 0;
  if (data.size < 20 || data.size % 4 != 0)
    return M_DRIVER_STATUS_BAD_BUNDLE;
  std::memcpy(&magic, data.data, sizeof(magic));
  if (magic != kSPIRVMagic)
    return M_DRIVER_STATUS_INCOMPATIBLE;

  ze_module_desc_t desc = {};
  desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  desc.inputSize = data.size;
  desc.pInputModule = reinterpret_cast<const uint8_t *>(data.data);
  std::string flags = buildFlags();
  desc.pBuildFlags = flags.c_str();
  const ZeApi &api = ze(*context);
  ze_module_build_log_handle_t log = nullptr;
  auto loaded = std::make_unique<M_driver_runtime_bundle>();
  ze_module_handle_t module = nullptr;
  ze_result_t result = api.zeModuleCreate(
      context->zeContext, context->device->zeDevice, &desc, &module, &log);
  std::string buildLog;
  if (log != nullptr) {
    size_t size = 0;
    if (result != ZE_RESULT_SUCCESS &&
        api.zeModuleBuildLogGetString(log, &size, nullptr) ==
            ZE_RESULT_SUCCESS &&
        size > 1) {
      buildLog.resize(size);
      api.zeModuleBuildLogGetString(log, &size, buildLog.data());
      buildLog.resize(std::strlen(buildLog.c_str()));
    }
    api.zeModuleBuildLogDestroy(log);
  }
  if (result != ZE_RESULT_SUCCESS)
    return fail(driverOf(*context),
                std::string("zeModuleCreate: ") + M::Intel::resultName(result) +
                    (buildLog.empty() ? "" : "\n" + compactBuildLog(buildLog)));
  loaded->module = std::make_shared<LoadedModule>(api, module);
  // The module builds anyway; zeKernelCreate would fail with
  // ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED and no names.
  if (std::vector<std::string> missing = unresolvedImports(api, module);
      !missing.empty()) {
    std::string names;
    for (const std::string &name : missing)
      names += (names.empty() ? "" : ", ") + name;
    return fail(driverOf(*context),
                "the kernels call functions nothing defines: " +
                    names); // `loaded` destroys the module
  }
  loaded->dynamicSharedKernels = dynamicSharedKernels(data.data, data.size);
  loaded->module->exported = exportedNames(data.data, data.size);
  const std::vector<std::string> &exported = loaded->module->exported;
  // On failure `loaded` destroys the module.
  if (exports(exported, kAbortFlagGlobal))
    if (M_driver_status_t status = pointGlobalAt(
            *context, module, kAbortFlagGlobal, context->abortFlag))
      return status;
  if (exports(exported, kHeapGlobal)) {
    void *heap = nullptr;
    if (M_driver_status_t status = heapOf(*context, &heap))
      return status;
    if (M_driver_status_t status =
            pointGlobalAt(*context, module, kHeapGlobal, heap))
      return status;
  }
  *runtime_bundle = loaded.release();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_bundle_unload(M_driver_context_t context,
                                         M_driver_runtime_bundle_t bundle) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (bundle == nullptr)
    return M_DRIVER_STATUS_BAD_BUNDLE;
  std::unique_ptr<M_driver_runtime_bundle> owned(bundle);
  // With no function loaded from it left, the module goes now, and a failure
  // is reported; otherwise the last function's unload destroys it.
  if (owned->module.use_count() > 1)
    return M_DRIVER_STATUS_SUCCESS;
  ze_result_t result = ze(*context).zeModuleDestroy(
      std::exchange(owned->module->handle, nullptr));
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(*context), "zeModuleDestroy", result);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_function_load(M_driver_context_t context,
                                         M_driver_runtime_bundle_t bundle,
                                         const char *function_name,
                                         uint64_t function_name_len,
                                         M_driver_function_t *function) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (bundle == nullptr)
    return M_DRIVER_STATUS_BAD_BUNDLE;
  if (function_name == nullptr || function == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  std::string name(function_name, function_name_len);
  ze_kernel_desc_t desc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                           name.c_str()};
  auto loaded = std::make_unique<M_driver_function>();
  loaded->context = context;
  loaded->module = bundle->module;
  ze_result_t result = ze(*context).zeKernelCreate(bundle->module->handle,
                                                   &desc, &loaded->kernel);
  if (result == ZE_RESULT_ERROR_INVALID_KERNEL_NAME)
    return M_DRIVER_STATUS_UNKNOWN_SYMBOL;
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(*context), "zeKernelCreate", result);
  ze_kernel_properties_t properties = {ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES};
  result = ze(*context).zeKernelGetProperties(loaded->kernel, &properties);
  if (result != ZE_RESULT_SUCCESS) {
    ze(*context).zeKernelDestroy(loaded->kernel);
    return failZe(driverOf(*context), "zeKernelGetProperties", result);
  }
  loaded->numArgs = properties.numKernelArgs;
  loaded->dynamicShared = bundle->dynamicSharedKernels.count(name) != 0;
  loaded->profileId =
      M::Intel::Profiling::functionId(function_name, function_name_len);
  *function = loaded.release();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_function_unload(M_driver_context_t context,
                                           M_driver_function_t function) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (function == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  std::unique_ptr<M_driver_function> owned(function);
  ze_result_t result = ze(*context).zeKernelDestroy(owned->kernel);
  if (result != ZE_RESULT_SUCCESS)
    return failZe(driverOf(*context), "zeKernelDestroy", result);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_function_property(M_driver_context_t context,
                                             M_driver_function_t function,
                                             const char *property_name,
                                             void *value) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (function == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  if (property_name == nullptr || value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (std::strcmp(property_name, M_DRIVER_FUNCTION_PROPERTY_NATIVE_HANDLE) ==
      0) {
    *static_cast<uint64_t *>(value) =
        reinterpret_cast<uint64_t>(function->module->handle);
    return M_DRIVER_STATUS_SUCCESS;
  }
  // "global_address:NAME" and "global_size:NAME" (uint64): the device
  // address and size of the global NAME that the function's module exports,
  // such as the constant memory a kernel imports by name (the compiler
  // plugin defines it, ConstantMemory.h). Constant memory mappings fill it
  // with memory_wrap and a copy, as cuModuleGetGlobal and a copy do on CUDA.
  // A module that doesn't export NAME reports
  // UNKNOWN_PROPERTY_NAME; any other failure is an error, so a caller that
  // skips a missing global never skips one a lost device hid.
  constexpr std::string_view kAddress = "global_address:";
  constexpr std::string_view kSize = "global_size:";
  std::string_view property(property_name);
  bool wantsAddress = property.substr(0, kAddress.size()) == kAddress;
  if (wantsAddress || property.substr(0, kSize.size()) == kSize) {
    std::string name(
        property.substr(wantsAddress ? kAddress.size() : kSize.size()));
    if (!exports(function->module->exported, name.c_str()))
      return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
    size_t size = 0;
    void *global = nullptr;
    ze_result_t result =
        ze(*function->context)
            .zeModuleGetGlobalPointer(function->module->handle, name.c_str(),
                                      &size, &global);
    if (result != ZE_RESULT_SUCCESS)
      return failZe(driverOf(*context),
                    ("zeModuleGetGlobalPointer(" + name + ")").c_str(), result);
    if (global == nullptr)
      return fail(driverOf(*context),
                  "zeModuleGetGlobalPointer(" + name + ") returned no address");
    *static_cast<uint64_t *>(value) =
        wantsAddress ? reinterpret_cast<uint64_t>(global) : uint64_t(size);
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_driver_status_t
M_driver_queue_execute(M_driver_queue_t queue, M_driver_function_t function,
                       M_driver_queue_execute_config_t *config, void **args,
                       uint64_t *arg_sizes, uint32_t num_args) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  uint64_t enter =
      queue->profile != nullptr ? M::Intel::Profiling::now() : uint64_t(0);
  if (function == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  if (config == nullptr ||
      (num_args > 0 && (args == nullptr || arg_sizes == nullptr)))
    return M_DRIVER_STATUS_INVALID_ARG;
  // A kernel can only run on command lists of the context it was loaded in.
  if (function->context != queue->context)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  if (config->mode != M_DRIVER_QUEUE_EXECUTE_MODE_GPU)
    return M_DRIVER_STATUS_EXECUTE_MODE_NOT_SUPPORTED;
  const M_driver_queue_execute_config_gpu_t &gpu = config->config.gpu;
  // Launch attributes are CUDA's (`CUlaunchAttribute`s, such as programmatic
  // dependent launch); Level Zero has none of them.
  if (gpu.num_attributes > 0)
    return M_DRIVER_STATUS_UNSUPPORTED;
  // A kernel that uses dynamic shared memory takes it as one more argument
  // than the launch passes; one that doesn't ignores the size, as in CUDA.
  bool dynamicShared = function->dynamicShared;
  if (function->numArgs != num_args + (dynamicShared ? 1 : 0))
    return M_DRIVER_STATUS_INVALID_ARG;
  const M_driver_context &context = *queue->context;
  const ZeApi &api = ze(context);

  std::lock_guard<std::mutex> functionLock(function->mutex);
  if (gpu.block.x != function->groupX || gpu.block.y != function->groupY ||
      gpu.block.z != function->groupZ) {
    ze_result_t result = api.zeKernelSetGroupSize(function->kernel, gpu.block.x,
                                                  gpu.block.y, gpu.block.z);
    if (result != ZE_RESULT_SUCCESS) {
      // The sizes name the cause (a work-group over the device's limit).
      std::string what = "zeKernelSetGroupSize(" + std::to_string(gpu.block.x) +
                         ", " + std::to_string(gpu.block.y) + ", " +
                         std::to_string(gpu.block.z) + ")";
      return failZe(driverOf(context), what.c_str(), result);
    }
    function->groupX = gpu.block.x;
    function->groupY = gpu.block.y;
    function->groupZ = gpu.block.z;
  }
  for (uint32_t i = 0; i < num_args; ++i) {
    ze_result_t result = api.zeKernelSetArgumentValue(
        function->kernel, i, static_cast<size_t>(arg_sizes[i]), args[i]);
    if (result != ZE_RESULT_SUCCESS)
      return fail(driverOf(context), std::string("zeKernelSetArgumentValue(") +
                                         std::to_string(i) +
                                         "): " + M::Intel::resultName(result));
  }
  if (dynamicShared) {
    // Level Zero sizes a local-memory argument with a null value. It can't
    // be empty; a kernel launched without any reads none.
    size_t bytes = std::max<size_t>(gpu.shared_mem_bytes, 1);
    ze_result_t result = api.zeKernelSetArgumentValue(function->kernel,
                                                      num_args, bytes, nullptr);
    if (result != ZE_RESULT_SUCCESS)
      return failZe(driverOf(context),
                    "zeKernelSetArgumentValue(dynamic shared memory)", result);
  }
  ze_group_count_t groups = {gpu.grid.x, gpu.grid.y, gpu.grid.z};
  std::lock_guard<std::mutex> queueLock(queue->mutex);
  M::Intel::Profiling::QueueProfile *profile = queue->profile.get();
  ze_event_handle_t signal =
      profile != nullptr && !queue->capturing ? profile->nextEvent() : nullptr;
  ze_result_t result = api.zeCommandListAppendLaunchKernel(
      queue->list, function->kernel, &groups, signal, 0, nullptr);
  if (result != ZE_RESULT_SUCCESS) {
    if (profile != nullptr)
      profile->abandonEvent();
    return failZe(driverOf(context), "zeCommandListAppendLaunchKernel", result);
  }
  ++queue->appended;
  if (profile != nullptr) {
    uint32_t grid[3] = {gpu.grid.x, gpu.grid.y, gpu.grid.z};
    uint32_t block[3] = {gpu.block.x, gpu.block.y, gpu.block.z};
    profile->kernel(function->profileId, grid, block, enter,
                    M::Intel::Profiling::now());
  }
  return M_DRIVER_STATUS_SUCCESS;
}
