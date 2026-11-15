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
// LLVM-level codegen for Intel SPIR-V targets. Before optimization the module's
// address spaces are renumbered for SPIR-V and the accelerator arch,
// which LLVM's SPIR-V backend doesn't know, is dropped. Assembly and SPIR-V
// binaries come from LLVM's in-tree SPIR-V backend through the host's llc.
// The whole module is one SPIR-V module: no splitting, no linking.
//
//===----------------------------------------------------------------------===//

#include "IntelBackend.h"

#include "IntelAddressSpace.h"
#include "IntelTraits.h"
#include "RemapAddressSpaces.h"
#include "SPIRVBinary.h"

#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Target/Host/HostBackend.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>

namespace M::KGEN::Intel {
namespace {

/// Named metadata carrying a failure from `finalizeModuleForTarget`, which
/// can't return one, to the emit hooks, which can.
constexpr llvm::StringLiteral kErrorMetadata = "mojo.intel.error";

/// The error recorded on `module`, if any.
std::optional<std::string> recordedError(const llvm::Module &module) {
  const llvm::NamedMDNode *node = module.getNamedMetadata(kErrorMetadata);
  if (!node || node->getNumOperands() == 0)
    return std::nullopt;
  const llvm::MDNode *entry = node->getOperand(0);
  if (entry->getNumOperands() == 0)
    return std::string("intel plugin: unknown error");
  if (auto *text = llvm::dyn_cast<llvm::MDString>(entry->getOperand(0)))
    return text->getString().str();
  return std::string("intel plugin: unknown error");
}

/// SPIR-V binaries start with this magic number (little endian on disk).
constexpr uint32_t kSPIRVMagic = 0x07230203;

bool isSPIRVBinary(const WriteableBuffer &buffer) {
  if (buffer.getBufferSize() < sizeof(uint32_t))
    return false;
  uint32_t word;
  std::memcpy(&word, buffer.getBufferStart(), sizeof(word));
  return word == kSPIRVMagic;
}

class IntelBackend final : public HostBackend {
public:
  const TargetTraits *traits() const override { return &IntelTraits::get(); }

  bool isOffload() const override { return true; }

  // One SPIR-V module per compilation: kernels and the device functions they
  // call must be in the same module, and SPIR-V objects aren't linked here.
  SplitStrategy splitStrategy(const CompilationOptions &) const override {
    return SplitStrategy::None;
  }
  bool isCodegenInterprocedural() const override { return true; }

  // The accelerator arch (e.g. `bmg-g31`) names a device for the Intel
  // toolchain after SPIR-V emission; LLVM's SPIR-V backend has no CPUs and
  // warns about any it is given.
  CompilationOptions
  adjustOptionsForTargetMachine(const CompilationOptions &options,
                                llvm::StringRef) const override {
    CompilationOptions adjusted = options;
    adjusted.targetCpu.clear();
    return adjusted;
  }

  void finalizeModuleForTarget(llvm::Module &module, llvm::TargetMachine &,
                               llvm::StringRef) const override {
    for (llvm::Function &func : module) {
      // The lowering copies the arch into every function's attributes too.
      func.removeFnAttr("target-cpu");
      func.removeFnAttr("tune-cpu");
      // The device has no C library: keep the optimizer from turning calls
      // into libc calls, such as `printf("%s\n", s)` into `puts(s)`, which
      // the driver can't link. OpenCL's own built-ins aren't LLVM libcalls.
      func.addFnAttr("no-builtins");
    }
    if (llvm::Error err = remapAddressSpacesForSPIRV(module)) {
      std::string message = llvm::toString(std::move(err));
      trace("backend: address-space renumbering failed: " + message);
      llvm::LLVMContext &ctx = module.getContext();
      module.getOrInsertNamedMetadata(kErrorMetadata)
          ->addOperand(
              llvm::MDNode::get(ctx, llvm::MDString::get(ctx, message)));
      return;
    }
    trace("backend: renumbered address spaces for SPIR-V");
  }

  ErrorOr<BufferRef> emitAssembly(llvm::Module &module,
                                  EmitContext &ctx) const override {
    if (std::optional<std::string> error = recordedError(module))
      return Error(*error);
    if (llvm::Error err = prepareForCodegen(module))
      return Error(llvm::toString(std::move(err)));
    trace("backend: emitAssembly via host llc");
    dumpBitcode(module);
    return HostBackend::emitAssembly(module, ctx);
  }

  /// A SPIR-V binary from LLVM's SPIR-V backend. Unlike the host path this
  /// links nothing: the binary is the device artifact.
  ErrorOr<BufferRef> emitObject(llvm::Module &module,
                                EmitContext &ctx) const override {
    if (std::optional<std::string> error = recordedError(module))
      return Error(*error);
    if (llvm::Error err = prepareForCodegen(module))
      return Error(llvm::toString(std::move(err)));
    trace("backend: emitObject via host llc");
    dumpBitcode(module);
    WriteableBufferRef binary = WriteableBuffer::get();
    if (ErrorOrSuccess err =
            ctx.runLlc(module, *binary, /*createObjectFile=*/true))
      return Error(llvm::Twine(err.getError()) +
                   ", llc failed to emit a SPIR-V binary");
    if (!isSPIRVBinary(*binary))
      return Error("intel plugin: llc output is not a SPIR-V binary");
    renameExtensions(binary->getBufferStart(), binary->getBufferSize());
    dump(module, ".spv", [&](llvm::raw_ostream &os) {
      os.write(binary->getBufferStart(), binary->getBufferSize());
    });
    return BufferRef::take(binary.release());
  }

  ErrorOr<BufferRef> createArchive(llvm::MutableArrayRef<BufferRef> objects,
                                   llvm::StringRef,
                                   EmitContext &) const override {
    // splitStrategy() is None, so there is exactly one SPIR-V module.
    if (objects.size() != 1)
      return Error("intel plugin: expected one SPIR-V module, got " +
                   llvm::Twine(objects.size()));
    return objects[0];
  }

protected:
  /// The rewrites that must see the optimized module, just before llc.
  static llvm::Error prepareForCodegen(llvm::Module &module) {
    expandStepVectors(module);
    expandGlobalAddressSpaceCasts(module);
    retypeAggregatePointers(module);
    flattenZeroLengthArrayGEPs(module);
    return llvm::Error::success();
  }

  /// With MOJO_INTEL_PLUGIN_DUMP=<dir>, writes each module the plugin emits
  /// as bitcode (before llc) and each SPIR-V binary to
  /// <dir>/<module name>.<n>.bc and .spv, for checking every kernel a program
  /// compiles. `n` numbers the files, since one program can compile several
  /// modules of the same name. The plugin can't print the module's IR:
  /// printing float constants runs this copy of LLVM's APFloat on the host's
  /// constants ("unknown format" assertion, see PromoteBFloat16.cpp). Mojo's
  /// `emission_kind="llvm-opt"` prints it from the host instead.
  template <typename WriteFn>
  static void dump(const llvm::Module &module, llvm::StringRef extension,
                   WriteFn write) {
    std::optional<std::string> dir =
        llvm::sys::Process::GetEnv("MOJO_INTEL_PLUGIN_DUMP");
    if (!dir)
      return;
    static std::atomic<unsigned> counter{0};
    llvm::SmallString<256> path(*dir);
    llvm::sys::path::append(path, llvm::sys::path::filename(module.getName()) +
                                      "." + llvm::Twine(counter++) + extension);
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
    if (ec) {
      trace(("backend: cannot write " + path + ": " + ec.message()).str());
      return;
    }
    write(os);
    trace(("backend: wrote " + path).str());
  }

  /// The module as bitcode, before llc, which rejects a module it can't emit
  /// (`llvm-dis` shows it).
  static void dumpBitcode(const llvm::Module &module) {
    dump(module, ".bc",
         [&](llvm::raw_ostream &os) { llvm::WriteBitcodeToFile(module, os); });
  }

  std::optional<unsigned> sharedMemoryAddressSpace() const override {
    // Shared memory is 3 in both Mojo and SPIR-V numbering (Workgroup).
    return static_cast<unsigned>(MojoAddressSpace::Shared);
  }

  bool isBaseTarget() const override { return false; }
};

} // namespace

const TargetBackend &getBackend() {
  static const IntelBackend instance;
  return instance;
}

} // namespace M::KGEN::Intel
