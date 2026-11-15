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
// Per-target metadata for Intel GPUs reached through OpenCL-flavored SPIR-V
// (`spirv64-unknown-unknown`). Served to the compiler through the plugin entry
// point `M_KGEN_getTargetTraits`; never registered as a builtin target.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_COMPILER_INTELTRAITS_H
#define MOJO_INTEL_COMPILER_INTELTRAITS_H

#include "Target/TargetTraits.h"

#include "llvm/TargetParser/Triple.h"

namespace M::KGEN::Intel {

/// Whether `MOJO_INTEL_PLUGIN_TRACE` is set; entry points log to stderr then.
bool traceEnabled();

/// Logs `message` with a plugin prefix when tracing is enabled.
void trace(llvm::StringRef message);

struct IntelTraits final : TargetTraits {
  llvm::StringRef name() const override { return "intel"; }
  bool isGPU() const override { return true; }
  bool matches(const llvm::Triple &triple) const override {
    return triple.getArch() == llvm::Triple::spirv64;
  }

  llvm::StringRef getAsmExtension() const override { return ".spvasm"; }
  llvm::StringRef getLLVMExtension() const override { return ".spirv.ll"; }
  llvm::StringRef getObjectExtension() const override { return ".spv"; }
  llvm::StringRef getBitcodeExtension() const override { return ".bc"; }
  llvm::ArrayRef<EmissionKind> supportedEmissionKinds() const override {
    return commonEmissionKinds();
  }

  llvm::StringRef acceleratorSectionTitle() const override {
    return "Intel (SPIR-V, plugin):";
  }
  llvm::ArrayRef<AcceleratorArch> supportedAcceleratorArchs() const override;

  /// Stack allocations live in Mojo's LOCAL address space, as on NVPTX and
  /// AMDGPU. The value is in Mojo numbering: it is compared against
  /// `pop.stack_allocation` before lowering, and becomes SPIR-V `Function`
  /// (the backend's alloca address space) when address spaces are renumbered.
  std::optional<unsigned> requiredStackAllocationAddressSpace() const override;

  // `defaultCPU` stays empty: LLVM's SPIR-V backend knows no CPU names and
  // ignores one with a warning, so the accelerator arch (e.g. `bmg-g31`) is
  // only meaningful to the Intel toolchain after SPIR-V emission.

  /// Shared stateless instance for the lowering's and backend's `traits()`.
  static const IntelTraits &get();

protected:
  // Accelerator target, gated on MAX like the builtin GPU targets.
  bool isBaseTarget() const override { return false; }
};

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_INTELTRAITS_H
