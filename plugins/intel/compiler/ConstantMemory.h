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
// Constant memory the host fills. A kernel that names a
// constant-space array, as `unsafe_stack_allocation[..., name="table",
// address_space=.CONSTANT]` does for MAX's `ConstantMemoryMapping`, only
// allocates it: Mojo emits an internal global with an `undef` initializer,
// which on CUDA the host finds by name in the loaded module
// (`cuModuleGetGlobal`) and copies the data into. LLVM's SPIR-V backend
// emits it as a UniformConstant variable with neither an initializer nor a
// linkage, which IGC takes for an import and refuses ("undefined
// reference").
//
// `defineConstantImports` gives each such global zeros and exports it
// under its name, so the module loads and the host can reach the global with
// `zeModuleGetGlobalPointer` and copy into it, as for the abort flag. It stays
// UniformConstant: measured on the B70, a host copy into an exported
// UniformConstant global reaches the kernels launched after it.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_COMPILER_CONSTANTMEMORY_H
#define MOJO_INTEL_COMPILER_CONSTANTMEMORY_H

namespace llvm {
class Module;
} // namespace llvm

namespace M::KGEN::Intel {

/// Defines every used global in the UniformConstant address space (SPIR-V
/// numbering) that has no value, being a declaration or initialized with
/// `undef`: a zero initializer, external linkage (SPIR-V `Export`) if it
/// has a name, not constant and externally initialized, so neither LLVM nor
/// IGC folds the zeros into its loads. A floating-point scalar
/// becomes a one-element array, since its zero would be a new floating-point
/// constant. Run after the address spaces are in SPIR-V numbering. Returns
/// whether the module changed.
bool defineConstantImports(llvm::Module &module);

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_CONSTANTMEMORY_H
