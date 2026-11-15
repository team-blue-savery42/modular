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
// Renumbers a module's address spaces from Mojo to SPIR-V numbering
// (IntelAddressSpace.h) right before SPIR-V codegen.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_COMPILER_REMAPADDRESSSPACES_H
#define MOJO_INTEL_COMPILER_REMAPADDRESSSPACES_H

#include "llvm/Support/Error.h"

namespace llvm {
class Module;
} // namespace llvm

namespace M::KGEN::Intel {

/// Rewrites every address space in `module` from Mojo to SPIR-V numbering:
/// globals, function signatures, instructions, constants and overloaded
/// intrinsic names. Besides the plain table it:
///  - keeps stack allocations in SPIR-V `Function` storage, casting them to
///    `Generic` where Mojo used a generic pointer, and turns fixed-size
///    shared-memory stack allocations into `Workgroup` globals;
///  - places globals Mojo left in its generic space in `CrossWorkgroup`;
///  - passes generic pointer parameters of `spir_kernel` functions as
///    `CrossWorkgroup` (OpenCL forbids generic kernel arguments), casting them
///    to `Generic` on entry.
/// Fails without changing `module` if it uses an address space Intel GPUs
/// don't have, a stack allocation it can't place, or global aliases or ifuncs.
llvm::Error remapAddressSpacesForSPIRV(llvm::Module &module);

/// Turns constant-expression address-space casts of globals inside function
/// bodies into instructions. LLVM's SPIR-V backend emits such a cast of an
/// array global as an `OpSpecConstantOp PtrCastToGeneric` whose result points
/// to the element type, which `spirv-val` rejects; as an instruction it is
/// emitted correctly. Run on the optimized module, just before codegen: the
/// optimizer folds the instructions back into constants.
void expandGlobalAddressSpaceCasts(llvm::Module &module);

/// Replaces calls to `llvm.stepvector`, which LLVM's SPIR-V backend can't
/// legalize, with the constant vector <0, 1, 2, ...> they produce. Integer
/// constants are safe to create in the plugin (unlike float ones, see
/// PromoteBFloat16.cpp). Returns whether anything changed.
bool expandStepVectors(llvm::Module &module);

/// Makes every pointer moved into or out of an aggregate a byte pointer.
/// LLVM's SPIR-V backend gives a pointer the element type its uses suggest
/// (an array, say), and an aggregate field of pointer type the type i8, and
/// doesn't reconcile the two: spirv-val rejects the OpCompositeExtract or
/// OpCompositeInsert. An i8 GEP by a frozen zero on each such pointer makes
/// both sides i8 (a constant zero GEP would be dropped); IGC folds it.
/// Returns whether anything changed.
bool retypeAggregatePointers(llvm::Module &module);

/// Rewrites GEPs over a zero-length array, `gep [0 x T], p, k, i...`, as
/// `gep T, p, i...` (the first index steps over zero bytes; with no other
/// index the GEP is just `p`). Mojo indexes dynamic shared memory that way
/// (`StaticTuple[T, 0]`), and LLVM's SPIR-V backend would type the pointer as
/// a runtime array, which OpenCL SPIR-V doesn't allow. Returns whether
/// anything changed.
bool flattenZeroLengthArrayGEPs(llvm::Module &module);

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_REMAPADDRESSSPACES_H
