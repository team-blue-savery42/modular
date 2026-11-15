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
// Marks OpenCL's work-group and sub-group built-ins, and the functions that
// call them, `convergent` before LLVM's optimizer runs.
//
// The MAX hook (max/_plugin/intel) reaches barriers, sub-group barriers,
// shuffles and ballots through `external_call` of the built-ins' mangled
// names (`_Z7barrierj`, ...), which Mojo declares as plain functions. (The
// stdlib hook calls only `get_sub_group_local_id`, a query, which stays
// unmarked.) LLVM may then make a call to one control
// dependent on more values than before: at -O3 SimpleLoopUnswitch copies a
// loop that holds a barrier once per value of a loop-invariant condition, so
// the work-items of one work-group (even of one sub-group) wait at different
// barriers, and a barrier no longer orders shared memory. NVPTX
// and AMDGPU barriers are convergent intrinsics, and clang marks every
// OpenCL function convergent; this gives Mojo's Intel modules the same
// guarantee for the built-ins that need it.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_COMPILER_CONVERGENT_H
#define MOJO_INTEL_COMPILER_CONVERGENT_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class Module;
} // namespace llvm

namespace M::KGEN::Intel {

/// Whether `name` (a mangled OpenCL or SPIR-V friendly built-in name) is a
/// built-in that all work-items of a work-group or sub-group must reach
/// together: `barrier`, `work_group_barrier`, `sub_group_*`, `work_group_*`,
/// `intel_sub_group_*`, `__spirv_ControlBarrier`, `__spirv_Group*` and
/// `__spirv_Subgroup*`. Queries such as `get_sub_group_local_id` aren't.
bool isConvergentBuiltin(llvm::StringRef name);

/// Marks each declared convergent built-in `convergent`, then each defined
/// function that calls a convergent function, until nothing changes (a call
/// is convergent when its callee is). Run before LLVM's optimizer. Returns
/// whether anything changed.
bool markConvergentFunctions(llvm::Module &module);

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_CONVERGENT_H
