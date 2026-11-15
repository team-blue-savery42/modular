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
// Dynamic shared memory for Intel GPUs. Mojo's `external_memory` declares an
// unsized external global in shared memory, sized per launch, as CUDA's
// `extern __shared__` arrays are. OpenCL SPIR-V has no such globals: a kernel
// takes per-launch local memory as a `__local` pointer parameter, which the
// driver sizes with `zeKernelSetArgumentValue(index, size, nullptr)`.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_COMPILER_DYNAMICSHAREDMEMORY_H
#define MOJO_INTEL_COMPILER_DYNAMICSHAREDMEMORY_H

#include "llvm/Support/Error.h"

namespace llvm {
class Module;
} // namespace llvm

namespace M::KGEN::Intel {

/// Replaces the module's external `Workgroup` globals (in SPIR-V numbering)
/// with a trailing `ptr addrspace(3)` parameter of each function that uses
/// them, and of every function that calls one of those, up to the kernels:
/// each call passes the caller's pointer on. Every such global aliases the
/// same memory, as in CUDA. A function called both from kernels that use the
/// memory and from ones that don't gets one signature: all its callers take
/// the pointer. The Level Zero driver plugin passes the launch's dynamic
/// shared memory size for a kernel with one argument more than the launch
/// gives. Errors when a function that needs the pointer is reached other
/// than by a direct call (an indirect call can't pass it).
llvm::Error addDynamicSharedMemoryParameter(llvm::Module &module);

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_DYNAMICSHAREDMEMORY_H
