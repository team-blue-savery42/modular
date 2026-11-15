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
// Nested struct arguments of internal functions. IGC gets nested by-value
// struct arguments wrong between functions as it does for kernels
// (`rewriteArrayParameters`): for a constant `{ i64, i64, { ptr, i64 } }` its
// vISA wrote the last field over the pointer, and the callee read a
// `SourceLocation`'s file name from address 0x35. It gets
// nested struct return values wrong too: `List._realloc` returns
// `{ { ptr, i64 }, i64 }`, and a kernel growing a List stored through a null
// pointer.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_COMPILER_NESTEDSTRUCTARGUMENTS_H
#define MOJO_INTEL_COMPILER_NESTEDSTRUCTARGUMENTS_H

namespace llvm {
class Module;
} // namespace llvm

namespace M::KGEN::Intel {

/// Passes each by-value argument of an internal function that is a struct
/// holding another struct (and no array) as a flat struct of its scalar
/// leaves, in order. Callers build the flat struct (a constant one stays a
/// constant); the function reads each leaf from it once, at its entry, and
/// its `extractvalue`s of the argument take those leaves. Any other use of
/// the argument or of a part of it (passing it whole to a call, storing it,
/// a phi) gets the part rebuilt from its leaves at the entry, after them. A
/// flat struct, not one parameter per leaf: with the pointer a parameter of
/// its own, IGC cloned the function per address space of its callers'
/// pointers and silently dropped the kernel. Functions called other than
/// directly, variadic ones and kernels are left alone. Such a function that
/// returns a struct holding another struct (and no array) returns the flat
/// struct of its leaves instead, and each caller reads the leaves from it
/// right after the call, as the function reads a flattened argument's. Run
/// on the optimized module. Returns whether anything changed.
bool flattenNestedStructArguments(llvm::Module &module);

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_NESTEDSTRUCTARGUMENTS_H
