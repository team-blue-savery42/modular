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
// Arrays in kernel arguments. Kernels take Mojo values holding arrays by
// value: closures capturing a `StaticTuple` or `IndexList`, and structs of
// them. IGC accepts no array in a kernel argument. Zero-length arrays are
// worse: Mojo represents an empty tuple, such as `StaticTuple[T, 0]` in an
// empty index list, as `[0 x T]`; LLVM's SPIR-V backend lowers it to
// OpTypeRuntimeArray, which OpenCL SPIR-V doesn't allow, and SPIR-V has no
// zero-length array. Dropping or reshaping fields would change the layout,
// which the host passes byte for byte.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_COMPILER_ZEROLENGTHARRAYS_H
#define MOJO_INTEL_COMPILER_ZEROLENGTHARRAYS_H

#include "llvm/Support/Error.h"

namespace llvm {
class Module;
class Type;
} // namespace llvm

namespace M::KGEN::Intel {

/// Whether `type` holds a zero-length array, at any depth.
bool containsZeroLengthArray(llvm::Type *type);

/// Gives each kernel parameter whose aggregate type holds an array (or is
/// one), holds another aggregate, or holds a `bool` (`i1`) or a bfloat16, a
/// struct of the original's scalar leaves instead, in order, without
/// zero-length arrays, each `bool` as an `i8` and each bfloat16 as an `i16`,
/// padded with single bytes so that every leaf keeps its offset and the
/// struct its size: the bytes the host passes don't change. A `bool` or
/// bfloat16 parameter becomes an `i8` or `i16` too. OpenCL allows no `bool`
/// in a kernel argument, not even inside a struct (clang rejects both), and
/// IGC 2.41 hung compiling a kernel that branched on such a struct field; it
/// read a bfloat16 field of a struct argument as zero. The kernel
/// reads a nonzero byte as true, and bitcasts the bits back to bfloat16.
/// IGC rejects a kernel argument of array type ("Array type is not allowed as
/// a kernel argument"), and reads a struct of scalars passed by value but not
/// one holding a byte array, nor a nested struct (the second field of
/// `{ { i64, i64 }, i64 }` read the first one's bytes). Where the parameter
/// holds a zero-length array, the kernel's `extractvalue`s of it read the
/// leaves, and any other use is an error that changes nothing (the original
/// value has no SPIR-V form); otherwise the original value is rebuilt from the
/// leaves at the kernel's entry, for any use. Run on the optimized module.
/// Returns whether anything changed.
llvm::Expected<bool> rewriteArrayParameters(llvm::Module &module);

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_ZEROLENGTHARRAYS_H
