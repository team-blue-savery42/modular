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
// Fix-ups of the SPIR-V binary LLVM's backend emits, for what Intel's driver
// compiler (IGC) reads.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_COMPILER_SPIRVBINARY_H
#define MOJO_INTEL_COMPILER_SPIRVBINARY_H

#include <cstddef>

namespace M::KGEN::Intel {

/// Renames extensions LLVM spells differently from IGC, in place. LLVM calls
/// arbitrary-precision integers (`i128` and other widths)
/// SPV_ALTERA_arbitrary_precision_integers; IGC knows the extension only as
/// SPV_INTEL_arbitrary_precision_integers, its earlier name, and rejects the
/// module otherwise. The capability's number didn't change, and both names
/// take ten words, so only the `OpExtension` string changes. `data` must
/// hold whole little-endian words. Returns the number of names changed.
unsigned renameExtensions(char *data, size_t size);

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_SPIRVBINARY_H
