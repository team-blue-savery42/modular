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
// Lane masks bitcast to odd-width integers. LLVM tests whether any or all
// lanes of a `<N x i1>` mask are set by bitcasting it to `iN` and comparing
// that with a constant, and counts them with `ctpop` of it; for N = 2 or 4
// (Mojo's float32 and float64 SIMD width on the B70) the SPIR-V backend emits
// an `i2` or `i4` (SPV_INTEL_arbitrary_precision_integers), on which IGC
// crashes or loops.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_COMPILER_MASKBITCASTS_H
#define MOJO_INTEL_COMPILER_MASKBITCASTS_H

#include "llvm/Support/Error.h"

namespace llvm {
class Module;
} // namespace llvm

namespace M::KGEN::Intel {

/// Removes each `bitcast <N x i1> %m to iN` with N not 8, 16, 32 or 64,
/// rewriting its users on the lanes of %m (lane 0 is bit 0):
/// - `icmp eq|ne` with a constant C: the lanes compared with the bits of C
///   and combined with `and`, negated for `ne`;
/// - another `icmp` with a constant, `zext` and `sext`: the same on the mask
///   built as an integer of a legal width from its lanes;
/// - `bitcast` back to the mask's type: %m;
/// - `llvm.ctpop`, when its users are `zext`s or unsigned compares with a
///   constant: the sum of the lanes.
/// Any other user is an error, before anything reaches IGC. Returns whether
/// anything changed.
llvm::Expected<bool> rewriteMaskBitcasts(llvm::Module &module);

/// Replaces each `trunc <N x iM> %x to <N x i1>` with
/// `icmp ne (and %x, 1), 0`, the same lanes. LLVM's SPIR-V backend lowers
/// such a truncation to an `OpBitwiseAnd` whose operands it types wrongly for
/// some shapes (`<8 x i32>`: "VReg's low-level type and register class have
/// different sizes", MAX's `apply_packed_bitmask`), and stops.
/// Returns whether anything changed.
bool expandMaskTruncs(llvm::Module &module);

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_MASKBITCASTS_H
