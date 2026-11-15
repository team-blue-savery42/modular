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
// 128-bit integers. IGC computes 128-bit integer arithmetic wrongly: a
// 128-bit constant operand keeps only its low 32 bits and adds lose their
// carry, with its defaults and with -cl-opt-disable (IGC
// 2.41.5), and IGC master asserts on any `i128`. Mojo's float
// formatting computes with `UInt128` (dragonbox), so `print` of a float64
// showed wrong digits. IGC never sees an `i128`: each one becomes
// two `i64` halves, low and high.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_COMPILER_INT128_H
#define MOJO_INTEL_COMPILER_INT128_H

#include "llvm/Support/Error.h"

namespace llvm {
class Module;
} // namespace llvm

namespace M::KGEN::Intel {

/// Rewrites every scalar `i128` value of the module's functions as a pair of
/// `i64` halves: `add` and `sub` (with carry and borrow), `mul` (the low 128
/// bits, from 32 x 32 -> 64 partial products), `and`, `or`, `xor`, `shl`,
/// `lshr` and `ashr` (by a constant or a variable amount), `icmp`, `zext` and
/// `sext` to `i128`, `trunc` from it, `select`, `freeze`, phis, constants,
/// bitcasts to and from 128-bit vectors, and loads and stores (two `i64`
/// accesses, the high one 8 bytes further). Globals, allocas and GEPs whose
/// types hold `i128`s get the type with each one as `[2 x i64]`, which the
/// data layout must lay out the same (float formatting's table becomes a
/// table of pairs). An internal function called only directly gets two `i64`
/// parameters for each `i128` one and returns `{ i64, i64 }` for an `i128`.
/// Only integer constants are made, and no intrinsic is declared. Anything
/// else that holds or computes on an `i128` (division, a conversion to or
/// from a float, a vector of `i128`, an aggregate value holding one, a
/// kernel or external function taking one) is refused with an error naming
/// it, so that no `i128` reaches IGC unnoticed. Run after the passes that add
/// instructions, just before codegen. Returns whether anything changed.
llvm::Expected<bool> lowerInt128(llvm::Module &module);

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_INT128_H
