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
// Aggregate phis. IGC loses values merged by a phi of a nested aggregate:
// the index of `for i, byte in enumerate(s.bytes())`, an i64 inside the
// iterator's `Optional` (a phi of `{ { i64, [8 x i8] }, i8 }` whose "none"
// side is `{ undef, 0 }`), is `undef` with IGC's defaults and 0 with
// -cl-opt-disable. IGC never sees such a phi:
// it becomes one scalar phi per leaf.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_COMPILER_AGGREGATEPHIS_H
#define MOJO_INTEL_COMPILER_AGGREGATEPHIS_H

namespace llvm {
class Module;
} // namespace llvm

namespace M::KGEN::Intel {

/// Replaces each phi of a struct or array type (at most 64 scalar leaves) by
/// one phi per scalar leaf, in the same block. Each incoming value's leaf is
/// the value inserted at that place when the incoming value is an
/// `insertvalue` chain, the leaf phi when it is another split phi, the
/// constant's own element when it is a constant that holds it (an undef or
/// poison leaf of an undef or poison constant; a zero integer or pointer of
/// a zero constant), and otherwise an `extractvalue` at the end of the
/// incoming block: no floating-point constant is created or folded. An
/// `extractvalue` of a whole leaf takes the leaf phi; any other use (a part
/// or the whole aggregate, stored, passed, returned) gets that part rebuilt
/// from the leaves with `insertvalue`s on poison. Run last before codegen,
/// after the passes that add phis (`returnAfterAbort`). Returns whether
/// anything changed.
bool splitAggregatePhis(llvm::Module &module);

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_AGGREGATEPHIS_H
