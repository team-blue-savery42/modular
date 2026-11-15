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
// bfloat16 arithmetic for Intel GPUs. SPV_KHR_bfloat16 gives SPIR-V the
// bfloat16 type and conversions to and from other float types, but no
// arithmetic, comparisons or OpenCL math on it (that is
// SPV_INTEL_bfloat16_arithmetic, which LLVM's SPIR-V backend otherwise asks
// for). Like targets without bfloat16 ALUs, compute it in float32.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_COMPILER_PROMOTEBFLOAT16_H
#define MOJO_INTEL_COMPILER_PROMOTEBFLOAT16_H

namespace llvm {
class Module;
} // namespace llvm

namespace M::KGEN::Intel {

/// Rewrites every floating-point operation on bfloat16 values (arithmetic,
/// comparisons, math intrinsics, and int conversions) to extend its operands
/// to float32, compute in float32, and truncate a bfloat16 result back. For
/// +, -, *, / and sqrt this gives the correctly rounded bfloat16 result, since
/// float32 carries more than twice bfloat16's precision. Then moves every
/// value that is or holds bfloat16 (selects, phis, loads, stores, vector and
/// aggregate moves) as i16, with bitcasts at the edges, so that only
/// conversions to and from other float types see bfloat16: IGC gets bfloat16
/// selects and phis wrong. Last, converts
/// between bfloat16 and float32 with integer operations (round to nearest
/// even, a NaN becomes the quiet NaN with its sign, bit for bit as the host
/// converts) instead of fpext and fptrunc, so that no conversion to or from
/// bfloat16 is left for IGC to get wrong or to narrow arithmetic around;
/// float64 goes through float32 both ways, as on the
/// host. The conversions create no floating-point constants and no intrinsic
/// declarations. Run on the optimized module, just
/// before codegen: the optimizer would fold the pairs away. Returns whether
/// anything changed.
bool promoteBFloat16(llvm::Module &module);

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_PROMOTEBFLOAT16_H
