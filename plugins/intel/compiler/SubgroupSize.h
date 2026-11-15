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
// The sub-group size each kernel is compiled for.
//
// A warp is a sub-group, and MAX's WARP_SIZE for Intel is a compile-time
// constant of the target: 16, or 32 for the targets whose features have
// `+sub-group-size-32` (max/_plugin/intel/targets.mojo). IGC otherwise picks
// SIMD8, 16 or 32 per kernel, which would break every warp primitive, so each
// kernel gets `intel_reqd_sub_group_size` (SPIR-V `OpExecutionMode
// SubgroupSize`) from its function's `target-features`, which Mojo's lowering
// copies from the target. The feature is ours, not LLVM's: it is removed from
// the features before the SPIR-V backend, which has none, sees them.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_COMPILER_SUBGROUPSIZE_H
#define MOJO_INTEL_COMPILER_SUBGROUPSIZE_H

#include "llvm/ADT/StringRef.h"

#include <string>

namespace llvm {
class Function;
class Module;
} // namespace llvm

namespace M::KGEN::Intel {

/// The size a kernel runs at unless its target asks for another one.
constexpr unsigned kDefaultSubgroupSize = 16;

/// The target feature (without its `+`) that asks for sub-group size 32.
constexpr llvm::StringLiteral kSubgroupSize32Feature = "sub-group-size-32";

/// The sub-group size a function with target features `features` (LLVM's
/// comma-separated `+a,-b` form) is compiled for.
unsigned subgroupSizeForFeatures(llvm::StringRef features);

/// `features` without the plugin's own features (enabled or disabled).
std::string withoutPluginFeatures(llvm::StringRef features);

/// Gives each SPIR kernel `intel_reqd_sub_group_size` from its
/// `target-features`, then removes the plugin's features from every
/// function's `target-features`. Returns whether a kernel asked for a size
/// other than the default.
bool setSubgroupSizes(llvm::Module &module);

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_SUBGROUPSIZE_H
