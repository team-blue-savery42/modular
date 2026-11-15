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

#ifndef MOJO_INTEL_COMPILER_INTELLOWERING_H
#define MOJO_INTEL_COMPILER_INTELLOWERING_H

namespace M::KGEN {
class TargetLowering;
} // namespace M::KGEN

namespace M::KGEN::Intel {

/// The plugin's lowering, returned by `M_KGEN_getTargetLowering`.
const TargetLowering &getLowering();

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_INTELLOWERING_H
