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

#include "IntelTraits.h"

#include "IntelAddressSpace.h"

#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

namespace M::KGEN::Intel {

bool traceEnabled() {
  static const bool enabled =
      llvm::sys::Process::GetEnv("MOJO_INTEL_PLUGIN_TRACE").has_value();
  return enabled;
}

void trace(llvm::StringRef message) {
  if (traceEnabled())
    llvm::errs() << "[intel-plugin] " << message << "\n";
}

const IntelTraits &IntelTraits::get() {
  static const IntelTraits instance;
  return instance;
}

// Arch names follow `ocloc -device`.
static constexpr TargetTraits::AcceleratorArch kIntelAcceleratorArchs[] = {
    {"bmg-g31", "Battlemage Xe2 (Arc Pro B70)"},
    {"bmg-g21", "Battlemage Xe2 (Arc B580, Arc Pro B60)"},
};

llvm::ArrayRef<TargetTraits::AcceleratorArch>
IntelTraits::supportedAcceleratorArchs() const {
  return kIntelAcceleratorArchs;
}

std::optional<unsigned>
IntelTraits::requiredStackAllocationAddressSpace() const {
  return static_cast<unsigned>(MojoAddressSpace::Local);
}

} // namespace M::KGEN::Intel
