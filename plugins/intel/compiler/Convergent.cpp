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

#include "Convergent.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace M::KGEN::Intel {
namespace {

/// The source name of an Itanium-mangled built-in, `_Z<length><name>...`,
/// or `name` itself if it isn't mangled that way.
StringRef unmangledName(StringRef name) {
  StringRef rest = name;
  if (!rest.consume_front("_Z"))
    return name;
  size_t digits = 0;
  while (digits < rest.size() && isDigit(rest[digits]))
    ++digits;
  unsigned length = 0;
  if (digits == 0 || rest.take_front(digits).getAsInteger(10, length) ||
      length > rest.size() - digits)
    return name;
  return rest.substr(digits, length);
}

} // namespace

bool isConvergentBuiltin(StringRef name) {
  StringRef base = unmangledName(name);
  return base == "barrier" || base == "work_group_barrier" ||
         base.starts_with("sub_group_") || base.starts_with("work_group_") ||
         base.starts_with("intel_sub_group_") ||
         base.starts_with("__spirv_ControlBarrier") ||
         base.starts_with("__spirv_Group") ||
         base.starts_with("__spirv_Subgroup");
}

bool markConvergentFunctions(Module &module) {
  bool changed = false;
  for (Function &func : module)
    if (func.isDeclaration() && !func.isIntrinsic() && !func.isConvergent() &&
        isConvergentBuiltin(func.getName())) {
      func.setConvergent();
      changed = true;
    }
  // Callers, to a fixed point: each round marks at least one more function.
  bool grew = true;
  while (grew) {
    grew = false;
    for (Function &func : module) {
      if (func.isDeclaration() || func.isConvergent())
        continue;
      for (Instruction &inst : instructions(func)) {
        auto *call = dyn_cast<CallBase>(&inst);
        if (call && call->isConvergent()) {
          func.setConvergent();
          grew = changed = true;
          break;
        }
      }
    }
  }
  return changed;
}

} // namespace M::KGEN::Intel
