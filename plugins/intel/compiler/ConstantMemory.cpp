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

#include "ConstantMemory.h"

#include "IntelAddressSpace.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace M::KGEN::Intel {

bool defineConstantImports(Module &module) {
  constexpr unsigned kUniformConstant =
      static_cast<unsigned>(SPIRVAddressSpace::UniformConstant);
  SmallVector<GlobalVariable *> imports;
  for (GlobalVariable &global : module.globals())
    if (global.getAddressSpace() == kUniformConstant && !global.use_empty() &&
        (global.isDeclaration() || isa<UndefValue>(global.getInitializer())))
      imports.push_back(&global);

  for (GlobalVariable *global : imports) {
    Type *type = global->getValueType();
    // A zero of an aggregate or vector is ConstantAggregateZero, and of an
    // integer or pointer an integer or null; a floating-point zero would be
    // a new APFloat, which the plugin must not create (two LLVM copies).
    if (type->isFloatingPointTy())
      type = ArrayType::get(type, 1);
    Constant *zero =
        type->isAggregateType() || type->isVectorTy()
            ? static_cast<Constant *>(ConstantAggregateZero::get(type))
            : Constant::getNullValue(type);
    // Only a name can be exported; a nameless one keeps its linkage.
    bool exported = global->hasName();
    auto *definition = new GlobalVariable(
        module, type, /*isConstant=*/false,
        exported ? GlobalValue::ExternalLinkage : global->getLinkage(), zero,
        "", global, GlobalValue::NotThreadLocal, kUniformConstant,
        /*isExternallyInitialized=*/true);
    definition->setAlignment(global->getAlign());
    definition->setUnnamedAddr(global->getUnnamedAddr());
    definition->takeName(global);
    global->replaceAllUsesWith(definition);
    global->eraseFromParent();
  }
  return !imports.empty();
}

} // namespace M::KGEN::Intel
