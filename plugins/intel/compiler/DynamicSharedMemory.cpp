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

#include "DynamicSharedMemory.h"

#include "IntelAddressSpace.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/Support/ModRef.h"

#include <algorithm>
#include <string>

using namespace llvm;

namespace M::KGEN::Intel {

namespace {

/// Whether `user` of a function only lists it for the linker (@llvm.used,
/// @llvm.compiler.used, through constants that end there); those lists follow
/// a replaced function. Any other global value, including one that refers to
/// the function directly (an alias, a variable's initializer, a function's
/// personality or prefix) or one listed for the linker itself, is a use of its
/// address. A constant with no users is dead: removeDeadConstantUsers stops
/// at a constant's first live user, so a dead one that sits behind a list
/// entry is still there. It names nothing and follows the replaced function
/// like the entry does, so it passes.
bool onlyInUsedLists(const User *user) {
  if (auto *global = dyn_cast<GlobalVariable>(user))
    return global->getName() == "llvm.used" ||
           global->getName() == "llvm.compiler.used";
  if (isa<GlobalValue>(user))
    return false;
  auto *constant = dyn_cast<Constant>(user);
  return constant && all_of(constant->users(), [](const User *inner) {
           return onlyInUsedLists(inner);
         });
}

/// `func` names the memory itself when `via` is null; otherwise it needs it
/// because it calls `via`.
Error notDirect(const Function &func, const Function *via, const Twine &why) {
  std::string needs = via ? (" calls " + via->getName() +
                             ", which needs dynamic shared memory, but ")
                                .str()
                          : std::string(" uses dynamic shared memory but ");
  return createStringError("intel plugin: " + func.getName() + needs + why +
                           "; only a direct call can pass the memory on");
}

/// Adds every caller of a function in `needs` to it, up to the kernels.
/// Recursion ends because each function is added once.
Error addCallers(SetVector<Function *> &needs) {
  // The callee each added caller came in through, for errors.
  DenseMap<const Function *, const Function *> via;
  for (size_t i = 0; i < needs.size(); ++i) {
    Function *func = needs[i];
    const Function *reason = via.lookup(func);
    func->removeDeadConstantUsers();
    for (Use &use : func->uses()) {
      User *user = use.getUser();
      if (auto *call = dyn_cast<CallBase>(user)) {
        if (!call->isCallee(&use))
          return notDirect(*func, reason, "is passed as an argument");
        if (!isa<CallInst>(call))
          return notDirect(*func, reason, "is invoked other than by a call");
        if (call->getFunctionType() != func->getFunctionType())
          return notDirect(*func, reason, "is called with another signature");
        if (needs.insert(call->getFunction()))
          via[call->getFunction()] = func;
      } else if (!onlyInUsedLists(user)) {
        return notDirect(*func, reason, "its address is taken");
      }
    }
  }
  return Error::success();
}

/// The memory becomes an argument: whatever a function or call did to the
/// global, it may now do to argument memory. Without this, an attribute such
/// as memory(write, argmem: none) would make the new accesses undefined.
MemoryEffects withArgumentMemory(MemoryEffects effects) {
  return effects | MemoryEffects::argMemOnly(effects.getModRef());
}

} // namespace

Error addDynamicSharedMemoryParameter(Module &module) {
  constexpr unsigned kWorkgroup =
      static_cast<unsigned>(SPIRVAddressSpace::Workgroup);
  SmallVector<GlobalVariable *> externs;
  for (GlobalVariable &global : module.globals())
    if (global.getAddressSpace() == kWorkgroup && global.isDeclaration())
      externs.push_back(&global);
  if (externs.empty())
    return Error::success();

  // Uses inside constant expressions become instructions, so every use
  // belongs to a function.
  SmallVector<Constant *> constants(externs.begin(), externs.end());
  convertUsersOfConstantsToInstructions(constants);

  SetVector<Function *> needs;
  Align alignment(1);
  for (GlobalVariable *global : externs) {
    global->removeDeadConstantUsers();
    alignment = std::max(alignment, global->getAlign().valueOrOne());
    for (User *user : global->users()) {
      auto *inst = dyn_cast<Instruction>(user);
      if (!inst)
        return createStringError("intel plugin: dynamic shared memory @" +
                                 global->getName() +
                                 " is used outside a function");
      needs.insert(inst->getFunction());
    }
  }
  if (Error err = addCallers(needs))
    return err;

  LLVMContext &ctx = module.getContext();
  auto *shared = PointerType::get(ctx, kWorkgroup);
  DenseMap<Function *, Function *> copies;
  for (Function *func : needs) {
    SmallVector<Type *> params(func->getFunctionType()->params());
    params.push_back(shared);
    auto *type =
        FunctionType::get(func->getReturnType(), params, func->isVarArg());
    Function *copy = Function::Create(type, func->getLinkage(),
                                      func->getAddressSpace(), "", &module);
    copy->copyAttributesFrom(func);
    copy->copyMetadata(func, 0);
    copy->splice(copy->end(), func);
    for (auto [from, to] : zip(func->args(), copy->args())) {
      from.replaceAllUsesWith(&to);
      to.takeName(&from);
    }
    Argument *memory = copy->getArg(copy->arg_size() - 1);
    memory->setName("dynamic_shared_memory");
    copy->addParamAttr(memory->getArgNo(),
                       Attribute::getWithAlignment(ctx, alignment));
    if (copy->hasFnAttribute(Attribute::Memory))
      copy->setMemoryEffects(withArgumentMemory(copy->getMemoryEffects()));
    copies[func] = copy;
  }

  // The bodies have moved, so a call's function is already a copy, whose
  // last argument is the pointer to pass on.
  for (Function *func : needs) {
    Function *copy = copies[func];
    for (Use &use : make_early_inc_range(func->uses())) {
      auto *call = dyn_cast<CallInst>(use.getUser());
      if (!call)
        continue; // @llvm.used, which follows the copy below.
      Function *caller = call->getFunction();
      SmallVector<Value *> args(call->args());
      args.push_back(caller->getArg(caller->arg_size() - 1));
      SmallVector<OperandBundleDef> bundles;
      call->getOperandBundlesAsDefs(bundles);
      CallInst *replacement =
          CallInst::Create(copy->getFunctionType(), copy, args, bundles, "",
                           call->getIterator());
      replacement->takeName(call);
      replacement->setCallingConv(call->getCallingConv());
      replacement->setTailCallKind(call->getTailCallKind());
      AttributeList attrs = call->getAttributes().addParamAttribute(
          ctx, args.size() - 1, Attribute::getWithAlignment(ctx, alignment));
      if (attrs.hasFnAttr(Attribute::Memory))
        attrs = attrs.addFnAttribute(
            ctx, Attribute::getWithMemoryEffects(
                     ctx, withArgumentMemory(attrs.getMemoryEffects())));
      replacement->setAttributes(attrs);
      replacement->copyMetadata(*call);
      call->replaceAllUsesWith(replacement);
      call->eraseFromParent();
    }
  }

  for (Function *func : needs) {
    Function *copy = copies[func];
    Argument *memory = copy->getArg(copy->arg_size() - 1);
    for (GlobalVariable *global : externs)
      global->replaceUsesWithIf(memory, [&](Use &use) {
        return cast<Instruction>(use.getUser())->getFunction() == copy;
      });
    copy->takeName(func);
    // References such as @llvm.used now name the copy.
    func->replaceAllUsesWith(copy);
    func->eraseFromParent();
  }
  for (GlobalVariable *global : externs)
    if (global->use_empty())
      global->eraseFromParent();
  return Error::success();
}

} // namespace M::KGEN::Intel
