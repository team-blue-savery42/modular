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

#include "Abort.h"

#include "IntelAddressSpace.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ModRef.h"

#include <map>

using namespace llvm;

namespace M::KGEN::Intel {

bool defineAbort(Module &module) {
  Function *abort = module.getFunction(kAbortFunction);
  if (!abort || !abort->isDeclaration())
    return abort != nullptr;

  LLVMContext &ctx = module.getContext();
  constexpr unsigned kCrossWorkgroup =
      static_cast<unsigned>(SPIRVAddressSpace::CrossWorkgroup);
  auto *global = PointerType::get(ctx, kCrossWorkgroup);
  // Externally initialized: the driver plugin writes it after loading the
  // module, so its null initializer must not be folded into loads.
  auto *flag = new GlobalVariable(
      module, global, /*isConstant=*/false, GlobalValue::ExternalLinkage,
      ConstantPointerNull::get(global), kAbortFlagGlobal,
      /*InsertBefore=*/nullptr, GlobalValue::NotThreadLocal, kCrossWorkgroup,
      /*isExternallyInitialized=*/true);
  flag->setAlignment(Align(8));

  // if (flag != null) *flag = 1; a module loaded some other way, with the
  // flag never set, aborts without a report rather than faulting.
  abort->setLinkage(GlobalValue::InternalLinkage);
  BasicBlock *entry = BasicBlock::Create(ctx, "entry", abort);
  BasicBlock *raise = BasicBlock::Create(ctx, "raise", abort);
  BasicBlock *done = BasicBlock::Create(ctx, "done", abort);
  IRBuilder<> builder(entry);
  LoadInst *target = builder.CreateLoad(global, flag, /*isVolatile=*/true);
  target->setAlignment(Align(8));
  builder.CreateCondBr(builder.CreateIsNotNull(target), raise, done);
  builder.SetInsertPoint(raise);
  builder.CreateAlignedStore(builder.getInt32(1), target, Align(4),
                             /*isVolatile=*/true);
  builder.CreateBr(done);
  builder.SetInsertPoint(done);
  builder.CreateRetVoid();
  return true;
}

bool dropUnusedAbortFlag(Module &module) {
  if (Function *abort = module.getFunction(kAbortFunction);
      abort != nullptr && abort->hasLocalLinkage() && abort->use_empty())
    abort->eraseFromParent();
  GlobalVariable *flag = module.getNamedGlobal(kAbortFlagGlobal);
  if (flag == nullptr || !flag->use_empty())
    return false;
  flag->eraseFromParent();
  return true;
}

namespace {

/// The functions that can abort: those that read the abort flag (the abort
/// hook, and any function the optimizer inlined it into), and those that call
/// one of them directly.
SetVector<Function *> abortingFunctions(GlobalVariable &flag) {
  SetVector<Function *> aborting;
  for (User *user : flag.users())
    if (auto *load = dyn_cast<LoadInst>(user))
      aborting.insert(load->getFunction());
  for (size_t i = 0; i < aborting.size(); ++i)
    for (Use &use : aborting[i]->uses())
      if (auto *call = dyn_cast<CallBase>(use.getUser());
          call && call->isCallee(&use))
        aborting.insert(call->getFunction());
  return aborting;
}

/// Whether `func` can take the aborted flag as an added parameter: an
/// internal function only ever called directly.
bool takesAbortedFlag(Function &func) {
  return !func.isDeclaration() && func.hasLocalLinkage() && !func.isVarArg() &&
         func.getCallingConv() != CallingConv::SPIR_KERNEL &&
         all_of(func.uses(), [](Use &use) {
           auto *call = dyn_cast<CallInst>(use.getUser());
           return call && call->isCallee(&use);
         });
}

/// The flag becomes an argument that a function writes when it aborts and
/// reads after a call that can abort: whatever memory a function or call
/// touched before, it may now also read and write argument memory. Without
/// this, an attribute the optimizer inferred, such as memory(readwrite,
/// argmem: none), would let a later pass move the flag's load across the
/// call that sets it.
MemoryEffects withFlagMemory(MemoryEffects effects) {
  return effects | MemoryEffects::argMemOnly(ModRefInfo::ModRef);
}

/// Gives each function in `aborting` that can take one a trailing pointer to
/// its work-item's aborted flag, and every other one (the kernels) the flag
/// itself, a private byte cleared at its entry. Each read of the abort flag
/// global (where a work-item aborts) sets the work-item's flag. Returns the
/// calls that pass the flag on, so that their callers can check it after them.
SmallVector<CallInst *>
threadAbortedFlag(Module &module, GlobalVariable &flag,
                  const SetVector<Function *> &aborting) {
  LLVMContext &ctx = module.getContext();
  unsigned privateSpace = module.getDataLayout().getAllocaAddrSpace();
  auto *pointer = PointerType::get(ctx, privateSpace);
  // Each function's flag pointer, and the new function for each old one.
  std::map<Function *, Value *> flagOf;
  // In `aborting`'s order, not by address: the output must not depend on
  // where the functions happen to live.
  SmallVector<std::pair<Function *, Function *>> replaced;
  for (Function *func : aborting) {
    if (!takesAbortedFlag(*func)) {
      IRBuilder<> entry(&*func->getEntryBlock().getFirstInsertionPt());
      AllocaInst *aborted =
          entry.CreateAlloca(Type::getInt8Ty(ctx), nullptr, "aborted");
      entry.CreateStore(entry.getInt8(0), aborted);
      flagOf[func] = aborted;
      continue;
    }
    SmallVector<Type *> params(func->getFunctionType()->params());
    params.push_back(pointer);
    auto *type = FunctionType::get(func->getReturnType(), params, false);
    Function *copy = Function::Create(type, func->getLinkage(),
                                      func->getAddressSpace(), "", &module);
    copy->copyAttributesFrom(func);
    if (copy->hasFnAttribute(Attribute::Memory))
      copy->setMemoryEffects(withFlagMemory(copy->getMemoryEffects()));
    copy->copyMetadata(func, 0);
    for (auto [from, to] : zip(func->args(), copy->args())) {
      to.takeName(&from);
      from.replaceAllUsesWith(&to);
    }
    copy->splice(copy->end(), func);
    copy->takeName(func);
    Argument *aborted = copy->getArg(copy->arg_size() - 1);
    aborted->setName("aborted");
    flagOf[copy] = aborted;
    replaced.emplace_back(func, copy);
  }

  for (User *user : flag.users())
    if (auto *load = dyn_cast<LoadInst>(user)) {
      IRBuilder<> builder(load->getNextNode());
      builder.CreateStore(builder.getInt8(1), flagOf.at(load->getFunction()));
    }

  SmallVector<CallInst *> checked;
  for (auto &[func, copy] : replaced) {
    for (User *user : make_early_inc_range(func->users())) {
      auto *call = cast<CallInst>(user);
      SmallVector<Value *> args(call->args());
      args.push_back(flagOf.at(call->getFunction()));
      auto *replacement = CallInst::Create(copy->getFunctionType(), copy, args,
                                           "", call->getIterator());
      replacement->setCallingConv(call->getCallingConv());
      AttributeList attrs = call->getAttributes();
      if (attrs.hasFnAttr(Attribute::Memory))
        attrs = attrs.addFnAttribute(
            ctx, Attribute::getWithMemoryEffects(
                     ctx, withFlagMemory(attrs.getMemoryEffects())));
      replacement->setAttributes(attrs);
      replacement->setTailCallKind(call->getTailCallKind());
      replacement->setDebugLoc(call->getDebugLoc());
      replacement->takeName(call);
      call->replaceAllUsesWith(replacement);
      call->eraseFromParent();
      checked.push_back(replacement);
    }
    func->eraseFromParent();
  }
  return checked;
}

} // namespace

bool returnAfterAbort(Module &module) {
  GlobalVariable *flag = module.getNamedGlobal(kAbortFlagGlobal);
  if (flag == nullptr)
    return false;
  LLVMContext &ctx = module.getContext();
  bool changed = false;
  SetVector<Function *> aborting = abortingFunctions(*flag);
  SmallVector<CallInst *> calls = threadAbortedFlag(module, *flag, aborting);
  changed |= !aborting.empty();
  SmallPtrSet<CallInst *, 16> passFlag(calls.begin(), calls.end());

  for (Function &func : module) {
    if (func.isDeclaration())
      continue;
    // `abort` returns (`defineAbort`), and so does everything that calls it.
    if (func.hasFnAttribute(Attribute::NoReturn)) {
      func.removeFnAttr(Attribute::NoReturn);
      changed = true;
    }
    for (Instruction &inst : instructions(func))
      if (auto *call = dyn_cast<CallBase>(&inst);
          call && call->getAttributes().hasFnAttr(Attribute::NoReturn)) {
        call->removeFnAttr(Attribute::NoReturn);
        changed = true;
      }
    // After a call that can abort, a work-item that aborted in it returns:
    // a value it got back is poison. The caller checks its
    // work-item's flag after each such call followed by more than a return
    // or an `unreachable` (which returns too), in program order, so that the
    // blocks and the return value's incoming values are the same on every
    // run.
    SmallVector<CallInst *> checks;
    for (Instruction &inst : instructions(func))
      if (auto *call = dyn_cast<CallInst>(&inst);
          call && passFlag.contains(call) &&
          !isa<ReturnInst, UnreachableInst>(call->getNextNode()))
        checks.push_back(call);
    SmallVector<CondBrInst *> aborts;
    for (CallInst *call : checks) {
      BasicBlock *block = call->getParent();
      BasicBlock *rest =
          block->splitBasicBlock(call->getNextNode(), block->getName());
      block->getTerminator()->eraseFromParent();
      IRBuilder<> builder(block);
      Value *aborted = builder.CreateIsNotNull(builder.CreateLoad(
          builder.getInt8Ty(), call->getArgOperand(call->arg_size() - 1)));
      // The return block is filled in below.
      aborts.push_back(builder.CreateCondBr(aborted, rest, rest));
    }
    SmallVector<UnreachableInst *> unreachables;
    SmallVector<ReturnInst *> returns;
    for (BasicBlock &block : func) {
      Instruction *terminator = block.getTerminator();
      if (auto *unreachable = dyn_cast<UnreachableInst>(terminator))
        unreachables.push_back(unreachable);
      else if (auto *ret = dyn_cast<ReturnInst>(terminator))
        returns.push_back(ret);
    }
    if (unreachables.empty() && aborts.empty())
      continue;
    changed = true;
    // One return for the function: the old returns, every `unreachable` and
    // every check that found the flag raised branch to it, the latter two
    // with poison for the value.
    Type *type = func.getReturnType();
    BasicBlock *exit = BasicBlock::Create(ctx, "return", &func);
    PHINode *value = nullptr;
    if (!type->isVoidTy())
      value = PHINode::Create(
          type, returns.size() + unreachables.size() + aborts.size(), "result",
          exit);
    ReturnInst::Create(ctx, value, exit);
    for (ReturnInst *ret : returns) {
      if (value)
        value->addIncoming(ret->getReturnValue(), ret->getParent());
      UncondBrInst::Create(exit, ret->getIterator());
      ret->eraseFromParent();
    }
    for (UnreachableInst *unreachable : unreachables) {
      if (value)
        value->addIncoming(PoisonValue::get(type), unreachable->getParent());
      UncondBrInst::Create(exit, unreachable->getIterator());
      unreachable->eraseFromParent();
    }
    for (CondBrInst *check : aborts) {
      check->setSuccessor(0, exit);
      if (value)
        value->addIncoming(PoisonValue::get(type), check->getParent());
    }
  }
  return changed;
}

} // namespace M::KGEN::Intel
