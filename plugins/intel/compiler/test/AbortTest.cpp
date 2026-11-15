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
// Tests for the abort contract with the Level Zero driver plugin.
//
//===----------------------------------------------------------------------===//

#include "Abort.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

using namespace llvm;
using M::KGEN::Intel::defineAbort;
using M::KGEN::Intel::dropUnusedAbortFlag;
using M::KGEN::Intel::kAbortFlagGlobal;
using M::KGEN::Intel::kAbortFunction;
using M::KGEN::Intel::returnAfterAbort;

namespace {

std::unique_ptr<Module> parse(LLVMContext &ctx, const std::string &body) {
  SMDiagnostic diag;
  std::unique_ptr<Module> module = parseAssemblyString(
      "target triple = \"spirv64-unknown-unknown\"\n" + body, diag, ctx);
  if (!module) {
    std::string text;
    raw_string_ostream os(text);
    diag.print("test", os);
    ADD_FAILURE() << text;
  }
  return module;
}

std::string print(const Module &module) {
  std::string text;
  raw_string_ostream os(text);
  module.print(os, nullptr);
  return text;
}

TEST(AbortTest, DefinesTheHookAndExportsTheFlag) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %p) {
  call void @__mojo_intel_abort()
  unreachable
}
declare void @__mojo_intel_abort()
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(defineAbort(*module));
  std::string text;
  raw_string_ostream os(text);
  ASSERT_FALSE(verifyModule(*module, &os)) << text << print(*module);

  GlobalVariable *flag = module->getNamedGlobal(kAbortFlagGlobal);
  ASSERT_NE(flag, nullptr) << print(*module);
  // The driver finds it by name and writes it after loading the module.
  EXPECT_EQ(flag->getLinkage(), GlobalValue::ExternalLinkage);
  EXPECT_TRUE(flag->isExternallyInitialized());
  EXPECT_EQ(flag->getAddressSpace(), 1u);
  EXPECT_EQ(flag->getValueType()->getPointerAddressSpace(), 1u);

  Function *abort = module->getFunction(kAbortFunction);
  ASSERT_FALSE(abort->isDeclaration());
  EXPECT_TRUE(abort->hasInternalLinkage());
  bool storesOne = false;
  for (Instruction &inst : instructions(*abort))
    if (auto *store = dyn_cast<StoreInst>(&inst))
      storesOne = store->isVolatile() &&
                  isa<ConstantInt>(store->getValueOperand()) &&
                  cast<ConstantInt>(store->getValueOperand())->isOne();
  EXPECT_TRUE(storesOne) << print(*module);
}

// After optimization removed every call to the abort hook, the hook and the
// flag go; while a call remains, both stay.
TEST(AbortTest, UnusedFlagIsDropped) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
declare void @__mojo_intel_abort()
define spir_kernel void @k(ptr addrspace(1) %p) {
  ret void
}
)");
  ASSERT_TRUE(module);
  ASSERT_TRUE(defineAbort(*module));
  ASSERT_NE(module->getNamedGlobal(kAbortFlagGlobal), nullptr);
  EXPECT_TRUE(dropUnusedAbortFlag(*module));
  EXPECT_EQ(module->getNamedGlobal(kAbortFlagGlobal), nullptr);
  EXPECT_EQ(module->getFunction(kAbortFunction), nullptr);
  EXPECT_FALSE(verifyModule(*module, &errs()));

  std::unique_ptr<Module> used = parse(ctx, R"(
declare void @__mojo_intel_abort()
define spir_kernel void @k(ptr addrspace(1) %p) {
  call void @__mojo_intel_abort()
  ret void
}
)");
  ASSERT_TRUE(used);
  ASSERT_TRUE(defineAbort(*used));
  EXPECT_FALSE(dropUnusedAbortFlag(*used));
  EXPECT_NE(used->getNamedGlobal(kAbortFlagGlobal), nullptr);
}

// Aborting returns, so a work-item that aborts deep in a call chain must
// unwind to the kernel: every function with an `unreachable` gets one return
// block that its returns and `unreachable`s branch to, and nothing is
// `noreturn` any more (IGC emitted nothing for the `unreachable`
// after a call to the stdlib's assertion reporter, and the work-item ran on
// into the next block).
TEST(AbortTest, AbortingCallChainsReturn) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define internal void @report(ptr %message) noreturn {
  call void @__mojo_intel_abort()
  unreachable
}
define internal i32 @checked(i32 %i) {
  %ok = icmp ult i32 %i, 4
  br i1 %ok, label %fine, label %bad
fine:
  ret i32 %i
bad:
  call void @report(ptr null) noreturn
  unreachable
}
define internal float @fails(i1 %c) noreturn {
  br i1 %c, label %a, label %b
a:
  call void @report(ptr null)
  unreachable
b:
  call void @report(ptr null)
  unreachable
}
define spir_kernel void @k(ptr addrspace(1) %p, i32 %i) {
  %v = call i32 @checked(i32 %i)
  store i32 %v, ptr addrspace(1) %p
  %c = icmp eq i32 %v, 0
  br i1 %c, label %fail, label %done
fail:
  %f = call float @fails(i1 %c) noreturn
  unreachable
done:
  ret void
}
declare void @__mojo_intel_abort() noreturn
)");
  ASSERT_TRUE(module);
  ASSERT_TRUE(defineAbort(*module));
  ASSERT_FALSE(dropUnusedAbortFlag(*module));
  EXPECT_TRUE(returnAfterAbort(*module));
  std::string text;
  raw_string_ostream os(text);
  ASSERT_FALSE(verifyModule(*module, &os)) << text << print(*module);

  for (Function &func : *module) {
    EXPECT_FALSE(func.hasFnAttribute(Attribute::NoReturn))
        << func.getName().str();
    unsigned returns = 0;
    for (Instruction &inst : instructions(func)) {
      EXPECT_FALSE(isa<UnreachableInst>(inst)) << print(*module);
      returns += isa<ReturnInst>(inst);
      if (auto *call = dyn_cast<CallBase>(&inst))
        EXPECT_FALSE(call->getAttributes().hasFnAttr(Attribute::NoReturn))
            << print(*module);
    }
    if (!func.isDeclaration())
      EXPECT_EQ(returns, 1u) << print(*module);
  }
  // A function that returns a value returns its own where it did, and
  // poison where it aborted.
  auto *ret =
      cast<ReturnInst>(module->getFunction("checked")->back().getTerminator());
  auto *result = cast<PHINode>(ret->getReturnValue());
  ASSERT_EQ(result->getNumIncomingValues(), 2u) << print(*module);
  EXPECT_TRUE(isa<Argument>(result->getIncomingValue(0)));
  EXPECT_TRUE(isa<PoisonValue>(result->getIncomingValue(1)));
}

// A work-item that aborted in a callee returns from its caller right after
// the call, rather than going on with the poison the callee handed back
// (an index or reference from a failed bounds check faulted). Each
// function that can abort takes a pointer to its work-item's flag, the
// kernel holds the flag, and aborting sets it.
TEST(AbortTest, CallersReturnAfterACalleeAborted) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define internal ptr addrspace(1) @element(ptr addrspace(1) %p, i64 %i) {
  %ok = icmp ult i64 %i, 4
  br i1 %ok, label %fine, label %bad
fine:
  %e = getelementptr i32, ptr addrspace(1) %p, i64 %i
  ret ptr addrspace(1) %e
bad:
  call void @__mojo_intel_abort()
  unreachable
}
define internal i32 @twice(ptr addrspace(1) %p, i64 %i) {
  %e = call ptr addrspace(1) @element(ptr addrspace(1) %p, i64 %i)
  %v = load i32, ptr addrspace(1) %e
  %w = add i32 %v, %v
  ret i32 %w
}
define spir_kernel void @k(ptr addrspace(1) %p, i64 %i) {
  %w = call i32 @twice(ptr addrspace(1) %p, i64 %i)
  store i32 %w, ptr addrspace(1) %p
  ret void
}
define internal i32 @pure(i32 %x) {
  ret i32 %x
}
declare void @__mojo_intel_abort() noreturn
)");
  ASSERT_TRUE(module);
  ASSERT_TRUE(defineAbort(*module));
  ASSERT_FALSE(dropUnusedAbortFlag(*module));
  EXPECT_TRUE(returnAfterAbort(*module));
  std::string text;
  raw_string_ostream os(text);
  ASSERT_FALSE(verifyModule(*module, &os)) << text << print(*module);

  // The kernel holds the flag, cleared at its entry.
  Function *kernel = module->getFunction("k");
  auto *aborted = dyn_cast<AllocaInst>(&kernel->getEntryBlock().front());
  ASSERT_NE(aborted, nullptr) << print(*module);
  auto *clear = dyn_cast<StoreInst>(aborted->getNextNode());
  ASSERT_NE(clear, nullptr) << print(*module);
  EXPECT_EQ(clear->getPointerOperand(), aborted);
  EXPECT_TRUE(cast<ConstantInt>(clear->getValueOperand())->isZero());

  // Every function that can abort takes the flag last; the others don't.
  for (const char *name : {"element", "twice", kAbortFunction}) {
    Function *func = module->getFunction(name);
    ASSERT_NE(func, nullptr) << name << print(*module);
    EXPECT_TRUE(func->getFunctionType()
                    ->getParamType(func->arg_size() - 1)
                    ->isPointerTy())
        << name;
    for (User *user : func->users())
      EXPECT_EQ(cast<CallInst>(user)->getArgOperand(func->arg_size() - 1),
                cast<CallInst>(user)->getFunction() == kernel
                    ? static_cast<Value *>(aborted)
                    : cast<CallInst>(user)->getFunction()->getArg(
                          cast<CallInst>(user)->getFunction()->arg_size() - 1))
          << print(*module);
  }
  EXPECT_EQ(module->getFunction("pure")->arg_size(), 1u);

  // Aborting sets the flag.
  Function *abort = module->getFunction(kAbortFunction);
  bool sets = false;
  for (Instruction &inst : instructions(*abort))
    if (auto *store = dyn_cast<StoreInst>(&inst))
      sets |= store->getPointerOperand() == abort->getArg(0) &&
              cast<ConstantInt>(store->getValueOperand())->isOne();
  EXPECT_TRUE(sets) << print(*module);

  // After each call that can abort and is followed by more than a return,
  // the caller loads the flag and returns if it is raised; the load of the
  // returned reference and the kernel's store happen only otherwise.
  for (const char *name : {"twice", "k"}) {
    Function *func = module->getFunction(name);
    BasicBlock &entry = func->getEntryBlock();
    auto *check = dyn_cast<CondBrInst>(entry.getTerminator());
    ASSERT_NE(check, nullptr) << name << print(*module);
    auto *raised = cast<ICmpInst>(check->getCondition());
    auto *flag = cast<LoadInst>(raised->getOperand(0));
    EXPECT_EQ(flag->getPointerOperand(),
              func == kernel ? static_cast<Value *>(aborted)
                             : func->getArg(func->arg_size() - 1));
    BasicBlock *exit = check->getSuccessor(0);
    EXPECT_TRUE(isa<ReturnInst>(exit->getTerminator())) << print(*module);
    BasicBlock *rest = check->getSuccessor(1);
    EXPECT_TRUE((isa<LoadInst, StoreInst>(&rest->front()))) << print(*module);
    unsigned returns = 0;
    for (Instruction &inst : instructions(*func))
      returns += isa<ReturnInst>(inst);
    EXPECT_EQ(returns, 1u) << print(*module);
  }
  // @element's call to abort is followed by `unreachable`, which returns:
  // no check there.
  for (Instruction &inst : instructions(*module->getFunction("element")))
    EXPECT_FALSE(isa<LoadInst>(inst)) << print(*module);
}

// After the optimizer, functions that touch only globals carry
// memory(..., argmem: none), and so do the calls to them. The aborted flag
// makes them write argument memory (when they abort) and read it (after a
// call that can abort), so their memory attributes and those of the calls
// must allow both, keeping the rest; otherwise a pass could move the
// caller's load of the flag across the call that sets it.
TEST(AbortTest, MemoryAttributesAllowTheFlag) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define internal i32 @checked(i32 %i) #0 {
  %ok = icmp ult i32 %i, 4
  br i1 %ok, label %fine, label %bad
fine:
  ret i32 %i
bad:
  call void @__mojo_intel_abort() #0
  unreachable
}
define internal i32 @twice(i32 %i) #0 {
  %c = call i32 @checked(i32 %i) #0
  %w = add i32 %c, %c
  ret i32 %w
}
define spir_kernel void @k(ptr addrspace(1) %p, i32 %i) {
  %w = call i32 @twice(i32 %i) #1
  store i32 %w, ptr addrspace(1) %p
  ret void
}
declare void @__mojo_intel_abort() noreturn
attributes #0 = { nounwind memory(readwrite, argmem: none, inaccessiblemem: none) }
attributes #1 = { nounwind memory(readwrite, argmem: none) }
)");
  ASSERT_TRUE(module);
  ASSERT_TRUE(defineAbort(*module));
  ASSERT_FALSE(dropUnusedAbortFlag(*module));
  EXPECT_TRUE(returnAfterAbort(*module));
  std::string text;
  raw_string_ostream os(text);
  ASSERT_FALSE(verifyModule(*module, &os)) << text << print(*module);

  for (const char *name : {"checked", "twice"}) {
    Function *func = module->getFunction(name);
    ASSERT_NE(func, nullptr) << name << print(*module);
    ASSERT_TRUE(func->hasFnAttribute(Attribute::Memory)) << print(*module);
    MemoryEffects effects = func->getMemoryEffects();
    EXPECT_EQ(effects.getModRef(IRMemLocation::ArgMem), ModRefInfo::ModRef)
        << name << print(*module);
    EXPECT_EQ(effects.getModRef(IRMemLocation::Other), ModRefInfo::ModRef)
        << name;
    EXPECT_EQ(effects.getModRef(IRMemLocation::InaccessibleMem),
              ModRefInfo::NoModRef)
        << name;
    for (User *user : func->users()) {
      auto *call = cast<CallInst>(user);
      MemoryEffects callEffects = call->getAttributes().getMemoryEffects();
      EXPECT_EQ(callEffects.getModRef(IRMemLocation::ArgMem),
                ModRefInfo::ModRef)
          << name << print(*module);
      EXPECT_TRUE(call->hasFnAttr(Attribute::NoUnwind)) << name;
    }
  }
}

// The checks after calls that can abort come in program order, whatever
// the order of the callees in memory: the return value's incoming values
// from them follow the calls (the output must be the same on every run).
// Only the first assertion, on those incoming values, tells the fix from the
// code before it, which visited the calls callee by callee, in the order of
// the callees' heap addresses, so not a fixed order (its one measured run
// gave "a b b "). The second is a sanity check that the splits left the
// calls where they were: splitBasicBlock puts the new block right after the
// old one, so the layout follows the program whatever order the splits come
// in, before the fix too.
TEST(AbortTest, ChecksFollowProgramOrder) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define internal i32 @a(i32 %i) {
  %ok = icmp ult i32 %i, 4
  br i1 %ok, label %fine, label %bad
fine:
  ret i32 %i
bad:
  call void @__mojo_intel_abort()
  unreachable
}
define internal i32 @b(i32 %i) {
  %ok = icmp ult i32 %i, 8
  br i1 %ok, label %fine, label %bad
fine:
  ret i32 %i
bad:
  call void @__mojo_intel_abort()
  unreachable
}
define internal i32 @c(i32 %i) {
  %x = call i32 @b(i32 %i)
  %y = call i32 @a(i32 %x)
  %z = call i32 @b(i32 %y)
  %w = add i32 %z, 1
  ret i32 %w
}
define spir_kernel void @k(ptr addrspace(1) %p, i32 %i) {
  %v = call i32 @c(i32 %i)
  store i32 %v, ptr addrspace(1) %p
  ret void
}
declare void @__mojo_intel_abort() noreturn
)");
  ASSERT_TRUE(module);
  ASSERT_TRUE(defineAbort(*module));
  EXPECT_TRUE(returnAfterAbort(*module));
  std::string text;
  raw_string_ostream os(text);
  ASSERT_FALSE(verifyModule(*module, &os)) << text << print(*module);
  Function *c = module->getFunction("c");
  PHINode *result = nullptr;
  for (Instruction &inst : instructions(*c))
    if (auto *phi = dyn_cast<PHINode>(&inst))
      result = phi;
  ASSERT_NE(result, nullptr) << print(*module);
  // The old return first, then one poison per checked call, in order.
  std::string callees;
  for (unsigned i = 0; i < result->getNumIncomingValues(); ++i) {
    if (!isa<PoisonValue>(result->getIncomingValue(i)))
      continue;
    for (Instruction &inst : *result->getIncomingBlock(i))
      if (auto *call = dyn_cast<CallInst>(&inst))
        callees += call->getCalledFunction()->getName().str() + " ";
  }
  EXPECT_EQ(callees, "b a b ") << print(*module);
  // The calls are where the program had them (true before the fix too).
  std::string order;
  for (BasicBlock &block : *c)
    for (Instruction &inst : block)
      if (auto *call = dyn_cast<CallInst>(&inst))
        order += call->getCalledFunction()->getName().str() + " ";
  EXPECT_EQ(order, "b a b ") << print(*module);
}

// The optimizer may inline the abort hook: a function that reads the abort
// flag global itself sets its work-item's flag there, and its callers check
// it.
TEST(AbortTest, InlinedAbortSetsTheFlag) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@__mojo_intel_abort_flag = addrspace(1) externally_initialized global ptr addrspace(1) null, align 8
define internal i32 @checked(i32 %i) {
  %ok = icmp ult i32 %i, 4
  br i1 %ok, label %fine, label %bad
fine:
  ret i32 %i
bad:
  %target = load volatile ptr addrspace(1), ptr addrspace(1) @__mojo_intel_abort_flag, align 8
  store volatile i32 1, ptr addrspace(1) %target, align 4
  unreachable
}
define spir_kernel void @k(ptr addrspace(1) %p, i32 %i) {
  %v = call i32 @checked(i32 %i)
  store i32 %v, ptr addrspace(1) %p
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(returnAfterAbort(*module));
  std::string text;
  raw_string_ostream os(text);
  ASSERT_FALSE(verifyModule(*module, &os)) << text << print(*module);
  Function *checked = module->getFunction("checked");
  ASSERT_EQ(checked->arg_size(), 2u) << print(*module);
  bool sets = false;
  for (Instruction &inst : instructions(*checked))
    if (auto *store = dyn_cast<StoreInst>(&inst))
      sets |= store->getPointerOperand() == checked->getArg(1);
  EXPECT_TRUE(sets) << print(*module);
  EXPECT_TRUE(isa<CondBrInst>(
      module->getFunction("k")->getEntryBlock().getTerminator()))
      << print(*module);
}

// Without abort nothing returns from a `noreturn` call, and `unreachable`
// stays for the optimizer.
TEST(AbortTest, UnreachableStaysWithoutAbort) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(i32 %i) {
  switch i32 %i, label %never [i32 0, label %done]
never:
  unreachable
done:
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_FALSE(defineAbort(*module));
  EXPECT_FALSE(returnAfterAbort(*module));
  EXPECT_TRUE(isa<UnreachableInst>(
      module->getFunction("k")->front().getNextNode()->getTerminator()));
}

TEST(AbortTest, NothingWithoutAbort) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %p) {
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_FALSE(defineAbort(*module));
  EXPECT_EQ(module->getNamedGlobal(kAbortFlagGlobal), nullptr);
}

} // namespace
