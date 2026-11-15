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
// Tests for dynamic shared memory as a kernel parameter, on hand-written IR in
// SPIR-V numbering (after RemapAddressSpaces), as the plugin sees it before
// codegen.
//
//===----------------------------------------------------------------------===//

#include "DynamicSharedMemory.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
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
using M::KGEN::Intel::addDynamicSharedMemoryParameter;

namespace {

constexpr const char *kHeader =
    "target datalayout = \"e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-"
    "v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1\"\n"
    "target triple = \"spirv64-unknown-unknown\"\n";

std::unique_ptr<Module> parse(LLVMContext &ctx, const std::string &body) {
  SMDiagnostic diag;
  std::unique_ptr<Module> module =
      parseAssemblyString(std::string(kHeader) + body, diag, ctx);
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

std::string verify(const Module &module) {
  std::string text;
  raw_string_ostream os(text);
  return verifyModule(module, &os) ? text : std::string();
}

// Two `external_memory` symbols, one used through a constant GEP, and a
// kernel that uses none.
constexpr const char *kKernels = R"(
@extern_ptr_syml = external addrspace(3) global [0 x i8], align 16
@other = external addrspace(3) global [0 x i8], align 128

define spir_kernel void @uses(ptr addrspace(1) %out, i64 %i) {
  %p = getelementptr i32, ptr addrspace(3) @extern_ptr_syml, i64 %i
  %v = load i32, ptr addrspace(3) %p, align 4
  %q = load i32, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @other, i64 4), align 4
  %s = add i32 %v, %q
  store i32 %s, ptr addrspace(1) %out, align 4
  ret void
}

define spir_kernel void @plain(ptr addrspace(1) %out) {
  store i32 1, ptr addrspace(1) %out, align 4
  ret void
}
)";

TEST(DynamicSharedMemoryTest, BecomesATrailingKernelParameter) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, kKernels);
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(addDynamicSharedMemoryParameter(*module)));
  ASSERT_EQ(verify(*module), "") << print(*module);

  Function *uses = module->getFunction("uses");
  ASSERT_NE(uses, nullptr) << print(*module);
  EXPECT_EQ(uses->getCallingConv(), CallingConv::SPIR_KERNEL);
  ASSERT_EQ(uses->arg_size(), 3u);
  EXPECT_EQ(uses->getArg(0)->getName(), "out");
  EXPECT_EQ(uses->getArg(1)->getName(), "i");
  Argument *memory = uses->getArg(2);
  EXPECT_EQ(memory->getType()->getPointerAddressSpace(), 3u);
  // The strictest alignment of the symbols it replaces.
  EXPECT_EQ(memory->getParamAlign().valueOrOne().value(), 128u);
  // Both symbols alias the parameter; the globals are gone.
  EXPECT_EQ(module->getNamedGlobal("extern_ptr_syml"), nullptr)
      << print(*module);
  EXPECT_EQ(module->getNamedGlobal("other"), nullptr) << print(*module);
  unsigned geps = 0;
  for (Instruction &inst : instructions(*uses))
    if (auto *gep = dyn_cast<GetElementPtrInst>(&inst)) {
      EXPECT_EQ(gep->getPointerOperand(), memory) << print(*module);
      ++geps;
    }
  EXPECT_EQ(geps, 2u) << print(*module);

  // A kernel without dynamic shared memory keeps its signature.
  EXPECT_EQ(module->getFunction("plain")->arg_size(), 1u);
}

TEST(DynamicSharedMemoryTest, ReferencesFollowTheNewKernel) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@smem = external addrspace(3) global [0 x i32], align 4
@llvm.used = appending global [1 x ptr] [ptr @k], section "llvm.metadata"
define spir_kernel void @k(ptr addrspace(1) %out) {
  %v = load i32, ptr addrspace(3) @smem, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(addDynamicSharedMemoryParameter(*module)));
  ASSERT_EQ(verify(*module), "") << print(*module);
  Function *kernel = module->getFunction("k");
  ASSERT_EQ(kernel->arg_size(), 2u) << print(*module);
  auto *used = cast<ConstantArray>(
      module->getNamedGlobal("llvm.used")->getInitializer());
  EXPECT_EQ(used->getOperand(0), kernel) << print(*module);
}

TEST(DynamicSharedMemoryTest, NothingToDoWithoutExternalSharedMemory) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@tile = internal addrspace(3) global [64 x i32] poison
define spir_kernel void @k(ptr addrspace(1) %out) {
  %v = load i32, ptr addrspace(3) @tile, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  std::string before = print(*module);
  ASSERT_FALSE(static_cast<bool>(addDynamicSharedMemoryParameter(*module)));
  EXPECT_EQ(print(*module), before);
}

// The memory parameter of `func` (its last), or null.
Argument *memoryOf(Function *func) {
  if (!func || func->arg_size() == 0)
    return nullptr;
  Argument *last = func->getArg(func->arg_size() - 1);
  return last->getName() == "dynamic_shared_memory" ? last : nullptr;
}

// The calls in `caller` to the function named `callee`.
SmallVector<CallInst *> callsTo(Function *caller, StringRef callee) {
  SmallVector<CallInst *> calls;
  for (Instruction &inst : instructions(*caller))
    if (auto *call = dyn_cast<CallInst>(&inst))
      if (call->getCalledFunction() &&
          call->getCalledFunction()->getName() == callee)
        calls.push_back(call);
  return calls;
}

// MAX's pattern 1. Only the callee names the memory; the kernel gets the
// parameter and passes it on.
TEST(DynamicSharedMemoryTest, ReachesACalleeOneLevelDown) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@callee_ext = external addrspace(3) global [0 x float], align 4
define internal float @helper(i64 %i) {
  %p = getelementptr float, ptr addrspace(3) @callee_ext, i64 %i
  %v = load float, ptr addrspace(3) %p, align 4
  ret float %v
}
define spir_kernel void @k(ptr addrspace(1) %out, i64 %i) {
  %v = call float @helper(i64 %i) #0
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}
attributes #0 = { nounwind }
)");
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(addDynamicSharedMemoryParameter(*module)));
  ASSERT_EQ(verify(*module), "") << print(*module);
  EXPECT_EQ(module->getNamedGlobal("callee_ext"), nullptr) << print(*module);

  Function *kernel = module->getFunction("k");
  Function *helper = module->getFunction("helper");
  ASSERT_NE(memoryOf(kernel), nullptr) << print(*module);
  ASSERT_NE(memoryOf(helper), nullptr) << print(*module);
  EXPECT_EQ(kernel->arg_size(), 3u);
  EXPECT_EQ(helper->arg_size(), 2u);
  EXPECT_EQ(helper->getArg(0)->getName(), "i");
  EXPECT_EQ(helper->getLinkage(), GlobalValue::InternalLinkage);
  EXPECT_EQ(memoryOf(helper)->getParamAlign().valueOrOne().value(), 4u);

  SmallVector<CallInst *> calls = callsTo(kernel, "helper");
  ASSERT_EQ(calls.size(), 1u) << print(*module);
  EXPECT_EQ(calls[0]->getArgOperand(1), memoryOf(kernel));
  EXPECT_TRUE(calls[0]->hasFnAttr(Attribute::NoUnwind));
  auto *gep = cast<GetElementPtrInst>(&*instructions(*helper).begin());
  EXPECT_EQ(gep->getPointerOperand(), memoryOf(helper)) << print(*module);
}

// MAX's pattern 2, one level deeper, with a pass-through that never names
// the memory, and a recursive function (no recursion on GPUs, but the pass
// must not loop on it).
TEST(DynamicSharedMemoryTest, ReachesACalleeSeveralLevelsDown) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@deep_ext = external addrspace(3) global [0 x i32], align 4
define internal void @baz(i32 %v) {
  store i32 %v, ptr addrspace(3) @deep_ext, align 4
  ret void
}
define internal void @bar(i32 %v) {
  call void @baz(i32 %v)
  call void @bar(i32 %v)
  ret void
}
define internal void @foo(i32 %v) {
  call void @bar(i32 %v)
  ret void
}
define spir_kernel void @k(i32 %v) {
  call void @foo(i32 %v)
  ret void
}
)");
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(addDynamicSharedMemoryParameter(*module)));
  ASSERT_EQ(verify(*module), "") << print(*module);
  for (const char *name : {"k", "foo", "bar", "baz"}) {
    Function *func = module->getFunction(name);
    ASSERT_NE(memoryOf(func), nullptr) << name << "\n" << print(*module);
    EXPECT_EQ(func->arg_size(), 2u) << name;
  }
  // Each caller passes its own parameter on, including the recursive call.
  Function *bar = module->getFunction("bar");
  for (CallInst *call : callsTo(bar, "baz"))
    EXPECT_EQ(call->getArgOperand(1), memoryOf(bar));
  ASSERT_EQ(callsTo(bar, "bar").size(), 1u) << print(*module);
  EXPECT_EQ(callsTo(bar, "bar")[0]->getArgOperand(1), memoryOf(bar));
  Function *foo = module->getFunction("foo");
  EXPECT_EQ(callsTo(foo, "bar")[0]->getArgOperand(1), memoryOf(foo));
  Function *kernel = module->getFunction("k");
  EXPECT_EQ(callsTo(kernel, "foo")[0]->getArgOperand(1), memoryOf(kernel));
}

// One callee, two kernels: one names the memory itself (another symbol, which
// aliases it), the other only through the callee. Both pass their parameter,
// so the callee has one signature. A callee that doesn't use the memory, from
// a kernel that does, keeps its signature.
TEST(DynamicSharedMemoryTest, ACalleeSharedByTwoKernelsHasOneSignature) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@callee_ext = external addrspace(3) global [0 x i32], align 4
@kernel_ext = external addrspace(3) global [0 x i32], align 16
define internal i32 @shared(i64 %i) {
  %p = getelementptr i32, ptr addrspace(3) @callee_ext, i64 %i
  %v = load i32, ptr addrspace(3) %p, align 4
  ret i32 %v
}
define internal i32 @plain(i32 %v) {
  %w = add i32 %v, 1
  ret i32 %w
}
define spir_kernel void @direct(ptr addrspace(1) %out) {
  %a = call i32 @shared(i64 1)
  %b = load i32, ptr addrspace(3) @kernel_ext, align 4
  %c = call i32 @plain(i32 %b)
  %s = add i32 %a, %c
  store i32 %s, ptr addrspace(1) %out, align 4
  ret void
}
define spir_kernel void @indirect(ptr addrspace(1) %out) {
  %a = call i32 @shared(i64 2)
  store i32 %a, ptr addrspace(1) %out, align 4
  ret void
}
define spir_kernel void @none(ptr addrspace(1) %out) {
  %a = call i32 @plain(i32 3)
  store i32 %a, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(addDynamicSharedMemoryParameter(*module)));
  ASSERT_EQ(verify(*module), "") << print(*module);
  EXPECT_EQ(module->getNamedGlobal("callee_ext"), nullptr) << print(*module);
  EXPECT_EQ(module->getNamedGlobal("kernel_ext"), nullptr) << print(*module);

  Function *shared = module->getFunction("shared");
  ASSERT_NE(memoryOf(shared), nullptr) << print(*module);
  EXPECT_EQ(shared->arg_size(), 2u);
  // The strictest alignment of all the symbols, everywhere.
  EXPECT_EQ(memoryOf(shared)->getParamAlign().valueOrOne().value(), 16u);
  for (const char *name : {"direct", "indirect"}) {
    Function *kernel = module->getFunction(name);
    ASSERT_NE(memoryOf(kernel), nullptr) << name << "\n" << print(*module);
    EXPECT_EQ(kernel->arg_size(), 2u) << name;
    SmallVector<CallInst *> calls = callsTo(kernel, "shared");
    ASSERT_EQ(calls.size(), 1u) << name;
    EXPECT_EQ(calls[0]->getArgOperand(1), memoryOf(kernel)) << name;
  }
  Function *direct = module->getFunction("direct");
  bool loadsParameter = false;
  for (Instruction &inst : instructions(*direct))
    if (auto *load = dyn_cast<LoadInst>(&inst))
      loadsParameter |= load->getPointerOperand() == memoryOf(direct);
  EXPECT_TRUE(loadsParameter) << print(*module);

  EXPECT_EQ(module->getFunction("plain")->arg_size(), 1u) << print(*module);
  EXPECT_EQ(module->getFunction("none")->arg_size(), 1u) << print(*module);
}

// A non-kernel function in @llvm.used follows its copy.
TEST(DynamicSharedMemoryTest, UsedListsFollowACallee) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@smem = external addrspace(3) global [0 x i32], align 4
@llvm.compiler.used = appending global [1 x ptr] [ptr @helper], section "llvm.metadata"
define internal void @helper() {
  store i32 1, ptr addrspace(3) @smem, align 4
  ret void
}
define spir_kernel void @k() {
  call void @helper()
  ret void
}
)");
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(addDynamicSharedMemoryParameter(*module)));
  ASSERT_EQ(verify(*module), "") << print(*module);
  Function *helper = module->getFunction("helper");
  ASSERT_NE(memoryOf(helper), nullptr) << print(*module);
  auto *used = cast<ConstantArray>(
      module->getNamedGlobal("llvm.compiler.used")->getInitializer());
  EXPECT_EQ(used->getOperand(0), helper) << print(*module);
}

// A dead constant on a function's @llvm.used entry stays in the module when
// it comes after the entry in the use list: removeDeadConstantUsers stops at
// the entry, a live user. It must not count as a taken address. Text IR
// can't hold an unused constant, so the test builds it.
TEST(DynamicSharedMemoryTest, AcceptsADeadConstantBehindAUsedListEntry) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@smem = external addrspace(3) global [0 x i32], align 4
define internal i32 @helper() {
  %v = load i32, ptr addrspace(3) @smem, align 4
  ret i32 %v
}
define spir_kernel void @k() {
  %v = call i32 @helper()
  ret void
}
)");
  ASSERT_TRUE(module);
  Function *helper = module->getFunction("helper");
  auto *global = PointerType::get(ctx, 1);
  Constant *entry = ConstantExpr::getAddrSpaceCast(helper, global);
  // Made before the list, so the list's use of `entry` comes first.
  Constant *dead = ConstantExpr::getGetElementPtr(
      Type::getInt8Ty(ctx), entry, ConstantInt::get(Type::getInt64Ty(ctx), 4));
  auto *listType = ArrayType::get(global, 1);
  auto *list = new GlobalVariable(
      *module, listType, false, GlobalValue::AppendingLinkage,
      ConstantArray::get(listType, {entry}), "llvm.used");
  list->setSection("llvm.metadata");
  // The setup the test is about: the dead constant survives the cleanup.
  helper->removeDeadConstantUsers();
  ASSERT_TRUE(dead->use_empty());
  ASSERT_TRUE(any_of(entry->users(), [&](User *u) { return u == dead; }))
      << "removeDeadConstantUsers removed the dead constant; the test no "
         "longer reaches the case";
  ASSERT_EQ(verify(*module), "") << print(*module);

  Error err = addDynamicSharedMemoryParameter(*module);
  ASSERT_FALSE(static_cast<bool>(err)) << toString(std::move(err));
  ASSERT_EQ(verify(*module), "") << print(*module);
  helper = module->getFunction("helper");
  ASSERT_NE(memoryOf(helper), nullptr) << print(*module);
  ASSERT_NE(memoryOf(module->getFunction("k")), nullptr) << print(*module);
  auto *used = cast<ConstantArray>(
      module->getNamedGlobal("llvm.used")->getInitializer());
  EXPECT_EQ(used->getOperand(0)->stripPointerCasts(), helper) << print(*module);
}

// An indirect call can't pass the memory: the function's address escapes.
TEST(DynamicSharedMemoryTest, RejectsAFunctionReachedIndirectly) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@extern_ptr_syml = external addrspace(3) global [0 x i8], align 16
define internal i32 @helper() {
  %v = load i32, ptr addrspace(3) @extern_ptr_syml, align 4
  ret i32 %v
}
define internal i32 @apply(ptr %f) {
  %v = call i32 %f()
  ret i32 %v
}
define spir_kernel void @k(ptr addrspace(1) %out) {
  %v = call i32 @apply(ptr @helper)
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  Error err = addDynamicSharedMemoryParameter(*module);
  ASSERT_TRUE(static_cast<bool>(err));
  std::string message = toString(std::move(err));
  EXPECT_NE(message.find("helper uses dynamic shared memory"),
            std::string::npos)
      << message;
  EXPECT_NE(message.find("only a direct call"), std::string::npos) << message;
}

// The error for `body`, which the pass must refuse, or a failure.
std::string refusal(const std::string &body) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, body);
  if (!module)
    return "";
  Error err = addDynamicSharedMemoryParameter(*module);
  if (!err) {
    ADD_FAILURE() << "accepted:\n" << print(*module);
    return "";
  }
  return toString(std::move(err));
}

constexpr const char *kHelper = R"(
@smem = external addrspace(3) global [0 x i32], align 4
define internal i32 @helper() {
  %v = load i32, ptr addrspace(3) @smem, align 4
  ret i32 %v
}
)";

// Every other way to reach a function than a direct call, each refused with
// its reason.
TEST(DynamicSharedMemoryTest, RejectsAStoredAddress) {
  std::string message = refusal(std::string(kHelper) + R"(
define spir_kernel void @k(ptr addrspace(1) %slot) {
  store ptr @helper, ptr addrspace(1) %slot, align 8
  ret void
}
)");
  EXPECT_NE(message.find("helper uses dynamic shared memory but its address "
                         "is taken"),
            std::string::npos)
      << message;
}

TEST(DynamicSharedMemoryTest, RejectsAFunctionTable) {
  std::string message = refusal(std::string(kHelper) + R"(
@table = internal addrspace(1) global [1 x ptr] [ptr @helper]
define spir_kernel void @k() {
  %v = call i32 @helper()
  ret void
}
)");
  EXPECT_NE(message.find("helper uses dynamic shared memory but its address "
                         "is taken"),
            std::string::npos)
      << message;
}

// An alias is a second name the pass can't follow, even unused; so is a
// personality. Neither counts as a linker list.
TEST(DynamicSharedMemoryTest, RejectsAnAliasOrAPersonality) {
  std::string alias = refusal(std::string(kHelper) + R"(
@other_name = internal alias i32 (), ptr @helper
define spir_kernel void @k() {
  %v = call i32 @helper()
  ret void
}
)");
  EXPECT_NE(alias.find("helper uses dynamic shared memory but its address is "
                       "taken"),
            std::string::npos)
      << alias;
  std::string personality = refusal(std::string(kHelper) + R"(
define spir_kernel void @k() personality ptr @helper {
  ret void
}
)");
  EXPECT_NE(personality.find("helper uses dynamic shared memory but its "
                             "address is taken"),
            std::string::npos)
      << personality;
}

// A global value that names the function is a taken address even when the
// linker lists that global: the list keeps it alive, it doesn't make it a
// call. Each one here is used, so only the global value check refuses it.
TEST(DynamicSharedMemoryTest, RejectsAGlobalValueEvenWhenListed) {
  const char *const kCases[] = {
      R"(
@other_name = alias i32 (), ptr @helper
@llvm.used = appending global [1 x ptr] [ptr @other_name], section "llvm.metadata"
define spir_kernel void @k() {
  %v = call i32 @helper()
  ret void
}
)",
      R"(
@fp = addrspace(1) global ptr @helper
@llvm.used = appending global [1 x ptr addrspace(1)] [ptr addrspace(1) @fp], section "llvm.metadata"
define spir_kernel void @k() {
  %v = call i32 @helper()
  ret void
}
)",
      R"(
@llvm.used = appending global [1 x ptr] [ptr @k], section "llvm.metadata"
define spir_kernel void @k() personality ptr @helper {
  ret void
}
)",
      R"(
@llvm.compiler.used = appending global [1 x ptr] [ptr @k], section "llvm.metadata"
define spir_kernel void @k() prefix ptr @helper {
  ret void
}
)",
  };
  for (const char *body : kCases) {
    std::string message = refusal(std::string(kHelper) + body);
    EXPECT_NE(message.find("helper uses dynamic shared memory but its "
                           "address is taken"),
              std::string::npos)
        << body << message;
  }
}

TEST(DynamicSharedMemoryTest, RejectsAnInvoke) {
  std::string message = refusal(std::string(kHelper) + R"(
declare i32 @personality(...)
define spir_kernel void @k() personality ptr @personality {
  %v = invoke i32 @helper() to label %ok unwind label %bad
ok:
  ret void
bad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}
)");
  EXPECT_NE(message.find("helper uses dynamic shared memory but is invoked "
                         "other than by a call"),
            std::string::npos)
      << message;
}

TEST(DynamicSharedMemoryTest, RejectsACallWithAnotherSignature) {
  std::string message = refusal(std::string(kHelper) + R"(
define spir_kernel void @k() {
  %v = call i32 @helper(i32 1)
  ret void
}
)");
  EXPECT_NE(message.find("helper uses dynamic shared memory but is called "
                         "with another signature"),
            std::string::npos)
      << message;
}

// A pass-through caller is named with the callee it needs the memory for.
TEST(DynamicSharedMemoryTest, NamesTheCalleeOfARejectedPassThrough) {
  std::string message = refusal(std::string(kHelper) + R"(
define internal i32 @middle() {
  %v = call i32 @helper()
  ret i32 %v
}
define internal i32 @apply(ptr %f) {
  %v = call i32 %f()
  ret i32 %v
}
define spir_kernel void @k() {
  %v = call i32 @apply(ptr @middle)
  ret void
}
)");
  EXPECT_NE(message.find("middle calls helper, which needs dynamic shared "
                         "memory, but is passed as an argument"),
            std::string::npos)
      << message;
}

// Optimized callees that touch only the memory carry memory(...,
// argmem: none); after the pass they touch argument memory, so their memory
// attributes (and those of the calls) must allow it, keeping the rest.
TEST(DynamicSharedMemoryTest, MemoryAttributesAllowTheArgument) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@smem = external addrspace(3) global [0 x i32], align 4
define internal void @write_only(i32 %v) #0 {
  store i32 %v, ptr addrspace(3) @smem, align 4
  ret void
}
define internal i32 @read_only() #1 {
  %v = load i32, ptr addrspace(3) @smem, align 4
  ret i32 %v
}
define spir_kernel void @k(ptr addrspace(1) %out) {
  call void @write_only(i32 1) #0
  %v = call i32 @read_only() #1
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
attributes #0 = { nounwind memory(write, argmem: none, inaccessiblemem: none) }
attributes #1 = { nounwind memory(read, argmem: none, inaccessiblemem: none) }
)");
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(addDynamicSharedMemoryParameter(*module)));
  ASSERT_EQ(verify(*module), "") << print(*module);

  Function *writer = module->getFunction("write_only");
  Function *reader = module->getFunction("read_only");
  ASSERT_NE(memoryOf(writer), nullptr) << print(*module);
  ASSERT_NE(memoryOf(reader), nullptr) << print(*module);
  MemoryEffects writes = writer->getMemoryEffects();
  EXPECT_TRUE(isModSet(writes.getModRef(IRMemLocation::ArgMem)))
      << print(*module);
  EXPECT_TRUE(writes.onlyWritesMemory()) << print(*module);
  EXPECT_FALSE(isModOrRefSet(writes.getModRef(IRMemLocation::InaccessibleMem)));
  MemoryEffects reads = reader->getMemoryEffects();
  EXPECT_TRUE(isRefSet(reads.getModRef(IRMemLocation::ArgMem)))
      << print(*module);
  EXPECT_TRUE(reads.onlyReadsMemory()) << print(*module);

  // The call sites' own attributes, apart from their callees'.
  Function *kernel = module->getFunction("k");
  SmallVector<CallInst *> writeCalls = callsTo(kernel, "write_only");
  SmallVector<CallInst *> readCalls = callsTo(kernel, "read_only");
  ASSERT_EQ(writeCalls.size(), 1u) << print(*module);
  ASSERT_EQ(readCalls.size(), 1u) << print(*module);
  MemoryEffects writeCall = writeCalls[0]->getAttributes().getMemoryEffects();
  EXPECT_TRUE(isModSet(writeCall.getModRef(IRMemLocation::ArgMem)))
      << print(*module);
  EXPECT_TRUE(writeCall.onlyWritesMemory());
  MemoryEffects readCall = readCalls[0]->getAttributes().getMemoryEffects();
  EXPECT_TRUE(isRefSet(readCall.getModRef(IRMemLocation::ArgMem)))
      << print(*module);
  EXPECT_TRUE(readCall.onlyReadsMemory());
  EXPECT_TRUE(readCalls[0]->hasFnAttr(Attribute::NoUnwind));
}

} // namespace
