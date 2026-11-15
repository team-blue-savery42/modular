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
// Tests for the device heap's contract with the Level Zero driver plugin
// (Heap.h). The allocator's behaviour (many work-items at once, out of
// memory, realloc's copy) is checked on the GPU by the driver plugin's
// heap tests (LevelZeroLaunchTest.cpp).
//
//===----------------------------------------------------------------------===//

#include "Heap.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

using namespace llvm;
using M::KGEN::Intel::defineHeapFunctions;
using M::KGEN::Intel::kHeapGlobal;

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

void expectValid(const Module &module) {
  std::string text;
  raw_string_ostream os(text);
  EXPECT_FALSE(verifyModule(module, &os)) << text << print(module);
}

/// The atomic instructions in `func` of type `T`.
template <typename T>
std::vector<T *> atomics(Function &func) {
  std::vector<T *> found;
  for (Instruction &inst : instructions(func))
    if (auto *atomic = dyn_cast<T>(&inst))
      found.push_back(atomic);
  return found;
}

// What a kernel holding a List calls (the stdlib's `_malloc` and `_free` on a
// GPU): malloc and free are defined in the module, malloc reserves with a
// device-scope compare-exchange, and the heap pointer is exported for the
// driver.
TEST(HeapTest, DefinesMallocAndFreeAndExportsTheHeap) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %p) {
  %m = call ptr addrspace(4) @malloc(i64 12)
  store i32 1, ptr addrspace(4) %m
  call void @free(ptr addrspace(4) %m)
  ret void
}
declare ptr addrspace(4) @malloc(i64)
declare void @free(ptr addrspace(4))
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(defineHeapFunctions(*module));
  expectValid(*module);

  // Nothing is left for the driver to link.
  EXPECT_EQ(module->getFunction("malloc"), nullptr) << print(*module);
  EXPECT_EQ(module->getFunction("free"), nullptr) << print(*module);
  for (Function &func : *module)
    EXPECT_FALSE(func.isDeclaration() && !func.isIntrinsic())
        << func.getName().str();

  GlobalVariable *heap = module->getNamedGlobal(kHeapGlobal);
  ASSERT_NE(heap, nullptr) << print(*module);
  EXPECT_EQ(heap->getLinkage(), GlobalValue::ExternalLinkage);
  EXPECT_TRUE(heap->isExternallyInitialized());
  EXPECT_EQ(heap->getAddressSpace(), 1u);
  EXPECT_EQ(heap->getValueType()->getPointerAddressSpace(), 1u);

  Function *malloc = module->getFunction("__mojo_intel_malloc");
  ASSERT_NE(malloc, nullptr) << print(*module);
  EXPECT_TRUE(malloc->hasInternalLinkage());
  EXPECT_TRUE(atomics<AtomicRMWInst>(*malloc).empty()) << print(*module);
  SmallVector<StringRef> scopes;
  ctx.getSyncScopeNames(scopes);
  // Two exchanges: one pops a freed block off its class's list
  // (acquire), one bumps the heap's counter; all device-scope.
  std::vector<AtomicCmpXchgInst *> exchanges =
      atomics<AtomicCmpXchgInst>(*malloc);
  ASSERT_EQ(exchanges.size(), 2u) << print(*module);
  for (AtomicCmpXchgInst *x : exchanges) {
    EXPECT_EQ(x->getPointerAddressSpace(), 1u);
    EXPECT_TRUE(x->getNewValOperand()->getType()->isIntegerTy(64));
    EXPECT_EQ(scopes[x->getSyncScopeID()], "device");
  }
  AtomicCmpXchgInst *popExchange = nullptr, *exchange = nullptr;
  for (AtomicCmpXchgInst *x : exchanges)
    (x->getSuccessOrdering() == AtomicOrdering::Acquire ? popExchange
                                                        : exchange) = x;
  ASSERT_NE(popExchange, nullptr) << print(*module);
  ASSERT_NE(exchange, nullptr) << print(*module);
  for (LoadInst *load : atomics<LoadInst>(*malloc))
    if (load->isAtomic())
      EXPECT_EQ(scopes[load->getSyncScopeID()], "device");

  // A request that doesn't fit leaves the counter alone: the exchange is
  // reached only when its new value (the block's end) is within the
  // capacity, so a failed request gives the rest of the heap to the next.
  BasicBlock *guard = exchange->getParent()->getSinglePredecessor();
  ASSERT_NE(guard, nullptr) << print(*module);
  auto *branch = dyn_cast<CondBrInst>(guard->getTerminator());
  ASSERT_NE(branch, nullptr) << print(*module);
  auto *fits = dyn_cast<ICmpInst>(branch->getCondition());
  ASSERT_NE(fits, nullptr) << print(*module);
  EXPECT_EQ(fits->getPredicate(), ICmpInst::ICMP_UGT);
  EXPECT_EQ(fits->getOperand(0), exchange->getNewValOperand());
  EXPECT_EQ(branch->getSuccessor(1), exchange->getParent());
  // A lost race retries from the value the exchange saw.
  auto *start = dyn_cast<PHINode>(exchange->getCompareOperand());
  ASSERT_NE(start, nullptr) << print(*module);
  EXPECT_EQ(start->getParent(), guard);
  bool retriesWithSeen = false;
  for (Value *incoming : start->incoming_values())
    if (auto *seen = dyn_cast<ExtractValueInst>(incoming))
      retriesWithSeen |=
          seen->getAggregateOperand() == exchange && seen->getIndices()[0] == 0;
  EXPECT_TRUE(retriesWithSeen) << print(*module);
  // So does a lost pop.
  auto *head = dyn_cast<PHINode>(popExchange->getCompareOperand());
  ASSERT_NE(head, nullptr) << print(*module);
  bool popRetriesWithSeen = false;
  for (Value *incoming : head->incoming_values())
    if (auto *seen = dyn_cast<ExtractValueInst>(incoming))
      popRetriesWithSeen |= seen->getAggregateOperand() == popExchange;
  EXPECT_TRUE(popRetriesWithSeen) << print(*module);

  // free pushes on the list with one release exchange, retried from the
  // head it saw, after linking the block with an atomic store.
  Function *free = module->getFunction("__mojo_intel_free");
  ASSERT_NE(free, nullptr) << print(*module);
  EXPECT_TRUE(free->hasInternalLinkage());
  EXPECT_TRUE(atomics<AtomicRMWInst>(*free).empty());
  std::vector<AtomicCmpXchgInst *> pushes = atomics<AtomicCmpXchgInst>(*free);
  ASSERT_EQ(pushes.size(), 1u) << print(*module);
  EXPECT_EQ(pushes[0]->getSuccessOrdering(), AtomicOrdering::Release);
  EXPECT_EQ(scopes[pushes[0]->getSyncScopeID()], "device");
  unsigned linkStores = 0;
  for (StoreInst *store : atomics<StoreInst>(*free))
    linkStores += store->isAtomic();
  EXPECT_EQ(linkStores, 1u) << print(*module);

  // The kernel calls the definitions, not new declarations.
  Function *kernel = module->getFunction("k");
  unsigned calls = 0;
  for (Instruction &inst : instructions(*kernel))
    if (auto *call = dyn_cast<CallInst>(&inst)) {
      ++calls;
      EXPECT_TRUE(call->getCalledFunction() == malloc ||
                  call->getCalledFunction() == free);
    }
  EXPECT_EQ(calls, 2u);
}

// A module that doesn't allocate is unchanged: no heap global for the driver
// to set.
TEST(HeapTest, ModulesThatDontAllocateAreUnchanged) {
  LLVMContext ctx;
  const char *source = R"(
define spir_kernel void @k(ptr addrspace(1) %p) {
  store i32 1, ptr addrspace(1) %p
  ret void
}
)";
  std::unique_ptr<Module> module = parse(ctx, source);
  ASSERT_TRUE(module);
  std::string before = print(*module);
  EXPECT_FALSE(defineHeapFunctions(*module));
  EXPECT_EQ(print(*module), before);
  EXPECT_EQ(module->getNamedGlobal(kHeapGlobal), nullptr);
}

// free alone (an optimizer removed the malloc) still reaches the heap: the
// block came from another module's malloc on the same context.
TEST(HeapTest, FreeAloneReachesTheHeap) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(4) %p) {
  call void @free(ptr addrspace(4) %p)
  ret void
}
declare void @free(ptr addrspace(4))
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(defineHeapFunctions(*module));
  expectValid(*module);
  EXPECT_EQ(module->getFunction("free"), nullptr);
  EXPECT_NE(module->getNamedGlobal(kHeapGlobal), nullptr) << print(*module);
}

// realloc comes with a malloc and a free of its own, which it calls, and
// copies from the old block before freeing it; its own declaration goes.
TEST(HeapTest, ReallocBringsMalloc) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %out, ptr addrspace(4) %p) {
  %q = call ptr addrspace(4) @realloc(ptr addrspace(4) %p, i64 64)
  store ptr addrspace(4) %q, ptr addrspace(1) %out
  ret void
}
declare ptr addrspace(4) @realloc(ptr addrspace(4), i64)
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(defineHeapFunctions(*module));
  expectValid(*module);
  EXPECT_EQ(module->getFunction("realloc"), nullptr);
  Function *malloc = module->getFunction("__mojo_intel_malloc");
  Function *realloc = module->getFunction("__mojo_intel_realloc");
  ASSERT_NE(malloc, nullptr) << print(*module);
  ASSERT_NE(realloc, nullptr) << print(*module);
  EXPECT_NE(module->getNamedGlobal(kHeapGlobal), nullptr);
  Function *free = module->getFunction("__mojo_intel_free");
  ASSERT_NE(free, nullptr) << print(*module);
  bool callsMalloc = false, callsFree = false, copies = false;
  for (Instruction &inst : instructions(*realloc)) {
    if (auto *call = dyn_cast<CallInst>(&inst)) {
      callsMalloc |= call->getCalledFunction() == malloc;
      callsFree |= call->getCalledFunction() == free;
    }
    copies |= isa<StoreInst>(inst);
  }
  EXPECT_TRUE(callsMalloc) << print(*module);
  EXPECT_TRUE(callsFree) << print(*module);
  EXPECT_TRUE(copies) << print(*module);
  // No memcpy: the plugin must not create calls a SPIR-V module can't
  // link (ctlz, umin and umax are SPIR-V instructions).
  for (Function &func : *module)
    EXPECT_FALSE(func.isIntrinsic() && func.getName().contains("memcpy"))
        << func.getName().str();
}

// A declaration of another shape isn't the C function: it's left for the
// driver plugin to report as unresolved.
TEST(HeapTest, OtherShapesAreLeftAlone) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %p) {
  %m = call i32 @malloc(ptr addrspace(1) %p)
  store i32 %m, ptr addrspace(1) %p
  ret void
}
declare i32 @malloc(ptr addrspace(1))
)");
  ASSERT_TRUE(module);
  EXPECT_FALSE(defineHeapFunctions(*module));
  ASSERT_NE(module->getFunction("malloc"), nullptr);
  EXPECT_TRUE(module->getFunction("malloc")->isDeclaration());
  EXPECT_EQ(module->getNamedGlobal(kHeapGlobal), nullptr);
}

} // namespace
