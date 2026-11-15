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
// Tests for aggregate phis split into scalar phis, on hand-written IR in
// SPIR-V numbering, as the plugin sees it before codegen.
//
//===----------------------------------------------------------------------===//

#include "AggregatePhis.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
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

using namespace llvm;
using M::KGEN::Intel::splitAggregatePhis;

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

unsigned aggregatePhis(Function &func) {
  unsigned count = 0;
  for (Instruction &inst : instructions(func))
    if (isa<PHINode>(inst) && inst.getType()->isAggregateType())
      ++count;
  return count;
}

PHINode *phiOf(Value *value) { return dyn_cast<PHINode>(value); }

/// The value `phi` takes from the block named `name`.
Value *incomingFrom(PHINode *phi, StringRef name) {
  for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i)
    if (phi->getIncomingBlock(i)->getName() == name)
      return phi->getIncomingValue(i);
  return nullptr;
}

// The shape IGC miscompiled: the Optional an `enumerate` loop's iterator
// returns, `{ { i64, [8 x i8] }, i8 }`, merged by a phi whose "none" side is
// `{ undef, 0 }`. IGC lost the index (its i64). After the split the store
// reads an i64 phi that takes the loop counter from the body, and no
// aggregate phi or insertvalue is left.
TEST(AggregatePhisTest, EnumerateOptional) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %out) {
entry:
  br label %head
head:
  %opt = phi { { i64, [8 x i8] }, i8 } [ { { i64, [8 x i8] } undef, i8 0 }, %entry ], [ %some, %body ]
  %i = phi i64 [ 0, %entry ], [ %next, %body ]
  %tag = extractvalue { { i64, [8 x i8] }, i8 } %opt, 1
  %idx = extractvalue { { i64, [8 x i8] }, i8 } %opt, 0, 0
  %byte = extractvalue { { i64, [8 x i8] }, i8 } %opt, 0, 1, 0
  %slot = getelementptr i32, ptr addrspace(1) %out, i64 %i
  %t = trunc i64 %idx to i32
  store i32 %t, ptr addrspace(1) %slot, align 4
  %bslot = getelementptr i8, ptr addrspace(1) %out, i64 %i
  store i8 %byte, ptr addrspace(1) %bslot, align 1
  %done = icmp eq i8 %tag, 0
  br i1 %done, label %exit, label %body
body:
  %next = add i64 %i, 1
  %inner = insertvalue { i64, [8 x i8] } undef, i64 %i, 0
  %b = insertvalue { i64, [8 x i8] } %inner, i8 48, 1, 0
  %s0 = insertvalue { { i64, [8 x i8] }, i8 } undef, { i64, [8 x i8] } %b, 0
  %some = insertvalue { { i64, [8 x i8] }, i8 } %s0, i8 1, 1
  %stop = icmp eq i64 %next, 3
  br i1 %stop, label %exit, label %head
exit:
  ret void
}
)");
  ASSERT_TRUE(module);
  Function *k = module->getFunction("k");
  ASSERT_EQ(aggregatePhis(*k), 1u);
  EXPECT_TRUE(splitAggregatePhis(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  EXPECT_EQ(aggregatePhis(*k), 0u) << print(*module);
  for (Instruction &inst : instructions(*k)) {
    EXPECT_FALSE(isa<InsertValueInst>(inst)) << print(*module);
    EXPECT_FALSE(isa<ExtractValueInst>(inst)) << print(*module);
  }

  Value *counter = nullptr;
  PHINode *index = nullptr, *byte = nullptr, *tag = nullptr;
  for (Instruction &inst : instructions(*k)) {
    if (inst.getName() == "next")
      counter = inst.getOperand(0); // %i
    if (auto *trunc = dyn_cast<TruncInst>(&inst))
      index = phiOf(trunc->getOperand(0));
    if (auto *store = dyn_cast<StoreInst>(&inst))
      if (store->getValueOperand()->getType()->isIntegerTy(8))
        byte = phiOf(store->getValueOperand());
    if (auto *cmp = dyn_cast<ICmpInst>(&inst))
      if (cmp->getOperand(0)->getType()->isIntegerTy(8))
        tag = phiOf(cmp->getOperand(0));
  }
  ASSERT_NE(index, nullptr) << print(*module);
  ASSERT_NE(byte, nullptr) << print(*module);
  ASSERT_NE(tag, nullptr) << print(*module);
  EXPECT_TRUE(index->getType()->isIntegerTy(64));
  EXPECT_EQ(incomingFrom(index, "body"), counter) << print(*module);
  EXPECT_TRUE(isa<UndefValue>(incomingFrom(index, "entry")));
  auto *b48 = dyn_cast<ConstantInt>(incomingFrom(byte, "body"));
  ASSERT_NE(b48, nullptr) << print(*module);
  EXPECT_EQ(b48->getZExtValue(), 48u);
  EXPECT_TRUE(isa<UndefValue>(incomingFrom(byte, "entry")));
  EXPECT_TRUE(cast<ConstantInt>(incomingFrom(tag, "body"))->isOne());
  EXPECT_TRUE(cast<ConstantInt>(incomingFrom(tag, "entry"))->isZero());
  // The leaves the loop never reads (bytes 1 to 7) leave no phi behind:
  // %i, the index, the first byte and the tag.
  unsigned phis = 0;
  for (Instruction &inst : instructions(*k))
    phis += isa<PHINode>(inst);
  EXPECT_EQ(phis, 4u) << print(*module);
}

// A float leaf of a zero constant is read with an extractvalue at the end of
// the incoming block, not made a new `float 0.0` (the plugin never creates
// floating-point constants); an integer leaf of it is a zero integer, and a
// float leaf of a constant struct is the struct's own element.
TEST(AggregatePhisTest, FloatLeavesOfConstants) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %out, i1 %c, float %f) {
entry:
  br i1 %c, label %a, label %join
a:
  %s = insertvalue { float, i32 } { float 2.5, i32 7 }, float %f, 0
  br i1 %c, label %b, label %join
b:
  br label %join
join:
  %p = phi { float, i32 } [ zeroinitializer, %entry ], [ %s, %a ], [ { float 2.5, i32 7 }, %b ]
  %x = extractvalue { float, i32 } %p, 0
  %n = extractvalue { float, i32 } %p, 1
  store float %x, ptr addrspace(1) %out, align 4
  %o = getelementptr i32, ptr addrspace(1) %out, i64 1
  store i32 %n, ptr addrspace(1) %o, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  Function *k = module->getFunction("k");
  Constant *original = nullptr;
  for (Instruction &inst : instructions(*k))
    if (auto *phi = dyn_cast<PHINode>(&inst))
      original =
          cast<Constant>(incomingFrom(phi, "b"))->getAggregateElement(0u);
  EXPECT_TRUE(splitAggregatePhis(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  EXPECT_EQ(aggregatePhis(*k), 0u) << print(*module);
  PHINode *x = nullptr, *n = nullptr;
  for (Instruction &inst : instructions(*k))
    if (auto *store = dyn_cast<StoreInst>(&inst)) {
      if (store->getValueOperand()->getType()->isFloatTy())
        x = phiOf(store->getValueOperand());
      else
        n = phiOf(store->getValueOperand());
    }
  ASSERT_NE(x, nullptr) << print(*module);
  ASSERT_NE(n, nullptr) << print(*module);
  auto *zero = dyn_cast<ExtractValueInst>(incomingFrom(x, "entry"));
  ASSERT_NE(zero, nullptr) << print(*module);
  EXPECT_EQ(zero->getParent()->getName(), "entry");
  EXPECT_TRUE(isa<ConstantAggregateZero>(zero->getAggregateOperand()));
  EXPECT_EQ(incomingFrom(x, "a"), k->getArg(2)) << print(*module);
  EXPECT_EQ(incomingFrom(x, "b"), original) << print(*module);
  EXPECT_TRUE(cast<ConstantInt>(incomingFrom(n, "entry"))->isZero());
  EXPECT_EQ(cast<ConstantInt>(incomingFrom(n, "a"))->getZExtValue(), 7u);
}

// A phi used whole (stored, passed to a call) or in part gets that part
// rebuilt from its leaf phis with insertvalues on poison; a split phi feeding
// another (a loop's phi into its exit's) gives its leaf phis directly; an
// incoming value that is neither a constant nor an insertvalue is read with
// extractvalues at the end of its block; a block coming in twice (a switch)
// gets one value per leaf.
TEST(AggregatePhisTest, WholeUsesAndPhiChains) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
declare void @use({ i64, { ptr addrspace(1), i64 } })
define void @f(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %sel) {
entry:
  %loaded = load { i64, { ptr addrspace(1), i64 } }, ptr addrspace(1) %in, align 8
  switch i32 %sel, label %head [ i32 1, label %done
                                 i32 2, label %done ]
head:
  %p = phi { i64, { ptr addrspace(1), i64 } } [ %loaded, %entry ], [ %q, %head ]
  %q = insertvalue { i64, { ptr addrspace(1), i64 } } %p, i64 5, 0
  %again = icmp eq i32 %sel, 0
  br i1 %again, label %head, label %done
done:
  %r = phi { i64, { ptr addrspace(1), i64 } } [ %loaded, %entry ], [ %loaded, %entry ], [ %p, %head ]
  store { i64, { ptr addrspace(1), i64 } } %r, ptr addrspace(1) %out, align 8
  call void @use({ i64, { ptr addrspace(1), i64 } } %r)
  %inner = extractvalue { i64, { ptr addrspace(1), i64 } } %r, 1
  %o = getelementptr i8, ptr addrspace(1) %out, i64 64
  store { ptr addrspace(1), i64 } %inner, ptr addrspace(1) %o, align 8
  ret void
}
)");
  ASSERT_TRUE(module);
  Function *f = module->getFunction("f");
  EXPECT_TRUE(splitAggregatePhis(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  EXPECT_EQ(aggregatePhis(*f), 0u) << print(*module);

  // Each leaf phi of %r takes %p's leaf phi from %head and one extract of
  // %loaded, in %entry, for both of %entry's entries.
  unsigned leaves = 0;
  for (Instruction &inst : instructions(*f)) {
    auto *phi = dyn_cast<PHINode>(&inst);
    if (!phi || phi->getParent()->getName() != "done")
      continue;
    ++leaves;
    ASSERT_EQ(phi->getNumIncomingValues(), 3u);
    Value *fromEntry = nullptr;
    for (unsigned i = 0; i < 3; ++i) {
      Value *value = phi->getIncomingValue(i);
      if (phi->getIncomingBlock(i)->getName() == "head") {
        auto *leaf = dyn_cast<PHINode>(value);
        ASSERT_NE(leaf, nullptr) << print(*module);
        EXPECT_EQ(leaf->getParent()->getName(), "head");
        continue;
      }
      if (fromEntry)
        EXPECT_EQ(value, fromEntry) << print(*module);
      fromEntry = value;
      auto *extract = dyn_cast<ExtractValueInst>(value);
      ASSERT_NE(extract, nullptr) << print(*module);
      EXPECT_EQ(extract->getParent()->getName(), "entry");
    }
  }
  EXPECT_EQ(leaves, 3u) << print(*module);

  // The whole value is rebuilt once for the store and the call, the inner
  // struct once for its extract.
  unsigned wholeInserts = 0, innerInserts = 0;
  for (Instruction &inst : instructions(*f))
    if (auto *insert = dyn_cast<InsertValueInst>(&inst)) {
      if (insert->getParent()->getName() != "done")
        continue;
      if (insert->getType()->getStructNumElements() == 2 &&
          insert->getType()->getStructElementType(0)->isIntegerTy(64))
        ++wholeInserts;
      else
        ++innerInserts;
    }
  EXPECT_EQ(wholeInserts, 3u) << print(*module);
  EXPECT_EQ(innerInserts, 2u) << print(*module);
}

// A phi of more than 64 leaves is left whole.
TEST(AggregatePhisTest, LargeArraysStay) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define void @f(ptr addrspace(1) %out, i1 %c, [100 x i8] %a) {
entry:
  br i1 %c, label %x, label %join
x:
  br label %join
join:
  %p = phi [100 x i8] [ zeroinitializer, %entry ], [ %a, %x ]
  store [100 x i8] %p, ptr addrspace(1) %out, align 1
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_FALSE(splitAggregatePhis(*module));
  EXPECT_EQ(aggregatePhis(*module->getFunction("f")), 1u);
}

/// The leaf phi named `name` (its phi's name, a dot and its leaf number).
PHINode *leafPhi(Function &func, StringRef name) {
  for (Instruction &inst : instructions(func))
    if (auto *phi = dyn_cast<PHINode>(&inst))
      if (phi->getName() == name)
        return phi;
  return nullptr;
}

// A constant data array: its integer elements are the leaves themselves
// (read with getElementAsConstant), but a float element is read with an
// extractvalue of the constant at the end of the incoming block, not made a
// `float 1.0` constant (the plugin never creates or folds floating-point
// constants), and no float constant appears in the function.
TEST(AggregatePhisTest, ConstantDataArrays) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %out, i1 %c, { [2 x i32], [2 x float] } %a) {
entry:
  br i1 %c, label %x, label %join
x:
  br label %join
join:
  %p = phi { [2 x i32], [2 x float] } [ { [2 x i32] [i32 1, i32 2], [2 x float] [float 1.0, float 2.0] }, %entry ], [ %a, %x ]
  %i0 = extractvalue { [2 x i32], [2 x float] } %p, 0, 0
  %i1 = extractvalue { [2 x i32], [2 x float] } %p, 0, 1
  %f0 = extractvalue { [2 x i32], [2 x float] } %p, 1, 0
  %f1 = extractvalue { [2 x i32], [2 x float] } %p, 1, 1
  store i32 %i0, ptr addrspace(1) %out, align 4
  %o1 = getelementptr i32, ptr addrspace(1) %out, i64 1
  store i32 %i1, ptr addrspace(1) %o1, align 4
  %o2 = getelementptr i32, ptr addrspace(1) %out, i64 2
  store float %f0, ptr addrspace(1) %o2, align 4
  %o3 = getelementptr i32, ptr addrspace(1) %out, i64 3
  store float %f1, ptr addrspace(1) %o3, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  Function *k = module->getFunction("k");
  Constant *original = nullptr;
  for (Instruction &inst : instructions(*k))
    if (auto *phi = dyn_cast<PHINode>(&inst))
      original = cast<Constant>(incomingFrom(phi, "entry"));
  ASSERT_NE(original, nullptr);
  // The parser makes both arrays constant data arrays, the case under test.
  ASSERT_TRUE(isa<ConstantDataArray>(original->getAggregateElement(0u)));
  ASSERT_TRUE(isa<ConstantDataArray>(original->getAggregateElement(1u)));

  EXPECT_TRUE(splitAggregatePhis(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  EXPECT_EQ(aggregatePhis(*k), 0u) << print(*module);

  for (unsigned i = 0; i < 2; ++i) {
    PHINode *leaf = leafPhi(*k, ("p." + Twine(i)).str());
    ASSERT_NE(leaf, nullptr) << print(*module);
    auto *element = dyn_cast<ConstantInt>(incomingFrom(leaf, "entry"));
    ASSERT_NE(element, nullptr) << print(*module);
    EXPECT_EQ(element->getZExtValue(), i + 1);
  }
  for (unsigned i = 0; i < 2; ++i) {
    PHINode *leaf = leafPhi(*k, ("p." + Twine(2 + i)).str());
    ASSERT_NE(leaf, nullptr) << print(*module);
    auto *extract = dyn_cast<ExtractValueInst>(incomingFrom(leaf, "entry"));
    ASSERT_NE(extract, nullptr) << print(*module);
    EXPECT_EQ(extract->getParent()->getName(), "entry");
    EXPECT_EQ(extract->getAggregateOperand(), original);
    EXPECT_EQ(extract->getIndices(), ArrayRef<unsigned>({1, i}));
  }
  for (Instruction &inst : instructions(*k))
    for (Value *operand : inst.operands())
      EXPECT_FALSE(isa<ConstantFP>(operand)) << print(*module);
}

// A loop's leaf the loop never reads is a phi only it reads: it is erased,
// and so is the extractvalue of the entry value that fed only it.
TEST(AggregatePhisTest, DeadLeafExtractsRemoved) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define void @f(ptr addrspace(1) %out, { i64, i64 } %a, i64 %n) {
entry:
  br label %head
head:
  %p = phi { i64, i64 } [ %a, %entry ], [ %q, %head ]
  %i = extractvalue { i64, i64 } %p, 0
  %next = add i64 %i, 1
  %q = insertvalue { i64, i64 } %p, i64 %next, 0
  %stop = icmp eq i64 %next, %n
  br i1 %stop, label %exit, label %head
exit:
  store i64 %i, ptr addrspace(1) %out, align 8
  ret void
}
)");
  ASSERT_TRUE(module);
  Function *f = module->getFunction("f");
  EXPECT_TRUE(splitAggregatePhis(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  EXPECT_EQ(aggregatePhis(*f), 0u) << print(*module);

  unsigned phis = 0, extracts = 0;
  for (Instruction &inst : instructions(*f)) {
    phis += isa<PHINode>(inst);
    EXPECT_FALSE(isa<InsertValueInst>(inst)) << print(*module);
    if (auto *extract = dyn_cast<ExtractValueInst>(&inst)) {
      ++extracts;
      EXPECT_EQ(extract->getIndices(), ArrayRef<unsigned>({0}))
          << print(*module);
    }
  }
  EXPECT_EQ(phis, 1u) << print(*module);
  EXPECT_EQ(extracts, 1u) << print(*module);
  EXPECT_EQ(leafPhi(*f, "p.1"), nullptr) << print(*module);
}

// Two split phis that feed each other around a loop: each leaf phi of one
// takes the other's leaf phi, with nothing extracted from either.
TEST(AggregatePhisTest, PhisFeedingEachOther) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define void @f(ptr addrspace(1) %out, { i64, i64 } %init, i1 %c, i64 %n) {
entry:
  br label %head
head:
  %a = phi { i64, i64 } [ %init, %entry ], [ %b, %latch ]
  br i1 %c, label %l1, label %l2
l1:
  br label %latch
l2:
  %a2 = insertvalue { i64, i64 } %a, i64 %n, 1
  br label %latch
latch:
  %b = phi { i64, i64 } [ %a, %l1 ], [ %a2, %l2 ]
  %x = extractvalue { i64, i64 } %b, 0
  %stop = icmp eq i64 %x, %n
  br i1 %stop, label %exit, label %head
exit:
  %y = extractvalue { i64, i64 } %b, 1
  store i64 %x, ptr addrspace(1) %out, align 8
  %o = getelementptr i64, ptr addrspace(1) %out, i64 1
  store i64 %y, ptr addrspace(1) %o, align 8
  ret void
}
)");
  ASSERT_TRUE(module);
  Function *f = module->getFunction("f");
  Argument *n = f->getArg(3);
  EXPECT_TRUE(splitAggregatePhis(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  EXPECT_EQ(aggregatePhis(*f), 0u) << print(*module);

  for (unsigned i = 0; i < 2; ++i) {
    PHINode *a = leafPhi(*f, ("a." + Twine(i)).str());
    PHINode *b = leafPhi(*f, ("b." + Twine(i)).str());
    ASSERT_NE(a, nullptr) << print(*module);
    ASSERT_NE(b, nullptr) << print(*module);
    EXPECT_EQ(incomingFrom(a, "latch"), b) << print(*module);
    EXPECT_EQ(incomingFrom(b, "l1"), a) << print(*module);
    EXPECT_EQ(incomingFrom(b, "l2"), i == 0 ? static_cast<Value *>(a) : n)
        << print(*module);
  }
  for (Instruction &inst : instructions(*f)) {
    EXPECT_FALSE(isa<InsertValueInst>(inst)) << print(*module);
    if (auto *extract = dyn_cast<ExtractValueInst>(&inst)) {
      EXPECT_EQ(extract->getAggregateOperand(), f->getArg(1)) << print(*module);
      EXPECT_EQ(extract->getParent()->getName(), "entry");
    }
  }
}

// An insertvalue that reads itself, which the verifier allows in a block
// unreachable from the entry, coming into a split phi: following its
// aggregate operand would loop forever, so the leaf it doesn't insert is read
// with an extractvalue of it instead.
TEST(AggregatePhisTest, SelfReferentialInsertInUnreachableBlock) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define void @f(ptr addrspace(1) %out, { i64, i64 } %a) {
entry:
  br label %join
dead:
  %x = insertvalue { i64, i64 } %x, i64 1, 0
  br label %join
join:
  %p = phi { i64, i64 } [ %a, %entry ], [ %x, %dead ]
  %v = extractvalue { i64, i64 } %p, 1
  store i64 %v, ptr addrspace(1) %out, align 8
  ret void
}
)");
  ASSERT_TRUE(module);
  ASSERT_EQ(verify(*module), "");
  Function *f = module->getFunction("f");
  EXPECT_TRUE(splitAggregatePhis(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  EXPECT_EQ(aggregatePhis(*f), 0u) << print(*module);
  PHINode *leaf = leafPhi(*f, "p.1");
  ASSERT_NE(leaf, nullptr) << print(*module);
  auto *extract = dyn_cast<ExtractValueInst>(incomingFrom(leaf, "dead"));
  ASSERT_NE(extract, nullptr) << print(*module);
  EXPECT_EQ(extract->getAggregateOperand()->getName(), "x");
  EXPECT_EQ(extract->getIndices(), ArrayRef<unsigned>({1}));
}

// A split phi whose whole value reaches another split phi only through an
// insertvalue, with no loop: once the second phi is split, the chain that
// rebuilt the first for that insertvalue is dead, and deleting it also
// deletes the first phi's leaf phi it alone read (p.0). The leaves still
// read (p.1, r.1) stay; the one nothing reads (r.0) goes, with its extract.
TEST(AggregatePhisTest, LeafDeletedWithTheChainThatReadIt) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
declare { i64, i64 } @g(i64)

define void @f(ptr addrspace(1) %out, i1 %c, i1 %d) {
entry:
  br i1 %c, label %A, label %B
A:
  %va = call { i64, i64 } @g(i64 1)
  br label %merge
B:
  %vb = call { i64, i64 } @g(i64 2)
  br label %merge
merge:
  %p = phi { i64, i64 } [ %va, %A ], [ %vb, %B ]
  %q = insertvalue { i64, i64 } %p, i64 5, 0
  br i1 %d, label %next, label %C
C:
  %vc = call { i64, i64 } @g(i64 3)
  br label %next
next:
  %r = phi { i64, i64 } [ %q, %merge ], [ %vc, %C ]
  %x = extractvalue { i64, i64 } %r, 1
  store i64 %x, ptr addrspace(1) %out, align 8
  ret void
}
)");
  ASSERT_TRUE(module);
  ASSERT_EQ(verify(*module), "");
  Function *f = module->getFunction("f");
  EXPECT_TRUE(splitAggregatePhis(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  EXPECT_EQ(aggregatePhis(*f), 0u) << print(*module);

  PHINode *p1 = leafPhi(*f, "p.1");
  PHINode *r1 = leafPhi(*f, "r.1");
  ASSERT_NE(p1, nullptr) << print(*module);
  ASSERT_NE(r1, nullptr) << print(*module);
  EXPECT_EQ(leafPhi(*f, "p.0"), nullptr) << print(*module);
  EXPECT_EQ(leafPhi(*f, "r.0"), nullptr) << print(*module);
  EXPECT_EQ(incomingFrom(r1, "merge"), p1) << print(*module);
  unsigned phis = 0, extracts = 0;
  for (Instruction &inst : instructions(*f)) {
    phis += isa<PHINode>(inst);
    EXPECT_FALSE(isa<InsertValueInst>(inst)) << print(*module);
    if (auto *extract = dyn_cast<ExtractValueInst>(&inst)) {
      ++extracts;
      EXPECT_EQ(extract->getIndices(), ArrayRef<unsigned>({1}))
          << print(*module);
    }
  }
  EXPECT_EQ(phis, 2u) << print(*module);
  EXPECT_EQ(extracts, 3u) << print(*module);
  auto *store = dyn_cast<StoreInst>(f->back().getTerminator()->getPrevNode());
  ASSERT_NE(store, nullptr) << print(*module);
  EXPECT_EQ(store->getValueOperand(), r1) << print(*module);
}

} // namespace
