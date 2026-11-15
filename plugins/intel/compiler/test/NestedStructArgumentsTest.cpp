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
// Tests for nested struct arguments of internal functions passed as flat
// structs, on hand-written IR in SPIR-V numbering, as the plugin sees it
// before codegen.
//
//===----------------------------------------------------------------------===//

#include "NestedStructArguments.h"

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
using M::KGEN::Intel::flattenNestedStructArguments;

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

/// The leaf at `index` of the flat struct `value` was built from: follows
/// the insertvalue chain back from its end.
Value *insertedAt(Value *value, unsigned index) {
  while (auto *insert = dyn_cast<InsertValueInst>(value)) {
    if (insert->getNumIndices() == 1 && insert->getIndices()[0] == index)
      return insert->getInsertedValueOperand();
    value = insert->getAggregateOperand();
  }
  return nullptr;
}

// A SourceLocation-shaped argument of an internal function goes as a flat
// struct of its leaves, a constant one as a constant (IGC wrote the length
// over the pointer of the nested one).
TEST(NestedStructArgumentsTest, BecomeFlat) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@name = internal addrspace(1) constant [4 x i8] c"a.m\00"
define internal i64 @report(i32 %n, { i64, i64, { ptr addrspace(4), i64 } } %loc) noinline {
  %file = extractvalue { i64, i64, { ptr addrspace(4), i64 } } %loc, 2, 0
  %byte = load i8, ptr addrspace(4) %file, align 1
  %line = extractvalue { i64, i64, { ptr addrspace(4), i64 } } %loc, 0
  %name = extractvalue { i64, i64, { ptr addrspace(4), i64 } } %loc, 2
  %length = extractvalue { ptr addrspace(4), i64 } %name, 1
  %sum = add i64 %line, %length
  ret i64 %sum
}
define spir_kernel void @k(ptr addrspace(1) %out) {
  %r = call i64 @report(i32 1, { i64, i64, { ptr addrspace(4), i64 } } { i64 7, i64 9, { ptr addrspace(4), i64 } { ptr addrspace(4) addrspacecast (ptr addrspace(1) @name to ptr addrspace(4)), i64 3 } })
  store i64 %r, ptr addrspace(1) %out, align 8
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(flattenNestedStructArguments(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);

  Function *report = module->getFunction("report");
  ASSERT_NE(report, nullptr) << print(*module);
  ASSERT_EQ(report->arg_size(), 2u) << print(*module);
  EXPECT_TRUE(report->getArg(0)->getType()->isIntegerTy(32));
  auto *flat = cast<StructType>(report->getArg(1)->getType());
  ASSERT_EQ(flat->getNumElements(), 4u) << print(*module);
  EXPECT_TRUE(flat->getElementType(2)->isPointerTy());
  for (Type *field : flat->elements())
    EXPECT_FALSE(field->isAggregateType()) << print(*module);
  EXPECT_TRUE(report->hasFnAttribute(Attribute::NoInline));
  // Each read is of a field of the flat struct: `%name` is only read
  // through, so nothing is rebuilt.
  for (Instruction &inst : instructions(*report))
    if (auto *extract = dyn_cast<ExtractValueInst>(&inst))
      EXPECT_EQ(extract->getAggregateOperand(), report->getArg(1))
          << print(*module);
  for (Instruction &inst : instructions(*report))
    EXPECT_FALSE(isa<InsertValueInst>(&inst)) << print(*module);
  for (Instruction &inst : instructions(*module->getFunction("k")))
    if (auto *call = dyn_cast<CallInst>(&inst)) {
      ASSERT_EQ(call->getCalledFunction(), report);
      auto *constant = dyn_cast<Constant>(call->getArgOperand(1));
      ASSERT_NE(constant, nullptr) << print(*module);
      EXPECT_EQ(constant->getAggregateElement(2u)->stripPointerCasts(),
                module->getNamedGlobal("name"));
    }
}

// A nested struct result goes back flat: `List._realloc`'s
// `{ { ptr, i64 }, i64 }` reached the kernel with a null pointer.
// Each return builds the flat struct, each caller reads its leaves from it,
// and a use of a whole part gets the part rebuilt from them.
TEST(NestedStructArgumentsTest, ReturnValuesBecomeFlat) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define internal { { ptr addrspace(4), i64 }, i64 } @grow(ptr addrspace(4) %p, i64 %n, i1 %c) noinline {
  br i1 %c, label %a, label %b
a:
  %inner = insertvalue { ptr addrspace(4), i64 } poison, ptr addrspace(4) %p, 0
  %inner.1 = insertvalue { ptr addrspace(4), i64 } %inner, i64 %n, 1
  %outer = insertvalue { { ptr addrspace(4), i64 }, i64 } poison, { ptr addrspace(4), i64 } %inner.1, 0
  %outer.1 = insertvalue { { ptr addrspace(4), i64 }, i64 } %outer, i64 7, 1
  ret { { ptr addrspace(4), i64 }, i64 } %outer.1
b:
  ret { { ptr addrspace(4), i64 }, i64 } poison
}
define internal void @keep({ ptr addrspace(4), i64 } %part) noinline {
  ret void
}
define spir_kernel void @k(ptr addrspace(1) %out, ptr addrspace(4) %p, i1 %c) {
  %r = call { { ptr addrspace(4), i64 }, i64 } @grow(ptr addrspace(4) %p, i64 3, i1 %c)
  %data = extractvalue { { ptr addrspace(4), i64 }, i64 } %r, 0, 0
  %length = extractvalue { { ptr addrspace(4), i64 }, i64 } %r, 1
  store i64 %length, ptr addrspace(4) %data, align 8
  %part = extractvalue { { ptr addrspace(4), i64 }, i64 } %r, 0
  call void @keep({ ptr addrspace(4), i64 } %part)
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(flattenNestedStructArguments(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);

  Function *grow = module->getFunction("grow");
  ASSERT_NE(grow, nullptr) << print(*module);
  auto *flat = dyn_cast<StructType>(grow->getReturnType());
  ASSERT_NE(flat, nullptr) << print(*module);
  ASSERT_EQ(flat->getNumElements(), 3u) << print(*module);
  EXPECT_TRUE(flat->getElementType(0)->isPointerTy());
  for (Type *field : flat->elements())
    EXPECT_FALSE(field->isAggregateType()) << print(*module);
  EXPECT_TRUE(grow->hasFnAttribute(Attribute::NoInline));
  // The first return hands back the pointer leaf of the value it returned.
  for (BasicBlock &block : *grow)
    if (auto *ret = dyn_cast<ReturnInst>(block.getTerminator());
        ret && block.getName() == "a") {
      auto *pointer = dyn_cast_or_null<ExtractValueInst>(
          insertedAt(ret->getReturnValue(), 0));
      ASSERT_NE(pointer, nullptr) << print(*module);
      EXPECT_EQ(pointer->getAggregateOperand()->getName(), "outer.1");
      EXPECT_EQ(pointer->getIndices(), ArrayRef<unsigned>({0, 0}));
    }

  // The kernel reads each leaf from the flat result; only the part passed
  // whole is rebuilt, and nothing else is an aggregate of the old type.
  Function *kernel = module->getFunction("k");
  CallInst *call = nullptr;
  for (Instruction &inst : instructions(*kernel))
    if (auto *found = dyn_cast<CallInst>(&inst);
        found && found->getCalledFunction() == grow)
      call = found;
  ASSERT_NE(call, nullptr) << print(*module);
  auto *store =
      cast<StoreInst>(&*find_if(instructions(*kernel), [](Instruction &inst) {
        return isa<StoreInst>(inst);
      }));
  auto *data = dyn_cast<ExtractValueInst>(store->getPointerOperand());
  ASSERT_NE(data, nullptr) << print(*module);
  EXPECT_EQ(data->getAggregateOperand(), call);
  EXPECT_EQ(data->getIndices()[0], 0u);
  auto *length = dyn_cast<ExtractValueInst>(store->getValueOperand());
  ASSERT_NE(length, nullptr) << print(*module);
  EXPECT_EQ(length->getAggregateOperand(), call);
  EXPECT_EQ(length->getIndices()[0], 2u);
  for (Instruction &inst : instructions(*kernel))
    if (auto *keep = dyn_cast<CallInst>(&inst);
        keep && keep->getCalledFunction() == module->getFunction("keep")) {
      Value *part = keep->getArgOperand(0);
      EXPECT_EQ(insertedAt(part, 0), data) << print(*module);
    }
}

// A nested struct without leaves (a struct of empty structs) holds nothing
// to misread and keeps its type, as a result and as an argument: flattened
// to `{}`, its returns, its callers' uses and its argument's uses kept the
// old type (an assertion in RAUW). Next to a nested argument that is
// flattened, the result still keeps its type and the module stays valid.
TEST(NestedStructArgumentsTest, StructsWithoutLeavesStay) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define internal { {}, {} } @empty(i1 %c) noinline {
  ret { {}, {} } zeroinitializer
}
define internal { {}, {} } @mixed({ { i64 }, i64 } %s, ptr addrspace(1) %out) noinline {
  %a = extractvalue { { i64 }, i64 } %s, 0, 0
  store i64 %a, ptr addrspace(1) %out
  ret { {}, {} } zeroinitializer
}
define internal void @take({ {}, {} } %e) noinline {
  ret void
}
define spir_kernel void @k(ptr addrspace(1) %out) {
  %e = call { {}, {} } @empty(i1 true)
  call void @take({ {}, {} } %e)
  %m = call { {}, {} } @mixed({ { i64 }, i64 } zeroinitializer, ptr addrspace(1) %out)
  call void @take({ {}, {} } %m)
  ret void
}
)");
  ASSERT_TRUE(module);
  Type *nested = module->getFunction("empty")->getReturnType();
  EXPECT_TRUE(flattenNestedStructArguments(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  EXPECT_EQ(module->getFunction("empty")->getReturnType(), nested);
  Function *mixed = module->getFunction("mixed");
  ASSERT_NE(mixed, nullptr) << print(*module);
  EXPECT_EQ(mixed->getReturnType(), nested) << print(*module);
  auto *flat = dyn_cast<StructType>(mixed->getArg(0)->getType());
  ASSERT_NE(flat, nullptr) << print(*module);
  EXPECT_EQ(flat->getNumElements(), 2u) << print(*module);
  EXPECT_EQ(module->getFunction("take")->getArg(0)->getType(), nested);
}

// A function whose only nested types are without leaves is left alone:
// nothing to flatten, nothing changes.
TEST(NestedStructArgumentsTest, AResultWithoutLeavesAloneChangesNothing) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define internal { {}, {} } @empty(i1 %c) noinline {
  ret { {}, {} } zeroinitializer
}
define internal void @take({ {}, {} } %e) noinline {
  ret void
}
define spir_kernel void @k(ptr addrspace(1) %out) {
  %e = call { {}, {} } @empty(i1 true)
  call void @take({ {}, {} } %e)
  ret void
}
)");
  ASSERT_TRUE(module);
  std::string before = print(*module);
  EXPECT_FALSE(flattenNestedStructArguments(*module));
  EXPECT_EQ(print(*module), before);
}

TEST(NestedStructArgumentsTest, StayWhenTheAddressIsTaken) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@table = internal global ptr @f
define internal i64 @f({ i64, { i64 } } %a) {
  %v = extractvalue { i64, { i64 } } %a, 1, 0
  ret i64 %v
}
define internal i64 @flat({ i64, i64 } %a) {
  %v = extractvalue { i64, i64 } %a, 1
  ret i64 %v
}
define spir_kernel void @k(ptr addrspace(1) %out) {
  %r = call i64 @f({ i64, { i64 } } zeroinitializer)
  %s = call i64 @flat({ i64, i64 } zeroinitializer)
  ret void
}
)");
  ASSERT_TRUE(module);
  std::string before = print(*module);
  EXPECT_FALSE(flattenNestedStructArguments(*module));
  EXPECT_EQ(print(*module), before);
}

// An argument, or a part of one, used other than through extractvalue (passed
// whole to a call, stored, merged by a phi) is rebuilt from the leaves at the
// function's entry, after the reads of the leaves it inserts.
TEST(NestedStructArgumentsTest, WholeUsesAreRebuiltAfterTheLeaves) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
declare void @sink({ i64, { ptr addrspace(4), i64 } })
define internal void @use({ i64, { ptr addrspace(4), i64 } } %loc, i1 %c, ptr addrspace(1) %out) noinline {
entry:
  %line = extractvalue { i64, { ptr addrspace(4), i64 } } %loc, 0
  call void @sink({ i64, { ptr addrspace(4), i64 } } %loc)
  %name = extractvalue { i64, { ptr addrspace(4), i64 } } %loc, 1
  store { ptr addrspace(4), i64 } %name, ptr addrspace(1) %out, align 8
  br i1 %c, label %then, label %join
then:
  br label %join
join:
  %merged = phi { ptr addrspace(4), i64 } [ %name, %entry ], [ zeroinitializer, %then ]
  %length = extractvalue { ptr addrspace(4), i64 } %merged, 1
  store i64 %length, ptr addrspace(1) %out, align 8
  store i64 %line, ptr addrspace(1) %out, align 8
  ret void
}
define spir_kernel void @k(ptr addrspace(4) %p, i1 %c, ptr addrspace(1) %out) {
  %a = insertvalue { i64, { ptr addrspace(4), i64 } } poison, i64 1, 0
  %b = insertvalue { i64, { ptr addrspace(4), i64 } } %a, ptr addrspace(4) %p, 1, 0
  %d = insertvalue { i64, { ptr addrspace(4), i64 } } %b, i64 3, 1, 1
  call void @use({ i64, { ptr addrspace(4), i64 } } %d, i1 %c, ptr addrspace(1) %out)
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(flattenNestedStructArguments(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);

  Function *use = module->getFunction("use");
  ASSERT_NE(use, nullptr) << print(*module);
  Argument *flat = use->getArg(0);
  ASSERT_EQ(cast<StructType>(flat->getType())->getNumElements(), 3u);
  // Every extractvalue reads a leaf of the flat struct, once, in the entry.
  unsigned reads = 0;
  for (Instruction &inst : instructions(*use))
    if (auto *extract = dyn_cast<ExtractValueInst>(&inst)) {
      if (extract->getAggregateOperand() == flat) {
        EXPECT_EQ(extract->getParent(), &use->getEntryBlock());
        ++reads;
      } else {
        // Only %merged, a phi of a rebuilt part, is read through.
        EXPECT_TRUE(isa<PHINode>(extract->getAggregateOperand()))
            << print(*module);
      }
    }
  EXPECT_EQ(reads, 3u) << print(*module);

  // @sink gets the whole argument rebuilt from all three leaves.
  CallInst *sink = nullptr;
  for (Instruction &inst : instructions(*use))
    if (auto *call = dyn_cast<CallInst>(&inst))
      sink = call;
  ASSERT_NE(sink, nullptr);
  auto *whole = dyn_cast<InsertValueInst>(sink->getArgOperand(0));
  ASSERT_NE(whole, nullptr) << print(*module);
  unsigned inserted = 0;
  for (Value *value = whole; auto *insert = dyn_cast<InsertValueInst>(value);
       value = insert->getAggregateOperand()) {
    auto *leaf = dyn_cast<ExtractValueInst>(insert->getInsertedValueOperand());
    ASSERT_NE(leaf, nullptr) << print(*module);
    EXPECT_EQ(leaf->getAggregateOperand(), flat);
    ++inserted;
  }
  EXPECT_EQ(inserted, 3u) << print(*module);

  // The stored part and the phi's incoming value are the `{ ptr, i64 }`
  // rebuilt from its two leaves.
  auto *phi = cast<PHINode>(&*module->getFunction("use")->back().begin());
  auto *part = dyn_cast<InsertValueInst>(phi->getIncomingValue(0));
  ASSERT_NE(part, nullptr) << print(*module);
  EXPECT_EQ(part->getParent(), &use->getEntryBlock());
}

// A floating-point leaf of a constant argument is read with an instruction:
// folding it out of the constant would create a floating-point constant
// (the zero of a `zeroinitializer`), which the plugin must never do.
TEST(NestedStructArgumentsTest, FloatingPointLeavesOfConstantsAreNotFolded) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define internal float @f({ i64, { float, i64 } } %a) noinline {
  %v = extractvalue { i64, { float, i64 } } %a, 1, 0
  ret float %v
}
define spir_kernel void @k(ptr addrspace(1) %out) {
  %r = call float @f({ i64, { float, i64 } } zeroinitializer)
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(flattenNestedStructArguments(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  CallInst *call = nullptr;
  for (Instruction &inst : instructions(*module->getFunction("k")))
    if (auto *found = dyn_cast<CallInst>(&inst))
      call = found;
  ASSERT_NE(call, nullptr);
  Value *leaf = insertedAt(call->getArgOperand(0), 1);
  ASSERT_NE(leaf, nullptr) << print(*module);
  EXPECT_TRUE(isa<ExtractValueInst>(leaf)) << print(*module);
  EXPECT_FALSE(isa<ConstantFP>(leaf)) << print(*module);
}

} // namespace
