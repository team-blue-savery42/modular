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
// Tests for rewriting the users of lane masks bitcast to odd-width integers,
// on hand-written IR as the plugin sees it before codegen. Each rewritten
// function is evaluated for every mask and compared with what the original
// IR computes.
//
//===----------------------------------------------------------------------===//

#include "MaskBitcasts.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <bit>
#include <cstdint>
#include <memory>
#include <string>

using namespace llvm;
using M::KGEN::Intel::expandMaskTruncs;
using M::KGEN::Intel::rewriteMaskBitcasts;

namespace {

std::unique_ptr<Module> parse(LLVMContext &ctx, const std::string &body) {
  SMDiagnostic diag;
  std::unique_ptr<Module> module = parseAssemblyString(
      "target datalayout = \"e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-"
      "v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1\"\n"
      "target triple = \"spirv64-unknown-unknown\"\n" +
          body,
      diag, ctx);
  if (!module) {
    std::string text;
    raw_string_ostream os(text);
    diag.print("test", os);
    ADD_FAILURE() << text;
  }
  return module;
}

/// `value` truncated to `bits` bits.
uint64_t truncate(uint64_t value, unsigned bits) {
  return bits >= 64 ? value : value & ((uint64_t(1) << bits) - 1);
}

/// `value`, of `bits` bits, sign-extended to 64.
int64_t signExtend(uint64_t value, unsigned bits) {
  return bits >= 64 ? int64_t(value)
                    : int64_t(value << (64 - bits)) >> (64 - bits);
}

/// What `value` computes when the mask argument's lane i is bit i of `lanes`,
/// zero-extended to 64 bits; handles what the rewrite emits.
uint64_t evaluate(const Value *value, uint64_t lanes) {
  unsigned bits = value->getType()->getIntegerBitWidth();
  if (const auto *constant = dyn_cast<ConstantInt>(value))
    return truncate(constant->getValue().getZExtValue(), bits);
  if (const auto *extract = dyn_cast<ExtractElementInst>(value)) {
    uint64_t lane =
        cast<ConstantInt>(extract->getIndexOperand())->getZExtValue();
    return (lanes >> lane) & 1;
  }
  if (const auto *zext = dyn_cast<ZExtInst>(value))
    return evaluate(zext->getOperand(0), lanes);
  if (const auto *select = dyn_cast<SelectInst>(value))
    return evaluate(select->getCondition(), lanes)
               ? evaluate(select->getTrueValue(), lanes)
               : evaluate(select->getFalseValue(), lanes);
  if (const auto *cmp = dyn_cast<ICmpInst>(value)) {
    unsigned width = cmp->getOperand(0)->getType()->getIntegerBitWidth();
    uint64_t a = evaluate(cmp->getOperand(0), lanes);
    uint64_t b = evaluate(cmp->getOperand(1), lanes);
    int64_t sa = signExtend(a, width), sb = signExtend(b, width);
    switch (cmp->getPredicate()) {
    case ICmpInst::ICMP_EQ:
      return a == b;
    case ICmpInst::ICMP_NE:
      return a != b;
    case ICmpInst::ICMP_ULT:
      return a < b;
    case ICmpInst::ICMP_ULE:
      return a <= b;
    case ICmpInst::ICMP_UGT:
      return a > b;
    case ICmpInst::ICMP_UGE:
      return a >= b;
    case ICmpInst::ICMP_SLT:
      return sa < sb;
    case ICmpInst::ICMP_SLE:
      return sa <= sb;
    case ICmpInst::ICMP_SGT:
      return sa > sb;
    case ICmpInst::ICMP_SGE:
      return sa >= sb;
    default:
      ADD_FAILURE() << "unexpected predicate";
      return 0;
    }
  }
  const auto *op = dyn_cast<BinaryOperator>(value);
  if (op == nullptr) {
    ADD_FAILURE() << "unexpected value " << value->getName().str();
    return 0;
  }
  uint64_t a = evaluate(op->getOperand(0), lanes);
  uint64_t b = evaluate(op->getOperand(1), lanes);
  switch (op->getOpcode()) {
  case Instruction::And:
    return a & b;
  case Instruction::Or:
    return a | b;
  case Instruction::Xor:
    return truncate(a ^ b, bits);
  case Instruction::Add:
    return truncate(a + b, bits);
  case Instruction::Shl:
    return truncate(a << b, bits);
  default:
    ADD_FAILURE() << "unexpected " << op->getOpcodeName();
    return 0;
  }
}

const Value *returned(const Function &function) {
  for (const Instruction &inst : instructions(function))
    if (const auto *ret = dyn_cast<ReturnInst>(&inst))
      return ret->getReturnValue();
  return nullptr;
}

bool hasBitcast(const Function &function) {
  for (const Instruction &inst : instructions(function))
    if (isa<BitCastInst>(inst))
      return true;
  return false;
}

/// Whether any instruction of `function` makes or takes an integer IGC
/// can't compile.
bool hasOddWidthInteger(const Function &function) {
  auto odd = [](const Type *type) {
    if (!type->isIntegerTy())
      return false;
    unsigned bits = type->getIntegerBitWidth();
    return bits != 1 && bits != 8 && bits != 16 && bits != 32 && bits != 64;
  };
  for (const Instruction &inst : instructions(function)) {
    if (odd(inst.getType()))
      return true;
    for (const Value *operand : inst.operands())
      if (odd(operand->getType()))
        return true;
  }
  return false;
}

/// Checks `function`'s result for every mask of `lanes` lanes against
/// `expected`, which computes it from the mask's bits.
template <typename Expected>
void expectResults(const Module &module, const char *name, unsigned lanes,
                   Expected expected) {
  const Function *function = module.getFunction(name);
  ASSERT_NE(function, nullptr) << name;
  EXPECT_FALSE(hasBitcast(*function)) << name;
  EXPECT_FALSE(hasOddWidthInteger(*function)) << name;
  const Value *result = returned(*function);
  ASSERT_NE(result, nullptr);
  unsigned bits = result->getType()->getIntegerBitWidth();
  for (uint64_t mask = 0; mask < (uint64_t(1) << lanes); ++mask)
    EXPECT_EQ(evaluate(result, mask), truncate(expected(mask), bits))
        << name << " with mask " << mask;
}

TEST(MaskBitcastsTest, AnyAllAndOtherConstantsBecomeLaneCompares) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define i1 @any4(<4 x i1> %m) {
  %b = bitcast <4 x i1> %m to i4
  %c = icmp ne i4 %b, 0
  ret i1 %c
}
define i1 @all4(<4 x i1> %m) {
  %b = bitcast <4 x i1> %m to i4
  %c = icmp eq i4 %b, -1
  ret i1 %c
}
define i1 @five_on_the_left(<4 x i1> %m) {
  %b = bitcast <4 x i1> %m to i4
  %c = icmp eq i4 5, %b
  ret i1 %c
}
define i1 @none2(<2 x i1> %m) {
  %b = bitcast <2 x i1> %m to i2
  %c = icmp eq i2 %b, 0
  ret i1 %c
}
)");
  ASSERT_TRUE(module);
  Expected<bool> changed = rewriteMaskBitcasts(*module);
  ASSERT_TRUE(bool(changed)) << toString(changed.takeError());
  EXPECT_TRUE(*changed);
  EXPECT_FALSE(verifyModule(*module, &errs()));
  expectResults(*module, "any4", 4, [](uint64_t m) { return m != 0; });
  expectResults(*module, "all4", 4, [](uint64_t m) { return m == 0xf; });
  expectResults(*module, "five_on_the_left", 4,
                [](uint64_t m) { return m == 5; });
  expectResults(*module, "none2", 2, [](uint64_t m) { return m == 0; });
}

// What LLVM's combiner makes of masks besides any/all tests: the bits as an
// integer, ordered and signed compares, a count of set lanes, a round trip.
TEST(MaskBitcastsTest, OtherUsersAreRewrittenOnTheLanes) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
declare i4 @llvm.ctpop.i4(i4)
define i32 @zext4(<4 x i1> %m) {
  %b = bitcast <4 x i1> %m to i4
  %z = zext i4 %b to i32
  ret i32 %z
}
define i8 @sext2(<2 x i1> %m) {
  %b = bitcast <2 x i1> %m to i2
  %s = sext i2 %b to i8
  ret i8 %s
}
define i1 @below5(<4 x i1> %m) {
  %b = bitcast <4 x i1> %m to i4
  %c = icmp ult i4 %b, 5
  ret i1 %c
}
define i1 @negative(<4 x i1> %m) {
  %b = bitcast <4 x i1> %m to i4
  %c = icmp slt i4 %b, 0
  ret i1 %c
}
define i1 @above_minus3(<4 x i1> %m) {
  %b = bitcast <4 x i1> %m to i4
  %c = icmp sgt i4 -3, %b
  ret i1 %c
}
define i32 @popcount(<4 x i1> %m) {
  %b = bitcast <4 x i1> %m to i4
  %p = call i4 @llvm.ctpop.i4(i4 %b)
  %z = zext i4 %p to i32
  ret i32 %z
}
define i1 @at_least_3(<4 x i1> %m) {
  %b = bitcast <4 x i1> %m to i4
  %p = call i4 @llvm.ctpop.i4(i4 %b)
  %c = icmp uge i4 %p, 3
  ret i1 %c
}
define i32 @round_trip(<4 x i1> %m) {
  %b = bitcast <4 x i1> %m to i4
  %c = icmp ne i4 %b, 0
  %v = bitcast i4 %b to <4 x i1>
  %l = extractelement <4 x i1> %v, i32 2
  %s = select i1 %c, i1 %l, i1 false
  %z = zext i1 %s to i32
  ret i32 %z
}
)");
  ASSERT_TRUE(module);
  Expected<bool> changed = rewriteMaskBitcasts(*module);
  ASSERT_TRUE(bool(changed)) << toString(changed.takeError());
  EXPECT_TRUE(*changed);
  EXPECT_FALSE(verifyModule(*module, &errs()));
  EXPECT_EQ(module->getFunction("llvm.ctpop.i4"), nullptr)
      << "the unused i4 declaration stays";
  expectResults(*module, "zext4", 4, [](uint64_t m) { return m; });
  expectResults(*module, "sext2", 2,
                [](uint64_t m) { return uint64_t(signExtend(m, 2)); });
  expectResults(*module, "below5", 4, [](uint64_t m) { return m < 5; });
  expectResults(*module, "negative", 4,
                [](uint64_t m) { return signExtend(m, 4) < 0; });
  expectResults(*module, "above_minus3", 4,
                [](uint64_t m) { return -3 > signExtend(m, 4); });
  expectResults(*module, "popcount", 4,
                [](uint64_t m) { return uint64_t(std::popcount(m)); });
  expectResults(*module, "at_least_3", 4,
                [](uint64_t m) { return std::popcount(m) >= 3; });
  expectResults(*module, "round_trip", 4,
                [](uint64_t m) { return m != 0 && ((m >> 2) & 1); });
}

// A use the rewrite doesn't know fails the compile, naming it, rather than
// reach IGC; masks of legal widths are left alone.
TEST(MaskBitcastsTest, UnknownUsersFailAndLegalWidthsStay) {
  LLVMContext ctx;
  std::unique_ptr<Module> unknown = parse(ctx, R"(
define i4 @selected(<4 x i1> %m, i1 %c) {
  %b = bitcast <4 x i1> %m to i4
  %s = select i1 %c, i4 %b, i4 0
  ret i4 %s
}
)");
  ASSERT_TRUE(unknown);
  Expected<bool> failed = rewriteMaskBitcasts(*unknown);
  ASSERT_FALSE(bool(failed));
  std::string message = toString(failed.takeError());
  EXPECT_NE(message.find("4-bit integer"), std::string::npos) << message;
  EXPECT_NE(message.find("select i1 %c, i4 %b, i4 0"), std::string::npos)
      << message;
  EXPECT_NE(message.find("selected"), std::string::npos) << message;

  std::unique_ptr<Module> legal = parse(ctx, R"(
define i1 @byte(<8 x i1> %m) {
  %b = bitcast <8 x i1> %m to i8
  %c = icmp ne i8 %b, 0
  ret i1 %c
}
)");
  ASSERT_TRUE(legal);
  Expected<bool> changed = rewriteMaskBitcasts(*legal);
  ASSERT_TRUE(bool(changed)) << toString(changed.takeError());
  EXPECT_FALSE(*changed);
  EXPECT_TRUE(hasBitcast(*legal->getFunction("byte")));
}

TEST(MaskBitcastsTest, MaskTruncsBecomeLowBitCompares) {
  // Each truncation to a lane mask becomes `icmp ne (and x, 1), 0`, and the
  // lanes agree with the truncation's for every value of a lane's low bits.
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define <8 x i1> @t8(<8 x i32> %x) {
  %m = trunc <8 x i32> %x to <8 x i1>
  ret <8 x i1> %m
}
define <4 x i1> @t4(<4 x i16> %x) {
  %m = trunc <4 x i16> %x to <4 x i1>
  ret <4 x i1> %m
}
define i1 @scalar(i32 %x) {
  %m = trunc i32 %x to i1
  ret i1 %m
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(expandMaskTruncs(*module));
  EXPECT_FALSE(verifyModule(*module, &errs()));
  for (const char *name : {"t8", "t4"}) {
    Function &function = *module->getFunction(name);
    unsigned truncs = 0, compares = 0;
    for (Instruction &inst : instructions(function)) {
      truncs += isa<TruncInst>(inst);
      if (auto *cmp = dyn_cast<ICmpInst>(&inst)) {
        ++compares;
        EXPECT_EQ(cmp->getPredicate(), ICmpInst::ICMP_NE);
        EXPECT_TRUE(isa<Constant>(cmp->getOperand(1)) &&
                    cast<Constant>(cmp->getOperand(1))->isNullValue());
        auto *low = dyn_cast<BinaryOperator>(cmp->getOperand(0));
        ASSERT_TRUE(low && low->getOpcode() == Instruction::And);
        auto *one = dyn_cast<Constant>(low->getOperand(1));
        ASSERT_TRUE(one && one->getSplatValue() &&
                    cast<ConstantInt>(one->getSplatValue())->isOne());
      }
    }
    EXPECT_EQ(truncs, 0u) << name;
    EXPECT_EQ(compares, 1u) << name;
  }
  // A scalar truncation is left alone (the backend lowers it right).
  unsigned scalarTruncs = 0;
  for (Instruction &inst : instructions(*module->getFunction("scalar")))
    scalarTruncs += isa<TruncInst>(inst);
  EXPECT_EQ(scalarTruncs, 1u);
  // Nothing left to do the second time.
  EXPECT_FALSE(expandMaskTruncs(*module));
}

} // namespace
