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
// Tests for 128-bit integers split into 64-bit halves, on hand-written
// IR. The arithmetic tests' function takes its 128-bit operands as 64-bit
// halves, computes with i128 inside and returns the result's halves; the
// memory tests' kernels load and store i128s, through globals, phis and
// calls. After the rewrite no i128 may be left, and an interpreter of the
// rewritten IR must give what APInt gives, on operands with zero, one,
// all-ones, sign-bit and random halves. A 64-bit shift by 64 or more fails
// the test: it is poison.
//
//===----------------------------------------------------------------------===//

#include "Int128.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace llvm;
using M::KGEN::Intel::lowerInt128;

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

/// Whether any instruction, argument or global of `module` is or computes
/// on an i128.
bool hasWide(const Module &module) {
  auto wide = [](Type *type) { return type->isIntegerTy(128); };
  for (const GlobalVariable &global : module.globals())
    if (wide(global.getValueType()))
      return true;
  for (const Function &func : module) {
    for (Type *param : func.getFunctionType()->params())
      if (wide(param))
        return true;
    if (wide(func.getReturnType()))
      return true;
    for (const Instruction &inst : instructions(func)) {
      if (wide(inst.getType()))
        return true;
      for (const Value *op : inst.operands())
        if (wide(op->getType()))
          return true;
    }
  }
  return false;
}

/// A value while interpreting: its scalars (one for an integer, the fields
/// of a `{ i64, i64 }` or the elements of a vector; a pointer is its object
/// and byte offset).
using Scalars = std::vector<APInt>;

/// Interprets the integer, memory and control-flow instructions the lowered
/// code uses, on 8-byte memory cells. A 64-bit shift by 64 or more (poison),
/// a read of a cell never written, or anything it doesn't know fails the
/// test.
class Interpreter {
public:
  explicit Interpreter(const Module &module) : layout(module.getDataLayout()) {
    for (const GlobalVariable &global : module.globals()) {
      objects[&global] = nextObject;
      if (global.hasInitializer())
        fill(nextObject, 0, global.getInitializer());
      ++nextObject;
    }
  }

  /// A new object, for an argument pointer.
  Scalars object() { return {APInt(64, nextObject++), APInt(64, 0)}; }

  Scalars call(const Function &func, const std::vector<Scalars> &args) {
    DenseMap<const Value *, Scalars> frame;
    for (const Argument &arg : func.args())
      frame[&arg] = args[arg.getArgNo()];
    const BasicBlock *block = &func.getEntryBlock();
    const BasicBlock *from = nullptr;
    for (int steps = 0; steps < 100000; ++steps) {
      // A block's phis read the values as they were on entry.
      std::vector<std::pair<const PHINode *, Scalars>> entry;
      for (const PHINode &phi : block->phis())
        entry.push_back({&phi, get(frame, phi.getIncomingValueForBlock(from))});
      for (auto &[phi, value] : entry)
        frame[phi] = value;
      for (const Instruction &inst : *block) {
        if (isa<PHINode>(inst))
          continue;
        if (const auto *ret = dyn_cast<ReturnInst>(&inst))
          return ret->getReturnValue() ? get(frame, ret->getReturnValue())
                                       : Scalars();
        if (const auto *br = dyn_cast<CondBrInst>(&inst)) {
          from = block;
          block = br->getSuccessor(
              get(frame, br->getCondition())[0].getBoolValue() ? 0 : 1);
          break;
        }
        if (const auto *br = dyn_cast<UncondBrInst>(&inst)) {
          from = block;
          block = br->getSuccessor();
          break;
        }
        frame[&inst] = execute(frame, inst);
      }
    }
    ADD_FAILURE() << "too many steps";
    return {};
  }

  /// The 8-byte cells of memory, by object and offset.
  std::map<std::pair<uint64_t, int64_t>, APInt> memory;

private:
  void fill(uint64_t object, int64_t offset, const Constant *constant) {
    Type *type = constant->getType();
    if (const auto *integer = dyn_cast<ConstantInt>(constant)) {
      for (unsigned bit = 0; bit < integer->getBitWidth(); bit += 64)
        memory[{object, offset + bit / 8}] =
            integer->getValue().extractBits(64, bit);
      return;
    }
    if (const auto *data = dyn_cast<ConstantDataSequential>(constant)) {
      for (unsigned i = 0; i < data->getNumElements(); ++i)
        fill(object, offset + 8 * i, data->getElementAsConstant(i));
      return;
    }
    if (isa<ConstantAggregateZero>(constant)) {
      for (uint64_t at = 0; at < layout.getTypeAllocSize(type); at += 8)
        memory[{object, offset + static_cast<int64_t>(at)}] = APInt(64, 0);
      return;
    }
    if (auto *structType = dyn_cast<StructType>(type)) {
      const StructLayout *fields = layout.getStructLayout(structType);
      for (unsigned i = 0; i < structType->getNumElements(); ++i)
        fill(object, offset + fields->getElementOffset(i),
             constant->getAggregateElement(i));
      return;
    }
    if (const auto *array = dyn_cast<ArrayType>(type)) {
      uint64_t stride = layout.getTypeAllocSize(array->getElementType());
      for (unsigned i = 0; i < array->getNumElements(); ++i)
        fill(object, offset + stride * i, constant->getAggregateElement(i));
      return;
    }
    ADD_FAILURE() << "unexpected initializer";
  }

  static unsigned count(Type *type) {
    if (type->isPointerTy())
      return 2;
    if (auto *vector = dyn_cast<FixedVectorType>(type))
      return vector->getNumElements();
    if (type->isAggregateType())
      return type->getStructNumElements();
    return 1;
  }

  static unsigned width(Type *type) {
    if (auto *vector = dyn_cast<FixedVectorType>(type))
      return vector->getScalarSizeInBits();
    if (auto *structType = dyn_cast<StructType>(type))
      return structType->getElementType(0)->getIntegerBitWidth();
    return type->getPrimitiveSizeInBits();
  }

  Scalars get(DenseMap<const Value *, Scalars> &frame, const Value *value) {
    auto found = frame.find(value);
    if (found != frame.end())
      return found->second;
    if (const auto *integer = dyn_cast<ConstantInt>(value))
      return {integer->getValue()};
    if (const auto *global = dyn_cast<GlobalVariable>(value))
      return {APInt(64, objects.lookup(global)), APInt(64, 0)};
    if (isa<UndefValue>(value) || isa<ConstantAggregateZero>(value))
      return Scalars(
          count(value->getType()),
          APInt(value->getType()->isPointerTy() ? 64 : width(value->getType()),
                0));
    if (const auto *expr = dyn_cast<ConstantExpr>(value)) {
      Instruction *inst = expr->getAsInstruction();
      Scalars result = execute(frame, *inst);
      inst->deleteValue();
      return result;
    }
    ADD_FAILURE() << "unknown value";
    return Scalars(count(value->getType()), APInt(64, 0));
  }

  Scalars execute(DenseMap<const Value *, Scalars> &frame,
                  const Instruction &inst) {
    auto op = [&](unsigned i) { return get(frame, inst.getOperand(i)); };
    auto one = [&](unsigned i) { return op(i)[0]; };
    unsigned bits = inst.getType()->isIntegerTy()
                        ? inst.getType()->getIntegerBitWidth()
                        : 0;
    auto amount = [&](unsigned i) {
      APInt by = one(i);
      if (by.uge(bits)) {
        ADD_FAILURE() << "a shift of an i" << bits << " by "
                      << by.getZExtValue() << " (poison)";
        return 0u;
      }
      return static_cast<unsigned>(by.getZExtValue());
    };
    switch (inst.getOpcode()) {
    case Instruction::Trunc:
      return {one(0).trunc(bits)};
    case Instruction::ZExt:
      return {one(0).zext(bits)};
    case Instruction::SExt:
      return {one(0).sext(bits)};
    case Instruction::Shl:
      return {one(0).shl(amount(1))};
    case Instruction::LShr:
      return {one(0).lshr(amount(1))};
    case Instruction::AShr:
      return {one(0).ashr(amount(1))};
    case Instruction::And:
      return {one(0) & one(1)};
    case Instruction::Or:
      return {one(0) | one(1)};
    case Instruction::Xor:
      return {one(0) ^ one(1)};
    case Instruction::Add:
      return {one(0) + one(1)};
    case Instruction::Sub:
      return {one(0) - one(1)};
    case Instruction::Mul:
      return {one(0) * one(1)};
    case Instruction::Freeze:
    case Instruction::AddrSpaceCast:
      return op(0);
    case Instruction::Select:
      return one(0).getBoolValue() ? op(1) : op(2);
    case Instruction::ICmp:
      return {APInt(1, ICmpInst::compare(one(0), one(1),
                                         cast<ICmpInst>(inst).getPredicate()))};
    case Instruction::InsertValue: {
      Scalars result = op(0);
      result[cast<InsertValueInst>(inst).getIndices()[0]] = one(1);
      return result;
    }
    case Instruction::ExtractValue:
      return {op(0)[cast<ExtractValueInst>(inst).getIndices()[0]]};
    case Instruction::InsertElement: {
      Scalars result = op(0);
      result[one(2).getZExtValue()] = one(1);
      return result;
    }
    case Instruction::ExtractElement:
      return {op(0)[one(1).getZExtValue()]};
    case Instruction::ShuffleVector: {
      Scalars x = op(0), y = op(1), result;
      for (int lane : cast<ShuffleVectorInst>(inst).getShuffleMask())
        result.push_back(
            lane < static_cast<int>(x.size()) ? x[lane] : y[lane - x.size()]);
      return result;
    }
    case Instruction::BitCast: {
      // Through the bits, element 0 lowest.
      Scalars from = op(0);
      unsigned fromWidth = width(inst.getOperand(0)->getType());
      APInt all(fromWidth * from.size(), 0);
      for (unsigned i = 0; i < from.size(); ++i)
        all.insertBits(from[i], i * fromWidth);
      unsigned toWidth = width(inst.getType());
      Scalars result;
      for (unsigned i = 0; i < count(inst.getType()); ++i)
        result.push_back(all.extractBits(toWidth, i * toWidth));
      return result;
    }
    case Instruction::Alloca:
      return object();
    case Instruction::GetElementPtr: {
      const auto &gep = cast<GetElementPtrInst>(inst);
      Scalars pointer = op(0);
      int64_t offset = pointer[1].getSExtValue();
      for (gep_type_iterator it = gep_type_begin(gep), end = gep_type_end(gep);
           it != end; ++it) {
        int64_t index = get(frame, it.getOperand())[0].getSExtValue();
        if (StructType *structType = it.getStructTypeOrNull())
          offset += layout.getStructLayout(structType)
                        ->getElementOffset(static_cast<unsigned>(index));
        else
          offset += index *
                    static_cast<int64_t>(it.getSequentialElementStride(layout));
      }
      return {pointer[0], APInt(64, offset)};
    }
    case Instruction::Load: {
      Scalars pointer = op(0);
      auto cell =
          memory.find({pointer[0].getZExtValue(), pointer[1].getSExtValue()});
      if (!inst.getType()->isIntegerTy(64) || cell == memory.end()) {
        ADD_FAILURE() << "a load of unwritten memory or of a type not i64";
        return {APInt(64, 0)};
      }
      return {cell->second};
    }
    case Instruction::Store: {
      Scalars pointer = op(1);
      if (!inst.getOperand(0)->getType()->isIntegerTy(64))
        ADD_FAILURE() << "a store of a type not i64";
      memory[{pointer[0].getZExtValue(), pointer[1].getSExtValue()}] = one(0);
      return {};
    }
    case Instruction::Call: {
      const auto &call = cast<CallInst>(inst);
      std::vector<Scalars> args;
      for (const Use &arg : call.args())
        args.push_back(get(frame, arg));
      return this->call(*call.getCalledFunction(), args);
    }
    default:
      ADD_FAILURE() << "unexpected instruction " << inst.getOpcodeName();
      return {APInt(64, 0)};
    }
  }

  const DataLayout &layout;
  DenseMap<const GlobalVariable *, uint64_t> objects;
  uint64_t nextObject = 1;
};

/// 64-bit halves: zero, one, small, 32-bit edges, the sign bit and below it,
/// all ones, and two arbitrary values.
const uint64_t kHalves[] = {0,
                            1,
                            3,
                            0xffffffffULL,
                            0x100000000ULL,
                            0x7fffffffffffffffULL,
                            0x8000000000000000ULL,
                            0x8fe78ab543e1de28ULL,
                            0x3c89ed7106ddc391ULL,
                            0xffffffffffffffffULL};

/// 128-bit operands from every pair of halves, and random ones.
std::vector<APInt> operands() {
  std::vector<APInt> values;
  for (uint64_t high : kHalves)
    for (uint64_t low : kHalves)
      values.push_back(APInt(128, {low, high}));
  std::mt19937_64 random(23);
  for (int i = 0; i < 20; ++i) {
    uint64_t low = random();
    uint64_t high = random();
    values.push_back(APInt(128, {low, high}));
  }
  return values;
}

/// Internal functions taking and returning i128, as dragonbox's helpers
/// would if not inlined.
constexpr const char *kCallees = R"(
define internal i128 @umul128(i64 %x, i128 noundef %y) {
  %xz = zext i64 %x to i128
  %p = mul i128 %xz, %y
  ret i128 %p
}

define internal fastcc i128 @add(i128 %x, i128 %y) {
  %s = add i128 %x, %y
  ret i128 %s
}
)";

/// A function `f(a.lo, a.hi, b.lo, b.hi)` that rebuilds `%a` and `%b` as i128,
/// runs `body` (which defines the i128 `%r`), and returns `%r`'s halves.
/// The internal functions of `kCallees` are there for it to call.
std::string function(const std::string &body) {
  return std::string(kCallees) + R"(
define { i64, i64 } @f(i64 %al, i64 %ah, i64 %bl, i64 %bh) {
  %a0 = zext i64 %al to i128
  %a1 = zext i64 %ah to i128
  %a2 = shl i128 %a1, 64
  %a = or i128 %a2, %a0
  %b0 = zext i64 %bl to i128
  %b1 = zext i64 %bh to i128
  %b2 = shl i128 %b1, 64
  %b = or i128 %b2, %b0
)" + body +
         R"(
  %rl = trunc i128 %r to i64
  %r1 = lshr i128 %r, 64
  %rh = trunc i128 %r1 to i64
  %s0 = insertvalue { i64, i64 } poison, i64 %rl, 0
  %s1 = insertvalue { i64, i64 } %s0, i64 %rh, 1
  ret { i64, i64 } %s1
}
)";
}

using Reference = std::function<APInt(const APInt &, const APInt &)>;

/// Lowers `function(body)` and checks it against `want` on `as` x `bs`.
void check(const std::string &body, const Reference &want,
           const std::vector<APInt> &as, const std::vector<APInt> &bs) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, function(body));
  ASSERT_TRUE(module);
  Expected<bool> lowered = lowerInt128(*module);
  ASSERT_TRUE(static_cast<bool>(lowered)) << toString(lowered.takeError());
  EXPECT_TRUE(*lowered);
  EXPECT_FALSE(verifyModule(*module, &errs()));
  EXPECT_FALSE(hasWide(*module)) << print(*module);
  const Function *func = module->getFunction("f");
  int wrong = 0;
  for (const APInt &a : as)
    for (const APInt &b : bs) {
      Interpreter interpreter(*module);
      Scalars got = interpreter.call(*func, {{a.trunc(64)},
                                             {a.extractBits(64, 64)},
                                             {b.trunc(64)},
                                             {b.extractBits(64, 64)}});
      APInt result = got[1].zext(128).shl(64) | got[0].zext(128);
      APInt expected = want(a, b);
      if (result != expected && ++wrong <= 3)
        ADD_FAILURE() << "a " << toString(a, 16, false) << " b "
                      << toString(b, 16, false) << ": got "
                      << toString(result, 16, false) << ", expected "
                      << toString(expected, 16, false);
    }
  EXPECT_EQ(wrong, 0) << body;
}

void check(const std::string &body, const Reference &want) {
  check(body, want, operands(), operands());
}

/// For a `body` that reads only `%a` (or only `%a`'s halves).
void checkUnary(const std::string &body, const Reference &want) {
  check(body, want, operands(), {APInt(128, 0)});
}

/// Shift amounts: every case of a half boundary.
std::vector<APInt> amounts() {
  std::vector<APInt> values;
  for (uint64_t n : {0, 1, 2, 31, 32, 33, 63, 64, 65, 96, 100, 126, 127})
    values.push_back(APInt(128, n));
  return values;
}

TEST(Int128Test, AddWithCarry) {
  check("%r = add i128 %a, %b",
        [](const APInt &a, const APInt &b) { return a + b; });
}

TEST(Int128Test, SubtractWithBorrow) {
  check("%r = sub i128 %a, %b",
        [](const APInt &a, const APInt &b) { return a - b; });
}

TEST(Int128Test, MultiplyLow128Bits) {
  check("%r = mul i128 %a, %b",
        [](const APInt &a, const APInt &b) { return a * b; });
}

// 64 x 64 -> 128, with zero-extended operands.
TEST(Int128Test, MultiplyWidening) {
  check(R"(
  %az = zext i64 %al to i128
  %bz = zext i64 %bl to i128
  %r = mul i128 %az, %bz
)",
        [](const APInt &a, const APInt &b) {
          return a.trunc(64).zext(128) * b.trunc(64).zext(128);
        });
}

// What LLVM makes of dragonbox's _umul128_upper64, the form IGC gets
// wrong: a mask with the 128-bit constant 2^64 - 1, a product, its high half.
TEST(Int128Test, MultiplyMaskedHighHalf) {
  check(R"(
  %am = and i128 %a, 18446744073709551615
  %bz = zext i64 %bl to i128
  %p = mul nuw i128 %am, %bz
  %r = lshr i128 %p, 64
)",
        [](const APInt &a, const APInt &b) {
          return (a.trunc(64).zext(128) * b.trunc(64).zext(128)).lshr(64);
        });
}

TEST(Int128Test, MultiplyByConstants) {
  checkUnary("%r = mul i128 %a, 340282366920938463463374607431768211455",
             [](const APInt &a, const APInt &) { return -a; });
  checkUnary("%r = mul i128 %a, 1512366075204170947332355369683137040",
             [](const APInt &a, const APInt &) {
               return a * APInt(128,
                                {0xfedcba9876543210ULL, 0x0123456789abcdefULL});
             });
}

TEST(Int128Test, Logic) {
  check("%r = and i128 %a, %b",
        [](const APInt &a, const APInt &b) { return a & b; });
  check("%r = or i128 %a, %b",
        [](const APInt &a, const APInt &b) { return a | b; });
  check("%r = xor i128 %a, %b",
        [](const APInt &a, const APInt &b) { return a ^ b; });
}

// IGC keeps only the low 32 bits of a 128-bit constant operand.
TEST(Int128Test, WideConstantOperands) {
  APInt mixed(128, {0xfedcba9876543210ULL, 0x0123456789abcdefULL});
  checkUnary("%r = and i128 %a, 18446744073709551615",
             [](const APInt &a, const APInt &) {
               return a & APInt::getLowBitsSet(128, 64);
             });
  checkUnary("%r = xor i128 %a, 1512366075204170947332355369683137040",
             [&](const APInt &a, const APInt &) { return a ^ mixed; });
  checkUnary("%r = or i128 %a, -18446744073709551616",
             [](const APInt &a, const APInt &) {
               return a | APInt::getHighBitsSet(128, 64);
             });
  checkUnary("%r = add i128 %a, -1",
             [](const APInt &a, const APInt &) { return a - 1; });
  checkUnary(
      "%r = sub i128 18446744073709551616, %a",
      [](const APInt &a, const APInt &) { return APInt(128, {0, 1}) - a; });
}

TEST(Int128Test, ShiftByConstant) {
  for (const APInt &n : amounts()) {
    unsigned by = static_cast<unsigned>(n.getZExtValue());
    std::string amount = std::to_string(by);
    checkUnary("%r = shl i128 %a, " + amount,
               [by](const APInt &a, const APInt &) { return a.shl(by); });
    checkUnary("%r = lshr i128 %a, " + amount,
               [by](const APInt &a, const APInt &) { return a.lshr(by); });
    checkUnary("%r = ashr i128 %a, " + amount,
               [by](const APInt &a, const APInt &) { return a.ashr(by); });
  }
}

TEST(Int128Test, ShiftByValue) {
  auto by = [](const APInt &b) {
    return static_cast<unsigned>(b.getZExtValue());
  };
  check(
      "%r = shl i128 %a, %b",
      [&](const APInt &a, const APInt &b) { return a.shl(by(b)); }, operands(),
      amounts());
  check(
      "%r = lshr i128 %a, %b",
      [&](const APInt &a, const APInt &b) { return a.lshr(by(b)); }, operands(),
      amounts());
  check(
      "%r = ashr i128 %a, %b",
      [&](const APInt &a, const APInt &b) { return a.ashr(by(b)); }, operands(),
      amounts());
}

TEST(Int128Test, Compares) {
  for (auto predicate :
       {CmpInst::ICMP_EQ, CmpInst::ICMP_NE, CmpInst::ICMP_UGT,
        CmpInst::ICMP_UGE, CmpInst::ICMP_ULT, CmpInst::ICMP_ULE,
        CmpInst::ICMP_SGT, CmpInst::ICMP_SGE, CmpInst::ICMP_SLT,
        CmpInst::ICMP_SLE}) {
    std::string name = CmpInst::getPredicateName(predicate).str();
    check("%c = icmp " + name + " i128 %a, %b\n  %r = zext i1 %c to i128",
          [predicate](const APInt &a, const APInt &b) {
            return APInt(128, ICmpInst::compare(a, b, predicate));
          });
  }
}

TEST(Int128Test, ExtendAndTruncate) {
  check(R"(
  %t = trunc i128 %a to i32
  %r = sext i32 %t to i128
)",
        [](const APInt &a, const APInt &) { return a.trunc(32).sext(128); });
  check("%r = sext i64 %bh to i128", [](const APInt &, const APInt &b) {
    return b.extractBits(64, 64).sext(128);
  });
  check(R"(
  %c = icmp ult i128 %a, %b
  %r = zext i1 %c to i128
)",
        [](const APInt &a, const APInt &b) { return APInt(128, a.ult(b)); });
  check(R"(
  %t = trunc i128 %b to i16
  %r = zext i16 %t to i128
)",
        [](const APInt &, const APInt &b) { return b.trunc(16).zext(128); });
}

TEST(Int128Test, SelectAndFreeze) {
  check(R"(
  %c = icmp slt i128 %a, %b
  %m = select i1 %c, i128 %a, i128 %b
  %r = freeze i128 %m
)",
        [](const APInt &a, const APInt &b) { return a.slt(b) ? a : b; });
  check(R"(
  %c = icmp eq i64 %al, 0
  %r = select i1 %c, i128 poison, i128 %b
)",
        [](const APInt &a, const APInt &b) {
          return a.trunc(64).isZero() ? APInt(128, 0) : b;
        });
}

// A chain as dragonbox computes it: a widening product's high half plus a
// shifted partial sum (_umul192_upper128).
TEST(Int128Test, DragonboxChain) {
  check(R"(
  %am = and i128 %a, 18446744073709551615
  %bz = zext i64 %bl to i128
  %p = mul nuw i128 %am, %bz
  %h = lshr i128 %p, 64
  %hz = zext i64 %ah to i128
  %s = add nuw nsw i128 %h, %hz
  %bs = shl nuw i128 %bz, 64
  %r = add i128 %s, %bs
)",
        [](const APInt &a, const APInt &b) {
          APInt m = (a.trunc(64).zext(128) * b.trunc(64).zext(128)).lshr(64);
          return m + a.extractBits(64, 64).zext(128) +
                 b.trunc(64).zext(128).shl(64);
        });
}

/// Lowers `ir`, which must succeed and leave no i128.
std::unique_ptr<Module> lowered(LLVMContext &ctx, const std::string &ir) {
  std::unique_ptr<Module> module = parse(ctx, ir);
  if (!module)
    return nullptr;
  Expected<bool> changed = lowerInt128(*module);
  if (!changed) {
    ADD_FAILURE() << toString(changed.takeError());
    return nullptr;
  }
  EXPECT_TRUE(*changed);
  EXPECT_FALSE(verifyModule(*module, &errs()));
  EXPECT_FALSE(hasWide(*module)) << print(*module);
  return module;
}

/// The error lowering `ir` gives, which must fail.
std::string refusal(const std::string &ir) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, ir);
  if (!module)
    return "";
  Expected<bool> changed = lowerInt128(*module);
  if (changed) {
    ADD_FAILURE() << "not refused";
    return "";
  }
  return toString(changed.takeError());
}

/// `in`'s 128-bit values in an interpreter's memory as object `in`.
void store(Interpreter &interpreter, const Scalars &in,
           const std::vector<APInt> &values) {
  for (size_t i = 0; i < values.size(); ++i)
    for (unsigned half = 0; half < 2; ++half)
      interpreter.memory[{in[0].getZExtValue(), 16 * i + 8 * half}] =
          values[i].extractBits(64, 64 * half);
}

/// The 128-bit value at byte `offset` of object `out`.
APInt loaded(Interpreter &interpreter, const Scalars &out, int64_t offset) {
  APInt value(128, 0);
  for (unsigned half = 0; half < 2; ++half) {
    auto cell =
        interpreter.memory.find({out[0].getZExtValue(), offset + 8 * half});
    if (cell == interpreter.memory.end()) {
      ADD_FAILURE() << "nothing stored at " << offset + 8 * half;
      return value;
    }
    value.insertBits(cell->second, 64 * half);
  }
  return value;
}

// mul128.mojo's kernel: loads, the forms IGC gets wrong, stores.
TEST(Int128Test, LoadsAndStores) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = lowered(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %a = load i128, ptr addrspace(1) %in, align 8
  %p = getelementptr i8, ptr addrspace(1) %in, i64 16
  %b = load volatile i128, ptr addrspace(1) %p, align 16
  %s = add i128 %a, %b
  store i128 %s, ptr addrspace(1) %out, align 8
  %am = and i128 %a, 18446744073709551615
  %bm = and i128 %b, 18446744073709551615
  %m = mul nuw i128 %am, %bm
  %o1 = getelementptr i8, ptr addrspace(1) %out, i64 16
  store i128 %m, ptr addrspace(1) %o1, align 8
  %h = lshr i128 %m, 64
  %o2 = getelementptr i8, ptr addrspace(1) %out, i64 32
  store i128 %h, ptr addrspace(1) %o2, align 8
  ret void
}
)");
  ASSERT_TRUE(module);
  const Function *kernel = module->getFunction("k");
  // The volatile load's halves stay volatile; the high one is 8-aligned.
  std::vector<uint64_t> aligns;
  for (const Instruction &inst : instructions(*kernel))
    if (const auto *load = dyn_cast<LoadInst>(&inst);
        load && load->isVolatile())
      aligns.push_back(load->getAlign().value());
  EXPECT_EQ(aligns, (std::vector<uint64_t>{16, 8}));
  for (const APInt &a : operands())
    for (const APInt &b : {operands()[11], operands()[99], operands()[105]}) {
      Interpreter interpreter(*module);
      Scalars in = interpreter.object(), out = interpreter.object();
      store(interpreter, in, {a, b});
      interpreter.call(*kernel, {in, out});
      APInt m = a.trunc(64).zext(128) * b.trunc(64).zext(128);
      EXPECT_EQ(loaded(interpreter, out, 0), a + b);
      EXPECT_EQ(loaded(interpreter, out, 16), m);
      EXPECT_EQ(loaded(interpreter, out, 32), m.lshr(64));
    }
}

// A loop carrying an i128 through phis, and a select of two.
TEST(Int128Test, PhisAndSelects) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = lowered(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %sum = phi i128 [ 0, %entry ], [ %add, %loop ]
  %big = phi i128 [ 0, %entry ], [ %max, %loop ]
  %offset = shl i64 %i, 4
  %p = getelementptr i8, ptr addrspace(1) %in, i64 %offset
  %v = load i128, ptr addrspace(1) %p, align 8
  %add = add i128 %sum, %v
  %gt = icmp ugt i128 %v, %big
  %max = select i1 %gt, i128 %v, i128 %big
  %next = add i64 %i, 1
  %done = icmp eq i64 %next, 8
  br i1 %done, label %exit, label %loop
exit:
  store i128 %add, ptr addrspace(1) %out, align 8
  %o = getelementptr i8, ptr addrspace(1) %out, i64 16
  store i128 %max, ptr addrspace(1) %o, align 8
  ret void
}
)");
  ASSERT_TRUE(module);
  std::vector<APInt> all = operands();
  for (size_t first = 0; first + 8 <= all.size(); first += 8) {
    std::vector<APInt> values(all.begin() + first, all.begin() + first + 8);
    Interpreter interpreter(*module);
    Scalars in = interpreter.object(), out = interpreter.object();
    store(interpreter, in, values);
    interpreter.call(*module->getFunction("k"), {in, out});
    APInt sum(128, 0), max(128, 0);
    for (const APInt &v : values) {
      sum += v;
      max = v.ugt(max) ? v : max;
    }
    EXPECT_EQ(loaded(interpreter, out, 0), sum);
    EXPECT_EQ(loaded(interpreter, out, 16), max);
  }
}

// Float formatting's table: a program-scope constant struct holding an
// array of i128, read through a byte GEP; and a typed GEP and a constant GEP
// expression over i128 (IGC master asserts on a program-scope i128).
TEST(Int128Test, GlobalTables) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = lowered(ctx, R"(
@table = internal addrspace(1) constant { [3 x i128] } { [3 x i128] [i128 -707734658592001144036750341951516805, i128 1512366075204170947332355369683137040, i128 1] }, align 8
@pairs = internal addrspace(1) global [2 x { i64, i128 }] [{ i64, i128 } { i64 7, i128 -1 }, { i64, i128 } zeroinitializer]

define spir_kernel void @k(ptr addrspace(1) %out, i64 %i) {
  %p = getelementptr inbounds nuw [16 x i8], ptr addrspace(4) addrspacecast (ptr addrspace(1) @table to ptr addrspace(4)), i64 %i
  %v = load i128, ptr addrspace(4) %p, align 8
  store i128 %v, ptr addrspace(1) %out, align 8
  %q = getelementptr [2 x { i64, i128 }], ptr addrspace(1) @pairs, i64 0, i64 %i, i32 1
  %w = load i128, ptr addrspace(1) %q, align 8
  %o1 = getelementptr i8, ptr addrspace(1) %out, i64 16
  store i128 %w, ptr addrspace(1) %o1, align 8
  %x = load i128, ptr addrspace(1) getelementptr ([2 x { i64, i128 }], ptr addrspace(1) @pairs, i64 0, i64 0, i32 1), align 8
  %o2 = getelementptr i8, ptr addrspace(1) %out, i64 32
  store i128 %x, ptr addrspace(1) %o2, align 8
  ret void
}
)");
  ASSERT_TRUE(module);
  GlobalVariable *table = module->getGlobalVariable("table", true);
  ASSERT_TRUE(table);
  EXPECT_TRUE(table->isConstant());
  EXPECT_EQ(table->getAlign(), MaybeAlign(8));
  EXPECT_EQ(table->getAddressSpace(), 1u);
  EXPECT_TRUE(module->getGlobalVariable("pairs", true));
  const APInt values[] = {
      APInt(128, "-707734658592001144036750341951516805", 10),
      APInt(128, {0xfedcba9876543210ULL, 0x0123456789abcdefULL}),
      APInt(128, 1)};
  for (uint64_t i = 0; i < 2; ++i) {
    Interpreter interpreter(*module);
    Scalars out = interpreter.object();
    interpreter.call(*module->getFunction("k"), {out, {APInt(64, i)}});
    EXPECT_EQ(loaded(interpreter, out, 0), values[i]);
    EXPECT_EQ(loaded(interpreter, out, 16),
              i == 0 ? APInt::getAllOnes(128) : APInt(128, 0));
    EXPECT_EQ(loaded(interpreter, out, 32), APInt::getAllOnes(128));
  }
}

// An i128 on the stack, and through vectors.
TEST(Int128Test, AllocasAndBitcasts) {
  check(R"(
  %m = alloca i128, align 8
  store i128 %a, ptr %m, align 8
  %v = load i128, ptr %m, align 8
  %w = bitcast i128 %v to <4 x i32>
  %x = shufflevector <4 x i32> %w, <4 x i32> poison, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  %r = bitcast <4 x i32> %x to i128
)",
        [](const APInt &a, const APInt &) {
          APInt r(128, 0);
          for (unsigned i = 0; i < 4; ++i)
            r.insertBits(a.extractBits(32, 32 * (i ^ 1)), 32 * i);
          return r;
        },
        operands(), {APInt(128, 0)});
}

// Calls of kCallees: each i128 parameter becomes two i64s, the result a
// pair.
TEST(Int128Test, Calls) {
  check(R"(
  %p = call i128 @umul128(i64 %al, i128 noundef %b)
  %r = call fastcc i128 @add(i128 %p, i128 %a)
)",
        [](const APInt &a, const APInt &b) {
          return a.trunc(64).zext(128) * b + a;
        });
}

TEST(Int128Test, CalleesKeepTheirNamesAndAttributes) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = lowered(ctx, std::string(kCallees) + R"(
define i64 @f(i64 %a) {
  %w = zext i64 %a to i128
  %p = call fastcc i128 @add(i128 %w, i128 %w)
  %t = trunc i128 %p to i64
  ret i64 %t
}
)");
  ASSERT_TRUE(module);
  Function *add = module->getFunction("add");
  ASSERT_TRUE(add);
  EXPECT_EQ(add->getCallingConv(), CallingConv::Fast);
  EXPECT_EQ(add->arg_size(), 4u);
  EXPECT_EQ(add->getArg(0)->getName(), "x.lo");
  Function *umul = module->getFunction("umul128");
  ASSERT_TRUE(umul);
  EXPECT_EQ(umul->arg_size(), 3u);
  EXPECT_FALSE(umul->hasParamAttribute(1, Attribute::NoUndef));
}

TEST(Int128Test, Refusals) {
  EXPECT_NE(refusal(R"(
define spir_kernel void @k(i128 %x) {
  ret void
}
)")
                .find("`k` takes or returns 128-bit integers"),
            std::string::npos);
  EXPECT_NE(refusal(R"(
declare i128 @external(i128)
define i64 @f(i64 %a) {
  %w = zext i64 %a to i128
  %r = call i128 @external(i128 %w)
  %t = trunc i128 %r to i64
  ret i64 %t
}
)")
                .find("`call external` in `f`"),
            std::string::npos);
  EXPECT_NE(refusal(R"(
define i64 @f(ptr %p) {
  %s = load { i128, i64 }, ptr %p, align 8
  %t = extractvalue { i128, i64 } %s, 1
  ret i64 %t
}
)")
                .find("an aggregate holding 128-bit integers"),
            std::string::npos);
  EXPECT_NE(refusal(R"(
@g = internal global <2 x i128> zeroinitializer
)")
                .find("`g` holds 128-bit integers: a vector"),
            std::string::npos);
}

TEST(Int128Test, DivisionIsRefused) {
  LLVMContext ctx;
  std::unique_ptr<Module> module =
      parse(ctx, function("%r = udiv i128 %a, %b"));
  ASSERT_TRUE(module);
  Expected<bool> lowered = lowerInt128(*module);
  ASSERT_FALSE(static_cast<bool>(lowered));
  std::string message = toString(lowered.takeError());
  EXPECT_NE(message.find("`udiv` in `f`"), std::string::npos) << message;
}

TEST(Int128Test, FloatConversionIsRefused) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define double @f(i64 %x) {
  %w = zext i64 %x to i128
  %d = uitofp i128 %w to double
  ret double %d
}
)");
  ASSERT_TRUE(module);
  Expected<bool> lowered = lowerInt128(*module);
  ASSERT_FALSE(static_cast<bool>(lowered));
  std::string message = toString(lowered.takeError());
  EXPECT_NE(message.find("`uitofp` in `f`"), std::string::npos) << message;
}

TEST(Int128Test, ModulesWithoutWideIntegersAreUnchanged) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@g = internal addrspace(1) constant [2 x i64] [i64 1, i64 2]
define i64 @f(i64 %a, double %d, <2 x i64> %v) {
  %p = mul i64 %a, %a
  %x = fadd double %d, 1.5
  %e = extractelement <2 x i64> %v, i32 1
  %s = add i64 %p, %e
  ret i64 %s
}
)");
  ASSERT_TRUE(module);
  std::string before = print(*module);
  Expected<bool> lowered = lowerInt128(*module);
  ASSERT_TRUE(static_cast<bool>(lowered)) << toString(lowered.takeError());
  EXPECT_FALSE(*lowered);
  EXPECT_EQ(before, print(*module));
}

} // namespace
