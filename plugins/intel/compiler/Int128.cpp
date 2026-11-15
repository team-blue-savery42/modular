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

#include "Int128.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Operator.h"
#include "llvm/Transforms/Utils/Local.h"

#include <string>
#include <utility>

using namespace llvm;

namespace M::KGEN::Intel {

namespace {

// Nothing is folded: the folder would also fold the floating-point constants
// of a bitcast, with this copy of LLVM's APFloat (see PromoteBFloat16.cpp).
// The arithmetic below skips the operations a constant half makes trivial.
using Builder = IRBuilder<NoFolder>;

/// An `i128` as its low and high 64 bits.
struct Halves {
  Value *low = nullptr;
  Value *high = nullptr;
};

bool isWide(const Type *type) { return type->isIntegerTy(128); }

/// Whether `type` is or holds an `i128`: a vector, struct or array of them,
/// or a function type that takes or returns one.
bool holdsWide(Type *type, SmallPtrSetImpl<Type *> &seen) {
  if (isWide(type))
    return true;
  if (!seen.insert(type).second)
    return false;
  return any_of(type->subtypes(),
                [&](Type *sub) { return holdsWide(sub, seen); });
}

bool holdsWide(Type *type) {
  SmallPtrSet<Type *, 8> seen;
  return holdsWide(type, seen);
}

/// Whether `inst` computes on, makes, stores or addresses an `i128`.
bool touchesWide(Instruction &inst) {
  if (holdsWide(inst.getType()))
    return true;
  if (any_of(inst.operands(),
             [](Use &use) { return holdsWide(use->getType()); }))
    return true;
  if (auto *alloca = dyn_cast<AllocaInst>(&inst))
    return holdsWide(alloca->getAllocatedType());
  if (auto *gep = dyn_cast<GetElementPtrInst>(&inst))
    return holdsWide(gep->getSourceElementType());
  if (auto *call = dyn_cast<CallBase>(&inst))
    return holdsWide(call->getFunctionType());
  return false;
}

bool isConstant(Value *value, uint64_t constant) {
  auto *integer = dyn_cast<ConstantInt>(value);
  return integer && integer->getValue() == constant;
}

bool isAllOnes(Value *value) {
  auto *integer = dyn_cast<ConstantInt>(value);
  return integer && integer->isMinusOne();
}

/// What `inst` is, for a diagnostic: its opcode, and the callee of a call.
/// The instruction itself isn't printed: printing a float constant would run
/// this copy of LLVM's APFloat.
std::string describe(const Instruction &inst) {
  std::string text = inst.getOpcodeName();
  if (const auto *call = dyn_cast<CallBase>(&inst)) {
    if (const Function *callee = call->getCalledFunction())
      text += " " + callee->getName().str();
    else
      text += " (indirect)";
  }
  return text;
}

class Lowering {
public:
  explicit Lowering(Module &module)
      : module(module), i64(Type::getInt64Ty(module.getContext())) {}

  Expected<bool> run() {
    bool changed = false;
    for (GlobalVariable &global : make_early_inc_range(module.globals()))
      if (holdsWide(global.getValueType())) {
        if (Error err = retypeGlobal(global))
          return std::move(err);
        changed = true;
      }
    for (Function &func : make_early_inc_range(module))
      if (!func.isDeclaration() && holdsWide(func.getFunctionType())) {
        if (Error err = splitSignature(func))
          return std::move(err);
        changed = true;
      }
    for (Function &func : module) {
      if (func.isDeclaration())
        continue;
      changed |= retypeConstantGEPs(func);
      Expected<bool> lowered = lowerFunction(func);
      if (!lowered)
        return lowered.takeError();
      changed |= *lowered;
    }
    // The functions whose signatures were split: their calls now call the
    // copies, which take their names.
    for (auto &[old, copy] : rewritten) {
      if (!old->use_empty())
        return createStringError(inconvertibleErrorCode(),
                                 "intel plugin: `" + old->getName() +
                                     "` is still used after its 128-bit "
                                     "arguments were split");
      copy->takeName(old);
      old->eraseFromParent();
    }
    // Declarations nothing calls any more, such as an intrinsic on i128.
    SmallVector<Function *> unused;
    for (Function &func : module)
      if (func.isDeclaration() && func.use_empty() &&
          holdsWide(func.getFunctionType()))
        unused.push_back(&func);
    for (Function *func : unused)
      func->eraseFromParent();
    changed |= !unused.empty();
    if (Error err = checkNoneLeft())
      return std::move(err);
    return changed;
  }

private:
  Error unsupported(const Instruction &inst, const Twine &why = "") {
    return createStringError(
        inconvertibleErrorCode(),
        "intel plugin: cannot lower the 128-bit integer operation `" +
            describe(inst) + "` in `" + inst.getFunction()->getName() + "`" +
            (why.isTriviallyEmpty() ? Twine("") : ": " + why) +
            " (IGC computes 128-bit integers wrongly)");
  }

  Expected<bool> lowerFunction(Function &func) {
    if (none_of(instructions(func),
                [](Instruction &inst) { return touchesWide(inst); }))
      return false;
    // Unreachable code may use a value before its definition; it goes.
    removeUnreachableBlocks(func);
    split.clear();
    replaced.clear();
    for (auto &[arg, halves] : argumentHalves[&func])
      split[arg] = halves;
    // Phis first: a loop's phi is used before its incoming value is made.
    SmallVector<PHINode *> phis;
    for (Instruction &inst : instructions(func))
      if (auto *phi = dyn_cast<PHINode>(&inst); phi && isWide(phi->getType())) {
        phis.push_back(phi);
        define(*phi,
               {PHINode::Create(i64, phi->getNumIncomingValues(),
                                phi->getName() + ".lo", phi->getIterator()),
                PHINode::Create(i64, phi->getNumIncomingValues(),
                                phi->getName() + ".hi", phi->getIterator())});
      }
    // In reverse post-order, each value is split before its uses.
    ReversePostOrderTraversal<Function *> order(&func);
    for (BasicBlock *block : order)
      for (Instruction &inst : make_early_inc_range(*block))
        if (touchesWide(inst))
          if (Error err = lower(inst))
            return std::move(err);
    for (PHINode *phi : phis) {
      Halves halves = split[phi];
      for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
        Expected<Halves> in = halvesOf(phi->getIncomingValue(i), *phi);
        if (!in)
          return in.takeError();
        cast<PHINode>(halves.low)
            ->addIncoming(in->low, phi->getIncomingBlock(i));
        cast<PHINode>(halves.high)
            ->addIncoming(in->high, phi->getIncomingBlock(i));
      }
    }
    for (Instruction *inst : replaced)
      inst->dropAllReferences();
    for (Instruction *inst : replaced) {
      if (!inst->use_empty())
        inst->replaceAllUsesWith(PoisonValue::get(inst->getType()));
      inst->eraseFromParent();
    }
    return true;
  }

  /// The halves of `value`, an `i128` operand: split before, or a constant.
  Expected<Halves> halvesOf(Value *value, const Instruction &user) {
    auto found = split.find(value);
    if (found != split.end())
      return found->second;
    if (auto *constant = dyn_cast<ConstantInt>(value)) {
      const APInt &bits = constant->getValue();
      return Halves{ConstantInt::get(i64, bits.trunc(64)),
                    ConstantInt::get(i64, bits.extractBits(64, 64))};
    }
    if (isa<PoisonValue>(value))
      return Halves{PoisonValue::get(i64), PoisonValue::get(i64)};
    if (isa<UndefValue>(value))
      return Halves{UndefValue::get(i64), UndefValue::get(i64)};
    std::string what = "an operand it cannot split";
    if (auto *inst = dyn_cast<Instruction>(value))
      what = std::string("an operand made by `") + inst->getOpcodeName() + "`";
    else if (auto *expr = dyn_cast<ConstantExpr>(value))
      what =
          std::string("a constant expression `") + expr->getOpcodeName() + "`";
    else if (isa<Argument>(value))
      what = "an argument";
    return unsupported(user, what);
  }

  // 64-bit operations that skip what a constant operand makes trivial.

  Value *and64(Builder &b, Value *x, Value *y) {
    if (isConstant(x, 0) || isAllOnes(y))
      return x;
    if (isConstant(y, 0) || isAllOnes(x))
      return y;
    return b.CreateAnd(x, y);
  }

  Value *or64(Builder &b, Value *x, Value *y) {
    if (isConstant(y, 0))
      return x;
    if (isConstant(x, 0))
      return y;
    return b.CreateOr(x, y);
  }

  Value *xor64(Builder &b, Value *x, Value *y) {
    if (isConstant(y, 0))
      return x;
    if (isConstant(x, 0))
      return y;
    return b.CreateXor(x, y);
  }

  Value *add64(Builder &b, Value *x, Value *y) {
    if (isConstant(y, 0))
      return x;
    if (isConstant(x, 0))
      return y;
    return b.CreateAdd(x, y);
  }

  Value *mul64(Builder &b, Value *x, Value *y) {
    if (isConstant(x, 0) || isConstant(y, 1))
      return x;
    if (isConstant(y, 0) || isConstant(x, 1))
      return y;
    return b.CreateMul(x, y);
  }

  /// The high 64 bits of the 128-bit product of `x` and `y`, from four
  /// 32 x 32 -> 64 products (no intrinsic, no 128-bit type).
  Value *mulHigh(Builder &b, Value *x, Value *y) {
    if (isConstant(x, 0) || isConstant(y, 0))
      return ConstantInt::get(i64, 0);
    Value *x1 = b.CreateLShr(x, 32);
    Value *x0 = b.CreateAnd(x, 0xffffffffULL);
    Value *y1 = b.CreateLShr(y, 32);
    Value *y0 = b.CreateAnd(y, 0xffffffffULL);
    Value *p00 = b.CreateMul(x0, y0);
    Value *p01 = b.CreateMul(x0, y1);
    Value *p10 = b.CreateMul(x1, y0);
    Value *p11 = b.CreateMul(x1, y1);
    // At most 3 * (2^32 - 1): no carry out of 64 bits.
    Value *middle = b.CreateAdd(
        b.CreateAdd(b.CreateLShr(p00, 32), b.CreateAnd(p01, 0xffffffffULL)),
        b.CreateAnd(p10, 0xffffffffULL));
    return b.CreateAdd(b.CreateAdd(b.CreateAdd(p11, b.CreateLShr(middle, 32)),
                                   b.CreateLShr(p01, 32)),
                       b.CreateLShr(p10, 32));
  }

  Halves add(Builder &b, Halves x, Halves y) {
    if (isConstant(y.low, 0) || isConstant(x.low, 0))
      return {add64(b, x.low, y.low), add64(b, x.high, y.high)};
    Value *low = b.CreateAdd(x.low, y.low);
    // The low sum wrapped exactly when it is below an operand.
    Value *carry = b.CreateZExt(b.CreateICmpULT(low, x.low), i64);
    return {low, b.CreateAdd(add64(b, x.high, y.high), carry)};
  }

  Halves sub(Builder &b, Halves x, Halves y) {
    if (isConstant(y.low, 0))
      return {x.low,
              isConstant(y.high, 0) ? x.high : b.CreateSub(x.high, y.high)};
    Value *borrow = b.CreateZExt(b.CreateICmpULT(x.low, y.low), i64);
    Value *high = isConstant(y.high, 0) ? x.high : b.CreateSub(x.high, y.high);
    return {b.CreateSub(x.low, y.low), b.CreateSub(high, borrow)};
  }

  /// The low 128 bits of the product: the high 64 bits of the low halves'
  /// product plus the low 64 bits of the two cross products.
  Halves mul(Builder &b, Halves x, Halves y) {
    Value *low = mul64(b, x.low, y.low);
    Value *cross = add64(b, mul64(b, x.low, y.high), mul64(b, x.high, y.low));
    return {low, add64(b, mulHigh(b, x.low, y.low), cross)};
  }

  /// `x` shifted by the constant `amount`, 0 to 127.
  Halves shiftBy(Builder &b, unsigned opcode, Halves x, unsigned amount) {
    if (amount == 0)
      return x;
    Value *zero = ConstantInt::get(i64, 0);
    if (opcode == Instruction::Shl) {
      if (amount >= 64)
        return {zero, amount == 64 ? x.low : b.CreateShl(x.low, amount - 64)};
      return {b.CreateShl(x.low, amount),
              or64(b, b.CreateShl(x.high, amount),
                   b.CreateLShr(x.low, 64 - amount))};
    }
    bool arithmetic = opcode == Instruction::AShr;
    auto shiftHigh = [&](unsigned by) -> Value * {
      if (by == 0)
        return x.high;
      return arithmetic ? b.CreateAShr(x.high, by) : b.CreateLShr(x.high, by);
    };
    Value *fill = arithmetic ? b.CreateAShr(x.high, 63) : zero;
    if (amount >= 64)
      return {shiftHigh(amount - 64), fill};
    return {
        or64(b, b.CreateLShr(x.low, amount), b.CreateShl(x.high, 64 - amount)),
        shiftHigh(amount)};
  }

  /// `x` shifted by the variable `amount` (its low half; 128 or more is
  /// poison). No half is shifted by 64 or more, which would be poison too:
  /// the bits crossing between the halves move by 1, then by 63 - n.
  Halves shiftByValue(Builder &b, unsigned opcode, Halves x, Value *amount) {
    Value *zero = ConstantInt::get(i64, 0);
    Value *n = b.CreateAnd(amount, 63);
    Value *rest = b.CreateXor(n, 63); // 63 - n
    Value *wide = b.CreateICmpNE(b.CreateAnd(amount, 64), zero);
    if (opcode == Instruction::Shl) {
      Value *low = b.CreateShl(x.low, n);
      Value *high = b.CreateOr(b.CreateShl(x.high, n),
                               b.CreateLShr(b.CreateLShr(x.low, 1), rest));
      return {b.CreateSelect(wide, zero, low), b.CreateSelect(wide, low, high)};
    }
    bool arithmetic = opcode == Instruction::AShr;
    Value *high =
        arithmetic ? b.CreateAShr(x.high, n) : b.CreateLShr(x.high, n);
    Value *low = b.CreateOr(b.CreateLShr(x.low, n),
                            b.CreateShl(b.CreateShl(x.high, 1), rest));
    Value *fill = arithmetic ? b.CreateAShr(x.high, 63) : zero;
    return {b.CreateSelect(wide, high, low), b.CreateSelect(wide, fill, high)};
  }

  Value *compare(Builder &b, CmpInst::Predicate predicate, Halves x, Halves y) {
    if (predicate == CmpInst::ICMP_EQ)
      return b.CreateAnd(b.CreateICmpEQ(x.low, y.low),
                         b.CreateICmpEQ(x.high, y.high));
    if (predicate == CmpInst::ICMP_NE)
      return b.CreateOr(b.CreateICmpNE(x.low, y.low),
                        b.CreateICmpNE(x.high, y.high));
    // The high halves decide (signed for a signed compare), unless they are
    // equal; then the low halves, unsigned.
    Value *byHigh = b.CreateICmp(predicate, x.high, y.high);
    Value *byLow =
        b.CreateICmp(ICmpInst::getUnsignedPredicate(predicate), x.low, y.low);
    return b.CreateSelect(b.CreateICmpEQ(x.high, y.high), byLow, byHigh);
  }

  /// Records that `inst`, an `i128`, is `halves`; it goes once the function
  /// is done.
  void define(Instruction &inst, Halves halves) {
    split[&inst] = halves;
    replaced.push_back(&inst);
  }

  /// Replaces `inst`, which reads an `i128` but isn't one, with `value`.
  void replace(Instruction &inst, Value *value) {
    inst.replaceAllUsesWith(value);
    replaced.push_back(&inst);
  }

  /// The halves of `count` operands of `inst` from operand `first` on.
  Error operandHalves(Instruction &inst, unsigned first, unsigned count,
                      Halves *out) {
    for (unsigned i = 0; i < count; ++i) {
      Expected<Halves> halves = halvesOf(inst.getOperand(first + i), inst);
      if (!halves)
        return halves.takeError();
      out[i] = *halves;
    }
    return Error::success();
  }

  Error lower(Instruction &inst) {
    Builder b(&inst);
    Halves x[2];
    unsigned opcode = inst.getOpcode();
    if (inst.isBinaryOp() || opcode == Instruction::ICmp) {
      if (inst.isIntDivRem())
        return unsupported(inst, "128-bit division");
      if (!isWide(inst.getOperand(0)->getType()))
        return unsupported(inst, "a vector of 128-bit integers");
      if (Error err = operandHalves(inst, 0, 2, x))
        return err;
      if (opcode == Instruction::ICmp)
        replace(inst,
                compare(b, cast<ICmpInst>(inst).getPredicate(), x[0], x[1]));
      else
        define(inst, binary(b, opcode, x[0], x[1]));
      return Error::success();
    }
    switch (opcode) {
    case Instruction::ZExt:
    case Instruction::SExt: {
      Value *from = inst.getOperand(0);
      if (!isWide(inst.getType()) || !from->getType()->isIntegerTy() ||
          from->getType()->getIntegerBitWidth() > 64)
        return unsupported(inst, "only scalars of up to 64 bits extend");
      bool isSigned = opcode == Instruction::SExt;
      Value *low = from;
      if (!from->getType()->isIntegerTy(64))
        low = isSigned ? b.CreateSExt(from, i64) : b.CreateZExt(from, i64);
      define(inst, {low, isSigned ? b.CreateAShr(low, 63)
                                  : ConstantInt::get(i64, 0)});
      return Error::success();
    }
    case Instruction::Trunc:
      if (!isWide(inst.getOperand(0)->getType()) ||
          !inst.getType()->isIntegerTy() ||
          inst.getType()->getIntegerBitWidth() > 64)
        return unsupported(inst, "only scalars of up to 64 bits truncate");
      if (Error err = operandHalves(inst, 0, 1, x))
        return err;
      replace(inst, inst.getType()->isIntegerTy(64)
                        ? x[0].low
                        : b.CreateTrunc(x[0].low, inst.getType()));
      return Error::success();
    case Instruction::Select:
      if (!isWide(inst.getType()) ||
          !inst.getOperand(0)->getType()->isIntegerTy(1))
        return unsupported(inst, "a vector of 128-bit integers");
      if (Error err = operandHalves(inst, 1, 2, x))
        return err;
      define(inst, {b.CreateSelect(inst.getOperand(0), x[0].low, x[1].low),
                    b.CreateSelect(inst.getOperand(0), x[0].high, x[1].high)});
      return Error::success();
    case Instruction::Freeze:
      if (!isWide(inst.getType()))
        return unsupported(inst, "a vector of 128-bit integers");
      if (Error err = operandHalves(inst, 0, 1, x))
        return err;
      define(inst, {b.CreateFreeze(x[0].low), b.CreateFreeze(x[0].high)});
      return Error::success();
    case Instruction::PHI:
      if (split.count(&inst))
        return Error::success(); // made before the walk
      return unsupported(inst, "an aggregate holding 128-bit integers");
    case Instruction::Load:
    case Instruction::Store:
      return lowerAccess(inst, b);
    case Instruction::Alloca: {
      auto *alloca = cast<AllocaInst>(&inst);
      Expected<Type *> type = memoryType(alloca->getAllocatedType(), inst);
      if (!type)
        return type.takeError();
      auto *copy = b.CreateAlloca(*type, alloca->getAddressSpace(),
                                  alloca->getArraySize());
      copy->setAlignment(alloca->getAlign());
      copy->takeName(alloca);
      replace(inst, copy);
      return Error::success();
    }
    case Instruction::GetElementPtr: {
      auto *gep = cast<GetElementPtrInst>(&inst);
      Expected<Type *> type = memoryType(gep->getSourceElementType(), inst);
      if (!type)
        return type.takeError();
      SmallVector<Value *> indices(gep->indices());
      gep->setSourceElementType(*type);
      gep->setResultElementType(
          GetElementPtrInst::getIndexedType(*type, indices));
      return Error::success();
    }
    case Instruction::BitCast:
      return lowerBitCast(inst, b);
    case Instruction::Call:
      return lowerCall(cast<CallInst>(inst), b);
    case Instruction::Ret: {
      if (!isWide(inst.getOperand(0)->getType()))
        return unsupported(inst, "an aggregate holding 128-bit integers");
      if (Error err = operandHalves(inst, 0, 1, x))
        return err;
      Type *pair = inst.getFunction()->getReturnType();
      Value *result = b.CreateInsertValue(PoisonValue::get(pair), x[0].low, 0);
      b.CreateRet(b.CreateInsertValue(result, x[0].high, 1));
      replaced.push_back(&inst);
      return Error::success();
    }
    default:
      return unsupported(inst);
    }
  }

  /// A load or store of an `i128`, as two of `i64`s, the high one 8 bytes
  /// further.
  Error lowerAccess(Instruction &inst, Builder &b) {
    auto *load = dyn_cast<LoadInst>(&inst);
    auto *store = dyn_cast<StoreInst>(&inst);
    Type *type = load ? load->getType() : store->getValueOperand()->getType();
    if (!isWide(type))
      return unsupported(inst, "an aggregate holding 128-bit integers");
    if (load ? load->isAtomic() : store->isAtomic())
      return unsupported(inst, "an atomic access");
    Value *pointer = getLoadStorePointerOperand(&inst);
    Align align = getLoadStoreAlignment(&inst);
    bool isVolatile = load ? load->isVolatile() : store->isVolatile();
    Value *high = b.CreateConstGEP1_64(b.getInt8Ty(), pointer, 8);
    Align highAlign = commonAlignment(align, 8);
    if (load) {
      define(inst, {b.CreateAlignedLoad(i64, pointer, align, isVolatile),
                    b.CreateAlignedLoad(i64, high, highAlign, isVolatile)});
      return Error::success();
    }
    Halves x;
    if (Error err = operandHalves(inst, 0, 1, &x))
      return err;
    b.CreateAlignedStore(x.low, pointer, align, isVolatile);
    b.CreateAlignedStore(x.high, high, highAlign, isVolatile);
    replaced.push_back(&inst);
    return Error::success();
  }

  /// A bitcast between an `i128` and a 128-bit vector (or fp128), through
  /// `<2 x i64>`, whose elements are the low and high halves.
  Error lowerBitCast(Instruction &inst, Builder &b) {
    Value *from = inst.getOperand(0);
    auto *pair = FixedVectorType::get(i64, 2);
    if (isWide(from->getType()) && isWide(inst.getType())) {
      Halves x;
      if (Error err = operandHalves(inst, 0, 1, &x))
        return err;
      define(inst, x);
      return Error::success();
    }
    if (isWide(inst.getType()) && !holdsWide(from->getType())) {
      Value *vector =
          from->getType() == pair ? from : b.CreateBitCast(from, pair);
      define(inst, {b.CreateExtractElement(vector, uint64_t(0)),
                    b.CreateExtractElement(vector, uint64_t(1))});
      return Error::success();
    }
    if (!isWide(from->getType()) || holdsWide(inst.getType()))
      return unsupported(inst, "a vector of 128-bit integers");
    Halves x;
    if (Error err = operandHalves(inst, 0, 1, &x))
      return err;
    Value *vector =
        b.CreateInsertElement(PoisonValue::get(pair), x.low, uint64_t(0));
    vector = b.CreateInsertElement(vector, x.high, uint64_t(1));
    replace(inst, inst.getType() == pair
                      ? vector
                      : b.CreateBitCast(vector, inst.getType()));
    return Error::success();
  }

  /// A call of a function whose signature was split: the `i128` arguments
  /// as their halves, an `i128` result read from the `{ i64, i64 }`.
  Error lowerCall(CallInst &call, Builder &b) {
    Function *callee = call.getCalledFunction();
    auto found = callee ? rewritten.find(callee) : rewritten.end();
    if (found == rewritten.end())
      return unsupported(call, callee ? "a call of a function that isn't "
                                        "internal"
                                      : "an indirect call");
    Function *copy = found->second;
    SmallVector<Value *> args;
    for (Use &arg : call.args()) {
      if (!isWide(arg->getType())) {
        args.push_back(arg);
        continue;
      }
      Halves x;
      if (Error err = operandHalves(call, arg.getOperandNo(), 1, &x))
        return err;
      args.append({x.low, x.high});
    }
    CallInst *result = b.CreateCall(copy->getFunctionType(), copy, args);
    result->setCallingConv(call.getCallingConv());
    result->setTailCallKind(call.getTailCallKind());
    result->setAttributes(
        splitAttributes(call.getAttributes(), call.getFunctionType()));
    if (isWide(call.getType())) {
      define(call, {b.CreateExtractValue(result, 0),
                    b.CreateExtractValue(result, 1)});
      return Error::success();
    }
    result->takeName(&call);
    replace(call, result);
    return Error::success();
  }

  /// `attrs` for the split signature of `type`: none on the halves, nor on
  /// a split result.
  AttributeList splitAttributes(AttributeList attrs, FunctionType *type) {
    SmallVector<AttributeSet> params;
    for (unsigned i = 0; i < type->getNumParams(); ++i) {
      if (isWide(type->getParamType(i)))
        params.append(2, AttributeSet());
      else
        params.push_back(attrs.getParamAttrs(i));
    }
    return AttributeList::get(
        module.getContext(), attrs.getFnAttrs(),
        isWide(type->getReturnType()) ? AttributeSet() : attrs.getRetAttrs(),
        params);
  }

  /// Gives internal function `func`, called only directly, a copy whose
  /// `i128` parameters are each two `i64`s and whose `i128` result is an
  /// `{ i64, i64 }`, with its body. The body is lowered with the rest.
  Error splitSignature(Function &func) {
    FunctionType *type = func.getFunctionType();
    auto refuse = [&](const Twine &why) {
      return createStringError(inconvertibleErrorCode(),
                               "intel plugin: the function `" + func.getName() +
                                   "` " + why);
    };
    if (!func.hasLocalLinkage() || func.isVarArg() ||
        func.getCallingConv() == CallingConv::SPIR_KERNEL ||
        any_of(func.uses(), [](Use &use) {
          auto *call = dyn_cast<CallInst>(use.getUser());
          return !call || !call->isCallee(&use);
        }))
      return refuse("takes or returns 128-bit integers and isn't an internal "
                    "function called only directly");
    LLVMContext &ctx = module.getContext();
    SmallVector<Type *> params;
    for (Type *param : type->params()) {
      if (isWide(param))
        params.append(2, i64);
      else if (holdsWide(param))
        return refuse("takes an aggregate holding 128-bit integers");
      else
        params.push_back(param);
    }
    Type *result = type->getReturnType();
    if (isWide(result))
      result = StructType::get(ctx, {i64, i64});
    else if (holdsWide(result))
      return refuse("returns an aggregate holding 128-bit integers");
    Function *copy = Function::Create(FunctionType::get(result, params, false),
                                      func.getLinkage(), func.getAddressSpace(),
                                      "", &module);
    copy->copyAttributesFrom(&func);
    copy->copyMetadata(&func, 0);
    copy->setAttributes(splitAttributes(func.getAttributes(), type));
    copy->splice(copy->end(), &func);
    auto to = copy->arg_begin();
    for (Argument &arg : func.args()) {
      if (!isWide(arg.getType())) {
        to->takeName(&arg);
        arg.replaceAllUsesWith(&*to++);
        continue;
      }
      Argument *low = &*to++;
      Argument *high = &*to++;
      low->setName(arg.getName() + ".lo");
      high->setName(arg.getName() + ".hi");
      argumentHalves[copy].push_back({&arg, {low, high}});
    }
    rewritten.insert({&func, copy});
    return Error::success();
  }

  /// `type` with each `i128` as `[2 x i64]`, which has the same size and,
  /// checked here, the same alignment, so every offset stays the same.
  Expected<Type *> memoryType(Type *type, const Instruction &user) {
    Expected<Type *> mapped = mapType(type);
    if (!mapped)
      return unsupported(user, toString(mapped.takeError()));
    return mapped;
  }

  Expected<Type *> mapType(Type *type) {
    if (!holdsWide(type))
      return type;
    LLVMContext &ctx = module.getContext();
    const DataLayout &layout = module.getDataLayout();
    if (isWide(type)) {
      Type *pair = ArrayType::get(i64, 2);
      if (layout.getABITypeAlign(type) != layout.getABITypeAlign(pair) ||
          layout.getTypeAllocSize(type) != layout.getTypeAllocSize(pair))
        return createStringError(inconvertibleErrorCode(),
                                 "the data layout lays out i128 unlike "
                                 "[2 x i64]");
      return pair;
    }
    if (auto *array = dyn_cast<ArrayType>(type)) {
      Expected<Type *> element = mapType(array->getElementType());
      if (!element)
        return element.takeError();
      return ArrayType::get(*element, array->getNumElements());
    }
    if (auto *structType = dyn_cast<StructType>(type)) {
      SmallVector<Type *> elements;
      for (Type *element : structType->elements()) {
        Expected<Type *> mapped = mapType(element);
        if (!mapped)
          return mapped.takeError();
        elements.push_back(*mapped);
      }
      if (structType->isLiteral())
        return StructType::get(ctx, elements, structType->isPacked());
      return StructType::create(ctx, elements, structType->getName(),
                                structType->isPacked());
    }
    return createStringError(inconvertibleErrorCode(),
                             "a vector of 128-bit integers");
  }

  /// `constant` as a constant of `type`, its mapped type: each `i128` as an
  /// `[2 x i64]` of its halves. Floating-point constants are kept as they
  /// are, never made. Null if it holds an `i128` it can't read.
  Constant *mapConstant(Constant *constant, Type *type) {
    if (constant->getType() == type)
      return constant;
    if (isa<PoisonValue>(constant))
      return PoisonValue::get(type);
    if (isa<UndefValue>(constant))
      return UndefValue::get(type);
    if (isa<ConstantAggregateZero>(constant) || constant->isNullValue())
      return Constant::getNullValue(type);
    if (auto *integer = dyn_cast<ConstantInt>(constant)) {
      const APInt &bits = integer->getValue();
      uint64_t halves[] = {bits.trunc(64).getZExtValue(),
                           bits.extractBits(64, 64).getZExtValue()};
      return ConstantDataArray::get(module.getContext(),
                                    ArrayRef<uint64_t>(halves));
    }
    if (!isa<ConstantAggregate>(constant))
      return nullptr;
    SmallVector<Constant *> elements;
    for (unsigned i = 0; i < constant->getNumOperands(); ++i) {
      Type *element = isa<StructType>(type)
                          ? cast<StructType>(type)->getElementType(i)
                          : cast<ArrayType>(type)->getElementType();
      Constant *mapped = mapConstant(constant->getAggregateElement(i), element);
      if (!mapped)
        return nullptr;
      elements.push_back(mapped);
    }
    if (auto *structType = dyn_cast<StructType>(type))
      return ConstantStruct::get(structType, elements);
    return ConstantArray::get(cast<ArrayType>(type), elements);
  }

  /// Replaces global `global`, which holds `i128`s (float formatting's
  /// table), with one of the mapped type and initializer.
  Error retypeGlobal(GlobalVariable &global) {
    auto refuse = [&](const Twine &why) {
      return createStringError(inconvertibleErrorCode(),
                               "intel plugin: the global `" + global.getName() +
                                   "` holds 128-bit "
                                   "integers: " +
                                   why);
    };
    Expected<Type *> type = mapType(global.getValueType());
    if (!type)
      return refuse(toString(type.takeError()));
    Constant *init = nullptr;
    if (global.hasInitializer()) {
      init = mapConstant(global.getInitializer(), *type);
      if (!init)
        return refuse("its initializer has a constant it can't split");
    }
    auto *copy = new GlobalVariable(
        module, *type, global.isConstant(), global.getLinkage(), init, "",
        &global, global.getThreadLocalMode(), global.getAddressSpace(),
        global.isExternallyInitialized());
    copy->copyAttributesFrom(&global);
    copy->copyMetadata(&global, 0);
    copy->takeName(&global);
    global.replaceAllUsesWith(copy);
    global.eraseFromParent();
    return Error::success();
  }

  /// A constant expression with a GEP over a type that holds an `i128`
  /// (the instruction's type can't show it), rebuilt as a GEP of bytes; null
  /// if there is none in it.
  Constant *retypeConstantGEP(Constant *constant) {
    auto *expr = dyn_cast<ConstantExpr>(constant);
    if (!expr)
      return nullptr;
    bool changed = false;
    SmallVector<Constant *> operands;
    for (Use &use : expr->operands()) {
      Constant *mapped = retypeConstantGEP(cast<Constant>(use));
      changed |= mapped != nullptr;
      operands.push_back(mapped ? mapped : cast<Constant>(use));
    }
    // As `getelementptr i8` by the same offset: the layout is the same.
    if (auto *gep = dyn_cast<GEPOperator>(expr))
      if (holdsWide(gep->getSourceElementType()))
        if (Constant *bytes = ConstantExpr::getGetElementPtr(
                module.getDataLayout(), gep->getSourceElementType(),
                operands[0], ArrayRef(operands).drop_front(),
                gep->getNoWrapFlags(), gep->getInRange()))
          return bytes;
    return changed ? expr->getWithOperands(operands) : nullptr;
  }

  bool retypeConstantGEPs(Function &func) {
    bool changed = false;
    for (Instruction &inst : instructions(func))
      for (Use &use : inst.operands())
        if (auto *constant = dyn_cast<Constant>(use))
          if (Constant *mapped = retypeConstantGEP(constant)) {
            use.set(mapped);
            changed = true;
          }
    return changed;
  }

  Halves binary(Builder &b, unsigned opcode, Halves x, Halves y) {
    switch (opcode) {
    case Instruction::Add:
      return add(b, x, y);
    case Instruction::Sub:
      return sub(b, x, y);
    case Instruction::Mul:
      return mul(b, x, y);
    case Instruction::And:
      return {and64(b, x.low, y.low), and64(b, x.high, y.high)};
    case Instruction::Or:
      return {or64(b, x.low, y.low), or64(b, x.high, y.high)};
    case Instruction::Xor:
      return {xor64(b, x.low, y.low), xor64(b, x.high, y.high)};
    default:
      break;
    }
    auto *low = dyn_cast<ConstantInt>(y.low);
    auto *high = dyn_cast<ConstantInt>(y.high);
    if (low && high) {
      if (!high->isZero() || low->getValue().uge(128))
        return {PoisonValue::get(i64), PoisonValue::get(i64)};
      return shiftBy(b, opcode, x, static_cast<unsigned>(low->getZExtValue()));
    }
    return shiftByValue(b, opcode, x, y.low);
  }

  /// An error naming the first `i128` left in the module, if any.
  Error checkNoneLeft() {
    for (GlobalVariable &global : module.globals())
      if (holdsWide(global.getValueType()))
        return createStringError(inconvertibleErrorCode(),
                                 "intel plugin: the global `" +
                                     global.getName() +
                                     "` holds 128-bit integers");
    for (Function &func : module) {
      if (holdsWide(func.getFunctionType()))
        return createStringError(inconvertibleErrorCode(),
                                 "intel plugin: the function `" +
                                     func.getName() +
                                     "` takes or returns 128-bit integers");
      for (Instruction &inst : instructions(func))
        if (touchesWide(inst))
          return unsupported(inst, "left by the lowering");
    }
    return Error::success();
  }

  Module &module;
  Type *i64;
  /// Each `i128` instruction of the function being lowered, as its halves.
  DenseMap<Value *, Halves> split;
  /// The instructions replaced, erased when the function is done.
  SmallVector<Instruction *, 32> replaced;
  /// Each function whose signature was split, and its copy.
  MapVector<Function *, Function *> rewritten;
  /// Each copy's `i128` arguments (the old ones, still used in its body),
  /// and the halves that replace them.
  DenseMap<Function *, SmallVector<std::pair<Argument *, Halves>>>
      argumentHalves;
};

} // namespace

Expected<bool> lowerInt128(Module &module) { return Lowering(module).run(); }

} // namespace M::KGEN::Intel
