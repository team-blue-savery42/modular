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

#include "MaskBitcasts.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace llvm;

namespace M::KGEN::Intel {

namespace {

/// Whether IGC takes integers of `bits` bits.
bool isLegalWidth(unsigned bits) {
  return bits == 1 || bits == 8 || bits == 16 || bits == 32 || bits == 64;
}

/// A bitcast of a fixed `<N x i1>` mask to an integer IGC can't take.
bool isOddMaskBitcast(const Instruction &inst) {
  const auto *cast = dyn_cast<BitCastInst>(&inst);
  if (cast == nullptr)
    return false;
  const auto *mask = dyn_cast<FixedVectorType>(cast->getSrcTy());
  const auto *to = dyn_cast<IntegerType>(cast->getDestTy());
  return mask != nullptr && to != nullptr &&
         mask->getElementType()->isIntegerTy(1) &&
         !isLegalWidth(to->getBitWidth());
}

unsigned laneCount(Value *mask) {
  return cast<FixedVectorType>(mask->getType())->getNumElements();
}

/// The smallest legal integer type holding `bits` bits.
IntegerType *legalType(LLVMContext &ctx, unsigned bits) {
  for (unsigned width : {8u, 16u, 32u, 64u})
    if (bits <= width)
      return IntegerType::get(ctx, width);
  return nullptr;
}

/// The mask as an integer of `type`, bit i being lane i, as the bitcast then
/// a zext or, with `signExtend`, a sext to `type` computes it.
Value *maskBits(IRBuilder<NoFolder> &builder, Value *mask, IntegerType *type,
                bool signExtend) {
  unsigned lanes = laneCount(mask);
  Value *bits = nullptr;
  for (unsigned i = 0; i < lanes; ++i) {
    Value *lane = builder.CreateZExt(
        builder.CreateExtractElement(mask, builder.getInt32(i)), type);
    if (i > 0)
      lane = builder.CreateShl(lane, i);
    bits = bits == nullptr ? lane : builder.CreateOr(bits, lane);
  }
  unsigned width = type->getBitWidth();
  if (signExtend && width > lanes) {
    Value *last =
        builder.CreateExtractElement(mask, builder.getInt32(lanes - 1));
    Value *high = builder.CreateOr(
        bits,
        ConstantInt::get(type, APInt::getHighBitsSet(width, width - lanes)));
    bits = builder.CreateSelect(last, high, bits);
  }
  return bits;
}

/// The number of set lanes of the mask, as an integer of `type`.
Value *laneCountSet(IRBuilder<NoFolder> &builder, Value *mask,
                    IntegerType *type) {
  Value *count = nullptr;
  for (unsigned i = 0, lanes = laneCount(mask); i < lanes; ++i) {
    Value *lane = builder.CreateZExt(
        builder.CreateExtractElement(mask, builder.getInt32(i)), type);
    count = count == nullptr ? lane : builder.CreateAdd(count, lane);
  }
  return count;
}

/// `cmp`'s replacement, for an equality with a constant: lane i of `mask`
/// equals bit i of `bits`, for every lane, negated for `ne`.
Value *compareLanes(ICmpInst &cmp, Value *mask, const APInt &bits) {
  IRBuilder<NoFolder> builder(&cmp);
  Value *all = nullptr;
  for (unsigned i = 0, lanes = laneCount(mask); i < lanes; ++i) {
    Value *lane = builder.CreateExtractElement(mask, builder.getInt32(i));
    if (!bits[i])
      lane = builder.CreateNot(lane);
    all = all == nullptr ? lane : builder.CreateAnd(all, lane);
  }
  if (cmp.getPredicate() == ICmpInst::ICMP_NE)
    all = builder.CreateNot(all);
  return all;
}

/// `cmp`, comparing `value` (an odd-width integer) with a constant, redone
/// on `wide`, the same value in a legal type, extended as the predicate
/// reads it.
Value *compareWidened(ICmpInst &cmp, Value *value, Value *wide) {
  IRBuilder<NoFolder> builder(&cmp);
  unsigned width = wide->getType()->getIntegerBitWidth();
  auto widen = [&](Value *operand) -> Value * {
    if (operand == value)
      return wide;
    const APInt &constant = cast<ConstantInt>(operand)->getValue();
    return ConstantInt::get(wide->getType(), cmp.isSigned()
                                                 ? constant.sext(width)
                                                 : constant.zext(width));
  };
  return builder.CreateICmp(cmp.getPredicate(), widen(cmp.getOperand(0)),
                            widen(cmp.getOperand(1)));
}

/// The operand of `cmp` other than `value`, if it is a constant.
ConstantInt *otherConstant(ICmpInst &cmp, Value *value) {
  return dyn_cast<ConstantInt>(cmp.getOperand(0) == value ? cmp.getOperand(1)
                                                          : cmp.getOperand(0));
}

Error unsupported(Instruction &user, Value *value) {
  std::string text;
  raw_string_ostream os(text);
  user.print(os);
  return createStringError(
      "intel plugin: a lane mask bitcast to " +
      std::to_string(value->getType()->getIntegerBitWidth()) +
      "-bit integer, which IGC can't compile, is used by `" + text + "` in " +
      user.getFunction()->getName());
}

void replace(Instruction &old, Value *replacement) {
  replacement->takeName(&old);
  old.replaceAllUsesWith(replacement);
  old.eraseFromParent();
}

/// Rewrites the users of `count`, a `ctpop` of the mask, on the count in
/// legal types.
Error rewriteCountUsers(CallInst &count, Value *mask) {
  SmallVector<User *> users(count.users());
  for (User *user : users) {
    auto *inst = cast<Instruction>(user);
    IRBuilder<NoFolder> builder(inst);
    if (auto *zext = dyn_cast<ZExtInst>(inst)) {
      replace(*zext, laneCountSet(builder, mask,
                                  cast<IntegerType>(zext->getDestTy())));
      continue;
    }
    // The count fits the mask's width, so unsigned compares see it whole.
    auto *cmp = dyn_cast<ICmpInst>(inst);
    if (cmp != nullptr && !cmp->isSigned() && otherConstant(*cmp, &count)) {
      IntegerType *type =
          legalType(count.getContext(), count.getType()->getIntegerBitWidth());
      replace(*cmp,
              compareWidened(*cmp, &count, laneCountSet(builder, mask, type)));
      continue;
    }
    return unsupported(*inst, &count);
  }
  return Error::success();
}

/// Rewrites `user`, a user of `bitcast`, which casts `mask`, to no longer
/// need an odd-width integer.
Error rewriteUser(Instruction &user, BitCastInst &bitcast, Value *mask) {
  IRBuilder<NoFolder> builder(&user);
  if (auto *cmp = dyn_cast<ICmpInst>(&user)) {
    ConstantInt *constant = otherConstant(*cmp, &bitcast);
    if (constant == nullptr)
      return unsupported(user, &bitcast);
    if (cmp->isEquality()) {
      replace(*cmp, compareLanes(*cmp, mask, constant->getValue()));
      return Error::success();
    }
    IntegerType *type = legalType(user.getContext(), laneCount(mask));
    replace(*cmp,
            compareWidened(*cmp, &bitcast,
                           maskBits(builder, mask, type, cmp->isSigned())));
    return Error::success();
  }
  if (isa<ZExtInst, SExtInst>(user)) {
    replace(user, maskBits(builder, mask, cast<IntegerType>(user.getType()),
                           isa<SExtInst>(user)));
    return Error::success();
  }
  if (isa<BitCastInst>(user) && user.getType() == mask->getType()) {
    replace(user, mask);
    return Error::success();
  }
  // Intrinsic IDs differ between the host's LLVM and the plugin's: by name.
  auto *call = dyn_cast<CallInst>(&user);
  Function *callee = call != nullptr ? call->getCalledFunction() : nullptr;
  if (callee != nullptr && callee->getName().starts_with("llvm.ctpop.")) {
    if (Error err = rewriteCountUsers(*call, mask))
      return err;
    if (call->use_empty())
      call->eraseFromParent();
    return Error::success();
  }
  return unsupported(user, &bitcast);
}

} // namespace

Expected<bool> rewriteMaskBitcasts(Module &module) {
  SmallVector<BitCastInst *> casts;
  for (Function &function : module)
    for (Instruction &inst : instructions(function))
      if (isOddMaskBitcast(inst))
        casts.push_back(cast<BitCastInst>(&inst));
  for (BitCastInst *bitcast : casts) {
    Value *mask = bitcast->getOperand(0);
    // Collected first: each rewrite edits the bitcast's use list.
    SmallVector<User *> users(bitcast->users());
    for (User *user : users)
      if (Error err = rewriteUser(*cast<Instruction>(user), *bitcast, mask))
        return std::move(err);
    bitcast->eraseFromParent();
  }
  // Declarations of the odd-width intrinsics the rewrite stopped calling.
  SmallVector<Function *> unused;
  for (Function &function : module)
    if (function.isDeclaration() && function.use_empty() &&
        function.getName().starts_with("llvm.ctpop.") &&
        !isLegalWidth(function.getReturnType()->getScalarSizeInBits()))
      unused.push_back(&function);
  for (Function *function : unused)
    function->eraseFromParent();
  return !casts.empty();
}

bool expandMaskTruncs(Module &module) {
  SmallVector<TruncInst *> truncs;
  for (Function &function : module)
    for (Instruction &inst : instructions(function))
      if (auto *trunc = dyn_cast<TruncInst>(&inst))
        if (auto *type = dyn_cast<FixedVectorType>(trunc->getType()))
          if (type->getElementType()->isIntegerTy(1))
            truncs.push_back(trunc);
  for (TruncInst *trunc : truncs) {
    IRBuilder<NoFolder> builder(trunc);
    Value *source = trunc->getOperand(0);
    Value *low =
        builder.CreateAnd(source, ConstantInt::get(source->getType(), 1));
    Value *lanes =
        builder.CreateICmpNE(low, Constant::getNullValue(source->getType()));
    lanes->takeName(trunc);
    trunc->replaceAllUsesWith(lanes);
    trunc->eraseFromParent();
  }
  return !truncs.empty();
}

} // namespace M::KGEN::Intel
