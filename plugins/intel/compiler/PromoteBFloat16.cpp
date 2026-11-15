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

#include "PromoteBFloat16.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"

#include <optional>

using namespace llvm;

namespace M::KGEN::Intel {
namespace {

// The plugin links its own copy of LLVM, and APFloat identifies a float
// format by the address of its semantics object. A constant folded here
// would carry this copy's semantics, which the host's codegen can't read
// (an "unknown format" assertion). So nothing is folded: a conversion of a
// constant stays an instruction, which codegen folds with the host's APFloat.
// The one exception is bf16Bits below, which reads a bfloat16 constant's bits
// into an integer constant, a type both copies agree on.
using Builder = IRBuilder<NoFolder>;

bool isBF16(Type *type) { return type->getScalarType()->isBFloatTy(); }

/// `type` with float32 elements in place of bfloat16 ones.
Type *widened(Type *type) {
  Type *f32 = Type::getFloatTy(type->getContext());
  if (auto *vector = dyn_cast<VectorType>(type))
    return VectorType::get(f32, vector->getElementCount());
  return f32;
}

Value *extend(Builder &builder, Value *value) {
  return isBF16(value->getType())
             ? builder.CreateFPExt(value, widened(value->getType()))
             : value;
}

/// Whether `inst` computes on bfloat16 values in a way SPIR-V can't express
/// without SPV_INTEL_bfloat16_arithmetic. Moves, selects, phis, loads, stores
/// and conversions between float types aren't promoted (moveAsIntegers then
/// moves the values as i16).
bool needsPromotion(Instruction &inst) {
  if (isa<BinaryOperator>(inst) || isa<UnaryOperator>(inst))
    return inst.getType()->isFPOrFPVectorTy() && isBF16(inst.getType());
  if (auto *cmp = dyn_cast<FCmpInst>(&inst))
    return isBF16(cmp->getOperand(0)->getType());
  if (isa<SIToFPInst>(inst) || isa<UIToFPInst>(inst))
    return isBF16(inst.getType());
  if (isa<FPToSIInst>(inst) || isa<FPToUIInst>(inst))
    return isBF16(inst.getOperand(0)->getType());
  if (auto *call = dyn_cast<IntrinsicInst>(&inst)) {
    // Only intrinsics that compute on their floating-point operands; memory,
    // debug and similar intrinsics may carry bfloat16 values untouched. One
    // that touches memory (llvm.masked.load, say) must: as float32 it would
    // access twice the bytes.
    if (call->mayReadOrWriteMemory())
      return false;
    if (!call->getType()->isFPOrFPVectorTy() &&
        call->getIntrinsicID() != Intrinsic::is_fpclass)
      return false;
    if (isBF16(call->getType()))
      return true;
    return any_of(call->args(), [](Value *arg) {
      return arg->getType()->isFPOrFPVectorTy() && isBF16(arg->getType());
    });
  }
  return false;
}

/// Integer `value` (signed if `isSigned`) as float32 rounded to odd: cut to
/// float32's 24 significant bits, with the lowest kept bit set if any bit
/// below it was. Rounding that to bfloat16 then gives the correctly rounded
/// result; a plain conversion to float32 rounds to nearest first, and the
/// second rounding can then go the wrong way at a tie (2^24 + 2^16 + 1 would
/// become 2^24, not 2^24 + 2^17). Only integer constants are made here.
Value *roundedToOdd(Builder &builder, Value *value, bool isSigned, Type *f32) {
  Type *type = value->getType();
  unsigned bits = type->getScalarSizeInBits();
  if (bits <= 24) // exact in float32
    return isSigned ? builder.CreateSIToFP(value, f32)
                    : builder.CreateUIToFP(value, f32);
  Value *zero = Constant::getNullValue(type);
  Value *one = ConstantInt::get(type, 1);
  Value *negative = nullptr;
  Value *magnitude = value;
  if (isSigned) {
    // The most negative value negates to itself, which as unsigned is its
    // magnitude.
    negative = builder.CreateICmpSLT(value, zero);
    magnitude = builder.CreateSelect(negative, builder.CreateNeg(value), value);
  }
  Value *leading = builder.CreateBinaryIntrinsic(Intrinsic::ctlz, magnitude,
                                                 builder.getFalse());
  // Bits below the top 24 significant ones: max(bits - leading - 24, 0).
  Value *excess = builder.CreateSub(
      builder.CreateSub(ConstantInt::get(type, bits), leading),
      ConstantInt::get(type, 24));
  Value *shift =
      builder.CreateSelect(builder.CreateICmpSGT(excess, zero), excess, zero);
  Value *unit = builder.CreateShl(one, shift);
  Value *below = builder.CreateSub(unit, one);
  Value *sticky =
      builder.CreateICmpNE(builder.CreateAnd(magnitude, below), zero);
  Value *odd =
      builder.CreateOr(builder.CreateAnd(magnitude, builder.CreateNot(below)),
                       builder.CreateSelect(sticky, unit, zero));
  // At most 24 significant bits: exact.
  Value *result = builder.CreateUIToFP(odd, f32);
  if (negative)
    result = builder.CreateSelect(negative, builder.CreateFNeg(result), result);
  return result;
}

/// The float32 version of a bfloat16 `call` to an overloaded intrinsic.
Value *promoteIntrinsic(Builder &builder, IntrinsicInst &call) {
  SmallVector<Value *> args;
  SmallVector<Type *> argTypes;
  for (Value *arg : call.args()) {
    args.push_back(extend(builder, arg));
    argTypes.push_back(args.back()->getType());
  }
  Type *type =
      isBF16(call.getType()) ? widened(call.getType()) : call.getType();
  // Resolves the overloads from the widened types.
  Function *wide = Intrinsic::getOrInsertDeclaration(
      call.getModule(), call.getIntrinsicID(), type, argTypes);
  CallInst *result = builder.CreateCall(wide, args);
  // Calls with a non-float result, such as llvm.is.fpclass, have none.
  if (isa<FPMathOperator>(result))
    result->copyFastMathFlags(&call);
  return result;
}

/// Whether `type` is or holds bfloat16 values (in vectors, structs, arrays).
bool holdsBF16(Type *type) {
  if (isBF16(type))
    return true;
  if (auto *st = dyn_cast<StructType>(type))
    return any_of(st->elements(), holdsBF16);
  if (auto *array = dyn_cast<ArrayType>(type))
    return holdsBF16(array->getElementType());
  return false;
}

/// `type` with i16 in place of each bfloat16, at any depth.
Type *asIntegers(Type *type) {
  LLVMContext &ctx = type->getContext();
  if (isBF16(type)) {
    Type *i16 = Type::getInt16Ty(ctx);
    if (auto *vector = dyn_cast<VectorType>(type))
      return VectorType::get(i16, vector->getElementCount());
    return i16;
  }
  if (auto *st = dyn_cast<StructType>(type)) {
    if (!holdsBF16(st))
      return st;
    SmallVector<Type *> elements;
    for (Type *element : st->elements())
      elements.push_back(asIntegers(element));
    return StructType::get(ctx, elements, st->isPacked());
  }
  if (auto *array = dyn_cast<ArrayType>(type))
    return ArrayType::get(asIntegers(array->getElementType()),
                          array->getNumElements());
  return type;
}

/// Whether `inst` only moves values that are or hold bfloat16: these become
/// moves of i16 (see moveAsIntegers). Conversions to other float types,
/// calls and returns keep their bfloat16 operands and results, converted at
/// the edge.
bool movesBF16(Instruction &inst) {
  if (auto *store = dyn_cast<StoreInst>(&inst))
    return holdsBF16(store->getValueOperand()->getType());
  if (auto *extract = dyn_cast<ExtractValueInst>(&inst))
    return holdsBF16(extract->getAggregateOperand()->getType());
  if (isa<LoadInst>(inst) || isa<SelectInst>(inst) || isa<PHINode>(inst) ||
      isa<FreezeInst>(inst) || isa<ExtractElementInst>(inst) ||
      isa<InsertElementInst>(inst) || isa<ShuffleVectorInst>(inst) ||
      isa<InsertValueInst>(inst))
    return holdsBF16(inst.getType());
  return false;
}

/// The bits of the bfloat16 constant `fp`, or nothing for a signaling NaN.
/// The host's LLVM made the constant, so its APFloat names the host's
/// bfloat16 semantics, whose address this copy's bitcastToAPInt doesn't know.
/// Converting it to this copy's bfloat16 semantics reads the host's by value
/// (the same precision and exponent range in the same LLVM version), changes
/// no bit of a finite value, infinity or quiet NaN, and leaves an APFloat
/// this copy reads. A signaling NaN would be quieted, so it stays a bitcast.
std::optional<uint16_t> bf16Bits(const ConstantFP *fp) {
  APFloat value = fp->getValueAPF();
  bool lossy = false;
  if (value.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven, &lossy) !=
          APFloat::opOK ||
      lossy)
    return std::nullopt;
  return static_cast<uint16_t>(value.bitcastToAPInt().getZExtValue());
}

/// `constant`, a bfloat16 scalar or vector, as the integer constant of
/// `type` holding the same bits; null when an element can't be read
/// (bf16Bits) or the constant is an expression. Without this, a bitcast of
/// the bfloat16 constant stays in the SPIR-V, keeping the bfloat16 type in a
/// module that otherwise has none; IGC on a device without bfloat16 support
/// (Xe-LP) rejects the module for the SPV_KHR_bfloat16 extension.
Constant *bf16ConstantBits(Constant *constant, Type *type) {
  if (!isBF16(constant->getType()) || !type->isIntOrIntVectorTy(16))
    return nullptr;
  if (auto *fp = dyn_cast<ConstantFP>(constant)) {
    std::optional<uint16_t> bits = bf16Bits(fp);
    return bits ? ConstantInt::get(type, *bits) : nullptr;
  }
  if (auto *data = dyn_cast<ConstantDataVector>(constant))
    return ConstantDataVector::getRaw(data->getRawDataValues(),
                                      data->getNumElements(),
                                      type->getScalarType());
  if (auto *vector = dyn_cast<ConstantVector>(constant)) {
    SmallVector<Constant *, 16> elements;
    for (Value *operand : vector->operands()) {
      auto *element = cast<Constant>(operand);
      if (isa<PoisonValue>(element))
        elements.push_back(PoisonValue::get(type->getScalarType()));
      else if (isa<UndefValue>(element))
        elements.push_back(UndefValue::get(type->getScalarType()));
      else if (auto *fp = dyn_cast<ConstantFP>(element)) {
        std::optional<uint16_t> bits = bf16Bits(fp);
        if (!bits)
          return nullptr;
        elements.push_back(ConstantInt::get(type->getScalarType(), *bits));
      } else
        return nullptr;
    }
    return ConstantVector::get(elements);
  }
  return nullptr;
}

/// `constant` of aggregate type retyped to `type` as a constant, element by
/// element; null when an element is a constant `bf16ConstantBits` can't read.
/// A constant aggregate left to the extract-and-insert chain of `retyped`
/// would stay an OpCompositeExtract of a bfloat16 composite constant, since
/// the builder doesn't fold (NoFolder).
Constant *aggregateConstantBits(Constant *constant, Type *type) {
  if (constant->getType() == type)
    return constant;
  if (isa<PoisonValue>(constant))
    return PoisonValue::get(type);
  if (isa<UndefValue>(constant))
    return UndefValue::get(type);
  if (constant->isNullValue())
    return Constant::getNullValue(type);
  if (!type->isAggregateType()) {
    if (isBF16(constant->getType()))
      return bf16ConstantBits(constant, type);
    return nullptr;
  }
  unsigned count = isa<StructType>(type)
                       ? cast<StructType>(type)->getNumElements()
                       : cast<ArrayType>(type)->getNumElements();
  SmallVector<Constant *> elements;
  for (unsigned i = 0; i < count; ++i) {
    Constant *element = constant->getAggregateElement(i);
    Type *elementType = isa<StructType>(type)
                            ? cast<StructType>(type)->getElementType(i)
                            : cast<ArrayType>(type)->getElementType();
    if (!element)
      return nullptr;
    element = aggregateConstantBits(element, elementType);
    if (!element)
      return nullptr;
    elements.push_back(element);
  }
  if (auto *structType = dyn_cast<StructType>(type))
    return ConstantStruct::get(structType, elements);
  return ConstantArray::get(cast<ArrayType>(type), elements);
}

/// `value` retyped to `type` (asIntegers of its type, or back), bit for bit,
/// built just before `before`: a bitcast for scalars and vectors, element by
/// element for structs and arrays. A bfloat16 constant becomes the integer
/// constant of its bits (bf16ConstantBits); any other constant is not read
/// (its APFloat belongs to the host's LLVM, see above) and becomes a bitcast
/// instruction, which codegen folds or emits.
Value *retyped(Value *value, Type *type, Instruction *before) {
  if (value->getType() == type)
    return value;
  if (isa<PoisonValue>(value))
    return PoisonValue::get(type);
  if (isa<UndefValue>(value))
    return UndefValue::get(type);
  if (auto *constant = dyn_cast<Constant>(value)) {
    if (constant->isNullValue())
      return Constant::getNullValue(type);
    if (Constant *bits = bf16ConstantBits(constant, type))
      return bits;
    if (type->isAggregateType())
      if (Constant *bits = aggregateConstantBits(constant, type))
        return bits;
  }
  Builder builder(before);
  if (!type->isAggregateType())
    return builder.CreateBitCast(value, type);
  unsigned count = isa<StructType>(type)
                       ? cast<StructType>(type)->getNumElements()
                       : cast<ArrayType>(type)->getNumElements();
  Value *result = PoisonValue::get(type);
  for (unsigned i = 0; i < count; ++i) {
    Type *elementType = isa<StructType>(type)
                            ? cast<StructType>(type)->getElementType(i)
                            : cast<ArrayType>(type)->getElementType();
    Value *element =
        retyped(builder.CreateExtractValue(value, i), elementType, before);
    result = builder.CreateInsertValue(result, element, i);
  }
  return result;
}

/// The i16 version of bfloat16 move `inst`, built before it with poison in
/// place of each operand that holds bfloat16 (moveAsIntegers fills them).
Instruction *movedShell(Instruction &inst) {
  Builder builder(&inst);
  auto shell = [](Value *value) -> Value * {
    return holdsBF16(value->getType())
               ? PoisonValue::get(asIntegers(value->getType()))
               : value;
  };
  Instruction *moved = nullptr;
  if (auto *store = dyn_cast<StoreInst>(&inst)) {
    StoreInst *s = builder.CreateAlignedStore(
        shell(store->getValueOperand()), store->getPointerOperand(),
        store->getAlign(), store->isVolatile());
    s->setAtomic(store->getOrdering(), store->getSyncScopeID());
    moved = s;
  } else if (auto *load = dyn_cast<LoadInst>(&inst)) {
    LoadInst *l = builder.CreateAlignedLoad(
        asIntegers(load->getType()), load->getPointerOperand(),
        load->getAlign(), load->isVolatile());
    l->setAtomic(load->getOrdering(), load->getSyncScopeID());
    moved = l;
  } else if (auto *phi = dyn_cast<PHINode>(&inst)) {
    PHINode *p = builder.CreatePHI(asIntegers(phi->getType()),
                                   phi->getNumIncomingValues());
    for (BasicBlock *from : phi->blocks())
      p->addIncoming(PoisonValue::get(p->getType()), from);
    moved = p;
  } else if (auto *select = dyn_cast<SelectInst>(&inst)) {
    // Fast-math flags don't apply to integers and are dropped.
    moved = cast<Instruction>(builder.CreateSelect(
        select->getCondition(), shell(select->getTrueValue()),
        shell(select->getFalseValue())));
  } else {
    // freeze, vector element and aggregate moves: the same operation on i16
    // operands.
    moved = inst.clone();
    moved->mutateType(asIntegers(inst.getType()));
    for (Use &use : moved->operands())
      use.set(shell(use.get()));
    moved->insertBefore(inst.getIterator());
  }
  if (!isa<PHINode>(moved) && !isa<CallBase>(moved))
    moved->copyMetadata(inst);
  return moved;
}

/// Moves bfloat16 values as i16: IGC computes wrong bits for
/// BFloat16KHR selects and phis, also with -cl-opt-disable when a select's
/// result is converted. After this only
/// conversions to and from other float types, bitcasts to and from i16,
/// calls and returns see bfloat16 values; aggregates holding bfloat16 hold
/// i16 instead.
bool moveAsIntegers(Module &module) {
  SmallVector<Instruction *> work;
  for (Function &func : module)
    for (Instruction &inst : instructions(func))
      if (movesBF16(inst))
        work.push_back(&inst);

  // Every move gets its i16 version first, so that operands can refer to
  // moves later in the function (loop phis).
  DenseMap<Value *, Instruction *> moved;
  for (Instruction *inst : work)
    moved[inst] = movedShell(*inst);

  // The i16 version of operand `value` of a move, used just before `before`.
  auto integers = [&](Value *value, Instruction *before) -> Value * {
    if (Instruction *m = moved.lookup(value))
      return m;
    return retyped(value, asIntegers(value->getType()), before);
  };
  for (Instruction *inst : work) {
    Instruction *m = moved[inst];
    if (auto *phi = dyn_cast<PHINode>(inst)) {
      // Each incoming value is retyped at the end of its block. llc's
      // CodeGenPrepare turns an integer or float phi into one of the type its
      // incoming values are bitcast from or its users bitcast it to, when
      // they are all bitcasts, loads, element extracts, constants or phis
      // (optimizePhiType), which would make it a bfloat16 phi again; freezing
      // each incoming value, which the SPIR-V backend emits as a copy, stops
      // that. A block can come in more than once (a switch with several
      // cases going to the same block); its entries must stay one value, so
      // each block's is made once and reused.
      auto *p = cast<PHINode>(m);
      DenseMap<BasicBlock *, Value *> fromBlock;
      for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
        BasicBlock *from = phi->getIncomingBlock(i);
        Value *&incoming = fromBlock[from];
        if (!incoming) {
          Instruction *end = from->getTerminator();
          incoming = integers(phi->getIncomingValue(i), end);
          if (!p->getType()->isAggregateType() && !p->getType()->isVectorTy())
            incoming = Builder(end).CreateFreeze(incoming);
        }
        p->setIncomingValue(i, incoming);
      }
      continue;
    }
    for (unsigned i = 0; i < inst->getNumOperands(); ++i)
      if (holdsBF16(inst->getOperand(i)->getType()))
        m->setOperand(i, integers(inst->getOperand(i), m));
  }

  // Users that aren't moves (conversions, calls, returns) get the value back
  // in its bfloat16 type.
  for (Instruction *inst : work) {
    Instruction *m = moved[inst];
    m->takeName(inst);
    auto notMoved = [&](Use &use) {
      return !moved.count(cast<Instruction>(use.getUser()));
    };
    if (none_of(inst->uses(), notMoved))
      continue;
    Instruction *at = isa<PHINode>(m) ? &*m->getParent()->getFirstInsertionPt()
                                      : m->getNextNode();
    inst->replaceUsesWithIf(retyped(m, inst->getType(), at), notMoved);
  }
  for (Instruction *inst : work)
    inst->dropAllReferences();
  for (Instruction *inst : work)
    inst->eraseFromParent();
  return !work.empty();
}

/// `type` (float32 or bfloat16, scalar or vector) with `bits`-wide integer
/// elements.
Type *integersOf(Type *type, unsigned bits) {
  Type *integer = Type::getIntNTy(type->getContext(), bits);
  if (auto *vector = dyn_cast<VectorType>(type))
    return VectorType::get(integer, vector->getElementCount());
  return integer;
}

/// The i16 bits of bfloat16 `value`, used before `before`: the i16 value it
/// was retyped from when it is a bitcast of one, the integer constant of a
/// bfloat16 constant (bf16ConstantBits), else a bitcast of it.
Value *bitsOf(Value *value, Instruction *before) {
  Type *type = integersOf(value->getType(), 16);
  if (auto *cast = dyn_cast<BitCastInst>(value))
    if (cast->getOperand(0)->getType() == type)
      return cast->getOperand(0);
  if (auto *constant = dyn_cast<Constant>(value))
    if (Constant *bits = bf16ConstantBits(constant, type))
      return bits;
  return Builder(before).CreateBitCast(value, type);
}

/// The float32 bits (i32) of bfloat16 `bits` (i16): bfloat16 is float32's
/// upper half, so this is exact, NaN payloads included.
Value *extendedBits(Builder &builder, Value *bits) {
  Type *i32 = integersOf(bits->getType(), 32);
  Value *wide = builder.CreateZExt(bits, i32);
  return builder.CreateShl(wide, ConstantInt::get(i32, 16));
}

/// The bfloat16 bits (i16 or a vector of them) of float32 `value` rounded to
/// nearest even, bit for bit as the host converts (Mojo's __truncsfbf2,
/// Mojo/lib/CompilerRT/Support.cpp): add 0x7fff plus the lowest kept bit and
/// keep the upper half (a carry into the exponent gives the next binade or
/// infinity). A NaN keeps its sign and becomes quiet: from a float32 source it
/// also keeps the upper half of its payload, as the host converts float32 since
/// upstream's LLVM 9b2201bd0e35 (inline); `canonicalNaN` gives 0x7fc0 with the
/// sign instead, as the host converts float64 (through __truncdfbf2, whose
/// float2bfloat makes every NaN 0x7fc0).
Value *roundedBits(Builder &builder, Value *value, bool canonicalNaN) {
  Type *i32 = integersOf(value->getType(), 32);
  auto k = [&](uint32_t v) { return ConstantInt::get(i32, v); };
  Value *bits = builder.CreateBitCast(value, i32);
  Value *magnitude = builder.CreateAnd(bits, k(0x7fffffff));
  Value *nan = builder.CreateICmpUGT(magnitude, k(0x7f800000));
  Value *lowestKept = builder.CreateAnd(builder.CreateLShr(bits, k(16)), k(1));
  Value *rounded =
      builder.CreateAdd(bits, builder.CreateAdd(lowestKept, k(0x7fff)));
  Value *quiet = canonicalNaN
                     ? builder.CreateOr(builder.CreateAnd(bits, k(0x80000000)),
                                        k(0x7fc00000))
                     : builder.CreateOr(bits, k(0x00400000));
  Value *result = builder.CreateSelect(nan, quiet, rounded);
  return builder.CreateTrunc(builder.CreateLShr(result, k(16)),
                             integersOf(value->getType(), 16));
}

/// `convert` applied to `value`, lane by lane when it is a vector (extract,
/// convert the scalar, insert). IGC's optimizer gets the vector form wrong:
/// a probe of the stdlib's bfloat16 erf at widths 4, 8 and 16 gave NaN for 4
/// to 7 of its 8 inputs with the vector form, all right lane by lane, as at
/// width 1; without the NaN select the vector form was right too.
template <typename Convert>
Value *perLane(Builder &builder, Value *value, Type *resultElement,
               Convert convert) {
  auto *vector = dyn_cast<FixedVectorType>(value->getType());
  if (!vector)
    return convert(builder, value);
  Value *result = PoisonValue::get(
      FixedVectorType::get(resultElement, vector->getNumElements()));
  for (unsigned i = 0; i < vector->getNumElements(); ++i) {
    Value *lane = builder.CreateExtractElement(value, builder.getInt64(i));
    result = builder.CreateInsertElement(result, convert(builder, lane),
                                         builder.getInt64(i));
  }
  return result;
}

/// Converts between bfloat16 and float32 with integer operations instead of
/// fpext and fptrunc: IGC's optimizer turns float32 arithmetic between
/// such conversions into bfloat16 arithmetic and computes that wrongly;
/// without an OpFConvert there is
/// nothing to narrow. Vectors convert lane by lane (perLane). bfloat16 to
/// float64 goes through float32 (exact), and float64 to bfloat16 too (an
/// fptrunc to float32 first): that rounds twice, as the host does (Mojo's
/// __truncdfbf2).
bool convertAsIntegers(Module &module) {
  SmallVector<Instruction *> work;
  for (Function &func : module)
    for (Instruction &inst : instructions(func)) {
      if (isa<FPExtInst>(inst) && isBF16(inst.getOperand(0)->getType()))
        work.push_back(&inst);
      if (isa<FPTruncInst>(inst) && isBF16(inst.getType()))
        work.push_back(&inst);
    }

  for (Instruction *inst : work) {
    Builder builder(inst);
    Value *source = inst->getOperand(0);
    if (isa<FPExtInst>(inst)) {
      // Lanes are built as integers, so no float vector constant (poison)
      // is made.
      Value *wide =
          perLane(builder, bitsOf(source, inst), builder.getInt32Ty(),
                  [](Builder &b, Value *v) { return extendedBits(b, v); });
      Value *result = builder.CreateBitCast(wide, widened(source->getType()));
      if (result->getType() != inst->getType())
        result = builder.CreateFPExt(result, inst->getType());
      result->takeName(inst);
      inst->replaceAllUsesWith(result);
    } else {
      Value *narrow = source;
      bool fromFloat32 = narrow->getType()->getScalarType()->isFloatTy();
      if (!fromFloat32)
        narrow = builder.CreateFPTrunc(narrow, widened(inst->getType()));
      Value *bits = perLane(builder, narrow, builder.getInt16Ty(),
                            [&](Builder &b, Value *v) {
                              return roundedBits(b, v, !fromFloat32);
                            });
      // Users that take the bits (moves, through moveAsIntegers's bitcasts)
      // get them directly; the others get them as bfloat16.
      Value *asBF16 = nullptr;
      for (Use &use : make_early_inc_range(inst->uses())) {
        auto *cast = dyn_cast<BitCastInst>(use.getUser());
        if (cast && cast->getType() == bits->getType()) {
          cast->replaceAllUsesWith(bits);
          cast->eraseFromParent();
          continue;
        }
        if (!asBF16)
          asBF16 = builder.CreateBitCast(bits, inst->getType());
        use.set(asBF16);
      }
      if (asBF16)
        asBF16->takeName(inst);
    }
    inst->eraseFromParent();
    // A bitcast from i16 that only fed the conversion.
    if (auto *cast = dyn_cast<BitCastInst>(source))
      if (cast->use_empty())
        cast->eraseFromParent();
  }
  return !work.empty();
}

} // namespace

/// Folds a bitcast of a bitcast back to the first operand's type (i16 ->
/// bfloat16 -> i16, which the moves and conversions build in either order
/// when a value's producer and users are rewritten separately) and erases
/// the dead casts. The SPIR-V backend emits the pair as written, and the
/// bfloat16 type with it; a module that otherwise has none then still asks
/// for SPV_KHR_bfloat16, which IGC on Xe-LP refuses. Returns whether
/// anything changed.
bool foldBitcastPairs(Module &module) {
  bool changed = false;
  for (Function &func : module) {
    SmallVector<BitCastInst *> dead;
    for (Instruction &inst : instructions(func)) {
      auto *outer = dyn_cast<BitCastInst>(&inst);
      if (outer == nullptr)
        continue;
      auto *inner = dyn_cast<BitCastInst>(outer->getOperand(0));
      if (inner == nullptr ||
          inner->getOperand(0)->getType() != outer->getType())
        continue;
      outer->replaceAllUsesWith(inner->getOperand(0));
      dead.push_back(outer);
      changed = true;
    }
    for (BitCastInst *cast : dead) {
      Value *inner = cast->getOperand(0);
      cast->eraseFromParent();
      if (auto *innerCast = dyn_cast<BitCastInst>(inner))
        if (innerCast->use_empty())
          innerCast->eraseFromParent();
    }
  }
  return changed;
}

/// Replaces llvm.vector.interleave2 and llvm.vector.deinterleave2 of bfloat16
/// vectors with shufflevectors, which moveAsIntegers moves as i16 vectors. The
/// intrinsics (a rope's even/odd pairs) would otherwise reach the SPIR-V
/// backend with bfloat16 operands and ask for SPV_KHR_bfloat16, and the
/// promotion to float32 would not see their struct result. Returns whether
/// anything changed.
bool lowerBF16Interleaves(Module &module) {
  SmallVector<IntrinsicInst *> work;
  for (Function &func : module)
    for (Instruction &inst : instructions(func))
      if (auto *call = dyn_cast<IntrinsicInst>(&inst))
        if ((call->getIntrinsicID() == Intrinsic::vector_interleave2 ||
             call->getIntrinsicID() == Intrinsic::vector_deinterleave2) &&
            isa<FixedVectorType>(call->getOperand(0)->getType()) &&
            isBF16(call->getOperand(0)->getType()))
          work.push_back(call);

  for (IntrinsicInst *call : work) {
    Builder builder(call);
    Value *source = call->getOperand(0);
    unsigned count = cast<FixedVectorType>(source->getType())->getNumElements();
    Value *result;
    if (call->getIntrinsicID() == Intrinsic::vector_interleave2) {
      SmallVector<int> mask;
      for (unsigned i = 0; i < count; ++i) {
        mask.push_back(i);
        mask.push_back(count + i);
      }
      result = builder.CreateShuffleVector(source, call->getOperand(1), mask);
    } else {
      // The result is a struct of the even and the odd elements.
      result = PoisonValue::get(call->getType());
      for (unsigned part = 0; part < 2; ++part) {
        SmallVector<int> mask;
        for (unsigned i = 0; i < count / 2; ++i)
          mask.push_back(2 * i + part);
        Value *half = builder.CreateShuffleVector(
            source, PoisonValue::get(source->getType()), mask);
        result = builder.CreateInsertValue(result, half, part);
      }
    }
    result->takeName(call);
    call->replaceAllUsesWith(result);
    call->eraseFromParent();
  }
  return !work.empty();
}

bool promoteBFloat16(Module &module) {
  bool lowered = lowerBF16Interleaves(module);
  SmallVector<Instruction *> work;
  for (Function &func : module)
    for (Instruction &inst : instructions(func))
      if (needsPromotion(inst))
        work.push_back(&inst);

  for (Instruction *inst : work) {
    Builder builder(inst);
    if (isa<FPMathOperator>(inst))
      builder.setFastMathFlags(inst->getFastMathFlags());
    Value *result = nullptr;
    if (auto *binary = dyn_cast<BinaryOperator>(inst)) {
      result = builder.CreateBinOp(binary->getOpcode(),
                                   extend(builder, binary->getOperand(0)),
                                   extend(builder, binary->getOperand(1)));
    } else if (auto *unary = dyn_cast<UnaryOperator>(inst)) {
      result = builder.CreateUnOp(unary->getOpcode(),
                                  extend(builder, unary->getOperand(0)));
    } else if (auto *cmp = dyn_cast<FCmpInst>(inst)) {
      result = builder.CreateFCmp(cmp->getPredicate(),
                                  extend(builder, cmp->getOperand(0)),
                                  extend(builder, cmp->getOperand(1)));
    } else if (isa<SIToFPInst>(inst) || isa<UIToFPInst>(inst)) {
      result = roundedToOdd(builder, inst->getOperand(0), isa<SIToFPInst>(inst),
                            widened(inst->getType()));
    } else if (isa<FPToSIInst>(inst) || isa<FPToUIInst>(inst)) {
      result = builder.CreateCast(cast<CastInst>(inst)->getOpcode(),
                                  extend(builder, inst->getOperand(0)),
                                  inst->getType());
    } else if (auto *call = dyn_cast<IntrinsicInst>(inst)) {
      result = promoteIntrinsic(builder, *call);
    }
    if (!result)
      continue;
    if (isBF16(inst->getType()) && !isBF16(result->getType()))
      result = builder.CreateFPTrunc(result, inst->getType());
    result->takeName(inst);
    inst->replaceAllUsesWith(result);
    inst->eraseFromParent();
  }
  bool moved = moveAsIntegers(module);
  bool converted = convertAsIntegers(module);
  bool folded = foldBitcastPairs(module);
  return lowered || !work.empty() || moved || converted || folded;
}

} // namespace M::KGEN::Intel
