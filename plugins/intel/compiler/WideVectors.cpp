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

#include "WideVectors.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Scalar/Scalarizer.h"

#include <optional>

using namespace llvm;

namespace M::KGEN::Intel {
namespace {

bool isWide(Type *type) {
  auto *vector = dyn_cast<FixedVectorType>(type);
  return vector && vector->getNumElements() > kMaxVectorElements;
}

/// Whether `type` holds a wide vector, at any depth.
bool holdsWide(Type *type) {
  if (isWide(type))
    return true;
  return any_of(type->subtypes(), holdsWide);
}

/// `type` with every wide vector in it, at any depth, an array of its
/// elements.
Type *withArrays(Type *type) {
  if (auto *vector = dyn_cast<FixedVectorType>(type))
    return isWide(type) ? ArrayType::get(vector->getElementType(),
                                         vector->getNumElements())
                        : type;
  if (auto *structType = dyn_cast<StructType>(type)) {
    SmallVector<Type *> elements;
    for (Type *element : structType->elements())
      elements.push_back(withArrays(element));
    return StructType::get(type->getContext(), elements,
                           structType->isPacked());
  }
  if (auto *array = dyn_cast<ArrayType>(type))
    return ArrayType::get(withArrays(array->getElementType()),
                          array->getNumElements());
  return type;
}

/// Whether vector elements of type `element` lie in memory as an array's
/// do: only if they fill whole bytes (an i1 vector is packed in bits).
bool fillsBytes(Type *element, const DataLayout &layout) {
  return layout.getTypeSizeInBits(element) ==
         layout.getTypeAllocSizeInBits(element);
}

/// Whether `a` and `b` lay out alike: the same size, and every member at the
/// same offset.
bool sameLayout(Type *a, Type *b, const DataLayout &layout) {
  if (layout.getTypeAllocSize(a) != layout.getTypeAllocSize(b))
    return false;
  if (auto *vector = dyn_cast<FixedVectorType>(a))
    return fillsBytes(vector->getElementType(), layout);
  if (auto *structA = dyn_cast<StructType>(a)) {
    auto *structB = cast<StructType>(b);
    const StructLayout *layoutA = layout.getStructLayout(structA);
    const StructLayout *layoutB = layout.getStructLayout(structB);
    for (unsigned i = 0; i < structA->getNumElements(); ++i)
      if (layoutA->getElementOffset(i) != layoutB->getElementOffset(i) ||
          !sameLayout(structA->getElementType(i), structB->getElementType(i),
                      layout))
        return false;
    return true;
  }
  if (auto *array = dyn_cast<ArrayType>(a))
    return sameLayout(array->getElementType(),
                      cast<ArrayType>(b)->getElementType(), layout);
  return true;
}

/// The type `type` takes in memory: itself where it holds no wide vector,
/// else the same type with its wide vectors as arrays of their elements,
/// where that lays out the same (for example a vector placed by explicit
/// padding), else null.
Type *memoryType(Type *type, const DataLayout &layout) {
  if (!holdsWide(type))
    return type;
  Type *arrays = withArrays(type);
  return sameLayout(type, arrays, layout) ? arrays : nullptr;
}

/// Gives GEPs and allocas whose element type holds a wide vector the same
/// type with arrays instead: the Scalarizer splits values, not the types
/// pointers are indexed by, and SPIR-V would declare the vector type for
/// them.
void retypeWidePointees(Function &func) {
  const DataLayout &layout = func.getParent()->getDataLayout();
  for (Instruction &inst : instructions(func)) {
    if (auto *gep = dyn_cast<GetElementPtrInst>(&inst)) {
      Type *type = memoryType(gep->getSourceElementType(), layout);
      if (!type || type == gep->getSourceElementType())
        continue;
      SmallVector<Value *> indices(gep->indices());
      gep->setSourceElementType(type);
      // getIndexedType skips the first index itself.
      gep->setResultElementType(
          GetElementPtrInst::getIndexedType(type, indices));
    } else if (auto *alloca = dyn_cast<AllocaInst>(&inst)) {
      Type *type = memoryType(alloca->getAllocatedType(), layout);
      if (type && type != alloca->getAllocatedType())
        alloca->setAllocatedType(type);
    }
  }
}

/// Recreates globals whose type holds a wide vector with arrays instead,
/// where the layout is the same and the initializer carries no data
/// (shared memory: poison or undef; or zero).
void retypeWideGlobals(Module &module) {
  const DataLayout &layout = module.getDataLayout();
  SmallVector<GlobalVariable *> globals;
  for (GlobalVariable &global : module.globals())
    if (holdsWide(global.getValueType()))
      globals.push_back(&global);
  for (GlobalVariable *global : globals) {
    Type *type = memoryType(global->getValueType(), layout);
    if (!type)
      continue;
    Constant *init = nullptr;
    if (global->hasInitializer()) {
      Constant *old = global->getInitializer();
      if (isa<PoisonValue>(old))
        init = PoisonValue::get(type);
      else if (isa<UndefValue>(old))
        init = UndefValue::get(type);
      else if (old->isNullValue())
        init = Constant::getNullValue(type);
      else
        continue;
    }
    auto *copy = new GlobalVariable(
        module, type, global->isConstant(), global->getLinkage(), init, "",
        global, global->getThreadLocalMode(), global->getAddressSpace(),
        global->isExternallyInitialized());
    copy->copyAttributesFrom(global);
    copy->setAlignment(global->getAlign().value_or(
        layout.getPrefTypeAlign(global->getValueType())));
    copy->takeName(global);
    global->replaceAllUsesWith(copy);
    global->eraseFromParent();
  }
}

bool usesWideVectors(Function &func) {
  for (Instruction &inst : instructions(func)) {
    if (isWide(inst.getType()))
      return true;
    if (auto *gep = dyn_cast<GetElementPtrInst>(&inst);
        gep && holdsWide(gep->getSourceElementType()))
      return true;
    if (auto *alloca = dyn_cast<AllocaInst>(&inst);
        alloca && holdsWide(alloca->getAllocatedType()))
      return true;
    for (Value *operand : inst.operands())
      if (isWide(operand->getType()))
        return true;
  }
  return false;
}

/// Where to compute a replacement for `use`: before its user, or for a phi,
/// at the end of the incoming block.
BasicBlock::iterator insertionPoint(Use &use) {
  auto *user = cast<Instruction>(use.getUser());
  if (auto *phi = dyn_cast<PHINode>(user))
    return phi->getIncomingBlock(use)->getTerminator()->getIterator();
  return user->getIterator();
}

/// The bits of `value`, a float the host made. This copy of LLVM can't read
/// them directly: APFloat finds a format's encoding by the address of its
/// semantics object, and the host's isn't this copy's. Converting to this
/// copy's semantics of the same format reads only the format's parameters,
/// is exact, and gives a value this copy can encode.
APInt bitsOf(const APFloat &value, Type *type) {
  APFloat local = value;
  bool losesInfo = false;
  local.convert(type->getFltSemantics(), APFloat::rmNearestTiesToEven,
                &losesInfo);
  return local.bitcastToAPInt();
}

/// The bits of `constant`, a float vector, as an integer array of the same
/// length; null where they can't be read (a constant expression element).
Constant *bitsArray(Constant *constant) {
  auto *type = cast<FixedVectorType>(constant->getType());
  Type *element = type->getElementType();
  unsigned count = type->getNumElements();
  unsigned bits = type->getScalarSizeInBits();
  auto *ints =
      ArrayType::get(IntegerType::get(type->getContext(), bits), count);
  if (auto *data = dyn_cast<ConstantDataVector>(constant))
    return ConstantDataArray::getRaw(data->getRawDataValues(), count,
                                     ints->getElementType());
  if (constant->isNullValue())
    return ConstantAggregateZero::get(ints);
  // A splat (a vector-typed ConstantFP) or a vector of elements, some undef.
  SmallVector<Constant *> elements;
  for (unsigned i = 0; i < count; ++i) {
    APInt value(bits, 0);
    if (auto *splat = dyn_cast<ConstantFP>(constant)) {
      value = bitsOf(splat->getValueAPF(), element);
    } else {
      auto *vector = dyn_cast<ConstantVector>(constant);
      if (!vector)
        return nullptr;
      Constant *item = vector->getOperand(i);
      if (auto *fp = dyn_cast<ConstantFP>(item))
        value = bitsOf(fp->getValueAPF(), element);
      else if (!isa<UndefValue>(item))
        return nullptr;
    }
    elements.push_back(ConstantInt::get(ints->getElementType(), value));
  }
  return ConstantArray::get(ints, elements);
}

/// `constant`, a float vector whose bits bitsArray can't read, built by
/// instructions the Scalarizer splits without folding a float constant: a
/// constant vector lane by lane (a float lane as a bitcast of its bits, a
/// constant expression lane as it is), a bitcast of an integer vector as a
/// bitcast of that vector frozen (its lanes are then split as integers).
/// Those are the only float vector constants LLVM has besides the ones
/// bitsArray reads; anything else is left as it is.
Value *rebuilt(IRBuilder<NoFolder> &builder, Constant *constant) {
  auto *type = cast<FixedVectorType>(constant->getType());
  if (auto *vector = dyn_cast<ConstantVector>(constant)) {
    Value *result = PoisonValue::get(type);
    for (unsigned i = 0; i < type->getNumElements(); ++i) {
      Constant *lane = vector->getOperand(i);
      Value *value = lane;
      if (auto *fp = dyn_cast<ConstantFP>(lane))
        value = builder.CreateBitCast(
            builder.getInt(bitsOf(fp->getValueAPF(), lane->getType())),
            lane->getType());
      result = builder.CreateInsertElement(result, value, i);
    }
    return result;
  }
  if (auto *expr = dyn_cast<ConstantExpr>(constant);
      expr && expr->getOpcode() == Instruction::BitCast &&
      !expr->getOperand(0)->getType()->getScalarType()->isFloatingPointTy())
    return builder.CreateBitCast(builder.CreateFreeze(expr->getOperand(0)),
                                 type);
  return constant;
}

/// Moves the function's float vector constants into private constant-memory
/// globals of their bits, loaded as integers and bitcast. The Scalarizer then
/// splits each into integer loads and per-element bitcasts, where splitting
/// the constant itself would fold out new float constants. The globals are
/// arrays: a wide vector initializer would reach codegen whole. A constant
/// whose bits can't be read is rebuilt at each use instead (`rebuilt`).
void loadFloatConstants(Function &func) {
  Module &module = *func.getParent();
  const DataLayout &layout = module.getDataLayout();
  constexpr unsigned kUniformConstant = 2;
  SmallVector<Use *> uses;
  for (Instruction &inst : instructions(func))
    for (Use &use : inst.operands()) {
      auto *constant = dyn_cast<Constant>(use.get());
      if (constant && !isa<UndefValue>(constant) &&
          isa<FixedVectorType>(constant->getType()) &&
          constant->getType()->getScalarType()->isFloatingPointTy())
        uses.push_back(&use);
    }
  // Null for a constant bitsArray can't read, which is rebuilt per use.
  DenseMap<Constant *, GlobalVariable *> globals;
  for (Use *use : uses) {
    auto *constant = cast<Constant>(use->get());
    auto *type = cast<FixedVectorType>(constant->getType());
    auto [entry, inserted] = globals.try_emplace(constant, nullptr);
    if (inserted)
      if (Constant *bits = bitsArray(constant)) {
        auto *global = new GlobalVariable(
            module, bits->getType(), /*isConstant=*/true,
            GlobalValue::PrivateLinkage, bits, "wide.constant",
            /*InsertBefore=*/nullptr, GlobalValue::NotThreadLocal,
            kUniformConstant);
        global->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
        global->setAlignment(layout.getABITypeAlign(type));
        entry->second = global;
      }
    GlobalVariable *global = entry->second;
    IRBuilder<NoFolder> builder(insertionPoint(*use)->getParent(),
                                insertionPoint(*use));
    if (!global) {
      use->set(rebuilt(builder, constant));
      continue;
    }
    auto *ints = FixedVectorType::get(
        builder.getIntNTy(type->getScalarSizeInBits()), type->getNumElements());
    Value *loaded = builder.CreateAlignedLoad(ints, global,
                                              global->getAlign().valueOrOne());
    use->set(builder.CreateBitCast(loaded, type));
  }
}

/// `type` with a wide vector as an array of its elements, or for a kernel
/// parameter as a struct of them: the host passes the vector's bytes, and IGC
/// reads a struct of scalars passed by value but not one holding an array.
/// Either has the vector's elements at the vector's offsets.
Type *unwidened(Type *type, bool kernelParameter = false) {
  if (!isWide(type))
    return type;
  auto *vector = cast<FixedVectorType>(type);
  if (kernelParameter)
    return StructType::get(type->getContext(),
                           SmallVector<Type *>(vector->getNumElements(),
                                               vector->getElementType()));
  return ArrayType::get(vector->getElementType(), vector->getNumElements());
}

/// `value` as `to`: a wide vector and an aggregate of its elements (an array
/// or a struct), either way round, element by element; the Scalarizer splits
/// the element moves. Nothing is folded, so no float constant is made.
Value *convert(IRBuilder<NoFolder> &builder, Value *value, Type *to) {
  Value *result = PoisonValue::get(to);
  bool toAggregate = !isa<FixedVectorType>(to);
  unsigned count =
      toAggregate ? cast<FixedVectorType>(value->getType())->getNumElements()
                  : cast<FixedVectorType>(to)->getNumElements();
  for (unsigned i = 0; i < count; ++i) {
    Value *element = toAggregate ? builder.CreateExtractElement(value, i)
                                 : builder.CreateExtractValue(value, i);
    result = toAggregate ? builder.CreateInsertValue(result, element, i)
                         : builder.CreateInsertElement(result, element, i);
  }
  return result;
}

/// Passes wide vectors between functions as arrays, and to kernels as
/// structs: the Scalarizer splits what a function computes, not its
/// signature. Functions whose address is taken keep theirs.
void unwidenSignatures(Module &module) {
  SmallVector<Function *> functions;
  for (Function &func : module) {
    if (func.isDeclaration())
      continue;
    bool wide = isWide(func.getReturnType()) ||
                any_of(func.args(),
                       [](Argument &arg) { return isWide(arg.getType()); });
    bool directlyCalled = all_of(func.users(), [&](User *user) {
      auto *call = dyn_cast<CallInst>(user);
      return call && call->getCalledOperand() == &func;
    });
    if (wide && directlyCalled)
      functions.push_back(&func);
  }

  for (Function *func : functions) {
    bool kernel = func->getCallingConv() == CallingConv::SPIR_KERNEL;
    SmallVector<Type *> params;
    for (Argument &arg : func->args())
      params.push_back(unwidened(arg.getType(), kernel));
    auto *type = FunctionType::get(unwidened(func->getReturnType()), params,
                                   func->isVarArg());
    Function *copy = Function::Create(type, func->getLinkage(),
                                      func->getAddressSpace(), "", &module);
    copy->copyAttributesFrom(func);
    copy->copyMetadata(func, 0);
    // Attributes of the vectors don't apply to the arrays.
    AttributeList attrs = copy->getAttributes();
    for (Argument &arg : func->args())
      if (isWide(arg.getType()))
        attrs =
            attrs.removeParamAttributes(module.getContext(), arg.getArgNo());
    if (isWide(func->getReturnType()))
      attrs = attrs.removeRetAttributes(module.getContext(),
                                        AttributeMask(attrs.getRetAttrs()));
    copy->setAttributes(attrs);
    copy->splice(copy->end(), func);

    IRBuilder<NoFolder> entry(&*copy->getEntryBlock().getFirstInsertionPt());
    for (auto [from, to] : zip(func->args(), copy->args())) {
      to.takeName(&from);
      from.replaceAllUsesWith(isWide(from.getType())
                                  ? convert(entry, &to, from.getType())
                                  : static_cast<Value *>(&to));
    }
    if (isWide(func->getReturnType()))
      for (BasicBlock &block : *copy)
        if (auto *ret = dyn_cast<ReturnInst>(block.getTerminator())) {
          IRBuilder<NoFolder> builder(ret);
          ret->setOperand(0, convert(builder, ret->getReturnValue(),
                                     copy->getReturnType()));
        }

    SmallVector<CallInst *> calls;
    for (User *user : func->users())
      calls.push_back(cast<CallInst>(user));
    for (CallInst *call : calls) {
      IRBuilder<NoFolder> builder(call);
      SmallVector<Value *> args;
      for (Use &arg : call->args())
        args.push_back(isWide(arg->getType())
                           ? convert(builder, arg, unwidened(arg->getType()))
                           : arg.get());
      CallInst *replacement = builder.CreateCall(copy, args);
      replacement->setCallingConv(call->getCallingConv());
      // The call's own attributes, less those of its wide vectors.
      AttributeList callAttrs = call->getAttributes();
      for (Use &arg : call->args())
        if (isWide(arg->getType()))
          callAttrs = callAttrs.removeParamAttributes(
              module.getContext(), call->getArgOperandNo(&arg));
      if (isWide(call->getType()))
        callAttrs = callAttrs.removeRetAttributes(
            module.getContext(), AttributeMask(callAttrs.getRetAttrs()));
      replacement->setAttributes(callAttrs);
      replacement->setDebugLoc(call->getDebugLoc());
      Value *result = replacement;
      if (isWide(call->getType())) {
        IRBuilder<NoFolder> after(call->getNextNode());
        result = convert(after, replacement, call->getType());
      }
      result->takeName(call);
      call->replaceAllUsesWith(result);
      call->eraseFromParent();
    }
    copy->takeName(func);
    func->eraseFromParent();
  }
}

/// The element type of a local array that holds a vector with `vector`'s
/// elements at the array's indices: the element type itself, or where it
/// doesn't fill whole bytes (i1), the integer of its allocated size, which
/// the vector is extended to before it's stored.
Type *storedElement(FixedVectorType *vector, const DataLayout &layout) {
  Type *element = vector->getElementType();
  if (fillsBytes(element, layout))
    return element;
  return IntegerType::get(element->getContext(),
                          layout.getTypeAllocSizeInBits(element));
}

/// Moves elements of wide vectors at run-time indices through memory: the
/// vector is stored to a local array and the element loaded (or stored and
/// the vector reloaded) at the index. The Scalarizer splits the vector's
/// store and load into one access per element; splitting the indexed access
/// itself would compare the index with every element, per access. A vector
/// read at run-time indices is stored once, where it's defined, and every
/// read loads from that copy: an SSA value doesn't change, so a read in a
/// loop costs one load, not a store of the whole vector.
void indexThroughMemory(Function &func) {
  SmallVector<Instruction *> accesses;
  for (Instruction &inst : instructions(func))
    if ((isa<ExtractElementInst>(inst) &&
         isWide(inst.getOperand(0)->getType()) &&
         !isa<Constant>(inst.getOperand(1))) ||
        (isa<InsertElementInst>(inst) && isWide(inst.getType()) &&
         !isa<Constant>(inst.getOperand(2))))
      accesses.push_back(&inst);
  if (accesses.empty())
    return;
  const DataLayout &layout = func.getParent()->getDataLayout();
  IRBuilder<NoFolder> entry(&*func.getEntryBlock().getFirstInsertionPt());
  // A local array for a vector, and its type.
  auto local = [&](FixedVectorType *vector) {
    auto *array =
        ArrayType::get(storedElement(vector, layout), vector->getNumElements());
    AllocaInst *alloca = entry.CreateAlloca(array, nullptr, "wide.index");
    alloca->setAlignment(layout.getPrefTypeAlign(array));
    return alloca;
  };
  // Stores `value` into `array` as its elements, extended where needed.
  auto store = [&](IRBuilder<NoFolder> &builder, Value *value,
                   AllocaInst *array) {
    auto *vector = cast<FixedVectorType>(value->getType());
    Type *element =
        cast<ArrayType>(array->getAllocatedType())->getElementType();
    if (element != vector->getElementType())
      value = builder.CreateZExt(
          value, FixedVectorType::get(element, vector->getNumElements()));
    builder.CreateStore(value, array);
  };
  // The copy of each vector that's read at run-time indices.
  DenseMap<Value *, AllocaInst *> copies;
  auto copyOf = [&](Value *vector) {
    AllocaInst *&copy = copies[vector];
    if (copy)
      return copy;
    copy = local(cast<FixedVectorType>(vector->getType()));
    std::optional<BasicBlock::iterator> after;
    if (auto *inst = dyn_cast<Instruction>(vector))
      after = inst->getInsertionPointAfterDef();
    else // an argument or a constant: the entry block, after the allocas
      after = entry.GetInsertPoint();
    // INVARIANT: the accesses are extracts and inserts, and no vector
    // definition before them is a terminator (an invoke).
    IRBuilder<NoFolder> builder(copy->getContext());
    builder.SetInsertPoint(*after);
    store(builder, vector, copy);
    return copy;
  };
  for (Instruction *access : accesses) {
    auto *vector = cast<FixedVectorType>(access->getOperand(0)->getType());
    Type *element = vector->getElementType();
    IRBuilder<NoFolder> builder(access);
    Value *index = access->getOperand(isa<ExtractElementInst>(access) ? 1 : 2);
    Value *result;
    if (isa<ExtractElementInst>(access)) {
      AllocaInst *copy = copyOf(access->getOperand(0));
      Type *stored = copy->getAllocatedType();
      result = builder.CreateLoad(
          stored->getArrayElementType(),
          builder.CreateInBoundsGEP(stored, copy, {builder.getInt64(0), index},
                                    "wide.element"));
      if (result->getType() != element)
        result = builder.CreateTrunc(result, element);
    } else {
      // An insert makes a new vector: its own array, rewritten each time.
      AllocaInst *array = local(vector);
      Type *stored = array->getAllocatedType();
      Type *storedElem = stored->getArrayElementType();
      store(builder, access->getOperand(0), array);
      Value *value = access->getOperand(1);
      if (storedElem != element)
        value = builder.CreateZExt(value, storedElem);
      builder.CreateStore(
          value,
          builder.CreateInBoundsGEP(stored, array, {builder.getInt64(0), index},
                                    "wide.element"));
      result = builder.CreateLoad(
          FixedVectorType::get(storedElem, vector->getNumElements()), array);
      if (storedElem != element)
        result = builder.CreateTrunc(result, vector);
    }
    result->takeName(access);
    access->replaceAllUsesWith(result);
  }
  // Erased only now: the copies map holds accesses as vectors too.
  for (Instruction *access : accesses)
    access->eraseFromParent();
}

} // namespace

bool splitWideVectors(Module &module) {
  // Every function touching wide vectors first loads its float constants,
  // before anything takes one apart; then signatures lose their wide
  // vectors, and then each function is split.
  bool globals = any_of(module.globals(), [](GlobalVariable &global) {
    return holdsWide(global.getValueType());
  });
  retypeWideGlobals(module);
  SmallVector<Function *> wide;
  for (Function &func : module)
    if (!func.isDeclaration() && usesWideVectors(func))
      wide.push_back(&func);
  if (wide.empty())
    return globals;
  for (Function *func : wide)
    loadFloatConstants(*func);
  unwidenSignatures(module);

  bool changed = false;
  for (Function &func : module) {
    if (func.isDeclaration() || !usesWideVectors(func))
      continue;
    indexThroughMemory(func);
    FunctionAnalysisManager analyses;
    analyses.registerPass([] { return DominatorTreeAnalysis(); });
    analyses.registerPass([] { return AssumptionAnalysis(); });
    analyses.registerPass([] { return PassInstrumentationAnalysis(); });
    // Without a target machine: the generic cost model, which the
    // Scalarizer only asks which target intrinsics it may split.
    analyses.registerPass([] { return TargetIRAnalysis(); });
    ScalarizerPassOptions options;
    options.ScalarizeMinBits = 0;
    options.ScalarizeVariableInsertExtract = true;
    options.ScalarizeLoadStore = true;
    ScalarizerPass(options).run(func, analyses);
    retypeWidePointees(func);
    changed = true;
  }
  return changed;
}

} // namespace M::KGEN::Intel
