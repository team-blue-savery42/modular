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

#include "ZeroLengthArrays.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Alignment.h"

#include <map>
#include <tuple>

using namespace llvm;

namespace M::KGEN::Intel {

bool containsZeroLengthArray(Type *type) {
  if (auto *array = dyn_cast<ArrayType>(type))
    return array->getNumElements() == 0 ||
           containsZeroLengthArray(array->getElementType());
  if (auto *structType = dyn_cast<StructType>(type))
    return any_of(structType->elements(), containsZeroLengthArray);
  return false;
}

namespace {

/// Whether `type` is or holds an array, at any depth.
bool holdsArray(Type *type) {
  if (isa<ArrayType>(type))
    return true;
  if (auto *structType = dyn_cast<StructType>(type))
    return any_of(structType->elements(), holdsArray);
  return false;
}

/// The type a scalar kernel argument, or a scalar leaf of one, is passed
/// as: a `bool` (`i1`) as its byte, a bfloat16 as its bits; anything else as
/// itself. Each takes the same bytes either way.
Type *passedAs(Type *type) {
  if (type->isIntegerTy(1))
    return Type::getInt8Ty(type->getContext());
  if (type->isBFloatTy())
    return Type::getInt16Ty(type->getContext());
  return type;
}

/// The value of original type `type` from `passed`, its `passedAs` form: a
/// nonzero byte is true; bfloat16 bits are bitcast back.
Value *readAs(IRBuilder<> &builder, Value *passed, Type *type) {
  if (type->isIntegerTy(1))
    return builder.CreateIsNotNull(passed);
  if (type->isBFloatTy())
    return builder.CreateBitCast(passed, type);
  return passed;
}

/// Whether `type` is or holds a scalar passed as another type (`passedAs`),
/// at any depth.
bool holdsRetyped(Type *type) {
  if (auto *array = dyn_cast<ArrayType>(type))
    return holdsRetyped(array->getElementType());
  if (auto *structType = dyn_cast<StructType>(type))
    return any_of(structType->elements(), holdsRetyped);
  return passedAs(type) != type;
}

/// Whether kernel parameter `arg` is aggregate and passed in another form:
/// one that is or holds an array, holds another aggregate, or holds a `bool`
/// or a bfloat16. IGC reads a flat struct of scalars passed by value, but
/// read the second field of a nested `{ { i64, i64 }, i64 }` from the first
/// one's bytes. OpenCL allows no `bool` in a kernel argument, not even as a
/// struct's field (clang rejects both), and IGC 2.41 hung compiling a kernel
/// that branched on one; it read a bfloat16 field of a struct argument as
/// zero (both IGC 2.41).
bool isRewritten(const Argument &arg) {
  Type *type = arg.getType();
  if (!type->isAggregateType())
    return false;
  return holdsArray(type) || holdsRetyped(type) ||
         any_of(type->subtypes(),
                [](Type *member) { return member->isAggregateType(); });
}

/// Whether kernel parameter `arg` is a scalar passed as another type
/// (`passedAs`): a `bool` or a bfloat16.
bool isRetyped(const Argument &arg) {
  return passedAs(arg.getType()) != arg.getType();
}

using Path = SmallVector<unsigned, 4>;

/// A parameter's replacement: its non-aggregate leaves in order, with the
/// zero-length arrays left out, each `bool` and bfloat16 as `passedAs`, and
/// single-byte padding where natural alignment wouldn't put a leaf at its
/// original offset. IGC reads such a struct of scalars correctly, but not a
/// struct holding a byte array.
struct Flattened {
  StructType *type = nullptr;
  /// The new struct's field for each leaf, by its index path in the
  /// original type.
  std::map<Path, unsigned> fields;
};

void collectLeaves(
    const DataLayout &layout, Type *type, uint64_t offset, Path &path,
    SmallVectorImpl<std::tuple<Path, Type *, uint64_t>> &leaves) {
  if (auto *structType = dyn_cast<StructType>(type)) {
    const StructLayout *members = layout.getStructLayout(structType);
    for (unsigned i = 0; i < structType->getNumElements(); ++i) {
      path.push_back(i);
      collectLeaves(layout, structType->getElementType(i),
                    offset + members->getElementOffset(i), path, leaves);
      path.pop_back();
    }
  } else if (auto *array = dyn_cast<ArrayType>(type)) {
    uint64_t size = layout.getTypeAllocSize(array->getElementType());
    for (unsigned i = 0; i < array->getNumElements(); ++i) {
      path.push_back(i);
      collectLeaves(layout, array->getElementType(), offset + i * size, path,
                    leaves);
      path.pop_back();
    }
  } else {
    leaves.emplace_back(path, type, offset);
  }
}

Flattened flatten(const DataLayout &layout, Type *type) {
  LLVMContext &ctx = type->getContext();
  SmallVector<std::tuple<Path, Type *, uint64_t>> leaves;
  Path path;
  collectLeaves(layout, type, 0, path, leaves);
  Flattened result;
  SmallVector<Type *> fields;
  uint64_t end = 0;
  auto pad = [&](uint64_t bytes) {
    fields.append(bytes, Type::getInt8Ty(ctx));
    end += bytes;
  };
  // A packed original can put a leaf where its alignment wouldn't, or end
  // short of it; the new struct is then packed too, placed by padding alone.
  uint64_t size = layout.getTypeAllocSize(type);
  bool packed = any_of(leaves, [&](auto &leaf) {
    Align align = layout.getABITypeAlign(std::get<1>(leaf));
    return !isAligned(align, std::get<2>(leaf)) || !isAligned(align, size);
  });
  for (auto &[leafPath, leaf, offset] : leaves) {
    // Otherwise the original placed the leaf at an offset aligned for it,
    // so padding up to that offset puts it there.
    uint64_t next = packed ? end : alignTo(end, layout.getABITypeAlign(leaf));
    if (next != offset)
      pad(offset - end);
    result.fields[leafPath] = fields.size();
    // `passedAs` keeps the size, so the offsets don't change.
    fields.push_back(passedAs(leaf));
    end = offset + layout.getTypeAllocSize(leaf);
  }
  pad(size - end);
  result.type = StructType::get(ctx, fields, packed);
  return result;
}

/// Builds the value of the original type `type` at `path` (an aggregate
/// without zero-length arrays, or a leaf) from the new parameter's fields.
Value *rebuild(IRBuilder<> &builder, Value *param, const Flattened &flat,
               Type *type, Path &path) {
  if (!type->isAggregateType()) {
    return readAs(
        builder, builder.CreateExtractValue(param, flat.fields.at(path)), type);
  }
  Value *value = PoisonValue::get(type);
  auto *structType = dyn_cast<StructType>(type);
  unsigned count = structType ? structType->getNumElements()
                              : cast<ArrayType>(type)->getNumElements();
  for (unsigned i = 0; i < count; ++i) {
    Type *member = structType ? structType->getElementType(i)
                              : cast<ArrayType>(type)->getElementType();
    path.push_back(i);
    value = builder.CreateInsertValue(
        value, rebuild(builder, param, flat, member, path), i);
    path.pop_back();
  }
  return value;
}

/// Errors unless every use of `value` reaches its fields through
/// `extractvalue`, following parts that still hold a zero-length array.
Error checkExtracts(Value *value) {
  for (User *user : value->users()) {
    auto *extract = dyn_cast<ExtractValueInst>(user);
    if (!extract)
      return createStringError(
          "intel plugin: a kernel argument holding a zero-length array is used "
          "other than through extractvalue, in " +
          cast<Instruction>(user)->getFunction()->getName());
    if (containsZeroLengthArray(extract->getType()))
      if (Error err = checkExtracts(extract))
        return err;
  }
  return Error::success();
}

/// Replaces the `extractvalue` users of `value`, the part of the original
/// parameter at `path`, with reads of the new parameter's fields. A part that
/// still holds a zero-length array is followed into its own users; nothing
/// reads a zero-length array itself, as it has no bytes.
/// `checkExtracts` must have passed.
void rewriteExtracts(Value *value, Path &path, Value *param,
                     const Flattened &flat) {
  SmallVector<User *> users(value->users());
  for (User *user : users) {
    auto *extract = cast<ExtractValueInst>(user);
    size_t depth = path.size();
    path.append(extract->idx_begin(), extract->idx_end());
    if (containsZeroLengthArray(extract->getType())) {
      rewriteExtracts(extract, path, param, flat);
    } else {
      IRBuilder<> builder(extract);
      Value *read = rebuild(builder, param, flat, extract->getType(), path);
      read->takeName(extract);
      extract->replaceAllUsesWith(read);
    }
    path.resize(depth);
    extract->eraseFromParent();
  }
}

/// Gives each kernel parameter passed in memory (`byval`) whose struct holds
/// a zero-length array the struct of its leaves (`flatten`) as its `byval`
/// type, with the original's alignment: the same bytes at the same offsets,
/// which the kernel reads through the pointer. Returns whether any changed.
bool retypeByValParameters(Module &module) {
  const DataLayout &layout = module.getDataLayout();
  LLVMContext &ctx = module.getContext();
  bool changed = false;
  for (Function &func : module) {
    if (func.getCallingConv() != CallingConv::SPIR_KERNEL ||
        func.isDeclaration())
      continue;
    for (Argument &arg : func.args()) {
      if (!arg.hasByValAttr() ||
          !containsZeroLengthArray(arg.getParamByValType()))
        continue;
      Type *original = arg.getParamByValType();
      Align align =
          arg.getParamAlign().value_or(layout.getABITypeAlign(original));
      unsigned index = arg.getArgNo();
      func.removeParamAttr(index, Attribute::ByVal);
      func.removeParamAttr(index, Attribute::Alignment);
      func.addParamAttr(index, Attribute::getWithByValType(
                                   ctx, flatten(layout, original).type));
      func.addParamAttr(index, Attribute::getWithAlignment(ctx, align));
      changed = true;
    }
  }
  return changed;
}

} // namespace

Expected<bool> rewriteArrayParameters(Module &module) {
  const DataLayout &layout = module.getDataLayout();
  bool retyped = retypeByValParameters(module);
  SmallVector<Function *> kernels;
  for (Function &func : module)
    if (func.getCallingConv() == CallingConv::SPIR_KERNEL &&
        !func.isDeclaration() && any_of(func.args(), [](const Argument &arg) {
          return isRewritten(arg) || isRetyped(arg);
        }))
      kernels.push_back(&func);

  for (Function *kernel : kernels)
    for (Argument &arg : kernel->args())
      if (isRewritten(arg) && containsZeroLengthArray(arg.getType()))
        if (Error err = checkExtracts(&arg))
          return std::move(err);

  for (Function *kernel : kernels) {
    SmallVector<Type *> params(kernel->getFunctionType()->params());
    std::map<unsigned, Flattened> flattened;
    for (Argument &arg : kernel->args())
      if (isRewritten(arg)) {
        flattened[arg.getArgNo()] = flatten(layout, arg.getType());
        params[arg.getArgNo()] = flattened[arg.getArgNo()].type;
      } else if (isRetyped(arg)) {
        params[arg.getArgNo()] = passedAs(arg.getType());
      }

    auto *type =
        FunctionType::get(kernel->getReturnType(), params, kernel->isVarArg());
    Function *copy = Function::Create(type, kernel->getLinkage(),
                                      kernel->getAddressSpace(), "", &module);
    copy->copyAttributesFrom(kernel);
    copy->copyMetadata(kernel, 0);
    // Attributes of the old aggregates (noundef and the like) don't apply.
    for (Argument &arg : kernel->args())
      if (isRewritten(arg) || isRetyped(arg))
        copy->setAttributes(copy->getAttributes().removeParamAttributes(
            module.getContext(), arg.getArgNo()));
    for (auto [from, to] : zip(kernel->args(), copy->args())) {
      to.takeName(&from);
      if (isRetyped(from)) {
        IRBuilder<> entry(&*kernel->getEntryBlock().getFirstInsertionPt());
        from.replaceAllUsesWith(readAs(entry, &to, from.getType()));
        continue;
      }
      if (!flattened.count(from.getArgNo())) {
        from.replaceAllUsesWith(&to);
        continue;
      }
      Path path;
      if (containsZeroLengthArray(from.getType())) {
        rewriteExtracts(&from, path, &to, flattened[from.getArgNo()]);
        continue;
      }
      IRBuilder<> entry(&*kernel->getEntryBlock().getFirstInsertionPt());
      from.replaceAllUsesWith(rebuild(entry, &to, flattened[from.getArgNo()],
                                      from.getType(), path));
    }
    copy->splice(copy->end(), kernel);
    copy->takeName(kernel);
    // References such as @llvm.used now name the copy.
    kernel->replaceAllUsesWith(copy);
    kernel->eraseFromParent();
  }
  return retyped || !kernels.empty();
}

} // namespace M::KGEN::Intel
