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

#include "NestedStructArguments.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include <algorithm>
#include <map>

using namespace llvm;

namespace M::KGEN::Intel {

namespace {

using Path = SmallVector<unsigned, 4>;

/// Whether `type` is or holds an array, at any depth.
bool holdsArray(Type *type) {
  if (isa<ArrayType>(type))
    return true;
  if (auto *structType = dyn_cast<StructType>(type))
    return any_of(structType->elements(), holdsArray);
  return false;
}

/// Whether `type` is a struct that holds another struct and no array.
bool isNestedStruct(Type *type) {
  auto *structType = dyn_cast<StructType>(type);
  return structType && !holdsArray(structType) &&
         any_of(structType->elements(),
                [](Type *member) { return member->isStructTy(); });
}

/// The index path and type of each scalar leaf of `type`, a struct of
/// structs and scalars, in order.
void collectLeaves(Type *type, Path &path, SmallVectorImpl<Path> &paths,
                   SmallVectorImpl<Type *> &types) {
  if (auto *structType = dyn_cast<StructType>(type)) {
    for (unsigned i = 0; i < structType->getNumElements(); ++i) {
      path.push_back(i);
      collectLeaves(structType->getElementType(i), path, paths, types);
      path.pop_back();
    }
    return;
  }
  paths.push_back(path);
  types.push_back(type);
}

/// Whether `type` holds a scalar, at any depth: a struct made only of empty
/// structs holds none.
bool hasLeaves(Type *type) {
  if (auto *structType = dyn_cast<StructType>(type))
    return any_of(structType->elements(), hasLeaves);
  return true;
}

/// Whether an argument or result of `type` is passed as the flat struct of
/// its leaves: a nested struct that has leaves. One without (a struct of
/// empty structs) holds nothing to misread and keeps its type: flattened to
/// `{}`, its uses would have kept the old type.
bool isFlatPassed(Type *type) {
  return isNestedStruct(type) && hasLeaves(type);
}

bool isFlattened(Function &func) {
  if (func.isDeclaration() || func.isVarArg() || !func.hasLocalLinkage() ||
      func.getCallingConv() == CallingConv::SPIR_KERNEL)
    return false;
  if (!isFlatPassed(func.getReturnType()) &&
      none_of(func.args(),
              [](Argument &arg) { return isFlatPassed(arg.getType()); }))
    return false;
  return all_of(func.uses(), [](Use &use) {
    auto *call = dyn_cast<CallInst>(use.getUser());
    return call && call->isCallee(&use);
  });
}

/// The parameter attributes for the flattened signature: each kept
/// parameter's own, none for a flattened one.
AttributeList flatAttributes(LLVMContext &ctx, AttributeList attrs,
                             ArrayRef<SmallVector<Path>> leaves) {
  SmallVector<AttributeSet> params;
  for (auto [index, argLeaves] : enumerate(leaves))
    params.push_back(argLeaves.empty() ? attrs.getParamAttrs(index)
                                       : AttributeSet());
  return AttributeList::get(ctx, attrs.getFnAttrs(), attrs.getRetAttrs(),
                            params);
}

/// `value`, of a nested struct type whose leaves are at `paths`, as the flat
/// struct `flatType` of them, built before `builder`'s insertion point.
Value *flatten(IRBuilder<> &builder, Value *value, StructType *flatType,
               ArrayRef<Path> paths) {
  Value *flat = PoisonValue::get(flatType);
  for (auto [field, leafPath] : enumerate(paths)) {
    // Folding a floating-point leaf out of a constant would create a
    // floating-point constant (a `zeroinitializer`'s zero), whose semantics
    // this LLVM copy doesn't share with the compiler's: read it with an
    // instruction instead.
    Value *leaf =
        isa<Constant>(value) &&
                flatType->getElementType(field)->isFPOrFPVectorTy()
            ? builder.Insert(ExtractValueInst::Create(value, leafPath))
            : builder.CreateExtractValue(value, leafPath);
    flat = builder.CreateInsertValue(flat, leaf, field);
  }
  return flat;
}

/// One flattened argument's leaves, read from the new parameter at the
/// function's entry.
struct Leaves {
  std::map<Path, Value *> byPath;
  /// The last of the reads: a part rebuilt from the leaves goes after it,
  /// so that it is dominated by every leaf it inserts.
  Instruction *last = nullptr;
};

/// Replaces `value`, the part at `path` of a flattened argument, with the
/// leaves read from the new parameter: each `extractvalue` of a leaf takes its
/// leaf, one of a part is followed into its own users, and any other use gets
/// the part rebuilt from its leaves at the function's entry, after the reads.
void readLeaves(Value *value, Path &path, const Leaves &leaves) {
  for (User *user : make_early_inc_range(value->users())) {
    auto *extract = dyn_cast<ExtractValueInst>(user);
    if (!extract)
      continue;
    size_t depth = path.size();
    path.append(extract->idx_begin(), extract->idx_end());
    if (auto found = leaves.byPath.find(path); found != leaves.byPath.end())
      extract->replaceAllUsesWith(found->second);
    else
      readLeaves(extract, path, leaves);
    path.resize(depth);
    if (extract->use_empty())
      extract->eraseFromParent();
  }
  if (value->use_empty())
    return;
  Value *rebuilt = PoisonValue::get(value->getType());
  if (leaves.last) {
    IRBuilder<> builder(leaves.last->getNextNode());
    for (auto &[leafPath, leaf] : leaves.byPath)
      if (leafPath.size() > path.size() &&
          std::equal(path.begin(), path.end(), leafPath.begin()))
        rebuilt = builder.CreateInsertValue(
            rebuilt, leaf, ArrayRef(leafPath).drop_front(path.size()));
  }
  value->replaceAllUsesWith(rebuilt);
}

} // namespace

bool flattenNestedStructArguments(Module &module) {
  LLVMContext &ctx = module.getContext();
  SmallVector<Function *> functions;
  for (Function &func : module)
    if (isFlattened(func))
      functions.push_back(&func);

  for (Function *func : functions) {
    // Each argument's leaf paths (none for one passed as it is).
    SmallVector<SmallVector<Path>> leaves;
    SmallVector<Type *> params;
    for (Argument &arg : func->args()) {
      SmallVector<Path> &argLeaves = leaves.emplace_back();
      if (!isFlatPassed(arg.getType())) {
        params.push_back(arg.getType());
        continue;
      }
      SmallVector<Type *> fields;
      Path path;
      collectLeaves(arg.getType(), path, argLeaves, fields);
      params.push_back(StructType::get(ctx, fields));
    }

    // The result's leaf paths, if it is returned flat.
    SmallVector<Path> resultLeaves;
    Type *result = func->getReturnType();
    if (isFlatPassed(result)) {
      SmallVector<Type *> fields;
      Path path;
      collectLeaves(result, path, resultLeaves, fields);
      result = StructType::get(ctx, fields);
    }
    auto *type = FunctionType::get(result, params, false);
    Function *copy = Function::Create(type, func->getLinkage(),
                                      func->getAddressSpace(), "", &module);
    copy->copyAttributesFrom(func);
    copy->copyMetadata(func, 0);
    copy->setAttributes(flatAttributes(ctx, func->getAttributes(), leaves));

    for (auto [from, to] : zip(func->args(), copy->args())) {
      to.takeName(&from);
      ArrayRef<Path> argLeaves = leaves[from.getArgNo()];
      if (argLeaves.empty()) {
        from.replaceAllUsesWith(&to);
        continue;
      }
      IRBuilder<> entry(&*func->getEntryBlock().getFirstInsertionPt());
      Leaves read;
      for (auto [field, leafPath] : enumerate(argLeaves)) {
        Value *leaf = entry.CreateExtractValue(&to, field);
        read.byPath[leafPath] = leaf;
        read.last = cast<Instruction>(leaf);
      }
      Path path;
      readLeaves(&from, path, read);
      for (auto &[leafPath, leaf] : read.byPath)
        if (leaf->use_empty())
          cast<Instruction>(leaf)->eraseFromParent();
    }
    copy->splice(copy->end(), func);
    if (!resultLeaves.empty())
      for (BasicBlock &block : *copy)
        if (auto *ret = dyn_cast<ReturnInst>(block.getTerminator())) {
          IRBuilder<> builder(ret);
          ret->setOperand(0, flatten(builder, ret->getReturnValue(),
                                     cast<StructType>(result), resultLeaves));
        }

    for (User *user : make_early_inc_range(func->users())) {
      auto *call = cast<CallInst>(user);
      IRBuilder<> builder(call);
      SmallVector<Value *> args;
      for (auto [index, argLeaves] : enumerate(leaves)) {
        Value *arg = call->getArgOperand(index);
        if (argLeaves.empty()) {
          args.push_back(arg);
          continue;
        }
        auto *flatType = cast<StructType>(type->getParamType(args.size()));
        args.push_back(flatten(builder, arg, flatType, argLeaves));
      }
      CallInst *replacement = builder.CreateCall(type, copy, args);
      replacement->setCallingConv(call->getCallingConv());
      replacement->setAttributes(
          flatAttributes(ctx, call->getAttributes(), leaves));
      replacement->setTailCallKind(call->getTailCallKind());
      replacement->setDebugLoc(call->getDebugLoc());
      replacement->takeName(call);
      if (resultLeaves.empty()) {
        call->replaceAllUsesWith(replacement);
        call->eraseFromParent();
        continue;
      }
      // The caller reads the result's leaves from the flat struct right
      // after the call, as a flattened function reads its argument's.
      Leaves read;
      for (auto [field, leafPath] : enumerate(resultLeaves)) {
        Value *leaf = builder.CreateExtractValue(replacement, field);
        read.byPath[leafPath] = leaf;
        read.last = cast<Instruction>(leaf);
      }
      Path path;
      readLeaves(call, path, read);
      for (auto &[leafPath, leaf] : read.byPath)
        if (leaf->use_empty())
          cast<Instruction>(leaf)->eraseFromParent();
      call->eraseFromParent();
    }
    copy->takeName(func);
    func->eraseFromParent();
  }
  return !functions.empty();
}

} // namespace M::KGEN::Intel
