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

#include "AggregatePhis.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Transforms/Utils/Local.h"

#include <algorithm>
#include <cstdint>
#include <map>

using namespace llvm;

namespace M::KGEN::Intel {

namespace {

using Path = SmallVector<unsigned, 4>;

/// More leaves than this and a phi is left whole: splitting a large array
/// would make many phis for no measured gain.
constexpr uint64_t kMaxLeaves = 64;

/// The number of scalar leaves of `type`, or kMaxLeaves + 1 if more.
uint64_t leafCount(Type *type) {
  if (auto *structType = dyn_cast<StructType>(type)) {
    uint64_t count = 0;
    for (Type *element : structType->elements()) {
      count += leafCount(element);
      if (count > kMaxLeaves)
        return kMaxLeaves + 1;
    }
    return count;
  }
  if (auto *arrayType = dyn_cast<ArrayType>(type)) {
    uint64_t each = leafCount(arrayType->getElementType());
    if (each == 0)
      return 0;
    if (arrayType->getNumElements() > kMaxLeaves)
      return kMaxLeaves + 1;
    return std::min(each * arrayType->getNumElements(), kMaxLeaves + 1);
  }
  return 1;
}

bool isSplit(Type *type) {
  if (!type->isAggregateType())
    return false;
  uint64_t count = leafCount(type);
  return count > 0 && count <= kMaxLeaves;
}

/// The index path and type of each scalar leaf of `type`, in order. Vectors
/// are leaves.
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
  if (auto *arrayType = dyn_cast<ArrayType>(type)) {
    for (unsigned i = 0; i < arrayType->getNumElements(); ++i) {
      path.push_back(i);
      collectLeaves(arrayType->getElementType(), path, paths, types);
      path.pop_back();
    }
    return;
  }
  paths.push_back(path);
  types.push_back(type);
}

struct Leaves {
  SmallVector<Path, 8> paths;
  SmallVector<Type *, 8> types;

  /// The leaf at `path`, or -1 if `path` isn't a whole leaf's.
  int find(ArrayRef<unsigned> path) const {
    for (unsigned k = 0; k < paths.size(); ++k)
      if (ArrayRef<unsigned>(paths[k]) == path)
        return static_cast<int>(k);
    return -1;
  }
};

/// The leaf at `path` of constant `constant`, of type `leafType`, when the
/// constant holds it: an element of a constant struct or array, an integer
/// of a constant data array, undef or poison inside undef or poison, a zero
/// integer, pointer or integer vector inside a zero constant. Null when
/// reading it would make a new constant of a floating-point type (a zero or
/// data-array float): the plugin never creates or folds those.
Constant *constantLeaf(Constant *constant, ArrayRef<unsigned> path,
                       Type *leafType) {
  bool isFloat = leafType->getScalarType()->isFloatingPointTy();
  for (unsigned i : path) {
    if (isa<UndefValue>(constant))
      break;
    if (isa<ConstantAggregate>(constant)) {
      constant = cast<Constant>(constant->getOperand(i));
      continue;
    }
    if (isa<ConstantAggregateZero>(constant))
      return isFloat ? nullptr : Constant::getNullValue(leafType);
    if (auto *data = dyn_cast<ConstantDataArray>(constant)) {
      if (data->getElementType()->isFloatingPointTy())
        return nullptr;
      constant = data->getElementAsConstant(i);
      continue;
    }
    return nullptr;
  }
  if (isa<PoisonValue>(constant))
    return PoisonValue::get(leafType);
  if (isa<UndefValue>(constant))
    return UndefValue::get(leafType);
  return constant->getType() == leafType ? constant : nullptr;
}

class Splitter {
public:
  explicit Splitter(ArrayRef<PHINode *> phis) : phis(phis) {}

  void run() {
    for (PHINode *phi : phis) {
      const Leaves &leaves = leavesOf(phi->getType());
      SmallVector<PHINode *, 8> &split = splitOf[phi];
      for (unsigned k = 0; k < leaves.types.size(); ++k)
        split.push_back(PHINode::Create(
            leaves.types[k], phi->getNumIncomingValues(),
            phi->getName() + "." + Twine(k), phi->getIterator()));
    }
    for (PHINode *phi : phis)
      fillIncoming(phi);
    for (PHINode *phi : phis)
      replaceUses(phi);

    SmallVector<WeakTrackingVH, 16> incoming;
    for (PHINode *phi : phis)
      for (Value *value : phi->incoming_values())
        if (isa<Instruction>(value) && !isa<PHINode>(value))
          incoming.push_back(value);
    for (PHINode *phi : phis)
      phi->dropAllReferences();
    for (PHINode *phi : phis)
      phi->eraseFromParent();
    // Deleting the chains below can delete leaf phis too (a leaf only the
    // rebuild for a non-loop insertvalue read becomes trivially dead), so the
    // leaves are held by handles that become null when that happens.
    SmallVector<WeakVH, 16> leaves;
    for (auto &[phi, split] : splitOf)
      leaves.append(split.begin(), split.end());
    splitOf.clear();
    // The insertvalue chains that built the incoming values, now unused, go
    // first: a loop's update (`insertvalue %p, ..., 0`) reads %p rebuilt from
    // all its leaf phis, which keeps the leaves it doesn't change live until
    // it is gone.
    for (WeakTrackingVH &value : incoming)
      if (auto *inst = dyn_cast_or_null<Instruction>(value))
        if (isInstructionTriviallyDead(inst))
          RecursivelyDeleteTriviallyDeadInstructions(inst);
    removeDeadLeaves(leaves);
  }

private:
  /// Erases the leaf phis nothing but other dead leaf phis reads (a loop's
  /// leaves the body never reads), and the extracts that fed only them.
  /// `leaves` are all the leaf phis made; those already deleted are null.
  void removeDeadLeaves(ArrayRef<WeakVH> leaves) {
    DenseSet<PHINode *> all, live;
    SmallVector<PHINode *, 16> work;
    for (const WeakVH &leaf : leaves)
      if (auto *phi = cast_or_null<PHINode>(leaf))
        all.insert(phi);
    for (PHINode *leaf : all)
      if (any_of(leaf->users(), [&](User *user) {
            auto *userPhi = dyn_cast<PHINode>(user);
            return !userPhi || !all.count(userPhi);
          }))
        if (live.insert(leaf).second)
          work.push_back(leaf);
    while (!work.empty()) {
      PHINode *leaf = work.pop_back_val();
      for (Value *value : leaf->incoming_values())
        if (auto *from = dyn_cast<PHINode>(value))
          if (all.count(from) && live.insert(from).second)
            work.push_back(from);
    }
    SmallVector<WeakTrackingVH, 16> feeding;
    SmallVector<PHINode *, 16> dead;
    for (PHINode *leaf : all)
      if (!live.count(leaf)) {
        dead.push_back(leaf);
        for (Value *value : leaf->incoming_values())
          if (isa<Instruction>(value) && !all.count(dyn_cast<PHINode>(value)))
            feeding.push_back(value);
      }
    for (PHINode *leaf : dead)
      leaf->dropAllReferences();
    for (PHINode *leaf : dead)
      leaf->eraseFromParent();
    for (WeakTrackingVH &value : feeding)
      if (auto *inst = dyn_cast_or_null<Instruction>(value))
        if (isInstructionTriviallyDead(inst))
          RecursivelyDeleteTriviallyDeadInstructions(inst);
  }

  const Leaves &leavesOf(Type *type) {
    auto [it, inserted] = leavesByType.try_emplace(type);
    if (inserted) {
      Path path;
      collectLeaves(type, path, it->second.paths, it->second.types);
    }
    return it->second;
  }

  /// Leaf `path` (of type `leafType`) of incoming value `value`, read at the
  /// end of its block, just before `end`.
  Value *leafOf(Value *value, ArrayRef<unsigned> path, Type *leafType,
                Instruction *end) {
    Path rest(path.begin(), path.end());
    // Only code unreachable from the entry can hold a cycle of insertvalues
    // (the verifier allows `%x = insertvalue {i64, i64} %x, i64 1, 0` there):
    // stop at a value seen before and read the leaf with an extractvalue.
    SmallPtrSet<Value *, 8> seen;
    while (!rest.empty() && seen.insert(value).second) {
      if (auto *phi = dyn_cast<PHINode>(value)) {
        auto it = splitOf.find(phi);
        if (it != splitOf.end())
          return it->second[leavesOf(phi->getType()).find(rest)];
      }
      if (auto *insert = dyn_cast<InsertValueInst>(value)) {
        ArrayRef<unsigned> at = insert->getIndices();
        size_t common =
            std::mismatch(at.begin(), at.end(), rest.begin(), rest.end())
                .first -
            at.begin();
        if (common == at.size()) {
          value = insert->getInsertedValueOperand();
          rest.erase(rest.begin(), rest.begin() + at.size());
          continue;
        }
        if (common < rest.size()) {
          value = insert->getAggregateOperand();
          continue;
        }
        break;
      }
      if (auto *constant = dyn_cast<Constant>(value))
        if (Constant *leaf = constantLeaf(constant, rest, leafType))
          return leaf;
      break;
    }
    if (rest.empty())
      return value;
    return ExtractValueInst::Create(value, rest, "", end->getIterator());
  }

  void fillIncoming(PHINode *phi) {
    const Leaves &leaves = leavesOf(phi->getType());
    SmallVector<PHINode *, 8> &split = splitOf[phi];
    for (unsigned k = 0; k < split.size(); ++k) {
      // A block can come in more than once (a switch); its entries must stay
      // one value.
      DenseMap<BasicBlock *, Value *> fromBlock;
      for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
        BasicBlock *from = phi->getIncomingBlock(i);
        Value *&value = fromBlock[from];
        if (!value)
          value = leafOf(phi->getIncomingValue(i), leaves.paths[k],
                         leaves.types[k], from->getTerminator());
        split[k]->addIncoming(value, from);
      }
    }
  }

  /// The part of `phi` at `prefix`, of type `type`, rebuilt from its leaf
  /// phis just before `before`.
  Value *rebuild(PHINode *phi, ArrayRef<unsigned> prefix, Type *type,
                 Instruction *before) {
    const Leaves &leaves = leavesOf(phi->getType());
    Value *result = PoisonValue::get(type);
    for (unsigned k = 0; k < leaves.paths.size(); ++k) {
      ArrayRef<unsigned> path = leaves.paths[k];
      if (path.size() <= prefix.size() ||
          path.take_front(prefix.size()) != prefix)
        continue;
      result = InsertValueInst::Create(result, splitOf[phi][k],
                                       path.drop_front(prefix.size()), "",
                                       before->getIterator());
    }
    return result;
  }

  void replaceUses(PHINode *phi) {
    const Leaves &leaves = leavesOf(phi->getType());
    Value *whole = nullptr;
    for (Use &use : make_early_inc_range(phi->uses())) {
      auto *user = cast<Instruction>(use.getUser());
      if (auto *userPhi = dyn_cast<PHINode>(user))
        if (splitOf.count(userPhi))
          continue; // erased; its leaves read this phi's leaves
      if (auto *extract = dyn_cast<ExtractValueInst>(user)) {
        int k = leaves.find(extract->getIndices());
        Value *part = k >= 0 ? static_cast<Value *>(splitOf[phi][k])
                             : rebuild(phi, extract->getIndices(),
                                       extract->getType(), extract);
        extract->replaceAllUsesWith(part);
        extract->eraseFromParent();
        continue;
      }
      if (!whole)
        whole = rebuild(phi, {}, phi->getType(),
                        &*phi->getParent()->getFirstInsertionPt());
      use.set(whole);
    }
  }

  ArrayRef<PHINode *> phis;
  DenseMap<PHINode *, SmallVector<PHINode *, 8>> splitOf;
  // A std::map: references to its entries stay valid as it grows.
  std::map<Type *, Leaves> leavesByType;
};

} // namespace

bool splitAggregatePhis(Module &module) {
  SmallVector<PHINode *> phis;
  for (Function &func : module)
    for (Instruction &inst : instructions(func))
      if (auto *phi = dyn_cast<PHINode>(&inst))
        if (isSplit(phi->getType()))
          phis.push_back(phi);
  if (phis.empty())
    return false;
  Splitter(phis).run();
  return true;
}

} // namespace M::KGEN::Intel
