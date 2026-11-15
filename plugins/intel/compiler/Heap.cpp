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

#include "Heap.h"

#include "IntelAddressSpace.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <string>

using namespace llvm;

namespace M::KGEN::Intel {

namespace {

constexpr unsigned kCrossWorkgroup =
    static_cast<unsigned>(SPIRVAddressSpace::CrossWorkgroup);
constexpr unsigned kGeneric = static_cast<unsigned>(SPIRVAddressSpace::Generic);

/// The function the module declares as `name` with `params` parameters (by
/// their kinds: 'p' a pointer, 'i' an integer) and, if `returnsPointer`, a
/// pointer result (void otherwise); null if it declares none, or one of
/// another shape, which is left for the driver plugin to report unresolved.
Function *declared(Module &module, StringRef name, StringRef params,
                   bool returnsPointer) {
  Function *func = module.getFunction(name);
  if (func == nullptr || !func->isDeclaration() || func->isVarArg())
    return nullptr;
  FunctionType *type = func->getFunctionType();
  if (type->getNumParams() != params.size())
    return nullptr;
  for (auto [kind, param] : zip(params, type->params()))
    if (kind == 'p' ? !param->isPointerTy() : !param->isIntegerTy())
      return nullptr;
  Type *result = type->getReturnType();
  if (returnsPointer ? !result->isPointerTy() : !result->isVoidTy())
    return nullptr;
  return func;
}

/// An internal function of `type` named `name`, with one entry block.
Function *create(Module &module, FunctionType *type, const Twine &name) {
  Function *func = Function::Create(
      type, GlobalValue::InternalLinkage,
      module.getDataLayout().getProgramAddressSpace(), name, &module);
  BasicBlock::Create(module.getContext(), "entry", func);
  return func;
}

/// Replaces the declaration `decl` with `definition`, which takes its uses.
void replace(Function *decl, Function *definition) {
  decl->replaceAllUsesWith(definition);
  decl->eraseFromParent();
}

/// The exported heap pointer the driver plugin sets (Heap.h).
GlobalVariable *heapGlobal(Module &module) {
  if (GlobalVariable *heap = module.getNamedGlobal(kHeapGlobal))
    return heap;
  auto *pointer = PointerType::get(module.getContext(), kCrossWorkgroup);
  // Externally initialized: the driver plugin writes it after loading the
  // module, so its null initializer must not be folded into loads.
  auto *heap = new GlobalVariable(
      module, pointer, /*isConstant=*/false, GlobalValue::ExternalLinkage,
      ConstantPointerNull::get(pointer), kHeapGlobal, /*InsertBefore=*/nullptr,
      GlobalValue::NotThreadLocal, kCrossWorkgroup,
      /*isExternallyInitialized=*/true);
  heap->setAlignment(Align(8));
  return heap;
}

/// The device-scope sync scope the heap's atomics use (SPIR-V Scope Device).
SyncScope::ID deviceScope(LLVMContext &ctx) {
  return ctx.getOrInsertSyncScopeID("device");
}

/// An atomic (device-scope, monotonic) i64 load from `at`.
Value *atomicLoad(IRBuilder<> &builder, Value *at, const Twine &name) {
  LoadInst *load =
      builder.CreateAlignedLoad(builder.getInt64Ty(), at, Align(8), name);
  load->setAtomic(AtomicOrdering::Monotonic, deviceScope(builder.getContext()));
  return load;
}

/// An atomic (device-scope, monotonic) i64 store of `value` to `at`.
void atomicStore(IRBuilder<> &builder, Value *value, Value *at) {
  StoreInst *store = builder.CreateAlignedStore(value, at, Align(8));
  store->setAtomic(AtomicOrdering::Monotonic,
                   deviceScope(builder.getContext()));
}

/// log2 of the block that holds `size` bytes after its header: the next
/// power of two of size + 16, at least `kHeapMinClassLog2`. `size` must be
/// at most the heap's capacity (< 2^62), so the sum can't wrap.
Value *classLog2(IRBuilder<> &builder, Value *size) {
  Value *bytes =
      builder.CreateAdd(size, builder.getInt64(kHeapBlockHeaderBytes));
  Value *leading = builder.CreateBinaryIntrinsic(
      Intrinsic::ctlz, builder.CreateSub(bytes, builder.getInt64(1)),
      builder.getFalse());
  Value *log2 = builder.CreateSub(builder.getInt64(64), leading);
  return builder.CreateBinaryIntrinsic(Intrinsic::umax, log2,
                                       builder.getInt64(kHeapMinClassLog2),
                                       nullptr, "class.log2");
}

/// The free-list head of class `log2` in the heap at `base`.
Value *freeListHead(IRBuilder<> &builder, Value *base, Value *log2) {
  Value *index = builder.CreateSub(log2, builder.getInt64(kHeapMinClassLog2));
  Value *offset =
      builder.CreateAdd(builder.getInt64(kHeapFreeListsOffset),
                        builder.CreateShl(index, builder.getInt64(3)));
  return builder.CreateInBoundsGEP(builder.getInt8Ty(), base, offset,
                                   "list.head");
}

/// A head whose change count is one more than `old`'s and whose first block
/// is `link` (an i64 holding a 32-bit link).
Value *nextHead(IRBuilder<> &builder, Value *old, Value *link) {
  Value *count =
      builder.CreateAdd(builder.CreateLShr(old, 32), builder.getInt64(1));
  return builder.CreateOr(builder.CreateShl(count, 32),
                          builder.CreateAnd(link, builder.getInt64(0xffffffff)),
                          "head.new");
}

/// Defines malloc(size) as `type`:
///
///   heap = __mojo_intel_heap
///   if (heap == null || size > capacity) return null
///   log2 = class of size + 16; bytes = 1 << log2
///   if (bytes > capacity) return null
///   if (log2 <= max linked class):            // reuse a freed block
///     old = head[log2]
///     while (old & 0xffffffff):
///       block = first + ((old & 0xffffffff) - 1) * 32
///       new = (old >> 32) + 1 << 32 | block->link
///       (old, ok) = cmpxchg(&head[log2], old, new)  // device, acquire
///       if (ok) goto found
///   used = 0
///   loop:                                     // or take new bytes
///     if (used + bytes > capacity) return null
///     (seen, ok) = cmpxchg(&heap->used, used, used + bytes)  // device scope
///     if (!ok) { used = seen; goto loop }
///   block = first + used
///   found: block->size = size; return block + 16
///
/// A request that doesn't fit in what is left never moves `used`, so the
/// rest of the heap stays for smaller ones (an atomic add followed by a
/// check would keep the bytes of every failed request). The first exchange
/// guesses an empty heap and so also reads `used`. `size > capacity` first
/// keeps `size + 16` from wrapping; `used` never passes `capacity`, which the
/// driver plugin keeps under 2^62 (Module.cpp), so `used + bytes` can't wrap.
/// A popped block's link may be read after another work-item took the block
/// and wrote into it; its exchange then fails, since the head's count moved.
Function *defineMalloc(Module &module, FunctionType *type) {
  LLVMContext &ctx = module.getContext();
  Function *func = create(module, type, "__mojo_intel_malloc");
  Type *i8 = Type::getInt8Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  auto *heapPointer = PointerType::get(ctx, kCrossWorkgroup);
  auto *result = cast<PointerType>(type->getReturnType());
  GlobalVariable *heap = heapGlobal(module);

  BasicBlock *entry = &func->getEntryBlock();
  BasicBlock *check = BasicBlock::Create(ctx, "check", func);
  BasicBlock *classes = BasicBlock::Create(ctx, "class", func);
  BasicBlock *reuse = BasicBlock::Create(ctx, "reuse", func);
  BasicBlock *pop = BasicBlock::Create(ctx, "pop", func);
  BasicBlock *popExchange = BasicBlock::Create(ctx, "pop.exchange", func);
  BasicBlock *bump = BasicBlock::Create(ctx, "bump", func);
  BasicBlock *loop = BasicBlock::Create(ctx, "loop", func);
  BasicBlock *exchange = BasicBlock::Create(ctx, "exchange", func);
  BasicBlock *bumped = BasicBlock::Create(ctx, "bumped", func);
  BasicBlock *found = BasicBlock::Create(ctx, "found", func);
  BasicBlock *none = BasicBlock::Create(ctx, "none", func);

  IRBuilder<> builder(entry);
  Value *size = builder.CreateZExtOrTrunc(func->getArg(0), i64, "size");
  LoadInst *base =
      builder.CreateAlignedLoad(heapPointer, heap, Align(8), "heap");
  builder.CreateCondBr(builder.CreateIsNull(base), none, check);

  builder.SetInsertPoint(check);
  Value *capacityAt =
      builder.CreateConstInBoundsGEP1_64(i8, base, 8, "capacity.at");
  LoadInst *capacity =
      builder.CreateAlignedLoad(i64, capacityAt, Align(8), "capacity");
  Value *first =
      builder.CreateConstInBoundsGEP1_64(i8, base, kHeapHeaderBytes, "first");
  builder.CreateCondBr(builder.CreateICmpUGT(size, capacity), none, classes);

  builder.SetInsertPoint(classes);
  Value *log2 = classLog2(builder, size);
  Value *bytes = builder.CreateShl(builder.getInt64(1), log2, "bytes");
  Value *headAt = freeListHead(builder, base, log2);
  Value *linked =
      builder.CreateICmpULE(log2, builder.getInt64(kHeapMaxClassLog2));
  builder.CreateCondBr(builder.CreateICmpUGT(bytes, capacity), none, reuse);

  // A class too large to be linked takes new bytes.
  builder.SetInsertPoint(reuse);
  BasicBlock *readHead = BasicBlock::Create(ctx, "head", func, pop);
  builder.CreateCondBr(linked, readHead, bump);

  builder.SetInsertPoint(readHead);
  Value *head = atomicLoad(builder, headAt, "head");
  builder.CreateBr(pop);

  builder.SetInsertPoint(pop);
  PHINode *old = builder.CreatePHI(i64, 2, "old");
  old->addIncoming(head, readHead);
  Value *units = builder.CreateAnd(old, builder.getInt64(0xffffffff), "units");
  builder.CreateCondBr(builder.CreateICmpEQ(units, builder.getInt64(0)), bump,
                       popExchange);

  builder.SetInsertPoint(popExchange);
  Value *popOffset = builder.CreateShl(
      builder.CreateSub(units, builder.getInt64(1)), builder.getInt64(5));
  Value *popped = builder.CreateInBoundsGEP(i8, first, popOffset, "popped");
  Value *link = atomicLoad(builder, popped, "link");
  Value *popPair = builder.CreateAtomicCmpXchg(
      headAt, old, nextHead(builder, old, link), Align(8),
      AtomicOrdering::Acquire, AtomicOrdering::Monotonic, deviceScope(ctx));
  old->addIncoming(builder.CreateExtractValue(popPair, 0, "head.seen"),
                   popExchange);
  builder.CreateCondBr(builder.CreateExtractValue(popPair, 1, "taken"), found,
                       pop);

  builder.SetInsertPoint(bump);
  builder.CreateBr(loop);

  builder.SetInsertPoint(loop);
  PHINode *start = builder.CreatePHI(i64, 2, "start");
  start->addIncoming(builder.getInt64(0), bump);
  Value *end = builder.CreateAdd(start, bytes, "end");
  builder.CreateCondBr(builder.CreateICmpUGT(end, capacity), none, exchange);

  builder.SetInsertPoint(exchange);
  Value *pair = builder.CreateAtomicCmpXchg(
      base, start, end, Align(8), AtomicOrdering::Monotonic,
      AtomicOrdering::Monotonic, deviceScope(ctx));
  Value *seen = builder.CreateExtractValue(pair, 0, "seen");
  start->addIncoming(seen, exchange);
  builder.CreateCondBr(builder.CreateExtractValue(pair, 1, "reserved"), bumped,
                       loop);

  builder.SetInsertPoint(bumped);
  Value *fresh = builder.CreateInBoundsGEP(i8, first, start, "fresh");
  builder.CreateBr(found);

  builder.SetInsertPoint(found);
  PHINode *block = builder.CreatePHI(popped->getType(), 2, "block");
  block->addIncoming(popped, popExchange);
  block->addIncoming(fresh, bumped);
  builder.CreateAlignedStore(size, block, Align(kHeapBlockHeaderBytes));
  Value *memory = builder.CreateConstInBoundsGEP1_64(
      i8, block, kHeapBlockHeaderBytes, "memory");
  builder.CreateRet(
      builder.CreatePointerBitCastOrAddrSpaceCast(memory, result));

  builder.SetInsertPoint(none);
  builder.CreateRet(ConstantPointerNull::get(result));
  return func;
}

/// Defines free(pointer) as `type`:
///
///   if (pointer == null || heap == null) return
///   block = pointer - 16; offset = block - first
///   if (offset >= capacity || offset >= max linked offset) return
///   log2 = class of block->size
///   if (log2 > max linked class) return
///   old = head[log2]
///   do:
///     block->link = old & 0xffffffff
///     new = (old >> 32) + 1 << 32 | offset / 32 + 1
///     (old, ok) = cmpxchg(&head[log2], old, new)  // device scope, release
///   while (!ok)
///
/// A pointer that isn't a heap block's is left alone (as a double free
/// isn't caught: C's free leaves both undefined).
Function *defineFree(Module &module, FunctionType *type) {
  LLVMContext &ctx = module.getContext();
  Function *func = create(module, type, "__mojo_intel_free");
  Type *i8 = Type::getInt8Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  auto *heapPointer = PointerType::get(ctx, kCrossWorkgroup);
  GlobalVariable *heap = heapGlobal(module);

  BasicBlock *entry = &func->getEntryBlock();
  BasicBlock *check = BasicBlock::Create(ctx, "check", func);
  BasicBlock *inHeap = BasicBlock::Create(ctx, "in.heap", func);
  BasicBlock *linkable = BasicBlock::Create(ctx, "linkable", func);
  BasicBlock *push = BasicBlock::Create(ctx, "push", func);
  BasicBlock *done = BasicBlock::Create(ctx, "done", func);

  IRBuilder<> builder(entry);
  Value *pointer = func->getArg(0);
  LoadInst *base =
      builder.CreateAlignedLoad(heapPointer, heap, Align(8), "heap");
  builder.CreateCondBr(builder.CreateOr(builder.CreateIsNull(pointer),
                                        builder.CreateIsNull(base)),
                       done, check);

  builder.SetInsertPoint(check);
  Value *global = builder.CreatePointerBitCastOrAddrSpaceCast(
      pointer, heapPointer, "global");
  Value *block = builder.CreateConstInBoundsGEP1_64(
      i8, global, -static_cast<int64_t>(kHeapBlockHeaderBytes), "block");
  Value *first =
      builder.CreateConstInBoundsGEP1_64(i8, base, kHeapHeaderBytes, "first");
  Value *offset =
      builder.CreateSub(builder.CreatePtrToInt(block, i64),
                        builder.CreatePtrToInt(first, i64), "offset");
  Value *capacityAt =
      builder.CreateConstInBoundsGEP1_64(i8, base, 8, "capacity.at");
  LoadInst *capacity =
      builder.CreateAlignedLoad(i64, capacityAt, Align(8), "capacity");
  Value *limit = builder.CreateBinaryIntrinsic(
      Intrinsic::umin, capacity, builder.getInt64(kHeapMaxLinkedOffset));
  builder.CreateCondBr(builder.CreateICmpUGE(offset, limit), done, inHeap);

  builder.SetInsertPoint(inHeap);
  Value *size = builder.CreateAlignedLoad(i64, block,
                                          Align(kHeapBlockHeaderBytes), "size");
  // A size past the capacity isn't a block's (and would wrap the class).
  Value *log2 = builder.CreateSelect(
      builder.CreateICmpUGT(size, capacity),
      builder.getInt64(kHeapMaxClassLog2 + 1),
      classLog2(builder,
                builder.CreateBinaryIntrinsic(Intrinsic::umin, size, capacity)),
      "log2");
  builder.CreateCondBr(
      builder.CreateICmpUGT(log2, builder.getInt64(kHeapMaxClassLog2)), done,
      linkable);

  builder.SetInsertPoint(linkable);
  Value *headAt = freeListHead(builder, base, log2);
  Value *units = builder.CreateAdd(builder.CreateLShr(offset, 5),
                                   builder.getInt64(1), "units");
  Value *head = atomicLoad(builder, headAt, "head");
  builder.CreateBr(push);

  builder.SetInsertPoint(push);
  PHINode *old = builder.CreatePHI(i64, 2, "old");
  old->addIncoming(head, linkable);
  atomicStore(builder, builder.CreateAnd(old, builder.getInt64(0xffffffff)),
              block);
  Value *pair = builder.CreateAtomicCmpXchg(
      headAt, old, nextHead(builder, old, units), Align(8),
      AtomicOrdering::Release, AtomicOrdering::Monotonic, deviceScope(ctx));
  old->addIncoming(builder.CreateExtractValue(pair, 0, "head.seen"), push);
  builder.CreateCondBr(builder.CreateExtractValue(pair, 1, "pushed"), done,
                       push);

  builder.SetInsertPoint(done);
  builder.CreateRetVoid();
  return func;
}

/// Defines realloc(pointer, size) as `type`, with `malloc` and `free`, as
/// C's: a new block holding the first min(old size, size) bytes of the old
/// one, which is freed; null, with the old block kept, if the heap is used
/// up; malloc(size) for a null pointer.
Function *defineRealloc(Module &module, FunctionType *type, Function *malloc,
                        Function *free) {
  LLVMContext &ctx = module.getContext();
  Function *func = create(module, type, "__mojo_intel_realloc");
  Type *i8 = Type::getInt8Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  auto *generic = PointerType::get(ctx, kGeneric);
  auto *result = cast<PointerType>(type->getReturnType());

  BasicBlock *entry = &func->getEntryBlock();
  BasicBlock *copy = BasicBlock::Create(ctx, "copy", func);
  BasicBlock *loop = BasicBlock::Create(ctx, "loop", func);
  BasicBlock *release = BasicBlock::Create(ctx, "release", func);
  BasicBlock *done = BasicBlock::Create(ctx, "done", func);

  IRBuilder<> builder(entry);
  Value *old = builder.CreatePointerBitCastOrAddrSpaceCast(func->getArg(0),
                                                           generic, "old");
  Value *size = builder.CreateZExtOrTrunc(func->getArg(1), i64, "size");
  FunctionType *mallocType = malloc->getFunctionType();
  Value *fresh = builder.CreateCall(
      malloc, builder.CreateZExtOrTrunc(size, mallocType->getParamType(0)),
      "fresh");
  Value *freshGeneric =
      builder.CreatePointerBitCastOrAddrSpaceCast(fresh, generic, "new");
  builder.CreateCondBr(builder.CreateOr(builder.CreateIsNull(old),
                                        builder.CreateIsNull(freshGeneric)),
                       done, copy);

  builder.SetInsertPoint(copy);
  Value *header = builder.CreateConstInBoundsGEP1_64(
      i8, old, -static_cast<int64_t>(kHeapBlockHeaderBytes), "header");
  Value *oldSize = builder.CreateAlignedLoad(i64, header, Align(8), "old.size");
  Value *count = builder.CreateSelect(builder.CreateICmpULT(oldSize, size),
                                      oldSize, size, "count");
  builder.CreateCondBr(builder.CreateICmpEQ(count, builder.getInt64(0)),
                       release, loop);

  // One byte a step: realloc is rare on a GPU, and the blocks are small.
  builder.SetInsertPoint(loop);
  PHINode *index = builder.CreatePHI(i64, 2, "index");
  index->addIncoming(builder.getInt64(0), copy);
  Value *byte =
      builder.CreateLoad(i8, builder.CreateInBoundsGEP(i8, old, index), "byte");
  builder.CreateStore(byte, builder.CreateInBoundsGEP(i8, freshGeneric, index));
  Value *next = builder.CreateAdd(index, builder.getInt64(1), "next");
  index->addIncoming(next, loop);
  builder.CreateCondBr(builder.CreateICmpEQ(next, count), release, loop);

  builder.SetInsertPoint(release);
  builder.CreateCall(
      free, builder.CreatePointerBitCastOrAddrSpaceCast(
                func->getArg(0), free->getFunctionType()->getParamType(0)));
  builder.CreateBr(done);

  builder.SetInsertPoint(done);
  builder.CreateRet(builder.CreatePointerBitCastOrAddrSpaceCast(fresh, result));
  return func;
}

} // namespace

bool defineHeapFunctions(Module &module) {
  Function *malloc = declared(module, "malloc", "i", /*returnsPointer=*/true);
  Function *free = declared(module, "free", "p", /*returnsPointer=*/false);
  Function *realloc =
      declared(module, "realloc", "pi", /*returnsPointer=*/true);
  if (malloc == nullptr && free == nullptr && realloc == nullptr)
    return false;

  LLVMContext &ctx = module.getContext();
  Function *mallocDefinition = nullptr;
  if (malloc != nullptr) {
    mallocDefinition = defineMalloc(module, malloc->getFunctionType());
    replace(malloc, mallocDefinition);
  } else if (realloc != nullptr) {
    auto *generic = PointerType::get(ctx, kGeneric);
    mallocDefinition = defineMalloc(
        module, FunctionType::get(generic, {Type::getInt64Ty(ctx)}, false));
  }
  Function *freeDefinition = nullptr;
  if (free != nullptr) {
    freeDefinition = defineFree(module, free->getFunctionType());
    replace(free, freeDefinition);
  } else if (realloc != nullptr) {
    freeDefinition = defineFree(
        module, FunctionType::get(Type::getVoidTy(ctx),
                                  {PointerType::get(ctx, kGeneric)}, false));
  }
  if (realloc != nullptr)
    replace(realloc, defineRealloc(module, realloc->getFunctionType(),
                                   mallocDefinition, freeDefinition));
  return true;
}

} // namespace M::KGEN::Intel
