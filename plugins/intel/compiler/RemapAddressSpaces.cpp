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

#include "RemapAddressSpaces.h"

#include "IntelAddressSpace.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <optional>
#include <string>
#include <utility>

using namespace llvm;

namespace M::KGEN::Intel {
namespace {

constexpr unsigned kMojoGeneric =
    static_cast<unsigned>(MojoAddressSpace::Generic);
constexpr unsigned kMojoLocal = static_cast<unsigned>(MojoAddressSpace::Local);
constexpr unsigned kMojoShared =
    static_cast<unsigned>(MojoAddressSpace::Shared);
constexpr unsigned kFunction =
    static_cast<unsigned>(SPIRVAddressSpace::Function);
constexpr unsigned kCrossWorkgroup =
    static_cast<unsigned>(SPIRVAddressSpace::CrossWorkgroup);
constexpr unsigned kUniformConstant =
    static_cast<unsigned>(SPIRVAddressSpace::UniformConstant);
constexpr unsigned kGeneric = static_cast<unsigned>(SPIRVAddressSpace::Generic);

/// The first address space in `type` with no SPIR-V equivalent, if any.
std::optional<unsigned> unmappedIn(Type *type, SmallPtrSetImpl<Type *> &seen) {
  if (!seen.insert(type).second)
    return std::nullopt;
  if (auto *ptr = dyn_cast<PointerType>(type)) {
    unsigned as = ptr->getAddressSpace();
    if (!toSPIRVAddressSpace(as))
      return as;
    return std::nullopt;
  }
  for (Type *sub : type->subtypes())
    if (std::optional<unsigned> as = unmappedIn(sub, seen))
      return as;
  return std::nullopt;
}

/// Errors unless `placeStackAllocations` can place `alloca`.
Error checkStackAllocation(const AllocaInst &alloca, const Twine &where) {
  unsigned as = alloca.getAddressSpace();
  if (as == kMojoGeneric || as == kMojoLocal)
    return Error::success();
  if (as == kMojoShared) {
    if (alloca.isStaticAlloca())
      return Error::success();
    return createStringError(
        "intel plugin: shared-memory stack allocation in " + where +
        " needs a constant size in the entry block");
  }
  return createStringError("intel plugin: stack allocation in address space " +
                           Twine(as) + " in " + where +
                           " is not supported on Intel GPUs");
}

/// Errors naming the first value of `module` that uses an unmapped address
/// space, or global aliases/ifuncs, which the rebuild below doesn't handle.
Error checkRemappable(Module &module) {
  if (!module.alias_empty() || !module.ifunc_empty())
    return createStringError(
        "intel plugin: global aliases and ifuncs are not supported");

  SmallPtrSet<Type *, 32> seen;
  auto check = [&](Type *type, const Twine &what) -> Error {
    if (std::optional<unsigned> as = unmappedIn(type, seen))
      return createStringError("intel plugin: address space " + Twine(*as) +
                               " in " + what +
                               " has no equivalent on Intel GPUs");
    return Error::success();
  };

  for (GlobalVariable &global : module.globals()) {
    if (Error err = check(global.getType(), "global @" + global.getName()))
      return err;
    if (Error err = check(global.getValueType(), "global @" + global.getName()))
      return err;
  }
  for (Function &func : module) {
    if (Error err =
            check(func.getFunctionType(), "function @" + func.getName()))
      return err;
    const std::string where = ("function @" + func.getName()).str();
    for (Instruction &inst : instructions(func)) {
      if (Error err = check(inst.getType(), where))
        return err;
      for (Value *operand : inst.operands())
        if (Error err = check(operand->getType(), where))
          return err;
      if (auto *alloca = dyn_cast<AllocaInst>(&inst)) {
        if (Error err = check(alloca->getAllocatedType(), where))
          return err;
        if (Error err = checkStackAllocation(*alloca, where))
          return err;
      }
      if (auto *gep = dyn_cast<GetElementPtrInst>(&inst))
        if (Error err = check(gep->getSourceElementType(), where))
          return err;
    }
  }
  return Error::success();
}

/// Maps types from Mojo to SPIR-V address spaces. With opaque pointers,
/// identified structs can't be recursive, so a struct whose elements change is
/// recreated under a new name and the rest are kept.
class AddressSpaceTypeMapper final : public ValueMapTypeRemapper {
public:
  Type *remapType(Type *type) override {
    if (auto it = cache.find(type); it != cache.end())
      return it->second;
    Type *result = remapUncached(type);
    cache[type] = result;
    return result;
  }

private:
  Type *remapUncached(Type *type) {
    LLVMContext &ctx = type->getContext();
    if (auto *ptr = dyn_cast<PointerType>(type)) {
      // INVARIANT: checkRemappable rejected modules with unmapped spaces.
      std::optional<SPIRVAddressSpace> as =
          toSPIRVAddressSpace(ptr->getAddressSpace());
      return PointerType::get(ctx, static_cast<unsigned>(*as));
    }
    if (auto *fn = dyn_cast<FunctionType>(type)) {
      SmallVector<Type *> params;
      for (Type *param : fn->params())
        params.push_back(remapType(param));
      return FunctionType::get(remapType(fn->getReturnType()), params,
                               fn->isVarArg());
    }
    if (auto *array = dyn_cast<ArrayType>(type))
      return ArrayType::get(remapType(array->getElementType()),
                            array->getNumElements());
    if (auto *vector = dyn_cast<VectorType>(type))
      return VectorType::get(remapType(vector->getElementType()),
                             vector->getElementCount());
    if (auto *st = dyn_cast<StructType>(type)) {
      if (st->isOpaque())
        return st;
      SmallVector<Type *> elements;
      bool changed = false;
      for (Type *element : st->elements()) {
        elements.push_back(remapType(element));
        changed |= elements.back() != element;
      }
      if (!changed)
        return st;
      if (st->isLiteral())
        return StructType::get(ctx, elements, st->isPacked());
      return StructType::create(ctx, elements, (st->getName() + ".spirv").str(),
                                st->isPacked());
    }
    return type;
  }

  DenseMap<Type *, Type *> cache;
};

/// Removes the lifetime markers on `alloca`. LLVM requires them to name an
/// alloca directly, which the rewritten uses no longer do; they're only hints.
void dropLifetimeMarkers(AllocaInst *alloca) {
  for (User *user : make_early_inc_range(alloca->users()))
    if (auto *intrinsic = dyn_cast<IntrinsicInst>(user))
      if (intrinsic->isLifetimeStartOrEnd())
        intrinsic->eraseFromParent();
}

/// Puts every stack allocation where SPIR-V expects it:
///  - generic ones move to Mojo's LOCAL space (SPIR-V `Function`, the alloca
///    address space) and are cast back to generic for their users;
///  - shared-memory ones become module-scope `Workgroup` globals, SPIR-V's only
///    form of shared memory.
/// `checkStackAllocation` has rejected anything else.
void placeStackAllocations(Module &module) {
  // Collect first: rewriting erases lifetime markers next to the allocations.
  SmallVector<AllocaInst *> allocas;
  for (Function &func : module)
    for (Instruction &inst : instructions(func))
      if (auto *alloca = dyn_cast<AllocaInst>(&inst))
        if (alloca->getAddressSpace() != kMojoLocal)
          allocas.push_back(alloca);

  for (AllocaInst *alloca : allocas) {
    dropLifetimeMarkers(alloca);
    if (alloca->getAddressSpace() == kMojoShared) {
      Type *type = alloca->getAllocatedType();
      if (alloca->isArrayAllocation())
        type = ArrayType::get(
            type, cast<ConstantInt>(alloca->getArraySize())->getZExtValue());
      auto *shared = new GlobalVariable(
          module, type, /*isConstant=*/false, GlobalValue::InternalLinkage,
          PoisonValue::get(type), "mojo.shared", /*InsertBefore=*/nullptr,
          GlobalValue::NotThreadLocal, kMojoShared);
      shared->setAlignment(alloca->getAlign());
      alloca->replaceAllUsesWith(shared);
      alloca->eraseFromParent();
      continue;
    }
    IRBuilder<> builder(alloca);
    AllocaInst *local = builder.CreateAlloca(
        alloca->getAllocatedType(), kMojoLocal, alloca->getArraySize());
    local->setAlignment(alloca->getAlign());
    local->takeName(alloca);
    Value *generic = builder.CreateAddrSpaceCast(local, alloca->getType());
    alloca->replaceAllUsesWith(generic);
    alloca->eraseFromParent();
  }
}

/// The parameter type `func` gets for its parameter `index` of type `param`:
/// the mapped type, except that a kernel's generic pointers arrive as
/// `CrossWorkgroup`, and its `byval` pointers (an argument passed in memory)
/// as `Function`: the kernel's private copy of the bytes the host
/// set, as clang passes an OpenCL kernel's struct argument for SPIR.
Type *newParamType(const Function &func, unsigned index, Type *param,
                   AddressSpaceTypeMapper &mapper) {
  if (func.getCallingConv() == CallingConv::SPIR_KERNEL &&
      param->isPointerTy() && param->getPointerAddressSpace() == kMojoGeneric)
    return PointerType::get(param->getContext(),
                            func.hasParamAttribute(index, Attribute::ByVal)
                                ? kFunction
                                : kCrossWorkgroup);
  return mapper.remapType(param);
}

/// Renames the AMD synchronization scopes Mojo code uses to the ones LLVM's
/// SPIR-V backend knows, which reads any other name as CrossDevice: correct,
/// but stronger and slower than asked. `agent` is the device, a `wavefront` a
/// sub-group, and the `one-as` variants (one address space) have no SPIR-V
/// form, so they get the scope without it (`one-as` alone is the system).
void renameSyncScopes(Module &module) {
  LLVMContext &ctx = module.getContext();
  SmallVector<StringRef> names;
  ctx.getSyncScopeNames(names);
  DenseMap<SyncScope::ID, SyncScope::ID> renamed;
  for (StringRef name : names) {
    StringRef base = name == "one-as" ? StringRef() : name;
    base.consume_back("-one-as");
    StringRef spirv = base == "agent"       ? "device"
                      : base == "wavefront" ? "subgroup"
                                            : base;
    if (spirv == name)
      continue;
    // The empty name is the system scope, which has its own ID.
    renamed[ctx.getOrInsertSyncScopeID(name)] =
        spirv.empty() ? SyncScope::System : ctx.getOrInsertSyncScopeID(spirv);
  }
  if (renamed.empty())
    return;
  for (Function &func : module)
    for (Instruction &inst : instructions(func))
      if (std::optional<SyncScope::ID> scope = getAtomicSyncScopeID(&inst))
        if (auto it = renamed.find(*scope); it != renamed.end())
          setAtomicSyncScopeID(&inst, it->second);
}

/// The constant C string `value` points to, if any.
GlobalVariable *constantCString(Value *value) {
  auto *global = dyn_cast<GlobalVariable>(value->stripPointerCasts());
  if (!global || !global->isConstant() || !global->hasInitializer())
    return nullptr;
  auto *data = dyn_cast<ConstantDataSequential>(global->getInitializer());
  return data && data->isCString() ? global : nullptr;
}

/// The OpenCL `printf` that takes its format through a generic pointer. The
/// SPIR-V backend recognizes `printf` by its demangled name, so this lowers
/// to the same `OpExtInst printf`, with the format in `Generic`.
constexpr const char *kRuntimeFormatPrintf = "_Z6printfPU3AS4Kcz";

/// Moves `printf` format strings into `UniformConstant`, the storage class
/// OpenCL's `printf` takes them from. Mojo passes them as generic pointers to
/// constant globals, which the rebuild above put in global memory; each call
/// gets a constant copy instead, and `printf` is redeclared to take it.
/// String-literal arguments move too: IGC formats `%s` only from a constant
/// string it can record in the module, and the driver reads any other `%s`
/// argument as a host address. A format that isn't a constant string (one
/// built at run time, as the stdlib's Intel debug_assert hook does) stays
/// where it is and goes to `kRuntimeFormatPrintf`; the SPIR-V then needs
/// SPV_EXT_relaxed_printf_string_address_space, and IGC copies the string
/// into the printf buffer, where the driver formats it.
Error placePrintfFormats(Module &module) {
  Function *printf = module.getFunction("printf");
  if (!printf)
    return Error::success();
  FunctionType *type = printf->getFunctionType();
  if (!type->isVarArg() || type->getNumParams() != 1 ||
      !type->getParamType(0)->isPointerTy())
    return createStringError(
        "intel plugin: printf is not declared as printf(ptr, ...)");

  // Check every use first, so a failure leaves the module as it was.
  SmallVector<std::pair<CallInst *, GlobalVariable *>> calls;
  for (User *user : printf->users()) {
    auto *call = dyn_cast<CallInst>(user);
    if (!call || call->getCalledOperand() != printf)
      return createStringError(
          "intel plugin: printf is used other than by a direct call");
    auto *format =
        dyn_cast<GlobalVariable>(call->getArgOperand(0)->stripPointerCasts());
    if (format && (!format->isConstant() || !format->hasInitializer()))
      format = nullptr;
    if (!format &&
        call->getArgOperand(0)->getType()->getPointerAddressSpace() != kGeneric)
      return createStringError("intel plugin: printf in " +
                               call->getFunction()->getName() +
                               " takes its format neither from a constant "
                               "string nor through a generic pointer");
    calls.emplace_back(call, format);
  }

  auto *openclType = FunctionType::get(
      type->getReturnType(),
      {PointerType::get(module.getContext(), kUniformConstant)},
      /*isVarArg=*/true);
  Function *opencl =
      Function::Create(openclType, printf->getLinkage(), "", module);
  opencl->copyAttributesFrom(printf);
  Function *runtimeFormat = nullptr;

  // One constant copy per string, shared by every call.
  DenseMap<GlobalVariable *, GlobalVariable *> constants;
  auto constantCopy = [&](GlobalVariable *global) {
    GlobalVariable *&copy = constants[global];
    if (!copy) {
      copy = new GlobalVariable(
          module, global->getValueType(), /*isConstant=*/true,
          GlobalValue::PrivateLinkage, global->getInitializer(),
          global->getName() + ".printf", /*InsertBefore=*/nullptr,
          GlobalValue::NotThreadLocal, kUniformConstant);
      copy->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      copy->setAlignment(global->getAlign());
    }
    return copy;
  };
  for (auto [call, format] : calls) {
    SmallVector<Value *> args(call->args());
    Function *callee = opencl;
    if (format) {
      args[0] = constantCopy(format);
    } else {
      if (!runtimeFormat) {
        runtimeFormat = Function::Create(
            FunctionType::get(type->getReturnType(),
                              {PointerType::get(module.getContext(), kGeneric)},
                              /*isVarArg=*/true),
            printf->getLinkage(), kRuntimeFormatPrintf, module);
        runtimeFormat->copyAttributesFrom(printf);
      }
      callee = runtimeFormat;
    }
    for (Value *&arg : drop_begin(args))
      if (arg->getType()->isPointerTy())
        if (GlobalVariable *literal = constantCString(arg))
          arg = constantCopy(literal);
    auto *replacement = CallInst::Create(callee->getFunctionType(), callee,
                                         args, "", call->getIterator());
    replacement->setCallingConv(call->getCallingConv());
    replacement->setAttributes(call->getAttributes());
    replacement->setDebugLoc(call->getDebugLoc());
    replacement->takeName(call);
    call->replaceAllUsesWith(replacement);
    call->eraseFromParent();
  }
  opencl->takeName(printf);
  printf->eraseFromParent();
  for (auto [global, copy] : constants) {
    global->removeDeadConstantUsers();
    if (global->use_empty())
      global->eraseFromParent();
  }
  return Error::success();
}

} // namespace

Error remapAddressSpacesForSPIRV(Module &module) {
  if (Error err = checkRemappable(module))
    return err;

  placeStackAllocations(module);

  AddressSpaceTypeMapper mapper;
  ValueToValueMapTy vmap;

  // Snapshot the originals: the copies are added to the same module.
  SmallVector<GlobalVariable *> oldGlobals(
      llvm::make_pointer_range(module.globals()));
  SmallVector<Function *> oldFunctions(llvm::make_pointer_range(module));

  // Initializers are mapped only once every global value has its copy: they
  // may refer to any of them.
  SmallVector<std::pair<GlobalVariable *, GlobalVariable *>> globals;
  for (GlobalVariable *oldGlobal : oldGlobals) {
    GlobalVariable &global = *oldGlobal;
    unsigned as = global.getAddressSpace();
    // Globals can't live in the generic space; give Mojo-generic ones global
    // memory and keep generic-pointer users working through a cast.
    unsigned newAS = as == kMojoGeneric
                         ? kCrossWorkgroup
                         : static_cast<unsigned>(*toSPIRVAddressSpace(as));
    auto *copy = new GlobalVariable(
        module, mapper.remapType(global.getValueType()), global.isConstant(),
        global.getLinkage(), /*Initializer=*/nullptr, "", &global,
        global.getThreadLocalMode(), newAS, global.isExternallyInitialized());
    copy->copyAttributesFrom(&global);
    Type *wanted = mapper.remapType(global.getType());
    vmap[&global] = copy->getType() == wanted
                        ? static_cast<Constant *>(copy)
                        : ConstantExpr::getAddrSpaceCast(copy, wanted);
    globals.emplace_back(&global, copy);
  }

  SmallVector<std::pair<Function *, Function *>> functions;
  for (Function *oldFunc : oldFunctions) {
    Function &func = *oldFunc;
    SmallVector<Type *> params;
    for (auto [index, param] : enumerate(func.getFunctionType()->params()))
      params.push_back(newParamType(func, index, param, mapper));
    auto *type = FunctionType::get(mapper.remapType(func.getReturnType()),
                                   params, func.isVarArg());
    Function *copy = Function::Create(type, func.getLinkage(),
                                      func.getAddressSpace(), "", &module);
    copy->copyAttributesFrom(&func);
    vmap[&func] = copy;
    functions.emplace_back(&func, copy);
  }

  for (auto [global, copy] : globals)
    if (global->hasInitializer())
      copy->setInitializer(
          MapValue(global->getInitializer(), vmap, RF_None, &mapper));

  for (auto [func, copy] : functions) {
    if (func->isDeclaration())
      continue;
    // Arguments whose type isn't the plain mapping (kernel pointers) get a
    // cast, built in a scratch block and moved into the entry block below.
    BasicBlock *scratch = nullptr;
    SmallVector<Instruction *> casts;
    for (auto [arg, newArg] : zip(func->args(), copy->args())) {
      newArg.setName(arg.getName());
      Type *mapped = mapper.remapType(arg.getType());
      if (newArg.getType() == mapped) {
        vmap[&arg] = &newArg;
        continue;
      }
      if (!scratch)
        scratch = BasicBlock::Create(module.getContext(), "", copy);
      auto *toGeneric = new AddrSpaceCastInst(
          &newArg, mapped, arg.getName() + ".generic", scratch);
      casts.push_back(toGeneric);
      vmap[&arg] = toGeneric;
    }
    SmallVector<ReturnInst *> returns;
    CloneFunctionInto(copy, func, vmap, CloneFunctionChangeType::GlobalChanges,
                      returns, "", nullptr, &mapper);
    // CloneFunctionInto keeps only the attributes of arguments it maps to
    // arguments. A kernel's `byval` argument, mapped to its cast, must keep
    // `byval` (and its alignment): the SPIR-V backend then passes the
    // argument by value (FuncParamAttr ByVal), not as a pointer to private
    // memory the host never set.
    for (Argument &arg : func->args()) {
      if (!arg.getType()->isPointerTy() || !arg.hasByValAttr() ||
          isa<Argument>(vmap[&arg]))
        continue;
      Type *byVal = arg.getParamByValType();
      AttrBuilder attrs(module.getContext());
      attrs.addByValAttr(mapper.remapType(byVal));
      if (MaybeAlign align = arg.getParamAlign())
        attrs.addAlignmentAttr(*align);
      copy->addParamAttrs(arg.getArgNo(), attrs);
    }
    if (scratch) {
      auto *entry = cast<BasicBlock>(vmap[&func->getEntryBlock()]);
      for (Instruction *toGeneric : casts)
        toGeneric->moveBefore(entry->getFirstInsertionPt());
      scratch->eraseFromParent();
    }
  }

  for (auto [func, copy] : functions) {
    copy->takeName(func);
    // `takeName`, unlike `setName`, leaves the cached intrinsic ID stale.
    copy->updateAfterNameChange();
  }
  for (auto [global, copy] : globals)
    copy->takeName(global);
  for (auto [func, copy] : functions)
    func->dropAllReferences();
  for (auto [global, copy] : globals)
    global->dropAllReferences();
  for (auto [func, copy] : functions) {
    func->removeDeadConstantUsers();
    func->eraseFromParent();
  }
  for (auto [global, copy] : globals) {
    global->removeDeadConstantUsers();
    global->eraseFromParent();
  }

  // Overloaded intrinsics carry pointer address spaces in their names.
  for (Function &func : make_early_inc_range(module)) {
    if (!func.isIntrinsic())
      continue;
    if (std::optional<Function *> remangled =
            Intrinsic::remangleIntrinsicFunction(&func)) {
      func.replaceAllUsesWith(*remangled);
      func.eraseFromParent();
    }
  }
  renameSyncScopes(module);
  return placePrintfFormats(module);
}

bool expandStepVectors(Module &module) {
  bool changed = false;
  for (Function &func : make_early_inc_range(module)) {
    if (func.getIntrinsicID() != Intrinsic::stepvector)
      continue;
    auto *type = dyn_cast<FixedVectorType>(func.getReturnType());
    if (!type)
      continue; // scalable vectors don't reach SPIR-V
    SmallVector<Constant *> steps;
    for (unsigned i = 0; i < type->getNumElements(); ++i)
      steps.push_back(ConstantInt::get(type->getElementType(), i));
    Constant *value = ConstantVector::get(steps);
    for (User *user : make_early_inc_range(func.users()))
      if (auto *call = dyn_cast<CallInst>(user)) {
        call->replaceAllUsesWith(value);
        call->eraseFromParent();
        changed = true;
      }
    if (func.use_empty())
      func.eraseFromParent();
  }
  return changed;
}

namespace {

/// `pointer` as a byte pointer: `gep i8, pointer, freeze 0`, inserted before
/// `where`.
Value *bytePointer(Value *pointer, BasicBlock::iterator where) {
  LLVMContext &ctx = pointer->getContext();
  auto *zero =
      new FreezeInst(ConstantInt::get(Type::getInt64Ty(ctx), 0), "zero", where);
  return GetElementPtrInst::Create(Type::getInt8Ty(ctx), pointer, {zero},
                                   pointer->getName() + ".bytes", where);
}

} // namespace

bool retypeAggregatePointers(Module &module) {
  SmallVector<ExtractValueInst *> extracts;
  SmallVector<InsertValueInst *> inserts;
  for (Function &func : module)
    for (Instruction &inst : instructions(func)) {
      if (auto *extract = dyn_cast<ExtractValueInst>(&inst);
          extract && extract->getType()->isPointerTy())
        extracts.push_back(extract);
      if (auto *insert = dyn_cast<InsertValueInst>(&inst);
          insert && insert->getInsertedValueOperand()->getType()->isPointerTy())
        inserts.push_back(insert);
    }
  for (ExtractValueInst *extract : extracts) {
    if (extract->use_empty())
      continue;
    Value *bytes = bytePointer(extract, std::next(extract->getIterator()));
    extract->replaceUsesWithIf(bytes, [&](Use &use) {
      return use.getUser() != bytes &&
             use.getUser() != cast<Instruction>(bytes)->getOperand(1);
    });
  }
  for (InsertValueInst *insert : inserts)
    insert->setOperand(
        InsertValueInst::getInsertedValueOperandIndex(),
        bytePointer(insert->getInsertedValueOperand(), insert->getIterator()));
  return !extracts.empty() || !inserts.empty();
}

bool flattenZeroLengthArrayGEPs(Module &module) {
  SmallVector<GetElementPtrInst *> geps;
  for (Function &func : module)
    for (Instruction &inst : instructions(func))
      if (auto *gep = dyn_cast<GetElementPtrInst>(&inst))
        if (auto *array = dyn_cast<ArrayType>(gep->getSourceElementType());
            array && array->getNumElements() == 0)
          geps.push_back(gep);
  for (GetElementPtrInst *gep : geps) {
    Value *flat = gep->getPointerOperand();
    if (gep->getNumIndices() > 1) {
      SmallVector<Value *> indices(drop_begin(gep->indices()));
      auto *inner = GetElementPtrInst::Create(
          cast<ArrayType>(gep->getSourceElementType())->getElementType(),
          gep->getPointerOperand(), indices, "", gep->getIterator());
      inner->setNoWrapFlags(gep->getNoWrapFlags());
      inner->takeName(gep);
      flat = inner;
    }
    // A single index offsets by nothing, so the pointer itself replaces the
    // GEP; it keeps its own name (a global's is its symbol).
    gep->replaceAllUsesWith(flat);
    gep->eraseFromParent();
  }
  return !geps.empty();
}

void expandGlobalAddressSpaceCasts(Module &module) {
  SmallVector<Constant *> casts;
  for (GlobalVariable &global : module.globals())
    for (User *user : global.users())
      if (auto *cast = dyn_cast<ConstantExpr>(user);
          cast && cast->getOpcode() == Instruction::AddrSpaceCast)
        casts.push_back(cast);
  // Expands the casts and every constant expression built on them (such as a
  // GEP of the generic pointer) where instructions use them.
  convertUsersOfConstantsToInstructions(casts, /*RestrictToFunc=*/nullptr,
                                        /*RemoveDeadConstants=*/true,
                                        /*IncludeSelf=*/true);
  for (GlobalVariable &global : module.globals()) {
    global.removeDeadConstantUsers();
    // The backend types a cast of an array global as a pointer to the array,
    // and doesn't reconcile that with a byte pointer where the value is
    // inserted into an aggregate (OpCompositeInsert of a string's pointer).
    // An i8 GEP makes the pointer a byte pointer; its zero offset is frozen
    // because the backend drops a constant zero GEP (IGC folds the freeze).
    if (!isa<ArrayType>(global.getValueType()))
      continue;
    for (User *user : make_early_inc_range(global.users())) {
      auto *cast = dyn_cast<AddrSpaceCastInst>(user);
      if (!cast)
        continue;
      LLVMContext &ctx = module.getContext();
      auto *zero = new FreezeInst(ConstantInt::get(Type::getInt64Ty(ctx), 0),
                                  "zero", cast->getNextNode()->getIterator());
      auto *bytes = GetElementPtrInst::Create(
          Type::getInt8Ty(ctx), cast, {zero}, global.getName() + ".bytes",
          zero->getNextNode()->getIterator());
      cast->replaceUsesWithIf(bytes,
                              [&](Use &use) { return use.getUser() != bytes; });
    }
  }
}

} // namespace M::KGEN::Intel
