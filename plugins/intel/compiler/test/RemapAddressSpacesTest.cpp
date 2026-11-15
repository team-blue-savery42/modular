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
// Tests for the Mojo -> SPIR-V address-space renumbering, on hand-written IR in
// Mojo numbering, checked as IR and as SPIR-V from the tree's SPIR-V backend.
//
//===----------------------------------------------------------------------===//

#include "RemapAddressSpaces.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

extern "C" void LLVMInitializeSPIRVTargetInfo();
extern "C" void LLVMInitializeSPIRVTarget();
extern "C" void LLVMInitializeSPIRVTargetMC();
extern "C" void LLVMInitializeSPIRVAsmPrinter();

using namespace llvm;
using M::KGEN::Intel::expandGlobalAddressSpaceCasts;
using M::KGEN::Intel::expandStepVectors;
using M::KGEN::Intel::flattenZeroLengthArrayGEPs;
using M::KGEN::Intel::remapAddressSpacesForSPIRV;
using M::KGEN::Intel::retypeAggregatePointers;

namespace {

constexpr const char *kHeader =
    "target datalayout = \"e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-"
    "v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1\"\n"
    "target triple = \"spirv64-unknown-unknown\"\n";

// A kernel in Mojo numbering: generic (0) parameters, global (1), shared (3)
// and constant (4) globals, a generic stack allocation, a device function, a
// shared -> generic cast and an overloaded intrinsic.
constexpr const char *kKernel = R"(
@counter = addrspace(1) global i32 0
@tile = internal addrspace(3) global [64 x float] poison
@table = internal addrspace(4) constant [2 x i32] [i32 1, i32 2]

define internal void @store(ptr %dst, i32 %v) {
  store i32 %v, ptr %dst, align 4
  ret void
}

define spir_kernel void @kernel(ptr %out, ptr %in, i32 %n) {
entry:
  %tmp = alloca [4 x i32], align 4
  %first = load i32, ptr %in, align 4
  store i32 %first, ptr %tmp, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %tmp, i64 16, i1 false)
  %shared = getelementptr [64 x float], ptr addrspace(3) @tile, i64 0, i64 1
  %generic = addrspacecast ptr addrspace(3) %shared to ptr
  store float 1.0, ptr %generic, align 4
  %c = load i32, ptr addrspace(4) @table, align 4
  call void @store(ptr %out, i32 %c)
  store i32 %n, ptr addrspace(1) @counter, align 4
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
)";

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

std::string verify(const Module &module) {
  std::string text;
  raw_string_ostream os(text);
  return verifyModule(module, &os) ? text : std::string();
}

unsigned pointerAS(Type *type) { return type->getPointerAddressSpace(); }

TEST(RemapAddressSpacesTest, RenumbersAKernel) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, kKernel);
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(remapAddressSpacesForSPIRV(*module)));
  ASSERT_EQ(verify(*module), "") << print(*module);

  // Globals: global stays 1, shared stays 3, constant 4 -> 2.
  EXPECT_EQ(module->getNamedGlobal("counter")->getAddressSpace(), 1u);
  EXPECT_EQ(module->getNamedGlobal("tile")->getAddressSpace(), 3u);
  EXPECT_EQ(module->getNamedGlobal("table")->getAddressSpace(), 2u);

  // Kernel: generic pointer parameters arrive as CrossWorkgroup (1).
  Function *kernel = module->getFunction("kernel");
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(kernel->getCallingConv(), CallingConv::SPIR_KERNEL);
  EXPECT_EQ(pointerAS(kernel->getArg(0)->getType()), 1u);
  EXPECT_EQ(pointerAS(kernel->getArg(1)->getType()), 1u);
  EXPECT_EQ(kernel->getArg(0)->getName(), "out");

  // ... and are cast to Generic (4) at the top of the entry block, which still
  // holds the stack allocation in Function (0).
  BasicBlock &entry = kernel->getEntryBlock();
  auto *cast0 = dyn_cast<AddrSpaceCastInst>(&entry.front());
  ASSERT_NE(cast0, nullptr) << print(*module);
  EXPECT_EQ(pointerAS(cast0->getDestTy()), 4u);
  bool sawAlloca = false;
  for (Instruction &inst : entry)
    if (auto *alloca = dyn_cast<AllocaInst>(&inst)) {
      sawAlloca = true;
      EXPECT_EQ(alloca->getAddressSpace(), 0u);
    }
  EXPECT_TRUE(sawAlloca);

  // Device functions take Generic pointers.
  Function *store = module->getFunction("store");
  ASSERT_NE(store, nullptr);
  EXPECT_EQ(pointerAS(store->getArg(0)->getType()), 4u);

  // The intrinsic is remangled for its new pointer types.
  EXPECT_EQ(module->getFunction("llvm.memcpy.p0.p0.i64"), nullptr);
  EXPECT_NE(module->getFunction("llvm.memcpy.p4.p4.i64"), nullptr);
}

TEST(RemapAddressSpacesTest, PointerInitializerFollowsItsGlobal) {
  // IR only: LLVM's SPIR-V backend crashes on this shape whatever the address
  // spaces.
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@table = internal addrspace(4) constant [2 x i32] [i32 1, i32 2]
@table_ref = addrspace(1) global ptr addrspace(4) @table
)");
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(remapAddressSpacesForSPIRV(*module)));
  ASSERT_EQ(verify(*module), "") << print(*module);
  GlobalVariable *ref = module->getNamedGlobal("table_ref");
  EXPECT_EQ(pointerAS(ref->getValueType()), 2u);
  EXPECT_EQ(ref->getInitializer(), module->getNamedGlobal("table"));
}

TEST(RemapAddressSpacesTest, GenericGlobalMovesToGlobalMemory) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@g = global i32 7
define i32 @read() {
  %v = load i32, ptr @g, align 4
  ret i32 %v
}
)");
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(remapAddressSpacesForSPIRV(*module)));
  ASSERT_EQ(verify(*module), "") << print(*module);
  EXPECT_EQ(module->getNamedGlobal("g")->getAddressSpace(), 1u);
  EXPECT_NE(print(*module).find("addrspacecast"), std::string::npos)
      << print(*module);
}

TEST(RemapAddressSpacesTest, SharedStackAllocationBecomesWorkgroupGlobal) {
  // Without the stdlib hook, shared memory arrives as an alloca in Mojo's
  // SHARED space, with lifetime markers.
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr %out) {
  %tile = alloca [64 x float], align 16, addrspace(3)
  call void @llvm.lifetime.start.p3(ptr addrspace(3) %tile)
  %slot = getelementptr float, ptr addrspace(3) %tile, i64 3
  store float 1.0, ptr addrspace(3) %slot, align 4
  %v = load float, ptr addrspace(3) %slot, align 4
  store float %v, ptr %out, align 4
  call void @llvm.lifetime.end.p3(ptr addrspace(3) %tile)
  ret void
}
declare void @llvm.lifetime.start.p3(ptr addrspace(3))
declare void @llvm.lifetime.end.p3(ptr addrspace(3))
)");
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(remapAddressSpacesForSPIRV(*module)));
  ASSERT_EQ(verify(*module), "") << print(*module);

  GlobalVariable *shared = nullptr;
  for (GlobalVariable &global : module->globals())
    if (global.getAddressSpace() == 3u)
      shared = &global;
  ASSERT_NE(shared, nullptr) << print(*module);
  EXPECT_TRUE(shared->hasInternalLinkage());
  EXPECT_EQ(shared->getValueType()->getArrayNumElements(), 64u);
  EXPECT_EQ(shared->getAlign(), MaybeAlign(16));
  for (Instruction &inst : instructions(*module->getFunction("k"))) {
    EXPECT_FALSE(isa<AllocaInst>(inst)) << print(*module);
    // Lifetime markers must name an alloca; the rewrite drops them.
    if (auto *intrinsic = dyn_cast<IntrinsicInst>(&inst))
      EXPECT_FALSE(intrinsic->isLifetimeStartOrEnd()) << print(*module);
  }
}

TEST(RemapAddressSpacesTest, GenericStackAllocationWithLifetimeMarkers) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr %out) {
  %tmp = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr %tmp)
  store i32 7, ptr %tmp, align 4
  %v = load i32, ptr %tmp, align 4
  store i32 %v, ptr %out, align 4
  call void @llvm.lifetime.end.p0(ptr %tmp)
  ret void
}
declare void @llvm.lifetime.start.p0(ptr)
declare void @llvm.lifetime.end.p0(ptr)
)");
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(remapAddressSpacesForSPIRV(*module)));
  ASSERT_EQ(verify(*module), "") << print(*module);
  bool sawPrivateAlloca = false;
  for (Instruction &inst : instructions(*module->getFunction("k")))
    if (auto *alloca = dyn_cast<AllocaInst>(&inst))
      sawPrivateAlloca |= alloca->getAddressSpace() == 0u;
  EXPECT_TRUE(sawPrivateAlloca) << print(*module);
}

TEST(RemapAddressSpacesTest, RejectsStackAllocationsItCannotPlace) {
  for (const char *body : {
           // Global memory can't be stack allocated.
           "define void @f() {\n  %p = alloca i32, addrspace(1)\n"
           "  ret void\n}\n",
           // Shared memory needs a fixed size.
           "define void @f(i64 %n) {\n"
           "  %p = alloca float, i64 %n, addrspace(3)\n  ret void\n}\n",
       }) {
    LLVMContext ctx;
    std::unique_ptr<Module> module = parse(ctx, body);
    ASSERT_TRUE(module);
    std::string before = print(*module);
    Error err = remapAddressSpacesForSPIRV(*module);
    ASSERT_TRUE(static_cast<bool>(err)) << body;
    std::string message = toString(std::move(err));
    EXPECT_NE(message.find("stack allocation"), std::string::npos) << message;
    EXPECT_EQ(print(*module), before) << "module changed on failure";
  }
}

TEST(RemapAddressSpacesTest, RejectsUnmappedAddressSpaces) {
  for (const char *body : {
           // NVIDIA shared-cluster memory.
           "define void @f(ptr addrspace(7) %p) { ret void }\n",
           // AMD buffer resource, only inside a function body.
           "define void @f() {\n  %p = inttoptr i64 0 to ptr addrspace(8)\n"
           "  ret void\n}\n",
           // Not a Mojo address space at all.
           "@g = addrspace(2) global i32 0\n",
       }) {
    LLVMContext ctx;
    std::unique_ptr<Module> module = parse(ctx, body);
    ASSERT_TRUE(module);
    std::string before = print(*module);
    Error err = remapAddressSpacesForSPIRV(*module);
    ASSERT_TRUE(static_cast<bool>(err)) << body;
    std::string message = toString(std::move(err));
    EXPECT_NE(message.find("no equivalent on Intel GPUs"), std::string::npos)
        << message;
    EXPECT_EQ(print(*module), before) << "module changed on failure";
  }
}

// OpenCL's printf takes its format string only from constant memory; Mojo's
// is a generic pointer to an ordinary constant global. IGC's `%s` takes only
// string literals, which must move too; other pointers stay.
constexpr const char *kPrintf = R"(
@fmt = internal constant [7 x i8] c"v=%d\0A\00\00"
@sfmt = internal constant [6 x i8] c"%s%p\0A\00"
@name = internal constant [4 x i8] c"abc\00"

define spir_kernel void @kernel(ptr %in) {
  %v = load i32, ptr %in, align 4
  %r = call i32 (ptr, ...) @printf(ptr @fmt, i32 %v)
  %s = call i32 (ptr, ...) @printf(ptr @fmt, i32 %r)
  %t = call i32 (ptr, ...) @printf(ptr @sfmt, ptr @name, ptr %in)
  ret void
}

declare i32 @printf(ptr, ...)
)";

TEST(RemapAddressSpacesTest, PrintfFormatMovesToConstantMemory) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, kPrintf);
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(remapAddressSpacesForSPIRV(*module)));
  ASSERT_EQ(verify(*module), "") << print(*module);

  Function *printf = module->getFunction("printf");
  ASSERT_NE(printf, nullptr) << print(*module);
  EXPECT_TRUE(printf->isVarArg());
  EXPECT_EQ(pointerAS(printf->getFunctionType()->getParamType(0)), 2u);
  // Calls share one constant copy per string; unused originals are gone.
  EXPECT_EQ(module->getNamedGlobal("fmt"), nullptr) << print(*module);
  EXPECT_EQ(module->getNamedGlobal("name"), nullptr) << print(*module);
  GlobalVariable *format = module->getNamedGlobal("fmt.printf");
  ASSERT_NE(format, nullptr) << print(*module);
  EXPECT_EQ(format->getAddressSpace(), 2u);
  EXPECT_EQ(format->getNumUses(), 2u);
  GlobalVariable *name = module->getNamedGlobal("name.printf");
  ASSERT_NE(name, nullptr) << print(*module);
  EXPECT_EQ(name->getAddressSpace(), 2u);
  for (User *user : printf->users()) {
    auto *call = cast<CallInst>(user);
    EXPECT_EQ(pointerAS(call->getArgOperand(0)->getType()), 2u);
    if (call->arg_size() == 3) {
      EXPECT_EQ(call->getArgOperand(1), name);
      // A runtime pointer is not a literal and stays as it was.
      EXPECT_EQ(pointerAS(call->getArgOperand(2)->getType()), 4u);
    }
  }
}

// A format built at run time (the Intel debug_assert hook's) stays where it
// is and goes to the generic-pointer printf; constant formats still move.
constexpr const char *kRuntimePrintf = R"(
@fmt = internal constant [4 x i8] c"%d\0A\00"
define spir_kernel void @kernel(ptr %fmt, i32 %v) {
  %r = call i32 (ptr, ...) @printf(ptr %fmt, i32 %v)
  %s = call i32 (ptr, ...) @printf(ptr @fmt, i32 %v)
  ret void
}
declare i32 @printf(ptr, ...)
)";

TEST(RemapAddressSpacesTest, RuntimePrintfFormatStaysGeneric) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, kRuntimePrintf);
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(remapAddressSpacesForSPIRV(*module)));
  ASSERT_EQ(verify(*module), "") << print(*module);

  Function *runtime = module->getFunction("_Z6printfPU3AS4Kcz");
  ASSERT_NE(runtime, nullptr) << print(*module);
  EXPECT_TRUE(runtime->isVarArg());
  EXPECT_EQ(pointerAS(runtime->getFunctionType()->getParamType(0)), 4u);
  ASSERT_EQ(runtime->getNumUses(), 1u);
  auto *call = cast<CallInst>(runtime->user_back());
  // The kernel's own pointer, not a copy.
  EXPECT_TRUE(isa<Argument>(call->getArgOperand(0)->stripPointerCasts()))
      << print(*module);

  Function *printf = module->getFunction("printf");
  ASSERT_NE(printf, nullptr) << print(*module);
  EXPECT_EQ(pointerAS(printf->getFunctionType()->getParamType(0)), 2u);
  EXPECT_EQ(printf->getNumUses(), 1u);
}

TEST(RemapAddressSpacesTest, RejectsPrintfFormatInGlobalMemory) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define void @f(ptr addrspace(1) %fmt) {
  %r = call i32 (ptr addrspace(1), ...) @printf(ptr addrspace(1) %fmt)
  ret void
}
declare i32 @printf(ptr addrspace(1), ...)
)");
  ASSERT_TRUE(module);
  Error err = remapAddressSpacesForSPIRV(*module);
  ASSERT_TRUE(static_cast<bool>(err));
  EXPECT_NE(toString(std::move(err)).find("generic pointer"),
            std::string::npos);
}

std::string emitSPIRV(Module &module) {
  LLVMInitializeSPIRVTargetInfo();
  LLVMInitializeSPIRVTarget();
  LLVMInitializeSPIRVTargetMC();
  LLVMInitializeSPIRVAsmPrinter();
  std::string error;
  Triple triple("spirv64-unknown-unknown");
  const Target *target = TargetRegistry::lookupTarget(triple, error);
  EXPECT_NE(target, nullptr) << error;
  if (!target)
    return "";
  std::unique_ptr<TargetMachine> machine(target->createTargetMachine(
      triple, "", "", TargetOptions(), std::nullopt));
  SmallString<0> buffer;
  raw_svector_ostream os(buffer);
  legacy::PassManager passes;
  EXPECT_FALSE(machine->addPassesToEmitFile(passes, os, nullptr,
                                            CodeGenFileType::AssemblyFile));
  passes.run(module);
  return std::string(buffer.str());
}

TEST(RemapAddressSpacesTest, SPIRVBackendLowersPrintf) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, kPrintf);
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(remapAddressSpacesForSPIRV(*module)));
  std::string spirv = emitSPIRV(*module);
  EXPECT_NE(spirv.find("printf"), std::string::npos) << spirv;
  EXPECT_NE(spirv.find("OpExtInst"), std::string::npos) << spirv;
  EXPECT_EQ(spirv.find("SPV_EXT_relaxed_printf"), std::string::npos) << spirv;
}

// The generic-pointer printf lowers to OpenCL's printf too, with the
// extension IGC needs for a format outside UniformConstant (the stdlib turns
// it on for spirv64).
TEST(RemapAddressSpacesTest, SPIRVBackendLowersRuntimeFormatPrintf) {
  static const bool parsed = [] {
    const char *argv[] = {
        "test", "--spirv-ext=+SPV_EXT_relaxed_printf_string_address_space"};
    return cl::ParseCommandLineOptions(2, argv);
  }();
  ASSERT_TRUE(parsed);
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, kRuntimePrintf);
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(remapAddressSpacesForSPIRV(*module)));
  std::string spirv = emitSPIRV(*module);
  EXPECT_NE(spirv.find("SPV_EXT_relaxed_printf_string_address_space"),
            std::string::npos)
      << spirv;
  // Both calls are OpenCL.std printf; nothing calls the mangled name.
  size_t calls = 0;
  for (size_t at = spirv.find(" printf "); at != std::string::npos;
       at = spirv.find(" printf ", at + 1))
    ++calls;
  EXPECT_EQ(calls, 2u) << spirv;
  EXPECT_EQ(spirv.find("OpFunctionCall"), std::string::npos) << spirv;
  EXPECT_EQ(spirv.find("_Z6printf"), std::string::npos) << spirv;
}

// AMD scope names, which Mojo code uses, become the SPIR-V backend's.
TEST(RemapAddressSpacesTest, RenamesAMDSyncScopes) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define void @f(ptr addrspace(1) %p) {
  %a = atomicrmw add ptr addrspace(1) %p, i32 1 syncscope("agent") monotonic
  %b = atomicrmw add ptr addrspace(1) %p, i32 1 syncscope("wavefront-one-as") monotonic
  %c = atomicrmw add ptr addrspace(1) %p, i32 1 syncscope("workgroup") monotonic
  %d = cmpxchg ptr addrspace(1) %p, i32 0, i32 1 syncscope("agent-one-as") acq_rel monotonic
  fence syncscope("agent") release
  %e = load atomic i32, ptr addrspace(1) %p syncscope("one-as") acquire, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(remapAddressSpacesForSPIRV(*module)));
  ASSERT_EQ(verify(*module), "") << print(*module);
  std::string text = print(*module);
  EXPECT_EQ(text.find("agent"), std::string::npos) << text;
  EXPECT_EQ(text.find("wavefront"), std::string::npos) << text;
  EXPECT_EQ(text.find("one-as"), std::string::npos) << text;
  // Instructions in the same order as the source.
  SmallVector<std::string> scopes;
  SmallVector<StringRef> names;
  ctx.getSyncScopeNames(names);
  for (Instruction &inst : instructions(*module->getFunction("f")))
    if (std::optional<SyncScope::ID> scope = getAtomicSyncScopeID(&inst))
      scopes.push_back(names[*scope].str());
  ASSERT_EQ(scopes.size(), 6u) << text;
  EXPECT_EQ(scopes[0], "device");
  EXPECT_EQ(scopes[1], "subgroup");
  EXPECT_EQ(scopes[2], "workgroup");
  EXPECT_EQ(scopes[3], "device");
  EXPECT_EQ(scopes[4], "device");
  EXPECT_EQ(scopes[5], ""); // "one-as" alone is the system scope
}

// A struct passed by value holding a pointer that the kernel uses as an
// array, and a pointer stored into a struct, as Mojo's closures and slices
// build them.
TEST(RemapAddressSpacesTest, SPIRVBackendMovesPointersThroughAggregates) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
declare void @use({ ptr addrspace(4), i64 })
define spir_kernel void @kernel({ float, { ptr addrspace(4) } } %arg) {
  %p = extractvalue { float, { ptr addrspace(4) } } %arg, 1, 0
  %e = getelementptr [4 x i8], ptr addrspace(4) %p, i64 1
  store i8 1, ptr addrspace(4) %e, align 1
  %s = insertvalue { ptr addrspace(4), i64 } undef, ptr addrspace(4) %e, 0
  call void @use({ ptr addrspace(4), i64 } %s)
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(retypeAggregatePointers(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  // The extracted pointer's only use is its byte GEP, and the inserted value
  // is one.
  for (Instruction &inst : instructions(*module->getFunction("kernel"))) {
    if (auto *extract = dyn_cast<ExtractValueInst>(&inst)) {
      ASSERT_TRUE(extract->hasOneUse()) << print(*module);
      auto *gep = dyn_cast<GetElementPtrInst>(extract->user_back());
      ASSERT_NE(gep, nullptr) << print(*module);
      EXPECT_TRUE(gep->getSourceElementType()->isIntegerTy(8));
    }
    if (auto *insert = dyn_cast<InsertValueInst>(&inst)) {
      auto *gep =
          dyn_cast<GetElementPtrInst>(insert->getInsertedValueOperand());
      ASSERT_NE(gep, nullptr) << print(*module);
      EXPECT_TRUE(gep->getSourceElementType()->isIntegerTy(8));
    }
  }
  std::string spirv = emitSPIRV(*module);
  EXPECT_NE(spirv.find("OpCompositeExtract"), std::string::npos) << spirv;
}

// Dynamic shared memory indexed as Mojo's StaticTuple[T, 0] does.
TEST(RemapAddressSpacesTest, ZeroLengthArrayGEPsBecomeElementGEPs) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @kernel(ptr addrspace(3) %shared, i64 %i) {
  %p = getelementptr inbounds [0 x float], ptr addrspace(3) %shared, i64 0, i64 %i
  store float 0.0, ptr addrspace(3) %p, align 4
  %q = getelementptr [0 x float], ptr addrspace(3) %shared, i64 %i
  store float 0.0, ptr addrspace(3) %q, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(flattenZeroLengthArrayGEPs(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  std::string text = print(*module);
  EXPECT_EQ(text.find("[0 x float]"), std::string::npos) << text;
  // The second GEP's only index stepped over zero bytes.
  Argument *shared = module->getFunction("kernel")->getArg(0);
  unsigned direct = 0;
  for (User *user : shared->users())
    direct += isa<StoreInst>(user);
  EXPECT_EQ(direct, 1u) << text;
  // The argument keeps its name; the GEP's goes nowhere.
  EXPECT_EQ(shared->getName(), "shared") << text;
  std::string spirv = emitSPIRV(*module);
  EXPECT_EQ(spirv.find("OpTypeRuntimeArray"), std::string::npos) << spirv;
}

TEST(RemapAddressSpacesTest, ZeroLengthArrayGEPKeepsItsGlobalsName) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@table = external addrspace(1) global [0 x float]
define spir_kernel void @kernel(i64 %i) {
  %q = getelementptr [0 x float], ptr addrspace(1) @table, i64 %i
  store float 0.0, ptr addrspace(1) %q, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(flattenZeroLengthArrayGEPs(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  // Renaming it would change the symbol.
  EXPECT_NE(module->getNamedGlobal("table"), nullptr) << print(*module);
}

TEST(RemapAddressSpacesTest, StepVectorsBecomeConstants) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define <4 x i64> @f(<4 x i64> %x) {
  %s = call <4 x i64> @llvm.stepvector.v4i64()
  %r = add <4 x i64> %x, %s
  ret <4 x i64> %r
}
declare <4 x i64> @llvm.stepvector.v4i64()
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(expandStepVectors(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  EXPECT_EQ(module->getFunction("llvm.stepvector.v4i64"), nullptr);
  EXPECT_NE(print(*module).find("<i64 0, i64 1, i64 2, i64 3>"),
            std::string::npos)
      << print(*module);
}

// A generic pointer to a string global, as Mojo's `print` builds it: a
// constant-expression cast stored and offset inside the function.
constexpr const char *kStringCast = R"(
@s = internal addrspace(1) constant [3 x i8] c"ab\00", align 16

define spir_kernel void @kernel(ptr addrspace(1) %out) {
  %p = addrspacecast ptr addrspace(1) %out to ptr addrspace(4)
  store ptr addrspace(4) addrspacecast (ptr addrspace(1) @s to ptr addrspace(4)), ptr addrspace(4) %p, align 8
  %q = getelementptr i8, ptr addrspace(4) %p, i64 8
  store ptr addrspace(4) getelementptr (i8, ptr addrspace(4) addrspacecast (ptr addrspace(1) @s to ptr addrspace(4)), i64 1), ptr addrspace(4) %q, align 8
  ret void
}
)";

TEST(RemapAddressSpacesTest, GlobalCastsBecomeInstructions) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, kStringCast);
  ASSERT_TRUE(module);
  expandGlobalAddressSpaceCasts(*module);
  ASSERT_EQ(verify(*module), "") << print(*module);
  // Each cast is an instruction, used only through a byte GEP.
  for (User *user : module->getNamedGlobal("s")->users()) {
    auto *cast = dyn_cast<AddrSpaceCastInst>(user);
    ASSERT_NE(cast, nullptr) << print(*module);
    ASSERT_TRUE(cast->hasOneUse()) << print(*module);
    auto *bytes = dyn_cast<GetElementPtrInst>(cast->user_back());
    ASSERT_NE(bytes, nullptr) << print(*module);
    EXPECT_TRUE(bytes->getSourceElementType()->isIntegerTy(8));
  }
  unsigned casts = 0;
  for (Instruction &inst : instructions(*module->getFunction("kernel")))
    casts += isa<AddrSpaceCastInst>(inst);
  EXPECT_EQ(casts, 3u) << print(*module);
}

TEST(RemapAddressSpacesTest, SPIRVBackendCastsGlobalsByInstruction) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, kStringCast);
  ASSERT_TRUE(module);
  expandGlobalAddressSpaceCasts(*module);
  std::string spirv = emitSPIRV(*module);
  EXPECT_EQ(spirv.find("OpSpecConstantOp"), std::string::npos) << spirv;
  EXPECT_NE(spirv.find("OpPtrCastToGeneric"), std::string::npos) << spirv;
}

// A string's generic pointer inserted into a {ptr, len} pair, as `print`
// builds string slices: the backend must see a byte pointer, not an array
// pointer, or `spirv-val` rejects the OpCompositeInsert.
TEST(RemapAddressSpacesTest, SPIRVBackendInsertsStringPointersAsBytes) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@s = internal addrspace(1) constant [2 x i8] c" \00", align 16
declare void @use({ ptr addrspace(4), i64 })
define spir_kernel void @kernel() {
  %v = insertvalue { ptr addrspace(4), i64 } undef, ptr addrspace(4) addrspacecast (ptr addrspace(1) @s to ptr addrspace(4)), 0
  call void @use({ ptr addrspace(4), i64 } %v)
  ret void
}
)");
  ASSERT_TRUE(module);
  expandGlobalAddressSpaceCasts(*module);
  ASSERT_EQ(verify(*module), "") << print(*module);
  std::string spirv = emitSPIRV(*module);
  size_t insert = spirv.find("OpCompositeInsert");
  ASSERT_NE(insert, std::string::npos) << spirv;
  // The inserted value is the access chain's byte pointer.
  size_t chain = spirv.find("OpPtrAccessChain");
  EXPECT_NE(chain, std::string::npos) << spirv;
  EXPECT_LT(chain, insert) << spirv;
}

// The renumbered kernel must come out of the SPIR-V backend with the storage
// classes OpenCL expects.
TEST(RemapAddressSpacesTest, SPIRVBackendSeesOpenCLStorageClasses) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, kKernel);
  ASSERT_TRUE(module);
  ASSERT_FALSE(static_cast<bool>(remapAddressSpacesForSPIRV(*module)));
  std::string spirv = emitSPIRV(*module);

  EXPECT_NE(spirv.find("OpEntryPoint Kernel"), std::string::npos) << spirv;
  EXPECT_NE(spirv.find("OpTypePointer CrossWorkgroup"), std::string::npos)
      << spirv;
  EXPECT_NE(spirv.find("OpTypePointer Generic"), std::string::npos) << spirv;
  EXPECT_NE(spirv.find("OpTypePointer Workgroup"), std::string::npos) << spirv;
  EXPECT_NE(spirv.find("OpTypePointer UniformConstant"), std::string::npos)
      << spirv;
  EXPECT_NE(spirv.find("OpTypePointer Function"), std::string::npos) << spirv;
}

} // namespace
