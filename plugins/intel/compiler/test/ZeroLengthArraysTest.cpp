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
// Tests for kernel arguments holding zero-length arrays, on hand-written IR
// in SPIR-V numbering, as the plugin sees it before codegen.
//
//===----------------------------------------------------------------------===//

#include "ZeroLengthArrays.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

#include "gtest/gtest.h"

#include <map>
#include <memory>
#include <string>

extern "C" void LLVMInitializeSPIRVTargetInfo();
extern "C" void LLVMInitializeSPIRVTarget();
extern "C" void LLVMInitializeSPIRVTargetMC();
extern "C" void LLVMInitializeSPIRVAsmPrinter();

using namespace llvm;
using M::KGEN::Intel::containsZeroLengthArray;
using M::KGEN::Intel::rewriteArrayParameters;

namespace {

constexpr const char *kHeader =
    "target datalayout = \"e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-"
    "v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1\"\n"
    "target triple = \"spirv64-unknown-unknown\"\n";

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

// A struct with a zero-length array inside a nested struct, whose 8-byte
// alignment places the following i32 at offset 16: { float @0, { ptr @8,
// [0 x i64] @16 }, i32 @16 }, 24 bytes.
constexpr const char *kKernel = R"(
define spir_kernel void @k(ptr addrspace(1) %out, { float, { ptr addrspace(1), [0 x i64] }, i32 } %arg) {
  %f = extractvalue { float, { ptr addrspace(1), [0 x i64] }, i32 } %arg, 0
  %inner = extractvalue { float, { ptr addrspace(1), [0 x i64] }, i32 } %arg, 1
  %p = extractvalue { ptr addrspace(1), [0 x i64] } %inner, 0
  %n = extractvalue { float, { ptr addrspace(1), [0 x i64] }, i32 } %arg, 2
  %v = load i32, ptr addrspace(1) %p, align 4
  %s = add i32 %v, %n
  store i32 %s, ptr addrspace(1) %out, align 4
  %o = getelementptr i8, ptr addrspace(1) %out, i64 4
  store float %f, ptr addrspace(1) %o, align 4
  ret void
}
)";

TEST(ZeroLengthArraysTest, ParametersBecomeStructsOfTheirLeaves) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, kKernel);
  ASSERT_TRUE(module);
  Expected<bool> changed = rewriteArrayParameters(*module);
  ASSERT_TRUE(static_cast<bool>(changed)) << toString(changed.takeError());
  EXPECT_TRUE(*changed);
  ASSERT_EQ(verify(*module), "") << print(*module);

  Function *kernel = module->getFunction("k");
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(kernel->getCallingConv(), CallingConv::SPIR_KERNEL);
  // { float, ptr, i32, i8 x 4 }: the leaves at their offsets, 24 bytes.
  auto *arg = cast<StructType>(kernel->getArg(1)->getType());
  EXPECT_FALSE(containsZeroLengthArray(arg));
  const DataLayout &layout = module->getDataLayout();
  EXPECT_EQ(layout.getTypeAllocSize(arg), 24u);
  ASSERT_EQ(arg->getNumElements(), 7u) << print(*module);
  const StructLayout *fields = layout.getStructLayout(arg);
  EXPECT_EQ(fields->getElementOffset(1), 8u);
  EXPECT_EQ(fields->getElementOffset(2), 16u);

  // Each field is read from its leaf.
  std::map<std::string, unsigned> reads;
  for (Instruction &inst : instructions(*kernel))
    if (auto *extract = dyn_cast<ExtractValueInst>(&inst)) {
      ASSERT_EQ(extract->getAggregateOperand(), kernel->getArg(1));
      reads[extract->getType()->isFloatTy()     ? "float"
            : extract->getType()->isPointerTy() ? "ptr"
                                                : "i32"] =
          extract->getIndices()[0];
    }
  EXPECT_EQ(reads["float"], 0u) << print(*module);
  EXPECT_EQ(reads["ptr"], 1u) << print(*module);
  EXPECT_EQ(reads["i32"], 2u) << print(*module);

  std::string spirv = emitSPIRV(*module);
  EXPECT_EQ(spirv.find("OpTypeRuntimeArray"), std::string::npos) << spirv;
  EXPECT_NE(spirv.find("OpEntryPoint Kernel"), std::string::npos) << spirv;
}

// The zero-length array's alignment moves the next field: { i32 @0,
// [0 x i64] @8, i32 @8 }, 16 bytes, becomes { i32, i8 x 4, i32, i8 x 4 }.
TEST(ZeroLengthArraysTest, PaddingKeepsShiftedFields) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %out, { i32, [0 x i64], i32 } %arg) {
  %b = extractvalue { i32, [0 x i64], i32 } %arg, 2
  store i32 %b, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  Expected<bool> changed = rewriteArrayParameters(*module);
  ASSERT_TRUE(static_cast<bool>(changed)) << toString(changed.takeError());
  ASSERT_EQ(verify(*module), "") << print(*module);
  Function *kernel = module->getFunction("k");
  auto *arg = cast<StructType>(kernel->getArg(1)->getType());
  const DataLayout &layout = module->getDataLayout();
  EXPECT_EQ(layout.getTypeAllocSize(arg), 16u) << print(*module);
  ASSERT_EQ(arg->getNumElements(), 10u) << print(*module);
  EXPECT_EQ(layout.getStructLayout(arg)->getElementOffset(5), 8u);
  for (Instruction &inst : instructions(*kernel))
    if (auto *extract = dyn_cast<ExtractValueInst>(&inst))
      EXPECT_EQ(extract->getIndices()[0], 5u) << print(*module);
}

TEST(ZeroLengthArraysTest, PackedStructsStayPacked) {
  // The i32 sits at offset 1, where only a packed struct can put it.
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %out, <{ i8, i32, [0 x i8], i16 }> %arg) {
  %b = extractvalue <{ i8, i32, [0 x i8], i16 }> %arg, 1
  store i32 %b, ptr addrspace(1) %out, align 4
  %c = extractvalue <{ i8, i32, [0 x i8], i16 }> %arg, 3
  store i16 %c, ptr addrspace(1) %out, align 2
  ret void
}
)");
  ASSERT_TRUE(module);
  Expected<bool> changed = rewriteArrayParameters(*module);
  ASSERT_TRUE(static_cast<bool>(changed)) << toString(changed.takeError());
  ASSERT_EQ(verify(*module), "") << print(*module);
  Function *kernel = module->getFunction("k");
  auto *arg = cast<StructType>(kernel->getArg(1)->getType());
  const DataLayout &layout = module->getDataLayout();
  EXPECT_TRUE(arg->isPacked()) << print(*module);
  EXPECT_EQ(layout.getTypeAllocSize(arg), 7u) << print(*module);
  ASSERT_EQ(arg->getNumElements(), 3u) << print(*module);
  EXPECT_EQ(layout.getStructLayout(arg)->getElementOffset(1), 1u);
  EXPECT_EQ(layout.getStructLayout(arg)->getElementOffset(2), 5u);
}

TEST(ZeroLengthArraysTest, ReferencesFollowTheNewKernel) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@llvm.used = appending global [1 x ptr] [ptr @k], section "llvm.metadata"
define spir_kernel void @k(ptr addrspace(1) %out, { i32, [0 x i64] } %arg) {
  %n = extractvalue { i32, [0 x i64] } %arg, 0
  store i32 %n, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  Expected<bool> changed = rewriteArrayParameters(*module);
  ASSERT_TRUE(static_cast<bool>(changed)) << toString(changed.takeError());
  ASSERT_EQ(verify(*module), "") << print(*module);
  auto *used = cast<ConstantArray>(
      module->getNamedGlobal("llvm.used")->getInitializer());
  EXPECT_EQ(used->getOperand(0), module->getFunction("k")) << print(*module);
}

// IGC rejects array-typed kernel arguments, and misreads structs holding
// arrays: both become structs of their leaves, rebuilt at entry for any use.
TEST(ZeroLengthArraysTest, ArraysBecomeStructsOfTheirLeaves) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
declare void @use([3 x i64])
define spir_kernel void @k(ptr addrspace(1) %out, [3 x i64] %a, { i32, [2 x float] } %s) {
  call void @use([3 x i64] %a)
  %f = extractvalue { i32, [2 x float] } %s, 1, 1
  store float %f, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  Expected<bool> changed = rewriteArrayParameters(*module);
  ASSERT_TRUE(static_cast<bool>(changed)) << toString(changed.takeError());
  EXPECT_TRUE(*changed);
  ASSERT_EQ(verify(*module), "") << print(*module);
  Function *kernel = module->getFunction("k");
  const DataLayout &layout = module->getDataLayout();
  for (unsigned i : {1u, 2u}) {
    auto *param = dyn_cast<StructType>(kernel->getArg(i)->getType());
    ASSERT_NE(param, nullptr) << print(*module);
    for (Type *field : param->elements())
      EXPECT_FALSE(field->isAggregateType()) << print(*module);
  }
  EXPECT_EQ(layout.getTypeAllocSize(kernel->getArg(1)->getType()), 24u);
  EXPECT_EQ(layout.getTypeAllocSize(kernel->getArg(2)->getType()), 12u);
  std::string spirv = emitSPIRV(*module);
  EXPECT_NE(spirv.find("OpEntryPoint Kernel"), std::string::npos) << spirv;
}

// IGC misreads nested structs passed by value (MAX's `Coord` in a struct):
// they become flat structs of their leaves too.
TEST(ZeroLengthArraysTest, NestedStructsBecomeFlat) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k({ { i64, i64 }, i64 } %box, ptr addrspace(1) %out) {
  %b = extractvalue { { i64, i64 }, i64 } %box, 0, 1
  store i64 %b, ptr addrspace(1) %out, align 8
  ret void
}
)");
  ASSERT_TRUE(module);
  Expected<bool> changed = rewriteArrayParameters(*module);
  ASSERT_TRUE(static_cast<bool>(changed)) << toString(changed.takeError());
  EXPECT_TRUE(*changed);
  ASSERT_EQ(verify(*module), "") << print(*module);
  auto *param =
      cast<StructType>(module->getFunction("k")->getArg(0)->getType());
  ASSERT_EQ(param->getNumElements(), 3u) << print(*module);
  for (Type *field : param->elements())
    EXPECT_TRUE(field->isIntegerTy(64)) << print(*module);
}

TEST(ZeroLengthArraysTest, RejectsOtherUses) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
declare void @use({ i32, [0 x i64] })
define spir_kernel void @k({ i32, [0 x i64] } %arg) {
  call void @use({ i32, [0 x i64] } %arg)
  ret void
}
)");
  ASSERT_TRUE(module);
  Expected<bool> changed = rewriteArrayParameters(*module);
  ASSERT_FALSE(static_cast<bool>(changed));
  EXPECT_NE(toString(changed.takeError()).find("other than through"),
            std::string::npos);
}

// A struct passed in memory (MAX's rowwise kernels take one whose index
// list is empty): { i64 @0, [0 x { i32, i1 }] @8, ptr @8, i32 @16 }, 24
// bytes, read through the pointer by byte offset. Its `byval` type becomes
// { i64, ptr, i32, i8 x 4 }, aligned as the original (8), so SPIR-V gets no
// runtime array.
TEST(ZeroLengthArraysTest, ByValStructsLoseTheirZeroLengthArrays) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr byval({ i64, [0 x { i32, i1 }], ptr addrspace(1), i32 }) %arg) {
  %pp = getelementptr inbounds i8, ptr %arg, i64 8
  %p = load ptr addrspace(1), ptr %pp, align 8
  %np = getelementptr inbounds i8, ptr %arg, i64 16
  %n = load i32, ptr %np, align 4
  store i32 %n, ptr addrspace(1) %p, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  Expected<bool> changed = rewriteArrayParameters(*module);
  ASSERT_TRUE(static_cast<bool>(changed)) << toString(changed.takeError());
  EXPECT_TRUE(*changed);
  ASSERT_EQ(verify(*module), "") << print(*module);

  Function *kernel = module->getFunction("k");
  ASSERT_NE(kernel, nullptr);
  Argument *arg = kernel->getArg(0);
  ASSERT_TRUE(arg->hasByValAttr());
  Type *byVal = arg->getParamByValType();
  EXPECT_FALSE(containsZeroLengthArray(byVal)) << print(*module);
  const DataLayout &layout = module->getDataLayout();
  EXPECT_EQ(layout.getTypeAllocSize(byVal), 24u);
  const StructLayout *fields = layout.getStructLayout(cast<StructType>(byVal));
  EXPECT_EQ(fields->getElementOffset(1), 8u);
  EXPECT_EQ(fields->getElementOffset(2), 16u);
  EXPECT_EQ(arg->getParamAlign().valueOrOne().value(), 8u);

  std::string spirv = emitSPIRV(*module);
  EXPECT_EQ(spirv.find("OpTypeRuntimeArray"), std::string::npos) << spirv;
  EXPECT_NE(spirv.find("OpEntryPoint Kernel"), std::string::npos) << spirv;
}

TEST(ZeroLengthArraysTest, LeavesOtherKernelsAlone) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %out, { i32, i64 } %arg) {
  %n = extractvalue { i32, i64 } %arg, 0
  store i32 %n, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  std::string before = print(*module);
  Expected<bool> changed = rewriteArrayParameters(*module);
  ASSERT_TRUE(static_cast<bool>(changed));
  EXPECT_FALSE(*changed);
  EXPECT_EQ(print(*module), before);
}

// OpenCL allows no bool in a kernel argument, not even as a struct's field,
// and IGC 2.41 hung compiling MAX's rank-3 rowwise kernels, which branch on
// `FastDiv._is_pow2` read from a flat struct argument: a bool
// parameter becomes a byte, and a struct holding one a struct with the byte
// in its place, at the same offsets; the kernel reads a nonzero byte as true.
TEST(ZeroLengthArraysTest, BoolsArePassedAsBytes) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %out, { i64, i32, i1, i8 } %s, i1 zeroext %b) {
  %n = extractvalue { i64, i32, i1, i8 } %s, 0
  %f = extractvalue { i64, i32, i1, i8 } %s, 2
  %c = and i1 %f, %b
  br i1 %c, label %then, label %done
then:
  store i64 %n, ptr addrspace(1) %out, align 8
  br label %done
done:
  ret void
}
)");
  ASSERT_TRUE(module);
  const DataLayout &layout = module->getDataLayout();
  uint64_t size =
      layout.getTypeAllocSize(module->getFunction("k")->getArg(1)->getType());
  Expected<bool> changed = rewriteArrayParameters(*module);
  ASSERT_TRUE(static_cast<bool>(changed)) << toString(changed.takeError());
  EXPECT_TRUE(*changed);
  ASSERT_EQ(verify(*module), "") << print(*module);
  Function *kernel = module->getFunction("k");
  ASSERT_NE(kernel, nullptr);
  auto *arg = dyn_cast<StructType>(kernel->getArg(1)->getType());
  ASSERT_NE(arg, nullptr) << print(*module);
  // { i64, i32, i8, i8, i8 x 2 }: the bool's byte, then the tail padding.
  ASSERT_EQ(arg->getNumElements(), 6u) << print(*module);
  EXPECT_TRUE(arg->getElementType(2)->isIntegerTy(8)) << print(*module);
  EXPECT_EQ(layout.getTypeAllocSize(arg), size);
  EXPECT_EQ(layout.getStructLayout(arg)->getElementOffset(2), 12u);
  EXPECT_TRUE(kernel->getArg(2)->getType()->isIntegerTy(8)) << print(*module);
  EXPECT_FALSE(kernel->hasParamAttribute(2, Attribute::ZExt));
  // Each byte is compared with zero, not truncated.
  unsigned compares = 0;
  for (Instruction &inst : instructions(*kernel))
    if (auto *cmp = dyn_cast<ICmpInst>(&inst))
      compares += cmp->getPredicate() == ICmpInst::ICMP_NE &&
                  cmp->getOperand(0)->getType()->isIntegerTy(8);
  EXPECT_EQ(compares, 2u) << print(*module);
  std::string spirv = emitSPIRV(*module);
  EXPECT_NE(spirv.find("OpEntryPoint Kernel"), std::string::npos) << spirv;
}

// IGC read a bfloat16 field of a struct kernel argument as zero: after
// upstream's elementwise moved off IndexList, test_powf_bf16's closure
// (two pointers and lengths, the bfloat16 exponent, then i64s) raised every
// input to the power 0. A bfloat16 parameter, or leaf of one, is
// passed as its bits and bitcast back, as PromoteBFloat16 moves bfloat16
// values.
TEST(ZeroLengthArraysTest, BFloat16sArePassedAsBits) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %out, { ptr addrspace(1), i64, bfloat, i64 } %s, bfloat %b) {
  %e = extractvalue { ptr addrspace(1), i64, bfloat, i64 } %s, 2
  store bfloat %e, ptr addrspace(1) %out, align 2
  %o = getelementptr i8, ptr addrspace(1) %out, i64 2
  store bfloat %b, ptr addrspace(1) %o, align 2
  ret void
}
)");
  ASSERT_TRUE(module);
  const DataLayout &layout = module->getDataLayout();
  uint64_t size =
      layout.getTypeAllocSize(module->getFunction("k")->getArg(1)->getType());
  Expected<bool> changed = rewriteArrayParameters(*module);
  ASSERT_TRUE(static_cast<bool>(changed)) << toString(changed.takeError());
  EXPECT_TRUE(*changed);
  ASSERT_EQ(verify(*module), "") << print(*module);
  Function *kernel = module->getFunction("k");
  ASSERT_NE(kernel, nullptr);
  auto *arg = dyn_cast<StructType>(kernel->getArg(1)->getType());
  ASSERT_NE(arg, nullptr) << print(*module);
  for (Type *field : arg->elements())
    EXPECT_FALSE(field->isBFloatTy()) << print(*module);
  ASSERT_GE(arg->getNumElements(), 3u) << print(*module);
  EXPECT_TRUE(arg->getElementType(2)->isIntegerTy(16)) << print(*module);
  EXPECT_EQ(layout.getTypeAllocSize(arg), size);
  EXPECT_EQ(layout.getStructLayout(arg)->getElementOffset(2), 16u);
  EXPECT_TRUE(kernel->getArg(2)->getType()->isIntegerTy(16)) << print(*module);
  unsigned bitcasts = 0;
  for (Instruction &inst : instructions(*kernel))
    if (auto *cast = dyn_cast<BitCastInst>(&inst))
      bitcasts +=
          cast->getSrcTy()->isIntegerTy(16) && cast->getDestTy()->isBFloatTy();
  EXPECT_EQ(bitcasts, 2u) << print(*module);
}

} // namespace
