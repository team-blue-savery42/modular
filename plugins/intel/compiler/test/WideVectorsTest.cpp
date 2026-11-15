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
// Tests for splitting vectors wider than SPIR-V allows, on hand-written IR in
// SPIR-V numbering, as the plugin sees it before codegen.
//
//===----------------------------------------------------------------------===//

#include "WideVectors.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

using namespace llvm;
using M::KGEN::Intel::kMaxVectorElements;
using M::KGEN::Intel::splitWideVectors;

namespace {

std::unique_ptr<Module> parse(LLVMContext &ctx, const std::string &body) {
  SMDiagnostic diag;
  std::unique_ptr<Module> module = parseAssemblyString(
      "target triple = \"spirv64-unknown-unknown\"\n" + body, diag, ctx);
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

bool wide(Type *type) {
  auto *vector = dyn_cast<FixedVectorType>(type);
  return vector && vector->getNumElements() > kMaxVectorElements;
}

// A 32-wide conversion between element sizes, a float constant vector, a
// zero vector, and a constant vector with an undef element.
constexpr const char *kKernel = R"(
define spir_kernel void @k(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %v = load <32 x float>, ptr addrspace(1) %in, align 4
  %a = fadd <32 x float> %v, <float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0>
  %b = call <32 x float> @llvm.maxnum.v32f32(<32 x float> %a, <32 x float> zeroinitializer)
  %c = fmul <32 x float> %b, <float undef, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0>
  %t = call <32 x float> @twice(<32 x float> %c)
  %h = fptrunc <32 x float> %t to <32 x half>
  store <32 x half> %h, ptr addrspace(1) %out, align 2
  ret void
}
declare <32 x float> @llvm.maxnum.v32f32(<32 x float>, <32 x float>)

define internal nofpclass(nan) <32 x float> @twice(<32 x float> noundef %x) {
  %y = fadd <32 x float> %x, %x
  ret <32 x float> %y
}

define spir_kernel void @narrow(ptr addrspace(1) %p) {
  %v = load <4 x float>, ptr addrspace(1) %p, align 16
  %a = fadd <4 x float> %v, <float 1.0, float 1.0, float 1.0, float 1.0>
  store <4 x float> %a, ptr addrspace(1) %p, align 16
  ret void
}
)";

TEST(WideVectorsTest, SplitsWithoutMakingFloatConstants) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, kKernel);
  ASSERT_TRUE(module);
  EXPECT_TRUE(splitWideVectors(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);

  // A wide vector passed between functions is an array of its elements.
  Function *twice = module->getFunction("twice");
  ASSERT_NE(twice, nullptr) << print(*module);
  EXPECT_TRUE(twice->getReturnType()->isArrayTy());
  EXPECT_TRUE(twice->getArg(0)->getType()->isArrayTy());
  for (Instruction &inst : instructions(*twice))
    EXPECT_FALSE(wide(inst.getType())) << print(*module);

  for (Instruction &inst : instructions(*module->getFunction("k"))) {
    EXPECT_FALSE(wide(inst.getType())) << print(*module);
    for (Value *operand : inst.operands()) {
      EXPECT_FALSE(wide(operand->getType())) << print(*module);
      // No float constant may come from the plugin: the constants are loaded
      // from globals the host's constants initialize.
      EXPECT_FALSE(isa<ConstantFP>(operand)) << print(*module);
      EXPECT_FALSE(isa<ConstantDataVector>(operand) &&
                   operand->getType()->getScalarType()->isFloatingPointTy())
          << print(*module);
    }
  }

  // Three constants, three integer-array globals in constant memory with
  // the same bits.
  unsigned globals = 0;
  for (GlobalVariable &global : module->globals()) {
    if (global.getAddressSpace() != 2 || !global.isConstant())
      continue;
    ++globals;
    auto *array = dyn_cast<ArrayType>(global.getValueType());
    ASSERT_NE(array, nullptr) << print(*module);
    EXPECT_TRUE(array->getElementType()->isIntegerTy(32));
  }
  EXPECT_EQ(globals, 3u) << print(*module);
  // The 1.0, ... constant: 1.0f is 0x3F800000; the undef-holding one: 2.0f.
  std::string text = print(*module);
  EXPECT_NE(text.find("i32 1065353216"), std::string::npos) << text;
  EXPECT_NE(text.find("i32 1073741824"), std::string::npos) << text;

  // A function without wide vectors is left as it was.
  bool narrowIntact = false;
  for (Instruction &inst : instructions(*module->getFunction("narrow")))
    narrowIntact |=
        isa<FixedVectorType>(inst.getType()) &&
        cast<FixedVectorType>(inst.getType())->getNumElements() == 4;
  EXPECT_TRUE(narrowIntact) << print(*module);
}

// A splat, which this LLVM represents as a vector-typed ConstantFP.
TEST(WideVectorsTest, SplatsBecomeTheirBits) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %p) {
  %v = load <32 x float>, ptr addrspace(1) %p, align 4
  %a = fmul <32 x float> %v, splat (float 0.5)
  store <32 x float> %a, ptr addrspace(1) %p, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(splitWideVectors(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  std::string text = print(*module);
  // 0.5f is 0x3F000000.
  EXPECT_NE(text.find("i32 1056964608"), std::string::npos) << text;
  for (Instruction &inst : instructions(*module->getFunction("k")))
    for (Value *operand : inst.operands())
      EXPECT_FALSE(isa<ConstantFP>(operand)) << text;
}

// A kernel taking a wide vector by value gets a struct of its elements,
// with the vector's size and offsets.
TEST(WideVectorsTest, KernelParametersBecomeStructsOfElements) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %out, <32 x i8> %v) {
  %w = add <32 x i8> %v, %v
  store <32 x i8> %w, ptr addrspace(1) %out, align 1
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(splitWideVectors(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  Function *kernel = module->getFunction("k");
  EXPECT_EQ(kernel->getCallingConv(), CallingConv::SPIR_KERNEL);
  auto *arg = dyn_cast<StructType>(kernel->getArg(1)->getType());
  ASSERT_NE(arg, nullptr) << print(*module);
  EXPECT_EQ(arg->getNumElements(), 32u);
  EXPECT_EQ(module->getDataLayout().getTypeAllocSize(arg), 32u);
  for (Instruction &inst : instructions(*kernel))
    EXPECT_FALSE(wide(inst.getType())) << print(*module);
}

// Run-time indices into a wide vector go through memory, one access each,
// not a comparison with every element.
TEST(WideVectorsTest, RunTimeIndicesGoThroughMemory) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %p, i64 %i) {
  %v = load <64 x i32>, ptr addrspace(1) %p, align 4
  %e = extractelement <64 x i32> %v, i64 %i
  %w = insertelement <64 x i32> %v, i32 %e, i64 %i
  store <64 x i32> %w, ptr addrspace(1) %p, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(splitWideVectors(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  unsigned selects = 0, indexed = 0;
  for (Instruction &inst : instructions(*module->getFunction("k"))) {
    EXPECT_FALSE(wide(inst.getType())) << print(*module);
    selects += isa<SelectInst>(inst);
    if (auto *gep = dyn_cast<GetElementPtrInst>(&inst))
      indexed +=
          gep->getNumIndices() == 2 && !isa<Constant>(gep->getOperand(2));
  }
  EXPECT_EQ(selects, 0u) << print(*module);
  EXPECT_EQ(indexed, 2u) << print(*module);
}

// Two reads of one vector at run-time indices, in a loop, share one copy
// stored where the vector is defined, before the loop.
TEST(WideVectorsTest, ReadsShareOneCopy) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %p, i64 %n) {
entry:
  %v = load <64 x i32>, ptr addrspace(1) %p, align 4
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %a = extractelement <64 x i32> %v, i64 %i
  %j = xor i64 %i, 1
  %b = extractelement <64 x i32> %v, i64 %j
  %s = add i32 %a, %b
  store i32 %s, ptr addrspace(1) %p, align 4
  %next = add i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(splitWideVectors(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  unsigned copies = 0, loopStores = 0;
  for (Instruction &inst : instructions(*module->getFunction("k"))) {
    EXPECT_FALSE(wide(inst.getType())) << print(*module);
    copies += isa<AllocaInst>(inst);
    loopStores += isa<StoreInst>(inst) && inst.getParent()->getName() == "loop";
  }
  EXPECT_EQ(copies, 1u) << print(*module);
  // Only the sum is stored in the loop.
  EXPECT_EQ(loopStores, 1u) << print(*module);
}

// Bools are packed in bits in a vector but take a byte each in an array, so
// they go through memory as bytes.
TEST(WideVectorsTest, BoolsGoThroughMemoryAsBytes) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %p, <32 x i32> %x, i64 %i) {
  %m = icmp sgt <32 x i32> %x, zeroinitializer
  %e = extractelement <32 x i1> %m, i64 %i
  %w = insertelement <32 x i1> %m, i1 false, i64 %i
  %z = zext <32 x i1> %w to <32 x i8>
  store <32 x i8> %z, ptr addrspace(1) %p, align 1
  %b = zext i1 %e to i8
  store i8 %b, ptr addrspace(1) %p, align 1
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(splitWideVectors(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  unsigned arrays = 0;
  for (Instruction &inst : instructions(*module->getFunction("k"))) {
    EXPECT_FALSE(wide(inst.getType())) << print(*module);
    for (Value *operand : inst.operands())
      EXPECT_FALSE(wide(operand->getType())) << print(*module);
    if (auto *alloca = dyn_cast<AllocaInst>(&inst)) {
      ++arrays;
      EXPECT_TRUE(
          alloca->getAllocatedType()->getArrayElementType()->isIntegerTy(8))
          << print(*module);
    }
  }
  EXPECT_EQ(arrays, 2u) << print(*module);
}

// Float vector constants whose bits can't be read without folding: a lane
// that is a constant expression, and a bitcast of an integer vector.
TEST(WideVectorsTest, UnreadableConstantsAreRebuilt) {
  std::string lanes = "float bitcast (i32 ptrtoint (ptr @g to i32) to float)";
  for (int i = 1; i < 32; ++i)
    lanes += ", float 2.0";
  std::string words = "i64 ptrtoint (ptr @g to i64)";
  for (int i = 1; i < 16; ++i)
    words += ", i64 4611686018427387904";
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@g = external addrspace(0) global i32
define spir_kernel void @k(ptr addrspace(1) %p) {
  %v = load <32 x float>, ptr addrspace(1) %p, align 4
  %a = fmul <32 x float> %v, <)" + lanes + R"(>
  %b = fadd <32 x float> %a, bitcast (<16 x i64> <)" +
                                                  words +
                                                  R"(> to <32 x float>)
  store <32 x float> %b, ptr addrspace(1) %p, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(splitWideVectors(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  for (Instruction &inst : instructions(*module->getFunction("k"))) {
    EXPECT_FALSE(wide(inst.getType())) << print(*module);
    for (Value *operand : inst.operands()) {
      EXPECT_FALSE(isa<ConstantFP>(operand)) << print(*module);
      EXPECT_FALSE(wide(operand->getType())) << print(*module);
    }
  }
  // 2.0f is 0x40000000.
  std::string text = print(*module);
  EXPECT_NE(text.find("i32 1073741824"), std::string::npos) << text;
}

// A rewritten call keeps its own attributes, not the callee's.
TEST(WideVectorsTest, CallsKeepTheirAttributes) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %p, i32 %n) {
  %v = load <32 x float>, ptr addrspace(1) %p, align 4
  %t = call <32 x float> @twice(<32 x float> noundef %v, i32 noundef %n) #0
  store <32 x float> %t, ptr addrspace(1) %p, align 4
  ret void
}
define internal nofpclass(nan) <32 x float> @twice(<32 x float> %x, i32 range(i32 0, 8) %n) {
  %y = fadd <32 x float> %x, %x
  ret <32 x float> %y
}
attributes #0 = { convergent }
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(splitWideVectors(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  CallInst *call = nullptr;
  for (Instruction &inst : instructions(*module->getFunction("k")))
    if (auto *c = dyn_cast<CallInst>(&inst))
      call = c;
  ASSERT_NE(call, nullptr) << print(*module);
  EXPECT_TRUE(call->hasFnAttr(Attribute::Convergent)) << print(*module);
  EXPECT_TRUE(call->paramHasAttr(1, Attribute::NoUndef)) << print(*module);
  // The callee's own parameter attribute isn't copied to the call.
  EXPECT_FALSE(call->getAttributes().hasParamAttr(1, Attribute::Range))
      << print(*module);
}

// Wide vectors as pointee types only: a GEP over vectors, and a local.
TEST(WideVectorsTest, PointeeTypesBecomeArrays) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %p, i64 %i) {
  %local = alloca <64 x i32>, align 256
  %row = getelementptr <64 x i32>, ptr addrspace(1) %p, i64 %i, i64 3
  %v = load i32, ptr addrspace(1) %row, align 4
  store i32 %v, ptr %local, align 4
  %back = load i32, ptr %local, align 4
  store i32 %back, ptr addrspace(1) %p, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(splitWideVectors(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  for (Instruction &inst : instructions(*module->getFunction("k"))) {
    if (auto *gep = dyn_cast<GetElementPtrInst>(&inst))
      EXPECT_TRUE(gep->getSourceElementType()->isArrayTy()) << print(*module);
    if (auto *alloca = dyn_cast<AllocaInst>(&inst))
      EXPECT_TRUE(alloca->getAllocatedType()->isArrayTy()) << print(*module);
  }
}

// Shared memory whose element struct places a wide vector by explicit
// padding, as Mojo lays it out: the global and the GEPs into it get arrays,
// with the same offsets.
TEST(WideVectorsTest, GlobalsWithPaddedVectorsBecomeArrays) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@shared = internal addrspace(3) global [4 x { float, [252 x i8], <64 x float> }] poison, align 256
define spir_kernel void @k(ptr addrspace(1) %out, i64 %i) {
  %f = getelementptr [4 x { float, [252 x i8], <64 x float> }], ptr addrspace(3) @shared, i64 0, i64 %i, i32 2, i64 5
  %v = load float, ptr addrspace(3) %f, align 4
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  const DataLayout &layout = module->getDataLayout();
  uint64_t size =
      layout.getTypeAllocSize(module->getNamedGlobal("shared")->getValueType());
  EXPECT_TRUE(splitWideVectors(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  GlobalVariable *shared = module->getNamedGlobal("shared");
  ASSERT_NE(shared, nullptr) << print(*module);
  EXPECT_EQ(print(*module).find("<64 x float>"), std::string::npos)
      << print(*module);
  EXPECT_EQ(layout.getTypeAllocSize(shared->getValueType()), size);
  EXPECT_EQ(shared->getAlign().valueOrOne().value(), 256u);
  EXPECT_EQ(shared->getAddressSpace(), 3u);
}

TEST(WideVectorsTest, NothingToDoWithoutWideVectors) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %p) {
  %v = load <16 x i32>, ptr addrspace(1) %p, align 64
  %a = add <16 x i32> %v, %v
  store <16 x i32> %a, ptr addrspace(1) %p, align 64
  ret void
}
)");
  ASSERT_TRUE(module);
  std::string before = print(*module);
  EXPECT_FALSE(splitWideVectors(*module));
  EXPECT_EQ(print(*module), before);
}

} // namespace
