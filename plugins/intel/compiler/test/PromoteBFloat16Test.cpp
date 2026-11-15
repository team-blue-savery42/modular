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
// Tests for computing bfloat16 in float32, on hand-written IR as the plugin
// sees it before codegen.
//
//===----------------------------------------------------------------------===//

#include "PromoteBFloat16.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace llvm;
using M::KGEN::Intel::promoteBFloat16;

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

/// Whether any instruction in `func` computes on bfloat16, rather than
/// converting it to or from another float type or moving it.
bool computesOnBF16(Function &func) {
  for (Instruction &inst : instructions(func)) {
    if (isa<FPExtInst>(inst) || isa<FPTruncInst>(inst) || isa<LoadInst>(inst) ||
        isa<StoreInst>(inst) || isa<ReturnInst>(inst) ||
        isa<ExtractElementInst>(inst) || isa<InsertElementInst>(inst) ||
        isa<ShuffleVectorInst>(inst) || isa<SelectInst>(inst) ||
        isa<PHINode>(inst) || isa<BitCastInst>(inst) || isa<FreezeInst>(inst))
      continue;
    auto bf16 = [](Type *type) { return type->getScalarType()->isBFloatTy(); };
    if (bf16(inst.getType()))
      return true;
    for (Value *operand : inst.operands())
      if (bf16(operand->getType()))
        return true;
  }
  return false;
}

/// Whether `type` is or holds bfloat16 values.
bool holdsBF16(Type *type) {
  if (type->getScalarType()->isBFloatTy())
    return true;
  return any_of(type->subtypes(), holdsBF16);
}

/// Whether every instruction in `func` that has an operand or result holding
/// bfloat16 is a conversion to or from another float type, a bitcast to or
/// from integers, a call, a return, or takes a field of a constant or of a
/// call's result (the edges, where aggregates are retyped field by field): no
/// select, phi, load, store, vector or aggregate move does.
bool onlyConvertsBF16(Function &func) {
  for (Instruction &inst : instructions(func)) {
    if (isa<FPExtInst>(inst) || isa<FPTruncInst>(inst) ||
        isa<BitCastInst>(inst) || isa<CallInst>(inst) || isa<ReturnInst>(inst))
      continue;
    if (auto *extract = dyn_cast<ExtractValueInst>(&inst))
      if (isa<Constant, CallInst>(extract->getAggregateOperand()))
        continue;
    if (holdsBF16(inst.getType()))
      return false;
    for (Value *operand : inst.operands())
      if (holdsBF16(operand->getType()))
        return false;
  }
  return true;
}

/// Whether the module's text names the bfloat type at all.
bool mentionsBFloat(const Module &module) {
  return print(module).find("bfloat") != std::string::npos;
}

constexpr const char *kKernel = R"(
define spir_kernel void @k(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %a = load bfloat, ptr addrspace(1) %p, align 2
  %b = fadd fast bfloat %a, 0xR3F80
  %c = fneg bfloat %b
  %d = call bfloat @llvm.sqrt.bf16(bfloat %c)
  %v = load <4 x bfloat>, ptr addrspace(1) %p, align 8
  %w = call <4 x bfloat> @llvm.fma.v4bf16(<4 x bfloat> %v, <4 x bfloat> %v, <4 x bfloat> %v)
  %e = extractelement <4 x bfloat> %w, i64 0
  %f = call bfloat @llvm.powi.bf16.i32(bfloat %e, i32 3)
  %lt = fcmp olt bfloat %d, %f
  %g = select i1 %lt, bfloat %d, bfloat %f
  %i = fptosi bfloat %g to i32
  %h = sitofp i32 %i to bfloat
  %nan = call i1 @llvm.is.fpclass.bf16(bfloat %h, i32 3)
  %j = zext i1 %nan to i32
  store i32 %j, ptr addrspace(1) %q, align 4
  store bfloat %h, ptr addrspace(1) %p, align 2
  store i32 %i, ptr addrspace(1) %q, align 4
  ret void
}
declare bfloat @llvm.sqrt.bf16(bfloat)
declare <4 x bfloat> @llvm.fma.v4bf16(<4 x bfloat>, <4 x bfloat>, <4 x bfloat>)
declare bfloat @llvm.powi.bf16.i32(bfloat, i32)
declare i1 @llvm.is.fpclass.bf16(bfloat, i32)
)";

TEST(PromoteBFloat16Test, ComputesInFloat32) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, kKernel);
  ASSERT_TRUE(module);
  EXPECT_TRUE(promoteBFloat16(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  Function *kernel = module->getFunction("k");
  EXPECT_FALSE(computesOnBF16(*kernel)) << print(*module);
  EXPECT_TRUE(onlyConvertsBF16(*kernel)) << print(*module);

  // The intrinsics are their float32 versions, keeping integer overloads.
  for (Instruction &inst : instructions(*kernel))
    if (auto *call = dyn_cast<IntrinsicInst>(&inst))
      for (Value *arg : call->args())
        EXPECT_FALSE(arg->getType()->getScalarType()->isBFloatTy())
            << print(*module);
  EXPECT_NE(module->getFunction("llvm.is.fpclass.f32"), nullptr)
      << print(*module);
  EXPECT_NE(module->getFunction("llvm.powi.f32.i32"), nullptr)
      << print(*module);
  EXPECT_NE(module->getFunction("llvm.fma.v4f32"), nullptr) << print(*module);

  // Fast-math flags carry over.
  bool sawFastAdd = false;
  for (Instruction &inst : instructions(*kernel))
    if (inst.getOpcode() == Instruction::FAdd)
      sawFastAdd = inst.hasAllowReassoc() && inst.getType()->isFloatTy();
  EXPECT_TRUE(sawFastAdd) << print(*module);
}

TEST(PromoteBFloat16Test, LeavesMemoryIntrinsicsAlone) {
  // As float32, a masked load would read twice the bytes.
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define void @k(ptr addrspace(1) %p, <8 x i1> %m, <8 x bfloat> %v) {
  %l = call <8 x bfloat> @llvm.masked.load.v8bf16.p1(ptr addrspace(1) %p, i32 2, <8 x i1> %m, <8 x bfloat> %v)
  call void @llvm.masked.store.v8bf16.p1(<8 x bfloat> %l, ptr addrspace(1) %p, i32 2, <8 x i1> %m)
  ret void
}
declare <8 x bfloat> @llvm.masked.load.v8bf16.p1(ptr addrspace(1), i32, <8 x i1>, <8 x bfloat>)
declare void @llvm.masked.store.v8bf16.p1(<8 x bfloat>, ptr addrspace(1), i32, <8 x i1>)
)");
  ASSERT_TRUE(module);
  std::string before = print(*module);
  EXPECT_FALSE(promoteBFloat16(*module));
  EXPECT_EQ(print(*module), before);
}

/// The bfloat16 bits `func`, which returns a bfloat16, gives once its
/// instructions are folded (test-side: this LLVM made the constants).
std::optional<uint64_t> foldedResult(Function &func) {
  const DataLayout &layout = func.getParent()->getDataLayout();
  for (Instruction &inst : make_early_inc_range(instructions(func)))
    if (Constant *folded = ConstantFoldInstruction(&inst, layout)) {
      inst.replaceAllUsesWith(folded);
      inst.eraseFromParent();
    }
  for (Instruction &inst : instructions(func))
    if (auto *ret = dyn_cast<ReturnInst>(&inst))
      if (auto *value = dyn_cast<ConstantFP>(ret->getReturnValue()))
        return value->getValueAPF().bitcastToAPInt().getZExtValue();
  return std::nullopt;
}

TEST(PromoteBFloat16Test, IntegersRoundOnce) {
  // Through float32 to nearest, each rounds twice and lands a step lower:
  // 2^24 + 2^16 + 1 and 2^60 + 2^52 + 1 are just above halfway between two
  // bfloat16 values, and float32 rounds them down to exactly halfway.
  struct Case {
    const char *conversion;
    uint64_t bits;
  };
  for (Case c : {
           Case{"uitofp i32 16842753", 0x4B81},            // 2^24 + 2^17
           Case{"sitofp i32 -16842753", 0xCB81},           // -(2^24 + 2^17)
           Case{"sitofp i64 1157425104234217473", 0x5D81}, // 2^60 + 2^53
           Case{"uitofp i32 16842752", 0x4B80},    // halfway: to even, 2^24
           Case{"sitofp i32 -2147483648", 0xCF00}, // -2^31, exact
           Case{"uitofp i16 1000", 0x447A},        // 1000, exact
       }) {
    LLVMContext ctx;
    std::unique_ptr<Module> module =
        parse(ctx, std::string("define bfloat @k() {\n  %h = ") + c.conversion +
                       " to bfloat\n  ret bfloat %h\n}\n");
    ASSERT_TRUE(module);
    EXPECT_TRUE(promoteBFloat16(*module));
    ASSERT_EQ(verify(*module), "") << print(*module);
    std::optional<uint64_t> bits = foldedResult(*module->getFunction("k"));
    ASSERT_TRUE(bits) << c.conversion << "\n" << print(*module);
    EXPECT_EQ(*bits, c.bits) << c.conversion;
  }
}

TEST(PromoteBFloat16Test, LeavesOtherTypesAlone) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define half @k(half %a, float %b, ptr %p) {
  %c = fadd half %a, %a
  %d = fmul float %b, %b
  %e = load half, ptr %p, align 2
  %s = select i1 true, half %e, half %c
  store half %s, ptr %p, align 2
  ret half %c
}
)");
  ASSERT_TRUE(module);
  std::string before = print(*module);
  EXPECT_FALSE(promoteBFloat16(*module));
  EXPECT_EQ(print(*module), before);
}

TEST(PromoteBFloat16Test, MovesAsIntegers) {
  // Loads, stores, vector element moves and freezes of bfloat16 become i16:
  // nothing is left of the type.
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define void @k(ptr addrspace(1) %p, ptr addrspace(1) %q, i32 %i) {
  %a = load volatile bfloat, ptr addrspace(1) %p, align 2, !nontemporal !0
  %v = load <4 x bfloat>, ptr addrspace(1) %p, align 8
  %e = extractelement <4 x bfloat> %v, i32 %i
  %w = insertelement <4 x bfloat> %v, bfloat %a, i64 1
  %s = shufflevector <4 x bfloat> %w, <4 x bfloat> poison, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 0, i32 1, i32 2, i32 poison>
  %f = freeze bfloat %e
  store atomic bfloat %f, ptr addrspace(1) %q syncscope("device") monotonic, align 2
  store <8 x bfloat> %s, ptr addrspace(1) %q, align 16
  ret void
}
!0 = !{i32 1}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(promoteBFloat16(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  EXPECT_FALSE(mentionsBFloat(*module)) << print(*module);
  Function *kernel = module->getFunction("k");
  // Volatile, atomic and metadata carry over.
  bool sawVolatile = false, sawAtomic = false;
  for (Instruction &inst : instructions(*kernel)) {
    if (auto *load = dyn_cast<LoadInst>(&inst))
      if (load->isVolatile()) {
        sawVolatile = load->getType()->isIntegerTy(16) &&
                      load->hasMetadata(LLVMContext::MD_nontemporal);
      }
    if (auto *store = dyn_cast<StoreInst>(&inst))
      if (store->isAtomic())
        sawAtomic = store->getOrdering() == AtomicOrdering::Monotonic &&
                    store->getSyncScopeID() != SyncScope::System;
  }
  EXPECT_TRUE(sawVolatile) << print(*module);
  EXPECT_TRUE(sawAtomic) << print(*module);
}

TEST(PromoteBFloat16Test, MovesSelectsAndConstantsAsIntegers) {
  // A select of a loaded value and a constant (as MAX's log: x == inf ? inf
  // : ...). Each bfloat16 constant becomes the i16 constant of its bits, so
  // the bfloat16 type leaves the module with it.
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define i16 @k(ptr addrspace(1) %p, i1 %c) {
  %a = load bfloat, ptr addrspace(1) %p, align 2
  %s = select nnan i1 %c, bfloat %a, bfloat 0xR7F80
  %t = select i1 %c, bfloat 0xR3FC0, bfloat 0xR0000
  store bfloat %t, ptr addrspace(1) %p, align 2
  store bfloat poison, ptr addrspace(1) %p, align 2
  %x = fpext bfloat %s to float
  %y = fptrunc float %x to bfloat
  %z = bitcast bfloat %y to i16
  ret i16 %z
}
define bfloat @inf() {
  ret bfloat 0xR7F80
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(promoteBFloat16(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  Function *kernel = module->getFunction("k");
  EXPECT_TRUE(onlyConvertsBF16(*kernel)) << print(*module);
  unsigned selects = 0;
  for (Instruction &inst : instructions(*kernel)) {
    // The i32 select is the rounding's NaN case (convertAsIntegers).
    if (auto *select = dyn_cast<SelectInst>(&inst);
        select && !select->getType()->isIntegerTy(32)) {
      ++selects;
      EXPECT_TRUE(select->getType()->isIntegerTy(16)) << print(*module);
    }
    // No bitcast of a constant is left: the constants were read.
    if (auto *cast = dyn_cast<BitCastInst>(&inst))
      EXPECT_FALSE(isa<Constant>(cast->getOperand(0))) << print(*module);
  }
  EXPECT_EQ(selects, 2u);
  std::string text = print(*module);
  EXPECT_NE(text.find("i16 32640"), std::string::npos) << text; // 0x7F80
  EXPECT_NE(text.find("i16 16320"), std::string::npos) << text; // 0x3FC0
  EXPECT_NE(text.find("i16 0"), std::string::npos) << text;
  EXPECT_NE(text.find("store i16 poison"), std::string::npos) << text;
}

TEST(PromoteBFloat16Test, MovesAggregatesAsIntegers) {
  // A struct of two bfloat16 values (as the stdlib's modf returns), built,
  // selected and taken apart; one built from a constant. Its fields are i16.
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define float @k(bfloat %a, bfloat %b, i1 %c, ptr %p) {
  %s = insertvalue { bfloat, bfloat } undef, bfloat %a, 0
  %t = insertvalue { bfloat, bfloat } %s, bfloat %b, 1
  %u = insertvalue { { bfloat, bfloat } } undef, { bfloat, bfloat } %t, 0
  %k = insertvalue { { bfloat, bfloat } } undef, { bfloat, bfloat } { bfloat 0xR3FC0, bfloat undef }, 0
  %v = select i1 %c, { { bfloat, bfloat } } %u, { { bfloat, bfloat } } %k
  %e = extractvalue { { bfloat, bfloat } } %v, 0, 1
  %x = fpext bfloat %e to float
  store { bfloat, bfloat } %t, ptr %p, align 2
  ret float %x
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(promoteBFloat16(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  Function *kernel = module->getFunction("k");
  EXPECT_TRUE(onlyConvertsBF16(*kernel)) << print(*module);
  unsigned selects = 0;
  for (Instruction &inst : instructions(*kernel))
    if (auto *select = dyn_cast<SelectInst>(&inst))
      selects += select->getType() ==
                 StructType::get(StructType::get(Type::getInt16Ty(ctx),
                                                 Type::getInt16Ty(ctx)));
  EXPECT_EQ(selects, 1u) << print(*module);
}

TEST(PromoteBFloat16Test, RetypesConstantAggregatesWhole) {
  // A constant struct of bfloat16 values, moved into another struct, becomes
  // an i16 constant: an extractvalue of the bfloat16 constant would stay in
  // the SPIR-V as a composite extract of a bfloat16 composite.
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define void @k(ptr %p, bfloat %a) {
  %s = insertvalue { bfloat, bfloat } { bfloat 0xR0000, bfloat undef }, bfloat %a, 1
  %t = insertvalue { { bfloat, bfloat } } undef, { bfloat, bfloat } %s, 0
  store { { bfloat, bfloat } } %t, ptr %p, align 2
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(promoteBFloat16(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  for (Instruction &inst : instructions(*module->getFunction("k")))
    if (auto *extract = dyn_cast<ExtractValueInst>(&inst))
      EXPECT_FALSE(isa<Constant>(extract->getAggregateOperand()))
          << print(*module);
  EXPECT_TRUE(onlyConvertsBF16(*module->getFunction("k"))) << print(*module);
}

TEST(PromoteBFloat16Test, ShufflesInterleavesOfBFloat16) {
  // A rope's even/odd split and its inverse: the intrinsics become shuffles,
  // which move as i16 vectors, so the SPIR-V has no bfloat16 vector.
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define void @k(ptr %p, ptr %q) {
  %v = load <8 x bfloat>, ptr %p, align 16
  %d = call { <4 x bfloat>, <4 x bfloat> } @llvm.vector.deinterleave2.v8bf16(<8 x bfloat> %v)
  %e = extractvalue { <4 x bfloat>, <4 x bfloat> } %d, 0
  %o = extractvalue { <4 x bfloat>, <4 x bfloat> } %d, 1
  %w = call <8 x bfloat> @llvm.vector.interleave2.v8bf16(<4 x bfloat> %o, <4 x bfloat> %e)
  store <8 x bfloat> %w, ptr %q, align 16
  ret void
}
declare { <4 x bfloat>, <4 x bfloat> } @llvm.vector.deinterleave2.v8bf16(<8 x bfloat>)
declare <8 x bfloat> @llvm.vector.interleave2.v8bf16(<4 x bfloat>, <4 x bfloat>)
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(promoteBFloat16(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  Function *kernel = module->getFunction("k");
  EXPECT_TRUE(onlyConvertsBF16(*kernel)) << print(*module);
  for (Instruction &inst : instructions(*kernel))
    EXPECT_FALSE(isa<IntrinsicInst>(inst)) << print(*module);
}

TEST(PromoteBFloat16Test, MovesPhisAsIntegers) {
  // A loop phi whose back-edge value is computed (and so moved) after it, and
  // a join of a constant and a computed value (as MAX's pow).
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define void @k(ptr addrspace(1) %p, i32 %n, i1 %c) {
entry:
  %a = load bfloat, ptr addrspace(1) %p, align 2
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %j, %loop ]
  %x = phi bfloat [ %a, %entry ], [ %y, %loop ]
  %y = fmul bfloat %x, 0xR3FC0
  %j = add i32 %i, 1
  %done = icmp eq i32 %j, %n
  br i1 %done, label %exit, label %loop
exit:
  br i1 %c, label %then, label %join
then:
  %z = fadd bfloat %y, %y
  br label %join
join:
  %r = phi bfloat [ %z, %then ], [ 0xR7FC0, %exit ]
  store bfloat %r, ptr addrspace(1) %p, align 2
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(promoteBFloat16(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  Function *kernel = module->getFunction("k");
  EXPECT_FALSE(computesOnBF16(*kernel)) << print(*module);
  EXPECT_TRUE(onlyConvertsBF16(*kernel)) << print(*module);
  unsigned phis = 0;
  for (Instruction &inst : instructions(*kernel)) {
    if (auto *phi = dyn_cast<PHINode>(&inst)) {
      phis += phi->getType()->isIntegerTy(16);
      // Frozen, so that llc's CodeGenPrepare can't retype the phi back.
      if (phi->getType()->isIntegerTy(16))
        for (Value *incoming : phi->incoming_values())
          EXPECT_TRUE(isa<FreezeInst>(incoming)) << print(*module);
    }
    // No bitcast pair is left: a value moved through i16 isn't cast back
    // and forth.
    if (auto *cast = dyn_cast<BitCastInst>(&inst))
      EXPECT_FALSE(isa<BitCastInst>(cast->getOperand(0))) << print(*module);
  }
  EXPECT_EQ(phis, 2u) << print(*module);
}

TEST(PromoteBFloat16Test, MovesPhisWithRepeatedBlocks) {
  // A switch whose cases go to the same block gives the phis one entry per
  // case from that block; the entries of one block must stay one value. A
  // scalar phi's entries are each frozen, so %r's entries of the moved %a,
  // and %c's constants, would each get their own freeze. A vector or
  // aggregate entry that is a moved instruction (%v, %s) is one value
  // already; the constants, the argument %va and the call result %u are not
  // moved, and each entry of theirs would get its own bitcast or field copy.
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
declare { bfloat, i32 } @g()

define void @k(ptr addrspace(1) %p, i32 %x, <2 x bfloat> %va) {
entry:
  %a = load bfloat, ptr addrspace(1) %p, align 2
  %v = load <2 x bfloat>, ptr addrspace(1) %p, align 4
  %s = insertvalue { bfloat, i32 } poison, bfloat %a, 0
  %u = call { bfloat, i32 } @g()
  switch i32 %x, label %join [ i32 1, label %join
                               i32 2, label %join
                               i32 3, label %other ]
other:
  br label %join
join:
  %r = phi bfloat [ %a, %entry ], [ %a, %entry ], [ %a, %entry ], [ %a, %other ]
  %c = phi bfloat [ 0xR3FC0, %entry ], [ 0xR3FC0, %entry ], [ 0xR3FC0, %entry ], [ 0xR3FC0, %other ]
  %w = phi <2 x bfloat> [ %v, %entry ], [ %v, %entry ], [ %v, %entry ], [ <bfloat 0xR3F80, bfloat 0xR4000>, %other ]
  %wc = phi <2 x bfloat> [ <bfloat 0xR3F80, bfloat 0xR4000>, %entry ], [ <bfloat 0xR3F80, bfloat 0xR4000>, %entry ], [ <bfloat 0xR3F80, bfloat 0xR4000>, %entry ], [ %v, %other ]
  %wa = phi <2 x bfloat> [ %va, %entry ], [ %va, %entry ], [ %va, %entry ], [ %v, %other ]
  %t = phi { bfloat, i32 } [ %s, %entry ], [ %s, %entry ], [ %s, %entry ], [ { bfloat 0xR3F80, i32 1 }, %other ]
  %tc = phi { bfloat, i32 } [ { bfloat 0xR3F80, i32 1 }, %entry ], [ { bfloat 0xR3F80, i32 1 }, %entry ], [ { bfloat 0xR3F80, i32 1 }, %entry ], [ %s, %other ]
  %tu = phi { bfloat, i32 } [ %u, %entry ], [ %u, %entry ], [ %u, %entry ], [ %s, %other ]
  %f = fadd bfloat %r, %c
  %e = extractelement <2 x bfloat> %w, i64 1
  %g = fadd bfloat %f, %e
  %h = extractvalue { bfloat, i32 } %t, 0
  %i = fadd bfloat %g, %h
  %ec = extractelement <2 x bfloat> %wc, i64 1
  %i2 = fadd bfloat %i, %ec
  %ea = extractelement <2 x bfloat> %wa, i64 0
  %i3 = fadd bfloat %i2, %ea
  %hc = extractvalue { bfloat, i32 } %tc, 0
  %i4 = fadd bfloat %i3, %hc
  %hu = extractvalue { bfloat, i32 } %tu, 0
  %i5 = fadd bfloat %i4, %hu
  store bfloat %i5, ptr addrspace(1) %p, align 2
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(promoteBFloat16(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  Function *kernel = module->getFunction("k");
  EXPECT_TRUE(onlyConvertsBF16(*kernel)) << print(*module);
  unsigned phis = 0;
  for (Instruction &inst : instructions(*kernel))
    if (auto *phi = dyn_cast<PHINode>(&inst)) {
      ++phis;
      Value *fromEntry = phi->getIncomingValue(0);
      for (unsigned i = 1; i < 3; ++i)
        EXPECT_EQ(phi->getIncomingValue(i), fromEntry) << print(*module);
      EXPECT_NE(phi->getIncomingValue(3), fromEntry) << print(*module);
    }
  EXPECT_EQ(phis, 8u) << print(*module);
}

/// Every floating-point constant `module`'s instructions use.
std::set<Constant *> floatConstants(Module &module) {
  std::set<Constant *> found;
  for (Function &func : module)
    for (Instruction &inst : instructions(func))
      for (Value *operand : inst.operands())
        if (auto *constant = dyn_cast<Constant>(operand))
          if (constant->getType()->isFPOrFPVectorTy())
            found.insert(constant);
  return found;
}

TEST(PromoteBFloat16Test, ConvertsWithIntegers) {
  // Conversions between bfloat16 and float32 become integer operations:
  // no fpext or fptrunc from or to bfloat16 is left, float64 goes through
  // float32 both ways (as the host's __truncdfbf2 does), nothing reads a
  // constant (0xR3FC0 is bitcast as it is), and no floating-point constant or
  // declaration is added.
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define void @k(ptr addrspace(1) %p, float %x, <4 x float> %v, double %d) {
  %a = load bfloat, ptr addrspace(1) %p, align 2
  %b = fadd bfloat %a, 0xR3FC0
  %w = load <4 x bfloat>, ptr addrspace(1) %p, align 8
  %e = fpext <4 x bfloat> %w to <4 x float>
  %t = fptrunc <4 x float> %e to <4 x bfloat>
  store <4 x bfloat> %t, ptr addrspace(1) %p, align 8
  %f = fpext bfloat %b to double
  store double %f, ptr addrspace(1) %p, align 8
  %g = fptrunc float %x to bfloat
  %h = fptrunc double %d to bfloat
  %s = select i1 true, bfloat %g, bfloat %h
  store bfloat %s, ptr addrspace(1) %p, align 2
  %u = fptrunc <4 x float> %v to <4 x bfloat>
  %c = call float @use(bfloat %g, <4 x bfloat> %u)
  ret void
}
declare float @use(bfloat, <4 x bfloat>)
)");
  ASSERT_TRUE(module);
  std::set<Constant *> constantsBefore = floatConstants(*module);
  size_t functionsBefore = module->size();
  EXPECT_TRUE(promoteBFloat16(*module));
  ASSERT_EQ(verify(*module), "") << print(*module);
  Function *kernel = module->getFunction("k");
  EXPECT_TRUE(onlyConvertsBF16(*kernel)) << print(*module);
  unsigned truncs = 0, exts = 0;
  for (Instruction &inst : instructions(*kernel)) {
    if (isa<FPTruncInst>(inst)) {
      ++truncs; // float64 to float32, for %h
      EXPECT_TRUE(inst.getOperand(0)->getType()->isDoubleTy())
          << print(*module);
      EXPECT_TRUE(inst.getType()->isFloatTy()) << print(*module);
    }
    if (isa<FPExtInst>(inst)) {
      ++exts; // float32 to float64, for %f
      EXPECT_TRUE(inst.getOperand(0)->getType()->isFloatTy()) << print(*module);
    }
  }
  EXPECT_EQ(truncs, 1u) << print(*module);
  EXPECT_EQ(exts, 1u) << print(*module);
  // Vectors convert lane by lane: no vector integer arithmetic or select
  // (IGC's optimizer gets the vector rounding wrong).
  for (Instruction &inst : instructions(*kernel))
    if (isa<BinaryOperator, SelectInst, ICmpInst, TruncInst, ZExtInst>(inst))
      EXPECT_FALSE(inst.getType()->isVectorTy()) << print(*module);
  for (Constant *constant : floatConstants(*module))
    EXPECT_TRUE(constantsBefore.count(constant)) << print(*module);
  EXPECT_EQ(module->size(), functionsBefore) << print(*module);
  // The call still gets bfloat16 values, made from the rounded bits.
  for (Instruction &inst : instructions(*kernel))
    if (auto *call = dyn_cast<CallInst>(&inst))
      for (Value *arg : call->args())
        EXPECT_TRUE(isa<BitCastInst>(arg)) << print(*module);
}

/// A function that calls @sink(i32 index, iN bits) once per input, with
/// `convert`'s result for that input's bits, built without folding.
template <typename Convert>
std::unique_ptr<Module> sinkModule(LLVMContext &ctx, unsigned resultBits,
                                   const std::vector<uint32_t> &inputs,
                                   Convert convert) {
  std::unique_ptr<Module> module =
      parse(ctx, "declare void @sink(i32, i" + std::to_string(resultBits) +
                     ")\ndefine void @k() {\n  ret void\n}\n");
  if (!module)
    return module;
  Function *kernel = module->getFunction("k");
  IRBuilder<NoFolder> builder(kernel->getEntryBlock().getTerminator());
  Function *sink = module->getFunction("sink");
  for (size_t i = 0; i < inputs.size(); ++i)
    builder.CreateCall(sink,
                       {builder.getInt32(i), convert(builder, inputs[i])});
  return module;
}

/// The bits each @sink call passes once `module`'s instructions are folded
/// (test-side: this LLVM made the constants), by index.
std::vector<uint64_t> sunkBits(Module &module, size_t count) {
  Function &kernel = *module.getFunction("k");
  const DataLayout &layout = module.getDataLayout();
  for (Instruction &inst : make_early_inc_range(instructions(kernel)))
    if (Constant *folded = ConstantFoldInstruction(&inst, layout)) {
      inst.replaceAllUsesWith(folded);
      inst.eraseFromParent();
    }
  std::vector<uint64_t> bits(count, ~uint64_t(0));
  for (Instruction &inst : instructions(kernel))
    if (auto *call = dyn_cast<CallInst>(&inst)) {
      auto *index = dyn_cast<ConstantInt>(call->getArgOperand(0));
      auto *value = dyn_cast<ConstantInt>(call->getArgOperand(1));
      if (index && value && index->getZExtValue() < count)
        bits[index->getZExtValue()] = value->getZExtValue();
    }
  return bits;
}

TEST(PromoteBFloat16Test, IntegerRoundingMatchesAPFloat) {
  // float32 to bfloat16 with integers: for every sign and exponent, mantissas
  // at both ends of the binade, with the lower half at each rounding edge
  // (below, at and above the tie, and the extremes). The reference is
  // APFloat's round to nearest even, and for a NaN its sign and the upper
  // half of its payload, made quiet, as the host converts since upstream's
  // LLVM 9b2201bd0e35.
  std::vector<uint32_t> inputs;
  for (uint32_t upper = 0; upper < 0x10000; ++upper) {
    uint32_t mantissa = upper & 0x7f;
    if (mantissa > 1 && mantissa < 0x7e)
      continue;
    for (uint32_t lower :
         {0x0000u, 0x0001u, 0x7fffu, 0x8000u, 0x8001u, 0xffffu})
      inputs.push_back(upper << 16 | lower);
  }
  LLVMContext ctx;
  std::unique_ptr<Module> module =
      sinkModule(ctx, 16, inputs, [](IRBuilder<NoFolder> &b, uint32_t bits) {
        Value *x = b.CreateBitCast(b.getInt32(bits), b.getFloatTy());
        Value *rounded = b.CreateFPTrunc(x, b.getBFloatTy());
        return b.CreateBitCast(rounded, b.getInt16Ty());
      });
  ASSERT_TRUE(module);
  EXPECT_TRUE(promoteBFloat16(*module));
  ASSERT_EQ(verify(*module), "");
  for (Instruction &inst : instructions(*module->getFunction("k")))
    ASSERT_FALSE(isa<FPTruncInst>(inst));
  std::vector<uint64_t> got = sunkBits(*module, inputs.size());
  unsigned wrong = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    APFloat want(APFloat::IEEEsingle(), APInt(32, inputs[i]));
    uint64_t wantBits = (inputs[i] | 0x00400000) >> 16;
    if (!want.isNaN()) {
      bool losesInfo;
      want.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven, &losesInfo);
      wantBits = want.bitcastToAPInt().getZExtValue();
    }
    if (got[i] != wantBits && wrong++ < 5)
      ADD_FAILURE() << std::hex << "0x" << inputs[i] << ": 0x" << got[i]
                    << ", want 0x" << wantBits;
  }
  EXPECT_EQ(wrong, 0u) << "of " << inputs.size();
}

TEST(PromoteBFloat16Test, IntegerExtensionIsExact) {
  // bfloat16 to float32 with integers, all 65,536 values: the bits shifted
  // up, so every value is APFloat's exact conversion and every NaN keeps its
  // payload, signaling ones included (APFloat would make them quiet).
  std::vector<uint32_t> inputs;
  for (uint32_t bits = 0; bits < 0x10000; ++bits)
    inputs.push_back(bits);
  LLVMContext ctx;
  std::unique_ptr<Module> module =
      sinkModule(ctx, 32, inputs, [](IRBuilder<NoFolder> &b, uint32_t bits) {
        Value *x = b.CreateBitCast(b.getInt16(bits), b.getBFloatTy());
        Value *wide = b.CreateFPExt(x, b.getFloatTy());
        return b.CreateBitCast(wide, b.getInt32Ty());
      });
  ASSERT_TRUE(module);
  EXPECT_TRUE(promoteBFloat16(*module));
  ASSERT_EQ(verify(*module), "");
  for (Instruction &inst : instructions(*module->getFunction("k")))
    ASSERT_FALSE(isa<FPExtInst>(inst));
  std::vector<uint64_t> got = sunkBits(*module, inputs.size());
  unsigned wrong = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    APFloat value(APFloat::BFloat(), APInt(16, inputs[i]));
    uint64_t wantBits = uint64_t(inputs[i]) << 16;
    if (!value.isNaN()) {
      bool losesInfo;
      value.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven,
                    &losesInfo);
      EXPECT_EQ(value.bitcastToAPInt().getZExtValue(), wantBits);
    }
    if (got[i] != wantBits && wrong++ < 5)
      ADD_FAILURE() << std::hex << "0x" << inputs[i] << ": 0x" << got[i]
                    << ", want 0x" << wantBits;
  }
  EXPECT_EQ(wrong, 0u);
}

} // namespace
