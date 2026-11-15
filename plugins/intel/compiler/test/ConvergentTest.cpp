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
// Tests for marking the work-group and sub-group built-ins convergent.
//
//===----------------------------------------------------------------------===//

#include "Convergent.h"

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

using namespace llvm;
using M::KGEN::Intel::isConvergentBuiltin;
using M::KGEN::Intel::markConvergentFunctions;

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

/// Runs LLVM's non-trivial loop unswitching, as -O3 does, on `module`.
void unswitch(Module &module) {
  LoopAnalysisManager lam;
  FunctionAnalysisManager fam;
  CGSCCAnalysisManager cgam;
  ModuleAnalysisManager mam;
  PassBuilder builder(module.getContext());
  builder.registerModuleAnalyses(mam);
  builder.registerCGSCCAnalyses(cgam);
  builder.registerFunctionAnalyses(fam);
  builder.registerLoopAnalyses(lam);
  builder.crossRegisterProxies(lam, fam, cgam, mam);
  ModulePassManager mpm;
  Error err = builder.parsePassPipeline(
      mpm, "function(loop-mssa(simple-loop-unswitch<nontrivial>))");
  ASSERT_FALSE(err) << toString(std::move(err));
  mpm.run(module, mam);
}

unsigned countCalls(Function &func, StringRef callee) {
  unsigned count = 0;
  for (Instruction &inst : instructions(func))
    if (auto *call = dyn_cast<CallBase>(&inst))
      if (call->getCalledFunction() &&
          call->getCalledFunction()->getName() == callee)
        ++count;
  return count;
}

// The shape of MAX's test_matmul_1_sram: each trip of a loop stores a value
// to shared memory, loaded only where a loop-invariant condition holds (the
// work-item's column is inside the matrix), then waits at a barrier and reads
// what the other work-items stored.
const char *kTiledLoop = R"(
@tile = internal addrspace(3) global [64 x float] poison

define spir_kernel void @k(ptr addrspace(1) %b, ptr addrspace(1) %c, i64 %col, i64 %n, i64 %lid) {
entry:
  %inside = icmp slt i64 %col, %n
  br label %loop

loop:
  %t = phi i64 [ 0, %entry ], [ %t.next, %latch ]
  %acc = phi float [ 0.0, %entry ], [ %acc.next, %latch ]
  br i1 %inside, label %load, label %store

load:
  %src = getelementptr float, ptr addrspace(1) %b, i64 %t
  %v.in = load float, ptr addrspace(1) %src, align 4
  br label %store

store:
  %v = phi float [ %v.in, %load ], [ 0.0, %loop ]
  %dst = getelementptr float, ptr addrspace(3) @tile, i64 %lid
  store float %v, ptr addrspace(3) %dst, align 4
  call void @_Z7barrierj(i32 3)
  %other = load float, ptr addrspace(3) @tile, align 4
  %acc.next = fadd float %acc, %other
  call void @_Z7barrierj(i32 3)
  br label %latch

latch:
  %t.next = add i64 %t, 1
  %more = icmp slt i64 %t.next, 16
  br i1 %more, label %loop, label %exit

exit:
  %out = getelementptr float, ptr addrspace(1) %c, i64 %lid
  store float %acc.next, ptr addrspace(1) %out, align 4
  ret void
}

declare void @_Z7barrierj(i32)
)";

TEST(ConvergentTest, NamesTheGroupBuiltins) {
  EXPECT_TRUE(isConvergentBuiltin("_Z7barrierj"));
  EXPECT_TRUE(isConvergentBuiltin("_Z18work_group_barrierj"));
  EXPECT_TRUE(isConvergentBuiltin("_Z17sub_group_barrierj"));
  EXPECT_TRUE(isConvergentBuiltin("_Z17sub_group_shufflejj"));
  EXPECT_TRUE(isConvergentBuiltin("_Z16sub_group_balloti"));
  EXPECT_TRUE(isConvergentBuiltin("_Z20sub_group_reduce_addi"));
  EXPECT_TRUE(isConvergentBuiltin("_Z20work_group_broadcastjm"));
  EXPECT_TRUE(isConvergentBuiltin("_Z27intel_sub_group_block_read8PU3AS1Kj"));
  EXPECT_TRUE(isConvergentBuiltin("_Z22__spirv_ControlBarrieriii"));
  // SPV_INTEL_split_barrier's halves.
  EXPECT_TRUE(isConvergentBuiltin("_Z33__spirv_ControlBarrierArriveINTELiii"));
  EXPECT_TRUE(isConvergentBuiltin("_Z31__spirv_ControlBarrierWaitINTELiii"));
  EXPECT_TRUE(isConvergentBuiltin("_Z28__spirv_GroupNonUniformBallotib"));
  EXPECT_TRUE(isConvergentBuiltin("_Z28__spirv_SubgroupShuffleINTELij"));
  // Queries and other built-ins aren't.
  EXPECT_FALSE(isConvergentBuiltin("_Z22get_sub_group_local_idv"));
  EXPECT_FALSE(isConvergentBuiltin("_Z12get_local_idj"));
  EXPECT_FALSE(isConvergentBuiltin("_Z6printfPU3AS4Kcz"));
  EXPECT_FALSE(isConvergentBuiltin("_Z8prefetchPU3AS1Kcm"));
  EXPECT_FALSE(isConvergentBuiltin("__mojo_intel_abort"));
  // Not a whole name: `barrier` must be the entire source name.
  EXPECT_FALSE(isConvergentBuiltin("_Z10barrierxyzj"));
  EXPECT_FALSE(isConvergentBuiltin("barrier_count"));
  // Malformed manglings are taken whole.
  EXPECT_FALSE(isConvergentBuiltin("_Z99barrier"));
  EXPECT_FALSE(isConvergentBuiltin("_Zbarrier"));
}

// Without the mark, LLVM copies the loop once per value of the condition:
// the work-items inside and outside the matrix then wait at different
// barriers. This is what broke test_matmul_1_sram's edge work-groups.
TEST(ConvergentTest, UnmarkedBarrierLoopIsUnswitched) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, kTiledLoop);
  ASSERT_TRUE(module);
  unswitch(*module);
  ASSERT_FALSE(verifyModule(*module, &errs()));
  EXPECT_EQ(countCalls(*module->getFunction("k"), "_Z7barrierj"), 4u)
      << print(*module);
}

TEST(ConvergentTest, MarkedBarrierLoopIsNotUnswitched) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, kTiledLoop);
  ASSERT_TRUE(module);
  EXPECT_TRUE(markConvergentFunctions(*module));
  EXPECT_TRUE(module->getFunction("_Z7barrierj")->isConvergent());
  unswitch(*module);
  ASSERT_FALSE(verifyModule(*module, &errs()));
  EXPECT_EQ(countCalls(*module->getFunction("k"), "_Z7barrierj"), 2u)
      << print(*module);
}

// Functions that reach a built-in through calls are convergent too, so a
// call to them isn't unswitched either; others are left alone.
TEST(ConvergentTest, MarksCallersTransitively) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k() {
  call void @outer()
  call void @plain()
  ret void
}
define internal void @outer() {
  call void @inner()
  ret void
}
define internal void @inner() {
  %x = call i32 @_Z17sub_group_shufflejj(i32 1, i32 2)
  ret void
}
define internal void @plain() {
  %id = call i64 @_Z12get_local_idj(i32 0)
  ret void
}
declare i32 @_Z17sub_group_shufflejj(i32, i32)
declare i64 @_Z12get_local_idj(i32)
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(markConvergentFunctions(*module));
  EXPECT_TRUE(module->getFunction("_Z17sub_group_shufflejj")->isConvergent());
  EXPECT_TRUE(module->getFunction("inner")->isConvergent());
  EXPECT_TRUE(module->getFunction("outer")->isConvergent());
  EXPECT_TRUE(module->getFunction("k")->isConvergent());
  EXPECT_FALSE(module->getFunction("plain")->isConvergent());
  EXPECT_FALSE(module->getFunction("_Z12get_local_idj")->isConvergent());
  // A second run finds nothing to do.
  EXPECT_FALSE(markConvergentFunctions(*module));
}

// A module with no group built-ins is unchanged.
TEST(ConvergentTest, LeavesOtherModulesAlone) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k(ptr addrspace(1) %p) {
  %id = call i64 @_Z12get_local_idj(i32 0)
  store i64 %id, ptr addrspace(1) %p, align 8
  ret void
}
declare i64 @_Z12get_local_idj(i32)
)");
  ASSERT_TRUE(module);
  std::string before = print(*module);
  EXPECT_FALSE(markConvergentFunctions(*module));
  EXPECT_EQ(print(*module), before);
}

} // namespace
