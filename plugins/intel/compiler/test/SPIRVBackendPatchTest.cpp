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
// Tests for the patches to LLVM's SPIR-V backend
// (bazel/public-patches/llvm-spirv-*.patch), on hand-written IR run through
// the backend in this process. The debug build has LLVM's assertions on, so a
// test whose patch is missing aborts.
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
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

#include <memory>
#include <string>

extern "C" void LLVMInitializeSPIRVTargetInfo();
extern "C" void LLVMInitializeSPIRVTarget();
extern "C" void LLVMInitializeSPIRVTargetMC();
extern "C" void LLVMInitializeSPIRVAsmPrinter();

using namespace llvm;

namespace {

std::unique_ptr<Module> parse(LLVMContext &ctx, const char *text) {
  SMDiagnostic diag;
  std::unique_ptr<Module> module = parseAssemblyString(text, diag, ctx);
  if (!module) {
    std::string message;
    raw_string_ostream os(message);
    diag.print("test", os);
    ADD_FAILURE() << message;
  }
  return module;
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
  // The default optimization level, so that CodeGenPrepare runs.
  std::unique_ptr<TargetMachine> machine(target->createTargetMachine(
      triple, "", "", TargetOptions(), std::nullopt));
  module.setDataLayout(machine->createDataLayout());
  SmallString<0> buffer;
  raw_svector_ostream os(buffer);
  legacy::PassManager passes;
  EXPECT_FALSE(machine->addPassesToEmitFile(passes, os, nullptr,
                                            CodeGenFileType::AssemblyFile));
  passes.run(module);
  return std::string(buffer.str());
}

// llvm-spirv-ptr-add-on-cast.patch, reduced from MAX's
// test_metal_memory: CodeGenPrepare sinks the load's address into %b as the
// constant getelementptr (i8, addrspacecast (@g to ptr addrspace(4)), 24),
// a G_PTR_ADD whose base is a cast, not a global variable. The instruction
// selector asserted that it is one.
TEST(SPIRVBackendPatchTest, ConstantOffsetFromACastOfAGlobal) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
target triple = "spirv64-unknown-unknown"
@g = external addrspace(3) global [8 x float]
define spir_kernel void @k(ptr addrspace(1) %out) {
  br label %b
b:
  %v = load float, ptr addrspace(4) inttoptr (i64 add (i64 ptrtoint (ptr addrspace(3) @g to i64), i64 24) to ptr addrspace(4)), align 4
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  ASSERT_FALSE(verifyModule(*module, &errs()));
  std::string spirv = emitSPIRV(*module);
  // The cast and the access chain are instructions in the function: an
  // OpSpecConstantOp can't use the cast's result.
  EXPECT_NE(spirv.find("OpPtrCastToGeneric"), std::string::npos) << spirv;
  EXPECT_NE(spirv.find("= OpPtrAccessChain"), std::string::npos) << spirv;
  EXPECT_EQ(spirv.find("OpSpecConstantOp"), std::string::npos) << spirv;
  EXPECT_NE(spirv.find("OpLoad"), std::string::npos) << spirv;
}

// The same with the cast's pointee type equal to the load's (@g an i8).
// Without assertions the unpatched selector built an OpSpecConstantOp on the
// cast's result, an instruction, which spirv-val rejects ("ID has not been
// defined").
TEST(SPIRVBackendPatchTest, ConstantOffsetFromACastOfAGlobalOfTheSameType) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
target triple = "spirv64-unknown-unknown"
@g = external addrspace(3) global i8
define spir_kernel void @k(ptr addrspace(1) %out) {
  br label %b
b:
  %v = load i8, ptr addrspace(4) inttoptr (i64 add (i64 ptrtoint (ptr addrspace(3) @g to i64), i64 24) to ptr addrspace(4)), align 4
  store i8 %v, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  ASSERT_FALSE(verifyModule(*module, &errs()));
  std::string spirv = emitSPIRV(*module);
  EXPECT_NE(spirv.find("OpPtrCastToGeneric"), std::string::npos) << spirv;
  EXPECT_NE(spirv.find("= OpPtrAccessChain"), std::string::npos) << spirv;
  EXPECT_EQ(spirv.find("OpSpecConstantOp"), std::string::npos) << spirv;
  EXPECT_NE(spirv.find("OpLoad"), std::string::npos) << spirv;
}

} // namespace
