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
// Tests for SubgroupSize: each kernel's sub-group size comes from
// its target features, and the SPIR-V backend, run in this process, emits it
// as the kernel's `OpExecutionMode SubgroupSize`.
//
//===----------------------------------------------------------------------===//

#include "SubgroupSize.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Regex.h"
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
using namespace M::KGEN::Intel;

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

unsigned requiredSize(const Function &func) {
  MDNode *node = func.getMetadata("intel_reqd_sub_group_size");
  if (!node || node->getNumOperands() != 1)
    return 0;
  return mdconst::extract<ConstantInt>(node->getOperand(0))->getZExtValue();
}

/// The `SubgroupSize` execution mode of the entry point named `name` in the
/// SPIR-V assembly `spirv`, or 0. FileCheck-like: finds `%id = OpString`-free
/// `OpEntryPoint Kernel %id "name"`, then `OpExecutionMode %id SubgroupSize N`.
unsigned emittedSize(const std::string &spirv, StringRef name) {
  SmallVector<StringRef, 3> match;
  Regex entry(("OpEntryPoint Kernel (%[A-Za-z0-9_]+) \"" + name + "\"").str());
  if (!entry.match(spirv, &match))
    return 0;
  Regex mode(("OpExecutionMode " + match[1] + " SubgroupSize ([0-9]+)").str());
  SmallVector<StringRef, 2> size;
  if (!mode.match(spirv, &size))
    return 0;
  unsigned value = 0;
  size[1].getAsInteger(10, value);
  return value;
}

const char *kTwoKernels = R"(
define spir_kernel void @at16(ptr addrspace(1) %p) #0 {
  store i32 1, ptr addrspace(1) %p
  ret void
}
define spir_kernel void @at32(ptr addrspace(1) %p) #1 {
  store i32 2, ptr addrspace(1) %p
  ret void
}
define spir_func void @helper() #1 {
  ret void
}
attributes #0 = { "target-features"="" }
attributes #1 = { "target-features"="+sub-group-size-32" }
)";

TEST(SubgroupSizeTest, FeatureStrings) {
  EXPECT_EQ(subgroupSizeForFeatures(""), 16u);
  EXPECT_EQ(subgroupSizeForFeatures("+sub-group-size-32"), 32u);
  EXPECT_EQ(subgroupSizeForFeatures("+a,+sub-group-size-32,-b"), 32u);
  EXPECT_EQ(subgroupSizeForFeatures("-sub-group-size-32"), 16u);
  EXPECT_EQ(subgroupSizeForFeatures("+sub-group-size-32,-sub-group-size-32"),
            16u);
  EXPECT_EQ(subgroupSizeForFeatures("+sub-group-size-320"), 16u);
  EXPECT_EQ(withoutPluginFeatures("+a,+sub-group-size-32,-b"), "+a,-b");
  EXPECT_EQ(withoutPluginFeatures("+sub-group-size-32"), "");
  EXPECT_EQ(withoutPluginFeatures("+a"), "+a");
}

TEST(SubgroupSizeTest, EachKernelGetsItsTargetsSize) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, kTwoKernels);
  ASSERT_TRUE(module);
  EXPECT_TRUE(setSubgroupSizes(*module));
  EXPECT_EQ(requiredSize(*module->getFunction("at16")), 16u);
  EXPECT_EQ(requiredSize(*module->getFunction("at32")), 32u);
  // Not a kernel: no execution mode of its own.
  EXPECT_EQ(requiredSize(*module->getFunction("helper")), 0u);
  // The feature is gone before LLVM's backend sees it.
  for (const Function &func : *module)
    EXPECT_FALSE(func.getFnAttribute("target-features")
                     .getValueAsString()
                     .contains("sub-group-size"))
        << func.getName();
}

TEST(SubgroupSizeTest, DefaultModuleIsUnchangedButFor16) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
define spir_kernel void @k() #0 {
  ret void
}
attributes #0 = { "target-features"="+other" }
)");
  ASSERT_TRUE(module);
  EXPECT_FALSE(setSubgroupSizes(*module));
  const Function &k = *module->getFunction("k");
  EXPECT_EQ(requiredSize(k), 16u);
  EXPECT_EQ(k.getFnAttribute("target-features").getValueAsString(), "+other");
}

TEST(SubgroupSizeTest, ExecutionModeInTheSPIRV) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, kTwoKernels);
  ASSERT_TRUE(module);
  setSubgroupSizes(*module);
  std::string spirv = emitSPIRV(*module);
  EXPECT_EQ(emittedSize(spirv, "at16"), 16u) << spirv;
  EXPECT_EQ(emittedSize(spirv, "at32"), 32u) << spirv;
}

} // namespace
