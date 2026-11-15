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
// Tests for the Intel target traits, the Mojo -> SPIR-V address-space table,
// and the assumptions both make about LLVM's SPIR-V target machine.
//
//===----------------------------------------------------------------------===//

#include "IntelTraits.h"
#include "IntelAddressSpace.h"

#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

extern "C" void LLVMInitializeSPIRVTargetInfo();
extern "C" void LLVMInitializeSPIRVTarget();
extern "C" void LLVMInitializeSPIRVTargetMC();

using namespace M::KGEN;
using namespace M::KGEN::Intel;

namespace {

constexpr const char *kTriple = "spirv64-unknown-unknown";

unsigned spirv(SPIRVAddressSpace as) { return static_cast<unsigned>(as); }

TEST(IntelTraitsTest, MatchesOnlySpirv64) {
  const IntelTraits &traits = IntelTraits::get();
  EXPECT_TRUE(traits.matches(llvm::Triple(kTriple)));
  EXPECT_TRUE(traits.matches(llvm::Triple("spirv64-intel-unknown")));
  EXPECT_FALSE(traits.matches(llvm::Triple("spirv32-unknown-unknown")));
  EXPECT_FALSE(traits.matches(llvm::Triple("spirv-unknown-vulkan1.3")));
  EXPECT_FALSE(traits.matches(llvm::Triple("nvptx64-nvidia-cuda")));
  EXPECT_FALSE(traits.matches(llvm::Triple("amdgcn-amd-amdhsa")));
  EXPECT_FALSE(traits.matches(llvm::Triple("x86_64-unknown-linux-gnu")));
}

TEST(IntelTraitsTest, Metadata) {
  const IntelTraits &traits = IntelTraits::get();
  EXPECT_EQ(traits.name(), "intel");
  EXPECT_TRUE(traits.isGPU());
  EXPECT_TRUE(traits.emitsOffloadObjectFile());
  EXPECT_EQ(traits.getAsmExtension(), ".spvasm");
  EXPECT_EQ(traits.getLLVMExtension(), ".spirv.ll");
  EXPECT_EQ(traits.getObjectExtension(), ".spv");
  EXPECT_EQ(traits.getBitcodeExtension(), ".bc");
  EXPECT_EQ(traits.codegenTriple(kTriple), kTriple);
  EXPECT_EQ(traits.forcedBitcodeVersion(), 0u);
  EXPECT_TRUE(traits.defaultCPU(llvm::Triple(kTriple)).empty());
  EXPECT_TRUE(traits.supportsEmissionKind("asm"));
  EXPECT_TRUE(traits.supportsEmissionKind("object"));
}

TEST(IntelTraitsTest, AcceleratorArchs) {
  const IntelTraits &traits = IntelTraits::get();
  EXPECT_FALSE(traits.acceleratorSectionTitle().empty());
  bool hasB70 = false;
  for (const TargetTraits::AcceleratorArch &arch :
       traits.supportedAcceleratorArchs())
    hasB70 |= arch.arch == "bmg-g31";
  EXPECT_TRUE(hasB70);
}

TEST(IntelTraitsTest, StackAllocationsAreMojoLocal) {
  std::optional<unsigned> stack =
      IntelTraits::get().requiredStackAllocationAddressSpace();
  ASSERT_TRUE(stack.has_value());
  EXPECT_EQ(*stack, static_cast<unsigned>(MojoAddressSpace::Local));
  EXPECT_EQ(toSPIRVAddressSpace(*stack), SPIRVAddressSpace::Function);
}

TEST(IntelAddressSpaceTest, MojoToSPIRV) {
  EXPECT_EQ(toSPIRVAddressSpace(0), SPIRVAddressSpace::Generic);
  EXPECT_EQ(toSPIRVAddressSpace(1), SPIRVAddressSpace::CrossWorkgroup);
  EXPECT_EQ(toSPIRVAddressSpace(3), SPIRVAddressSpace::Workgroup);
  EXPECT_EQ(toSPIRVAddressSpace(4), SPIRVAddressSpace::UniformConstant);
  EXPECT_EQ(toSPIRVAddressSpace(5), SPIRVAddressSpace::Function);
  // No Intel equivalent, or not a Mojo address space.
  EXPECT_EQ(toSPIRVAddressSpace(2), std::nullopt);
  EXPECT_EQ(toSPIRVAddressSpace(6), std::nullopt);
  EXPECT_EQ(toSPIRVAddressSpace(7), std::nullopt);
  EXPECT_EQ(toSPIRVAddressSpace(8), std::nullopt);
  EXPECT_EQ(toSPIRVAddressSpace(42), std::nullopt);
}

TEST(IntelAddressSpaceTest, MappingIsInjective) {
  // Renumbering must never merge two Mojo address spaces into one.
  bool seen[5] = {};
  for (unsigned mojo = 0; mojo < 16; ++mojo) {
    std::optional<SPIRVAddressSpace> as = toSPIRVAddressSpace(mojo);
    if (!as)
      continue;
    unsigned index = spirv(*as);
    ASSERT_LT(index, 5u);
    EXPECT_FALSE(seen[index]) << "two Mojo address spaces map to " << index;
    seen[index] = true;
  }
}

// The table and the traits assume facts about LLVM's SPIR-V target machine;
// check them against the target built into this tree.
class SPIRVTargetMachineTest : public ::testing::Test {
protected:
  void SetUp() override {
    LLVMInitializeSPIRVTargetInfo();
    LLVMInitializeSPIRVTarget();
    LLVMInitializeSPIRVTargetMC();
    std::string error;
    llvm::Triple triple(kTriple);
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget(triple, error);
    ASSERT_NE(target, nullptr) << error;
    machine.reset(target->createTargetMachine(
        triple, /*CPU=*/"", /*Features=*/"", llvm::TargetOptions(),
        /*RM=*/std::nullopt));
    ASSERT_NE(machine, nullptr);
  }

  std::unique_ptr<llvm::TargetMachine> machine;
};

TEST_F(SPIRVTargetMachineTest, DataLayoutMatchesTheTable) {
  llvm::DataLayout layout = machine->createDataLayout();
  EXPECT_TRUE(layout.isLittleEndian());
  // Stack allocations land in the alloca address space, which must be where
  // Mojo's LOCAL maps.
  EXPECT_EQ(layout.getAllocaAddrSpace(), spirv(SPIRVAddressSpace::Function));
  // Globals default to CrossWorkgroup (`G1`), where Mojo's GLOBAL maps.
  EXPECT_EQ(layout.getDefaultGlobalsAddressSpace(),
            spirv(SPIRVAddressSpace::CrossWorkgroup));
  // Every address space we map to uses 64-bit pointers (Physical64).
  for (unsigned mojo : {0u, 1u, 3u, 4u, 5u}) {
    std::optional<SPIRVAddressSpace> as = toSPIRVAddressSpace(mojo);
    ASSERT_TRUE(as.has_value());
    EXPECT_EQ(layout.getPointerSizeInBits(spirv(*as)), 64u)
        << "SPIR-V address space " << spirv(*as);
  }
}

TEST_F(SPIRVTargetMachineTest, KnowsNoCPUNames) {
  // Why `defaultCPU` stays empty: the SPIR-V backend has no processor table.
  EXPECT_FALSE(machine->getMCSubtargetInfo().isCPUStringValid("bmg-g31"));
}

} // namespace
