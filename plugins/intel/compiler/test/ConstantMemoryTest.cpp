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
// Tests for defining the constant-space globals a kernel imports.
//
//===----------------------------------------------------------------------===//

#include "ConstantMemory.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

using namespace llvm;
using M::KGEN::Intel::defineConstantImports;

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

void expectDefined(const GlobalVariable *global) {
  ASSERT_NE(global, nullptr);
  EXPECT_FALSE(global->isDeclaration());
  EXPECT_TRUE(global->hasExternalLinkage());
  EXPECT_FALSE(global->isConstant());
  EXPECT_TRUE(global->isExternallyInitialized());
  EXPECT_EQ(global->getAddressSpace(), 2u);
  EXPECT_TRUE(global->getInitializer()->isNullValue());
}

// MAX's test_external_constant_mem: a named constant-space array of 16
// floats, read at the work-item's index. Mojo emits it as an internal global
// with an undef initializer (address spaces already in SPIR-V numbering).
TEST(ConstantMemoryTest, DefinesANamedTableExported) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@static_constant = internal addrspace(2) global [16 x float] undef, align 8

define spir_kernel void @k(ptr addrspace(1) %out, i64 %i) {
  %p = getelementptr inbounds float, ptr addrspace(2) @static_constant, i64 %i
  %v = load float, ptr addrspace(2) %p, align 4
  %q = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i
  store float %v, ptr addrspace(1) %q, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(defineConstantImports(*module));
  EXPECT_FALSE(verifyModule(*module, &errs()));
  GlobalVariable *table = module->getNamedGlobal("static_constant");
  expectDefined(table);
  EXPECT_EQ(table->getValueType(), ArrayType::get(Type::getFloatTy(ctx), 16));
  EXPECT_EQ(table->getAlign(), Align(8));
  EXPECT_TRUE(isa<ConstantAggregateZero>(table->getInitializer()));
  EXPECT_NE(print(*module).find(
                "@static_constant = addrspace(2) externally_initialized "
                "global [16 x float] "
                "zeroinitializer, align 8"),
            std::string::npos)
      << print(*module);
  // Running again changes nothing.
  EXPECT_FALSE(defineConstantImports(*module));
}

// A declaration is defined the same way.
TEST(ConstantMemoryTest, DefinesADeclaration) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@table = external addrspace(2) global [4 x i32], align 4

define spir_kernel void @k(ptr addrspace(1) %out) {
  %v = load i32, ptr addrspace(2) @table, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(defineConstantImports(*module));
  EXPECT_FALSE(verifyModule(*module, &errs()));
  expectDefined(module->getNamedGlobal("table"));
}

// A nameless one can't be exported: it gets zeros and keeps its linkage.
TEST(ConstantMemoryTest, ZeroesANamelessOne) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@0 = internal addrspace(2) global [4 x i32] poison, align 4

define spir_kernel void @k(ptr addrspace(1) %out) {
  %v = load i32, ptr addrspace(2) @0, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(defineConstantImports(*module));
  EXPECT_FALSE(verifyModule(*module, &errs()));
  ASSERT_EQ(module->global_size(), 1u);
  GlobalVariable &global = *module->global_begin();
  EXPECT_FALSE(global.hasName());
  EXPECT_TRUE(global.hasInternalLinkage());
  EXPECT_TRUE(isa<ConstantAggregateZero>(global.getInitializer()));
  EXPECT_FALSE(global.isConstant());
}

// A floating-point scalar becomes a one-element array: its zero would be a
// new APFloat. Integer, vector and struct globals keep their types.
TEST(ConstantMemoryTest, DefinesScalarsVectorsAndStructs) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@f = internal addrspace(2) global float undef, align 4
@n = internal addrspace(2) global i32 undef, align 4
@v = external addrspace(2) global <4 x half>, align 8
@s = external addrspace(2) global { i64, double }, align 8

define spir_kernel void @k(ptr addrspace(1) %out) {
  %f = load float, ptr addrspace(2) @f, align 4
  store float %f, ptr addrspace(1) %out, align 4
  %n = load i32, ptr addrspace(2) @n, align 4
  store i32 %n, ptr addrspace(1) %out, align 4
  %v = load <4 x half>, ptr addrspace(2) @v, align 8
  store <4 x half> %v, ptr addrspace(1) %out, align 8
  %s = load i64, ptr addrspace(2) @s, align 8
  store i64 %s, ptr addrspace(1) %out, align 8
  ret void
}
)");
  ASSERT_TRUE(module);
  EXPECT_TRUE(defineConstantImports(*module));
  EXPECT_FALSE(verifyModule(*module, &errs()));
  GlobalVariable *f = module->getNamedGlobal("f");
  expectDefined(f);
  EXPECT_EQ(f->getValueType(), ArrayType::get(Type::getFloatTy(ctx), 1));
  EXPECT_TRUE(isa<ConstantAggregateZero>(f->getInitializer()));
  expectDefined(module->getNamedGlobal("n"));
  EXPECT_EQ(module->getNamedGlobal("n")->getValueType(), Type::getInt32Ty(ctx));
  expectDefined(module->getNamedGlobal("v"));
  EXPECT_TRUE(isa<ConstantAggregateZero>(
      module->getNamedGlobal("v")->getInitializer()));
  expectDefined(module->getNamedGlobal("s"));
  EXPECT_TRUE(isa<ConstantAggregateZero>(
      module->getNamedGlobal("s")->getInitializer()));
}

// Constants with values, unused globals and globals in other address spaces
// are left alone.
TEST(ConstantMemoryTest, LeavesOtherGlobalsAlone) {
  LLVMContext ctx;
  std::unique_ptr<Module> module = parse(ctx, R"(
@table = internal addrspace(2) constant [2 x i32] [i32 1, i32 2], align 4
@unused = internal addrspace(2) global [4 x i32] undef, align 4
@unused_import = external addrspace(2) global [4 x i32], align 4
@shared = internal addrspace(3) global [4 x i32] undef, align 4
@global = external addrspace(1) global i32, align 4

define spir_kernel void @k(ptr addrspace(1) %out) {
  %a = load i32, ptr addrspace(2) @table, align 4
  %b = load i32, ptr addrspace(3) @shared, align 4
  %c = load i32, ptr addrspace(1) @global, align 4
  %ab = add i32 %a, %b
  %abc = add i32 %ab, %c
  store i32 %abc, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_TRUE(module);
  std::string before = print(*module);
  EXPECT_FALSE(defineConstantImports(*module));
  EXPECT_EQ(print(*module), before);
}

} // namespace
