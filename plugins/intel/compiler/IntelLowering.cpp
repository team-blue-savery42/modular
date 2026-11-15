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
// KGEN/POP -> LLVM-dialect lowering policy for Intel SPIR-V targets.
//
//===----------------------------------------------------------------------===//

#include "IntelLowering.h"

#include "IntelTraits.h"

#include "Target/TargetLowering.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace M::KGEN::Intel {
namespace {

/// The attribute MAX's MAX_THREADS_PER_BLOCK_METADATA names for Intel
/// targets (max/gpu/globals.mojo).
constexpr llvm::StringLiteral kMaxWorkGroupSize =
    "pop.intel.max_work_group_size";

class IntelLowering final : public TargetLowering {
public:
  const TargetTraits *traits() const override { return &IntelTraits::get(); }

  void markExportedKernel(mlir::Operation *op) const override {
    auto func = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(op);
    if (!func) {
      trace("lowering: markExportedKernel: op is not an LLVMFuncOp to the "
            "plugin");
      return;
    }
    // A SPIR-V kernel returns nothing. An exported function that returns a
    // value (compile_info of a helper, for example) can't be a kernel entry
    // point and stays an exported function.
    if (!mlir::isa<mlir::LLVM::LLVMVoidType>(
            func.getFunctionType().getReturnType())) {
      trace("lowering: markExportedKernel: returns a value, not a kernel");
      return;
    }
    func.setCConv(mlir::LLVM::cconv::CConv::SPIR_KERNEL);
    trace("lowering: markExportedKernel: set spir_kernel");
  }

  /// MAX marks kernels with their maximum block size
  /// (`MAX_THREADS_PER_BLOCK_METADATA`, `pop.intel.max_work_group_size` for
  /// Intel). It is a hint, and SPIR-V's form of it (MaxWorkgroupSizeINTEL)
  /// belongs to an FPGA extension, so it is accepted and dropped for now.
  mlir::LogicalResult mapFuncMetadata(mlir::Operation *,
                                      mlir::NamedAttribute attr,
                                      mlir::NamedAttrList &,
                                      llvm::SmallVectorImpl<mlir::Attribute> &,
                                      bool &handled) const override {
    handled = attr.getName() == kMaxWorkGroupSize;
    return mlir::success();
  }

  bool isExportedKernel(mlir::Operation *op) const override {
    auto func = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(op);
    return func && func.getCConv() == mlir::LLVM::cconv::CConv::SPIR_KERNEL;
  }

  /// A kernel argument passed in memory (a borrowed struct that isn't
  /// register-passable, as MAX's graph kernels take) is `byval`, as clang
  /// passes an OpenCL kernel's struct argument for SPIR: the kernel gets a
  /// private copy and the host sets the struct's bytes as the argument.
  llvm::StringRef getKernelByValArgAttrName() const override {
    return mlir::LLVM::LLVMDialect::getByValAttrName();
  }

protected:
  bool isBaseTarget() const override { return false; }
};

} // namespace

const TargetLowering &getLowering() {
  static const IntelLowering instance;
  return instance;
}

} // namespace M::KGEN::Intel
