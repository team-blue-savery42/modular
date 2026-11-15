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

#include "SubgroupSize.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

namespace M::KGEN::Intel {
namespace {

bool isPluginFeature(llvm::StringRef name) {
  return name == kSubgroupSize32Feature;
}

} // namespace

unsigned subgroupSizeForFeatures(llvm::StringRef features) {
  llvm::SmallVector<llvm::StringRef, 8> tokens;
  features.split(tokens, ",", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  unsigned size = kDefaultSubgroupSize;
  // The last mention wins, as in LLVM's feature strings.
  for (llvm::StringRef token : tokens) {
    token = token.trim();
    if (token.size() > 1 && token.drop_front() == kSubgroupSize32Feature)
      size = token.front() == '+' ? 32 : kDefaultSubgroupSize;
  }
  return size;
}

std::string withoutPluginFeatures(llvm::StringRef features) {
  llvm::SmallVector<llvm::StringRef, 8> tokens;
  features.split(tokens, ",", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  std::string kept;
  for (llvm::StringRef token : tokens) {
    llvm::StringRef trimmed = token.trim();
    if (trimmed.size() > 1 &&
        (trimmed.front() == '+' || trimmed.front() == '-') &&
        isPluginFeature(trimmed.drop_front()))
      continue;
    if (!kept.empty())
      kept += ',';
    kept += trimmed.str();
  }
  return kept;
}

bool setSubgroupSizes(llvm::Module &module) {
  llvm::LLVMContext &ctx = module.getContext();
  bool nonDefault = false;
  for (llvm::Function &func : module) {
    llvm::StringRef features;
    if (func.hasFnAttribute("target-features"))
      features = func.getFnAttribute("target-features").getValueAsString();
    if (func.getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
      unsigned size = subgroupSizeForFeatures(features);
      nonDefault |= size != kDefaultSubgroupSize;
      func.setMetadata(
          "intel_reqd_sub_group_size",
          llvm::MDNode::get(
              ctx, llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                       llvm::Type::getInt32Ty(ctx), size))));
    }
    if (!features.empty()) {
      std::string kept = withoutPluginFeatures(features);
      if (kept.empty())
        func.removeFnAttr("target-features");
      else if (kept != features)
        func.addFnAttr("target-features", kept);
    }
  }
  return nonDefault;
}

} // namespace M::KGEN::Intel
