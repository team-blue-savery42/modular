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
// Tests for splitting comma-separated emission options.
//
//===----------------------------------------------------------------------===//

#include "Mojo/ToolCommon/KGENPasses.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>

namespace {

std::vector<std::string> split(llvm::StringRef options) {
  llvm::SmallVector<llvm::StringRef> out;
  M::KGEN::splitEmissionOptions(options, out);
  return std::vector<std::string>(out.begin(), out.end());
}

using Options = std::vector<std::string>;

TEST(EmissionOptionsTest, SplitsOnCommas) {
  EXPECT_EQ(split("target-abi=lp64d,some-opt"),
            (Options{"target-abi=lp64d", "some-opt"}));
  EXPECT_EQ(split("a=1,,b=2"), (Options{"a=1", "b=2"}));
  EXPECT_EQ(split(""), Options{});
}

TEST(EmissionOptionsTest, ListValuesStayOneOption) {
  EXPECT_EQ(split("spirv-ext=+SPV_A,+SPV_B,-SPV_C,target-abi=x"),
            (Options{"spirv-ext=+SPV_A,+SPV_B,-SPV_C", "target-abi=x"}));
  // A flag has no value to continue.
  EXPECT_EQ(split("flag,+x"), (Options{"flag", "+x"}));
}

} // namespace
