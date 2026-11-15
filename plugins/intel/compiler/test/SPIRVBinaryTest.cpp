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
// Tests for the SPIR-V binary fix-ups, on hand-assembled words.
//
//===----------------------------------------------------------------------===//

#include "SPIRVBinary.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using M::KGEN::Intel::renameExtensions;

namespace {

constexpr uint32_t kOpCapability = 17, kOpExtension = 10;

/// A module header followed by `OpCapability 5844` and one `OpExtension`
/// per name.
std::vector<uint32_t> module(const std::vector<std::string> &extensions) {
  std::vector<uint32_t> words = {0x07230203, 0x00010400, 0, 100, 0};
  words.push_back((2u << 16) | kOpCapability);
  words.push_back(5844);
  for (const std::string &name : extensions) {
    size_t nameWords = (name.size() + 1 + 3) / 4;
    words.push_back(static_cast<uint32_t>((1 + nameWords) << 16) |
                    kOpExtension);
    std::vector<uint32_t> packed(nameWords, 0);
    std::memcpy(packed.data(), name.data(), name.size());
    words.insert(words.end(), packed.begin(), packed.end());
  }
  return words;
}

std::string extensionAt(const std::vector<uint32_t> &words, size_t index) {
  return std::string(reinterpret_cast<const char *>(&words[index + 1]));
}

TEST(SPIRVBinaryTest, RenamesArbitraryPrecisionIntegers) {
  std::vector<uint32_t> words =
      module({"SPV_KHR_bfloat16", "SPV_ALTERA_arbitrary_precision_integers"});
  size_t bfloat = 7, arbitrary = 7 + 1 + 5;
  std::vector<uint32_t> before = words;
  EXPECT_EQ(renameExtensions(reinterpret_cast<char *>(words.data()),
                             words.size() * 4),
            1u);
  EXPECT_EQ(words.size(), before.size());
  EXPECT_EQ(extensionAt(words, bfloat), "SPV_KHR_bfloat16");
  EXPECT_EQ(extensionAt(words, arbitrary),
            "SPV_INTEL_arbitrary_precision_integers");
  // SPV_INTEL_ is a byte shorter than SPV_ALTERA_, so the name's padding
  // absorbs the rename: the instruction keeps its word count, and the
  // capability is untouched.
  EXPECT_EQ(words[arbitrary], before[arbitrary]);
  EXPECT_EQ(words[6], 5844u);
}

TEST(SPIRVBinaryTest, LeavesOtherModulesAlone) {
  std::vector<uint32_t> words = module({"SPV_KHR_bfloat16"});
  std::vector<uint32_t> before = words;
  EXPECT_EQ(renameExtensions(reinterpret_cast<char *>(words.data()),
                             words.size() * 4),
            0u);
  EXPECT_EQ(words, before);
}

} // namespace
