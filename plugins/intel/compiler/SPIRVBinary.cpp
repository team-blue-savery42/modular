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

#include "SPIRVBinary.h"

#include <cstdint>
#include <cstring>

namespace M::KGEN::Intel {
namespace {

struct Rename {
  const char *from;
  const char *to;
};

constexpr Rename kRenames[] = {
    {"SPV_ALTERA_arbitrary_precision_integers",
     "SPV_INTEL_arbitrary_precision_integers"},
};

/// Words a nul-terminated literal string of `length` characters takes.
size_t wordsFor(size_t length) { return (length + 1 + 3) / 4; }

} // namespace

unsigned renameExtensions(char *data, size_t size) {
  constexpr uint32_t kOpExtension = 10;
  unsigned renamed = 0;
  size_t words = size / 4;
  // Instructions follow the five-word header; each starts with its word
  // count (high half) and opcode (low half).
  for (size_t i = 5; i < words;) {
    uint32_t word;
    std::memcpy(&word, data + i * 4, 4);
    size_t count = word >> 16;
    if (count == 0 || i + count > words)
      break;
    if ((word & 0xFFFF) == kOpExtension) {
      char *name = data + (i + 1) * 4;
      size_t space = (count - 1) * 4;
      for (const Rename &rename : kRenames) {
        size_t fromLength = std::strlen(rename.from);
        size_t toLength = std::strlen(rename.to);
        if (fromLength + 1 > space ||
            std::memcmp(name, rename.from, fromLength + 1) != 0)
          continue;
        // Only a rename that keeps the word count fits in place.
        if (wordsFor(toLength) != count - 1)
          continue;
        std::memset(name, 0, space);
        std::memcpy(name, rename.to, toLength);
        ++renamed;
      }
    }
    i += count;
  }
  return renamed;
}

} // namespace M::KGEN::Intel
