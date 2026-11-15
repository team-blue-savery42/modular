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
// The compute capability over IP versions no installed GPU has: the B70
// (20.2) reaches only one branch of computeCapability, so its guard, and the
// device attribute's unsupported status for it (Limits.cpp), are checked
// here. Needs no GPU.
//
//===----------------------------------------------------------------------===//

#include "ComputeCapability.h"

#include "gtest/gtest.h"

#include <cstdint>

using M::Intel::computeCapability;
using M::Intel::computeCapabilityAttribute;

namespace {

/// A Level Zero IP version: architecture in bits 22-31, release in 14-21,
/// revision in 0-5 (ze_device_ip_version_ext_t, as Driver.cpp decodes it).
constexpr uint32_t ipVersion(uint32_t architecture, uint32_t release,
                             uint32_t revision) {
  return architecture << 22 | release << 14 | revision;
}

struct Case {
  const char *what;
  uint32_t ipVersion;
  bool answered;
  int32_t major, minor;
};

TEST(ComputeCapabilityTest, AnswersTheArchitectureAndRelease) {
  const Case cases[] = {
      {"no IP version", 0, false, 0, 0},
      {"pre-Xe 11.2", ipVersion(11, 2, 0), true, 11, 2},
      {"DG2 12.55, a two-digit release", ipVersion(12, 55, 8), true, 12, 5},
      {"arl-s 12.70", ipVersion(12, 70, 4), true, 12, 7},
      {"12.10, the first two-digit release", ipVersion(12, 10, 0), true, 12, 1},
      {"12.9, the last one-digit release", ipVersion(12, 9, 0), true, 12, 9},
      {"rpl-p 12.3", ipVersion(12, 3, 0), true, 12, 3},
      {"bmg-g31 20.2", ipVersion(20, 2, 4), true, 20, 2},
      {"a future 30.0", ipVersion(30, 0, 0), true, 30, 0},
      {"the revision is ignored", ipVersion(20, 1, 63), true, 20, 1},
  };
  for (const Case &c : cases) {
    int32_t major = -1, minor = -1;
    bool answered = computeCapability(c.ipVersion, major, minor);
    EXPECT_EQ(answered, c.answered) << c.what;
    if (c.answered && answered) {
      EXPECT_EQ(major, c.major) << c.what;
      EXPECT_EQ(minor, c.minor) << c.what;
    }
  }
}

TEST(ComputeCapabilityTest, AttributeIsUnsupportedWithoutAnAnswer) {
  const Case cases[] = {
      {"no IP version", 0, false, 0, 0},
      {"arl-s 12.70", ipVersion(12, 70, 4), true, 12, 7},
      {"bmg-g31 20.2", ipVersion(20, 2, 4), true, 20, 2},
  };
  for (const Case &c : cases) {
    for (bool wantMajor : {true, false}) {
      int32_t value = -1;
      M_driver_status_t status =
          computeCapabilityAttribute(c.ipVersion, wantMajor, &value);
      if (c.answered) {
        EXPECT_EQ(status, M_DRIVER_STATUS_SUCCESS) << c.what;
        EXPECT_EQ(value, wantMajor ? c.major : c.minor) << c.what;
      } else {
        EXPECT_EQ(status, M_DRIVER_STATUS_UNSUPPORTED) << c.what;
        EXPECT_EQ(value, -1) << c.what << ": value set";
      }
    }
  }
}

} // namespace
