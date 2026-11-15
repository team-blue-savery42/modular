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
// The compute capability attributes through the plugin's entry point
// (Limits.cpp) for IP versions no installed GPU has. ComputeCapabilityTest
// checks the header's arithmetic and status; this checks that the entry point
// returns that status and leaves the value alone when there is no answer. The
// B70 (20.2) reaches only the answered branch, so the test sets the device
// object's IP version (Plugin.h; the plugin built from the same header) and
// restores it.
//
//===----------------------------------------------------------------------===//

#include "Target/test/Fixture.h"

#include "plugins/intel/driver/Plugin.h"

#include <cstdint>

namespace {

/// A Level Zero IP version: architecture in bits 22-31, release in 14-21,
/// revision in 0-5.
constexpr uint32_t ipVersion(uint32_t architecture, uint32_t release,
                             uint32_t revision) {
  return architecture << 22 | release << 14 | revision;
}

/// Puts a device's IP version back however the test leaves.
struct RestoreIpVersion {
  M_driver_device_t device;
  uint32_t saved;
  ~RestoreIpVersion() { device->ipVersion = saved; }
};

TEST_F(DriverPluginDeviceTest, ComputeCapabilityEntryPointStatus) {
  constexpr int32_t kUntouched = -7;
  for (M_driver_device_t device : devices) {
    RestoreIpVersion restore{device, device->ipVersion};
    device->ipVersion = ipVersion(0, 0, 0);
    for (int32_t id : {75, 76}) {
      int32_t value = kUntouched;
      EXPECT_EQ(device_attribute(device, id, &value),
                M_DRIVER_STATUS_UNSUPPORTED)
          << "attribute " << id;
      EXPECT_EQ(value, kUntouched) << "attribute " << id;
    }
    device->ipVersion = ipVersion(12, 9, 0);
    int32_t major = kUntouched, minor = kUntouched;
    ASSERT_SUCCESS(device_attribute(device, 75, &major));
    ASSERT_SUCCESS(device_attribute(device, 76, &minor));
    EXPECT_EQ(major, 12);
    EXPECT_EQ(minor, 9);
    // A two-digit release keeps its first digit.
    device->ipVersion = ipVersion(12, 70, 4);
    ASSERT_SUCCESS(device_attribute(device, 75, &major));
    ASSERT_SUCCESS(device_attribute(device, 76, &minor));
    EXPECT_EQ(major, 12);
    EXPECT_EQ(minor, 7);
  }
}

} // namespace
