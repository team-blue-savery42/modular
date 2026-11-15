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
// The compute capability for a Level Zero IP version, and its
// device attribute's status. A header of its own so that a test can feed them
// IP versions no installed GPU has.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_DRIVER_COMPUTECAPABILITY_H
#define MOJO_INTEL_DRIVER_COMPUTECAPABILITY_H

#include "HAL/M_driver_common.h"

#include <cstdint>

namespace M::Intel {

/// The IP version's architecture and release, so that MAX's major * 10 +
/// minor is the target's `compute` times ten (bmg-g31, IP 20.2: 202), as an
/// NVIDIA sm_86 answers 8 and 6. A two-digit release keeps its first digit:
/// Arrow Lake-S (arl-s, IP 12.70) answers 12 and 7, as the target's
/// `compute` 12.7, and DG2 (12.55) 12 and 5; the second digit tells parts
/// of one generation apart, which the arch name does. False when Level
/// Zero gave no IP version (architecture 0). Any other architecture, pre-Xe
/// included, is answered as it is.
inline bool computeCapability(uint32_t ipVersion, int32_t &major,
                              int32_t &minor) {
  uint32_t architecture = ipVersion >> 22;
  uint32_t release = (ipVersion >> 14) & 0xff;
  if (architecture == 0)
    return false;
  major = static_cast<int32_t>(architecture);
  minor = static_cast<int32_t>(release > 9 ? release / 10 : release);
  return true;
}

/// The compute capability device attribute: `*value` is the major (or, with
/// `wantMajor` false, the minor) and the status success, or the status is
/// unsupported, leaving `*value` alone, when computeCapability can't answer.
inline M_driver_status_t
computeCapabilityAttribute(uint32_t ipVersion, bool wantMajor, int32_t *value) {
  int32_t major = 0, minor = 0;
  if (!computeCapability(ipVersion, major, minor))
    return M_DRIVER_STATUS_UNSUPPORTED;
  *value = wantMajor ? major : minor;
  return M_DRIVER_STATUS_SUCCESS;
}

} // namespace M::Intel

#endif // MOJO_INTEL_DRIVER_COMPUTECAPABILITY_H
