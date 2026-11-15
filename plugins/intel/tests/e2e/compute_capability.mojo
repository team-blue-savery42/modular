# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Sean Avery. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# DeviceContext.compute_capability() asks the driver for
# CUDA's device attributes 75 (COMPUTE_CAPABILITY_MAJOR) and 76 (_MINOR) and
# returns major * 10 + minor for every API but Metal. The Level Zero plugin
# answers the IP version's architecture and release, so the B70
# (bmg-g31, IP 20.2) reports 202, matching the target's `compute=20.2` the
# way an NVIDIA GPU with `compute=8.6` reports 86. Checked here against the
# target's compile-time GPUInfo.
# run-e2e: device-context

from max.gpu.host import DeviceAttribute, DeviceContext
from max.gpu.host.info import get_gpu_info


def main() raises:
    var ctx = DeviceContext()
    var major = ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MAJOR)
    var minor = ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MINOR)
    var cc = ctx.compute_capability()
    print("api:", ctx.api())
    print("major:", major, "minor:", minor)
    print("compute_capability:", cc)
    comptime info = get_gpu_info()
    var expected = Int(info.compute * 10 + 0.5)
    print("target", info.name, "compute:", info.compute, "expected:", expected)
    if cc == expected and cc == major * 10 + minor:
        print("PASS compute_capability", cc)
    else:
        print("FAIL compute_capability", cc, "expected", expected)
