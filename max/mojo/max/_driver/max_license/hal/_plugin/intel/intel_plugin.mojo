# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from std.sys.info import CompilationTarget
from .._trait import HalPluginHooks
from ..._machine import DeviceRef
from ...execution_config.gpu_execution_config import GPUExecutionConfiguration
from ...extensions.intel import IntelHALExtensions


struct IntelHalPluginHooks[target: CompilationTarget](HalPluginHooks):
    # Grids and blocks, plus the dynamic shared memory size the driver plugin
    # passes as the kernel's trailing local-memory argument. One shape for
    # every Xe2 part until a launch needs per-product limits.
    comptime ExecutionConfigType[device: DeviceRef] = GPUExecutionConfiguration
    """Returns the execution configuration for the given device."""

    # Intel device features (`SupportsIntelDeviceFeaturesV1`). No peer access:
    # Level Zero has it, the driver plugin doesn't yet.
    comptime extensions = IntelHALExtensions
