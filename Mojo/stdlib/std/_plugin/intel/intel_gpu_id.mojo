# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

# ===----------------------------------------------------------------------=== #
# ===----------------------------------------------------------------------=== #
"""GPU indices and sizes for Intel GPUs, read through the OpenCL work-item
built-ins, which LLVM's SPIR-V backend turns into SPIR-V `BuiltIn` variables.
"""

from std._gpu.primitives.id import _GPUIdPluginHookFnType, _is_cluster_id
from std.collections import OptionalReg
from std.ffi import external_call


def _intel_gpu_id_fn[
    name: StaticString, dim: StaticString
]() -> OptionalReg[_GPUIdPluginHookFnType]:
    """Returns the OpenCL read of `name` along `dim`, or `None` for the
    thread block cluster ids: Intel GPUs have no clusters.
    """
    comptime if _is_cluster_id(name):
        return None
    else:
        return comptime (
            OptionalReg[_GPUIdPluginHookFnType](_intel_gpu_id[name, dim])
        )


@inline(.always)
def _work_item_fn[builtin: StaticString, dim: StaticString]() -> Int:
    """Calls the OpenCL built-in `size_t builtin(uint dim)` by its mangled
    name, which is what LLVM's SPIR-V backend recognizes."""
    comptime index = 0 if dim == "x" else (1 if dim == "y" else 2)
    return Int(external_call[builtin, UInt64](UInt32(index)))


@inline(.always)
def _intel_gpu_id[name: StaticString, dim: StaticString]() -> Int:
    """Reads the GPU index or size `name` along `dim`.

    CUDA's thread, block, block size and grid size are OpenCL's local id,
    group id, local size and number of groups; a lane is a sub-group
    invocation.
    """
    comptime if name == "lane_id":
        return Int(external_call["_Z22get_sub_group_local_idv", UInt32]())
    elif name == "thread_idx":
        return _work_item_fn["_Z12get_local_idj", dim]()
    elif name == "block_idx":
        return _work_item_fn["_Z12get_group_idj", dim]()
    elif name == "block_dim":
        return _work_item_fn["_Z14get_local_sizej", dim]()
    else:
        comptime assert name == "grid_dim", "unknown GPU id"
        return _work_item_fn["_Z14get_num_groupsj", dim]()
