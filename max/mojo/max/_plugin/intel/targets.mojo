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
"""Intel GPU targets, each carrying the Intel stdlib plugin.

The triple is `spirv64-unknown-unknown`; `arch` is the `ocloc -device` name
the Level Zero driver plugin reports. The data layout must equal LLVM's
SPIR-V target machine's exactly: the Intel compiler plugin keeps it after
renumbering address spaces for SPIR-V.

Kernels run at sub-group size 16. Each target has a variant whose features
have `+sub-group-size-32` (`--target-accelerator=intel:bmg-g31-sg32`, or
`compile_info[kernel, target=ArcProB70SG32.target]` for one kernel): its
kernels run at 32 and its `WARP_SIZE` is 32 (the compiler plugin's
`SubgroupSize.h` and `IntelMaxPluginHooks` read the feature). Its `arch` is the
base target's, the device `ocloc` and the driver see. Kernels using DPAS stay
at 16 (at 32 `ocloc` crashes).
"""

from std.sys import CompilationTarget
from std._gpu.host.info import (
    AcceleratorArchitectureFamily,
    GPUInfo,
    TargetAccelerator,
    TargetAcceleratorCollection,
    TargetAcceleratorType,
    _K,
    _KB,
    _supported_archs_line,
)
from std._plugin.intel import IntelPlugin

from . import IntelMaxPlugin

comptime INTEL_PLUGINS = TypeList.of[IntelPlugin, IntelMaxPlugin]().values
"""The `opaque_plugin` payload of every Intel target."""


struct IntelTargets(TargetAcceleratorCollection):
    """Intel GPU targets that the Intel plugins contribute."""

    comptime vendor_name: String = "Intel"

    # fmt: off
    comptime RAW_TARGETS = TypeList.of[
        Trait=TargetAcceleratorType,
        ArcProB70    ._with_cli_values[["bmg-g31"]],
        ArcB580      ._with_cli_values[["bmg-g21"]],
        ArcProB70SG32._with_cli_values[["bmg-g31-sg32"]],
        ArcB580SG32  ._with_cli_values[["bmg-g21-sg32"]],
    ]().values
    # fmt: on

    @staticmethod
    def normalize_target_arch(target_arch0: StaticString) -> String:
        # "intel:bmg-g31" -> "bmg-g31", and "level_zero:bmg-g31" too: MAX's
        # runtime names a plugin's device "<api>:<arch>" with the plugin's
        # api unchanged (`apiToVendorPrefix`, MLRT's DeviceAPINames.h), and
        # the Level Zero driver plugin's api is "level_zero". Idempotent.
        return target_arch0.replace("intel:", "").replace("level_zero:", "")

    @staticmethod
    def supported_archs_message() -> String:
        return String("  ", _supported_archs_line[Self](), "\n\n")


comptime IntelXe2Family = AcceleratorArchitectureFamily(
    # The smallest Xe2 subgroup, which the Level Zero plugin reports too.
    warp_size=16,
    # Per Xe-core: 8 EUs x 8 hardware threads x SIMD16.
    threads_per_multiprocessor=_K,
    shared_memory_per_multiprocessor=128 * _KB,
    # Per Xe-core: 64 threads x 128 GRF registers x 16 32-bit lanes.
    max_registers_per_block=128 * _K,
    max_thread_block_size=_K,
)
"""Intel Xe2 (Battlemage) GPU architecture family."""

comptime IntelXe2SG32Family = AcceleratorArchitectureFamily(
    # The variant targets' sub-group size.
    warp_size=32,
    threads_per_multiprocessor=IntelXe2Family.threads_per_multiprocessor,
    shared_memory_per_multiprocessor=(
        IntelXe2Family.shared_memory_per_multiprocessor
    ),
    max_registers_per_block=IntelXe2Family.max_registers_per_block,
    max_thread_block_size=IntelXe2Family.max_thread_block_size,
)
"""Intel Xe2 with kernels at sub-group size 32."""

comptime _bmg_g31_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "spirv64-unknown-unknown", `,
        `stdlib_plugin = "intel", `,
        `opaque_plugin = `,
        INTEL_PLUGINS,
        `,`,
        `arch = "bmg-g31", `,
        `data_layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1", `,
        `index_bit_width = 64, `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for Battlemage G31 (Arc Pro B70)."""

comptime _bmg_g21_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "spirv64-unknown-unknown", `,
        `stdlib_plugin = "intel", `,
        `opaque_plugin = `,
        INTEL_PLUGINS,
        `,`,
        `arch = "bmg-g21", `,
        `data_layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1", `,
        `index_bit_width = 64, `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for Battlemage G21 (Arc B580, Arc Pro B60)."""

comptime _bmg_g31_sg32_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "spirv64-unknown-unknown", `,
        `stdlib_plugin = "intel", `,
        `opaque_plugin = `,
        INTEL_PLUGINS,
        `,`,
        `arch = "bmg-g31", `,
        `features = "+sub-group-size-32", `,
        `data_layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1", `,
        `index_bit_width = 64, `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Battlemage G31 with kernels at sub-group size 32."""

comptime _bmg_g21_sg32_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "spirv64-unknown-unknown", `,
        `stdlib_plugin = "intel", `,
        `opaque_plugin = `,
        INTEL_PLUGINS,
        `,`,
        `arch = "bmg-g21", `,
        `features = "+sub-group-size-32", `,
        `data_layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1", `,
        `index_bit_width = 64, `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Battlemage G21 with kernels at sub-group size 32."""

comptime ArcProB70 = TargetAccelerator[
    GPUInfo.from_family(
        family=IntelXe2Family,
        name="Arc Pro B70",
        api="level_zero",
        arch_name="bmg-g31",
        compute=20.2,  # Level Zero IP version: architecture 20, release 2
        version="xe2",
        sm_count=32,  # Xe-cores
    ),
    _bmg_g31_target,
]()
"""Intel Arc Pro B70 configuration."""

comptime ArcB580 = TargetAccelerator[
    GPUInfo.from_family(
        family=IntelXe2Family,
        name="Arc B580",
        api="level_zero",
        arch_name="bmg-g21",
        compute=20.1,  # Level Zero IP version: architecture 20, release 1
        version="xe2",
        sm_count=20,  # Xe-cores
    ),
    _bmg_g21_target,
]()
"""Intel Arc B580 configuration."""

comptime ArcProB70SG32 = TargetAccelerator[
    GPUInfo.from_family(
        family=IntelXe2SG32Family,
        name="Arc Pro B70 SG32",
        api="level_zero",
        arch_name="bmg-g31-sg32",
        compute=20.2,
        version="xe2",
        sm_count=32,
    ),
    _bmg_g31_sg32_target,
]()
"""Intel Arc Pro B70 with kernels at sub-group size 32."""

comptime ArcB580SG32 = TargetAccelerator[
    GPUInfo.from_family(
        family=IntelXe2SG32Family,
        name="Arc B580 SG32",
        api="level_zero",
        arch_name="bmg-g21-sg32",
        compute=20.1,
        version="xe2",
        sm_count=20,
    ),
    _bmg_g21_sg32_target,
]()
"""Intel Arc B580 with kernels at sub-group size 32."""
