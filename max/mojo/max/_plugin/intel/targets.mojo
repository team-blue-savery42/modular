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

Kernels run at sub-group size 16. Each Xe2 target has a variant whose features
have `+sub-group-size-32` (`--target-accelerator=intel:bmg-g31-sg32`, or
`compile_info[kernel, target=ArcProB70SG32.target]` for one kernel): its
kernels run at 32 and its `WARP_SIZE` is 32 (the compiler plugin's
`SubgroupSize.h` and `IntelMaxPluginHooks` read the feature). Its `arch` is the
base target's, the device `ocloc` and the driver see. Kernels using DPAS stay
at 16 (at 32 `ocloc` crashes).

The integrated-GPU targets (`rpl-p`, Xe-LP; `arl-s`, Xe-LPG) have `version`
"xe-lp" and "xe-lpg": the Intel GEMM and flash-attention routes, DPAS and 2D
block I/O, are taken for `version` "xe2" alone, so these targets run MAX's
generic kernels. Their families carry the limits Level Zero reports (the
work-group limit, shared local memory).
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
        RaptorLakeP  ._with_cli_values[["rpl-p"]],
        ArrowLakeS   ._with_cli_values[["arl-s"]],
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

comptime IntelXeLPFamily = AcceleratorArchitectureFamily(
    # Xe-LP runs SIMD8, 16 and 32 sub-groups; kernels run at 16, as on Xe2.
    warp_size=16,
    # Per subslice: 16 EUs x 7 hardware threads x SIMD8 (Level Zero on a
    # Raptor Lake-P: numEUsPerSubslice 16, numThreadsPerEU 7,
    # physicalEUSimdWidth 8).
    threads_per_multiprocessor=896,
    # maxSharedLocalMemory.
    shared_memory_per_multiprocessor=64 * _KB,
    # Per subslice: 112 threads x 128 GRF registers x 8 32-bit lanes.
    max_registers_per_block=112 * _K,
    # maxTotalGroupSize; the Xe2 parts allow 1024.
    max_thread_block_size=512,
)
"""Intel Xe-LP (Tiger Lake to Raptor Lake integrated GPUs): no DPAS, no 2D
block I/O, no float64, work-groups of at most 512 work-items."""

comptime IntelXeLPGFamily = AcceleratorArchitectureFamily(
    warp_size=16,
    # Per Xe-core: 16 EUs x 8 hardware threads x SIMD8 (Level Zero on an
    # Arrow Lake-S).
    threads_per_multiprocessor=_K,
    shared_memory_per_multiprocessor=64 * _KB,
    # Per Xe-core: 128 threads x 128 GRF registers x 8 32-bit lanes.
    max_registers_per_block=128 * _K,
    max_thread_block_size=_K,
)
"""Intel Xe-LPG (Meteor Lake and Arrow Lake integrated GPUs): no DPAS, no 2D
block I/O; float64 and 1024-wide work-groups."""

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

comptime _rpl_p_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "spirv64-unknown-unknown", `,
        `stdlib_plugin = "intel", `,
        `opaque_plugin = `,
        INTEL_PLUGINS,
        `,`,
        `arch = "rpl-p", `,
        `data_layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1", `,
        `index_bit_width = 64, `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for the Raptor Lake-P integrated GPU (Xe-LP)."""

comptime _arl_s_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "spirv64-unknown-unknown", `,
        `stdlib_plugin = "intel", `,
        `opaque_plugin = `,
        INTEL_PLUGINS,
        `,`,
        `arch = "arl-s", `,
        `data_layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1", `,
        `index_bit_width = 64, `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for the Arrow Lake-S integrated GPU (Xe-LPG)."""

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

comptime RaptorLakeP = TargetAccelerator[
    GPUInfo.from_family(
        family=IntelXeLPFamily,
        name="Raptor Lake-P UHD Graphics",
        api="level_zero",
        arch_name="rpl-p",
        compute=12.3,  # Level Zero IP version 12.3
        version="xe-lp",
        sm_count=5,  # subslices (80 EUs)
    ),
    _rpl_p_target,
]()
"""Intel Raptor Lake-P integrated GPU (Core i5-1334U and its family)."""

comptime ArrowLakeS = TargetAccelerator[
    GPUInfo.from_family(
        family=IntelXeLPGFamily,
        name="Arrow Lake-S Graphics",
        api="level_zero",
        arch_name="arl-s",
        compute=12.7,  # Level Zero IP version 12.70: the release's first digit
        version="xe-lpg",
        sm_count=4,  # Xe-cores (64 EUs)
    ),
    _arl_s_target,
]()
"""Intel Arrow Lake-S integrated GPU (Core Ultra 200S)."""
