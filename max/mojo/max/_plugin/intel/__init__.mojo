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
"""MAX support for Intel GPUs: the MAX plugin, and the target table in
`targets.mojo`.

Barriers, shuffles and ballots call the OpenCL built-ins by their mangled
names, which LLVM's SPIR-V backend lowers to `OpControlBarrier` and the
`OpGroupNonUniform*` instructions. A warp is a sub-group, which the Intel
compiler plugin fixes in every kernel (`intel_reqd_sub_group_size`): 16
work-items, or 32 for a target with the `sub-group-size-32` feature, so
`WARP_SIZE` always holds. `mma()` runs DPAS (`max.gpu.compute.arch.mma_intel`),
at sub-group size 16 only.
"""

from std.collections import OptionalReg
from std.collections.string.string_span import _get_kgen_string
from std.ffi import external_call
from std.sys import bit_width_of
from std.sys.info import CompilationTarget

from max.gpu.compute.arch.mma_intel import _mma_intel
from max.gpu.sync.sync import (
    _BarrierPluginHookFnType,
    _SyncwarpPluginHookFnType,
)

from .._shuffle import shuffle_words
from .._trait import MaxPlugin, MaxPluginHooks
from max._driver.max_license.hal._plugin.intel.intel_plugin import (
    IntelHalPluginHooks,
)

# OpenCL `cl_mem_fence_flags`.
comptime _CLK_LOCAL_MEM_FENCE = UInt32(1)
comptime _CLK_GLOBAL_MEM_FENCE = UInt32(2)


@inline(.always)
def _barrier():
    # CUDA's __syncthreads orders global as well as shared memory within the
    # block, so both fences.
    external_call["_Z7barrierj", NoneType](
        _CLK_LOCAL_MEM_FENCE | _CLK_GLOBAL_MEM_FENCE
    )


@inline(.always)
def _syncwarp(mask: Int):
    # A sub-group runs its active work-items together; there are no lane
    # masks, so `mask` is ignored, as on AMD.
    external_call["_Z17sub_group_barrierj", NoneType](
        _CLK_LOCAL_MEM_FENCE | _CLK_GLOBAL_MEM_FENCE
    )


@inline(.always)
def _shuffle_word(mask: UInt, word: UInt32, offset: UInt32) -> UInt32:
    # A sub-group runs its active work-items together; there are no lane
    # masks, so `mask` is ignored, as on AMD.
    return external_call["_Z17sub_group_shufflejj", UInt32](word, offset)


@inline(.always)
def _shuffle_xor_word(mask: UInt, word: UInt32, offset: UInt32) -> UInt32:
    # OpGroupNonUniformShuffleXor. The default hook reads the source lane
    # through `sub_group_shuffle` with a computed index, which IGC lowers to
    # an indirect register move per word; a reduction over 16 lanes then cost
    # ~0.75 us in the Gated DeltaNet recurrence's token loop.
    return external_call["_Z21sub_group_shuffle_xorjj", UInt32](word, offset)


@inline(.always)
def _ballot(val: Bool) -> SIMD[DType.uint32, 4]:
    """OpenCL's `sub_group_ballot`: bit `l` of the 128-bit mask is set when
    work-item `l` of the sub-group passes a nonzero value."""
    return external_call["_Z16sub_group_balloti", SIMD[DType.uint32, 4]](
        Int32(val)
    )


struct IntelMaxPlugin(MaxPlugin):
    """The MAX plugin for Intel GPUs."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["intel"]()
    comptime HookFor[target: CompilationTarget] = IntelMaxPluginHooks[target]


struct IntelMaxPluginHooks[target: CompilationTarget](MaxPluginHooks):
    """The MAX hooks for an Intel GPU target."""

    comptime _raw_mlir_target = Self.target._mlir_value
    # Xe2 runs SIMD16 and SIMD32 sub-groups. Kernels run at 16, the size the
    # Level Zero plugin reports, unless the target asks for 32; the Intel
    # compiler plugin reads the same feature.
    comptime warp_size = 32 if Self.target._has_feature[
        "sub-group-size-32"
    ]() else 16
    comptime HalHooks = IntelHalPluginHooks[Self.target]
    comptime barrier_fn: OptionalReg[_BarrierPluginHookFnType] = _barrier
    comptime syncwarp_fn: OptionalReg[_SyncwarpPluginHookFnType] = _syncwarp

    @inline(.always)
    @staticmethod
    def shuffle_idx_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        return shuffle_words[_shuffle_word](mask, val, offset)

    @inline(.always)
    @staticmethod
    def shuffle_xor_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        # A source lane outside the sub-group keeps the caller's value, as the
        # default hook does; OpenCL leaves that case undefined.
        var shuffled = shuffle_words[_shuffle_xor_word](mask, val, offset)
        return shuffled if offset < UInt32(Self.warp_size) else val

    @inline(.always)
    @staticmethod
    def vote_fn[ret_type: DType](val: Bool) -> Scalar[ret_type]:
        # 16 or 32 lanes fit the first word; a 64-bit mask takes two.
        var mask = _ballot(val)
        comptime if bit_width_of[ret_type]() <= 32:
            return mask[0].cast[ret_type]()
        else:
            return mask[0].cast[ret_type]() | (mask[1].cast[ret_type]() << 32)

    @inline(.always)
    @staticmethod
    def mma_fn[
        d_dtype: DType,
        d_width: SIMDLength,
        a_dtype: DType,
        a_width: SIMDLength,
        b_dtype: DType,
        b_width: SIMDLength,
        c_dtype: DType,
        c_width: SIMDLength,
        //,
    ](
        mut d: SIMD[d_dtype, d_width],
        a: SIMD[a_dtype, a_width],
        b: SIMD[b_dtype, b_width],
        c: SIMD[c_dtype, c_width],
    ):
        _mma_intel(d, a, b, c)
