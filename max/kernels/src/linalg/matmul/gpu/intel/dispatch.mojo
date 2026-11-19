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

# ===----------------------------------------------------------------------=== #
# ===----------------------------------------------------------------------=== #
"""`_matmul_gpu`'s Intel branch: picks an Intel GEMM kernel for a matmul and
launches it through `DeviceContext`, or declines (returns False) so that
`_matmul_gpu` carries on to its other paths (on Intel, the GEMV for M or
N = 1 and the naive kernel).

Taken: Xe2 targets at sub-group size 16 (`GPUInfo.api` "level_zero",
`version` "xe2", `warp_size` 16); float16 or bfloat16 A and B of one type,
float32 or that type for C; row-major tensors (unit inner stride, rows
packed) in generic memory with 64-byte aligned bases (`c` only gives the
shape when an epilogue stores the outputs); and the shapes
`gemm_supported` allows (N, K multiples of 8 and at least 32). M <=
`SKINNY_MAX_M` runs `skinny_kernel` (its defaults for the layout) and M <=
`SKINNY_WIDE_MAX_M` (batched decode) the same kernel with 32- or 64-row tiles;
larger M runs `gemm_kernel` with 256 x 256 work-group tiles, or 256 x
`SMALL_WG_N` when the output has fewer 256 x 256 tiles than the device has
Xe-cores.

Intel GPUs without DPAS (any other `version`, the integrated Xe-LP and
Xe-LPG): float16, bfloat16 or float32 A and B of one type, C float32 or that
type, row-major as above, M > 1, K a multiple of 4 (and N, for B as K x N):
`simt_kernel_fn`, a float32 shared-memory GEMM (`simt.mojo`). Anything else
carries on to the GEMV and the naive kernel.

`matmul_f8_dispatch_intel` is the FP8-weight sibling (MAX's static FP8 linear
layers on Intel, `mo.matmul.weight.only.scaled.float8.intel`): bfloat16 A,
float8_e4m3fn B^T and a float32 weight scale, C bfloat16 or float32. M <=
`SKINNY_MAX_M` runs `skinny_kernel_f8` (8 sub-groups splitting K when N is
above `F8_SKINNY_KS16_MAX_N`, else 16); larger M widens B^T to a bfloat16
scratch copy and runs `gemm_kernel_f8_widened` in the GEMM's tile classes.
"""

from std.math import ceildiv
from std.memory.address_space import AddressSpace

from layout import TileTensor
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext

from ....utils import ElementwiseComputeFn, ElementwiseEpilogueFn
from .matmul import (
    F4_UNIT,
    F4_WIDEN_WORDS,
    f4_widen_kernel,
    gemm_f4_supported,
    skinny_kernel_f4,
    SKINNY_MAX_M,
    SKINNY_WIDE_MAX_M,
    SMALL_WG_N,
    gemm_block,
    gemm_f8_supported,
    gemm_grid,
    F8_WIDEN_WORDS,
    f8_widen_kernel,
    gemm_kernel_f8_widened,
    gemm_kernel_fn,
    gemm_small_tiles,
    gemm_supported,
    skinny_block,
    skinny_grid,
    skinny_kernel_f8,
    skinny_kernel_fn,
    skinny_ks,
    skinny_wide_ks,
)
from .simt import SIMT_THREADS, simt_grid, simt_kernel_fn, simt_supported


@inline(.always)
def _row_major(t: TileTensor[...]) -> Bool:
    """Whether a rank-2 tensor is row-major with packed rows and a 64-byte
    aligned base (2D block I/O's rules)."""
    return (
        Int(t.layout.stride[1]().value()) == 1
        and Int(t.layout.stride[0]().value()) == Int(t.dim[1]())
        and Int(t.unsafe_ptr()) % 64 == 0
    )


@inline(.always)
def _raw[
    dtype: DType
](t: TileTensor[dtype, ...]) -> MutPointer[Scalar[dtype], MutAnyOrigin]:
    """The tensor's base as the kernels' pointer type (generic memory)."""
    return MutPointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=Int(t.unsafe_ptr())
    )


comptime F8_SKINNY_KS16_MAX_N = 2048
"""Largest N for which `skinny_kernel_f8` splits K over 16 sub-groups (8
above it)."""

comptime F4_SKINNY_SPLIT_N = 2048
"""N above which `skinny_kernel_f4` runs `F4_SKINNY_KS_WIDE` sub-groups per
work-group, else `F4_SKINNY_KS_NARROW`."""

comptime F4_SKINNY_KS_WIDE = 8
"""Sub-groups splitting K in `skinny_kernel_f4` for N above
`F4_SKINNY_SPLIT_N`."""

comptime F4_SKINNY_KS_NARROW = 16
"""Sub-groups splitting K in `skinny_kernel_f4` up to `F4_SKINNY_SPLIT_N`."""


@inline(.always)
def _raw_buf[
    dtype: DType
](buf: DeviceBuffer[dtype]) -> MutPointer[Scalar[dtype], MutAnyOrigin]:
    """A device buffer's base as the kernels' pointer type."""
    return MutPointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=Int(buf.unsafe_ptr())
    )


@inline(.always)
def matmul_dispatch_intel[
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    *,
    transpose_b: Bool,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor[mut=False, ...],
    b: TileTensor[mut=False, ...],
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
    ctx: DeviceContext,
) raises -> Bool:
    """Runs `c = a @ b` (`b` given as N x K with `transpose_b`) with an Intel
    GEMM kernel and returns True, or returns False, launching nothing, when
    the target, types, layouts or shape aren't the kernels'.

    Parameters:
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        transpose_b: Whether `b` is stored as `(N, K)`.
        has_epilogue_fn: Whether `epilogue_fn` stores each output.
        has_compute_fn: Whether `compute_fn` maps each output.

    Args:
        c: Output, `(M, N)`.
        a: `(M, K)`.
        b: `(K, N)`, or `(N, K)` with `transpose_b`.
        epilogue_fn: Stores each output (with `has_epilogue_fn`).
        compute_fn: Maps each output before the store (with
            `has_compute_fn`).
        ctx: Device context to launch on.

    Returns:
        Whether the matmul was launched.
    """
    comptime info = ctx.default_device_info
    comptime a_type = a.dtype
    comptime c_type = c.dtype
    comptime generic = (
        c.address_space == AddressSpace.GENERIC
        and a.address_space == AddressSpace.GENERIC
        and b.address_space == AddressSpace.GENERIC
    )
    # Intel GPUs without DPAS (Xe-LP, Xe-LPG): the shared-memory GEMM of
    # `simt.mojo` for every M above 1 (M = 1 is MAX's GEMV).
    comptime taken_simt = (
        info.api == "level_zero"
        and info.version != "xe2"
        and (
            a_type == DType.float16
            or a_type == DType.bfloat16
            or a_type == DType.float32
        )
        and b.dtype == a_type
        and (c_type == DType.float32 or c_type == a_type)
        and generic
    )
    comptime taken = (
        info.api == "level_zero"
        and info.version == "xe2"
        and info.warp_size == 16
        and (a_type == DType.float16 or a_type == DType.bfloat16)
        and b.dtype == a_type
        and (c_type == DType.float32 or c_type == a_type)
        and generic
    )
    comptime if taken_simt:
        var m = Int(c.dim[0]())
        var n = Int(c.dim[1]())
        var k = Int(a.dim[1]())
        if not (
            simt_supported(m, n, k, transpose_b)
            and (has_epilogue_fn or _row_major(c))
            and _row_major(a)
            and _row_major(b)
        ):
            return False
        comptime kern = simt_kernel_fn[
            a_type,
            c_type,
            EpilogueFnType,
            ComputeFnType,
            transpose_b,
            has_epilogue_fn,
            has_compute_fn,
        ]
        ctx.enqueue_function[kern](
            _raw(c),
            _raw(a),
            _raw(b),
            Int64(m),
            Int64(n),
            Int64(k),
            host_args=(epilogue_fn, compute_fn),
            grid_dim=simt_grid(m, n),
            block_dim=SIMT_THREADS,
        )
        return True
    elif not taken:
        return False
    else:
        var m = Int(c.dim[0]())
        var n = Int(c.dim[1]())
        var k = Int(a.dim[1]())
        if not (
            gemm_supported[a_type, c_type](m, n, k)
            # A store epilogue writes every output itself: `c` gives only
            # the shape (MAX may pass a null base).
            and (has_epilogue_fn or _row_major(c))
            and _row_major(a)
            and _row_major(b)
        ):
            return False
        var cp = _raw(c)
        var ap = _raw(a)
        var bp = _raw(b)
        if m <= SKINNY_MAX_M:
            comptime kern = skinny_kernel_fn[
                a_type,
                c_type,
                EpilogueFnType,
                ComputeFnType,
                transpose_b,
                has_epilogue_fn,
                has_compute_fn,
            ]
            ctx.enqueue_function[kern](
                cp,
                ap,
                bp,
                Int64(m),
                Int64(n),
                Int64(k),
                host_args=(epilogue_fn, compute_fn),
                grid_dim=skinny_grid(n),
                block_dim=skinny_block[skinny_ks[transpose_b]()](),
            )
            return True
        # Batched decode: 32 or 64 rows still stream B once, in wider tiles.
        if m <= 32:
            comptime ks = skinny_wide_ks[transpose_b, 32]()
            comptime kern = skinny_kernel_fn[
                a_type,
                c_type,
                EpilogueFnType,
                ComputeFnType,
                transpose_b,
                has_epilogue_fn,
                has_compute_fn,
                sg_m=32,
                ks=ks,
            ]
            ctx.enqueue_function[kern](
                cp,
                ap,
                bp,
                Int64(m),
                Int64(n),
                Int64(k),
                host_args=(epilogue_fn, compute_fn),
                grid_dim=skinny_grid(n),
                block_dim=skinny_block[ks](),
            )
            return True
        if m <= SKINNY_WIDE_MAX_M:
            comptime ks = skinny_wide_ks[transpose_b, 64]()
            comptime kern = skinny_kernel_fn[
                a_type,
                c_type,
                EpilogueFnType,
                ComputeFnType,
                transpose_b,
                has_epilogue_fn,
                has_compute_fn,
                sg_m=64,
                ks=ks,
            ]
            ctx.enqueue_function[kern](
                cp,
                ap,
                bp,
                Int64(m),
                Int64(n),
                Int64(k),
                host_args=(epilogue_fn, compute_fn),
                grid_dim=skinny_grid(n),
                block_dim=skinny_block[ks](),
            )
            return True
        # The Xe-core count from the device (Level Zero slices x
        # sub-slices, through the HAL's device attribute).
        var xe_cores = ctx.device().get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        if gemm_small_tiles(m, n, xe_cores):
            comptime kern = gemm_kernel_fn[
                a_type,
                c_type,
                EpilogueFnType,
                ComputeFnType,
                transpose_b,
                has_epilogue_fn,
                has_compute_fn,
                wg_n=SMALL_WG_N,
            ]
            ctx.enqueue_function[kern](
                cp,
                ap,
                bp,
                Int64(m),
                Int64(n),
                Int64(k),
                host_args=(epilogue_fn, compute_fn),
                grid_dim=gemm_grid[256, SMALL_WG_N](m, n),
                block_dim=gemm_block[256, SMALL_WG_N](),
            )
        else:
            comptime kern = gemm_kernel_fn[
                a_type,
                c_type,
                EpilogueFnType,
                ComputeFnType,
                transpose_b,
                has_epilogue_fn,
                has_compute_fn,
            ]
            ctx.enqueue_function[kern](
                cp,
                ap,
                bp,
                Int64(m),
                Int64(n),
                Int64(k),
                host_args=(epilogue_fn, compute_fn),
                grid_dim=gemm_grid(m, n),
                block_dim=gemm_block(),
            )
        return True


@inline(.always)
def matmul_f8_dispatch_intel(
    c: TileTensor[mut=True, ...],
    a: TileTensor[DType.bfloat16, ...],
    b: TileTensor[DType.float8_e4m3fn, ...],
    scale: Float32,
    ctx: DeviceContext,
) raises -> Bool:
    """Runs `c = scale * a @ b^T` (`b` given as N x K float8_e4m3fn) with an
    Intel FP8-weight kernel and returns True, or returns False, launching
    nothing, when the target, layouts or shape aren't the kernels' (the
    rules of `matmul_dispatch_intel`, and `gemm_f8_supported`).

    Args:
        c: Output, `(M, N)`, bfloat16 or float32.
        a: `(M, K)`, bfloat16.
        b: `(N, K)`, float8_e4m3fn.
        scale: The weight scale (one per tensor).
        ctx: Device context to launch on.

    Returns:
        Whether the matmul was launched.
    """
    comptime info = ctx.default_device_info
    comptime c_type = c.dtype
    comptime taken = (
        info.api == "level_zero"
        and info.version == "xe2"
        and info.warp_size == 16
        and (c_type == DType.float32 or c_type == DType.bfloat16)
        and c.address_space == AddressSpace.GENERIC
        and a.address_space == AddressSpace.GENERIC
        and b.address_space == AddressSpace.GENERIC
    )
    comptime if not taken:
        return False
    else:
        var m = Int(c.dim[0]())
        var n = Int(c.dim[1]())
        var k = Int(a.dim[1]())
        if not (
            gemm_f8_supported[c_type](m, n, k)
            and Int(b.dim[0]()) == n
            and Int(b.dim[1]()) == k
            and _row_major(c)
            and _row_major(a)
            and _row_major(b)
        ):
            return False
        var cp = _raw(c)
        var ap = _raw(a)
        var bp = _raw(b)
        if m <= SKINNY_MAX_M:
            # Measured on the B70: 8 sub-groups for N 5120 and
            # 8192, 16 for N 1024 (fewer work-groups).
            if n > F8_SKINNY_KS16_MAX_N:
                comptime kern = skinny_kernel_f8[c_type, ks=8]
                ctx.enqueue_function[kern](
                    cp,
                    ap,
                    bp,
                    Int64(m),
                    Int64(n),
                    Int64(k),
                    scale,
                    grid_dim=skinny_grid(n),
                    block_dim=skinny_block[8](),
                )
            else:
                comptime kern = skinny_kernel_f8[c_type, ks=16]
                ctx.enqueue_function[kern](
                    cp,
                    ap,
                    bp,
                    Int64(m),
                    Int64(n),
                    Int64(k),
                    scale,
                    grid_dim=skinny_grid(n),
                    block_dim=skinny_block[16](),
                )
            return True
        # Widen B^T once to bfloat16 (values / 256), then the bfloat16 GEMM
        # with the scale; the scratch copy is freed after the GEMM runs
        # (frees wait for the queue's earlier work).
        var wide = ctx.enqueue_create_buffer[DType.bfloat16](n * k)
        var wp = _raw_buf(wide)
        comptime widen_block = 256
        ctx.enqueue_function[f8_widen_kernel](
            wp,
            bp,
            Int64(n * k),
            grid_dim=ceildiv(n * k, 4 * F8_WIDEN_WORDS * widen_block),
            block_dim=widen_block,
        )
        var xe_cores = ctx.device().get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        if gemm_small_tiles(m, n, xe_cores):
            comptime kern = gemm_kernel_f8_widened[c_type, SMALL_WG_N]
            ctx.enqueue_function[kern](
                cp,
                ap,
                wp,
                Int64(m),
                Int64(n),
                Int64(k),
                scale,
                grid_dim=gemm_grid[256, SMALL_WG_N](m, n),
                block_dim=gemm_block[256, SMALL_WG_N](),
            )
        else:
            comptime kern = gemm_kernel_f8_widened[c_type]
            ctx.enqueue_function[kern](
                cp,
                ap,
                wp,
                Int64(m),
                Int64(n),
                Int64(k),
                scale,
                grid_dim=gemm_grid(m, n),
                block_dim=gemm_block(),
            )
        return True


@inline(.always)
def _launch_skinny_f4[
    c_type: DType, ks: Int
](
    ctx: DeviceContext,
    c: MutPointer[Scalar[c_type], MutAnyOrigin],
    a: MutPointer[Scalar[DType.bfloat16], MutAnyOrigin],
    b: MutPointer[Scalar[DType.uint8], MutAnyOrigin],
    scales: MutPointer[Scalar[DType.uint8], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    scale: Float32,
) raises:
    """Launches `skinny_kernel_f4` with `ks` sub-groups per work-group."""
    comptime kern = skinny_kernel_f4[c_type, ks=ks]
    ctx.enqueue_function[kern](
        c,
        a,
        b,
        scales,
        Int64(m),
        Int64(n),
        Int64(k),
        scale,
        grid_dim=skinny_grid(n),
        block_dim=skinny_block[ks](),
    )


@inline(.always)
def matmul_f4_dispatch_intel(
    c: TileTensor[mut=True, ...],
    a: TileTensor[DType.bfloat16, ...],
    b: TileTensor[DType.uint8, ...],
    scales: TileTensor[DType.float8_e4m3fn, ...],
    scale: Float32,
    ctx: DeviceContext,
) raises -> Bool:
    """Runs `c = scale * a @ dequant(b)^T` (`b` given as N x K / 2 bytes of
    E2M1 codes, low nibble first, `scales` as N x K / 16 E4M3 block scales:
    NVFP4) with an Intel FP4-weight kernel and returns True, or returns
    False, launching nothing, when the target, layouts or shape aren't the
    kernels' (the rules of `matmul_dispatch_intel`, and
    `gemm_f4_supported`).

    Args:
        c: Output, `(M, N)`, bfloat16 or float32.
        a: `(M, K)`, bfloat16.
        b: `(N, K / 2)`, packed E2M1 codes.
        scales: `(N, K / 16)`, float8_e4m3fn block scales.
        scale: The weight scale (one per tensor: NVFP4's `weight_scale_2`).
        ctx: Device context to launch on.

    Returns:
        Whether the matmul was launched.
    """
    comptime info = ctx.default_device_info
    comptime c_type = c.dtype
    comptime taken = (
        info.api == "level_zero"
        and info.version == "xe2"
        and info.warp_size == 16
        and (c_type == DType.float32 or c_type == DType.bfloat16)
        and c.address_space == AddressSpace.GENERIC
        and a.address_space == AddressSpace.GENERIC
        and b.address_space == AddressSpace.GENERIC
        and scales.address_space == AddressSpace.GENERIC
    )
    comptime if not taken:
        return False
    else:
        var m = Int(c.dim[0]())
        var n = Int(c.dim[1]())
        var k = Int(a.dim[1]())
        if not (
            gemm_f4_supported[c_type](m, n, k)
            and Int(b.dim[0]()) == n
            and Int(b.dim[1]()) * 2 == k
            and Int(scales.dim[0]()) == n
            and Int(scales.dim[1]()) * 16 == k
            and _row_major(c)
            and _row_major(a)
            and _row_major(b)
            and _row_major(scales)
        ):
            return False
        var cp = _raw(c)
        var ap = _raw(a)
        var bp = _raw(b)
        var sp = _raw(scales).unsafe_bitcast[UInt8]()
        var f4_scale = scale * F4_UNIT
        if m <= SKINNY_MAX_M:
            if n > F4_SKINNY_SPLIT_N:
                _launch_skinny_f4[c_type, F4_SKINNY_KS_WIDE](
                    ctx, cp, ap, bp, sp, m, n, k, f4_scale
                )
            else:
                _launch_skinny_f4[c_type, F4_SKINNY_KS_NARROW](
                    ctx, cp, ap, bp, sp, m, n, k, f4_scale
                )
            return True
        # Widen B^T once to bfloat16 (values times their block scale /
        # 2**22), then the FP8 path's bfloat16 GEMM with the scale; the
        # scratch copy is freed after the GEMM runs.
        var wide = ctx.enqueue_create_buffer[DType.bfloat16](n * k)
        var wp = _raw_buf(wide)
        comptime widen_block = 256
        ctx.enqueue_function[f4_widen_kernel](
            wp,
            bp,
            sp,
            Int64(n * k),
            grid_dim=ceildiv(n * k, 8 * F4_WIDEN_WORDS * widen_block),
            block_dim=widen_block,
        )
        var xe_cores = ctx.device().get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        if gemm_small_tiles(m, n, xe_cores):
            comptime kern = gemm_kernel_f8_widened[c_type, SMALL_WG_N]
            ctx.enqueue_function[kern](
                cp,
                ap,
                wp,
                Int64(m),
                Int64(n),
                Int64(k),
                f4_scale,
                grid_dim=gemm_grid[256, SMALL_WG_N](m, n),
                block_dim=gemm_block[256, SMALL_WG_N](),
            )
        else:
            comptime kern = gemm_kernel_f8_widened[c_type]
            ctx.enqueue_function[kern](
                cp,
                ap,
                wp,
                Int64(m),
                Int64(n),
                Int64(k),
                f4_scale,
                grid_dim=gemm_grid(m, n),
                block_dim=gemm_block(),
            )
        _ = wide^
        return True
