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
"""Intel Xe2 MMA: DPAS through `SPV_INTEL_subgroup_matrix_multiply_accumulate`.

`mma()` reaches `_mma_intel` through the Intel MAX plugin's `mma_fn` hook.
One call is `OpSubgroupMatrixMultiplyAccumulateINTEL`, which IGC lowers to
`dpas`: D (M x 16) = A (M x 16) * B (16 x 16) + C (M x 16) over a sub-group of
16 work-items, so N = 16 is the sub-group size and K = 16.

Fragments, from the extension (N = K = 16), for lane `n` of the sub-group:

- A: `SIMD[float16 | bfloat16, M]`, M = 1, 2, 4 or 8: element `r` is
  `A[r, n]` (lane n holds column n, one value per row).
- B: `SIMD[float16 | bfloat16, 16]` (the same type as A): element `k` is
  `B[k, n]` (lane n holds column n). It is passed as the extension's packed
  ("VNNI") form, `SIMD[int32, 8]` with rows 2j and 2j + 1 in the low and high
  half of word j, which is the same bits; a B already in that form may be
  passed as `SIMD[int32 | uint32, 8]`, and A's type then says how to read it.
- C and D: `SIMD[float32, M]`, or A's 16-bit type (`float16` with float16,
  `bfloat16` with bfloat16): element `r` is `C[r, n]` / `D[r, n]`.

These are the `cl_intel_subgroup_matrix_multiply_accumulate` shapes for a
device whose sub-group size is 16. DPAS needs sub-group size 16: at 32 `ocloc`
crashes, so a kernel built for an Intel `-sg32` target is rejected at compile
time. Other shapes, including NVIDIA's 16x8x16 fragments, are rejected at
compile time too.
"""

from std.ffi import external_call
from std.memory import bitcast

from max.gpu.globals import WARP_SIZE

from ..mma import _unsupported_mma_op

# Matrix Multiply Accumulate Operands (the extension's bitmask).
comptime _C_BF16 = 0x4
comptime _D_BF16 = 0x8
comptime _A_F16 = 0x400
comptime _B_F16 = 0x800
comptime _A_BF16 = 0x1000
comptime _B_BF16 = 0x2000


def _dpas_name[m: Int, acc: StaticString]() -> StaticString:
    """The OpenCL (Itanium) mangled name of the built-in for A `short<m>`,
    B `int8` and an accumulator of element `acc` (`f` float, `Dh` half, `s`
    short): one name per signature, so a module calling several has one
    declaration of each (LLVM's SPIR-V backend demangles it)."""
    comptime assert (m == 1 or m == 2 or m == 4 or m == 8) and (
        acc == "f" or acc == "Dh" or acc == "s"
    ), "no DPAS built-in for this shape"
    comptime if m == 1 and acc == "f":
        return "_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELisDv8_ifi"
    elif m == 1 and acc == "Dh":
        return "_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELisDv8_iDhi"
    elif m == 1 and acc == "s":
        return "_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELisDv8_isi"
    elif m == 2 and acc == "f":
        return (
            "_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_sDv8_iDv2_fi"
        )
    elif m == 2 and acc == "Dh":
        return "_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_sDv8_iDv2_Dhi"
    elif m == 2 and acc == "s":
        return "_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_sDv8_iS_i"
    elif m == 4 and acc == "f":
        return (
            "_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iDv4_fi"
        )
    elif m == 4 and acc == "Dh":
        return "_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iDv4_Dhi"
    elif m == 4 and acc == "s":
        return "_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iS_i"
    elif m == 8 and acc == "f":
        return (
            "_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi"
        )
    elif m == 8 and acc == "Dh":
        return "_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_Dhi"
    else:  # m == 8 and acc == "s"
        return "_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iS_i"


@inline(.always)
def _dpas[
    acc_dtype: DType, m: Int, acc: StaticString, flags: Int
](
    a: SIMD[DType.int16, m],
    b: SIMD[DType.int32, 8],
    c: SIMD[acc_dtype, m],
) -> SIMD[acc_dtype, m]:
    return external_call[_dpas_name[m, acc](), SIMD[acc_dtype, m]](
        Int32(16), a, b, c, Int32(flags)
    )


@inline(.always)
def _mma_intel(mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    """DPAS: `d = a * b + c` on one M x 16 x 16 tile; see the module."""
    comptime assert WARP_SIZE == 16, String(
        t"Intel DPAS (mma) needs sub-group size 16, but this kernel is built"
        t" for {WARP_SIZE}: build it for the target without `-sg32`"
    )
    comptime m = Int(a.length)
    comptime f16 = a.dtype == DType.float16
    comptime bf16 = a.dtype == DType.bfloat16
    comptime typed_b = b.dtype == a.dtype and Int(b.length) == 16
    comptime packed_b = (
        b.dtype in (DType.int32, DType.uint32) and Int(b.length) == 8
    )
    comptime ok = (
        (f16 or bf16)
        and (typed_b or packed_b)
        and (m == 1 or m == 2 or m == 4 or m == 8)
        and Int(c.length) == m
        and Int(d.length) == m
        and c.dtype == d.dtype
        and (d.dtype == DType.float32 or d.dtype == a.dtype)
    )
    comptime if not ok:
        _unsupported_mma_op[
            d.dtype,
            d.length,
            a.dtype,
            a.length,
            b.dtype,
            b.length,
            c.dtype,
            c.length,
        ]()
    else:
        var a16 = bitcast[DType.int16, m](rebind[SIMD[a.dtype, m]](a))
        var b32 = bitcast[DType.int32, 8](b)
        comptime ab = (_A_F16 | _B_F16) if f16 else (_A_BF16 | _B_BF16)

        comptime if d.dtype == DType.float32:
            d = rebind[type_of(d)](
                _dpas[DType.float32, m, "f", ab](
                    a16, b32, rebind[SIMD[DType.float32, m]](c)
                )
            )
        elif f16:
            d = rebind[type_of(d)](
                _dpas[DType.float16, m, "Dh", ab](
                    a16, b32, rebind[SIMD[DType.float16, m]](c)
                )
            )
        else:
            # bfloat16 C and D move as their bits, read as bfloat16
            # through the operand flags.
            var c16 = bitcast[DType.int16, m](
                rebind[SIMD[DType.bfloat16, m]](c)
            )
            var d16 = _dpas[DType.int16, m, "s", ab | _C_BF16 | _D_BF16](
                a16, b32, c16
            )
            d = rebind[type_of(d)](bitcast[DType.bfloat16, m](d16))
