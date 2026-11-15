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
"""Lowerings for the `std.math` functions `IntelPlugin` overrides.

On a GPU the stdlib builds `exp` and `log` for float16 and float32 from
`exp2` and `log2`, which the CUDA and HIP plugins hook to their hardware
instructions. Without a hook the stdlib's software `exp2` runs, which on the
B70 put `exp(-1.0847778)` 25 ulp off (and `erf`, built on `exp`, 29 ulp).
Here they are `llvm.exp2` and `llvm.log2`, which LLVM's SPIR-V backend emits
as OpenCL's `exp2` and `log2` (at most 3 ulp, IGC's implementation).
"""

from std.collections import OptionalReg
from std.collections.string.string_span import get_static_string
from std.math.math import _UnaryMathOp, _UnaryMathPluginHookFnType
from std.sys import llvm_intrinsic


comptime _MathFn = OptionalReg[_UnaryMathPluginHookFnType]


def _intel_unary_math_fn[op: _UnaryMathOp, dtype: DType]() -> _MathFn:
    """Returns the OpenCL lowering of the `std.math` function `op` for
    `dtype`, or `None` to keep the stdlib's own lowering."""
    # float16 widens to float32 in the stdlib. llvm.log2 and llvm.exp2 on
    # <4 x half> were off by over 1000 ulp on the B70.
    comptime supported = dtype == DType.float32
    comptime match op:
        case .exp2:
            return comptime (
                _MathFn(
                    _llvm_unary_math["llvm.exp2"]
                ) if supported else _MathFn(None)
            )
        case .log2:
            return comptime (
                _MathFn(
                    _llvm_unary_math["llvm.log2"]
                ) if supported else _MathFn(None)
            )
        case .exp | .log | .sqrt | .rsqrt | .recip | .sin | .cos | .tanh:
            return None


@inline(.always)
def _llvm_unary_math[
    dtype: DType, width: SIMDLength, //, intrinsic: StaticString
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Applies the overloaded LLVM intrinsic `intrinsic` to the whole vector;
    the SPIR-V backend emits the OpenCL instruction for it."""
    return llvm_intrinsic[intrinsic, SIMD[dtype, width], has_side_effect=False](
        x
    )
