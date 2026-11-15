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

# The float64 formatting path of print, step by step, device against host.
# One work-item runs each step of the stdlib's dragonbox
# (std.builtin._format_float) on float64 values loaded from memory and stores
# every intermediate value; the host runs the same functions and compares.
# Prints one line per step that differs, then "wrong K of M". Before the
# compiler plugin split 128-bit integers into 64-bit halves, steps 0-9 (the
# float's fields, the cache index, the 128-bit cache entry, _umul128) were
# right on the B70 and from step 10 (_umul128_upper64, which LLVM makes a
# 128-bit multiply of a value masked with the 128-bit constant 2^64 - 1) to
# the final digits wrong, with IGC's defaults and with
# MOJO_INTEL_IGC_OPTIONS=-cl-opt-disable: IGC's 128-bit integer arithmetic.
# Ends with PASS when no step differs. In check-e2e.
from std.memory import alloc
from std.builtin._format_float import (
    FP,
    _to_decimal,
    _floor_log10_pow2,
    _floor_log2_pow10,
    _compute_delta,
    _compute_mul,
    _divide_by_pow10,
    _umul128,
    _umul128_upper64,
    _umul192_upper128,
    _uint128_high,
    _uint128_low,
    cache_f64,
)
from std.builtin.globals import global_constant
from std.utils.numerics import FPUtils

from hal_gpu import launch, load, open_context

comptime NV = 7
comptime NS = 16


@always_inline
def steps(x: Float64, o: MutPointer[UInt64, MutAnyOrigin]):
    comptime dt = DType.float64
    var sig = rebind[UInt64](FPUtils.get_mantissa_uint(x))
    var exp = FPUtils.get_exponent_biased(x)
    o[unsafe_offset=0] = sig
    o[unsafe_offset=1] = UInt64(exp)
    # _to_decimal's normal-interval path, inputs and intermediates.
    var two_fc = (sig * 2) | (UInt64(1) << UInt64(FP[dt].sig_bits + 1))
    var binary_exp = exp + FP[dt].neg_exp_bias - FP[dt].sig_bits
    var minus_k = _floor_log10_pow2(binary_exp) - FP[dt].kappa
    var beta = binary_exp + _floor_log2_pow10(-minus_k)
    var cache_index = -minus_k - FP[dt].min_k
    o[unsafe_offset=2] = UInt64(minus_k)
    o[unsafe_offset=3] = UInt64(beta)
    o[unsafe_offset=4] = UInt64(cache_index)
    var cache = global_constant[cache_f64]()[cache_index]
    o[unsafe_offset=5] = _uint128_high(cache)
    o[unsafe_offset=6] = _uint128_low(cache)
    o[unsafe_offset=7] = _compute_delta[
        DType.uint64, FP[dt].total_bits, FP[dt].cache_bits
    ](cache_index, beta)
    var u = (two_fc | 1) << UInt64(beta)
    var hl = _umul128(u, _uint128_high(cache))
    o[unsafe_offset=8] = _uint128_high(hl)
    o[unsafe_offset=9] = _uint128_low(hl)
    o[unsafe_offset=10] = _umul128_upper64(u, _uint128_low(cache))
    var z = _compute_mul[DType.uint64](u, cache_index)
    o[unsafe_offset=11] = z.integer_part
    o[unsafe_offset=12] = _divide_by_pow10[
        FP[dt, DType.uint64].kappa + 1, FP[dt, DType.uint64].n_max_larger
    ](z.integer_part)
    # The whole conversion.
    var s2 = sig
    var e2 = exp
    _to_decimal[dt](s2, e2)
    o[unsafe_offset=13] = s2
    o[unsafe_offset=14] = UInt64(e2)
    # The digit loop of _write_float: last digit and digit count.
    var d = s2
    var n = UInt64(0)
    var last = UInt64(0)
    while d > 0:
        last = d % 10
        d //= 10
        n += 1
    o[unsafe_offset=15] = n * 16 + last


def kernel(
    src: MutPointer[Float64, MutAnyOrigin],
    dst: MutPointer[UInt64, MutAnyOrigin],
):
    for i in range(NV):
        steps(src[unsafe_offset=i], dst.unsafe_offset(i * NS))


def main() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var function = load[type_of(kernel), kernel](ctx[])
    var d_src = ctx[].alloc_sync(UInt64(8 * NV))
    var d_dst = ctx[].alloc_sync(UInt64(8 * NV * NS))
    var h_src = alloc[Float64]({length = NV}).unsafe_leak()
    var h_dst = alloc[UInt64]({length = NV * NS}).unsafe_leak()
    var want = alloc[UInt64]({length = NV * NS}).unsafe_leak()
    var values: List[Float64] = [
        0.5023639798164368,
        0.50390625,
        0.5018579959869385,
        7.2,
        1.7976931348623157e308,
        -1.0,
        0.5,
    ]
    for i in range(NV):
        h_src[unsafe_offset=i] = values[i]
        steps(
            values[i],
            rebind[MutPointer[UInt64, MutAnyOrigin]](
                want.unsafe_offset(i * NS)
            ),
        )
    queue[].copy_to_device(d_src.view(), h_src.unsafe_bitcast[UInt8]())
    launch(
        queue[],
        function,
        1,
        1,
        [ctx[].memory_get_address(d_src), ctx[].memory_get_address(d_dst)],
    )
    queue[].copy_from_device(h_dst.unsafe_bitcast[UInt8](), d_dst.view())
    queue[].synchronize()
    var wrong = 0
    for i in range(NV):
        for s in range(NS):
            var g = h_dst[unsafe_offset=i * NS + s]
            var w = want[unsafe_offset=i * NS + s]
            if g != w:
                wrong += 1
                print("value", i, "step", s, "device", g, "host", w)
    print("wrong", wrong, "of", NV * NS)
    print("PASS" if wrong == 0 else "FAIL")
