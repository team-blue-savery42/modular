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

# log2 and exp2 through the Intel math hooks (llvm.log2, llvm.exp2) at vector
# widths 1 to 64, float32 and float16, against the host. Widths above 16 have
# no SPIR-V vector type; the compiler plugin splits them. Prints FAIL for a
# width whose worst error exceeds 4 float32 ulps.
from std.math import exp2, log2
from std.memory import alloc

from hal_gpu import launch, load, open_context

# Elements per section of the buffer: inputs, then log2, then exp2.
comptime N = 64


def kernel[
    dtype: DType, w: Int
](p: MutPointer[Scalar[dtype], MutAnyOrigin]) where dtype.is_floating_point():
    var x = p.unsafe_load[width=w](0)
    p.unsafe_store[width=w](N, log2(x))
    p.unsafe_store[width=w](2 * N, exp2(x))


def check[dtype: DType, w: Int]() raises where dtype.is_floating_point():
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var function = load[type_of(kernel[dtype, w]), kernel[dtype, w]](ctx[])
    var bytes = 3 * N * (4 if dtype == DType.float32 else 2)
    var d = ctx[].alloc_sync(UInt64(bytes))
    var h = alloc[Scalar[dtype]]({length = 3 * N}).unsafe_leak()
    for i in range(3 * N):
        h[unsafe_offset=i] = Scalar[dtype](0)
    # 0.5 to 7.25 in steps of 0.25 then repeating: exact in float16 too.
    for i in range(N):
        h[unsafe_offset=i] = Scalar[dtype](Float32(i % 28) * 0.25 + 0.5)
    queue[].copy_to_device(d.view(), h.unsafe_bitcast[UInt8]())
    launch(queue[], function, 1, 1, [ctx[].memory_get_address(d)])
    queue[].copy_from_device(h.unsafe_bitcast[UInt8](), d.view())
    queue[].synchronize()
    var worst = Float32(0)
    for i in range(w):
        var x = h[unsafe_offset=i].cast[DType.float32]()
        var l = h[unsafe_offset=N + i].cast[DType.float32]()
        var e = h[unsafe_offset=2 * N + i].cast[DType.float32]()
        # Errors in ulps of the float32 result, against float64.
        var lr = log2(x.cast[DType.float64]())
        var er = exp2(x.cast[DType.float64]())
        var ulp_l = Float64(2.0) ** (Float64(Int(log2(abs(lr) + 1e-30))) - 23)
        var ulp_e = Float64(2.0) ** (Float64(Int(log2(er))) - 23)
        worst = max(
            worst,
            Float32(abs(l.cast[DType.float64]() - lr) / ulp_l),
            Float32(abs(e.cast[DType.float64]() - er) / ulp_e),
        )
    # float16 results are rounded to 10 bits: allow its ulp, 2^13 float32 ulps.
    var bound = Float32(4) if dtype == DType.float32 else Float32(4 * 8192)
    print(
        "PASS" if worst <= bound else "FAIL",
        dtype,
        "width",
        w,
        "worst error (float32 ulps)",
        worst,
        "log2(0.5)",
        h[unsafe_offset=N],
        "exp2(0.5)",
        h[unsafe_offset=2 * N],
    )


def main() raises:
    check[DType.float32, 1]()
    check[DType.float32, 2]()
    check[DType.float32, 4]()
    check[DType.float32, 8]()
    check[DType.float32, 16]()
    check[DType.float32, 32]()
    check[DType.float32, 64]()
    check[DType.float16, 1]()
    check[DType.float16, 4]()
    check[DType.float16, 8]()
    check[DType.float16, 32]()
