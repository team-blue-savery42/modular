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

# DPAS through MAX's `mma()` on the B70, through the HAL client, every
# result checked on the CPU.
#
# One sub-group of 16 computes D (M x 16) = A (M x 16) * B (16 x 16) + C for
# float16 and bfloat16 inputs, a float32 or a 16-bit accumulator (float16
# with float16, bfloat16 with bfloat16) and M = 1, 2, 4, 8 (the
# `arch/mma_intel.mojo` fragments: lane n holds column n of A, B, C and D).
# B goes in as 16 typed values per lane, and once more (M = 8, float32) in the
# packed VNNI form the extension takes (`SIMD[int32, 8]`, built on the host).
# Two data sets per case:
#
#   int     small integers in [-3, 3]: every product and sum is exact in
#           each type (|D| <= 147), so D must equal the CPU's.
#   random  uniform in [-1, 1), rounded to the input type: D must be within
#           the accumulator's rounding of the CPU's float64 result,
#           |D - ref| <= 2 * eps * (sum |a * b| + |c|) with eps 2^-23
#           (float32), 2^-10 (float16), 2^-7 (bfloat16).

from std.memory import alloc, bitcast
from std.sys.info import CompilationTarget

from max.gpu import lane_id
from max.gpu.compute.mma import mma

from hal_gpu import GpuContext, GpuQueue, launch, load, open_context

comptime K = 16
comptime N = 16


def mma_kernel[
    in_dtype: DType, acc_dtype: DType, m: Int
](
    a_ptr: MutPointer[Scalar[in_dtype], MutAnyOrigin],
    b_ptr: MutPointer[Scalar[in_dtype], MutAnyOrigin],
    c_ptr: MutPointer[Scalar[acc_dtype], MutAnyOrigin],
    d_ptr: MutPointer[Scalar[acc_dtype], MutAnyOrigin],
):
    var n = Int(lane_id())
    var a = SIMD[in_dtype, m]()
    var c = SIMD[acc_dtype, m]()
    comptime for r in range(m):
        a[r] = a_ptr[unsafe_offset=r * K + n]
        c[r] = c_ptr[unsafe_offset=r * N + n]
    var b = SIMD[in_dtype, K]()
    comptime for k in range(K):
        b[k] = b_ptr[unsafe_offset=k * N + n]
    var d = SIMD[acc_dtype, m]()
    mma(d, a, b, c)
    comptime for r in range(m):
        d_ptr[unsafe_offset=r * N + n] = d[r]


# B in the extension's packed form: word j of lane n holds B[2j, n] in its
# low and B[2j + 1, n] in its high 16 bits.
def mma_packed_kernel[
    in_dtype: DType
](
    a_ptr: MutPointer[Scalar[in_dtype], MutAnyOrigin],
    b_ptr: MutPointer[Int32, MutAnyOrigin],
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
):
    var n = Int(lane_id())
    var a = SIMD[in_dtype, 8]()
    var c = SIMD[DType.float32, 8]()
    var b = SIMD[DType.int32, 8]()
    comptime for r in range(8):
        a[r] = a_ptr[unsafe_offset=r * K + n]
        c[r] = c_ptr[unsafe_offset=r * N + n]
        b[r] = b_ptr[unsafe_offset=r * N + n]
    var d = SIMD[DType.float32, 8]()
    mma(d, a, b, c)
    comptime for r in range(8):
        d_ptr[unsafe_offset=r * N + n] = d[r]


struct Rng(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> UInt64:
        self.state ^= self.state >> 12
        self.state ^= self.state << 25
        self.state ^= self.state >> 27
        return self.state * 0x2545F4914F6CDD1D

    def small_int(mut self) -> Float32:
        return Float32(Int((self.next() >> 33) % 7) - 3)

    def uniform(mut self) -> Float32:
        # [-1, 1) in steps of 2^-23.
        return (
            Float32(Int((self.next() >> 40) % (1 << 24))) / Float32(1 << 23)
            - 1.0
        )


def eps_of[dtype: DType]() -> Float64:
    comptime if dtype == DType.float32:
        return 1.0 / Float64(1 << 23)
    elif dtype == DType.float16:
        return 1.0 / Float64(1 << 10)
    else:
        return 1.0 / Float64(1 << 7)


def type_name[dtype: DType]() -> String:
    comptime if dtype == DType.float32:
        return "float32"
    elif dtype == DType.float16:
        return "float16"
    else:
        return "bfloat16"


def run_case[
    in_dtype: DType, acc_dtype: DType, m: Int, packed: Bool
](mut ctx: GpuContext, queue: GpuQueue, random: Bool) raises -> Bool:
    # Host data, rounded to the device types; the reference in float64.
    var a = alloc[Scalar[in_dtype]]({length = m * K}).unsafe_leak()
    var b = alloc[Scalar[in_dtype]]({length = K * N}).unsafe_leak()
    var bp = alloc[Int32]({length = (K // 2) * N}).unsafe_leak()
    var c = alloc[Scalar[acc_dtype]]({length = m * N}).unsafe_leak()
    var d = alloc[Scalar[acc_dtype]]({length = m * N}).unsafe_leak()
    var rng = Rng(UInt64(0x9E3779B97F4A7C15) + UInt64(m * 131 + Int(random)))
    for i in range(m * K):
        var v = rng.uniform() if random else rng.small_int()
        a[unsafe_offset=i] = v.cast[in_dtype]()
    for i in range(K * N):
        var v = rng.uniform() if random else rng.small_int()
        b[unsafe_offset=i] = v.cast[in_dtype]()
    for i in range(m * N):
        var v = rng.uniform() if random else rng.small_int()
        c[unsafe_offset=i] = v.cast[acc_dtype]()
        d[unsafe_offset=i] = Scalar[acc_dtype](-999)
    for j in range(K // 2):
        for n in range(N):
            var lo = UInt32(
                bitcast[DType.uint16, 1](b[unsafe_offset=(2 * j) * N + n])
            )
            var hi = UInt32(
                bitcast[DType.uint16, 1](b[unsafe_offset=(2 * j + 1) * N + n])
            )
            bp[unsafe_offset=j * N + n] = Int32(lo | (hi << 16))

    var da = ctx.alloc_sync(UInt64(2 * m * K))
    var db = ctx.alloc_sync(UInt64(4 * (K // 2) * N))
    var dc = ctx.alloc_sync(UInt64(4 * m * N))
    var dd = ctx.alloc_sync(UInt64(4 * m * N))
    queue.copy_to_device(da.view(), a.unsafe_bitcast[UInt8]())
    comptime if packed:
        queue.copy_to_device(db.view(), bp.unsafe_bitcast[UInt8]())
    else:
        queue.copy_to_device(db.view(), b.unsafe_bitcast[UInt8]())
    queue.copy_to_device(dc.view(), c.unsafe_bitcast[UInt8]())
    queue.copy_to_device(dd.view(), d.unsafe_bitcast[UInt8]())
    var args: List[UInt64] = [
        ctx.memory_get_address(da),
        ctx.memory_get_address(db),
        ctx.memory_get_address(dc),
        ctx.memory_get_address(dd),
    ]
    comptime if packed:
        comptime kern = mma_packed_kernel[in_dtype]
        var f = load[type_of(kern), kern](ctx)
        launch(queue, f, 1, 16, args)
    else:
        comptime kern = mma_kernel[in_dtype, acc_dtype, m]
        var f = load[type_of(kern), kern](ctx)
        launch(queue, f, 1, 16, args)
    queue.copy_from_device(d.unsafe_bitcast[UInt8](), dd.view())
    queue.synchronize()

    var wrong = 0
    var worst = Float64(0)  # the largest |D - ref| / bound
    for r in range(m):
        for n in range(N):
            var want = Float64(c[unsafe_offset=r * N + n].cast[DType.float32]())
            var mag = abs(want)
            for k in range(K):
                var p = Float64(
                    a[unsafe_offset=r * K + k].cast[DType.float32]()
                ) * Float64(b[unsafe_offset=k * N + n].cast[DType.float32]())
                want += p
                mag += abs(p)
            var got = Float64(d[unsafe_offset=r * N + n].cast[DType.float32]())
            var err = abs(got - want)
            if random:
                var bound = 2.0 * eps_of[acc_dtype]() * mag
                worst = max(worst, err / bound)
                if err > bound:
                    wrong += 1
            elif err != 0:
                wrong += 1
    var name = String(
        type_name[in_dtype](),
        " -> ",
        type_name[acc_dtype](),
        " M=",
        m,
        " packed-B" if packed else "",
        " random" if random else " int",
    )
    var ok = wrong == 0
    print(
        "PASS" if ok else "FAIL",
        name,
        "wrong:",
        wrong,
        "of",
        m * N,
        "worst err/bound:",
        worst,
    )
    a.unsafe_free()
    b.unsafe_free()
    bp.unsafe_free()
    c.unsafe_free()
    d.unsafe_free()
    return ok


def run_type[
    in_dtype: DType, acc_dtype: DType
](mut ctx: GpuContext, queue: GpuQueue) raises -> Int:
    var failed = 0
    for random in [False, True]:
        failed += Int(
            not run_case[in_dtype, acc_dtype, 1, False](ctx, queue, random)
        )
        failed += Int(
            not run_case[in_dtype, acc_dtype, 2, False](ctx, queue, random)
        )
        failed += Int(
            not run_case[in_dtype, acc_dtype, 4, False](ctx, queue, random)
        )
        failed += Int(
            not run_case[in_dtype, acc_dtype, 8, False](ctx, queue, random)
        )
    return failed


def main() raises:
    var ctx_ref = open_context()
    var queue_ref = ctx_ref[].create_queue()
    ref ctx = ctx_ref[]
    ref queue = queue_ref[]
    var failed = 0
    failed += run_type[DType.float16, DType.float32](ctx, queue)
    failed += run_type[DType.bfloat16, DType.float32](ctx, queue)
    failed += run_type[DType.float16, DType.float16](ctx, queue)
    failed += run_type[DType.bfloat16, DType.bfloat16](ctx, queue)
    for random in [False, True]:
        failed += Int(
            not run_case[DType.float16, DType.float32, 8, True](
                ctx, queue, random
            )
        )
        failed += Int(
            not run_case[DType.bfloat16, DType.float32, 8, True](
                ctx, queue, random
            )
        )
    if failed == 0:
        print("PASS mma all")
    else:
        print("FAIL mma", failed, "cases")
