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

# Intel 2D block I/O (`max.gpu.memory.block_io_intel`) on the B70,
# through the HAL client, every result checked on the CPU.
#
# Every shape of the `cl_intel_subgroup_2d_block_io` tables, for 8-, 16- and
# 32-bit elements: load, load-transform (VNNI), load-transpose, store and
# prefetch (followed by a load), on a random matrix whose width (200 bytes)
# is less than its pitch (256 bytes), at three places:
#
#   inside   x = 4, y = 3: the whole block is in the matrix.
#   edge     x = the last 8 bytes' column, y = 4 rows from the bottom: the
#            block runs off the right and bottom edges (zero fill; stores
#            write only inside).
#   before   x = -4, y = -2: the block starts above and left of the matrix.
#
# One kernel per (operation, element size, block width, count) runs every
# block height of the table one after another. The expected value of each
# lane's element comes from a model of the extension's layout (lower columns
# to lower lanes; a 16-wide block gives lane l column l of each row, a wider
# one consecutive columns to each lane, a narrower one rows to groups of
# lanes; blocks one after another; transform packs rows into 32-bit words,
# transpose gives lane l row l (and l + 16) with one element per column), so
# a wrong layout fails. Last, a DPAS through mma() with A from a block load
# and B from a transform load.

from std.memory import alloc, bitcast
from std.sys import size_of

from max.gpu import lane_id
from max.gpu.compute.mma import mma
from max.gpu.memory.block_io_intel import (
    block_elements,
    block_load,
    block_load_transform,
    block_load_transpose,
    block_prefetch,
    block_store,
)

from hal_gpu import GpuContext, GpuQueue, launch, load, open_context

comptime ROWS = 70
comptime WIDTH_BYTES = 200
comptime PITCH_BYTES = 256

comptime LOAD = 0
comptime TRANSFORM = 1
comptime TRANSPOSE = 2
comptime STORE = 3
comptime PREFETCH = 4


def op_name(op: Int) -> String:
    if op == LOAD:
        return "load"
    if op == TRANSFORM:
        return "load_transform"
    if op == TRANSPOSE:
        return "load_transpose"
    if op == STORE:
        return "store"
    return "prefetch+load"


def lane_values(bw: Int, h: Int, cnt: Int) -> Int:
    return cnt * max(1, bw * h // 16)


def offset_of(bw: Int, cnt: Int, hmin: Int, h: Int) -> Int:
    """Where height `h`'s results start in a kernel's output (elements)."""
    var off = 0
    var k = hmin
    while k < h:
        off += 16 * lane_values(bw, k, cnt)
        k *= 2
    return off


def load_kernel[
    op: Int, dtype: DType, bw: Int, cnt: Int, hmin: Int, hmax: Int
](
    m: MutPointer[Scalar[dtype], MutAnyOrigin],
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    width: Int,
    height: Int,
    pitch: Int,
    x: Int,
    y: Int,
):
    var l = Int(lane_id())
    comptime for k in range(6):
        comptime h = 1 << k
        comptime if h >= hmin and h <= hmax:
            comptime n = block_elements[bw, h, cnt]()
            comptime off = offset_of(bw, cnt, hmin, h)
            var v: SIMD[dtype, n]
            comptime if op == TRANSFORM:
                v = block_load_transform[bw, h, cnt](
                    m, width, height, pitch, x, y
                )
            elif op == TRANSPOSE:
                v = block_load_transpose[bw, h, cnt](
                    m, width, height, pitch, x, y
                )
            else:
                comptime if op == PREFETCH:
                    block_prefetch[bw, h, cnt](m, width, height, pitch, x, y)
                v = block_load[bw, h, cnt](m, width, height, pitch, x, y)
            comptime for i in range(n):
                dst[unsafe_offset=off + l * n + i] = v[i]


# The prefetch-only shape (8-bit, 16 wide, 32 rows, count 1 and 2), then a
# load of the same rows.
def prefetch16_kernel(
    m: MutPointer[UInt8, MutAnyOrigin],
    dst: MutPointer[UInt8, MutAnyOrigin],
    width: Int,
    height: Int,
    pitch: Int,
    x: Int,
    y: Int,
):
    block_prefetch[16, 32, 1](m, width, height, pitch, x, y)
    block_prefetch[16, 32, 2](m, width, height, pitch, x, y)
    var v = block_load[32, 32, 1](m, width, height, pitch, x, y)
    comptime n = block_elements[32, 32]()
    var l = Int(lane_id())
    comptime for i in range(n):
        dst[unsafe_offset=l * n + i] = v[i]


def store_kernel[
    dtype: DType, bw: Int, h: Int
](
    m: MutPointer[Scalar[dtype], MutAnyOrigin],
    src: MutPointer[Scalar[dtype], MutAnyOrigin],
    width: Int,
    height: Int,
    pitch: Int,
    x: Int,
    y: Int,
):
    comptime n = block_elements[bw, h]()
    var l = Int(lane_id())
    var v = SIMD[dtype, n]()
    comptime for i in range(n):
        v[i] = src[unsafe_offset=l * n + i]
    block_store[bw, h](m, width, height, pitch, x, y, v)


struct Rng(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> UInt64:
        self.state ^= self.state >> 12
        self.state ^= self.state << 25
        self.state ^= self.state >> 27
        return self.state * 0x2545F4914F6CDD1D


def model(
    op: Int, bw: Int, h: Int, cnt: Int, lane: Int, idx: Int
) -> Tuple[Int, Int, Bool]:
    """The block row and column of element `idx` of lane `lane`, and whether
    it is part of the block (False: padding, reads as zero)."""
    var e = lane_values(bw, h, cnt)
    var b = idx // (e // cnt)
    var j = idx % (e // cnt)
    if op == TRANSFORM:
        # 16 wide: element j is row j of column `lane` (rows packed into
        # 32-bit words, lower rows in lower bits: the same bytes).
        return (j, b * bw + lane, True)
    if op == TRANSPOSE:
        # Column c of the block is element (h / 16) c + i of each lane;
        # lane l holds rows l + 16 i (measured on the B70, the same with
        # IGC master: for 32 rows not the extension's "consecutive
        # elements" rule, which would give rows 2l, 2l + 1).
        var per = h // 16
        return (lane + 16 * (j % per), b * bw + j // per, True)
    if bw >= 16:
        var per = bw // 16
        return (j // per, b * bw + lane * per + j % per, True)
    var k = 16 // bw
    var row = j * k + lane // bw
    return (row, b * bw + lane % bw, row < h)


def get(p: MutPointer[UInt8, _], bytes: Int, i: Int) -> UInt32:
    var v = UInt32(0)
    for k in range(bytes):
        v |= UInt32(p[unsafe_offset=i * bytes + k]) << UInt32(8 * k)
    return v


def run_load[
    op: Int, dtype: DType, bw: Int, cnt: Int, hmin: Int, hmax: Int
](mut ctx: GpuContext, queue: GpuQueue) raises -> Int:
    comptime bytes = size_of[dtype]()
    comptime cols = WIDTH_BYTES // bytes
    comptime pitch = PITCH_BYTES // bytes
    comptime total = offset_of(bw, cnt, hmin, 2 * hmax)
    var host = alloc[UInt8]({length = ROWS * PITCH_BYTES}).unsafe_leak()
    var rng = Rng(
        UInt64(0x9E3779B97F4A7C15) + UInt64(op * 1000 + bw * 10 + cnt)
    )
    for i in range(ROWS * PITCH_BYTES):
        host[unsafe_offset=i] = UInt8(rng.next() >> 56)
    var got = alloc[UInt8]({length = total * bytes}).unsafe_leak()
    var dm = ctx.alloc_sync(UInt64(ROWS * PITCH_BYTES))
    var dout = ctx.alloc_sync(UInt64(total * bytes))
    queue.copy_to_device(dm.view(), host)
    comptime kern = load_kernel[op, dtype, bw, cnt, hmin, hmax]
    var f = load[type_of(kern), kern](ctx)
    var failed = 0
    var xs: List[Int] = [4, cols - 8 // bytes, -4]
    var ys: List[Int] = [3, ROWS - 4, -2]
    var places: List[String] = ["inside", "edge", "before"]
    for p in range(3):
        var x = xs[p]
        var y = ys[p]
        queue.set_memory(dout.view(), 0xAB)
        launch(
            queue,
            f,
            1,
            16,
            [
                ctx.memory_get_address(dm),
                ctx.memory_get_address(dout),
                UInt64(cols),
                UInt64(ROWS),
                UInt64(pitch),
                UInt64(Int64(x)),
                UInt64(Int64(y)),
            ],
        )
        queue.copy_from_device(got, dout.view())
        queue.synchronize()
        var h = hmin
        while h <= hmax:
            var n = lane_values(bw, h, cnt)
            var off = offset_of(bw, cnt, hmin, h)
            var wrong = 0
            var zeros = 0
            for lane in range(16):
                for idx in range(n):
                    var rc = model(op, bw, h, cnt, lane, idx)
                    var row = y + rc[0]
                    var col = x + rc[1]
                    var want = UInt32(0)
                    if (
                        rc[2]
                        and row >= 0
                        and row < ROWS
                        and col >= 0
                        and col < cols
                    ):
                        want = get(host, bytes, row * pitch + col)
                    else:
                        zeros += 1
                    if get(got, bytes, off + lane * n + idx) != want:
                        wrong += 1
            var ok = wrong == 0
            if not ok:
                failed += 1
            print(
                "PASS" if ok else "FAIL",
                op_name(op),
                String(8 * bytes) + "-bit",
                String(bw) + "x" + String(h),
                "count",
                cnt,
                places[p],
                "wrong:",
                wrong,
                "of",
                16 * n,
                "zero-filled:",
                zeros,
            )
            h *= 2
    host.unsafe_free()
    got.unsafe_free()
    return failed


def run_store[
    dtype: DType, bw: Int, h: Int
](mut ctx: GpuContext, queue: GpuQueue) raises -> Int:
    comptime bytes = size_of[dtype]()
    comptime cols = WIDTH_BYTES // bytes
    comptime pitch = PITCH_BYTES // bytes
    comptime n = block_elements[bw, h]()
    var before = alloc[UInt8]({length = ROWS * PITCH_BYTES}).unsafe_leak()
    var got = alloc[UInt8]({length = ROWS * PITCH_BYTES}).unsafe_leak()
    var src = alloc[UInt8]({length = 16 * n * bytes}).unsafe_leak()
    var rng = Rng(UInt64(0x51ED270B27D9D8A3) + UInt64(bw * 100 + h * 7 + bytes))
    for i in range(ROWS * PITCH_BYTES):
        before[unsafe_offset=i] = UInt8(rng.next() >> 56)
    for i in range(16 * n * bytes):
        src[unsafe_offset=i] = UInt8(rng.next() >> 56)
    var dm = ctx.alloc_sync(UInt64(ROWS * PITCH_BYTES))
    var dsrc = ctx.alloc_sync(UInt64(16 * n * bytes))
    queue.copy_to_device(dsrc.view(), src)
    comptime kern = store_kernel[dtype, bw, h]
    var f = load[type_of(kern), kern](ctx)
    var failed = 0
    var xs: List[Int] = [4, cols - 8 // bytes, -4]
    var ys: List[Int] = [3, ROWS - 4, -2]
    var places: List[String] = ["inside", "edge", "before"]
    for p in range(3):
        var x = xs[p]
        var y = ys[p]
        queue.copy_to_device(dm.view(), before)
        launch(
            queue,
            f,
            1,
            16,
            [
                ctx.memory_get_address(dm),
                ctx.memory_get_address(dsrc),
                UInt64(cols),
                UInt64(ROWS),
                UInt64(pitch),
                UInt64(Int64(x)),
                UInt64(Int64(y)),
            ],
        )
        queue.copy_from_device(got, dm.view())
        queue.synchronize()
        # The expected matrix: the old one with the block's in-bounds
        # elements replaced (columns past the width, in the pitch, kept).
        var want = alloc[UInt8]({length = ROWS * PITCH_BYTES}).unsafe_leak()
        for i in range(ROWS * PITCH_BYTES):
            want[unsafe_offset=i] = before[unsafe_offset=i]
        var written = 0
        for lane in range(16):
            for idx in range(n):
                var rc = model(LOAD, bw, h, 1, lane, idx)
                var row = y + rc[0]
                var col = x + rc[1]
                if (
                    rc[2]
                    and row >= 0
                    and row < ROWS
                    and col >= 0
                    and col < cols
                ):
                    written += 1
                    for k in range(bytes):
                        want[
                            unsafe_offset=(row * pitch + col) * bytes + k
                        ] = src[unsafe_offset=(lane * n + idx) * bytes + k]
        var wrong = 0
        for i in range(ROWS * pitch):
            if get(got, bytes, i) != get(want, bytes, i):
                wrong += 1
        var ok = wrong == 0
        if not ok:
            failed += 1
        print(
            "PASS" if ok else "FAIL",
            "store",
            String(8 * bytes) + "-bit",
            String(bw) + "x" + String(h),
            places[p],
            "wrong:",
            wrong,
            "of",
            ROWS * pitch,
            "written:",
            written,
        )
        want.unsafe_free()
    before.unsafe_free()
    got.unsafe_free()
    src.unsafe_free()
    return failed


def run_prefetch16(mut ctx: GpuContext, queue: GpuQueue) raises -> Int:
    comptime n = block_elements[32, 32]()
    var host = alloc[UInt8]({length = ROWS * PITCH_BYTES}).unsafe_leak()
    var rng = Rng(UInt64(0x2545F4914F6CDD1D))
    for i in range(ROWS * PITCH_BYTES):
        host[unsafe_offset=i] = UInt8(rng.next() >> 56)
    var got = alloc[UInt8]({length = 16 * n}).unsafe_leak()
    var dm = ctx.alloc_sync(UInt64(ROWS * PITCH_BYTES))
    var dout = ctx.alloc_sync(UInt64(16 * n))
    queue.copy_to_device(dm.view(), host)
    var f = load[type_of(prefetch16_kernel), prefetch16_kernel](ctx)
    launch(
        queue,
        f,
        1,
        16,
        [
            ctx.memory_get_address(dm),
            ctx.memory_get_address(dout),
            UInt64(WIDTH_BYTES),
            UInt64(ROWS),
            UInt64(PITCH_BYTES),
            UInt64(4),
            UInt64(3),
        ],
    )
    queue.copy_from_device(got, dout.view())
    queue.synchronize()
    var wrong = 0
    for lane in range(16):
        for idx in range(n):
            var rc = model(LOAD, 32, 32, 1, lane, idx)
            var want = host[unsafe_offset=(3 + rc[0]) * PITCH_BYTES + 4 + rc[1]]
            if got[unsafe_offset=lane * n + idx] != want:
                wrong += 1
    print(
        "PASS" if wrong == 0 else "FAIL",
        "prefetch 8-bit 16x32 count 1 and 2, then load 32x32 wrong:",
        wrong,
        "of",
        16 * n,
    )
    host.unsafe_free()
    got.unsafe_free()
    return Int(wrong != 0)


# A (8 x 16) from a 16-wide 8-row block load, B (16 x 16) from a VNNI
# transform load, into mma(): the Intel GEMM's fragments.
def mma_kernel(
    a: MutPointer[Float16, MutAnyOrigin],
    b: MutPointer[Float16, MutAnyOrigin],
    d: MutPointer[Float32, MutAnyOrigin],
):
    var av = block_load[16, 8](a, 32, 8, 32, 0, 0)
    var bv = block_load_transform[16, 16](b, 32, 16, 32, 16, 0)
    var c = SIMD[DType.float32, 8](0)
    var dv = SIMD[DType.float32, 8]()
    mma(dv, av, bv, c)
    var l = Int(lane_id())
    comptime for r in range(8):
        d[unsafe_offset=r * 16 + l] = dv[r]


def run_mma(mut ctx: GpuContext, queue: GpuQueue) raises -> Int:
    # A is 8 x 32 (the block at column 0), B is 16 x 32 (the block at column
    # 16), both with 32-element (64-byte) rows; small integers, so exact.
    var a = alloc[Float16]({length = 8 * 32}).unsafe_leak()
    var b = alloc[Float16]({length = 16 * 32}).unsafe_leak()
    var d = alloc[Float32]({length = 8 * 16}).unsafe_leak()
    var rng = Rng(UInt64(77))
    for i in range(8 * 32):
        a[unsafe_offset=i] = Float16(Int((rng.next() >> 33) % 7) - 3)
    for i in range(16 * 32):
        b[unsafe_offset=i] = Float16(Int((rng.next() >> 33) % 7) - 3)
    var da = ctx.alloc_sync(UInt64(2 * 8 * 32))
    var db = ctx.alloc_sync(UInt64(2 * 16 * 32))
    var dd = ctx.alloc_sync(UInt64(4 * 8 * 16))
    queue.copy_to_device(da.view(), a.unsafe_bitcast[UInt8]())
    queue.copy_to_device(db.view(), b.unsafe_bitcast[UInt8]())
    var f = load[type_of(mma_kernel), mma_kernel](ctx)
    launch(
        queue,
        f,
        1,
        16,
        [
            ctx.memory_get_address(da),
            ctx.memory_get_address(db),
            ctx.memory_get_address(dd),
        ],
    )
    queue.copy_from_device(d.unsafe_bitcast[UInt8](), dd.view())
    queue.synchronize()
    var wrong = 0
    for r in range(8):
        for n in range(16):
            var want = Float32(0)
            for k in range(16):
                want += Float32(a[unsafe_offset=r * 32 + k]) * Float32(
                    b[unsafe_offset=k * 32 + 16 + n]
                )
            if d[unsafe_offset=r * 16 + n] != want:
                wrong += 1
    print(
        "PASS" if wrong == 0 else "FAIL",
        "mma with A from block_load and B from block_load_transform wrong:",
        wrong,
        "of 128",
    )
    a.unsafe_free()
    b.unsafe_free()
    d.unsafe_free()
    return Int(wrong != 0)


def main() raises:
    var ctx_ref = open_context()
    var queue_ref = ctx_ref[].create_queue()
    ref ctx = ctx_ref[]
    ref queue = queue_ref[]
    var failed = 0
    comptime u8 = DType.uint8
    comptime u16 = DType.uint16
    comptime u32 = DType.uint32
    # Loads, and the same shapes as prefetch then load.
    comptime for op in [LOAD, PREFETCH]:
        failed += run_load[op, u8, 32, 1, 1, 32](ctx, queue)
        failed += run_load[op, u8, 32, 2, 1, 32](ctx, queue)
        failed += run_load[op, u8, 16, 4, 8, 32](ctx, queue)
        failed += run_load[op, u16, 16, 1, 1, 32](ctx, queue)
        failed += run_load[op, u16, 16, 2, 1, 32](ctx, queue)
        failed += run_load[op, u32, 16, 1, 1, 32](ctx, queue)
        failed += run_load[op, u32, 8, 1, 1, 32](ctx, queue)
        failed += run_load[op, u32, 8, 2, 1, 32](ctx, queue)
    failed += run_prefetch16(ctx, queue)
    # bfloat16 and float16 are 16-bit loads too.
    failed += run_load[LOAD, DType.bfloat16, 16, 1, 8, 8](ctx, queue)
    failed += run_load[LOAD, DType.float16, 16, 2, 16, 16](ctx, queue)
    failed += run_load[TRANSFORM, u8, 16, 1, 32, 32](ctx, queue)
    failed += run_load[TRANSFORM, u8, 16, 2, 32, 32](ctx, queue)
    failed += run_load[TRANSFORM, u8, 16, 4, 32, 32](ctx, queue)
    failed += run_load[TRANSFORM, u16, 16, 1, 16, 32](ctx, queue)
    failed += run_load[TRANSFORM, u16, 16, 2, 16, 32](ctx, queue)
    failed += run_load[TRANSPOSE, u32, 8, 1, 16, 32](ctx, queue)
    comptime for h in [1, 2, 4, 8]:
        failed += run_store[u8, 16, h](ctx, queue)
        failed += run_store[u8, 32, h](ctx, queue)
        failed += run_store[u16, 16, h](ctx, queue)
        failed += run_store[u32, 16, h](ctx, queue)
    failed += run_mma(ctx, queue)
    if failed == 0:
        print("PASS block_io all")
    else:
        print("FAIL block_io", failed, "cases")
