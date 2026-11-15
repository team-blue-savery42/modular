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

# The warp and block primitives MAX's rowwise algorithms use, on the B70's
# 16-lane sub-groups, every result checked on the CPU.
#
#   shuffles    warp.shuffle_down, shuffle_up, shuffle_xor and shuffle_idx at
#               every offset 0-15, of a float32 (one word) and a uint64 (two
#               words, split by MAX's shuffle_words), in a work-group of 4
#               sub-groups. Down/up/xor keep the lane's own value where the
#               source lane is outside the sub-group, as MAX's default hooks
#               define them.
#   reductions  warp.sum, warp.max and warp.broadcast, and lane_group_sum and
#               lane_group_max over 2, 4, 8 and 16 lanes 1 apart and over 2,
#               4 and 8 lanes 2 apart.
#   ballot      warp.vote as uint32 and uint64 for 16 predicates.
#   block       block.sum, block.max and block.min (broadcast) in one
#               work-group of 32 to 1,024 work-items, including 272, 320 and
#               576 (17, 20 and 36 sub-groups), whose last group of 16
#               sub-groups is partial. MAX's block reduce stores one value per
#               sub-group to shared memory, then the first sub-group reduces
#               them, one per lane: at 1,024 work-items there are 64 values
#               for 16 lanes, and before the fix lanes 0-15 read only the
#               first 16, so the sum covered a quarter of the work-group (the
#               max too). Rowwise softmax's block tier runs
#               1,024-wide work-groups on a few long rows, and its row sum
#               came out a quarter of the true one.
#               block.max and block.min each run twice: once on values of the
#               sign that keeps 0 out of the result (max of values >= 0, min
#               of values <= 0) and once on values of the other sign (max of
#               values <= -86,000, min of values >= 86,000), so a lane with
#               no sub-group's value that held 0 instead of the identity
#               (MIN_FINITE for max, MAX_FINITE for min) changes the result.
#   fused       block._block_reduce with 2 (min and max, as topk_fi's
#               _block_minmax) and 4 (two sums, a min and a max, as its
#               _block_reduce_pivot_bounds) reductions in one call, at 256,
#               272, 576 and 1,024 work-items: past 256 the shared slots are
#               padded, and reduction i's live at i * (sub-groups + padding),
#               so a layout where a later reduction's slots overlap an
#               earlier one's, or a broadcast that overwrites one, shows.
#               Each reduction has its own values.
#   scan        block.prefix_sum, inclusive and exclusive, at 32 to 1,024
#               work-items. Its first sub-group scanned one sub-group total
#               per lane, so at 1,024 work-items (topk_fi's default block)
#               sub-groups 17 and up added only the previous sub-group's own
#               total instead of the running prefix.
#
# Values are small integers in float32, so every sum is exact.

from std.memory import alloc
from std.sys.defines import get_defined_int
from std.utils.static_tuple import StaticTuple

from max.gpu import WARP_SIZE, lane_id, thread_idx
from max.gpu.primitives import block, warp

from hal_gpu import launch, load, open_context

comptime LANES = 16
# Work-items in the warp-level kernels: 4 sub-groups.
comptime T = 64
comptime OFFSETS = 16
# Shuffle kinds: down, up, xor, idx.
comptime KINDS = 4
# Reduction outputs per work-item (see `reductions`).
comptime REDUCTIONS = 18
comptime PREDICATES = 16


@inline(.always)
def fvalue(tid: Int) -> Float32:
    return Float32((tid * 37) % 101 + tid // 16)


@inline(.always)
def extreme(tid: Int) -> Float32:
    # Grows with the sub-group, so block.max's result (and block.min's, of
    # the negation) comes from the last sub-group, the one a dropped group
    # would lose.
    return fvalue(tid) + Float32((tid // LANES) * 200)


@inline(.always)
def opposite(tid: Int) -> Float32:
    # All below -86,000 (extreme() < 14,000 up to 1,024 work-items), so a
    # stray 0 wins block.max (and its negation's block.min), yet still
    # growing with the sub-group like extreme(). Exact in float32.
    return extreme(tid) - Float32(100000)


@inline(.always)
def uvalue(tid: Int) -> UInt64:
    # Distinct high and low words, so a swapped or repeated word shows.
    return (UInt64(tid + 1) << 32) | UInt64(0xA000 + tid * 3)


@inline(.always)
def predicate(k: Int, lane: Int) -> Bool:
    if k < 4:
        return (lane >> k) & 1 == 1
    if k < 8:
        return lane % (k - 1) == 0
    if k == 15:
        return False
    return lane < 2 * (k - 7)


def shuffles(
    f: MutPointer[Float32, MutAnyOrigin],
    u: MutPointer[UInt64, MutAnyOrigin],
    lanes: MutPointer[Int32, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    var v = fvalue(tid)
    var w = uvalue(tid)
    lanes[unsafe_offset=tid] = Int32(lane_id())
    for off in range(OFFSETS):
        var o = UInt32(off)
        f[unsafe_offset=(0 * OFFSETS + off) * T + tid] = warp.shuffle_down(v, o)
        f[unsafe_offset=(1 * OFFSETS + off) * T + tid] = warp.shuffle_up(v, o)
        f[unsafe_offset=(2 * OFFSETS + off) * T + tid] = warp.shuffle_xor(v, o)
        f[unsafe_offset=(3 * OFFSETS + off) * T + tid] = warp.shuffle_idx(v, o)
        u[unsafe_offset=(0 * OFFSETS + off) * T + tid] = warp.shuffle_down(w, o)
        u[unsafe_offset=(1 * OFFSETS + off) * T + tid] = warp.shuffle_up(w, o)
        u[unsafe_offset=(2 * OFFSETS + off) * T + tid] = warp.shuffle_xor(w, o)
        u[unsafe_offset=(3 * OFFSETS + off) * T + tid] = warp.shuffle_idx(w, o)


def reductions(f: MutPointer[Float32, MutAnyOrigin]):
    var tid = Int(thread_idx.x)
    var v = fvalue(tid)
    f[unsafe_offset=0 * T + tid] = warp.sum(v)
    f[unsafe_offset=1 * T + tid] = warp.max(v)
    f[unsafe_offset=2 * T + tid] = warp.broadcast(v)
    f[unsafe_offset=3 * T + tid] = warp.lane_group_sum[num_lanes=2](v)
    f[unsafe_offset=4 * T + tid] = warp.lane_group_sum[num_lanes=4](v)
    f[unsafe_offset=5 * T + tid] = warp.lane_group_sum[num_lanes=8](v)
    f[unsafe_offset=6 * T + tid] = warp.lane_group_max[num_lanes=2](v)
    f[unsafe_offset=7 * T + tid] = warp.lane_group_max[num_lanes=4](v)
    f[unsafe_offset=8 * T + tid] = warp.lane_group_max[num_lanes=8](v)
    f[unsafe_offset=9 * T + tid] = warp.lane_group_sum[num_lanes=2, stride=2](v)
    f[unsafe_offset=10 * T + tid] = warp.lane_group_sum[num_lanes=4, stride=2](
        v
    )
    f[unsafe_offset=11 * T + tid] = warp.lane_group_max[num_lanes=8, stride=2](
        v
    )
    f[unsafe_offset=12 * T + tid] = warp.lane_group_sum[num_lanes=LANES](v)
    f[unsafe_offset=13 * T + tid] = warp.lane_group_max[num_lanes=LANES](v)
    f[unsafe_offset=14 * T + tid] = Float32(WARP_SIZE)
    f[unsafe_offset=15 * T + tid] = warp.lane_group_sum[num_lanes=8, stride=2](
        v
    )
    f[unsafe_offset=16 * T + tid] = warp.lane_group_max[num_lanes=2, stride=2](
        v
    )
    f[unsafe_offset=17 * T + tid] = warp.lane_group_max[num_lanes=4, stride=2](
        v
    )


def ballots(
    m32: MutPointer[UInt32, MutAnyOrigin],
    m64: MutPointer[UInt64, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    for k in range(PREDICATES):
        m32[unsafe_offset=k * T + tid] = warp.vote[DType.uint32](
            predicate(k, lane)
        )
        m64[unsafe_offset=k * T + tid] = warp.vote[DType.uint64](
            predicate(k, lane)
        )


def block_reduce[
    size: Int
](
    s: MutPointer[Float32, MutAnyOrigin],
    m: MutPointer[Float32, MutAnyOrigin],
    n: MutPointer[Float32, MutAnyOrigin],
    m2: MutPointer[Float32, MutAnyOrigin],
    n2: MutPointer[Float32, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    var v = fvalue(tid)
    s[unsafe_offset=tid] = block.sum[block_size=size, broadcast=True](v)
    m[unsafe_offset=tid] = block.max[block_size=size, broadcast=True](
        extreme(tid)
    )
    n[unsafe_offset=tid] = block.min[block_size=size, broadcast=True](
        -extreme(tid)
    )
    m2[unsafe_offset=tid] = block.max[block_size=size, broadcast=True](
        opposite(tid)
    )
    n2[unsafe_offset=tid] = block.min[block_size=size, broadcast=True](
        -opposite(tid)
    )


@inline(.always)
def fused_value(r: Int, tid: Int) -> Float32:
    """Reduction `r`'s value at `tid` in `block_fused`: two sums of different
    values, then a min and a max whose identities a stray 0 would beat."""
    if r == 0:
        return fvalue(tid)
    if r == 1:
        return Float32(2) * fvalue(tid) + Float32(1)
    if r == 2:
        return -opposite(tid)
    return opposite(tid)


def block_fused[
    size: Int
](
    two: MutPointer[Float32, MutAnyOrigin],
    four: MutPointer[Float32, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)

    @inline(.always)
    def min_max[
        dtype: DType, width: SIMDLength, reduction_idx: Int
    ](v: SIMD[dtype, width]) -> Scalar[dtype]:
        comptime if reduction_idx == 0:
            return warp.min(v)
        else:
            return warp.max(v)

    var mm = block._block_reduce[size, broadcast=True](
        StaticTuple[Float32, 2](fused_value(2, tid), fused_value(3, tid)),
        initial_vals=StaticTuple[Float32, 2](
            Float32.MAX_FINITE, Float32.MIN_FINITE
        ),
        warp_reduce_fn=min_max,
    )
    two[unsafe_offset=2 * tid] = mm[0]
    two[unsafe_offset=2 * tid + 1] = mm[1]

    @inline(.always)
    def sums_min_max[
        dtype: DType, width: SIMDLength, reduction_idx: Int
    ](v: SIMD[dtype, width]) -> Scalar[dtype]:
        comptime if reduction_idx < 2:
            return warp.sum(v)
        elif reduction_idx == 2:
            return warp.min(v)
        else:
            return warp.max(v)

    var r = block._block_reduce[size, broadcast=True](
        StaticTuple[Float32, 4](
            fused_value(0, tid),
            fused_value(1, tid),
            fused_value(2, tid),
            fused_value(3, tid),
        ),
        initial_vals=StaticTuple[Float32, 4](
            0, 0, Float32.MAX_FINITE, Float32.MIN_FINITE
        ),
        warp_reduce_fn=sums_min_max,
    )
    comptime for i in range(4):
        four[unsafe_offset=4 * tid + i] = r[i]


def block_scan[
    size: Int
](
    inclusive: MutPointer[Float32, MutAnyOrigin],
    exclusive: MutPointer[Float32, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    var v = fvalue(tid)
    inclusive[unsafe_offset=tid] = block.prefix_sum[block_size=size](v)
    exclusive[unsafe_offset=tid] = block.prefix_sum[
        block_size=size, exclusive=True
    ](v)


def report(name: String, wrong: Int, first: String):
    if wrong == 0:
        print("PASS", name)
    else:
        print("FAIL", name, "wrong:", wrong, "first:", first)


def expected_shuffle(kind: Int, off: Int, tid: Int) -> Int:
    """The work-item whose value `tid` reads, per MAX's default hooks."""
    var lane = tid % LANES
    var base = tid - lane
    var src: Int
    if kind == 0:
        src = lane + off if lane + off < LANES else lane
    elif kind == 1:
        src = lane - off if lane - off >= 0 else lane
    elif kind == 2:
        src = lane ^ off
    else:
        src = off
    return base + src


def group_reduce(tid: Int, lanes: Int, stride: Int, is_max: Bool) -> Float32:
    """The lane_group_* result over `lanes` lanes `stride` apart: the
    butterfly reaches the lanes whose IDs differ from `tid`'s only in the bits
    of (lanes - 1) * stride."""
    var lane = tid % LANES
    var base = tid - lane
    var mask = (lanes - 1) * stride
    var acc = Float32(0)
    for other in range(LANES):
        if (other & ~mask) == (lane & ~mask):
            var x = fvalue(base + other)
            acc = max(acc, x) if is_max else acc + x
    return acc


def check_shuffles() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    comptime nf = KINDS * OFFSETS * T
    var f_host = alloc[Float32]({length = nf}).unsafe_leak()
    var u_host = alloc[UInt64]({length = nf}).unsafe_leak()
    var l_host = alloc[Int32]({length = T}).unsafe_leak()
    var f_dev = ctx[].alloc_sync(UInt64(4 * nf))
    var u_dev = ctx[].alloc_sync(UInt64(8 * nf))
    var l_dev = ctx[].alloc_sync(UInt64(4 * T))
    var fn_ = load[type_of(shuffles), shuffles](ctx[])
    launch(
        queue[],
        fn_,
        1,
        T,
        [
            ctx[].memory_get_address(f_dev),
            ctx[].memory_get_address(u_dev),
            ctx[].memory_get_address(l_dev),
        ],
    )
    queue[].copy_from_device(f_host.unsafe_bitcast[UInt8](), f_dev.view())
    queue[].copy_from_device(u_host.unsafe_bitcast[UInt8](), u_dev.view())
    queue[].copy_from_device(l_host.unsafe_bitcast[UInt8](), l_dev.view())
    queue[].synchronize()
    var lane_wrong = 0
    for tid in range(T):
        if Int(l_host[unsafe_offset=tid]) != tid % LANES:
            lane_wrong += 1
    report("lane_id", lane_wrong, "")
    var names = ["shuffle_down", "shuffle_up", "shuffle_xor", "shuffle_idx"]
    for kind in range(KINDS):
        var fw = 0
        var uw = 0
        var first = String()
        for off in range(OFFSETS):
            for tid in range(T):
                var i = (kind * OFFSETS + off) * T + tid
                var src = expected_shuffle(kind, off, tid)
                if f_host[unsafe_offset=i] != fvalue(src):
                    if fw + uw == 0:
                        first = String(
                            "float32 offset ",
                            off,
                            " tid ",
                            tid,
                            " got ",
                            f_host[unsafe_offset=i],
                            " want ",
                            fvalue(src),
                        )
                    fw += 1
                if u_host[unsafe_offset=i] != uvalue(src):
                    if fw + uw == 0:
                        first = String(
                            "uint64 offset ",
                            off,
                            " tid ",
                            tid,
                            " got ",
                            u_host[unsafe_offset=i],
                            " want ",
                            uvalue(src),
                        )
                    uw += 1
        report(names[kind] + " offsets 0-15 float32 uint64", fw + uw, first)


def check_reductions() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    comptime nf = REDUCTIONS * T
    var f_host = alloc[Float32]({length = nf}).unsafe_leak()
    var f_dev = ctx[].alloc_sync(UInt64(4 * nf))
    var fn_ = load[type_of(reductions), reductions](ctx[])
    launch(queue[], fn_, 1, T, [ctx[].memory_get_address(f_dev)])
    queue[].copy_from_device(f_host.unsafe_bitcast[UInt8](), f_dev.view())
    queue[].synchronize()
    var names = [
        "warp.sum",
        "warp.max",
        "warp.broadcast",
        "lane_group_sum 2",
        "lane_group_sum 4",
        "lane_group_sum 8",
        "lane_group_max 2",
        "lane_group_max 4",
        "lane_group_max 8",
        "lane_group_sum 2 stride 2",
        "lane_group_sum 4 stride 2",
        "lane_group_max 8 stride 2",
        "lane_group_sum 16",
        "lane_group_max 16",
        "WARP_SIZE",
        "lane_group_sum 8 stride 2",
        "lane_group_max 2 stride 2",
        "lane_group_max 4 stride 2",
    ]
    for r in range(REDUCTIONS):
        var wrong = 0
        var first = String()
        for tid in range(T):
            var want: Float32
            var lane = tid % LANES
            if r == 0 or r == 12:
                want = group_reduce(tid, LANES, 1, False)
            elif r == 1 or r == 13:
                want = group_reduce(tid, LANES, 1, True)
            elif r == 2:
                want = fvalue(tid - lane)
            elif r <= 5:
                want = group_reduce(tid, 1 << (r - 2), 1, False)
            elif r <= 8:
                want = group_reduce(tid, 1 << (r - 5), 1, True)
            elif r == 9:
                want = group_reduce(tid, 2, 2, False)
            elif r == 10:
                want = group_reduce(tid, 4, 2, False)
            elif r == 11:
                want = group_reduce(tid, 8, 2, True)
            elif r == 14:
                want = Float32(LANES)
            elif r == 15:
                want = group_reduce(tid, 8, 2, False)
            elif r == 16:
                want = group_reduce(tid, 2, 2, True)
            else:
                want = group_reduce(tid, 4, 2, True)
            var got = f_host[unsafe_offset=r * T + tid]
            if got != want:
                if wrong == 0:
                    first = String("tid ", tid, " got ", got, " want ", want)
                wrong += 1
        report(names[r], wrong, first)


def check_ballots() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    comptime n = PREDICATES * T
    var m32_host = alloc[UInt32]({length = n}).unsafe_leak()
    var m64_host = alloc[UInt64]({length = n}).unsafe_leak()
    var m32_dev = ctx[].alloc_sync(UInt64(4 * n))
    var m64_dev = ctx[].alloc_sync(UInt64(8 * n))
    var fn_ = load[type_of(ballots), ballots](ctx[])
    launch(
        queue[],
        fn_,
        1,
        T,
        [
            ctx[].memory_get_address(m32_dev),
            ctx[].memory_get_address(m64_dev),
        ],
    )
    queue[].copy_from_device(m32_host.unsafe_bitcast[UInt8](), m32_dev.view())
    queue[].copy_from_device(m64_host.unsafe_bitcast[UInt8](), m64_dev.view())
    queue[].synchronize()
    var wrong = 0
    var first = String()
    for k in range(PREDICATES):
        var want = UInt64(0)
        for lane in range(LANES):
            if predicate(k, lane):
                want |= UInt64(1) << UInt64(lane)
        for tid in range(T):
            var got32 = m32_host[unsafe_offset=k * T + tid]
            var got64 = m64_host[unsafe_offset=k * T + tid]
            if UInt64(got32) != want or got64 != want:
                if wrong == 0:
                    first = String(
                        "predicate ",
                        k,
                        " tid ",
                        tid,
                        " got ",
                        got32,
                        " ",
                        got64,
                        " want ",
                        want,
                    )
                wrong += 1
    report("vote uint32 uint64 predicates 0-15", wrong, first)


def check_block[size: Int]() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var s_host = alloc[Float32]({length = size}).unsafe_leak()
    var m_host = alloc[Float32]({length = size}).unsafe_leak()
    var n_host = alloc[Float32]({length = size}).unsafe_leak()
    var m2_host = alloc[Float32]({length = size}).unsafe_leak()
    var n2_host = alloc[Float32]({length = size}).unsafe_leak()
    var s_dev = ctx[].alloc_sync(UInt64(4 * size))
    var m_dev = ctx[].alloc_sync(UInt64(4 * size))
    var n_dev = ctx[].alloc_sync(UInt64(4 * size))
    var m2_dev = ctx[].alloc_sync(UInt64(4 * size))
    var n2_dev = ctx[].alloc_sync(UInt64(4 * size))
    var fn_ = load[type_of(block_reduce[size]), block_reduce[size]](ctx[])
    launch(
        queue[],
        fn_,
        1,
        size,
        [
            ctx[].memory_get_address(s_dev),
            ctx[].memory_get_address(m_dev),
            ctx[].memory_get_address(n_dev),
            ctx[].memory_get_address(m2_dev),
            ctx[].memory_get_address(n2_dev),
        ],
    )
    queue[].copy_from_device(s_host.unsafe_bitcast[UInt8](), s_dev.view())
    queue[].copy_from_device(m_host.unsafe_bitcast[UInt8](), m_dev.view())
    queue[].copy_from_device(n_host.unsafe_bitcast[UInt8](), n_dev.view())
    queue[].copy_from_device(m2_host.unsafe_bitcast[UInt8](), m2_dev.view())
    queue[].copy_from_device(n2_host.unsafe_bitcast[UInt8](), n2_dev.view())
    queue[].synchronize()
    var want_s = Float32(0)
    var want_m = extreme(0)
    var want_n = -extreme(0)
    var want_m2 = opposite(0)
    var want_n2 = -opposite(0)
    for tid in range(size):
        want_s += fvalue(tid)
        want_m = max(want_m, extreme(tid))
        want_n = min(want_n, -extreme(tid))
        want_m2 = max(want_m2, opposite(tid))
        want_n2 = min(want_n2, -opposite(tid))
    var wrong = 0
    var first = String()
    for tid in range(size):
        var s = s_host[unsafe_offset=tid]
        var m = m_host[unsafe_offset=tid]
        var mn = n_host[unsafe_offset=tid]
        var m2 = m2_host[unsafe_offset=tid]
        var n2 = n2_host[unsafe_offset=tid]
        if (
            s != want_s
            or m != want_m
            or mn != want_n
            or m2 != want_m2
            or n2 != want_n2
        ):
            if wrong == 0:
                first = String(
                    "tid ",
                    tid,
                    " sum ",
                    s,
                    " want ",
                    want_s,
                    " max ",
                    m,
                    " want ",
                    want_m,
                    " min ",
                    mn,
                    " want ",
                    want_n,
                    " max(<0) ",
                    m2,
                    " want ",
                    want_m2,
                    " min(>0) ",
                    n2,
                    " want ",
                    want_n2,
                )
            wrong += 1
    report(
        String("block.sum block.max block.min (both signs) ", size),
        wrong,
        first,
    )


def check_fused[size: Int]() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var two_host = alloc[Float32]({length = 2 * size}).unsafe_leak()
    var four_host = alloc[Float32]({length = 4 * size}).unsafe_leak()
    var two_dev = ctx[].alloc_sync(UInt64(8 * size))
    var four_dev = ctx[].alloc_sync(UInt64(16 * size))
    var fn_ = load[type_of(block_fused[size]), block_fused[size]](ctx[])
    launch(
        queue[],
        fn_,
        1,
        size,
        [
            ctx[].memory_get_address(two_dev),
            ctx[].memory_get_address(four_dev),
        ],
    )
    queue[].copy_from_device(two_host.unsafe_bitcast[UInt8](), two_dev.view())
    queue[].copy_from_device(four_host.unsafe_bitcast[UInt8](), four_dev.view())
    queue[].synchronize()
    var want = [Float32(0), Float32(0), fused_value(2, 0), fused_value(3, 0)]
    for tid in range(size):
        want[0] += fused_value(0, tid)
        want[1] += fused_value(1, tid)
        want[2] = min(want[2], fused_value(2, tid))
        want[3] = max(want[3], fused_value(3, tid))
    var wrong = 0
    var first = String()
    for tid in range(size):
        for i in range(6):
            var got = (
                two_host[unsafe_offset=2 * tid + i] if i
                < 2 else four_host[unsafe_offset=4 * tid + i - 2]
            )
            var w = want[i + 2] if i < 2 else want[i - 2]
            if got != w:
                if wrong == 0:
                    first = String(
                        "tid ", tid, " output ", i, " got ", got, " want ", w
                    )
                wrong += 1
    report(String("block._block_reduce fused 2 and 4 ", size), wrong, first)


def check_scan[size: Int]() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var i_host = alloc[Float32]({length = size}).unsafe_leak()
    var e_host = alloc[Float32]({length = size}).unsafe_leak()
    var i_dev = ctx[].alloc_sync(UInt64(4 * size))
    var e_dev = ctx[].alloc_sync(UInt64(4 * size))
    var fn_ = load[type_of(block_scan[size]), block_scan[size]](ctx[])
    launch(
        queue[],
        fn_,
        1,
        size,
        [ctx[].memory_get_address(i_dev), ctx[].memory_get_address(e_dev)],
    )
    queue[].copy_from_device(i_host.unsafe_bitcast[UInt8](), i_dev.view())
    queue[].copy_from_device(e_host.unsafe_bitcast[UInt8](), e_dev.view())
    queue[].synchronize()
    var wrong = 0
    var first = String()
    var running = Float32(0)
    for tid in range(size):
        var want_e = running
        running += fvalue(tid)
        var got_i = i_host[unsafe_offset=tid]
        var got_e = e_host[unsafe_offset=tid]
        if got_i != running or got_e != want_e:
            if wrong == 0:
                first = String(
                    "tid ",
                    tid,
                    " inclusive ",
                    got_i,
                    " want ",
                    running,
                    " exclusive ",
                    got_e,
                    " want ",
                    want_e,
                )
            wrong += 1
    report(String("block.prefix_sum inclusive exclusive ", size), wrong, first)


def main() raises:
    check_shuffles()
    check_reductions()
    check_ballots()
    check_block[32]()
    check_block[64]()
    check_block[256]()
    check_block[272]()
    check_block[320]()
    check_block[512]()
    # -D MAX_BLOCK=512 on a device whose work-groups stop at 512 (Xe-LP).
    comptime if get_defined_int["MAX_BLOCK", 1024]() >= 1024:
        check_block[576]()
        check_block[1024]()
    check_fused[256]()
    check_fused[272]()
    comptime if get_defined_int["MAX_BLOCK", 1024]() >= 1024:
        check_fused[576]()
        check_fused[1024]()
    check_scan[32]()
    check_scan[256]()
    check_scan[272]()
    check_scan[512]()
    comptime if get_defined_int["MAX_BLOCK", 1024]() >= 1024:
        check_scan[576]()
        check_scan[1024]()
