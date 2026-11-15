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

# One program with kernels at sub-group size 16 and 32, every result checked
# on the CPU.
#
# The same kernels are compiled twice through the HAL client: for the B70's
# target (sub-group size 16) and for its `+sub-group-size-32` variant
# (`ArcProB70SG32`), and loaded into one context. At each size:
#
#   info        OpenCL's get_sub_group_size(), MAX's WARP_SIZE and lane_id()
#               in every work-item: the size the kernel runs at, the size its
#               code was written for, and the lane, must agree.
#   shuffles    warp.shuffle_down, shuffle_up, shuffle_xor and shuffle_idx at
#               every offset below the size, of a float32 (one word) and a
#               uint64 (two words), in a work-group of 128 work-items.
#   reductions  warp.sum, warp.max, warp.broadcast, lane_group_sum over the
#               whole sub-group and lane_group_max over 8 lanes.
#   ballot      warp.vote as uint32 and uint64 for 8 predicates.
#   block       block.sum and block.max (broadcast) in one work-group of 64,
#               320 and 1,024 work-items (at 16, 1,024 has more sub-groups
#               than lanes, the more-warps-than-lanes branch; at 32 it has
#               as many).
#
# Values are small integers in float32, so every sum is exact.

from std.ffi import external_call
from std.memory import alloc
from std.sys.info import CompilationTarget
from std.sys.defines import get_defined_int

from max.gpu import WARP_SIZE, lane_id, thread_idx
from max.gpu.primitives import block, warp
from max._plugin.intel.targets import ArcProB70, ArcProB70SG32

from hal_gpu import launch, load_for, open_context

comptime T16 = ArcProB70.target
comptime T32 = ArcProB70SG32.target

# Work-items in the warp-level kernels: 8 sub-groups of 16, 4 of 32.
comptime T = 128
# Offsets stored per kind; a kernel at 16 fills the first 16.
comptime MAX_LANES = 32
comptime KINDS = 4
comptime REDUCTIONS = 5
comptime PREDICATES = 8


@inline(.always)
def fvalue(tid: Int) -> Float32:
    return Float32((tid * 37) % 101 + tid // 16)


@inline(.always)
def uvalue(tid: Int) -> UInt64:
    # Distinct high and low words, so a swapped or repeated word shows.
    return (UInt64(tid + 1) << 32) | UInt64(0xA000 + tid * 3)


@inline(.always)
def predicate(k: Int, lane: Int) -> Bool:
    if k < 5:
        return (lane >> k) & 1 == 1
    if k == 5:
        return lane % 3 == 0
    if k == 6:
        return lane < 20
    return False


def info(o: MutPointer[UInt32, MutAnyOrigin]):
    var tid = Int(thread_idx.x)
    o[unsafe_offset=3 * tid] = external_call[
        "_Z18get_sub_group_sizev", UInt32
    ]()
    o[unsafe_offset=3 * tid + 1] = UInt32(WARP_SIZE)
    o[unsafe_offset=3 * tid + 2] = UInt32(lane_id())


def shuffles(
    f: MutPointer[Float32, MutAnyOrigin],
    u: MutPointer[UInt64, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    var v = fvalue(tid)
    var w = uvalue(tid)
    for off in range(WARP_SIZE):
        var o = UInt32(off)
        f[unsafe_offset=(0 * MAX_LANES + off) * T + tid] = warp.shuffle_down(
            v, o
        )
        f[unsafe_offset=(1 * MAX_LANES + off) * T + tid] = warp.shuffle_up(v, o)
        f[unsafe_offset=(2 * MAX_LANES + off) * T + tid] = warp.shuffle_xor(
            v, o
        )
        f[unsafe_offset=(3 * MAX_LANES + off) * T + tid] = warp.shuffle_idx(
            v, o
        )
        u[unsafe_offset=(0 * MAX_LANES + off) * T + tid] = warp.shuffle_down(
            w, o
        )
        u[unsafe_offset=(1 * MAX_LANES + off) * T + tid] = warp.shuffle_up(w, o)
        u[unsafe_offset=(2 * MAX_LANES + off) * T + tid] = warp.shuffle_xor(
            w, o
        )
        u[unsafe_offset=(3 * MAX_LANES + off) * T + tid] = warp.shuffle_idx(
            w, o
        )


def reductions(f: MutPointer[Float32, MutAnyOrigin]):
    var tid = Int(thread_idx.x)
    var v = fvalue(tid)
    f[unsafe_offset=0 * T + tid] = warp.sum(v)
    f[unsafe_offset=1 * T + tid] = warp.max(v)
    f[unsafe_offset=2 * T + tid] = warp.broadcast(v)
    f[unsafe_offset=3 * T + tid] = warp.lane_group_sum[num_lanes=WARP_SIZE](v)
    f[unsafe_offset=4 * T + tid] = warp.lane_group_max[num_lanes=8](v)


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
](s: MutPointer[Float32, MutAnyOrigin], m: MutPointer[Float32, MutAnyOrigin]):
    var tid = Int(thread_idx.x)
    s[unsafe_offset=tid] = block.sum[block_size=size, broadcast=True](
        fvalue(tid)
    )
    # Grows with the sub-group, so a dropped sub-group changes the max.
    m[unsafe_offset=tid] = block.max[block_size=size, broadcast=True](
        fvalue(tid) + Float32((tid // 16) * 200)
    )


def report(name: String, wrong: Int, first: String):
    if wrong == 0:
        print("PASS", name)
    else:
        print("FAIL", name, "wrong:", wrong, "first:", first)


def expected_shuffle(lanes: Int, kind: Int, off: Int, tid: Int) -> Int:
    """The work-item whose value `tid` reads, per MAX's default hooks."""
    var lane = tid % lanes
    var base = tid - lane
    var src: Int
    if kind == 0:
        src = lane + off if lane + off < lanes else lane
    elif kind == 1:
        src = lane - off if lane - off >= 0 else lane
    elif kind == 2:
        src = lane ^ off
    else:
        src = off
    return base + src


def group_reduce(lanes: Int, tid: Int, width: Int, is_max: Bool) -> Float32:
    """The reduction over the `width` consecutive lanes (a power of two) of
    `tid`'s sub-group that hold `tid`."""
    var lane = tid % lanes
    var first = tid - lane + (lane - lane % width)
    var acc = Float32(0)
    for other in range(first, first + width):
        acc = max(acc, fvalue(other)) if is_max else acc + fvalue(other)
    return acc


def check_size[lanes: Int, target: CompilationTarget]() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var tag = String(" at ", lanes)

    var i_host = alloc[UInt32]({length = 3 * T}).unsafe_leak()
    var i_dev = ctx[].alloc_sync(UInt64(4 * 3 * T))
    var f_info = load_for[type_of(info), info, target](ctx[])
    launch(queue[], f_info, 1, T, [ctx[].memory_get_address(i_dev)])
    queue[].copy_from_device(i_host.unsafe_bitcast[UInt8](), i_dev.view())
    queue[].synchronize()
    var wrong = 0
    var first = String()
    for tid in range(T):
        var got = (
            Int(i_host[unsafe_offset=3 * tid]),
            Int(i_host[unsafe_offset=3 * tid + 1]),
            Int(i_host[unsafe_offset=3 * tid + 2]),
        )
        if got[0] != lanes or got[1] != lanes or got[2] != tid % lanes:
            if wrong == 0:
                first = String(
                    "tid ",
                    tid,
                    " sub-group size ",
                    got[0],
                    " WARP_SIZE ",
                    got[1],
                    " lane ",
                    got[2],
                )
            wrong += 1
    report("sub-group size, WARP_SIZE, lane_id" + tag, wrong, first)

    comptime nf = KINDS * MAX_LANES * T
    var f_host = alloc[Float32]({length = nf}).unsafe_leak()
    var u_host = alloc[UInt64]({length = nf}).unsafe_leak()
    var f_dev = ctx[].alloc_sync(UInt64(4 * nf))
    var u_dev = ctx[].alloc_sync(UInt64(8 * nf))
    var f_shuffles = load_for[type_of(shuffles), shuffles, target](ctx[])
    launch(
        queue[],
        f_shuffles,
        1,
        T,
        [ctx[].memory_get_address(f_dev), ctx[].memory_get_address(u_dev)],
    )
    queue[].copy_from_device(f_host.unsafe_bitcast[UInt8](), f_dev.view())
    queue[].copy_from_device(u_host.unsafe_bitcast[UInt8](), u_dev.view())
    queue[].synchronize()
    var names = ["shuffle_down", "shuffle_up", "shuffle_xor", "shuffle_idx"]
    for kind in range(KINDS):
        wrong = 0
        first = String()
        for off in range(lanes):
            for tid in range(T):
                var i = (kind * MAX_LANES + off) * T + tid
                var src = expected_shuffle(lanes, kind, off, tid)
                if f_host[unsafe_offset=i] != fvalue(src) or u_host[
                    unsafe_offset=i
                ] != uvalue(src):
                    if wrong == 0:
                        first = String(
                            "offset ",
                            off,
                            " tid ",
                            tid,
                            " got ",
                            f_host[unsafe_offset=i],
                            " ",
                            u_host[unsafe_offset=i],
                            " want work-item ",
                            src,
                        )
                    wrong += 1
        report(
            names[kind] + " offsets 0-" + String(lanes - 1) + tag, wrong, first
        )

    comptime nr = REDUCTIONS * T
    var r_host = alloc[Float32]({length = nr}).unsafe_leak()
    var r_dev = ctx[].alloc_sync(UInt64(4 * nr))
    var f_red = load_for[type_of(reductions), reductions, target](ctx[])
    launch(queue[], f_red, 1, T, [ctx[].memory_get_address(r_dev)])
    queue[].copy_from_device(r_host.unsafe_bitcast[UInt8](), r_dev.view())
    queue[].synchronize()
    var rnames = [
        "warp.sum",
        "warp.max",
        "warp.broadcast",
        "lane_group_sum WARP_SIZE",
        "lane_group_max 8",
    ]
    for r in range(REDUCTIONS):
        wrong = 0
        first = String()
        for tid in range(T):
            var want: Float32
            if r == 0 or r == 3:
                want = group_reduce(lanes, tid, lanes, False)
            elif r == 1:
                want = group_reduce(lanes, tid, lanes, True)
            elif r == 2:
                want = fvalue(tid - tid % lanes)
            else:
                want = group_reduce(lanes, tid, 8, True)
            var got = r_host[unsafe_offset=r * T + tid]
            if got != want:
                if wrong == 0:
                    first = String("tid ", tid, " got ", got, " want ", want)
                wrong += 1
        report(rnames[r] + tag, wrong, first)

    comptime nb = PREDICATES * T
    var b32_host = alloc[UInt32]({length = nb}).unsafe_leak()
    var b64_host = alloc[UInt64]({length = nb}).unsafe_leak()
    var b32_dev = ctx[].alloc_sync(UInt64(4 * nb))
    var b64_dev = ctx[].alloc_sync(UInt64(8 * nb))
    var f_vote = load_for[type_of(ballots), ballots, target](ctx[])
    launch(
        queue[],
        f_vote,
        1,
        T,
        [ctx[].memory_get_address(b32_dev), ctx[].memory_get_address(b64_dev)],
    )
    queue[].copy_from_device(b32_host.unsafe_bitcast[UInt8](), b32_dev.view())
    queue[].copy_from_device(b64_host.unsafe_bitcast[UInt8](), b64_dev.view())
    queue[].synchronize()
    wrong = 0
    first = String()
    for k in range(PREDICATES):
        var want = UInt64(0)
        for lane in range(lanes):
            if predicate(k, lane):
                want |= UInt64(1) << UInt64(lane)
        for tid in range(T):
            var g32 = b32_host[unsafe_offset=k * T + tid]
            var g64 = b64_host[unsafe_offset=k * T + tid]
            if UInt64(g32) != want or g64 != want:
                if wrong == 0:
                    first = String(
                        "predicate ",
                        k,
                        " tid ",
                        tid,
                        " got ",
                        g32,
                        " ",
                        g64,
                        " want ",
                        want,
                    )
                wrong += 1
    report("vote uint32 uint64" + tag, wrong, first)

    check_block[64, lanes, target]()
    check_block[320, lanes, target]()
    # -D MAX_BLOCK=512 on a device whose work-groups stop at 512 (Xe-LP).
    comptime if get_defined_int["MAX_BLOCK", 1024]() >= 1024:
        check_block[1024, lanes, target]()


def check_block[size: Int, lanes: Int, target: CompilationTarget]() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var s_host = alloc[Float32]({length = size}).unsafe_leak()
    var m_host = alloc[Float32]({length = size}).unsafe_leak()
    var s_dev = ctx[].alloc_sync(UInt64(4 * size))
    var m_dev = ctx[].alloc_sync(UInt64(4 * size))
    var fn_ = load_for[type_of(block_reduce[size]), block_reduce[size], target](
        ctx[]
    )
    launch(
        queue[],
        fn_,
        1,
        size,
        [ctx[].memory_get_address(s_dev), ctx[].memory_get_address(m_dev)],
    )
    queue[].copy_from_device(s_host.unsafe_bitcast[UInt8](), s_dev.view())
    queue[].copy_from_device(m_host.unsafe_bitcast[UInt8](), m_dev.view())
    queue[].synchronize()
    var want_s = Float32(0)
    var want_m = Float32(0)
    for tid in range(size):
        want_s += fvalue(tid)
        want_m = max(want_m, fvalue(tid) + Float32((tid // 16) * 200))
    var wrong = 0
    var first = String()
    for tid in range(size):
        var s = s_host[unsafe_offset=tid]
        var m = m_host[unsafe_offset=tid]
        if s != want_s or m != want_m:
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
                )
            wrong += 1
    report(String("block.sum block.max ", size, " at ", lanes), wrong, first)


def main() raises:
    check_size[16, T16]()
    check_size[32, T32]()
