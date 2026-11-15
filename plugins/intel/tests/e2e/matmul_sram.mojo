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

# MAX's test_matmul_1_sram kernel (examples/
# test_matmul_1_sram.mojo) on the HAL client, 513 x 502 x 511 in 32 x 32
# work-groups, with every element of C checked on the CPU (MAX's test checks
# only the last 10 x 10).
#
# The kernel is a loop over K tiles: each work-item stores one element of A
# and one of B to shared memory, loaded only where its row (col) is inside the
# matrix, then waits at a barrier, sums a row by a column, and waits again.
# The loads' conditions are loop invariant, and before the fix LLVM's -O3 loop
# unswitching copied the loop once per value of them: the barrier wasn't
# `convergent`, so it let the work-items of an edge work-group (and of one
# sub-group) wait at different barriers, and some read shared memory before
# the others had written it. A run gave 1,030 wrong elements, all in the edge
# groups: 415 for 511 in the first sub-group of the right edge's top two
# groups, and 496 in the 6 elements (row 512, columns 496-501) of the corner
# group that MAX's test reports.
#
# Two variants:
#   race_free  shared memory indexed with stride 32 in every tile: no race,
#              so every element must be K. Its loads still go past A and B:
#              tile_and_unswitch runs the 31-wide last tile (offset 480) as a
#              full tile, which loads without the `< K` checks, so localRow 31
#              loads B's row K (up to 502 floats past its end) and work-item
#              (row 512, localCol 31) loads A[M * K]. With stride 32 those
#              values are stored where the k loop (k < 31) never reads them.
#   max        MAX's kernel as written: its last tile is 31 wide (K % 32), and
#              the tile function's `tile_size` argument shadows the comptime
#              one, so that tile is stored with stride 31: (r, 31) and
#              (r + 1, 0) store to the same element, and work-items read B's
#              row K, past its end. That is a race on any GPU (Modular's code,
#              upstream), and it can only make the first column of the right
#              edge groups (whose collision partner, a column past N, stores
#              0) short by up to 30, and the last column of a group short by
#              1. Anything else wrong, or anything in MAX's 10 x 10 corner, is
#              counted as `unexplained`. The max variant then prints where
#              the explained elements are: how many short last columns, in
#              how many groups and column groups, and how many short first
#              columns, with the lowest value.
#
# Both variants are MAX's loads as written, so the device buffers of A and B
# carry one extra zeroed row (A_ROWS, B_ROWS) for the reads past their ends:
# every load stays inside an allocation, whatever the allocator pads, so a
# variant can't page-fault and leave the next GPU process hanging.

from std.math import align_down, ceildiv
from std.memory import alloc, unsafe_stack_allocation
from std.algorithm.functional import tile_and_unswitch
from max.gpu import global_idx, thread_idx
from max.gpu.sync import barrier

from hal_gpu import launch, load, open_context

comptime tile_size = 32
comptime M = 513
comptime N = 502
comptime K = 511
# Rows of A and B as allocated: one more than the matrices', zeroed (see the
# top). The last tile's loads reach A[M * K] and B[K * N + N - 1] at most.
comptime A_ROWS = M + 1
comptime B_ROWS = K + 1


def matmul_sram[
    race_free: Bool
](
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    c: MutPointer[Float32, MutAnyOrigin],
):
    var a_shared = unsafe_stack_allocation[
        tile_size * tile_size, DType.float32, address_space=.SHARED
    ]()
    var b_shared = unsafe_stack_allocation[
        tile_size * tile_size, DType.float32, address_space=.SHARED
    ]()
    var col = global_idx.x
    var row = global_idx.y
    var localCol = thread_idx.x
    var localRow = thread_idx.y
    var result = Float32(0.0)
    var K_roundbytile = align_down(K, tile_size)
    var K_remainder = K - K_roundbytile if K - K_roundbytile > 0 else 1

    @inline(.always)
    def update_tile[
        full_tile: Bool
    ](offset: Int, end: Int, tile_size: Int) {
        var localCol,
        var a,
        var row,
        var a_shared,
        var localRow,
        var col,
        var b,
        var b_shared,
        mut result,
        imm,
    }:
        # MAX's kernel indexes with the argument `tile_size` (31 in the last
        # tile); the race-free variant always with 32.
        var stride = 32 if race_free else tile_size
        var a_val: Float32
        comptime if not full_tile:
            a_val = a[unsafe_offset=row * K + offset + localCol] if (
                row < M and offset + localCol < K
            ) else 0.0
        else:
            a_val = (
                a[unsafe_offset=row * K + offset + localCol] if row < M else 0.0
            )
        a_shared[localRow * stride + localCol] = a_val

        var b_val: Float32
        comptime if not full_tile:
            b_val = b[unsafe_offset=(offset + localRow) * N + col] if (
                col < N and offset + localRow < K
            ) else 0.0
        else:
            b_val = (
                b[unsafe_offset=(offset + localRow) * N + col] if col
                < N else 0.0
            )
        b_shared[localRow * stride + localCol] = b_val

        barrier()
        for k in range(tile_size):
            result += a_shared.unsafe_load(
                localRow * stride + k
            ) * b_shared.unsafe_load(k * stride + localCol)
        barrier()

    tile_and_unswitch(
        0, K, tile_size, K_remainder, workgroup_function=update_tile
    )

    if row < M and col < N:
        c[unsafe_offset=row * N + col] = result


def explained(i: Int, j: Int, v: Float32) -> Bool:
    """Whether MAX's own race (see the top) can give `v` at C[i, j]."""
    var lc = j % tile_size
    if lc == tile_size - 1:
        return v == Float32(K - 1)
    if lc == 0 and j >= align_down(N, tile_size):
        return v >= Float32(K - 30) and v < Float32(K)
    return False


def run[race_free: Bool](name: String) raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var a_host = alloc[Float32]({length = A_ROWS * K}).unsafe_leak()
    var b_host = alloc[Float32]({length = B_ROWS * N}).unsafe_leak()
    var c_host = alloc[Float32]({length = M * N}).unsafe_leak()
    for i in range(A_ROWS * K):
        a_host[unsafe_offset=i] = 1 if i < M * K else 0
    for i in range(B_ROWS * N):
        b_host[unsafe_offset=i] = 1 if i < K * N else 0
    var a_dev = ctx[].alloc_sync(UInt64(4 * A_ROWS * K))
    var b_dev = ctx[].alloc_sync(UInt64(4 * B_ROWS * N))
    var c_dev = ctx[].alloc_sync(UInt64(4 * M * N))
    queue[].copy_to_device(a_dev.view(), a_host.unsafe_bitcast[UInt8]())
    queue[].copy_to_device(b_dev.view(), b_host.unsafe_bitcast[UInt8]())
    queue[].set_memory(c_dev.view(), value=UInt8(0))
    var f = load[type_of(matmul_sram[race_free]), matmul_sram[race_free]](ctx[])
    launch(
        queue[],
        f,
        ceildiv(N, tile_size),
        tile_size,
        [
            ctx[].memory_get_address(a_dev),
            ctx[].memory_get_address(b_dev),
            ctx[].memory_get_address(c_dev),
        ],
        grid_y=ceildiv(M, tile_size),
        block_y=tile_size,
    )
    queue[].copy_from_device(c_host.unsafe_bitcast[UInt8](), c_dev.view())
    queue[].synchronize()
    var wrong = 0
    var unexplained = 0
    var corner = 0
    # Where the explained elements are: in which work-groups (row group,
    # column group), and in which of their columns, so the race's pattern is
    # measured in every run rather than assumed.
    comptime row_groups = ceildiv(M, tile_size)
    comptime col_groups = ceildiv(N, tile_size)
    var last_hit = alloc[Int32](
        {length = row_groups * col_groups}
    ).unsafe_leak()
    var first_hit = alloc[Int32](
        {length = row_groups * col_groups}
    ).unsafe_leak()
    for g in range(row_groups * col_groups):
        last_hit[unsafe_offset=g] = 0
        first_hit[unsafe_offset=g] = 0
    var last_col = 0
    var first_col = 0
    var first_min = Float32(K)
    for i in range(M):
        for j in range(N):
            var v = c_host[unsafe_offset=i * N + j]
            if v == Float32(K):
                continue
            wrong += 1
            var in_corner = i >= M - 10 and j >= N - 10
            if in_corner:
                corner += 1
            if race_free or in_corner or not explained(i, j, v):
                if unexplained < 10:
                    print("  ", name, "C[", i, ",", j, "] =", v)
                unexplained += 1
            else:
                var g = (i // tile_size) * col_groups + j // tile_size
                if j % tile_size == tile_size - 1:
                    last_col += 1
                    last_hit[unsafe_offset=g] = 1
                else:
                    first_col += 1
                    first_min = min(first_min, v)
                    first_hit[unsafe_offset=g] = 1
    print(name, "wrong:", wrong, "unexplained:", unexplained, "corner:", corner)
    if not race_free:
        # Groups with a short last column (of those that have a last column:
        # all but the right edge's), and the column groups they're in.
        var last_groups = 0
        var first_groups = 0
        var cols = String()
        for cg in range(col_groups):
            var hits = 0
            for rg in range(row_groups):
                hits += Int(last_hit[unsafe_offset=rg * col_groups + cg])
                first_groups += Int(
                    first_hit[unsafe_offset=rg * col_groups + cg]
                )
            last_groups += hits
            if hits > 0:
                cols += " " + String(cg)
        print(
            name,
            "last column short:",
            last_col,
            "in",
            last_groups,
            "of",
            row_groups * (col_groups - 1),
            "groups, column groups" + cols,
        )
        print(
            name,
            "first column short:",
            first_col,
            "in",
            first_groups,
            "of",
            row_groups,
            "right-edge groups, lowest",
            first_min,
        )
    last_hit.unsafe_free()
    first_hit.unsafe_free()
    a_host.unsafe_free()
    b_host.unsafe_free()
    c_host.unsafe_free()
    ctx[].unload_function(f)
    ctx[].free_sync(a_dev^)
    ctx[].free_sync(b_dev^)
    ctx[].free_sync(c_dev^)


def main() raises:
    run[True]("race_free")
    run[False]("max")
