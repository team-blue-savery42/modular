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
"""`BlockReducer.generic` combines every thread's state, in every round.

A two-word monoid (a sum and a count) goes through `generic`'s warp-shuffle
path, which stores one state per warp to shared memory and lets the first
warp combine them. The block sizes cover one warp per lane or fewer (256 at
16 lanes), more warps than lanes in whole groups (1,024: 64 warps of 16) and
in a partial last group, where some lanes have no warp to read (320: 20
warps, 576: 36 warps of 16). Each block calls `generic` `ROUNDS` times in a
loop on different values, as the block tier does when it grid-strides
several rows per block, so a warp that starts the next round early and
overwrites shared memory a lagging warp still reads shows up as a wrong
result. Every thread's result is checked on the CPU; the values are small
integers in float32, so the sums are exact.
"""

from algorithm.gpu.rowwise import BlockReducer
from algorithm.reduce_op import ReduceOp
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal

comptime BLOCKS = 4
comptime ROUNDS = 64


struct SumCount[W: Int](ReduceOp):
    """A sum and the number of values summed: 2 words at width 1."""

    comptime Single = SumCount[1]
    comptime width = Self.W

    var sum: SIMD[DType.float32, Self.W]
    var count: SIMD[DType.float32, Self.W]

    def __init__(out self):
        self.sum = 0
        self.count = 0

    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        self.sum += Self.pad[Self.W, DType.float32, 0](val)
        self.count += Self.pad[Self.W, DType.float32, 0](
            SIMD[DType.float32, w](1)
        )

    def join(mut self, other: Self):
        self.sum += other.sum
        self.count += other.count

    def __getitem__(self, j: Int) -> Self.Single:
        var r = Self.Single()
        r.sum[0] = self.sum[j]
        r.count[0] = self.count[j]
        return r

    def __setitem__(mut self, j: Int, s: Self.Single):
        self.sum[j] = s.sum[0]
        self.count[j] = s.count[0]


@inline(.always)
def value(blk: Int, rnd: Int, tid: Int) -> Int:
    return (tid * 37 + rnd * 11 + blk * 5) % 101 + (tid // 16) % 7


def kernel[
    block_size: Int
](dst: MutPointer[Float32, MutAnyOrigin], rounds_dev: Int32):
    var rounds = Int(rounds_dev)
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    for rnd in range(rounds):
        var state = SumCount[1]()
        state.sum = Float32(value(blk, rnd, tid))
        state.count = 1
        BlockReducer[block_size]().generic(state)
        var i = ((blk * rounds + rnd) * block_size + tid) * 2
        dst[unsafe_offset=i] = state.sum[0]
        dst[unsafe_offset=i + 1] = state.count[0]


def _check[block_size: Int]() raises:
    comptime n = BLOCKS * ROUNDS * block_size * 2
    with DeviceContext() as ctx:
        var out_h = ctx.enqueue_create_host_buffer[DType.float32](n)
        var out_d = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.enqueue_function[kernel[block_size]](
            out_d.unsafe_ptr(),
            Int32(ROUNDS),
            grid_dim=BLOCKS,
            block_dim=block_size,
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        var wrong = 0
        var first = String()
        for blk in range(BLOCKS):
            for rnd in range(ROUNDS):
                var want = 0
                for tid in range(block_size):
                    want += value(blk, rnd, tid)
                for tid in range(block_size):
                    var i = ((blk * ROUNDS + rnd) * block_size + tid) * 2
                    var s = out_h[i]
                    var c = out_h[i + 1]
                    if s != Float32(want) or c != Float32(block_size):
                        if wrong == 0:
                            first = String(
                                "block ",
                                blk,
                                " round ",
                                rnd,
                                " thread ",
                                tid,
                                ": sum ",
                                s,
                                " want ",
                                want,
                                ", count ",
                                c,
                                " want ",
                                block_size,
                            )
                        wrong += 1
        assert_equal(
            wrong, 0, msg=String("block_size ", block_size, ", first ", first)
        )
        _ = out_d^


def test_generic_256() raises:
    _check[256]()


def test_generic_320() raises:
    _check[320]()


def test_generic_576() raises:
    _check[576]()


def test_generic_1024() raises:
    _check[1024]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
