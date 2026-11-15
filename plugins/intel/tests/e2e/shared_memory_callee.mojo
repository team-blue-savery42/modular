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

# Dynamic shared memory used outside the kernel: MAX's
# test_external_memory_callee_patterns without the DeviceContext, plus the
# call graphs the compiler plugin must thread the memory through. Each kernel
# runs NUM_BLOCKS work-groups of BLOCK_SIZE; work-item i writes i (plus a
# per-kernel offset) into the dynamic shared memory, and work-item 0 sums it,
# so each block's result is checked against the CPU.
#   callee:   the kernel doesn't name the memory; a callee does (one level).
#   deep:     kernel -> pass-through -> the user (two levels).
#   direct:   the kernel names it AND calls the same callee as `callee`, which
#             two kernels now share, and a callee that doesn't use it.
# Before the fix the plugin refused the module: "dynamic shared memory @... is
# used in ..., which is not a kernel".

from std.memory import alloc
from max.gpu import block_idx, thread_idx
from max.gpu.memory import external_memory
from max.gpu.sync import barrier

from hal_gpu import launch, load, open_context

comptime BLOCK_SIZE = 64
comptime NUM_BLOCKS = 4


@inline(.never)
def _fill_and_reduce(
    data: MutPointer[Float32, MutAnyOrigin],
    local_idx: Int,
    blk_idx: Int,
    add: Float32,
):
    var smem = external_memory[
        Float32, address_space=.SHARED, alignment=4, name="callee_ext"
    ]()
    smem[unsafe_offset=local_idx] = Float32(local_idx) + add
    barrier()
    if local_idx == 0:
        var s = Float32(0)
        for i in range(BLOCK_SIZE):
            s += smem[unsafe_offset=i]
        data[unsafe_offset=blk_idx] = s


@inline(.never)
def _pass_through(
    data: MutPointer[Float32, MutAnyOrigin], local_idx: Int, blk_idx: Int
):
    _fill_and_reduce(data, local_idx, blk_idx, 1)


@inline(.never)
def _no_shared(x: Int) -> Float32:
    return Float32(x) * 2


def callee_kernel(data: MutPointer[Float32, MutAnyOrigin]):
    _fill_and_reduce(data, thread_idx.x, block_idx.x, 0)


def deep_kernel(data: MutPointer[Float32, MutAnyOrigin]):
    _pass_through(data, thread_idx.x, block_idx.x)


def direct_kernel(data: MutPointer[Float32, MutAnyOrigin]):
    # Names the memory itself (a different symbol: every one aliases the
    # same memory, as in CUDA) and calls both callees.
    var smem = external_memory[
        Float32, address_space=.SHARED, alignment=4, name="kernel_ext"
    ]()
    _fill_and_reduce(data, thread_idx.x, block_idx.x, 2)
    barrier()
    smem[unsafe_offset=thread_idx.x] = _no_shared(thread_idx.x)
    barrier()
    if thread_idx.x == 0:
        var s = Float32(0)
        for i in range(BLOCK_SIZE):
            s += smem[unsafe_offset=i]
        data[unsafe_offset=NUM_BLOCKS + block_idx.x] = s


def check(name: String, res: Pointer[Float32, _], n: Int, add: Float32) -> Int:
    var bad = 0
    var want = Float32(BLOCK_SIZE * (BLOCK_SIZE - 1) // 2) + add * BLOCK_SIZE
    for i in range(n):
        if res[unsafe_offset=i] != want:
            bad += 1
    print(name, "wrong:", bad, "first:", res[unsafe_offset=0], "want:", want)
    return bad


def main() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var dout = ctx[].alloc_sync(UInt64(4 * 2 * NUM_BLOCKS))
    var res = alloc[Float32]({length = 2 * NUM_BLOCKS}).unsafe_leak()
    var bad = 0

    var f1 = load[type_of(callee_kernel), callee_kernel](ctx[])
    launch(
        queue[],
        f1,
        NUM_BLOCKS,
        BLOCK_SIZE,
        [ctx[].memory_get_address(dout)],
        shared_mem_bytes=BLOCK_SIZE * 4,
    )
    queue[].copy_from_device(res.unsafe_bitcast[UInt8](), dout.view())
    queue[].synchronize()
    bad += check("callee", res, NUM_BLOCKS, 0)

    var f2 = load[type_of(deep_kernel), deep_kernel](ctx[])
    launch(
        queue[],
        f2,
        NUM_BLOCKS,
        BLOCK_SIZE,
        [ctx[].memory_get_address(dout)],
        shared_mem_bytes=BLOCK_SIZE * 4,
    )
    queue[].copy_from_device(res.unsafe_bitcast[UInt8](), dout.view())
    queue[].synchronize()
    bad += check("deep", res, NUM_BLOCKS, 1)

    var f3 = load[type_of(direct_kernel), direct_kernel](ctx[])
    launch(
        queue[],
        f3,
        NUM_BLOCKS,
        BLOCK_SIZE,
        [ctx[].memory_get_address(dout)],
        shared_mem_bytes=BLOCK_SIZE * 4,
    )
    queue[].copy_from_device(res.unsafe_bitcast[UInt8](), dout.view())
    queue[].synchronize()
    bad += check("direct callee", res, NUM_BLOCKS, 2)
    # Sum of 2 * i over the block: twice the plain sum, so add = sum / 64.
    bad += check(
        "direct kernel",
        res.unsafe_offset(NUM_BLOCKS),
        NUM_BLOCKS,
        Float32(BLOCK_SIZE - 1) / 2,
    )
    res.unsafe_free()
    ctx[].free_sync(dout^)
    if bad:
        raise Error("dynamic shared memory in callees gave wrong results")
