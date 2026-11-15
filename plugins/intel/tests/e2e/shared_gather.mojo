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

# Pointer.unsafe_gather, strided_load and strided_store on shared memory.
# With a generic address space the stdlib built each lane's address
# from `Int(pointer)`, which for a shared pointer is an offset into the
# work-group's window, not a generic address: the kernel read and wrote near
# address 0 and page-faulted. The offsets depend on `thread_idx.x` (0 here) so
# that LLVM can't fold the address back into an address-space cast.
# `-D KERNEL=<name>` runs one check alone (default: all), so that each can be
# shown to fail on its own: a fault ends the process. A name that matches no
# check is an error; the names live only in the check calls in main.

from std.memory import alloc, unsafe_stack_allocation
from std.sys.defines import get_defined_string
from std.sys.intrinsics import strided_load, strided_store
from std.testing import assert_equal

from hal_gpu import GpuContext, launch, load, open_context
from max.gpu import thread_idx

comptime _N = 8
comptime _KERNEL = get_defined_string["KERNEL", "all"]()


def _fill(shared: MutPointer[Float32, _, address_space=.SHARED]):
    for i in range(_N):
        shared.unsafe_offset(i).unsafe_store(Float32(i))


def _gather_kernel[masked: Bool](output: MutPointer[Float32, MutAnyOrigin]):
    var shared = unsafe_stack_allocation[
        _N, DType.float32, address_space=.SHARED
    ]()
    _fill(shared)
    var offsets = SIMD[.int32, 4](6, 4, 2, 0) + Int32(thread_idx.x)
    var got: SIMD[.float32, 4]
    comptime if masked:
        got = shared.unsafe_gather(
            offsets,
            SIMD[.bool, 4](True, False, True, False),
            SIMD[.float32, 4](-1),
        )
    else:
        got = shared.unsafe_gather(offsets)
    for i in range(4):
        output.unsafe_offset(i).unsafe_store(got[i])


def _strided_load_kernel(output: MutPointer[Float32, MutAnyOrigin]):
    var shared = unsafe_stack_allocation[
        _N, DType.float32, address_space=.SHARED
    ]()
    _fill(shared)
    var got = strided_load[4](shared.unsafe_offset(Int(thread_idx.x) + 1), 2)
    for i in range(4):
        output.unsafe_offset(i).unsafe_store(got[i])


def _strided_store_kernel(output: MutPointer[Float32, MutAnyOrigin]):
    var shared = unsafe_stack_allocation[
        _N, DType.float32, address_space=.SHARED
    ]()
    _fill(shared)
    strided_store(
        SIMD[.float32, 4](10, 20, 30, 40),
        shared.unsafe_offset(Int(thread_idx.x)),
        2,
    )
    for i in range(4):
        output.unsafe_offset(i).unsafe_store(
            shared.unsafe_offset(2 * i).unsafe_load()
            + shared.unsafe_offset(2 * i + 1).unsafe_load()
        )


def check[
    fn_type: TrivialRegisterPassable, func: fn_type
](
    mut ctx: GpuContext, name: String, expected: SIMD[DType.float32, 4]
) raises -> Int:
    """Runs the check unless `-D KERNEL` names another; returns how many ran."""
    if _KERNEL != "all" and _KERNEL != name:
        return 0
    var queue = ctx.create_queue()
    var function = load[fn_type, func](ctx)
    var out_dev = ctx.alloc_sync(UInt64(16))
    queue[].set_memory(out_dev.view(), value=UInt8(0))
    launch(queue[], function, 1, 1, [ctx.memory_get_address(out_dev)])
    var host = alloc[Float32]({length = 4}).unsafe_leak()
    queue[].copy_from_device(host.unsafe_bitcast[UInt8](), out_dev.view())
    queue[].synchronize()
    for i in range(4):
        assert_equal(
            host[unsafe_offset=i], expected[i], msg=name + " lane " + String(i)
        )
    host.unsafe_free()
    ctx.unload_function(function)
    ctx.free_sync(out_dev^)
    print("PASS", name, flush=True)
    return 1


def main() raises:
    var ctx = open_context()
    var ran = 0
    ran += check[type_of(_gather_kernel[False]), _gather_kernel[False]](
        ctx[], "gather_shared", SIMD[DType.float32, 4](6, 4, 2, 0)
    )
    ran += check[type_of(_gather_kernel[True]), _gather_kernel[True]](
        ctx[], "gather_shared_masked", SIMD[DType.float32, 4](6, -1, 2, -1)
    )
    # Elements 1, 3, 5, 7.
    ran += check[type_of(_strided_load_kernel), _strided_load_kernel](
        ctx[], "strided_load_shared", SIMD[DType.float32, 4](1, 3, 5, 7)
    )
    # Even elements become 10..40; each output adds the odd neighbour.
    ran += check[type_of(_strided_store_kernel), _strided_store_kernel](
        ctx[], "strided_store_shared", SIMD[DType.float32, 4](11, 23, 35, 47)
    )
    # A misspelt name would otherwise run nothing and exit 0.
    if ran == 0:
        raise Error("unknown -D KERNEL=" + _KERNEL)
