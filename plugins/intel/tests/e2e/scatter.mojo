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

# Port of the stdlib's Mojo/stdlib/test/gpu/memory/test_scatter.mojo to the
# HAL client: Pointer.unsafe_scatter in global and shared memory,
# masked and not. The kernels are the stdlib test's.

from std.memory import alloc, unsafe_stack_allocation
from std.testing import assert_equal

from hal_gpu import GpuContext, launch, load, open_context

comptime _OFFSETS = SIMD[.int32, 4](3, 2, 1, 0)
comptime _VALUES = SIMD[.float32, 4](1, 2, 3, 4)
comptime _EVEN_LANES = SIMD[.bool, 4](True, False, True, False)


def _scatter_global_kernel(output: MutPointer[Float32, MutAnyOrigin]):
    output.unsafe_scatter(_OFFSETS, _VALUES)


def _scatter_shared_kernel[
    masked: Bool
](output: MutPointer[Float32, MutAnyOrigin]):
    var shared = unsafe_stack_allocation[
        4, DType.float32, address_space=.SHARED
    ]()
    for i in range(4):
        shared.unsafe_offset(i).unsafe_store(Float32(-1))

    comptime if masked:
        shared.unsafe_scatter(_OFFSETS, _VALUES, _EVEN_LANES)
    else:
        shared.unsafe_scatter(_OFFSETS, _VALUES)

    for i in range(4):
        output.unsafe_offset(i).unsafe_store(
            shared.unsafe_offset(i).unsafe_load()
        )


def check[
    fn_type: TrivialRegisterPassable, func: fn_type
](mut ctx: GpuContext, name: String, expected: SIMD[DType.float32, 4]) raises:
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
    print("PASS", name)


def main() raises:
    var ctx = open_context()
    check[type_of(_scatter_global_kernel), _scatter_global_kernel](
        ctx[], "test_scatter_global", SIMD[DType.float32, 4](4, 3, 2, 1)
    )
    check[
        type_of(_scatter_shared_kernel[masked=False]),
        _scatter_shared_kernel[masked=False],
    ](ctx[], "test_scatter_shared", SIMD[DType.float32, 4](4, 3, 2, 1))
    # Lanes 0 and 2 write values 1 and 3 to offsets 3 and 1.
    check[
        type_of(_scatter_shared_kernel[masked=True]),
        _scatter_shared_kernel[masked=True],
    ](ctx[], "test_scatter_shared_masked", SIMD[DType.float32, 4](-1, 3, -1, 1))
