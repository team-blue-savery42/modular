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

# MAX's test_strided_load on the B70. The MAX test only checks that
# the kernel's LLVM IR names `@llvm.masked.gather`, NVIDIA's and AMD's
# lowering; Apple's GPUs skip it for the same reason as ours (its BUILD file:
# "Metal IR does not contain @llvm.masked.gather"). The Intel stdlib plugin
# sets `lowers_masked_gather_scatter = False` (vectors of pointers need
# SPV_INTEL_masked_gather_scatter, which IGC rejects), so `gather()` loads
# each lane through its own pointer and the IR has no such call. This runs
# the MAX test's kernel, unchanged (uint32 through a generic pointer), at
# widths 4 (the MAX test's) and 16 and strides 1, 3 and 0, and checks every
# lane against the CPU.

from std.memory import alloc
from std.sys.intrinsics import strided_load

from hal_gpu import GpuContext, launch, load, open_context

comptime _N = 64


def strided_load_kernel[
    *, dtype: DType = .uint32, width: SIMDLength = 1
](
    output: MutPointer[SIMD[dtype, width], MutAnyOrigin],
    ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin, address_space=.GENERIC],
    stride: Int,
):
    output[] = strided_load[width](ptr, stride)


def check[width: Int](mut ctx: GpuContext, stride: Int) raises -> Int:
    comptime kernel = strided_load_kernel[width=width]
    var queue = ctx.create_queue()
    var function = load[type_of(kernel), kernel](ctx)
    var in_dev = ctx.alloc_sync(UInt64(4 * _N))
    var out_dev = ctx.alloc_sync(UInt64(4 * width))
    var host = alloc[UInt32]({length = _N}).unsafe_leak()
    for i in range(_N):
        host[unsafe_offset=i] = UInt32(1000 + 7 * i)
    queue[].copy_to_device(in_dev.view(), host.unsafe_bitcast[UInt8]())
    queue[].set_memory(out_dev.view(), value=UInt8(0xFF))
    launch(
        queue[],
        function,
        1,
        1,
        [
            ctx.memory_get_address(out_dev),
            ctx.memory_get_address(in_dev),
            UInt64(stride),
        ],
    )
    var out = alloc[UInt32]({length = width}).unsafe_leak()
    queue[].copy_from_device(out.unsafe_bitcast[UInt8](), out_dev.view())
    queue[].synchronize()
    var wrong = 0
    for i in range(width):
        var want = host[unsafe_offset=i * stride]
        if out[unsafe_offset=i] != want:
            print(
                "width",
                width,
                "stride",
                stride,
                "lane",
                i,
                "got",
                out[unsafe_offset=i],
                "expected",
                want,
            )
            wrong += 1
    out.unsafe_free()
    host.unsafe_free()
    ctx.unload_function(function)
    ctx.free_sync(out_dev^)
    ctx.free_sync(in_dev^)
    if wrong == 0:
        print("PASS width", width, "stride", stride, flush=True)
    return wrong


def main() raises:
    var ctx = open_context()
    var wrong = 0
    for stride in [1, 3, 0]:
        wrong += check[4](ctx[], stride)
        wrong += check[16](ctx[], stride)
    if wrong:
        raise Error(String(wrong, " lanes wrong"))
