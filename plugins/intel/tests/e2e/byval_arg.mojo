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

# A kernel argument passed in memory. A struct that isn't
# register-passable is a borrowed in-memory argument of the kernel function;
# Mojo's lowering gives exported kernels' such arguments the target's
# by-value attribute (`addByValAttribute`, LowerKGENToLLVM.cpp), which the
# Intel lowering names `llvm.byval`, as clang passes an OpenCL
# kernel's struct argument for SPIR. MAX's graph kernels take such
# arguments. The host sets the struct's bytes as the argument; the kernel
# reads every field (40 bytes, an odd mix of widths) and writes
# i * scale + offset for i < count, and the bytes of `tag` and `flag` into
# the first two words past count. Checked against the CPU.

from max.gpu import global_idx
from std.memory import alloc, OpaquePointer, Pointer
from std.sys import size_of

from hal_gpu import GpuContext, load, open_context


struct Params(Copyable, Movable):
    """Copyable but not register-passable: passed to a kernel in memory."""

    var scale: Int64
    var offset: Int64
    var count: Int32
    var tag: UInt16
    var flag: UInt8
    var values: SIMD[DType.int32, 4]

    def __init__(out self, scale: Int64, offset: Int64, count: Int32):
        self.scale = scale
        self.offset = offset
        self.count = count
        self.tag = 0xBEEF
        self.flag = 7
        self.values = SIMD[DType.int32, 4](100, 101, 102, 103)


def byval_kernel(params: Params, dst: MutPointer[Int64, MutAnyOrigin]):
    var i = Int(global_idx.x)
    if i < Int(params.count):
        dst[unsafe_offset=i] = (
            Int64(i) * params.scale
            + params.offset
            + Int64(params.values[i % 4])
        )
    elif i == Int(params.count):
        dst[unsafe_offset=i] = Int64(params.tag)
    elif i == Int(params.count) + 1:
        dst[unsafe_offset=i] = Int64(params.flag)


def main() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var function = load[type_of(byval_kernel), byval_kernel](ctx[])
    comptime N = 1000
    comptime TOTAL = 1024
    var dst = ctx[].alloc_sync(UInt64(8 * TOTAL))
    queue[].set_memory(dst.view(), value=UInt8(0xFF))
    var params = Params(scale=3, offset=-5, count=N)
    var address = ctx[].memory_get_address(dst)
    var args = alloc[OpaquePointer[ImmUntrackedOrigin]](
        {length = 2}
    ).unsafe_leak()
    var sizes = alloc[UInt64]({length = 2}).unsafe_leak()
    args[unsafe_offset=0] = rebind[OpaquePointer[ImmUntrackedOrigin]](
        Pointer(to=params)
    )
    sizes[unsafe_offset=0] = UInt64(size_of[Params]())
    args[unsafe_offset=1] = rebind[OpaquePointer[ImmUntrackedOrigin]](
        Pointer(to=address)
    )
    sizes[unsafe_offset=1] = UInt64(8)
    queue[].execute(
        function,
        grid=(UInt32(TOTAL // 256), UInt32(1), UInt32(1)),
        block=(UInt32(256), UInt32(1), UInt32(1)),
        args=args,
        arg_sizes=rebind[Pointer[UInt64, MutAnyOrigin]](sizes),
        num_args=UInt32(2),
        shared_mem_bytes=UInt32(0),
    )
    var host = alloc[Int64]({length = TOTAL}).unsafe_leak()
    queue[].copy_from_device(host.unsafe_bitcast[UInt8](), dst.view())
    queue[].synchronize()
    var wrong = 0
    for i in range(TOTAL):
        var want: Int64 = -1
        if i < N:
            want = Int64(i) * 3 - 5 + Int64(100 + i % 4)
        elif i == N:
            want = 0xBEEF
        elif i == N + 1:
            want = 7
        if host[unsafe_offset=i] != want:
            if wrong < 3:
                print("  i", i, "device", host[unsafe_offset=i], "host", want)
            wrong += 1
    print("byval: struct of", size_of[Params](), "bytes, wrong:", wrong)
    print("PASS byval" if wrong == 0 else "FAIL byval")
    args.unsafe_free()
    sizes.unsafe_free()
    host.unsafe_free()
    ctx[].unload_function(function)
    ctx[].free_sync(dst^)
