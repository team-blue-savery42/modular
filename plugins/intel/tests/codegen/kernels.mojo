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

# Codegen corpus for the Intel compiler plugin: kernels for
# the B70 target from MAX's Intel target table, written against the stdlib and
# MAX GPU APIs that the Intel plugins implement.
# Run with utils/check-codegen.

from std.atomic import Atomic
from std.compile import compile_info
from std.ffi import external_call
from std.math import exp, sqrt
from std.memory import AddressSpace, stack_allocation, unsafe_stack_allocation
from std.memory.memory import _free, _malloc
from std.os import abort
from std.sys.info import CompilationTarget
from std.utils import StaticTuple

from max._plugin.intel.targets import ArcProB70, ArcProB70SG32
from max.gpu import (
    WARP_SIZE,
    block_dim,
    block_idx,
    global_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from max.gpu.compute.mma import mma
from max.gpu.primitives import warp
from max.gpu.memory import external_memory
from max.gpu.memory.block_io_intel import (
    block_load,
    block_load_transform,
    block_load_transpose,
    block_prefetch,
    block_store,
)
from max.gpu.sync import barrier, syncwarp

comptime _intel = CompilationTarget[_mlir_value=ArcProB70.mlir_target]()
# Its sub-group size 32 variant.
comptime _intel32 = CompilationTarget[_mlir_value=ArcProB70SG32.mlir_target]()


# A store through a global pointer.
def store_value(p: Pointer[Int32, MutAnyOrigin], v: Int32):
    p[0] = v


# A generic stack buffer.
def load_first(
    dst: Pointer[Int32, MutAnyOrigin], p: Pointer[Int32, MutAnyOrigin]
):
    var tmp = stack_allocation[4, DType.int32]()
    for i in range(4):
        tmp[i] = p[i]
    dst[0] = tmp[0] + tmp[3]


# A stack buffer in the local (private) address space.
def local_stack(p: Pointer[Int32, MutAnyOrigin]):
    var tmp = stack_allocation[
        4, DType.int32, address_space=AddressSpace.LOCAL
    ]()
    for i in range(4):
        tmp[i] = p[i]
    p[0] = tmp[0] + tmp[3]


# One work-item per element.
def vector_add(
    a: Pointer[Float32, MutAnyOrigin],
    b: Pointer[Float32, MutAnyOrigin],
    c: Pointer[Float32, MutAnyOrigin],
    n: Int,
):
    var i = global_idx.x
    if i < n:
        c[i] = a[i] + b[i]


# Every GPU index and size.
def ids(dst: Pointer[Int, MutAnyOrigin]):
    var i = global_idx.x
    dst[i * 5 + 0] = thread_idx.x + thread_idx.y + thread_idx.z
    dst[i * 5 + 1] = block_idx.x + block_idx.y + block_idx.z
    dst[i * 5 + 2] = block_dim.x + block_dim.y + block_dim.z
    dst[i * 5 + 3] = grid_dim.x + grid_dim.y + grid_dim.z
    dst[i * 5 + 4] = Int(lane_id())


# Atomics on global memory.
def atomic_count(
    counter: Pointer[Int32, MutAnyOrigin],
    flags: Pointer[Int32, MutAnyOrigin],
):
    if flags[global_idx.x] != 0:
        _ = Atomic.fetch_add(counter, 1)


# Math that lowers to OpenCL extended instructions.
def math_kernel(x: Pointer[Float32, MutAnyOrigin], n: Int):
    var i = global_idx.x
    if i < n:
        x[i] = exp(x[i]) + sqrt(x[i])


# Static shared memory and barriers: a tree sum per 64-wide block, written to
# p[block_idx.x].
def shared_reduce(p: Pointer[Int32, MutAnyOrigin]):
    var tile = stack_allocation[
        64, DType.int32, address_space=AddressSpace.SHARED
    ]()
    var t = thread_idx.x
    tile[t] = p[global_idx.x]
    barrier()
    var stride = 32
    while stride > 0:
        if t < stride:
            tile[t] += tile[t + stride]
        barrier()
        stride //= 2
    if t == 0:
        p[block_idx.x] = tile[0]


# Dynamic shared memory, sized at launch: reverses each block in place.
def reverse_shared(p: Pointer[Int32, MutAnyOrigin]):
    var tile = external_memory[
        Int32, address_space=AddressSpace.SHARED, alignment=16
    ]()
    var t = thread_idx.x
    tile[t] = p[global_idx.x]
    barrier()
    p[global_idx.x] = tile[block_dim.x - 1 - t]


# A sub-group barrier.
def warp_sync(p: Pointer[Int32, MutAnyOrigin]):
    p[global_idx.x] += 1
    syncwarp()


# A warp reduction, compiled for both sub-group sizes: its butterfly has
# log2(WARP_SIZE) shuffles, and it stores WARP_SIZE.
def warp_sum(p: Pointer[Float32, MutAnyOrigin]):
    p[global_idx.x] = warp.sum(p[global_idx.x]) + Float32(WARP_SIZE)


# DPAS through mma(): three signatures in one module (float32 and
# float16 accumulators, M = 8 and 1; a bfloat16 accumulator, M = 4), each its
# own declaration of the built-in, which the default extensions allow.
def dpas(p: Pointer[Float32, MutAnyOrigin]):
    var n = Int(lane_id())
    var b = SIMD[DType.float16, 16](Float16(Float32(n)))
    var d = SIMD[DType.float32, 8]()
    mma(d, SIMD[DType.float16, 8](1), b, SIMD[DType.float32, 8](p[n]))
    var h = SIMD[DType.float16, 1]()
    mma(h, SIMD[DType.float16, 1](2), b, SIMD[DType.float16, 1](0))
    var bb = SIMD[DType.bfloat16, 16](BFloat16(Float32(n)))
    var e = SIMD[DType.bfloat16, 4]()
    mma(e, SIMD[DType.bfloat16, 4](1), bb, SIMD[DType.bfloat16, 4](0))
    p[n] = d.reduce_add() + Float32(h[0]) + Float32(e.reduce_add())


# 2D block I/O: each operation is one instruction from the extension
# the spirv64 defaults enable, reading a global (CrossWorkgroup) matrix into
# a private (Function) array or storing from one.
def block_io(p: Pointer[UInt16, MutAnyOrigin]):
    block_prefetch[16, 8](p, 64, 64, 64, 0, 0)
    # The widths are block_elements[...]: 8, 16 and 8 per work-item.
    var a = rebind[SIMD[DType.uint16, 8]](
        block_load[16, 8](p, 64, 64, 64, 16, 8)
    )
    var b = rebind[SIMD[DType.uint16, 16]](
        block_load_transform[16, 16](p, 64, 64, 64, 0, 16)
    )
    var t = rebind[SIMD[DType.uint32, 8]](
        block_load_transpose[8, 16](
            p.unsafe_bitcast[UInt32](), 32, 64, 32, 0, 32
        )
    )
    var s = a + b.slice[8]() + t.cast[DType.uint16]()
    block_store[16, 8](p, 64, 64, 64, 16, 8, s)


# Split work-group barriers (SPV_INTEL_split_barrier, which the
# spirv64 defaults enable): the built-ins the Intel GEMM calls, work-group
# scope (2), no memory semantics (0).
def split_barrier(p: Pointer[Int32, MutAnyOrigin]):
    external_call["_Z33__spirv_ControlBarrierArriveINTELiii", NoneType](
        Int32(2), Int32(2), Int32(0)
    )
    p[thread_idx.x] = 1
    external_call["_Z31__spirv_ControlBarrierWaitINTELiii", NoneType](
        Int32(2), Int32(2), Int32(0)
    )


# A struct passed by value that holds a zero-length array, as an empty index
# list does: { i32 @0, [0 x i64] @8, i64 @8 }, 16 bytes.
@fieldwise_init
struct WithEmpty(TrivialRegisterPassable):
    var a: Int32
    var empty: StaticTuple[Int64, 0]
    var b: Int64


def zero_tuple(p: Pointer[Int32, MutAnyOrigin], s: WithEmpty):
    p[0] = s.a
    p[1] = Int32(s.b)


# A plain struct passed by value: { i32 @0, i64 @8 }, 16 bytes.
@fieldwise_init
struct Plain(TrivialRegisterPassable):
    var a: Int32
    var b: Int64


def plain_struct(p: Pointer[Int32, MutAnyOrigin], s: Plain):
    p[0] = s.a
    p[1] = Int32(s.b)


# A struct holding an address (a closure's capture of a buffer looks the
# same): the kernel writes through it, so the allocation is reached
# indirectly, not through a pointer argument the driver can see.
@fieldwise_init
struct Boxed(TrivialRegisterPassable):
    var address: Int
    var v: Int32


def boxed_pointer(p: Pointer[Int32, MutAnyOrigin], s: Boxed):
    var q = Pointer[Int32, MutAnyOrigin](unsafe_from_address=s.address)
    q[0] = s.v
    p[0] = s.v + 1


# p[i] = p[4 + i] / p[8 + i]: float32 division, which LLVM defines as
# correctly rounded (OpenCL allows 2.5 ulp unless told otherwise).
def divide(p: Pointer[Float32, MutAnyOrigin]):
    for i in range(4):
        p[i] = p[4 + i] / p[8 + i]


# A List on the device calls malloc and free: the compiler plugin defines
# them over the driver plugin's heap. Each work-item's list grows
# to (i % 7) + 1 values, i * 8 + k, and p[i] gets their sum.
def uses_heap(p: Pointer[Int32, MutAnyOrigin]):
    var i = global_idx.x
    var values = List[Int32]()
    for k in range(i % 7 + 1):
        values.append(Int32(i * 8 + k))
    var total: Int32 = 0
    for k in range(len(values)):
        total += values[k]
    p[i] = total


# malloc of p[i] bytes: p[i] becomes 1 if it returned memory, 0 if null (the
# heap is used up, as CUDA's malloc returns null).
def heap_exhaust(p: Pointer[Int32, MutAnyOrigin]):
    var i = global_idx.x
    var block = _malloc[Int32](Int(p[i]))
    p[i] = 1 if block else 0


# Freed blocks are reused: 256 rounds of malloc(p[i]) bytes,
# a value written to the block's first and last words and read back, then
# free; p[i] becomes the rounds whose block came back and held its values.
def heap_reuse(p: Pointer[Int32, MutAnyOrigin]):
    var i = global_idx.x
    var bytes = Int(p[i])
    var count: Int32 = 0
    for round in range(256):
        var block = _malloc[Int32](bytes)
        if not block:
            continue
        var q = block.unsafe_value()
        var last = bytes // 4 - 1
        q[0] = Int32(i * 1000 + round)
        q[last] = Int32(i * 1000 + round + 1)
        if q[0] == Int32(i * 1000 + round) and q[last] == Int32(
            i * 1000 + round + 1
        ):
            count += 1
        _free(block)
    p[i] = count


# abort(): reported to the host through the driver plugin's flag.
def aborts(p: Pointer[Int32, MutAnyOrigin]):
    if p[0] == 7:
        abort("corpus abort")
    p[0] = 1


# Constant memory the host fills, named as MAX's ConstantMemoryMapping names
# it: the plugin defines and exports the table the kernel imports.
def constant_import(p: Pointer[Int32, MutAnyOrigin]):
    var table = unsafe_stack_allocation[
        16,
        Int32,
        name=StaticString("constant_table"),
        address_space=AddressSpace.CONSTANT,
        alignment=8,
    ]()
    p[thread_idx.x] = table[thread_idx.x]


def emit[
    func_type: TrivialRegisterPassable, //, func: func_type
](name: StaticString,) raises:
    with open(String("out/", name, ".spvasm"), "w") as f:
        f.write(compile_info[func, target=_intel]().asm)
    with open(String("out/", name, ".spv"), "w") as f:
        f.write(compile_info[func, target=_intel, emission_kind="object"]().asm)


def main() raises:
    emit[store_value]("store_value")
    emit[load_first]("load_first")
    emit[local_stack]("local_stack")
    emit[vector_add]("vector_add")
    emit[ids]("ids")
    emit[atomic_count]("atomic_count")
    emit[math_kernel]("math_kernel")
    emit[shared_reduce]("shared_reduce")
    emit[reverse_shared]("reverse_shared")
    emit[warp_sync]("warp_sync")
    emit[aborts]("aborts")
    emit[zero_tuple]("zero_tuple")
    emit[plain_struct]("plain_struct")
    emit[boxed_pointer]("boxed_pointer")
    emit[divide]("divide")
    emit[uses_heap]("uses_heap")
    emit[heap_exhaust]("heap_exhaust")
    emit[heap_reuse]("heap_reuse")
    emit[constant_import]("constant_import")
    emit[warp_sum]("warp_sum")
    emit[dpas]("dpas")
    emit[block_io]("block_io")
    emit[split_barrier]("split_barrier")
    with open("out/warp_sum_sg32.spv", "w") as f:
        f.write(
            compile_info[
                warp_sum, target=_intel32, emission_kind="object"
            ]().asm
        )
    # SPIR-V extensions are enabled per compile through an emission option.
    with open("out/store_value_ext.spvasm", "w") as f:
        f.write(
            compile_info[
                store_value,
                target=_intel,
                compile_options="spirv-ext=+SPV_INTEL_subgroups",
            ]().asm
        )
