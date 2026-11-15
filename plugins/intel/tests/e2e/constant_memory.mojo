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

# Constant memory: MAX's test_constant_memory without the
# DeviceContext. A constant-space array filled at compile time (directly and
# through a function parameter), and a named constant-space array that the
# host fills, as MAX's ConstantMemoryMapping names one. Mojo emits the named
# one as an internal global with an undef initializer, which IGC refused
# ("undefined reference to `static_constant'"); the compiler plugin now
# defines it, zero-filled and exported. This checks it loads and reads zeros,
# then fills it twice as the compat layer's `enqueue_function` fills a
# mapping: `Context.get_optional_function_property` reads
# `global_address:`/`global_size:` (uint64), `wrap_memory` and a queue copy
# fill it.

from std.memory import alloc, unsafe_stack_allocation
from max.gpu import thread_idx

from hal_gpu import launch, load, open_context

comptime N = 16


def _fill_impl[
    n: Int
]() -> MutPointer[Float32, MutUntrackedOrigin, address_space=.CONSTANT]:
    var ptr = unsafe_stack_allocation[n, Float32, address_space=.CONSTANT]()

    comptime for i in range(n):
        ptr[unsafe_offset=i] = Float32(i)
    return ptr


def static_kernel(data: MutPointer[Float32, MutAnyOrigin]):
    comptime val = _fill_impl[N]()
    data[unsafe_offset=thread_idx.x] = val[unsafe_offset=thread_idx.x]


def via_func_kernel[
    get: def() thin -> MutPointer[
        Float32, MutUntrackedOrigin, address_space=.CONSTANT
    ]
](data: MutPointer[Float32, MutAnyOrigin]):
    comptime val = get()
    data[unsafe_offset=thread_idx.x] = val[unsafe_offset=thread_idx.x]


def external_kernel(data: MutPointer[Float32, MutAnyOrigin]):
    var static_constant = unsafe_stack_allocation[
        N,
        Float32,
        name=StaticString("static_constant"),
        address_space=.CONSTANT,
        alignment=8,
    ]()
    data[unsafe_offset=thread_idx.x] = static_constant[
        unsafe_offset=thread_idx.x
    ]


def check(name: String, res: Pointer[Float32, _], step: Float32 = 1) -> Int:
    var bad = 0
    for i in range(N):
        if res[unsafe_offset=i] != Float32(i) * step:
            bad += 1
    print(
        name,
        "wrong:",
        bad,
        "first:",
        res[unsafe_offset=0],
        res[unsafe_offset=1],
        res[unsafe_offset=N - 1],
    )
    return bad


def main() raises:
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var dout = ctx[].alloc_sync(UInt64(4 * N))
    var res = alloc[Float32]({length = N}).unsafe_leak()
    var bad = 0

    var f1 = load[type_of(static_kernel), static_kernel](ctx[])
    launch(queue[], f1, 1, N, [ctx[].memory_get_address(dout)])
    queue[].copy_from_device(res.unsafe_bitcast[UInt8](), dout.view())
    queue[].synchronize()
    bad += check("static", res)

    comptime k2 = via_func_kernel[_fill_impl[20]]
    var f2 = load[type_of(k2), k2](ctx[])
    launch(queue[], f2, 1, N, [ctx[].memory_get_address(dout)])
    queue[].copy_from_device(res.unsafe_bitcast[UInt8](), dout.view())
    queue[].synchronize()
    bad += check("via_func", res)

    var f3 = load[type_of(external_kernel), external_kernel](ctx[])
    print("external: loaded")
    launch(queue[], f3, 1, N, [ctx[].memory_get_address(dout)])
    queue[].copy_from_device(res.unsafe_bitcast[UInt8](), dout.view())
    queue[].synchronize()
    bad += check("external (unfilled)", res, step=0)

    var missing = ctx[].get_optional_function_property(
        f3, "global_address:no_such_global"
    )
    print("missing global:", "none" if not missing else "found")
    if missing:
        bad += 1
    var address = ctx[].get_optional_function_property(
        f3, "global_address:static_constant"
    )
    var size = ctx[].get_optional_function_property(
        f3, "global_size:static_constant"
    )
    if not address or not size:
        raise Error("static_constant: no global_address/global_size")
    print("global size:", size.value())
    if size.value() != UInt64(4 * N):
        bad += 1
    var table = alloc[Float32]({length = N}).unsafe_leak()
    for step in [Float32(1), Float32(2)]:
        for i in range(N):
            table[unsafe_offset=i] = Float32(i) * step
        var global_buf = ctx[].wrap_memory(address.value(), UInt64(4 * N))
        queue[].copy_to_device(global_buf.view(), table.unsafe_bitcast[UInt8]())
        ctx[].free_sync(global_buf^)
        launch(queue[], f3, 1, N, [ctx[].memory_get_address(dout)])
        queue[].copy_from_device(res.unsafe_bitcast[UInt8](), dout.view())
        queue[].synchronize()
        bad += check(
            String("external (filled x", Int(step), ")"), res, step=step
        )
    table.unsafe_free()
    res.unsafe_free()
    ctx[].free_sync(dout^)
    if bad:
        raise Error("constant memory gave wrong results")
