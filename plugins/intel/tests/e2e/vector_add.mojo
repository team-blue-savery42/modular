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

# A vector add on the B70 through the HAL client API (max._driver), the way
# Modular's MLRT/test/Driver/HAL/test_e2e_driver.mojo drives a GPU: our
# compiler plugin builds the kernel, our Level Zero driver plugin runs it.
# Checked against a CPU reference; prints launch timings. Run with
# utils/run-e2e.

from std.os import getenv
from std.memory import alloc, ArcPointer, OpaquePointer, Pointer
from std.sys import size_of
from std.time import perf_counter_ns

from max._driver import Driver, Context
from max._driver.max_license.hal.device import (
    Device,
    DeviceSpec,
    get_device_spec,
)
from max.gpu import global_idx


def vec_add(
    a: Pointer[Float32, ImmutAnyOrigin],
    b: Pointer[Float32, ImmutAnyOrigin],
    result: Pointer[Float32, MutAnyOrigin],
    n: Int,
):
    var tid = global_idx.x
    if tid >= n:
        return
    result[unsafe_offset=tid] = a[unsafe_offset=tid] + b[unsafe_offset=tid]


def run[device_spec: DeviceSpec](mut ctx: Context[device_spec]) raises:
    var start = perf_counter_ns()
    # Owned (see hal_gpu.mojo's `load`): a `ref` into the call's temporary
    # ends with the statement.
    var compiled_bundle = ctx.compile[type_of(vec_add), vec_add]()
    ref info = compiled_bundle[1]
    var func = ctx.load_function(compiled_bundle[0], info.function_name)
    var compiled = perf_counter_ns()
    print("kernel:", info.function_name)
    print("compile and load ms:", (compiled - start) // 1_000_000)

    comptime length = 1 << 20
    comptime bytes = UInt64(length * size_of[Float32]())
    var a_dev = ctx.alloc_sync(bytes)
    var b_dev = ctx.alloc_sync(bytes)
    var out_dev = ctx.alloc_sync(bytes)
    var queue = ctx.create_queue()

    var a = alloc[Float32]({length = length}).unsafe_leak()
    var b = alloc[Float32]({length = length}).unsafe_leak()
    var out = alloc[Float32]({length = length}).unsafe_leak()
    for i in range(length):
        a[unsafe_offset=i] = Float32(i)
        b[unsafe_offset=i] = Float32((i % 7) * -3)
        out[unsafe_offset=i] = Float32(-1.0)
    queue[].copy_to_device(a_dev.view(), a.unsafe_bitcast[UInt8]())
    queue[].copy_to_device(b_dev.view(), b.unsafe_bitcast[UInt8]())
    queue[].synchronize()

    var a_ptr = ctx.memory_get_address(a_dev)
    var b_ptr = ctx.memory_get_address(b_dev)
    var out_ptr = ctx.memory_get_address(out_dev)
    var n = Int(length)
    var args = alloc[OpaquePointer[ImmUntrackedOrigin]](
        {length = 4}
    ).unsafe_leak()
    var arg_sizes = alloc[UInt64]({length = 4}).unsafe_leak()
    args[unsafe_offset=0] = rebind[OpaquePointer[MutUntrackedOrigin]](
        Pointer(to=a_ptr)
    )
    args[unsafe_offset=1] = rebind[OpaquePointer[MutUntrackedOrigin]](
        Pointer(to=b_ptr)
    )
    args[unsafe_offset=2] = rebind[OpaquePointer[MutUntrackedOrigin]](
        Pointer(to=out_ptr)
    )
    args[unsafe_offset=3] = rebind[OpaquePointer[MutUntrackedOrigin]](
        Pointer(to=n)
    )
    for i in range(4):
        arg_sizes[unsafe_offset=i] = UInt64(8)

    comptime block = UInt32(256)
    comptime grid = UInt32(length) // block

    queue[].execute(
        func,
        grid=(grid, UInt32(1), UInt32(1)),
        block=(block, UInt32(1), UInt32(1)),
        args=args,
        arg_sizes=rebind[Pointer[UInt64, MutAnyOrigin]](arg_sizes),
        num_args=UInt32(4),
    )
    queue[].copy_from_device(out.unsafe_bitcast[UInt8](), out_dev.view())
    queue[].synchronize()

    var wrong = 0
    for i in range(length):
        if out[unsafe_offset=i] != a[unsafe_offset=i] + b[unsafe_offset=i]:
            if wrong < 5:
                print("mismatch at", i, ":", out[unsafe_offset=i])
            wrong += 1
    print("elements:", length, "wrong:", wrong)

    # Launch + synchronize, and back-to-back launches, of the whole vector (12
    # MiB moved) and of one block (launch overhead, as the driver plugin's
    # MojoVectorAddMatchesTheCpu measure it).
    comptime reps = 1000
    var t0 = perf_counter_ns()
    for _ in range(reps):
        queue[].execute(
            func,
            grid=(grid, UInt32(1), UInt32(1)),
            block=(block, UInt32(1), UInt32(1)),
            args=args,
            arg_sizes=rebind[Pointer[UInt64, MutAnyOrigin]](arg_sizes),
            num_args=UInt32(4),
        )
        queue[].synchronize()
    var t1 = perf_counter_ns()
    for _ in range(reps):
        queue[].execute(
            func,
            grid=(grid, UInt32(1), UInt32(1)),
            block=(block, UInt32(1), UInt32(1)),
            args=args,
            arg_sizes=rebind[Pointer[UInt64, MutAnyOrigin]](arg_sizes),
            num_args=UInt32(4),
        )
    queue[].synchronize()
    var t2 = perf_counter_ns()
    n = 1
    var t3 = perf_counter_ns()
    for _ in range(reps):
        queue[].execute(
            func,
            grid=(UInt32(1), UInt32(1), UInt32(1)),
            block=(UInt32(1), UInt32(1), UInt32(1)),
            args=args,
            arg_sizes=rebind[Pointer[UInt64, MutAnyOrigin]](arg_sizes),
            num_args=UInt32(4),
        )
        queue[].synchronize()
    var t4 = perf_counter_ns()
    for _ in range(reps):
        queue[].execute(
            func,
            grid=(UInt32(1), UInt32(1), UInt32(1)),
            block=(UInt32(1), UInt32(1), UInt32(1)),
            args=args,
            arg_sizes=rebind[Pointer[UInt64, MutAnyOrigin]](arg_sizes),
            num_args=UInt32(4),
        )
    queue[].synchronize()
    var t5 = perf_counter_ns()
    print("whole vector: launch+sync us:", Float64(t1 - t0) / reps / 1000)
    print(
        "whole vector: per launch, batched us:", Float64(t2 - t1) / reps / 1000
    )
    print("one work-item: launch+sync us:", Float64(t4 - t3) / reps / 1000)
    print(
        "one work-item: per launch, batched us:", Float64(t5 - t4) / reps / 1000
    )

    args.unsafe_free()
    arg_sizes.unsafe_free()
    a.unsafe_free()
    b.unsafe_free()
    out.unsafe_free()
    ctx.unload_function(func)
    ctx.free_sync(out_dev^)
    ctx.free_sync(b_dev^)
    ctx.free_sync(a_dev^)
    if wrong:
        raise Error("vector add gave wrong results")


def main() raises:
    var spec = getenv("MODULAR_DRIVER_PLUGINS")
    if not spec:
        raise Error("MODULAR_DRIVER_PLUGINS not set")
    var driver = Driver.create(spec)
    print("driver:", driver[].name(), "devices:", driver[].device_count())
    var device: ArcPointer[Device[get_device_spec[0]()]] = driver[].get_device[
        0
    ]()
    var ctx = device[].get_context()
    run(ctx[])
