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

# Launch helpers for running Mojo kernels on the B70 through the HAL client
# API, shared by the ports of the stdlib's and MAX's GPU tests in
# this directory. Kernels here take only 8-byte arguments (device addresses
# and 64-bit integers), passed as a list of words.

from std.compile import compile_info
from std.memory import alloc, ArcPointer, OpaquePointer, Pointer
from std.os import getenv
from std.sys.info import CompilationTarget

from max._driver import Driver, Context
from max._driver.max_license.hal.device import Device, get_device_spec
from max._driver.max_license.hal.extensions.intel import (
    SupportsIntelDeviceFeaturesV1,
)
from max._driver.max_license.hal.plugin import FunctionHandle
from max._driver.max_license.hal.queue import Queue

comptime Spec = get_device_spec[0]()
comptime GpuContext = Context[Spec]
comptime GpuQueue = Queue[Spec]


def open_context() raises -> ArcPointer[GpuContext]:
    """A context on device 0 of the driver plugin in MODULAR_DRIVER_PLUGINS."""
    var spec = getenv("MODULAR_DRIVER_PLUGINS")
    if not spec:
        raise Error("MODULAR_DRIVER_PLUGINS not set")
    var driver = Driver.create(spec)
    var device: ArcPointer[Device[Spec]] = driver[].get_device[0]()
    return device[].get_context()


def xe_cores() raises -> Int:
    """Device 0's Xe-cores (Level Zero slices x sub-slices), from the Intel
    device features."""
    var driver = Driver.create(getenv("MODULAR_DRIVER_PLUGINS"))
    var device: ArcPointer[Device[Spec]] = driver[].get_device[0]()
    comptime Ext = type_of(device[].extensions())
    comptime if conforms_to(Ext, SupportsIntelDeviceFeaturesV1):
        var query = device[].extensions().intel_device_features()
        if query:
            var f = query.value().device_features(
                device[]._raw[], device[]._handle
            )
            return Int(f.num_slices) * Int(f.num_subslices_per_slice)
    raise Error("the driver plugin reports no Intel device features")


def load[
    fn_type: TrivialRegisterPassable, func: fn_type
](mut ctx: GpuContext) raises -> FunctionHandle:
    """Compiles `func` for the target and loads it."""
    # Owned, not a `ref` into the call's temporary: Mojo now ends that
    # temporary when the destructuring statement ends, which unloaded the
    # bundle before the function was loaded from it.
    var compiled = ctx.compile[fn_type, func]()
    var handle = ctx.load_function(compiled[0], compiled[1].function_name)
    _ = compiled^
    return handle


def load_for[
    fn_type: TrivialRegisterPassable, func: fn_type, target: CompilationTarget
](mut ctx: GpuContext) raises -> FunctionHandle:
    """Compiles `func` for `target`, another target of the context's device
    (such as its sub-group size 32 variant), and loads it."""
    var info = compile_info[func, target=target, emission_kind="object"]()
    var bundle = ctx.load_bundle(info.asm)
    return ctx.load_function(bundle, info.function_name)


def launch(
    queue: GpuQueue,
    func: FunctionHandle,
    grid: Int,
    block: Int,
    words: List[UInt64],
    shared_mem_bytes: Int = 0,
    grid_y: Int = 1,
    block_y: Int = 1,
) raises:
    """Enqueues `func` on `queue` with a grid of `grid` x `grid_y` groups of
    `block` x `block_y` work-items; each word is one argument."""
    var count = len(words)
    var slots = max(count, 1)
    var args = alloc[OpaquePointer[ImmUntrackedOrigin]](
        {length = slots}
    ).unsafe_leak()
    var sizes = alloc[UInt64]({length = slots}).unsafe_leak()
    var values = alloc[UInt64]({length = slots}).unsafe_leak()
    for i in range(count):
        values[unsafe_offset=i] = words[i]
        args[unsafe_offset=i] = rebind[OpaquePointer[MutUntrackedOrigin]](
            Pointer(to=values[unsafe_offset=i])
        )
        sizes[unsafe_offset=i] = UInt64(8)
    try:
        queue.execute(
            func,
            grid=(UInt32(grid), UInt32(grid_y), UInt32(1)),
            block=(UInt32(block), UInt32(block_y), UInt32(1)),
            args=args,
            arg_sizes=rebind[Pointer[UInt64, MutAnyOrigin]](sizes),
            num_args=UInt32(count),
            shared_mem_bytes=UInt32(shared_mem_bytes),
        )
    finally:
        # The driver plugin copies arguments when the launch is enqueued.
        args.unsafe_free()
        sizes.unsafe_free()
        values.unsafe_free()
