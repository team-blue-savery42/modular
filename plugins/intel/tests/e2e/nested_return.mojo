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

# Nested struct return values of internal functions. A List that
# grows in a kernel calls `List._realloc`, which returns `{ { ptr, i64 }, i64 }`
# (the new buffer and capacity, and the length); with that value the kernel
# stored its first element at address 0 (page fault, write). Here a callee
# that isn't inlined returns the same shape: `Outer(Inner(output + i, i), 2 *
# i)`, and work-item i stores `inner.n + outer.m` (3 * i) through
# `inner.ptr`; the host checks every one. `-D SHAPE=flat` returns the three
# fields as one flat struct instead, for comparison.

from std.memory import alloc
from std.sys.defines import get_defined_string

from hal_gpu import launch, load, open_context
from max.gpu import global_idx

comptime _SHAPE = get_defined_string["SHAPE", "nested"]()
comptime _ITEMS = 256
comptime _Ptr = MutPointer[Int32, MutUntrackedOrigin]


@fieldwise_init
struct Inner(Copyable):
    var ptr: _Ptr
    var n: Int


@fieldwise_init
struct Outer(Copyable):
    var inner: Inner
    var m: Int


@fieldwise_init
struct Flat(Copyable):
    var ptr: _Ptr
    var n: Int
    var m: Int


@inline(.never)
def _nested(output: MutPointer[Int32, MutAnyOrigin], i: Int) -> Outer:
    return Outer(Inner(rebind[_Ptr](output.unsafe_offset(i)), i), 2 * i)


@inline(.never)
def _flat(output: MutPointer[Int32, MutAnyOrigin], i: Int) -> Flat:
    return Flat(rebind[_Ptr](output.unsafe_offset(i)), i, 2 * i)


def _kernel(output: MutPointer[Int32, MutAnyOrigin]):
    var i = Int(global_idx.x)
    comptime if _SHAPE == "flat":
        var f = _flat(output, i)
        f.ptr.unsafe_store(Int32(f.n + f.m))
    else:
        var o = _nested(output, i)
        o.inner.ptr.unsafe_store(Int32(o.inner.n + o.m))


def main() raises:
    if _SHAPE != "nested" and _SHAPE != "flat":
        raise Error("unknown -D SHAPE=" + _SHAPE)
    var ctx = open_context()
    var queue = ctx[].create_queue()
    var function = load[type_of(_kernel), _kernel](ctx[])
    var out_dev = ctx[].alloc_sync(UInt64(4 * _ITEMS))
    queue[].set_memory(out_dev.view(), value=UInt8(0))
    launch(
        queue[], function, _ITEMS // 64, 64, [ctx[].memory_get_address(out_dev)]
    )
    var host = alloc[Int32]({length = _ITEMS}).unsafe_leak()
    queue[].copy_from_device(host.unsafe_bitcast[UInt8](), out_dev.view())
    queue[].synchronize()
    var wrong = 0
    for i in range(_ITEMS):
        if host[unsafe_offset=i] != Int32(3 * i):
            if wrong < 8:
                print(
                    "out[", i, "] =", host[unsafe_offset=i], "expected", 3 * i
                )
            wrong += 1
    host.unsafe_free()
    ctx[].unload_function(function)
    ctx[].free_sync(out_dev^)
    if wrong:
        raise Error(String(_SHAPE, ": ", wrong, " of ", _ITEMS, " wrong"))
    print("PASS", _SHAPE, _ITEMS, "work-items", flush=True)
