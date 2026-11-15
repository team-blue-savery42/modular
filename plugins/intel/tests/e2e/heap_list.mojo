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

# Kernels that allocate: a List in a kernel calls malloc and
# free, which the compiler plugin defines over the driver plugin's heap.
# Each work-item i grows a list to (i % LEN) + 1 values i * 8 + k and writes
# their sum to out[i]; the host checks every one. `-D KERNEL=<name>` runs one
# check alone (default: all): `list` (the List), `direct` (malloc, a store
# and a load through the block, no List), `realloc` (C's realloc: from null,
# growing, then shrinking; each copy checked to be min(old, new) bytes).
# `-D ITEMS=<n>` sets the work-items (default 4096, groups of 64), `-D LEN=<n>`
# the list lengths (default 7), `-D ROUNDS=<n>` the launches of each kernel,
# each checked (default 1; freed blocks are reused, so the 8 MiB
# heap serves any number: without reuse `list` ran out after a few dozen).

from std.ffi import external_call
from std.memory import alloc
from std.memory.pointer import OptionalPointer
from std.os import abort
from std.memory.memory import _free, _malloc
from std.sys.defines import get_defined_int, get_defined_string

from hal_gpu import GpuContext, launch, load, open_context
from max.gpu import global_idx

comptime _KERNEL = get_defined_string["KERNEL", "all"]()
comptime _ITEMS = get_defined_int["ITEMS", 4096]()
comptime _LEN = get_defined_int["LEN", 7]()
comptime _ROUNDS = get_defined_int["ROUNDS", 1]()
comptime _ABORT = get_defined_string["ABORT", "no"]() == "yes"


def _expected(i: Int) -> Int32:
    var count = i % _LEN + 1
    return Int32(count * i * 8 + count * (count - 1) // 2)


def _list_kernel(output: MutPointer[Int32, MutAnyOrigin]):
    var i = Int(global_idx.x)
    var values = List[Int32]()
    for k in range(i % _LEN + 1):
        values.append(Int32(i * 8 + k))
    var total: Int32 = 0
    for k in range(len(values)):
        total += values[k]
    output.unsafe_offset(i).unsafe_store(total)


def _list_capacity_kernel(output: MutPointer[Int32, MutAnyOrigin]):
    var i = Int(global_idx.x)
    var values = List[Int32](capacity=_LEN + 1)
    for k in range(i % _LEN + 1):
        values.append(Int32(i * 8 + k))
    var total: Int32 = 0
    for k in range(len(values)):
        total += values[k]
    output.unsafe_offset(i).unsafe_store(total)


def _direct_kernel(output: MutPointer[Int32, MutAnyOrigin]):
    var i = Int(global_idx.x)
    var count = i % _LEN + 1
    var block = _malloc[Int32](4 * count)
    if not block:
        comptime if _ABORT:
            abort("heap_list: malloc returned null")
        output.unsafe_offset(i).unsafe_store(-1)
        return
    var p = block.unsafe_value()
    for k in range(count):
        p.unsafe_offset(k).unsafe_store(Int32(i * 8 + k))
    var total: Int32 = 0
    for k in range(count):
        total += p.unsafe_offset(k).unsafe_load()
    _free(block)
    output.unsafe_offset(i).unsafe_store(total)


comptime _Block = OptionalPointer[
    Int32, MutUntrackedOrigin, address_space=.GENERIC
]
comptime _CBlock = _Block.T._mlir_type


def _realloc(block: _Block, bytes: Int) -> _Block:
    """C's realloc, as a kernel calls it (the compiler plugin defines it)."""
    var c_block = Pointer(to=block).unsafe_bitcast[_CBlock]()[]
    var grown = external_call["realloc", _CBlock](c_block, bytes)
    return Pointer(to=grown).unsafe_bitcast[_Block]()[]


def _mark(i: Int, k: Int, salt: Int) -> Int32:
    """The value work-item i writes into slot k of its first block's padding
    in the launch whose salt (the host's round number) is `salt`: negative,
    nonzero and unique (for up to 4,096 work-items and 32,768 rounds), so no
    other block holds it, not even this work-item's first block of an
    earlier launch, which a later grow may reuse."""
    return Int32(-(1 << 30) - salt * 32768 - i * 8 - k)


def _fill(i: Int, k: Int) -> Int32:
    """The value work-item i writes into slot k of the part a grow added."""
    return Int32(-2_000_000 - i * 8 - k)


def _shrink_mark(i: Int, k: Int) -> Int32:
    """The value work-item i writes into slot k (1 to 3) of its grown block
    just before shrinking it: no other block ever holds it, so a shrunk block
    that holds it got more than 4 bytes copied (a reused block may hold any
    work-item's earlier values, its own included)."""
    return Int32(-3_000_000 - i * 8 - k)


def _realloc_kernel(output: MutPointer[Int32, MutAnyOrigin]):
    """Each work-item reallocs from null (a malloc) to `count` values i * 8 + k,
    grows the block by LEN values, then shrinks it to one value, and checks
    that each realloc copied exactly min(old, new) bytes. The heap rounds a
    block up to at least 16 bytes, so a block of `count`
    values has padding up to the next multiple of 4 values; a new block is
    fresh memory or one another work-item freed, which holds only values
    unique to that work-item. The first block's padding holds
    _mark values: a grow that copied more than the old size carries them into
    the new block. Values 1 to 3 of the grown block are set to _shrink_mark
    values, unique to the work-item, just before the shrink: a shrink to 4
    bytes that copied 8 bytes or more leaves at least value 1 in the new
    block's padding (and one that copies more than 16 writes past it, over
    the next block). The check compares whole values, so a copy of 5 to 7
    bytes, which changes only part of value 1, isn't caught.
    out[i] is the sum of the `count` values read after growing, as
    the other checks write; -1 if a realloc returned null, -2 if the shrunk
    block lost its first value, -3 if the grown part didn't hold its values
    (two work-items got overlapping memory), -4 if the grow copied more than
    the old size, -5 if one of the new block's values 1 to 3 is its
    _shrink_mark (a shrink that copied 8 bytes or more).
    (A shrink that copies the whole old size also writes over the next
    block's header; on the B70 another work-item's realloc then read a wrong
    size and the kernel faulted, so this check then fails by a fault: run a
    throwaway GPU process after it.)"""
    var i = Int(global_idx.x)
    var salt = Int(output.unsafe_offset(i).unsafe_load())
    var count = i % _LEN + 1
    var padded = (count + 3) // 4 * 4
    var first = _realloc(_Block(None), 4 * count)
    if not first:
        output.unsafe_offset(i).unsafe_store(-1)
        return
    var p = first.unsafe_value()
    for k in range(count):
        p.unsafe_offset(k).unsafe_store(Int32(i * 8 + k))
    for k in range(count, padded):
        p.unsafe_offset(k).unsafe_store(_mark(i, k, salt))
    var grown = _realloc(first, 4 * (count + _LEN))
    if not grown:
        output.unsafe_offset(i).unsafe_store(-1)
        return
    var q = grown.unsafe_value()
    for k in range(count, padded):
        if q.unsafe_offset(k).unsafe_load() == _mark(i, k, salt):
            output.unsafe_offset(i).unsafe_store(-4)
            return
    for k in range(_LEN):
        q.unsafe_offset(count + k).unsafe_store(_fill(i, k))
    var total: Int32 = 0
    for k in range(count):
        total += q.unsafe_offset(k).unsafe_load()
    for k in range(_LEN):
        if q.unsafe_offset(count + k).unsafe_load() != _fill(i, k):
            output.unsafe_offset(i).unsafe_store(-3)
            return
    for k in range(1, 4):
        q.unsafe_offset(k).unsafe_store(_shrink_mark(i, k))
    var shrunk = _realloc(grown, 4)
    if not shrunk:
        output.unsafe_offset(i).unsafe_store(-1)
        return
    var s = shrunk.unsafe_value()
    if s.unsafe_load() != Int32(i * 8):
        output.unsafe_offset(i).unsafe_store(-2)
        return
    for k in range(1, 4):
        if s.unsafe_offset(k).unsafe_load() == _shrink_mark(i, k):
            output.unsafe_offset(i).unsafe_store(-5)
            return
    _free(shrunk)
    output.unsafe_offset(i).unsafe_store(total)


def check[
    fn_type: TrivialRegisterPassable, func: fn_type
](mut ctx: GpuContext, name: String) raises:
    if _KERNEL != "all" and _KERNEL != name:
        return
    var queue = ctx.create_queue()
    var function = load[fn_type, func](ctx)
    var bytes = UInt64(4 * _ITEMS)
    var out_dev = ctx.alloc_sync(bytes)
    var groups = (_ITEMS + 63) // 64
    var host = alloc[Int32]({length = _ITEMS}).unsafe_leak()
    var wrong = 0
    for round in range(_ROUNDS):
        # out[i] starts as the round number: the realloc check's salt.
        for i in range(_ITEMS):
            host[unsafe_offset=i] = Int32(round)
        queue[].copy_to_device(out_dev.view(), host.unsafe_bitcast[UInt8]())
        launch(queue[], function, groups, 64, [ctx.memory_get_address(out_dev)])
        queue[].copy_from_device(host.unsafe_bitcast[UInt8](), out_dev.view())
        queue[].synchronize()
        for i in range(_ITEMS):
            var got = host[unsafe_offset=i]
            if got != _expected(i):
                if wrong < 8:
                    print(
                        name,
                        "round",
                        round,
                        "out[",
                        i,
                        "] =",
                        got,
                        "expected",
                        _expected(i),
                    )
                wrong += 1
    host.unsafe_free()
    ctx.unload_function(function)
    ctx.free_sync(out_dev^)
    if wrong:
        raise Error(
            String(name, ": ", wrong, " of ", _ITEMS * _ROUNDS, " wrong")
        )
    print("PASS", name, _ITEMS, "work-items", _ROUNDS, "rounds", flush=True)


def main() raises:
    comptime known = (
        _KERNEL == "all"
        or _KERNEL == "list"
        or _KERNEL == "list_capacity"
        or _KERNEL == "direct"
        or _KERNEL == "realloc"
    )
    if not known:
        raise Error("unknown -D KERNEL=" + _KERNEL)
    var ctx = open_context()
    check[type_of(_direct_kernel), _direct_kernel](ctx[], "direct")
    check[type_of(_list_capacity_kernel), _list_capacity_kernel](
        ctx[], "list_capacity"
    )
    check[type_of(_list_kernel), _list_kernel](ctx[], "list")
    check[type_of(_realloc_kernel), _realloc_kernel](ctx[], "realloc")
