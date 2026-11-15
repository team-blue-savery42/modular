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

# Constant memory mappings through MAX's DeviceContext:
# `enqueue_function(..., constant_memory=[ConstantMemoryMapping])`
# fills a kernel's named constant-space array before the launch. The compat
# layer's `_fill_constant_memory` reads the driver's `global_address:NAME`
# and `global_size:NAME` function properties, wraps the global and copies the
# mapping on the launch's stream. Before that change the compat layer refused
# every mapping. Each of its three call sites is reached here:
# `DeviceFunction._call_with_pack_checked` (enqueue_function[kernel], and a
# compiled function on the context and on a stream),
# `DeviceFunction._call_with_pack` (no public HAL caller, so called
# directly) and `DeviceExternalFunction._call_with_pack` (a function loaded
# with load_function, on the context and on a stream). Each launch gets a
# different table, so a launch that missed its copy shows the previous one.
# Then a mapping shorter than the global (only its bytes change), a mapping
# with no such global and one larger than the global (both refused).
# run-e2e: device-context

from max.gpu import thread_idx
from max.gpu.host import ConstantMemoryMapping, DeviceBuffer, DeviceContext
from max.gpu.host.compile import _compile_code
from std.memory import unsafe_stack_allocation

comptime N = 16


def kernel(data: MutPointer[Float32, MutAnyOrigin]):
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


def table(
    scale: Float32, length: Int = N, offset: Float32 = 0
) -> List[Float32]:
    var t = List[Float32](capacity=length)
    for i in range(length):
        t.append(offset + Float32(i) * scale)
    return t^


def mapping(
    name: StaticString, mut t: List[Float32]
) -> List[ConstantMemoryMapping]:
    return [
        ConstantMemoryMapping(
            name,
            t.unsafe_ptr()
            .unsafe_bitcast[NoneType]()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            t.byte_length(),
        )
    ]


def check(
    ctx: DeviceContext,
    what: String,
    out_buf: DeviceBuffer[DType.float32],
    want: List[Float32],
) raises -> Int:
    ctx.synchronize()
    var bad = 0
    with out_buf.map_to_host() as res:
        for i in range(N):
            if res[i] != want[i]:
                bad += 1
        print(
            "PASS" if bad == 0 else "FAIL",
            what,
            "wrong:",
            bad,
            "first:",
            res[0],
            res[1],
            res[N - 1],
        )
    return bad


def expect_refusal(what: String, err: String, want: String) -> Int:
    if want in err:
        print("PASS", what, "refused:", err)
        return 0
    print("FAIL", what, "refused with the wrong error:", err)
    return 1


def main() raises:
    var ctx = DeviceContext()
    var out_buf = ctx.enqueue_create_buffer[DType.float32](N)
    var bad = 0

    # Each subtest runs even if an earlier one raised, so a refusal at one
    # call site shows up as that subtest's FAIL.
    try:
        var t = table(1)
        ctx.enqueue_function[kernel](
            out_buf,
            grid_dim=1,
            block_dim=N,
            constant_memory=mapping("static_constant", t),
        )
        bad += check(ctx, "enqueue_function[kernel] (checked)", out_buf, t)
    except e:
        print("FAIL enqueue_function[kernel] (checked) raised:", e)
        bad += 1

    var compiled = ctx.compile_function[kernel]()
    try:
        var t = table(2)
        ctx.enqueue_function(
            compiled,
            out_buf,
            grid_dim=1,
            block_dim=N,
            constant_memory=mapping("static_constant", t),
        )
        bad += check(ctx, "compiled on the context (checked)", out_buf, t)
    except e:
        print("FAIL compiled on the context (checked) raised:", e)
        bad += 1

    try:
        var t = table(3)
        ctx.stream().enqueue_function(
            compiled,
            out_buf,
            grid_dim=1,
            block_dim=N,
            constant_memory=mapping("static_constant", t),
        )
        bad += check(ctx, "compiled on a stream (checked)", out_buf, t)
    except e:
        print("FAIL compiled on a stream (checked) raised:", e)
        bad += 1

    try:
        var t = table(4)
        compiled._call_with_pack(
            ctx,
            out_buf.unsafe_ptr(),
            grid_dim=1,
            block_dim=N,
            constant_memory=mapping("static_constant", t),
        )
        bad += check(ctx, "compiled _call_with_pack (unchecked)", out_buf, t)
    except e:
        print("FAIL compiled _call_with_pack (unchecked) raised:", e)
        bad += 1

    var info = _compile_code[kernel, emission_kind="object"]()
    var external = ctx.load_function[kernel](
        function_name=String(info.function_name), asm=String(info.asm)
    )
    try:
        var t = table(5)
        ctx.enqueue_function(
            external,
            out_buf.unsafe_ptr(),
            grid_dim=1,
            block_dim=N,
            constant_memory=mapping("static_constant", t),
        )
        bad += check(ctx, "external on the context", out_buf, t)
    except e:
        print("FAIL external on the context raised:", e)
        bad += 1

    try:
        var t = table(6)
        ctx.stream().enqueue_function(
            external,
            out_buf.unsafe_ptr(),
            grid_dim=1,
            block_dim=N,
            constant_memory=mapping("static_constant", t),
        )
        bad += check(ctx, "external on a stream", out_buf, t)
    except e:
        print("FAIL external on a stream raised:", e)
        bad += 1

    # The global holds 6i (if the last launch was right); a mapping of the
    # first 8 floats changes only those.
    try:
        var t = table(1, length=N // 2, offset=100)
        ctx.enqueue_function[kernel](
            out_buf,
            grid_dim=1,
            block_dim=N,
            constant_memory=mapping("static_constant", t),
        )
        var want = table(6)
        for i in range(N // 2):
            want[i] = t[i]
        bad += check(ctx, "partial mapping (8 of 16)", out_buf, want)
    except e:
        print("FAIL partial mapping (8 of 16) raised:", e)
        bad += 1

    try:
        var t = table(1)
        ctx.enqueue_function[kernel](
            out_buf,
            grid_dim=1,
            block_dim=N,
            constant_memory=mapping("no_such_global", t),
        )
        ctx.synchronize()
        print("FAIL missing global accepted")
        bad += 1
    except e:
        bad += expect_refusal(
            "missing global",
            String(e),
            "the driver exposes no global `no_such_global`",
        )

    try:
        var t = table(1, length=2 * N)
        ctx.enqueue_function[kernel](
            out_buf,
            grid_dim=1,
            block_dim=N,
            constant_memory=mapping("static_constant", t),
        )
        ctx.synchronize()
        print("FAIL oversized mapping accepted")
        bad += 1
    except e:
        bad += expect_refusal(
            "oversized mapping",
            String(e),
            "has 128 bytes, but the global has 64.",
        )

    _ = out_buf^
    if bad:
        raise Error("constant memory mappings: ", bad, " failures")
    print("PASS all")
