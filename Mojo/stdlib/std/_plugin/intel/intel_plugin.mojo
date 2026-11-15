# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

# ===----------------------------------------------------------------------=== #
# ===----------------------------------------------------------------------=== #
"""`PluginHooks` for Intel GPUs reached through OpenCL-flavored SPIR-V.

The Mojo compiler lowers these targets through the Intel compiler plugin
(`IntelCompilerPlugin.so`, loaded from `MODULAR_COMPILER_PLUGINS`), and the
Level Zero driver plugin runs them. Hooks are added here as each GPU feature
gains an Intel lowering; the rest keep the stdlib's defaults.
"""

from std._gpu.primitives.id import _GPUIdPluginHookFnType
from std._plugin._trait import (
    PluginHooks,
    StdPlugin,
    TargetDeviceKind,
    _GlobalPerfCounterNsPluginHookFnType,
    _PrintEmitPluginHookFnType,
    _PrintfEmitPluginHookFnType,
    _SleepPluginHookFnType,
)
from std.collections import OptionalReg
from std.collections.string.string_span import (
    _get_kgen_string,
    get_static_string,
)
from std.ffi import CStringSpan, external_call
from std.io.io import _printf
from std.reflection.location import SourceLocation
from std.io.file_descriptor import RawFileDescriptor
from std.math.math import _UnaryMathOp, _UnaryMathPluginHookFnType
from std.memory import AddressSpace
from std.sys.intrinsics import PrefetchOptions, _PrefetchPluginHookFnType
from std.sys.info import CompilationTarget

from .intel_gpu_id import _intel_gpu_id_fn
from .intel_math import _intel_unary_math_fn
from .intel_time import _clock_hz, _global_perf_counter_ns, _sleep


struct IntelPlugin(StdPlugin):
    """The standard library plugin for Intel GPUs."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["intel"]()
    comptime HookFor[target: CompilationTarget] = IntelPluginHooks[target]


struct IntelPluginHooks[target: CompilationTarget](PluginHooks):
    """The standard library hooks for an Intel GPU target."""

    comptime device_kind = TargetDeviceKind.gpu

    # LLVM's SPIR-V backend can't select llvm.vector.{extract,insert,splice},
    # so SIMD.slice, insert, rotate and shift build their results from
    # element accesses and shuffles.
    comptime lowers_subvector_intrinsics = False

    # Vectors of pointers need SPV_INTEL_masked_gather_scatter, which IGC
    # rejects, so gather() and scatter() go element by element.
    comptime lowers_masked_gather_scatter = False

    comptime prefetch_fn[
        address_space: AddressSpace, params: PrefetchOptions
    ]: OptionalReg[_PrefetchPluginHookFnType[address_space]] = _prefetch[
        address_space
    ]

    @staticmethod
    @inline(.always)
    def abort_fn():
        # IGC rejects SPV_KHR_abort, so LLVM drops the stdlib's trap that
        # follows and the work-item just ends. To report the abort, this
        # raises a flag the Level Zero driver plugin owns; the plugin fails
        # the next synchronize (the Intel compiler plugin defines the
        # function, see its Abort.h).
        external_call["__mojo_intel_abort", NoneType]()

    comptime unary_math_fn[
        op: _UnaryMathOp, dtype: DType, width: Int
    ]: OptionalReg[_UnaryMathPluginHookFnType] = _intel_unary_math_fn[
        op, dtype
    ]()

    comptime gpu_id_fn[name: StaticString, dim: StaticString]: OptionalReg[
        _GPUIdPluginHookFnType
    ] = _intel_gpu_id_fn[name, dim]()

    comptime global_perf_counter_ns_fn: OptionalReg[
        _GlobalPerfCounterNsPluginHookFnType
    ] = _global_perf_counter_ns[_clock_hz[Self.target]()]

    comptime sleep_fn: OptionalReg[_SleepPluginHookFnType] = _sleep[
        _clock_hz[Self.target]()
    ]

    comptime print_emit_fn: OptionalReg[
        _PrintEmitPluginHookFnType
    ] = _print_emit

    @staticmethod
    @inline(.always)
    def debug_assert_emit_fn[
        O: Origin
    ](message: Pointer[Byte, O], length: Int, loc: SourceLocation):
        _debug_assert_emit(message, length, loc)

    comptime _handles_debug_assert: Bool = True

    comptime printf_emit_fn[fmt: StaticString, *types: AnyType]: OptionalReg[
        _PrintfEmitPluginHookFnType[*types]
    ] = _printf_emit[fmt, *types]


@inline(.always)
def _printf_emit[
    fmt: StaticString, *types: AnyType
](*args: *types, file: RawFileDescriptor):
    """Formats `args` into `fmt` with OpenCL `printf`.

    The Intel compiler plugin moves the format string into the constant
    address space, where OpenCL requires it. The driver copies the device's
    printf buffer to the host's stdout, so `file` is ignored.
    """
    # Pass the values themselves, not references to them.
    _ = external_call["printf", Int32, num_fixed_args=1](
        get_static_string[fmt]().as_c_string_span(),
        args.get_loaded_kgen_pack(),
    )


@inline(.always)
def _print_emit[O: Origin](cstr: CStringSpan[O], file: RawFileDescriptor):
    """Writes `print()` output through OpenCL `printf`.

    The driver copies the device's printf buffer to the host's stdout, so
    `file` is ignored.
    """
    _printf_bytes(cstr.ptr().unsafe_bitcast[Byte](), len(cstr))


@inline(.always)
def _printf_bytes[O: Origin](bytes: Pointer[Byte, O], length: Int):
    """Writes `length` bytes through OpenCL `printf`.

    IGC takes `%s` only for string literals, which it records in the module;
    the driver reads any other `%s` argument as a host address. A runtime
    string therefore goes out with `%c`, eight bytes per call.
    """
    var i = 0
    while i + 8 <= length:
        _printf["%c%c%c%c%c%c%c%c"](
            Int32(bytes[unsafe_offset=i]),
            Int32(bytes[unsafe_offset=i + 1]),
            Int32(bytes[unsafe_offset=i + 2]),
            Int32(bytes[unsafe_offset=i + 3]),
            Int32(bytes[unsafe_offset=i + 4]),
            Int32(bytes[unsafe_offset=i + 5]),
            Int32(bytes[unsafe_offset=i + 6]),
            Int32(bytes[unsafe_offset=i + 7]),
        )
        i += 8
    while i < length:
        _printf["%c"](Int32(bytes[unsafe_offset=i]))
        i += 1


comptime _ASSERT_LINE_BYTES = 1024
"""The most bytes of a failed `debug_assert`'s line; a longer file name or
message is cut short."""

comptime _ASSERT_MIDDLE: StaticString = (
    ":%llu:%llu: block: [%llu,%llu,%llu] thread: [%llu,%llu,%llu] Assert"
    " Error: "
)
"""The specifiers between a failed `debug_assert`'s file name and message."""


struct _RuntimeFormat:
    """A `printf` format built on the device.

    IGC copies a format that isn't a string literal into the printf buffer,
    and the driver formats it there (the compiler plugin passes it through a
    generic pointer, with SPV_EXT_relaxed_printf_string_address_space). One
    `printf` per line keeps lines from different work-items whole: each call
    writes a work-item's entry in one piece, while separate calls interleave
    with other work-items'.
    """

    var _data: Array[Byte, _ASSERT_LINE_BYTES]
    var _pos: Int

    @inline(.always)
    def __init__(out self):
        self._data = {uninitialized = True}
        self._pos = 0

    @inline(.always)
    def specifiers(mut self, text: StaticString):
        """Appends `text` as it is, for `printf`'s arguments. The caller
        leaves room for it."""
        # `while`, not `for ... in range`: this runs inside the assertion
        # reporter, which nothing it calls may assert from again.
        var bytes = text.unsafe_ptr()
        var i = 0
        while i < text.byte_length():
            self._data.unsafe_ptr()[unsafe_offset=self._pos] = bytes[
                unsafe_offset=i
            ]
            self._pos += 1
            i += 1

    @inline(.always)
    def text[
        O: Origin
    ](mut self, bytes: Pointer[Byte, O], length: Int, end: Int):
        """Appends `length` bytes with each `%` doubled, stopping at `end`."""
        comptime percent = Byte(37)  # "%"
        var i = 0
        while i < length:
            var byte = bytes[unsafe_offset=i]
            var size = 2 if byte == percent else 1
            if self._pos + size > end:
                return
            self._data.unsafe_ptr()[unsafe_offset=self._pos] = byte
            self._pos += 1
            if size == 2:
                self._data.unsafe_ptr()[unsafe_offset=self._pos] = byte
                self._pos += 1
            i += 1

    @inline(.always)
    def nul_terminate(mut self) -> Pointer[Byte, origin_of(self._data)]:
        """Ends the format; the caller left room for the nul."""
        self._data.unsafe_ptr()[unsafe_offset=self._pos] = 0
        self._pos += 1
        return self._data.unsafe_ptr()


@inline(.always)
def _debug_assert_emit[
    O: Origin
](message: Pointer[Byte, O], length: Int, loc: SourceLocation):
    """Writes a failed `debug_assert` message through OpenCL `printf`, one
    `printf` for the whole line."""
    # `_debug_assert_msg` hands every assertion to this hook, including ones
    # evaluated at compile time, so it keeps the stdlib's compile-time message.
    if __is_run_in_comptime_interpreter:
        print("At: ", loc, ": Assert Error: ", message, sep="")
        return

    from std._gpu.primitives.id import block_idx, thread_idx

    # The file name and the message reach this hook at run time
    # (`_debug_assert_msg` isn't inlined), so neither is a literal IGC can
    # record for `%s` (the driver reads any other `%s` argument as a host
    # address and crashes in `snprintf`). Both become part of the format.
    comptime tail = 2  # "\n" and the nul
    var line = _RuntimeFormat()
    line.specifiers("At: ")
    var file_name = loc.file_name()
    line.text(
        file_name.unsafe_ptr(),
        file_name.byte_length(),
        _ASSERT_LINE_BYTES - _ASSERT_MIDDLE.byte_length() - tail,
    )
    line.specifiers(_ASSERT_MIDDLE)
    # `length` counts the message's nul terminator.
    var n = length
    if n > 0 and message[unsafe_offset=n - 1] == 0:
        n -= 1
    line.text(message, n, _ASSERT_LINE_BYTES - tail)
    line.specifiers("\n")
    _ = external_call["printf", Int32, num_fixed_args=1](
        line.nul_terminate(),
        loc.line(),
        loc.column(),
        UInt(block_idx.x),
        UInt(block_idx.y),
        UInt(block_idx.z),
        UInt(thread_idx.x),
        UInt(thread_idx.y),
        UInt(thread_idx.z),
    )


@inline(.always)
def _prefetch[
    address_space: AddressSpace
](addr: Pointer[NoneType, ImmUntrackedOrigin, address_space=address_space]):
    """Prefetches a global-memory address with OpenCL `prefetch`; a hint for
    any other address space is dropped, since OpenCL's `prefetch` takes only
    global pointers (a generic one may point anywhere)."""
    comptime if address_space == AddressSpace.GLOBAL:
        # void prefetch(const __global char *p, size_t num_gentypes)
        external_call["_Z8prefetchPU3AS1Kcm", NoneType](
            addr.unsafe_bitcast[Int8](), UInt64(1)
        )
