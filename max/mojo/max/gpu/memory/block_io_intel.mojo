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

# ===----------------------------------------------------------------------=== #
# ===----------------------------------------------------------------------=== #
"""Intel Xe2 2D block I/O through `SPV_INTEL_2d_block_io`.

A sub-group of 16 work-items loads, stores or prefetches a whole 2D block of
a row-major matrix in global memory with one instruction (IGC's
`load_block2d`, `store_block2d`, `prefetch`): the hardware does the address
arithmetic and the bounds check, and every element outside the matrix reads
as zero (loads) or is skipped (stores, prefetches), so a partial block at the
matrix's edge needs no special code.

The matrix is described by its base pointer and, in elements, its `width`
(columns), `height` (rows) and `pitch` (elements from one row's start to the
next); a block starts at column `x` and row `y` (either may be negative or
run past the edge). A block is `block_width` columns by `block_height` rows,
and `count` blocks side by side are loaded at once.

Each work-item gets `n = block_elements[block_width, block_height, count]()`
values, `SIMD[dtype, n]` (declare a variable with that `n`, or `rebind` the
result to a literal width), laid out as the extension says (lower columns to
lower lanes; blocks one after another):

- `block_load`, 16 columns (16- and 32-bit): element `r` of lane `l` is row
  `r`, column `l`.
  32 columns (8-bit): lane `l` holds columns `2l` and `2l + 1` of each row.
  8 columns (32-bit): element `j` of lane `l` is row `2j + l // 8`, column
  `l % 8` (lanes 0-7 hold even rows, lanes 8-15 odd ones).
- `block_load_transform` (VNNI, for DPAS's B): element `k` of lane `l` is row
  `k`, column `l`, packed so that the 32-bit words are the extension's
  (rows `2j`, `2j + 1` in word `j` for 16-bit data, four rows for 8-bit):
  `bitcast[DType.int32, n // 2]` of a 16-bit result is the packed B `mma()`
  takes, and the 16-bit result itself is its typed B.
- `block_load_transpose` (32-bit, 8 columns): for 16 rows, element `c` of
  lane `l` is row `l`, column `c` of the block (each lane gets one row); for
  32 rows, element `2c + i` of lane `l` is row `l + 16i`, column `c`
  (measured on the B70, IGC 2.41 and master; the extension's general rule
  would give rows `2l + i`). For 16-bit pairs, load them as 32-bit and
  bitcast.
- `block_store` writes `block_load`'s layout back.

Shapes are the `cl_intel_subgroup_2d_block_io` ones (the client limits; IGC
accepts more) and are checked at compile time:

- load: 8-bit 32 wide, count 1 or 2, and 16 wide x 8, 16 or 32 rows,
  count 4; 16-bit 16 wide, count 1 or 2; 32-bit 16 wide, count 1, and
  8 wide, count 1 or 2.
- prefetch: as load, and 8-bit 16 wide x 32 rows, count 1 or 2.
- load_transform: 8-bit 16 wide x 32 rows, count 1, 2 or 4; 16-bit 16 wide x
  16 or 32 rows, count 1 or 2.
- load_transpose: 32-bit 8 wide x 16 or 32 rows, count 1.
- store: 8-bit 16 or 32 wide, 16- and 32-bit 16 wide; 1, 2, 4 or 8 rows,
  count 1.

Rows are 1, 2, 4, 8, 16 or 32 where not given.

Run-time requirements (the extension's; not checked unless built with
`-D ASSERT=all`): the base address is 64-byte aligned; `width` is at least
64 bytes and a multiple of 4 bytes (of the element size for 32-bit data);
`pitch` is at least `width` and a multiple of 16 bytes; `x` is a multiple of
4 for 8-bit and of 2 for 16-bit data; the matrix is in global memory; every
argument is the same in all work-items of the sub-group, and the sub-group is
full (sub-group size 16: kernels built for an Intel `-sg32` target are
rejected at compile time).
"""

from std.ffi import external_call
from std.memory import stack_allocation
from std.memory.address_space import AddressSpace
from std.sys import size_of

from max.gpu.globals import WARP_SIZE

# OpenCL (Itanium) mangled names of the built-ins, as clang declares them
# (`const global void *` source, `private void *` destination), so LLVM's
# SPIR-V backend gives the pointers CrossWorkgroup and Function storage.
comptime _LOAD = "_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1KviiiDv2_iPv"
comptime _TRANSFORM = (
    "_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1KviiiDv2_iPv"
)
comptime _TRANSPOSE = (
    "_Z41__spirv_Subgroup2DBlockLoadTransposeINTELiiiiPU3AS1KviiiDv2_iPv"
)
comptime _PREFETCH = (
    "_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1KviiiDv2_i"
)
comptime _STORE = "_Z33__spirv_Subgroup2DBlockStoreINTELiiiiPKvPU3AS1viiiDv2_i"

comptime _OP_LOAD = 0
comptime _OP_TRANSFORM = 1
comptime _OP_TRANSPOSE = 2
comptime _OP_STORE = 3
comptime _OP_PREFETCH = 4


def _op_name(op: Int) -> StaticString:
    if op == _OP_LOAD:
        return "load"
    if op == _OP_TRANSFORM:
        return "load_transform"
    if op == _OP_TRANSPOSE:
        return "load_transpose"
    if op == _OP_STORE:
        return "store"
    return "prefetch"


def _pow2_rows(h: Int, lo: Int) -> Bool:
    """`h` is a power of two in [lo, 32]."""
    return h >= lo and h <= 32 and (h & (h - 1)) == 0


def _supported[
    op: Int, bytes: Int, width: Int, height: Int, count: Int
]() -> Bool:
    """The `cl_intel_subgroup_2d_block_io` shapes (sub-group size 16)."""
    var one_or_two = count == 1 or count == 2
    if op == _OP_LOAD or op == _OP_PREFETCH:
        if bytes == 1:
            # Prefetch also has 16 wide x 32 rows, count 1 or 2.
            var prefetch16 = (
                op == _OP_PREFETCH
                and width == 16
                and height == 32
                and one_or_two
            )
            return (
                prefetch16
                or (width == 32 and one_or_two and _pow2_rows(height, 1))
                or (width == 16 and count == 4 and _pow2_rows(height, 8))
            )
        if bytes == 2:
            return width == 16 and one_or_two and _pow2_rows(height, 1)
        if bytes == 4:
            return _pow2_rows(height, 1) and (
                (width == 16 and count == 1) or (width == 8 and one_or_two)
            )
        return False
    elif op == _OP_TRANSFORM:
        if bytes == 1:
            return width == 16 and height == 32 and (one_or_two or count == 4)
        if bytes == 2:
            return width == 16 and (height == 16 or height == 32) and one_or_two
        return False
    elif op == _OP_TRANSPOSE:
        return (
            bytes == 4
            and width == 8
            and (height == 16 or height == 32)
            and count == 1
        )
    else:  # store
        return (
            count == 1
            and _pow2_rows(height, 1)
            and height <= 8
            and (width == 16 or (bytes == 1 and width == 32))
            and (bytes == 1 or bytes == 2 or bytes == 4)
        )


def block_elements[
    block_width: Int, block_height: Int, count: Int = 1
]() -> Int:
    """The number of values each of the 16 work-items holds for a block
    operation of this shape (the `SIMD` width of its value)."""
    return count * max(1, block_width * block_height // 16)


@inline(.always)
def _check[op: Int, dtype: DType, width: Int, height: Int, count: Int]():
    comptime assert WARP_SIZE == 16, String(
        t"Intel 2D block I/O needs sub-group size 16, but this kernel is built"
        t" for {WARP_SIZE}: build it for the target without `-sg32`"
    )
    comptime bytes = size_of[dtype]()
    comptime assert _supported[op, bytes, width, height, count](), String(
        t"no Intel 2D block {_op_name(op)} of {count} block(s) of {width} x"
        t" {height} {bytes}-byte elements: see max.gpu.memory.block_io_intel"
    )


@inline(.always)
def _geometry[
    dtype: DType
](
    base: Pointer[Scalar[dtype], _, address_space=_],
    width: Int,
    height: Int,
    pitch: Int,
) -> Tuple[Int32, Int32, Int32]:
    """The extension's width and pitch in bytes and height in rows."""
    comptime bytes = size_of[dtype]()
    debug_assert(
        Int(base) % 64 == 0
        and width * bytes >= 64
        and pitch >= width
        and (pitch * bytes) % 16 == 0,
        "2D block I/O: the base must be 64-byte aligned, the width at least",
        " 64 bytes and the pitch at least the width and a multiple of 16 bytes",
    )
    return (Int32(width * bytes), Int32(height), Int32(pitch * bytes))


@inline(.always)
def _load[
    op: Int,
    name: StaticString,
    dtype: DType,
    block_width: Int,
    block_height: Int,
    count: Int,
](
    base: Pointer[Scalar[dtype], _, address_space=_],
    width: Int,
    height: Int,
    pitch: Int,
    x: Int,
    y: Int,
) -> SIMD[dtype, block_elements[block_width, block_height, count]()]:
    _check[op, dtype, block_width, block_height, count]()
    comptime n = block_elements[block_width, block_height, count]()
    var g = _geometry(base, width, height, pitch)
    var dst = stack_allocation[n, dtype, address_space=AddressSpace.LOCAL]()
    external_call[name, NoneType](
        Int32(size_of[dtype]()),
        Int32(block_width),
        Int32(block_height),
        Int32(count),
        base.unsafe_bitcast[UInt8]().unsafe_address_space_cast[
            AddressSpace.GLOBAL
        ](),
        g[0],
        g[1],
        g[2],
        SIMD[DType.int32, 2](Int32(x), Int32(y)),
        dst,
    )
    return dst.unsafe_load[width=n]()


@inline(.always)
def block_load[
    dtype: DType, //, block_width: Int, block_height: Int, count: Int = 1
](
    base: Pointer[Scalar[dtype], _, address_space=_],
    width: Int,
    height: Int,
    pitch: Int,
    x: Int,
    y: Int,
) -> SIMD[dtype, block_elements[block_width, block_height, count]()]:
    """Loads `count` blocks of `block_width` x `block_height` elements at
    column `x`, row `y` of the matrix (zeros outside it) into the sub-group;
    see the module for the layout and the shapes.

    Parameters:
        dtype: The element type (1, 2 or 4 bytes).
        block_width: Columns per block.
        block_height: Rows per block.
        count: Blocks side by side.

    Args:
        base: The matrix's first element, in global memory, 64-byte aligned.
        width: The matrix's columns.
        height: The matrix's rows.
        pitch: Elements from one row's start to the next.
        x: The block's first column.
        y: The block's first row.

    Returns:
        This work-item's values of the blocks.
    """
    return _load[_OP_LOAD, _LOAD, dtype, block_width, block_height, count](
        base, width, height, pitch, x, y
    )


@inline(.always)
def block_load_transform[
    dtype: DType, //, block_width: Int, block_height: Int, count: Int = 1
](
    base: Pointer[Scalar[dtype], _, address_space=_],
    width: Int,
    height: Int,
    pitch: Int,
    x: Int,
    y: Int,
) -> SIMD[dtype, block_elements[block_width, block_height, count]()]:
    """Loads blocks as `block_load` does, but packed for DPAS's B (VNNI):
    element `k` of lane `l` is row `k`, column `l` of the block, and each
    32-bit word holds consecutive rows (two 16-bit or four 8-bit values).

    Parameters:
        dtype: The element type (1 or 2 bytes).
        block_width: Columns per block (16).
        block_height: Rows per block.
        count: Blocks side by side.

    Args:
        base: The matrix's first element, in global memory, 64-byte aligned.
        width: The matrix's columns.
        height: The matrix's rows.
        pitch: Elements from one row's start to the next.
        x: The block's first column.
        y: The block's first row.

    Returns:
        This work-item's values of the blocks.
    """
    return _load[
        _OP_TRANSFORM, _TRANSFORM, dtype, block_width, block_height, count
    ](base, width, height, pitch, x, y)


@inline(.always)
def block_load_transpose[
    dtype: DType, //, block_width: Int, block_height: Int, count: Int = 1
](
    base: Pointer[Scalar[dtype], _, address_space=_],
    width: Int,
    height: Int,
    pitch: Int,
    x: Int,
    y: Int,
) -> SIMD[dtype, block_elements[block_width, block_height, count]()]:
    """Loads a block transposed (32-bit elements, 8 columns of 16 or 32
    rows): for 16 rows, element `c` of lane `l` is row `l`, column `c` of the
    block; for 32 rows, element `2c + i` of lane `l` is row `l + 16i`.

    Parameters:
        dtype: The element type (4 bytes).
        block_width: Columns per block (8).
        block_height: Rows per block (16 or 32).
        count: Blocks side by side (1).

    Args:
        base: The matrix's first element, in global memory, 64-byte aligned.
        width: The matrix's columns.
        height: The matrix's rows.
        pitch: Elements from one row's start to the next.
        x: The block's first column.
        y: The block's first row.

    Returns:
        This work-item's values of the block.
    """
    return _load[
        _OP_TRANSPOSE, _TRANSPOSE, dtype, block_width, block_height, count
    ](base, width, height, pitch, x, y)


@inline(.always)
def block_store[
    dtype: DType, //, block_width: Int, block_height: Int
](
    base: Pointer[Scalar[dtype], _, address_space=_],
    width: Int,
    height: Int,
    pitch: Int,
    x: Int,
    y: Int,
    value: SIMD,
):
    """Stores one block of `block_width` x `block_height` elements from the
    sub-group at column `x`, row `y` (elements outside the matrix are not
    written), `block_load`'s layout.

    Parameters:
        dtype: The element type (1, 2 or 4 bytes).
        block_width: Columns of the block.
        block_height: Rows of the block (1, 2, 4 or 8).

    Args:
        base: The matrix's first element, in global memory, 64-byte aligned.
        width: The matrix's columns.
        height: The matrix's rows.
        pitch: Elements from one row's start to the next.
        x: The block's first column.
        y: The block's first row.
        value: This work-item's values of the block: `block_elements` of
            them, of the matrix's type.
    """
    _check[_OP_STORE, dtype, block_width, block_height, 1]()
    comptime n = block_elements[block_width, block_height]()
    comptime assert value.dtype == dtype and Int(value.length) == n, String(
        t"an Intel 2D block store of {block_width} x {block_height} takes"
        t" SIMD[{dtype}, {n}] per work-item"
    )
    var g = _geometry(base, width, height, pitch)
    var src = stack_allocation[n, dtype, address_space=AddressSpace.LOCAL]()
    src.unsafe_store(0, rebind[SIMD[dtype, n]](value))
    external_call[_STORE, NoneType](
        Int32(size_of[dtype]()),
        Int32(block_width),
        Int32(block_height),
        Int32(1),
        src,
        base.unsafe_bitcast[UInt8]().unsafe_address_space_cast[
            AddressSpace.GLOBAL
        ](),
        g[0],
        g[1],
        g[2],
        SIMD[DType.int32, 2](Int32(x), Int32(y)),
    )


@inline(.always)
def block_prefetch[
    dtype: DType, //, block_width: Int, block_height: Int, count: Int = 1
](
    base: Pointer[Scalar[dtype], _, address_space=_],
    width: Int,
    height: Int,
    pitch: Int,
    x: Int,
    y: Int,
):
    """Prefetches `count` blocks of `block_width` x `block_height` elements
    at column `x`, row `y` into the cache (no effect on results).

    Parameters:
        dtype: The element type (1, 2 or 4 bytes).
        block_width: Columns per block.
        block_height: Rows per block.
        count: Blocks side by side.

    Args:
        base: The matrix's first element, in global memory, 64-byte aligned.
        width: The matrix's columns.
        height: The matrix's rows.
        pitch: Elements from one row's start to the next.
        x: The block's first column.
        y: The block's first row.
    """
    _check[_OP_PREFETCH, dtype, block_width, block_height, count]()
    var g = _geometry(base, width, height, pitch)
    external_call[_PREFETCH, NoneType](
        Int32(size_of[dtype]()),
        Int32(block_width),
        Int32(block_height),
        Int32(count),
        base.unsafe_bitcast[UInt8]().unsafe_address_space_cast[
            AddressSpace.GLOBAL
        ](),
        g[0],
        g[1],
        g[2],
        SIMD[DType.int32, 2](Int32(x), Int32(y)),
    )
