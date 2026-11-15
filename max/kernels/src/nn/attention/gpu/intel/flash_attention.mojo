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
"""Intel Xe2 flash attention over a paged KV cache: DPAS and 2D block I/O.

`intel_flash_attention` is `flash_attention_dispatch`'s Intel route for a
ragged batch of bfloat16 or float16 queries against a paged KV cache
(`mo.mha.ragged.paged`), any `MHAMask`. Two kernels share one sub-group
routine, `_attend`, which owns a few query rows (DPAS M):

- **Prefill** (`intel_fa_prefill_kernel`): a work-group owns
  `prefill_bq[depth]` queries (64 up to head dim 128, else `BQ`, 32) of one
  sequence and all `group` query heads that share one KV head (GQA), one
  sub-group of 16 work-items per `prefill_rows[depth]` queries (8 up to head
  dim 128, else 4) of one head (32 sub-groups for Qwen3.5's 8 / 2 heads and
  for Qwen3-8B's 32 / 8). The sub-groups of a work-group read the same K and
  V tiles, so all but the first find them in the cache. No shared local
  memory, no barriers.
- **Decode** (`intel_fa_decode_kernel` + `intel_fa_decode_stitch`): one
  sub-group per (split of `decode_split_keys` keys (32 or more), KV head,
  sequence), its rows the `group` query heads of the KV head (rows past
  `group` read zeros);
  each writes its unnormalised output, row max and row sum, and the stitch
  kernel combines the splits (log-sum-exp), as `naive_fa_decode_apple`'s
  stitch does.

`_attend` streams the keys in steps of `BN` (32): S = Q K^T as DPAS tiles
(A: 16 head dims of the rows from `block_load`; B: the packed K^T from
`block_load_transpose` of K read as 32-bit pairs, one key per lane), scale
(with log2 e folded in) and mask, online softmax in base 2 (each lane keeps
partial row sums, added up once at the end; the running row maxima, the same
in every lane, are only raised when a tile exceeds them by more than `LAZY`
bits, the one time `lane_group_max` runs and the sums and O are rescaled),
then O += P V (A: the score tile itself, cast to the input type --
DPAS's accumulator layout is its A layout; B: V's rows by
`block_load_transform`). The page lookups of a step are made once; the
step's V rows and the next step's K rows are prefetched (`block_prefetch`)
at its start (in prefill the sub-groups of a work-group, which read the same
tiles, split the prefetches: sub-group n issues blocks n, n + 16, ...).
O (`ROWS` x depth, float32) stays in registers: 64 floats per work-item at
depth 256. With 8 rows (128 floats) IGC spilled 4-12 KB per
thread and both kernels were 1.5-2x slower (measured on the B70).

The KV operand must be a paged cache whose pages hold rows of
`kv_num_heads * depth` elements (`PagedKVCache`'s layout): a key's row for a
head is `block_paged_ptr`'s pointer, the next key's `kv_num_heads * depth`
elements further on within a page, so a step of `BN` keys never crosses a
page when `page_size % BN == 0` (checked by `intel_fa_supported`). Rows past
the cache's length read as zeros (2D block loads clip at the given height).
Needs sub-group size 16 (DPAS and block I/O assert it).

Tuning defines (defaults measured best on the B70): `intel_fa_rows` (4),
`intel_fa_bn` (32), `intel_fa_bq` (32), `intel_fa_split` (32),
`intel_fa_fill` (256).
"""

from std.collections import OptionalReg
from std.math import ceildiv, exp2
from std.math.constants import log2e
from std.memory import stack_allocation
from std.memory.address_space import AddressSpace
from std.utils.numerics import get_accum_type
from std.sys.defines import get_defined_int

from layout import TileTensor, row_major
from max.gpu import WARP_SIZE, block_idx, grid_dim, lane_id, thread_idx
from max.gpu.compute.mma import mma
from max.gpu.host import DeviceContext
from max.gpu.memory.block_io_intel import (
    block_load,
    block_load_transform,
    block_load_transpose,
    block_prefetch,
    block_store,
)
from max.gpu.primitives.warp import lane_group_max, lane_group_sum, vote

from nn.attention.mha_mask import CausalMask, MHAMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.common import ImmutTileTensor1D

comptime ROWS = get_defined_int["intel_fa_rows", 4]()
"""Query rows of a sub-group (DPAS M)."""
comptime BN = get_defined_int["intel_fa_bn", 32]()
"""Keys per step."""
comptime BQ = get_defined_int["intel_fa_bq", 32]()
"""Queries of one head per prefill work-group."""


def prefill_rows[depth: Int]() -> Int:
    """Query rows per prefill sub-group (DPAS M): 8 up to head dim 128 (twice
    the K and V reuse of 4), 4 above (8 spilled registers at 256).
    `intel_fa_prefill_rows` overrides it."""
    return get_defined_int["intel_fa_prefill_rows", 8 if depth <= 128 else 4]()


def prefill_bq[depth: Int]() -> Int:
    """Queries of one head per prefill work-group: 64 up to head dim 128 (8%
    faster at 4,096 tokens than 32 with 8-row sub-groups), else
    `BQ` (the work-group would pass 1,024 work-items)."""
    return BQ if depth > 128 else get_defined_int["intel_fa_prefill_bq", 64]()


comptime DECODE_SPLIT = get_defined_int["intel_fa_split", 32]()
"""Fewest keys per decode sub-group (a multiple of `BN`)."""
comptime DECODE_FILL = get_defined_int["intel_fa_fill", 256]()
"""Decode sub-groups wanted in all: `decode_max_splits` gives each (sequence,
KV head) pair `DECODE_FILL` / (sequences x KV heads) of them, at least one."""


def decode_max_splits(keys: Int, batch_size: Int, kv_heads: Int) -> Int:
    """Splits a decode launch has per (sequence, KV head) pair: at most
    `DECODE_FILL` / (sequences x KV heads) (at least one), and no more than
    `DECODE_SPLIT` keys each allows for `keys`. It is the launch's grid size
    (a device graph captured at a longer context keeps it), so the keys of a
    split are worked out on the device from each sequence's own length:
    `decode_split_keys`."""
    return min(
        ceildiv(keys, DECODE_SPLIT),
        max(1, DECODE_FILL // (batch_size * kv_heads)),
    )


def decode_split_keys(num_keys: Int, max_splits: Int) -> Int:
    """Keys per decode sub-group for a sequence of `num_keys` keys in a launch
    of `max_splits` splits: `DECODE_SPLIT`, or more (a multiple of `BN`) when
    the sequence would need more splits than the launch has.

    More, smaller splits only pay while there are fewer sub-groups than the
    GPU can run: each split adds a partial output the stitch kernel reads
    one after another, and adjacent small splits read the cache from many
    places at once. On the B70 at Qwen3-8B's shape (8 KV heads) a 24,576-key
    sequence took 601 us in 32-key splits and 194 us in about 32 splits, and
    64 sequences of 600 keys 572 us and 379 us."""
    return max(DECODE_SPLIT, ceildiv(ceildiv(num_keys, max_splits), BN) * BN)


comptime NEG_INF = Float32(-3.0e38)
comptime LAZY = Float32(8.0)
"""A row's running maximum is raised when a score exceeds it by more than this
(log2 units), so that P = 2^(S - m) stays below 2^8."""


def intel_fa_supported[
    dtype: DType, depth: Int, group: Int, page_size: Int, kv_dtype: DType
]() -> Bool:
    """Whether `intel_flash_attention` takes these attention shapes."""
    return (
        dtype in (DType.bfloat16, DType.float16)
        and kv_dtype == dtype
        and depth % 32 == 0
        and depth >= 32
        and depth <= 256
        and group >= 1
        and group <= ROWS
        and page_size > 0
        and page_size % BN == 0
    )


@inline(.always)
def _load_v[
    depth: Int, kv_pitch: Int, dtype: DType, j: Int
](addr: UInt64, rows: Int) -> SIMD[dtype, 32]:
    """V's fragment j of 16 keys: lane n holds head dims 32 j + n and
    32 j + 16 + n, VNNI-packed: two B operands."""
    var vp = MutPointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=Int(addr)
    )
    return rebind[SIMD[dtype, 32]](
        block_load_transform[16, 16, 2](vp, depth, rows, kv_pitch, 32 * j, 0)
    )


@inline(.always)
def _attend[
    dtype: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    //,
    depth: Int,
    kv_pitch: Int,
    QR: Int = ROWS,
    COOP: Bool = False,
    PIPE: Bool = False,
](
    o: MutPointer[Scalar[DType.float32], _, address_space=AddressSpace.LOCAL],
    mut m: SIMD[DType.float32, QR],
    mut l: SIMD[DType.float32, QR],
    q_base: MutPointer[Scalar[dtype], MutAnyOrigin],
    q_rows: Int,
    q_pitch: Int,
    k: k_t,
    v: v_t,
    mask: mask_t,
    batch: Int,
    kv_head: Int,
    head0: Int,
    head_step: Int,
    pos0: Int,
    pos_step: Int,
    status_row: Int,
    status_rows: Int,
    key_begin: Int,
    key_end: Int,
    num_keys: Int,
    scale: Float32,
    pf_id: Int = 0,
):
    """Attends this sub-group's `QR` rows to keys [`key_begin`, `key_end`)
    (of `num_keys` in the sequence): updates the running output `o` (depth /
    16 DPAS tiles of `SIMD[float32, QR]`, element r of lane n =
    O[r, 16 t + n]), the row maxima `m` (scaled scores, the same in every
    lane, at most `LAZY` bits below a score's) and this lane's partial row
    sums `l`. Row r's query is row r of the matrix at `q_base` (`q_rows`
    rows, `q_pitch` apart; zeros past them), its head `head0 + head_step * r`
    and its position (score row) `pos0 + pos_step * r`;
    `status_row`/`status_rows` give the rows' span for the mask's tile
    status. With `COOP` the sub-group
    `pf_id` of its work-group issues only the prefetches whose block number
    is `pf_id` modulo the number of blocks of a step (the others' share). With
    `PIPE` the loads of a step are issued ahead of their DPAS (prefill; the
    decode kernel's 4 rows ran slower so)."""
    comptime KS = depth // 16  # DPAS K steps over the head dimension
    comptime NT = BN // 16  # 16-key column tiles per step
    comptime KK = KS // 2  # steps of 32 head dims
    comptime causal = mask_t == CausalMask
    var lane = Int(lane_id())
    var kv0 = key_begin
    # m is kept in log2 units inside the loop.
    m = max(m * log2e, SIMD[DType.float32, QR](NEG_INF))
    while kv0 < key_end:
        var status = mask.status(
            UInt32(batch), (status_row, kv0), (status_rows, BN)
        )
        if status == TileMaskStatus.FULL_MASK:
            comptime if causal:
                break
            else:
                kv0 += BN
                continue
        # This step's K and V rows: one lookup per 16 keys (a paged cache
        # reads its page table), hoisted out of the loops below.
        var kaddr = SIMD[DType.uint64, NT]()
        var vaddr = SIMD[DType.uint64, NT]()
        comptime for t in range(NT):
            var key0 = min(kv0 + 16 * t, num_keys - 1)
            kaddr[t] = UInt64(
                Int(
                    k.block_paged_ptr[16](
                        UInt32(batch), UInt32(key0), UInt32(kv_head)
                    )
                )
            )
            vaddr[t] = UInt64(
                Int(
                    v.block_paged_ptr[16](
                        UInt32(batch), UInt32(key0), UInt32(kv_head)
                    )
                )
            )
        # Prefetch this step's V rows (read after the softmax) and the next
        # step's K rows into the cache (10-13% faster prefill, measured).
        # With COOP the sub-groups of a work-group share the work: each takes
        # the blocks whose index is its own, modulo the number of blocks.
        comptime NB = 2 * NT * (depth // 32)
        comptime for t in range(NT):
            var key0 = kv0 + 16 * t
            if key0 < num_keys:
                var vp = MutPointer[Scalar[dtype], MutAnyOrigin](
                    unsafe_from_address=Int(vaddr[t])
                )
                var rows = min(16, num_keys - key0)
                comptime for c in range(depth // 32):
                    comptime bi = t * (depth // 32) + c
                    if not COOP or pf_id % NB == bi:
                        block_prefetch[16, 16, 2](
                            vp, depth, rows, kv_pitch, 32 * c, 0
                        )
            var nkey0 = kv0 + BN + 16 * t
            if nkey0 < key_end:
                var kp = k.block_paged_ptr[16](
                    UInt32(batch), UInt32(nkey0), UInt32(kv_head)
                )
                var rows = min(16, num_keys - nkey0)
                comptime for c in range(depth // 32):
                    comptime bi = NT * (depth // 32) + t * (depth // 32) + c
                    if not COOP or pf_id % NB == bi:
                        block_prefetch[16, 16, 2](
                            kp, depth, rows, kv_pitch, 32 * c, 0
                        )
        # S = Q K^T: NT tiles of QR rows x 16 keys.
        var s = stack_allocation[
            NT * QR, DType.float32, address_space=AddressSpace.LOCAL
        ]()
        comptime for t in range(NT):
            s.unsafe_store(QR * t, SIMD[DType.float32, QR](0))
        comptime NU = NT * KK
        var vs = stack_allocation[
            2 * 32, dtype, address_space=AddressSpace.LOCAL
        ]()
        comptime if PIPE:
            # S = Q K^T, NT tiles of QR rows x 16 keys, the head dimension in
            # steps of 32 (two A fragments of the rows, and per tile two packed
            # B fragments of K^T), unrolled so that the loads of a step are
            # in flight together (a loop paid each step's load latency in turn).
            var qs = stack_allocation[
                2 * 2 * QR, dtype, address_space=AddressSpace.LOCAL
            ]()
            var ks = stack_allocation[
                2 * NT * 2 * 8, DType.uint32, address_space=AddressSpace.LOCAL
            ]()
            comptime for kk in range(KK):
                comptime par = kk % 2
                qs.unsafe_store(
                    par * 2 * QR,
                    rebind[SIMD[dtype, 2 * QR]](
                        block_load[16, QR, 2](
                            q_base, depth, q_rows, q_pitch, 32 * kk, 0
                        )
                    ),
                )
                comptime for t in range(NT):
                    var key0 = kv0 + 16 * t
                    var kp = MutPointer[Scalar[DType.uint32], MutAnyOrigin](
                        unsafe_from_address=Int(kaddr[t])
                    )
                    var rows = max(1, min(16, num_keys - key0))
                    comptime for h in range(2):
                        # Lane l: key key0 + l, 8 words of head-dim pairs: the
                        # packed B of K^T. (Keys past the end give scores that
                        # the mask below replaces.)
                        ks.unsafe_store(
                            ((par * NT + t) * 2 + h) * 8,
                            rebind[SIMD[DType.uint32, 8]](
                                block_load_transpose[8, 16](
                                    kp,
                                    depth // 2,
                                    rows,
                                    kv_pitch // 2,
                                    16 * kk + 8 * h,
                                    0,
                                )
                            ),
                        )
                comptime for t in range(NT):
                    comptime for h in range(2):
                        var kb = ks.unsafe_load[width=8](
                            ((par * NT + t) * 2 + h) * 8
                        )
                        var qa = qs.unsafe_load[width=2 * QR](par * 2 * QR)
                        var d = SIMD[DType.float32, QR]()
                        mma(
                            d,
                            qa.slice[QR, offset=QR * h](),
                            kb,
                            s.unsafe_load[width=QR](QR * t),
                        )
                        s.unsafe_store(QR * t, d)
            # V's first fragment: its load overlaps the softmax.
            vs.unsafe_store(
                0,
                _load_v[depth, kv_pitch, dtype, 0](
                    vaddr[0], max(1, min(16, num_keys - kv0))
                ),
            )
        else:
            # The head dimension in steps of 32 (a loop: unrolled, it was no
            # faster).
            for kk in range(KS // 2):
                # 32 head dims of the rows: two A fragments.
                var qa = rebind[SIMD[dtype, 2 * QR]](
                    block_load[16, QR, 2](
                        q_base, depth, q_rows, q_pitch, 32 * kk, 0
                    )
                )
                comptime for t in range(NT):
                    var key0 = kv0 + 16 * t
                    if key0 < num_keys:
                        var kp = MutPointer[Scalar[DType.uint32], MutAnyOrigin](
                            unsafe_from_address=Int(kaddr[t])
                        )
                        var rows = min(16, num_keys - key0)
                        comptime for h in range(2):
                            # Lane l: key key0 + l, 8 words of head-dim pairs:
                            # the packed B of K^T.
                            var kb = rebind[SIMD[DType.uint32, 8]](
                                block_load_transpose[8, 16](
                                    kp,
                                    depth // 2,
                                    rows,
                                    kv_pitch // 2,
                                    16 * kk + 8 * h,
                                    0,
                                )
                            )
                            var d = SIMD[DType.float32, QR]()
                            mma(
                                d,
                                qa.slice[QR, offset=QR * h](),
                                kb,
                                s.unsafe_load[width=QR](QR * t),
                            )
                            s.unsafe_store(QR * t, d)
        # Scores in log2 units: S * scale * log2(e), so that P = exp2(S - m).
        var masked = status != TileMaskStatus.NO_MASK or kv0 + BN > num_keys
        var tile_max = SIMD[DType.float32, QR](NEG_INF)
        comptime for t in range(NT):
            var st = s.unsafe_load[width=QR](QR * t)
            if masked:
                var key = kv0 + 16 * t + lane
                st = st * scale
                comptime for r in range(QR):
                    var x = mask.mask(
                        (
                            batch,
                            head0 + head_step * r,
                            pos0 + pos_step * r,
                            key,
                        ),
                        st[r],
                    )
                    st[r] = x if key < num_keys else NEG_INF
                st = st.gt(NEG_INF * 0.5).select(
                    st * log2e, SIMD[DType.float32, QR](NEG_INF)
                )
            else:
                st = st * (scale * log2e)
            s.unsafe_store(QR * t, st)
            tile_max = max(tile_max, st)
        # The running maximum only has to bound the scores: it is raised (and
        # the sums rescaled) when some row's tile exceeds it by more than
        # LAZY, so P stays below 2^LAZY, and the cross-lane maximum is only
        # taken then.
        var raise_m = vote[DType.uint32](tile_max.gt(m + LAZY).reduce_or())
        if raise_m != 0:
            var m_new = max(m, lane_group_max[16](tile_max))
            var alpha = exp2(m - m_new)
            m = m_new
            l *= alpha
            comptime for j in range(KS):
                o.unsafe_store(QR * j, o.unsafe_load[width=QR](QR * j) * alpha)
        comptime if PIPE:
            var ps = stack_allocation[
                NT * QR, dtype, address_space=AddressSpace.LOCAL
            ]()
            comptime for t in range(NT):
                var st = s.unsafe_load[width=QR](QR * t)
                var p = exp2(st - m)
                if masked:
                    p = st.gt(NEG_INF * 0.5).select(
                        p, SIMD[DType.float32, QR](0)
                    )
                l += p
                ps.unsafe_store(QR * t, p.cast[dtype]())
            # O += P V, V's fragment u + 1 loaded before fragment u's DPAS. (A
            # tile past the keys has P = 0 and V rows that are zero or a real
            # key's, so no tile is skipped.)
            comptime for u in range(NU):
                comptime par = u % 2
                comptime if u + 1 < NU:
                    vs.unsafe_store(
                        (1 - par) * 32,
                        _load_v[depth, kv_pitch, dtype, (u + 1) % KK](
                            vaddr[(u + 1) // KK],
                            max(
                                1,
                                min(16, num_keys - kv0 - 16 * ((u + 1) // KK)),
                            ),
                        ),
                    )
                comptime t = u // KK
                comptime j = u % KK
                var vb = vs.unsafe_load[width=32](par * 32)
                var pa = ps.unsafe_load[width=QR](QR * t)
                comptime for h in range(2):
                    comptime oi = QR * (2 * j + h)
                    var d = SIMD[DType.float32, QR]()
                    mma(
                        d,
                        pa,
                        vb.slice[16, offset=16 * h](),
                        o.unsafe_load[width=QR](oi),
                    )
                    o.unsafe_store(oi, d)
        else:
            comptime for t in range(NT):
                var key0 = kv0 + 16 * t
                var st = s.unsafe_load[width=QR](QR * t)
                var p = exp2(st - m)
                if masked:
                    p = st.gt(NEG_INF * 0.5).select(
                        p, SIMD[DType.float32, QR](0)
                    )
                l += p
                if key0 < num_keys:
                    var pa = p.cast[dtype]()
                    var vp = MutPointer[Scalar[dtype], MutAnyOrigin](
                        unsafe_from_address=Int(vaddr[t])
                    )
                    var rows = min(16, num_keys - key0)
                    comptime for j in range(KS // 2):
                        # Lane n: head dims 32 j + n and 32 j + 16 + n of the
                        # 16 keys, VNNI-packed: two B operands.
                        var vb = rebind[SIMD[dtype, 32]](
                            block_load_transform[16, 16, 2](
                                vp, depth, rows, kv_pitch, 32 * j, 0
                            )
                        )
                        comptime for h in range(2):
                            comptime oi = QR * (2 * j + h)
                            var d = SIMD[DType.float32, QR]()
                            mma(
                                d,
                                pa,
                                vb.slice[16, offset=16 * h](),
                                o.unsafe_load[width=QR](oi),
                            )
                            o.unsafe_store(oi, d)
        kv0 += BN
    m = m * Float32(0.6931471805599453)


@inline(.always)
def _seq_bounds[
    k_t: MHAOperand, //, _is_cache_length_accurate: Bool
](
    k: k_t,
    row_offsets: MutPointer[Scalar[DType.uint32], MutAnyOrigin],
    batch: Int,
) -> Tuple[Int, Int, Int]:
    """(first ragged row, query length, keys attended) of sequence `batch`,
    as `mha_gpu_naive`'s ragged path computes them."""
    var seq_start = Int(row_offsets[batch])
    var q_len = Int(row_offsets[batch + 1]) - seq_start
    var num_keys: Int
    comptime if _is_cache_length_accurate:
        num_keys = q_len
    else:
        num_keys = k.cache_length(batch) + q_len
    return (seq_start, q_len, num_keys)


def intel_fa_prefill_kernel[
    dtype: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    depth: Int,
    num_heads: Int,
    group: Int,
    _is_cache_length_accurate: Bool,
](
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
    q: MutPointer[Scalar[dtype], MutAnyOrigin],
    k: k_t,
    v: v_t,
    mask: mask_t,
    row_offsets: MutPointer[Scalar[DType.uint32], MutAnyOrigin],
    scale: Float32,
):
    """Prefill: grid (ceildiv(max queries, prefill_bq), KV heads, sequences),
    `prefill_bq / prefill_rows[depth]() * group` sub-groups of 16."""
    comptime kv_heads = num_heads // group
    comptime kv_pitch = kv_heads * depth
    comptime KS = depth // 16
    comptime PR = prefill_rows[depth]()
    var sg = Int(thread_idx.x) // WARP_SIZE
    var batch = Int(block_idx.z)
    var kv_head = Int(block_idx.y)
    var head = kv_head * group + sg % group
    var q_block = Int(block_idx.x)
    comptime if mask_t == CausalMask:
        # The last query blocks attend the most keys: launch them first.
        q_block = Int(grid_dim.x) - 1 - q_block
    var q0 = q_block * prefill_bq[depth]() + PR * (sg // group)
    var b = _seq_bounds[_is_cache_length_accurate](k, row_offsets, batch)
    var seq_start = b[0]
    var q_len = b[1]
    var num_keys = b[2]
    if q0 >= q_len:
        return
    var pos0 = num_keys - q_len + q0
    var key_end = num_keys
    comptime if mask_t == CausalMask:
        key_end = min(num_keys, pos0 + PR)
    var q_base = q + ((seq_start + q0) * num_heads + head) * depth
    var o_base = output + ((seq_start + q0) * num_heads + head) * depth
    var rows = min(PR, q_len - q0)

    var o = stack_allocation[
        KS * PR, DType.float32, address_space=AddressSpace.LOCAL
    ]()
    comptime for j in range(KS):
        o.unsafe_store(PR * j, SIMD[DType.float32, PR](0))
    var m = SIMD[DType.float32, PR](NEG_INF)
    var l = SIMD[DType.float32, PR](0)
    _attend[depth, kv_pitch, PR, True, True](
        o,
        m,
        l,
        q_base,
        rows,
        num_heads * depth,
        k,
        v,
        mask,
        batch,
        kv_head,
        head,
        0,
        pos0,
        1,
        pos0,
        PR,
        0,
        key_end,
        num_keys,
        scale,
        sg,
    )
    var inv = 1.0 / lane_group_sum[16](l)
    comptime for j in range(KS):
        var out = (o.unsafe_load[width=PR](PR * j) * inv).cast[dtype]()
        block_store[16, PR](
            o_base, depth, rows, num_heads * depth, 16 * j, 0, out
        )


def intel_fa_decode_kernel[
    dtype: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    depth: Int,
    num_heads: Int,
    group: Int,
    _is_cache_length_accurate: Bool,
](
    o_part: MutPointer[Scalar[DType.float32], MutAnyOrigin],
    ml_part: MutPointer[Scalar[DType.float32], MutAnyOrigin],
    q: MutPointer[Scalar[dtype], MutAnyOrigin],
    k: k_t,
    v: v_t,
    mask: mask_t,
    row_offsets: MutPointer[Scalar[DType.uint32], MutAnyOrigin],
    scale: Float32,
    num_splits: Int32,
):
    """Decode (one query per sequence): grid (splits, KV heads, sequences),
    one sub-group of `decode_split_keys` keys. Writes split `s`'s
    unnormalised output of head h to
    `o_part[((batch * num_heads + h) * splits + s) * depth + d]` and its row
    max and lane-summed row sum to `ml_part[2 * (...) + 0/1]`."""
    comptime kv_heads = num_heads // group
    comptime kv_pitch = kv_heads * depth
    comptime KS = depth // 16
    var split = Int(block_idx.x)
    var kv_head = Int(block_idx.y)
    var batch = Int(block_idx.z)
    var splits = Int(num_splits)
    var b = _seq_bounds[_is_cache_length_accurate](k, row_offsets, batch)
    var seq_start = b[0]
    var q_len = b[1]
    var num_keys = b[2]
    var split_keys = decode_split_keys(num_keys, splits)
    var key_begin = split * split_keys
    if key_begin >= num_keys or q_len == 0:
        return
    var key_end = min(num_keys, key_begin + split_keys)
    var pos = num_keys - 1
    var q_base = q + (seq_start * num_heads + kv_head * group) * depth

    var o = stack_allocation[
        KS * ROWS, DType.float32, address_space=AddressSpace.LOCAL
    ]()
    comptime for j in range(KS):
        o.unsafe_store(ROWS * j, SIMD[DType.float32, ROWS](0))
    var m = SIMD[DType.float32, ROWS](NEG_INF)
    var l = SIMD[DType.float32, ROWS](0)
    _attend[depth, kv_pitch](
        o,
        m,
        l,
        q_base,
        group,
        depth,
        k,
        v,
        mask,
        batch,
        kv_head,
        kv_head * group,
        1,
        pos,
        0,
        pos,
        1,
        key_begin,
        key_end,
        num_keys,
        scale,
    )
    var lsum = lane_group_sum[16](l)
    var lane = Int(lane_id())
    comptime for r in range(group):
        var idx = (batch * num_heads + kv_head * group + r) * splits + split
        comptime for j in range(KS):
            o_part[idx * depth + 16 * j + lane] = o.unsafe_load[width=ROWS](
                ROWS * j
            )[r]
        if lane == 0:
            ml_part[2 * idx] = m[r]
            ml_part[2 * idx + 1] = lsum[r]


def intel_fa_decode_stitch[
    dtype: DType,
    k_t: MHAOperand,
    depth: Int,
    num_heads: Int,
    _is_cache_length_accurate: Bool,
](
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
    o_part: MutPointer[Scalar[DType.float32], MutAnyOrigin],
    ml_part: MutPointer[Scalar[DType.float32], MutAnyOrigin],
    k: k_t,
    row_offsets: MutPointer[Scalar[DType.uint32], MutAnyOrigin],
    num_splits: Int32,
):
    """Combines the decode splits: grid (heads, sequences), `depth`
    work-items."""
    var head = Int(block_idx.x)
    var batch = Int(block_idx.y)
    var d = Int(thread_idx.x)
    var splits = Int(num_splits)
    var b = _seq_bounds[_is_cache_length_accurate](k, row_offsets, batch)
    var seq_start = b[0]
    var q_len = b[1]
    var num_keys = b[2]
    if q_len == 0:
        return
    var active = ceildiv(num_keys, decode_split_keys(num_keys, splits))
    var base = (batch * num_heads + head) * splits
    # Eight splits per iteration, so that their loads are in flight together.
    comptime U = 8
    var mv = SIMD[DType.float32, U](NEG_INF)
    for s0 in range(0, active, U):
        comptime for i in range(U):
            if s0 + i < active:
                mv[i] = max(mv[i], ml_part[2 * (base + s0 + i)])
    var m = mv.reduce_max()
    var lv = SIMD[DType.float32, U](0)
    var av = SIMD[DType.float32, U](0)
    for s0 in range(0, active, U):
        comptime for i in range(U):
            if s0 + i < active:
                var idx = base + s0 + i
                var w = exp2((ml_part[2 * idx] - m) * log2e)
                lv[i] += w * ml_part[2 * idx + 1]
                av[i] += w * o_part[idx * depth + d]
    var l = lv.reduce_add()
    var acc = av.reduce_add()
    output[(seq_start * num_heads + head) * depth + d] = (acc / l).cast[dtype]()


@inline(.always)
def _raw[
    dtype: DType
](t: TileTensor[dtype, ...]) -> MutPointer[Scalar[dtype], MutAnyOrigin]:
    return MutPointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=Int(t.unsafe_ptr())
    )


def intel_flash_attention[
    dtype: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    //,
    depth: Int,
    num_heads: Int,
    group: Int,
    _is_cache_length_accurate: Bool = False,
](
    q: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    k: k_t,
    v: v_t,
    mask: mask_t,
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    row_offsets: ImmutTileTensor1D[.uint32],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    max_cache_size: Int,
    is_token_generation: Bool,
    ctx: DeviceContext,
) raises:
    """Launches Intel flash attention for a ragged batch (see the module):
    the decode pair when every sequence has one query (`is_token_generation`),
    else the prefill kernel. `max_cache_size` bounds the keys of any
    sequence; the caller checks `intel_fa_supported`."""
    comptime assert num_heads % group == 0
    comptime kv_heads = num_heads // group
    if batch_size == 0 or max_prompt_len == 0 or max_cache_size == 0:
        return
    var q_ptr = _raw(q)
    comptime assert output.dtype == dtype, "the output must be of Q's type"
    var o_ptr = MutPointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=Int(output.unsafe_ptr())
    )
    var offs = MutPointer[Scalar[DType.uint32], MutAnyOrigin](
        unsafe_from_address=Int(row_offsets.unsafe_ptr())
    )
    if is_token_generation:
        # A cache length that isn't accurate excludes the new token.
        var keys = max_cache_size + (
            0 if _is_cache_length_accurate else max_prompt_len
        )
        var splits = decode_max_splits(keys, batch_size, kv_heads)
        # One buffer: the partial outputs, then the row maxima and sums.
        var o_n = batch_size * num_heads * splits * depth
        var part = ctx.enqueue_create_buffer[DType.float32](
            o_n + 2 * batch_size * num_heads * splits
        )
        var o_part_ptr = MutPointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(part.unsafe_ptr())
        )
        var ml_part_ptr = o_part_ptr + o_n
        comptime decode = intel_fa_decode_kernel[
            dtype,
            k_t,
            v_t,
            mask_t,
            depth,
            num_heads,
            group,
            _is_cache_length_accurate,
        ]
        ctx.enqueue_function[decode](
            o_part_ptr,
            ml_part_ptr,
            q_ptr,
            k,
            v,
            mask,
            offs,
            scale,
            Int32(splits),
            grid_dim=(splits, kv_heads, batch_size),
            block_dim=WARP_SIZE,
        )
        comptime stitch = intel_fa_decode_stitch[
            dtype, k_t, depth, num_heads, _is_cache_length_accurate
        ]
        ctx.enqueue_function[stitch](
            o_ptr,
            o_part_ptr,
            ml_part_ptr,
            k,
            offs,
            Int32(splits),
            grid_dim=(num_heads, batch_size),
            block_dim=depth,
        )
        _ = part^
    else:
        comptime prefill = intel_fa_prefill_kernel[
            dtype,
            k_t,
            v_t,
            mask_t,
            depth,
            num_heads,
            group,
            _is_cache_length_accurate,
        ]
        ctx.enqueue_function[prefill](
            o_ptr,
            q_ptr,
            k,
            v,
            mask,
            offs,
            scale,
            grid_dim=(
                ceildiv(max_prompt_len, prefill_bq[depth]()),
                kv_heads,
                batch_size,
            ),
            block_dim=prefill_bq[depth]()
            // prefill_rows[depth]()
            * group
            * WARP_SIZE,
        )
