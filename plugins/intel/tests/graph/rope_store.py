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

"""Qwen3.5-0.8B's full-attention KV-cache store and fused Q/K RoPE
through MAX's graph on the B70, against NumPy.

Run on the GPU host: utils/run-max-py tests/graph/rope_store.py [--time]

The graph is the model's bf16 path (layers/attention.py): store_k_cache_ragged
and store_v_cache_ragged (`mo.kv_cache.store.paged.ragged`) write K and V into
a paged cache, then fused_qk_ragged_rope
(`mo.fused_qk_rope.ragged.paged.with_position_id`) rotates the last 64 of the
256 dims of Q (returned) and of K in the cache, interleaved, with per-token
rows of a frequency table chosen by `position_ids` [1, T] (the model computes
its M-RoPE table rows in the graph; the kernel sees one section). The table
is bf16, as the model casts it to the activations' type, so the rotation is
bf16 arithmetic (MAX's `rope_value`), which the reference models. Shapes: 8 Q
heads, 2 KV heads, head dim 256, bf16, page size 256 (MAX's page for this
model), and batches: decode (one token after 300 cached), prefill 128 and
2048, and a mixed ragged batch. Reference: the page layout from the lookup
table (token p of request b at page lut[b][p // 256], slot p % 256), the
rotation in float32 on the bf16 inputs rounded once; pass: K/V bit-exact where
not rotated, rotated values within 1 bf16 ulp (or 1e-5), every other cache
slot untouched (zero).

--time: per-execution time of the store + rope graph at decode and
prefill 2048, with the bytes moved (Q read and written, K and V read and
written to the cache, K's rotated slice re-read and written) against the
594.6 GB/s DRAM read ceiling.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
from elementwise import DRAM_GBS, compare, from_buf, rnd, to_buf
from max import tree
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops import (
    fused_qk_ragged_rope,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import MHAKVCacheParams

sys.path.insert(
    0,
    os.path.join(
        os.environ.get("MODULAR_DIR")
        # The checkout this file is in: plugins/intel/tests/graph.
        or os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        ),
        "max/tests/integration",
    ),
)
from test_common.simple_kv_cache import (
    block_ids_for_batch,
    paged_kv_cache_inputs,
)

NQ, NKV, HD, RD, PAGE = 8, 2, 256, 64, 256
BF = DType.bfloat16


def kv_params() -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=BF,
        n_kv_heads=NKV,
        head_dim=HD,
        num_layers=1,
        page_size=PAGE,
        devices=[DeviceRef.GPU()],
    )


def build(session, params, T, B, R):
    types = [
        TensorType(BF, [T, NQ, HD], device=DeviceRef.GPU()),
        TensorType(BF, [T, NKV, HD], device=DeviceRef.GPU()),
        TensorType(BF, [T, NKV, HD], device=DeviceRef.GPU()),
        TensorType(DType.uint32, [B + 1], device=DeviceRef.GPU()),
        TensorType(BF, [R, RD], device=DeviceRef.GPU()),
        TensorType(DType.uint32, [1, T], device=DeviceRef.GPU()),
    ]
    with Graph(
        "rope_store", input_types=[*types, *params.flattened_kv_inputs()]
    ) as g:
        q, k, v, offsets, freqs, pos = (x.tensor for x in g.inputs[:6])
        kv = params.unflatten_kv_inputs(iter(g.inputs[6:]))[0]
        layer = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        store_k_cache_ragged(kv, k, offsets, layer)
        store_v_cache_ragged(kv, v, offsets, layer)
        g.output(
            fused_qk_ragged_rope(
                params,
                q,
                offsets,
                kv,
                freqs,
                layer,
                interleaved=True,
                position_ids=pos,
            )
        )
    return session.load(g)


def rope_ref(x, freqs, rows):
    """x [T, H, 256] (bf16 values as float32): rotate dims 192..255 in pairs
    (2i, 2i + 1) by row rows[t] of freqs [R, 64] = (cos, sin) interleaved, as
    MAX's `rope_value` computes it in the table's type (bf16 here, as the
    model casts the table to the activations' type): ComplexSIMD's multiply,
    re = fma(a, c, -round(b s)), im = fma(a, s, round(b c)), each fma rounded
    once to bf16."""
    out = x.copy()
    f = freqs[rows].astype(np.float64)  # [T, 64]
    c, s = f[:, 0::2][:, None, :], f[:, 1::2][:, None, :]
    a = x[:, :, HD - RD :: 2].astype(np.float64)
    b = x[:, :, HD - RD + 1 :: 2].astype(np.float64)
    bs = rnd(b * s, BF).astype(np.float64)
    bc = rnd(b * c, BF).astype(np.float64)
    out[:, :, HD - RD :: 2] = rnd(a * c - bs, BF)
    out[:, :, HD - RD + 1 :: 2] = rnd(a * s + bc, BF)
    return out


class RopeStore:
    def __init__(self):
        self.dev = Accelerator()
        self.session = InferenceSession(devices=[self.dev])
        self.params = kv_params()
        self.rng = np.random.default_rng(4)
        self.fails = 0
        self.models = {}

    def inputs(self, prompt_lens, cache_lens):
        T, B = sum(prompt_lens), len(prompt_lens)
        R = max(c + p for c, p in zip(cache_lens, prompt_lens, strict=False))
        q = rnd(self.rng.standard_normal((T, NQ, HD)), BF)
        k = rnd(self.rng.standard_normal((T, NKV, HD)), BF)
        v = rnd(self.rng.standard_normal((T, NKV, HD)), BF)
        ang = self.rng.uniform(-math.pi, math.pi, (R, RD // 2))
        freqs = np.empty((R, RD), np.float32)
        freqs[:, 0::2], freqs[:, 1::2] = np.cos(ang), np.sin(ang)
        freqs = rnd(freqs.astype(np.float64), BF)
        # Positions: cache length + index, as text tokens get (all M-RoPE
        # rows equal), shuffled per token to test the indirection.
        pos = np.concatenate(
            [
                np.arange(c, c + p)
                for c, p in zip(cache_lens, prompt_lens, strict=False)
            ]
        )
        pos = self.rng.permutation(R)[pos].astype(np.uint32)
        offsets = np.concatenate([[0], np.cumsum(prompt_lens)]).astype(
            np.uint32
        )
        kvin = paged_kv_cache_inputs(
            self.params,
            prompt_lens,
            cache_lengths=cache_lens,
            total_num_pages=sum(
                -(-(c + p) // PAGE)
                for c, p in zip(cache_lens, prompt_lens, strict=False)
            )
            + 1,
        )
        return T, B, R, q, k, v, freqs, pos, offsets, kvin

    def model(self, T, B, R):
        key = (T, B, R)
        if key not in self.models:
            self.models[key] = build(self.session, self.params, T, B, R)
        return self.models[key]

    def args(self, q, k, v, offsets, freqs, pos, kvin):
        return [
            to_buf(q, BF, self.dev),
            to_buf(k, BF, self.dev),
            to_buf(v, BF, self.dev),
            Buffer.from_numpy(offsets).to(self.dev),
            to_buf(freqs, BF, self.dev),
            Buffer.from_numpy(pos[None, :]).to(self.dev),
            *tree.leaves(kvin),
        ]

    def check(self, label, prompt_lens, cache_lens):
        T, B, R, q, k, v, freqs, pos, offsets, kvin = self.inputs(
            prompt_lens, cache_lens
        )
        (qout,) = self.model(T, B, R).execute(
            *self.args(q, k, v, offsets, freqs, pos, kvin)
        )
        got_q = from_buf(qout, BF)
        blocks_bits = (
            kvin.kv_blocks.to(CPU())
            .view(DType.uint16, kvin.kv_blocks.shape)
            .to_numpy()
        )
        blocks = (blocks_bits.astype(np.uint32) << 16).view(
            np.float32
        )  # [pages, 2, 1, PAGE, NKV, HD]
        want_q = rope_ref(q, freqs, pos)
        want_k = rope_ref(k, freqs, pos)
        ok, detail = compare(got_q, want_q, BF)
        # The cache: every token's K (rotated) and V at its page and slot.
        expect = np.zeros_like(blocks)
        lut = block_ids_for_batch(
            [c + p for c, p in zip(cache_lens, prompt_lens, strict=False)], PAGE
        )
        t = 0
        for b, (c, p) in enumerate(zip(cache_lens, prompt_lens, strict=False)):
            for i in range(p):
                position = c + i
                page, slot = lut[b][position // PAGE], position % PAGE
                expect[page, 0, 0, slot] = want_k[t]
                expect[page, 1, 0, slot] = v[t]
                t += 1
        ok_k, detail_k = compare(blocks[:, 0], expect[:, 0], BF)
        ok_v, _detail_v = compare(blocks[:, 1], expect[:, 1], BF)
        v_exact = bool((blocks[:, 1] == expect[:, 1]).all())
        nonrot_k = bool(
            (blocks[:, 0, ..., : HD - RD] == expect[:, 0, ..., : HD - RD]).all()
        )
        good = ok and ok_k and ok_v and v_exact and nonrot_k
        self.fails += 0 if good else 1
        print(
            f"{'PASS' if good else 'FAIL'} {label} (prompt {prompt_lens}, cached {cache_lens}): "
            f"Q {detail}; K in cache {detail_k} (unrotated dims exact: {nonrot_k}); "
            f"V in cache exact: {v_exact}",
            flush=True,
        )

    def time(self, label, prompt_lens, cache_lens):
        T, B, R, q, k, v, freqs, pos, offsets, kvin = self.inputs(
            prompt_lens, cache_lens
        )
        model = self.model(T, B, R)
        args = self.args(q, k, v, offsets, freqs, pos, kvin)
        for _ in range(3):
            model.execute(*args)
        self.dev.synchronize()
        t0, n = time.perf_counter(), 0
        while time.perf_counter() - t0 < 0.02:
            model.execute(*args)
            n += 1
        self.dev.synchronize()
        n = max(10, int(n * 0.1 / (time.perf_counter() - t0)) + 1)
        time.sleep(3)
        for _ in range(n):
            model.execute(*args)
        self.dev.synchronize()
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(n):
                model.execute(*args)
            self.dev.synchronize()
            times.append((time.perf_counter() - t0) / n)
        t = float(np.median(times))
        # Q read + written; K, V read and written to the cache; K's rotated
        # 64 of 256 dims read back and written in the cache; the table rows.
        nbytes = 2 * (
            2 * T * NQ * HD + 4 * T * NKV * HD + 2 * T * NKV * RD + T * RD
        )
        gbs = nbytes / t / 1e9
        print(
            f"TIME store+rope {label} T={T}: {t * 1e6:.2f} us per execution (n={n}), "
            f"{nbytes / 1e6:.2f} MB, {gbs:.1f} GB/s = {gbs / DRAM_GBS:.3f} of {DRAM_GBS}",
            flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    a = ap.parse_args()
    r = RopeStore()
    print(
        "device:",
        r.dev,
        r.dev.api,
        r.dev.architecture_name,
        "kv block shape",
        r.params.shape_per_block,
        flush=True,
    )
    if a.time:
        r.time("decode", [1], [511])
        r.time("prefill", [2048], [0])
        return 0
    r.check("decode", [1], [300])
    r.check("decode at a page edge", [1], [255])
    r.check("prefill 128", [128], [0])
    r.check("prefill 2048", [2048], [0])
    r.check("ragged batch", [33, 1, 300], [0, 511, 5])
    print(f"{'PASS' if r.fails == 0 else 'FAIL'}: {r.fails} failing case(s)")
    return 0 if r.fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
