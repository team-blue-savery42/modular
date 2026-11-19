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

"""Qwen3.5-0.8B's full attention (`mo.mha.ragged.paged`) through MAX's
graph on the B70, against a float64 NumPy attention.

Run on the GPU host: utils/run-max-py tests/graph/attention.py [--time]

The graph is the model's call (layers/attention.py): `flash_attention_ragged`
with the causal mask over a paged KV cache, 8 query heads over 2 KV heads
(GQA 4), head dim 256, bf16, page size 256 (MAX's page for this model), scale
1/16. The cache holds random bf16 K and V on every page (the pages a request
owns come from `block_ids_for_batch`, as `paged_kv_cache_inputs` lays them
out); Q is random. Cases: decode (one query) after 511 and after 255 cached
tokens (page edge), a decode batch, prefill 128, 512 and 2048, and a mixed
ragged batch [33, 1, 300] with [0, 511, 5] cached. Reference: softmax(q k^T /
16) v over keys 0..position of each query, in float64 on the bf16 inputs;
pass: max |gpu - ref| <= 2e-2 (MAX's attention tests: atol 1e-2 against a
bf16 reference; here the outputs are <= 1 in magnitude, so 2e-2 is ~2-3 bf16
ulps).

--time: per-op time through the graph: graphs of 1 and of 9
attention ops chained (each op's output is the next op's Q), executions back
to back; per op = (t9 - t1) / 8, which removes the graph's per-execution
cost; decode (one query after 511 cached) and prefill 512, 2048
(ATTN_TIME_PROMPTS overrides the prefill lengths). ATTN_SHAPE=qwen3-8b uses
Qwen3-8B's heads, ATTN_SHAPE=gemma4-global Gemma-4-26B-A4B's full-attention
layers' (head dim 512, GQA 8).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from elementwise import from_buf, rnd, to_buf
from max import tree
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops import MHAMaskVariant, flash_attention_ragged
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

# ATTN_SHAPE=qwen3-8b: Qwen3-8B's attention, 32 query heads over 8
# KV heads (GQA 4), head dim 128, page 128, scale 1/sqrt(128).
# ATTN_SHAPE=gemma4-global: Gemma-4-26B-A4B's full-attention layers, 16
# query heads over 2 KV heads (GQA 8), head dim 512, page 128.
if os.environ.get("ATTN_SHAPE") == "qwen3-8b":
    NQ, NKV, HD, PAGE = 32, 8, 128, 128
elif os.environ.get("ATTN_SHAPE") == "gemma4-global":
    NQ, NKV, HD, PAGE = 16, 2, 512, 128
else:
    NQ, NKV, HD, PAGE = 8, 2, 256, 256
BF = DType.bfloat16
SCALE = 1.0 / HD**0.5
ATOL = 2e-2


def kv_params() -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=BF,
        n_kv_heads=NKV,
        head_dim=HD,
        num_layers=1,
        page_size=PAGE,
        devices=[DeviceRef.GPU()],
    )


def build(session, params, T, B, chain=1):
    types = [
        TensorType(BF, [T, NQ, HD], device=DeviceRef.GPU()),
        TensorType(DType.uint32, [B + 1], device=DeviceRef.GPU()),
    ]
    with Graph(
        "attention", input_types=[*types, *params.flattened_kv_inputs()]
    ) as g:
        q, offsets = (x.tensor for x in g.inputs[:2])
        kv = params.unflatten_kv_inputs(iter(g.inputs[2:]))[0]
        layer = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        x = q
        for _ in range(chain):
            x = flash_attention_ragged(
                params,
                x,
                offsets,
                kv,
                layer,
                mask_variant=MHAMaskVariant.CAUSAL_MASK,
                scale=SCALE,
            )
        g.output(x)
    return session.load(g)


class Attention:
    def __init__(self):
        self.dev = Accelerator()
        self.session = InferenceSession(devices=[self.dev])
        self.params = kv_params()
        self.rng = np.random.default_rng(6)
        self.fails = 0
        self.models = {}

    def inputs(self, prompt_lens, cache_lens):
        T, B = sum(prompt_lens), len(prompt_lens)
        q = rnd(self.rng.uniform(-1, 1, (T, NQ, HD)), BF)
        offsets = np.concatenate([[0], np.cumsum(prompt_lens)]).astype(
            np.uint32
        )
        pages = sum(
            -(-(c + p) // PAGE)
            for c, p in zip(cache_lens, prompt_lens, strict=False)
        )
        kvin = paged_kv_cache_inputs(
            self.params,
            prompt_lens,
            cache_lengths=cache_lens,
            total_num_pages=pages,
        )
        shape = list(kvin.kv_blocks.shape)  # [pages + 1, 2, 1, PAGE, NKV, HD]
        blocks = rnd(self.rng.uniform(-1, 1, shape), BF)
        kvin.kv_blocks.inplace_copy_from(to_buf(blocks, BF, self.dev))
        return T, B, q, offsets, kvin, blocks

    def model(self, T, B, chain=1):
        key = (T, B, chain)
        if key not in self.models:
            self.models[key] = build(self.session, self.params, T, B, chain)
        return self.models[key]

    def args(self, q, offsets, kvin):
        return [
            to_buf(q, BF, self.dev),
            Buffer.from_numpy(offsets).to(self.dev),
            *tree.leaves(kvin),
        ]

    def check(self, label, prompt_lens, cache_lens):
        T, B, q, offsets, kvin, blocks = self.inputs(prompt_lens, cache_lens)
        (out,) = self.model(T, B).execute(*self.args(q, offsets, kvin))
        got = from_buf(out, BF)
        lut = block_ids_for_batch(
            [c + p for c, p in zip(cache_lens, prompt_lens, strict=False)], PAGE
        )
        want = np.zeros((T, NQ, HD))
        t = 0
        for b, (c, p) in enumerate(zip(cache_lens, prompt_lens, strict=False)):
            n = c + p
            idx = np.arange(n)
            pg = np.asarray(lut[b])[idx // PAGE]
            k = blocks[pg, 0, 0, idx % PAGE].astype(np.float64)  # [n, NKV, HD]
            v = blocks[pg, 1, 0, idx % PAGE].astype(np.float64)
            for i in range(p):
                pos = c + i
                qi = q[t].astype(np.float64)  # [NQ, HD]
                kh = np.repeat(
                    k[: pos + 1], NQ // NKV, axis=1
                )  # [pos+1, NQ, HD]
                vh = np.repeat(v[: pos + 1], NQ // NKV, axis=1)
                s = np.einsum("hd,khd->hk", qi, kh) * SCALE
                s = np.exp(s - s.max(axis=1, keepdims=True))
                s /= s.sum(axis=1, keepdims=True)
                want[t] = np.einsum("hk,khd->hd", s, vh)
                t += 1
        err = float(np.abs(got - want).max())
        good = bool(np.isfinite(got).all()) and err <= ATOL
        self.fails += 0 if good else 1
        print(
            f"{'PASS' if good else 'FAIL'} {label} (prompt {prompt_lens}, cached {cache_lens}): "
            f"max |gpu - ref| {err:.2e} (atol {ATOL})",
            flush=True,
        )

    def time_graph(self, model, args):
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
        return float(np.median(times))

    def time(self, label, prompt_lens, cache_lens):
        T, B, q, offsets, kvin, _ = self.inputs(prompt_lens, cache_lens)
        args = self.args(q, offsets, kvin)
        t1 = self.time_graph(self.model(T, B, 1), args)
        t9 = self.time_graph(self.model(T, B, 9), args)
        per_op = (t9 - t1) / 8
        print(
            f"TIME attention {label} T={T} cached {cache_lens}: graph of 1 op "
            f"{t1 * 1e6:.2f} us, of 9 ops {t9 * 1e6:.2f} us per execution; "
            f"per op {per_op * 1e6:.2f} us",
            flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    a = ap.parse_args()
    r = Attention()
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
        for p in os.environ.get("ATTN_TIME_PROMPTS", "512 2048").split():
            r.time(f"prefill {p}", [int(p)], [0])
        return 0
    r.check("decode", [1], [511])
    r.check("decode at a page edge", [1], [255])
    r.check("decode batch", [1, 1, 1], [300, 0, 1000])
    r.check("prefill 128", [128], [0])
    r.check("prefill 512", [512], [0])
    r.check("prefill 2048", [2048], [0])
    r.check("ragged batch", [33, 1, 300], [0, 511, 5])
    print(f"{'PASS' if r.fails == 0 else 'FAIL'}: {r.fails} failing case(s)")
    return 0 if r.fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
