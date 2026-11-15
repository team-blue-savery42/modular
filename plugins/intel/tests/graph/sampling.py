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

"""Qwen3.5-0.8B's embedding, logits and sampling ops through MAX's
graph on the B70, against NumPy.

Run on the GPU host: utils/run-max-py tests/graph/sampling.py [--only NAME] [--time]

Checks (the model's top-level subgraph and its sampler graph):
- embedding: ops.gather of rows of the tied bfloat16 table [248320, 1024]
  for T = 1, 7 (including the first and last rows) and 512 token ids; exact.
- logits: ops.matmul(h, transpose(W)) with h bfloat16 [M, 1024] and W the
  same table (B^T, MAX's skinny kernel for M <= 16, the Intel GEMM above),
  M = 1, 16, 128; against float32 NumPy within 1 bfloat16 ulp (or 1e-5).
- greedy: ops.cast to float32 then the sampler's fused top-k op
  (`sampler.fused_token_sampling`, top_k a per-row tensor, max_k 1 on the
  CPU, as MAX's sampler graph passes them) over [B, 248320], B = 1, 4: the
  sampled id holds the row's maximum.
- top-k: the same op with k = 20, temperature 1, per-row seeds: 64 rows of
  random logits each sample one of their 20 largest; then 4,096 rows of one
  row of logits with a known top-20 distribution: every frequency within
  5 standard deviations of its probability.
- scatter_nd: `mo.scatter_nd.skip_oob_indices` (the image-token merge) of 16
  rows into [64, 1024] bfloat16, with indices -1 (the last row) and 64, 1000
  (out of bounds: skipped); exact.

--time: per-execution wall time of one-op graphs for the decode shapes;
run under MOJO_INTEL_PROFILE for the kernels' device times.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from elementwise import DRAM_GBS, Runner, compare, from_buf, rnd, to_buf
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops import scatter_nd_skip_oob_indices, topk_fused_sampling

VOCAB, HIDDEN = 248320, 1024
BF = DType.bfloat16


class SamplingRunner(Runner):
    def __init__(self):
        super().__init__()
        self._table = None

    def table(self):
        """The tied embedding table, bfloat16 values N(0, 0.02) as float32,
        and its device buffer."""
        if self._table is None:
            w = rnd(
                self.rng.standard_normal((VOCAB, HIDDEN), dtype=np.float32)
                * 0.02,
                BF,
            )
            self._table = (w, to_buf(w, BF, self.dev))
        return self._table

    def load_typed(self, name, types, body):
        with Graph(name, input_types=types) as g:
            outs = body(*(v.tensor for v in g.inputs))
            g.output(*(outs if isinstance(outs, tuple) else (outs,)))
        return self.session.load(g)

    def ids_buf(self, ids):
        return Buffer.from_numpy(np.ascontiguousarray(ids, dtype=np.int64)).to(
            self.dev
        )

    def check_gather(self, T):
        w, wbuf = self.table()
        ids = self.rng.integers(0, VOCAB, T)
        if T >= 2:
            ids[0], ids[-1] = 0, VOCAB - 1
        model = self.load_typed(
            f"gather_{T}",
            [
                TensorType(BF, (VOCAB, HIDDEN), device=self.gpu),
                TensorType(DType.int64, (T,), device=self.gpu),
            ],
            lambda t, i: ops.gather(t, i, axis=0),
        )
        (out,) = model.execute(wbuf, self.ids_buf(ids))
        got = from_buf(out, BF)
        bad = int((got != w[ids]).sum())
        self.report(
            f"gather bf16 [{VOCAB}, {HIDDEN}] x {T} ids",
            bad == 0,
            f"{bad} of {got.size} wrong",
        )

    def logits_model(self, M):
        return self.load_typed(
            f"logits_{M}",
            [
                TensorType(BF, (M, HIDDEN), device=self.gpu),
                TensorType(BF, (VOCAB, HIDDEN), device=self.gpu),
            ],
            lambda h, t: ops.matmul(h, ops.transpose(t, 0, 1)),
        )

    def check_logits(self, M):
        w, wbuf = self.table()
        h = rnd(self.rng.uniform(-1, 1, (M, HIDDEN)), BF)
        (out,) = self.logits_model(M).execute(to_buf(h, BF, self.dev), wbuf)
        got = from_buf(out, BF)
        want = rnd(h.astype(np.float32) @ w.T, BF)
        ok, detail = compare(got, want, BF)
        self.report(
            f"logits bf16 [{M}, {HIDDEN}] x [{VOCAB}, {HIDDEN}]^T", ok, detail
        )

    def sampler_model(self, B, k):
        """MAX's sampler graph's op sequence: cast to float32, then the fused
        top-k sampler with per-row k, temperature, top_p, min_p and seed
        tensors and max_k on the CPU."""

        def body(logits, top_k, temperature, top_p, seed):
            return topk_fused_sampling(
                logits=ops.cast(logits, DType.float32),
                top_k=top_k,
                max_k=ops.constant(k, DType.int64, device=DeviceRef.CPU()),
                temperature=temperature,
                top_p=top_p,
                min_top_p=ops.constant(
                    1.0, DType.float32, device=DeviceRef.CPU()
                ),
                min_p=ops.broadcast_to(
                    ops.constant(0.0, DType.float32, device=self.gpu), [B]
                ),
                seed=seed,
            )

        return self.load_typed(
            f"sampler_{B}_{k}",
            [
                TensorType(BF, (B, VOCAB), device=self.gpu),
                TensorType(DType.int64, (B,), device=self.gpu),
                TensorType(DType.float32, (B,), device=self.gpu),
                TensorType(DType.float32, (B,), device=self.gpu),
                TensorType(DType.uint64, (B,), device=self.gpu),
            ],
            body,
        )

    def sampler_args(self, logits, k, seeds):
        B = logits.shape[0]
        return [
            to_buf(logits, BF, self.dev),
            self.ids_buf(np.full(B, k)),
            Buffer.from_numpy(np.ones(B, np.float32)).to(self.dev),
            Buffer.from_numpy(np.ones(B, np.float32)).to(self.dev),
            Buffer.from_numpy(np.asarray(seeds, np.uint64)).to(self.dev),
        ]

    def check_greedy(self, B):
        logits = rnd(self.rng.standard_normal((B, VOCAB)) * 3.0, BF)
        (out,) = self.sampler_model(B, 1).execute(
            *self.sampler_args(logits, 1, range(B))
        )
        got = np.asarray(from_buf(out, DType.int64)).reshape(B)
        rows = np.arange(B)
        ok = bool((logits[rows, got] == logits.max(axis=1)).all())
        self.report(
            f"greedy (fused top-k, k = 1) f32 [{B}, {VOCAB}]",
            ok,
            f"sampled {got.tolist()}, argmax {logits.argmax(axis=1).tolist()}",
        )

    def check_topk_members(self, B=64, k=20):
        logits = rnd(self.rng.standard_normal((B, VOCAB)) * 3.0, BF)
        (out,) = self.sampler_model(B, k).execute(
            *self.sampler_args(logits, k, range(1, B + 1))
        )
        got = np.asarray(from_buf(out, DType.int64)).reshape(B)
        kth = np.sort(logits, axis=1)[:, -k]
        inside = logits[np.arange(B), got] >= kth
        argmax_share = float((got == logits.argmax(axis=1)).mean())
        self.report(
            f"top-k k = {k} f32 [{B}, {VOCAB}]: sample among the top {k}",
            bool(inside.all()),
            f"{int((~inside).sum())} of {B} outside; {argmax_share:.2f} were the argmax",
        )

    def check_topk_distribution(self, B=4096, k=20):
        # One row: 20 logits with known softmax probabilities, the rest far
        # below (never sampled); the same row in every batch row, seeds 1..B.
        p = self.rng.uniform(0.5, 2.0, k)
        p /= p.sum()
        row = np.full(VOCAB, -30.0)
        cols = self.rng.choice(VOCAB, k, replace=False)
        row[cols] = np.log(p) + 5.0
        row = rnd(row, BF)
        exp = np.exp(row[cols].astype(np.float64) - row[cols].max())
        prob = exp / exp.sum()
        logits = np.broadcast_to(row, (B, VOCAB)).copy()
        (out,) = self.sampler_model(B, k).execute(
            *self.sampler_args(logits, k, range(1, B + 1))
        )
        got = np.asarray(from_buf(out, DType.int64)).reshape(B)
        counts = np.array([(got == c).sum() for c in cols], np.float64)
        sigma = np.sqrt(B * prob * (1 - prob))
        z = np.abs(counts - B * prob) / sigma
        outside = int(B - counts.sum())
        self.report(
            f"top-k k = {k} distribution over {B} rows",
            outside == 0 and z.max() < 5,
            f"{outside} samples outside the top {k}; worst |z| {z.max():.2f}",
        )

    def check_scatter(self, T=64, n=16):
        x = rnd(self.rng.uniform(-1, 1, (T, HIDDEN)), BF)
        u = rnd(self.rng.uniform(-1, 1, (n, HIDDEN)), BF)
        idx = self.rng.choice(T - 1, n, replace=False).astype(np.int64)
        idx[0], idx[1], idx[2] = -1, T, 1000
        model = self.load_typed(
            "scatter_nd",
            [
                TensorType(BF, (T, HIDDEN), device=self.gpu),
                TensorType(BF, (n, HIDDEN), device=self.gpu),
                TensorType(DType.int64, (n, 1), device=self.gpu),
            ],
            scatter_nd_skip_oob_indices,
        )
        (out,) = model.execute(
            to_buf(x, BF, self.dev),
            to_buf(u, BF, self.dev),
            self.ids_buf(idx.reshape(n, 1)),
        )
        want = x.copy()
        for j, i in enumerate(idx):
            if -T <= i < T:
                want[i] = u[j]
        got = from_buf(out, BF)
        bad = int((got != want).sum())
        self.report(
            f"scatter_nd.skip_oob_indices bf16 [{T}, {HIDDEN}] <- {n} rows (3 oob)",
            bad == 0,
            f"{bad} of {got.size} wrong",
        )

    def time_ops(self, only):
        _w, wbuf = self.table()
        cases = []
        for M in (1, 16):
            h = to_buf(
                rnd(self.rng.uniform(-1, 1, (M, HIDDEN)), BF), BF, self.dev
            )
            cases.append(
                (
                    f"logits [{M}, {HIDDEN}] x [{VOCAB}, {HIDDEN}]^T",
                    self.logits_model(M),
                    [h, wbuf],
                    2 * VOCAB * HIDDEN + 2 * M * (HIDDEN + VOCAB),
                )
            )
        for T in (1, 512):
            m = self.load_typed(
                f"t_gather_{T}",
                [
                    TensorType(BF, (VOCAB, HIDDEN), device=self.gpu),
                    TensorType(DType.int64, (T,), device=self.gpu),
                ],
                lambda t, i: ops.gather(t, i, axis=0),
            )
            cases.append(
                (
                    f"gather {T} rows of [{VOCAB}, {HIDDEN}]",
                    m,
                    [wbuf, self.ids_buf(self.rng.integers(0, VOCAB, T))],
                    4 * T * HIDDEN,
                )
            )
        logits = rnd(self.rng.standard_normal((1, VOCAB)) * 3.0, BF)
        for k in (1, 20):
            cases.append(
                (
                    f"sampler k = {k} [1, {VOCAB}] (cast + fused top-k)",
                    self.sampler_model(1, k),
                    self.sampler_args(logits, k, [1]),
                    2 * VOCAB + 4 * VOCAB * 2,
                )
            )
        for name, model, args, nbytes in cases:
            if only and only not in name:
                continue
            t, n = self.per_exec(model, args)
            gbs = nbytes / t / 1e9
            print(
                f"TIME {name}: {t * 1e6:.2f} us per execution (n={n}), "
                f"{gbs:.1f} GB/s = {gbs / DRAM_GBS:.3f} of {DRAM_GBS}",
                flush=True,
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    r = SamplingRunner()
    print("device:", r.dev, r.dev.api, r.dev.architecture_name, flush=True)
    if args.time:
        r.time_ops(args.only)
        return 0
    want = lambda name: not args.only or args.only in name
    if want("gather"):
        for T in (1, 7, 512):
            r.check_gather(T)
    if want("logits"):
        for M in (1, 16, 128):
            r.check_logits(M)
    if want("greedy"):
        for B in (1, 4):
            r.check_greedy(B)
    if want("topk"):
        r.check_topk_members()
        r.check_topk_distribution()
    if want("scatter"):
        r.check_scatter()
    print(f"{'PASS' if r.fails == 0 else 'FAIL'}: {r.fails} failing case(s)")
    return 0 if r.fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
