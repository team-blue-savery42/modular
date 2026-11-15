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

"""The norms Qwen3.5-0.8B uses, through MAX's graph on the B70, against NumPy.

Run on the GPU host: utils/run-max-py tests/graph/norms.py [--time]

Checks: ops.rms_norm as the model calls it (eps 1e-6, multiply_before_cast
False): the layer norms over [T, 1024] with the (1 + w) weight, the q/k norms
over [T, 8, 256] and [T, 2, 256] (1 + w), the Gated DeltaNet's gated norm
over [T, 16, 128] with the plain weight (bfloat16 in the model's graph: its
float32 weight is cast to the input's type; float32 checked too), for
T = 1, 128, 512, 2048; and ops.layer_norm float32 over [T, 768] (the vision
tower). Reference (NumPy, MAX's formula, nn/normalization.mojo with
multiply_before_cast False): r = 1 / sqrt(mean(x^2) + eps) in float64,
y = round(x * r), out = round(y * round(w + offset)), r in float32 as the
kernel (and x * r moved by 2^-21 either way: the kernel's float32 rsqrt isn't
correctly rounded, which can move the first rounding); bfloat16 within 1 ulp
(or 1e-5 absolute), float32 within rtol = atol = 1e-5. LayerNorm:
(x - mean) / sqrt(var + eps) * gamma + beta, float32 within 1e-5.

--time: per-op time of one-op graphs (as tests/graph/elementwise.py)
at the decode shapes and [512, 1024], [2048, 1024], with GB/s.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
from elementwise import DRAM_GBS, Runner, compare, from_buf, rnd, to_buf
from max.dtype import DType
from max.graph import ops

EPS = 1e-6


def rms_ref(x, w, offset, dt):
    """MAX's formula in float32, as the kernel computes it (accumulation in
    float32; the statistics' sum in float64 here, then rounded)."""
    x32 = x.astype(np.float32)
    ms = np.float32((x32.astype(np.float64) ** 2).mean(axis=-1, keepdims=True))
    r = (np.float32(1) / np.sqrt(ms + np.float32(EPS))).astype(np.float32)
    g = rnd(w.astype(np.float64) + offset, dt).astype(np.float64)
    # The kernel's rsqrt is float32 within a few ulps, not correctly rounded:
    # where x * r lies that close to a rounding boundary of the first rounding,
    # either neighbour is MAX's formula: the three candidates are returned.
    return [
        rnd(
            rnd((x32 * r).astype(np.float64) * (1 + k * 2.0**-21), dt).astype(
                np.float64
            )
            * g,
            dt,
        )
        for k in (0, -1, 1)
    ]


class NormRunner(Runner):
    def check_rms(self, label, dt, shape, offset):
        x = rnd(self.rng.standard_normal(shape) * 2.0, dt)
        w = rnd(
            self.rng.uniform(-0.5, 0.5, (shape[-1],)) + (0 if offset else 1), dt
        )
        model = self.load(
            f"rms_{label}",
            [(dt, shape), (dt, (shape[-1],))],
            lambda a, b: ops.rms_norm(
                a, b, EPS, weight_offset=offset, multiply_before_cast=False
            ),
        )
        (out,) = model.execute(to_buf(x, dt, self.dev), to_buf(w, dt, self.dev))
        got = from_buf(out, dt)
        refs = rms_ref(x, w, offset, dt)
        # Per element, the candidate closest to the B70's value.
        from elementwise import ordered

        dist = np.stack(
            [np.abs(ordered(got, dt) - ordered(c, dt)) for c in refs]
        )
        best = np.take_along_axis(
            np.stack(refs), dist.argmin(axis=0)[None], axis=0
        )[0]
        ok, detail = compare(got, best, dt)
        name = f"rms_norm {label} {dt} {list(shape)}"
        if not ok:
            # The kernel rounds twice (x * r, then * (w + offset)); a 1-ulp
            # difference in float32 rsqrt can move the first rounding. MAX's
            # CPU kernels decide whether it is the formula or the B70.
            fn = lambda a, b: ops.rms_norm(
                a, b, EPS, weight_offset=offset, multiply_before_cast=False
            )
            cpu = self.cpu_result(
                f"rms_{label}",
                fn,
                dt,
                shape,
                [x, w],
                shapes=[shape, (shape[-1],)],
            )
            ok_c, detail_c = compare(got, cpu, dt)
            if ok_c:
                print(
                    f"PASS (= MAX CPU) {name}: against MAX's CPU kernels {detail_c}; "
                    f"against NumPy {detail}",
                    flush=True,
                )
                return
            detail += f"; against MAX's CPU kernels {detail_c}"
        self.report(name, ok, detail)

    def check_layer_norm(self, T, width=768):
        f32 = DType.float32
        x = self.rng.standard_normal((T, width)).astype(np.float32) * 3 + 0.5
        g = self.rng.uniform(0.5, 1.5, (width,)).astype(np.float32)
        b = self.rng.uniform(-0.5, 0.5, (width,)).astype(np.float32)
        model = self.load(
            "layer_norm",
            [(f32, (T, width)), (f32, (width,)), (f32, (width,))],
            lambda a, gg, bb: ops.layer_norm(a, gg, bb, EPS),
        )
        (out,) = model.execute(*(to_buf(v, f32, self.dev) for v in (x, g, b)))
        x64 = x.astype(np.float64)
        mu = x64.mean(axis=-1, keepdims=True)
        var = ((x64 - mu) ** 2).mean(axis=-1, keepdims=True)
        want = rnd((x64 - mu) / np.sqrt(var + EPS) * g + b, f32)
        ok, detail = compare(from_buf(out, f32), want, f32)
        self.report(f"layer_norm f32 [{T}, {width}]", ok, detail)

    def time_norms(self):
        bf = DType.bfloat16
        cases = [
            ("layer", (1, 1024), 1.0),
            ("qnorm", (1, 8, 256), 1.0),
            ("knorm", (1, 2, 256), 1.0),
            ("gated", (1, 16, 128), 0.0),
            ("layer", (128, 1024), 1.0),
            ("layer", (512, 1024), 1.0),
            ("layer", (2048, 1024), 1.0),
            ("qnorm", (2048, 8, 256), 1.0),
            ("gated", (2048, 16, 128), 0.0),
        ]
        for label, shape, offset in cases:
            x = rnd(self.rng.standard_normal(shape), bf)
            w = rnd(self.rng.uniform(-0.5, 0.5, (shape[-1],)), bf)
            model = self.load(
                f"t_rms_{label}",
                [(bf, shape), (bf, (shape[-1],))],
                lambda a, b, o=offset: ops.rms_norm(
                    a, b, EPS, weight_offset=o, multiply_before_cast=False
                ),
            )
            t, n = self.per_exec(
                model, [to_buf(x, bf, self.dev), to_buf(w, bf, self.dev)]
            )
            nbytes = 2 * 2 * math.prod(shape) + 2 * shape[-1]
            gbs = nbytes / t / 1e9
            print(
                f"TIME rms_norm {label} bf16 {list(shape)}: {t * 1e6:.2f} us per op (n={n}), "
                f"{gbs:.1f} GB/s = {gbs / DRAM_GBS:.3f} of {DRAM_GBS}",
                flush=True,
            )
        f32 = DType.float32
        for T in (1, 1024):
            x = self.rng.standard_normal((T, 768)).astype(np.float32)
            g = np.ones(768, np.float32)
            model = self.load(
                "t_ln",
                [(f32, (T, 768)), (f32, (768,)), (f32, (768,))],
                lambda a, gg, bb: ops.layer_norm(a, gg, bb, EPS),
            )
            t, n = self.per_exec(
                model, [to_buf(v, f32, self.dev) for v in (x, g, g)]
            )
            gbs = 2 * 4 * T * 768 / t / 1e9
            print(
                f"TIME layer_norm f32 [{T}, 768]: {t * 1e6:.2f} us per op (n={n}), "
                f"{gbs:.1f} GB/s = {gbs / DRAM_GBS:.3f} of {DRAM_GBS}",
                flush=True,
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    args = ap.parse_args()
    r = NormRunner()
    print("device:", r.dev, r.dev.api, r.dev.architecture_name, flush=True)
    if args.time:
        r.time_norms()
        return 0
    for T in (1, 128, 512, 2048):
        r.check_rms("layer", DType.bfloat16, (T, 1024), 1.0)
        r.check_rms("qnorm", DType.bfloat16, (T, 8, 256), 1.0)
        r.check_rms("knorm", DType.bfloat16, (T, 2, 256), 1.0)
        r.check_rms("gated", DType.bfloat16, (T, 16, 128), 0.0)
        r.check_rms("gated", DType.float32, (T, 16, 128), 0.0)
        r.check_layer_norm(T)
    print(f"{'PASS' if r.fails == 0 else 'FAIL'}: {r.fails} failing case(s)")
    return 0 if r.fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
