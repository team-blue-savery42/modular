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

"""The elementwise ops and activations Qwen3.5-0.8B uses, through MAX's
graph on the B70, against NumPy.

Run on the GPU host: utils/run-max-py tests/graph/elementwise.py [--time] [--only SUBSTR]

Checks (default): every op the model's graph uses (add, mul, sub, div, pow,
exp, log, negate, sigmoid, silu, gelu, gelu_tanh, cast bf16<->f32, where,
greater, greater_equal, logical_and, logical_not) as a one-op graph, bfloat16
and float32, at decode ([1, 1024], [1, 3584]) and prefill ([128, 3584],
[2048, 3584]) shapes; and the fused chains the model builds: the MLP's
silu(gate) * up from one [T, 2 x 3584] projection, the Gated DeltaNet's gates
(softplus with the x > 20 guard, decay = exp(-exp(A_log) * softplus), beta =
sigmoid(b)) and the full-attention output gate attn * sigmoid(gate).
Reference: NumPy on the same (bf16-rounded) inputs, computed in float64
(float32 for bfloat16 activations, as MAX computes them) and rounded once to
the op's output type; for chains, rounded after each op as the graph's types
say. Pass: bfloat16 within 1 ulp (or 1e-5 absolute, MAX's atol, where
the float32 formula cancels); float32 within MAX's activation-test
tolerance (rtol = atol = 1e-5, test_activations_gpu.py); bool exact. A case
outside that is run through MAX's own CPU kernels too: "PASS (= MAX CPU)"
means the B70 is within 1 ulp (bf16) or the f32 tolerance of MAX's CPU
result, i.e. the difference is MAX's algorithm (bf16 `pow` computes
exp(y * log(x)) in bfloat16 in the stdlib's generic `_powf`), not the B70's.

--time: per-op time of one-op graphs (executions back to back on one
queue, one synchronize; batches of >= 100 ms; 3 s idle, a warm-up batch, the
median of 5), bfloat16 at decode and [2048, 3584], with GB/s (bytes read and
written once over the time) against the 594.6 GB/s DRAM read ceiling.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

DRAM_GBS = 594.6


def bf16_round(x: np.ndarray) -> np.ndarray:
    """Values rounded to the nearest bfloat16 (ties to even), as float32."""
    f = np.asarray(x, dtype=np.float32)
    bits = f.view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).astype(
        np.uint32
    )
    out = rounded.view(np.float32)
    return np.where(np.isnan(f), np.float32(np.nan), out)


def rnd(x: np.ndarray, dt: DType) -> np.ndarray:
    """float64 results rounded once to `dt` (as float32 values for bf16)."""
    f32 = np.asarray(x, dtype=np.float64).astype(np.float32)
    return bf16_round(f32) if dt == DType.bfloat16 else f32


def to_buf(x: np.ndarray, dt: DType, dev) -> Buffer:
    if dt == DType.bfloat16:
        bits = (
            np.ascontiguousarray(x, dtype=np.float32).view(np.uint32) >> 16
        ).astype(np.uint16)
        return Buffer.from_numpy(bits).view(DType.bfloat16, x.shape).to(dev)
    if dt == DType.bool:
        return Buffer.from_numpy(np.ascontiguousarray(x, dtype=np.bool_)).to(
            dev
        )
    return Buffer.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)


def from_buf(buf: Buffer, dt: DType) -> np.ndarray:
    host = buf.to(CPU())
    if dt == DType.bfloat16:
        bits = host.view(DType.uint16, buf.shape).to_numpy()
        return (bits.astype(np.uint32) << 16).view(np.float32)
    return np.array(host.to_numpy())


def ordered(x: np.ndarray, dt: DType) -> np.ndarray:
    """Bit patterns as integers ordered like the values (for ulp distances)."""
    if dt == DType.bfloat16:
        b = (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.int64)
        return np.where(b & 0x8000, 0x8000 - b, b)
    b = np.asarray(x, np.float32).view(np.uint32).astype(np.int64)
    return np.where(b & 0x80000000, 0x80000000 - b, b)


def compare(got: np.ndarray, want: np.ndarray, dt: DType) -> tuple[bool, str]:
    if dt == DType.bool:
        bad = int((got != want).sum())
        return bad == 0, f"{bad} of {want.size} wrong"
    both_nan = np.isnan(got) & np.isnan(want)
    ulps = np.abs(ordered(got, dt) - ordered(want, dt))
    ulps = np.where(both_nan, 0, ulps)
    worst = int(ulps.max())
    if dt == DType.bfloat16:
        # 1 ulp, or MAX's absolute tolerance (1e-5) where the float32 formula
        # cancels (gelu_tanh's 1 + tanh(x) for x << 0: results ~1e-5 and
        # below, whose low bits depend on float32 tanh's last ulp).
        far = (ulps > 1) & (np.abs(got.astype(np.float64) - want) > 1e-5)
        return not far.any(), (
            f"worst {worst} ulp, {int((ulps > 0).sum())} of {want.size} not exact, "
            f"{int(far.sum())} above 1 ulp and 1e-5"
        )
    tol = 1e-5 + 1e-5 * np.abs(want)
    bad = int(((np.abs(got - want) > tol) & ~both_nan).sum())
    return (
        bad == 0,
        f"worst {worst} ulp (f32), {bad} of {want.size} outside rtol=atol=1e-5",
    )


def unif(lo, hi):
    return lambda rng, shape: rng.uniform(lo, hi, shape)


def away_from_zero(rng, shape):
    return np.sign(rng.uniform(-1, 1, shape)) * np.exp(
        rng.uniform(np.log(0.25), np.log(4.0), shape)
    )


def log_unif(rng, shape):
    return np.exp(rng.uniform(np.log(1e-3), np.log(100.0), shape))


# The activation references follow MAX's formulas (nn/activations.mojo) in
# MAX's precision: for bfloat16 a float32 computation rounded once (MAX
# computes bf16 activations in float32), whose cancellation near zero (gelu's
# 1 + erf for x << 0) a float64 reference wouldn't show.
def _erf(x):
    return np.vectorize(math.erf)(x.astype(np.float64)).astype(x.dtype)


def gelu_ref(x):
    return (
        x
        * x.dtype.type(0.5)
        * (x.dtype.type(1) + _erf(x / x.dtype.type(1.4142135623730951)))
    )


def gelu_tanh_ref(x):
    c = x.dtype.type
    return (
        x
        * c(0.5)
        * (
            c(1)
            + np.tanh(c(0.7978845608028654) * (x + c(0.044715) * x * x * x))
        )
    )


FLOAT_OPS = {
    # name: (inputs [generator per input], graph fn, reference fn)
    "add": ([unif(-4, 4), unif(-4, 4)], lambda a, b: a + b, lambda a, b: a + b),
    "mul": ([unif(-4, 4), unif(-4, 4)], lambda a, b: a * b, lambda a, b: a * b),
    "sub": ([unif(-4, 4), unif(-4, 4)], lambda a, b: a - b, lambda a, b: a - b),
    "div": (
        [unif(-4, 4), away_from_zero],
        lambda a, b: a / b,
        lambda a, b: a / b,
    ),
    "pow": ([unif(0.1, 4), unif(-3, 3)], ops.pow, np.power),
    "exp": ([unif(-10, 10)], ops.exp, np.exp),
    "log": ([log_unif], ops.log, np.log),
    "negate": ([unif(-4, 4)], ops.negate, np.negative),
    "sigmoid": (
        [unif(-12, 12)],
        ops.sigmoid,
        lambda x: 1.0 / (1.0 + np.exp(-x)),
    ),
    "silu": ([unif(-12, 12)], ops.silu, lambda x: x / (1.0 + np.exp(-x))),
    "gelu": ([unif(-6, 6)], ops.gelu, gelu_ref),
    "gelu_tanh": (
        [unif(-6, 6)],
        lambda x: ops.gelu(x, approximate="tanh"),
        gelu_tanh_ref,
    ),
}

ACTIVATIONS = {"exp", "log", "sigmoid", "silu", "gelu", "gelu_tanh"}

SHAPES = [(1, 1024), (1, 3584), (128, 3584), (2048, 3584)]


class Runner:
    def __init__(self):
        self.dev = Accelerator()
        self.session = InferenceSession(devices=[self.dev])
        self.gpu = DeviceRef.GPU()
        self.rng = np.random.default_rng(2)
        self.fails = 0
        self.cpu_session = None

    def load(self, name, in_types, body):
        types = [
            TensorType(dt, shape, device=self.gpu) for dt, shape in in_types
        ]
        with Graph(name, input_types=types) as g:
            outs = body(*(v.tensor for v in g.inputs))
            g.output(*(outs if isinstance(outs, tuple) else (outs,)))
        return self.session.load(g)

    def report(self, label, ok, detail):
        self.fails += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} {label}: {detail}", flush=True)

    def check_float_op(self, name, dt, shape):
        gens, gfn, rfn = FLOAT_OPS[name]
        xs = [rnd(gen(self.rng, shape), dt) for gen in gens]
        model = self.load(f"{name}_{dt}", [(dt, shape)] * len(xs), gfn)
        (out,) = model.execute(*(to_buf(x, dt, self.dev) for x in xs))
        got = from_buf(out, dt)
        ref_t = (
            np.float32
            if dt == DType.bfloat16 and name in ACTIVATIONS
            else np.float64
        )
        want = rnd(rfn(*(x.astype(ref_t) for x in xs)), dt)
        ok, detail = compare(got, want, dt)
        label = f"{name} {dt} {list(shape)}"
        if not ok:
            cpu = self.cpu_result(name, gfn, dt, shape, xs)
            ok_c, detail_c = compare(got, cpu, dt)
            if ok_c:
                print(
                    f"PASS (= MAX CPU) {label}: against MAX's CPU kernels {detail_c}; "
                    f"against NumPy {detail}",
                    flush=True,
                )
                return
            detail += f"; against MAX's CPU kernels {detail_c}"
        self.report(label, ok, detail)

    def cpu_result(self, name, gfn, dt, shape, xs, shapes=None):
        if self.cpu_session is None:
            self.cpu_session = InferenceSession(devices=[CPU()])
        types = [
            TensorType(dt, sh, device=DeviceRef.CPU())
            for sh in (shapes or [shape] * len(xs))
        ]
        with Graph(f"{name}_{dt}_cpu", input_types=types) as g:
            g.output(gfn(*(v.tensor for v in g.inputs)))
        (out,) = self.cpu_session.load(g).execute(
            *(to_buf(x, dt, CPU()) for x in xs)
        )
        return from_buf(out, dt)

    def check_cast(self, src, dst, shape):
        x = self.rng.standard_normal(shape) * 100.0
        x = rnd(x, src)
        model = self.load(
            f"cast_{src}_{dst}", [(src, shape)], lambda a: ops.cast(a, dst)
        )
        (out,) = model.execute(to_buf(x, src, self.dev))
        ok, detail = compare(
            from_buf(out, dst), rnd(x.astype(np.float64), dst), dst
        )
        self.report(f"cast {src}->{dst} {list(shape)}", ok, detail)

    def check_predicates(self, dt, shape):
        # Values on a coarse grid so equal pairs occur.
        a = rnd(np.round(self.rng.uniform(-4, 4, shape) * 2) / 2, dt)
        b = rnd(np.round(self.rng.uniform(-4, 4, shape) * 2) / 2, dt)
        for name, gfn, rfn in (
            ("greater", ops.greater, np.greater),
            ("greater_equal", ops.greater_equal, np.greater_equal),
        ):
            model = self.load(f"{name}_{dt}", [(dt, shape)] * 2, gfn)
            (out,) = model.execute(
                to_buf(a, dt, self.dev), to_buf(b, dt, self.dev)
            )
            ok, detail = compare(
                from_buf(out, DType.bool), rfn(a, b), DType.bool
            )
            self.report(f"{name} {dt} {list(shape)}", ok, detail)
        c = self.rng.uniform(size=shape) < 0.5
        model = self.load(
            f"where_{dt}",
            [(DType.bool, shape), (dt, shape), (dt, shape)],
            ops.where,
        )
        (out,) = model.execute(
            to_buf(c, DType.bool, self.dev),
            to_buf(a, dt, self.dev),
            to_buf(b, dt, self.dev),
        )
        ok, detail = compare(from_buf(out, dt), np.where(c, a, b), dt)
        self.report(f"where {dt} {list(shape)}", ok, detail)

    def check_logical(self, shape):
        p = self.rng.uniform(size=shape) < 0.5
        q = self.rng.uniform(size=shape) < 0.5
        model = self.load(
            "logical_and", [(DType.bool, shape)] * 2, ops.logical_and
        )
        (out,) = model.execute(
            to_buf(p, DType.bool, self.dev), to_buf(q, DType.bool, self.dev)
        )
        ok, detail = compare(from_buf(out, DType.bool), p & q, DType.bool)
        self.report(f"logical_and {list(shape)}", ok, detail)
        model = self.load("logical_not", [(DType.bool, shape)], ops.logical_not)
        (out,) = model.execute(to_buf(p, DType.bool, self.dev))
        ok, detail = compare(from_buf(out, DType.bool), ~p, DType.bool)
        self.report(f"logical_not {list(shape)}", ok, detail)

    def check_swiglu(self, T, width=3584):
        bf = DType.bfloat16
        gu = rnd(self.rng.uniform(-6, 6, (T, 2 * width)), bf)
        model = self.load(
            "swiglu",
            [(bf, (T, 2 * width))],
            lambda x: (lambda g, u: ops.silu(g) * u)(  # noqa: PLC3002
                *ops.split(x, [width, width], axis=-1)
            ),
        )
        (out,) = model.execute(to_buf(gu, bf, self.dev))
        g, u = (
            gu[:, :width].astype(np.float64),
            gu[:, width:].astype(np.float64),
        )
        want = rnd(rnd(g / (1.0 + np.exp(-g)), bf).astype(np.float64) * u, bf)
        ok, detail = compare(from_buf(out, bf), want, bf)
        self.report(f"chain silu(gate)*up bf16 [{T}, 2x{width}]", ok, detail)

    def check_output_gate(self, T, width=2048):
        bf = DType.bfloat16
        a = rnd(self.rng.uniform(-4, 4, (T, width)), bf)
        g = rnd(self.rng.uniform(-8, 8, (T, width)), bf)
        model = self.load(
            "out_gate", [(bf, (T, width))] * 2, lambda x, y: x * ops.sigmoid(y)
        )
        (out,) = model.execute(to_buf(a, bf, self.dev), to_buf(g, bf, self.dev))
        s = rnd(1.0 / (1.0 + np.exp(-g.astype(np.float64))), bf)
        want = rnd(a.astype(np.float64) * s, bf)
        ok, detail = compare(from_buf(out, bf), want, bf)
        self.report(f"chain attn*sigmoid(gate) bf16 [{T}, {width}]", ok, detail)

    def check_gates(self, T, nv=16):
        """gated_deltanet.py: decay and beta from a, b (bf16), A_log, dt_bias."""
        bf, f32 = DType.bfloat16, DType.float32
        a = rnd(self.rng.uniform(-8, 24, (T, nv)), bf)  # some x > 20
        b = rnd(self.rng.uniform(-8, 8, (T, nv)), bf)
        a_log = rnd(self.rng.uniform(-2, 2, (nv,)), bf)
        dt_bias = rnd(self.rng.uniform(-1, 1, (nv,)), bf)

        def body(a_, b_, alog_, dtb_):
            A = ops.exp(ops.cast(alog_, f32))
            x = ops.cast(a_, f32) + ops.cast(dtb_, f32)
            sp = ops.where(x > 20.0, x, ops.log(1.0 + ops.exp(x)))
            decay = ops.exp(ops.cast(-A * sp, f32))
            beta = ops.cast(ops.sigmoid(b_), f32)
            return decay, beta

        model = self.load(
            "gates",
            [(bf, (T, nv)), (bf, (T, nv)), (bf, (nv,)), (bf, (nv,))],
            body,
        )
        decay, beta = model.execute(
            *(to_buf(v, bf, self.dev) for v in (a, b, a_log, dt_bias))
        )
        # float32 ops, each rounded to float32 (the graph's types).
        A = rnd(np.exp(a_log.astype(np.float64)), f32).astype(np.float64)
        x = rnd(a.astype(np.float64) + dt_bias, f32).astype(np.float64)
        e = rnd(np.exp(x), f32).astype(np.float64)
        sp = np.where(
            x > 20.0, x, rnd(np.log(rnd(1.0 + e, f32).astype(np.float64)), f32)
        )
        prod = rnd(-A * sp, f32).astype(np.float64)
        want_decay = rnd(np.exp(prod), f32)
        want_beta = rnd(
            rnd(1.0 / (1.0 + np.exp(-b.astype(np.float64))), bf), f32
        )
        ok, detail = compare(from_buf(decay, f32), want_decay, f32)
        self.report(
            f"chain decay=exp(-exp(A_log)*softplus(a+dt_bias)) f32 [{T}, {nv}]",
            ok,
            detail,
        )
        ok, detail = compare(from_buf(beta, f32), want_beta, f32)
        self.report(f"chain beta=sigmoid(b) bf16->f32 [{T}, {nv}]", ok, detail)

    def per_exec(self, model, args):
        for _ in range(3):
            model.execute(*args)
        self.dev.synchronize()
        t0 = time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < 0.02:
            model.execute(*args)
            n += 1
        self.dev.synchronize()
        n = max(10, int(n * 0.1 / (time.perf_counter() - t0)) + 1)
        time.sleep(3)
        for _ in range(n):  # warm-up batch
            model.execute(*args)
        self.dev.synchronize()
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(n):
                model.execute(*args)
            self.dev.synchronize()
            times.append((time.perf_counter() - t0) / n)
        return float(np.median(times)), n

    def time_ops(self, only):
        bf = DType.bfloat16
        cases = [
            (name, len(FLOAT_OPS[name][0]), FLOAT_OPS[name][1])
            for name in (
                "add",
                "mul",
                "sub",
                "div",
                "pow",
                "exp",
                "log",
                "negate",
                "sigmoid",
                "silu",
                "gelu",
                "gelu_tanh",
            )
        ]
        for shape in [(1, 1024), (1, 3584), (2048, 3584)]:
            for name, arity, gfn in cases + [
                ("swiglu", 1, None),
                ("cast_f32", 1, None),
            ]:
                if only and only not in name:
                    continue
                if name == "swiglu":
                    w = shape[1]
                    model = self.load(
                        "t_swiglu",
                        [(bf, (shape[0], 2 * w))],
                        lambda x: (lambda g, u: ops.silu(g) * u)(  # noqa: PLC3002, B023
                            *ops.split(x, [w, w], axis=-1)  # noqa: B023
                        ),
                    )
                    args = [
                        to_buf(
                            rnd(self.rng.uniform(-4, 4, (shape[0], 2 * w)), bf),
                            bf,
                            self.dev,
                        )
                    ]
                    nbytes = 2 * shape[0] * 2 * w + 2 * shape[0] * w
                elif name == "cast_f32":
                    model = self.load(
                        "t_cast",
                        [(bf, shape)],
                        lambda a: ops.cast(a, DType.float32),
                    )
                    args = [
                        to_buf(
                            rnd(self.rng.uniform(-4, 4, shape), bf),
                            bf,
                            self.dev,
                        )
                    ]
                    nbytes = 6 * math.prod(shape)
                else:
                    gens = FLOAT_OPS[name][0]
                    model = self.load(f"t_{name}", [(bf, shape)] * arity, gfn)
                    args = [
                        to_buf(rnd(g(self.rng, shape), bf), bf, self.dev)
                        for g in gens
                    ]
                    nbytes = 2 * math.prod(shape) * (arity + 1)
                t, n = self.per_exec(model, args)
                gbs = nbytes / t / 1e9
                print(
                    f"TIME {name} bf16 {list(shape)}: {t * 1e6:.2f} us per op (n={n}), "
                    f"{gbs:.1f} GB/s = {gbs / DRAM_GBS:.3f} of {DRAM_GBS}",
                    flush=True,
                )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    r = Runner()
    print("device:", r.dev, r.dev.api, r.dev.architecture_name, flush=True)
    if args.time:
        r.time_ops(args.only)
        return 0
    for shape in SHAPES:
        for dt in (DType.bfloat16, DType.float32):
            for name in FLOAT_OPS:
                if args.only and args.only not in name:
                    continue
                r.check_float_op(name, dt, shape)
            if not args.only or "pred" in args.only:
                r.check_predicates(dt, shape)
        if not args.only or "cast" in args.only:
            r.check_cast(DType.bfloat16, DType.float32, shape)
            r.check_cast(DType.float32, DType.bfloat16, shape)
        if not args.only or "pred" in args.only:
            r.check_logical(shape)
    if not args.only or "chain" in args.only:
        for T in (1, 128, 2048):
            r.check_swiglu(T)
            r.check_output_gate(T)
            r.check_gates(T)
    print(f"{'PASS' if r.fails == 0 else 'FAIL'}: {r.fails} failing case(s)")
    return 0 if r.fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
