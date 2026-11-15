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

"""A MAX graph on the B70 through MAX's runtime on the HAL DeviceContext,
against NumPy.

Run on the GPU host: utils/run-max-py tests/graph/graph_smoke.py

One graph: y = rms_norm(a @ b + c, w, eps, weight_offset=1) with bfloat16
inputs (the Qwen3.5 layer norm's (1 + w) form, hidden 1024), then silu(y) * y.
Checks: the B70's output equals MAX's own CPU kernels on the same graph
(within 1 bf16 ulp), and is within 8 bf16 ulps of a float64 NumPy reference
(MAX rounds inside fused ops where NumPy doesn't: measured 0 ulps against the
CPU kernels and 6 ulps worst, p99 4, against NumPy). Then graphs of 1 and 100
adds on [1, 1024] tensors are timed (executions back to back, one
synchronize) as the first measure of the graph path's per-execution and
per-op cost.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

M, K, N = 64, 512, 1024
EPS = 1e-6


def bf16_round(x: np.ndarray) -> np.ndarray:
    """float32 values rounded to the nearest bfloat16 (ties to even)."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def to_device(x: np.ndarray, dev: Accelerator) -> Buffer:
    """A bfloat16 device buffer holding float32 `x` (already bf16 values)."""
    bits = (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    host = Buffer.from_numpy(bits).view(DType.bfloat16, x.shape)
    return host.to(dev)


def to_host(buf: Buffer) -> np.ndarray:
    """A bfloat16 buffer as float32 NumPy values."""
    from max.driver import CPU

    bits = buf.to(CPU()).view(DType.uint16, buf.shape).to_numpy()
    return (bits.astype(np.uint32) << 16).view(np.float32)


def main() -> int:
    rng = np.random.default_rng(6)
    dev = Accelerator()
    print("device:", dev, dev.api, dev.architecture_name)
    session = InferenceSession(devices=[dev])
    gpu = DeviceRef.GPU()

    types = [
        TensorType(DType.bfloat16, [M, K], device=gpu),
        TensorType(DType.bfloat16, [K, N], device=gpu),
        TensorType(DType.bfloat16, [M, N], device=gpu),
        TensorType(DType.bfloat16, [N], device=gpu),
    ]
    with Graph("graph_smoke", input_types=types) as g:
        a, b, c, w = (v.tensor for v in g.inputs)
        x = ops.matmul(a, b) + c
        y = ops.rms_norm(x, w, EPS, weight_offset=1.0)
        g.output(ops.silu(y) * y)
    model = session.load(g)

    a = bf16_round(rng.uniform(-1, 1, (M, K)).astype(np.float32))
    b = bf16_round(rng.uniform(-1, 1, (K, N)).astype(np.float32) / 16)
    c = bf16_round(rng.uniform(-1, 1, (M, N)).astype(np.float32))
    w = bf16_round(rng.uniform(-0.5, 0.5, (N,)).astype(np.float32))
    (out,) = model.execute(*(to_device(v, dev) for v in (a, b, c, w)))
    got = to_host(out)

    # Reference: the graph's bf16 roundings (matmul and add outputs, the
    # norm's output) applied in float64 otherwise.
    xr = bf16_round((a.astype(np.float64) @ b).astype(np.float32))
    xr = bf16_round((xr.astype(np.float64) + c).astype(np.float32))
    rms = np.sqrt(
        (xr.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + EPS
    )
    yr = bf16_round((xr / rms * (1.0 + w)).astype(np.float32)).astype(
        np.float64
    )
    sr = bf16_round((yr / (1.0 + np.exp(-yr))).astype(np.float32)).astype(
        np.float64
    )
    want = bf16_round((sr * yr).astype(np.float32)).astype(np.float64)
    # Within 2 bf16 ulps of the reference (ulp of |want|, at least the ulp
    # of 2^-8 of the row's largest value, for results near zero).
    floor = np.abs(want).max(axis=1, keepdims=True) * 2.0**-8
    mag = np.maximum(np.abs(want), floor)
    ulp = 2.0 ** (np.floor(np.log2(mag)) - 7)
    err = np.abs(got - want) / ulp
    worst = float(err.max())
    i, j = np.unravel_index(int(err.argmax()), err.shape)
    print(
        f"  worst at [{i},{j}]: got {got[i, j]:.6g} want {want[i, j]:.6g}; "
        f"ulp error p50 {np.percentile(err, 50):.2f} p99 {np.percentile(err, 99):.2f}; "
        f"elements > 2 ulps: {int((err > 2).sum())} of {err.size}"
    )
    # The same against a reference rounding only the inputs and the output
    # (the graph compiler fuses ops and keeps float32 between them).
    xf = a.astype(np.float64) @ b + c
    yf = xf / np.sqrt((xf**2).mean(axis=1, keepdims=True) + EPS) * (1.0 + w)
    wf = bf16_round((yf / (1.0 + np.exp(-yf)) * yf).astype(np.float32)).astype(
        np.float64
    )
    err_f = np.abs(got - wf) / ulp
    print(
        f"  unrounded reference: worst {err_f.max():.2f} ulps, p99 {np.percentile(err_f, 99):.2f}, "
        f"elements > 2 ulps: {int((err_f > 2).sum())}"
    )
    # MAX's own CPU kernels on the same graph (the CPU path MAX's model
    # can't take, but its ops can).
    from max.driver import CPU

    cpu_dev = CPU()
    cpu_types = [
        TensorType(t.dtype, t.shape, device=DeviceRef.CPU()) for t in types
    ]
    with Graph("graph_smoke_cpu", input_types=cpu_types) as gc:
        a_, b_, c_, w_ = (v.tensor for v in gc.inputs)
        y_ = ops.rms_norm(ops.matmul(a_, b_) + c_, w_, EPS, weight_offset=1.0)
        gc.output(ops.silu(y_) * y_)
    cpu_session = InferenceSession(devices=[cpu_dev])
    (cpu_out,) = cpu_session.load(gc).execute(
        *(to_device(v, cpu_dev) for v in (a, b, c, w))
    )
    cpu_got = to_host(cpu_out)
    err_c = np.abs(got - cpu_got) / ulp
    print(
        f"  B70 against MAX's CPU kernels: worst {err_c.max():.2f} ulps, "
        f"p99 {np.percentile(err_c, 99):.2f}, elements > 2 ulps: {int((err_c > 2).sum())}; "
        f"CPU against the reference: worst {float((np.abs(cpu_got - want) / ulp).max()):.2f} ulps"
    )
    # MAX's kernels round where the reference doesn't model (bf16 math in
    # silu): the check is the B70 against MAX's CPU kernels, within 1 ulp,
    # and both within 8 ulps of float64.
    ok = err_c.max() <= 1.0 and worst <= 8.0
    print(
        f"{'PASS' if ok else 'FAIL'} matmul+add+rms_norm+silu*y bf16 "
        f"[{M}x{K}]x[{K}x{N}]: worst error {worst:.3g} bf16 ulps against NumPy, "
        f"{err_c.max():.3g} against MAX's CPU kernels"
    )

    # Cost of the graph path (executions back to back, one
    # synchronize, median of 5 batches of 50): a graph adding two [1, 1024]
    # bf16 inputs once, and one chaining 100 such adds (v = v + y; the graph
    # compiler may fuse the chain into fewer kernels: unitrace counts them).
    def timed(model, *args, runs=50):
        for _ in range(20):
            model.execute(*args)
        dev.synchronize()
        times = []
        for _ in range(5):
            time.sleep(0.5)
            t0 = time.perf_counter()
            for _ in range(runs):
                model.execute(*args)
            dev.synchronize()
            times.append((time.perf_counter() - t0) / runs)
        return float(np.median(times))

    def rand_dev():
        return to_device(
            bf16_round(rng.uniform(-1, 1, (1, N)).astype(np.float32)), dev
        )

    x0, y0 = rand_dev(), rand_dev()
    for chain in (1, 100):
        in_type = TensorType(DType.bfloat16, [1, N], device=gpu)
        with Graph(f"add_chain_{chain}", input_types=[in_type, in_type]) as g2:
            v, y = (t.tensor for t in g2.inputs)
            for _ in range(chain):
                v = v + y
            g2.output(v)
        t = timed(session.load(g2), x0, y0)
        print(
            f"add_chain {chain}: {t * 1e6:.1f} us per execution "
            f"({t / chain * 1e6:.2f} us per add)"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
