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

"""The Gated DeltaNet's two custom ops on the B70 through MAX's graph,
against NumPy.

`gated_delta_conv1d_fwd` (causal depthwise conv, kernel 4, conv_dim 6144,
slot-indexed window pool) then `gated_delta_recurrence_fwd` (16 heads of
128, slot-indexed state pool) at Qwen3.5-0.8B's shapes, as one graph: one
sequence of T tokens (decode T = 1, prefill T = 128, 512, 2048), a decode
batch of four one-token sequences (the Intel decode kernel) and a ragged
batch [1, 300, 1] (a prefill row beside decode rows), pools in float32 and in
bfloat16 (the model's), from random state. The reference runs the same
recurrence in float64 (q and k L2-normalised with eps 1e-6 inside the root,
q scaled by 1/sqrt(128)). Checks: outputs and the float32 state within 1e-5
of the reference relative to its largest magnitude (the bfloat16 state
within 1 bf16 ulp of the rounded reference, ~4e-3), the window exact.

Run on the GPU host: utils/run-max-py tests/graph/gated_delta.py
Each case also runs with the conv taking a bfloat16 input and applying SiLU
(the `silu` flag: Qwen3.5's Intel path) from the first CONV_DIM columns of a
wider input (the projection it is a part of), the reference applying SiLU to
the conv output before the recurrence.
Times one graph execution per case (median of 20 after 3) for reference.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.nn._graph_ops.state_space import (
    gated_delta_conv1d_fwd,
    gated_delta_recurrence_fwd,
)

NV = NK = 16
KD = VD = 128
KEY_DIM = NK * KD
VALUE_DIM = NV * VD
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
KS = 4


def bf16_round(x: np.ndarray) -> np.ndarray:
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def to_buffer(x: np.ndarray, dtype: DType, dev) -> Buffer:
    if dtype == DType.bfloat16:
        bits = (bf16_round(x).view(np.uint32) >> 16).astype(np.uint16)
        return Buffer.from_numpy(bits).view(DType.bfloat16, x.shape).to(dev)
    return Buffer.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)


def from_buffer(b: Buffer, dtype: DType) -> np.ndarray:
    host = b.to(CPU())
    if dtype == DType.bfloat16:
        bits = host.view(DType.uint16, host.shape).to_numpy()
        return (bits.astype(np.uint32) << 16).view(np.float32)
    return host.to_numpy()


def reference(x, w, window, state, decay, beta, offsets, slots, silu=False):
    """Conv (with SiLU when `silu`, as the op's flag) then the recurrence, in
    float64."""
    x = x.astype(np.float64)
    conv = np.empty_like(x)
    window = window.astype(np.float64).copy()
    state = state.astype(np.float64).copy()
    out = np.empty((x.shape[0], VALUE_DIM))
    for b in range(len(slots)):
        s, e, slot = offsets[b], offsets[b + 1], slots[b]
        full = np.concatenate([window[slot].T, x[s:e]])  # [KS-1+T, C]
        for t in range(e - s):
            conv[s + t] = (w.T * full[t : t + KS]).sum(axis=0)
        window[slot] = full[-(KS - 1) :].T
        if silu:
            conv[s:e] = conv[s:e] / (1.0 + np.exp(-conv[s:e]))
        for t in range(s, e):
            row = conv[t]
            for h in range(NV):
                kh = h // (NV // NK)
                q = row[kh * KD : (kh + 1) * KD]
                k = row[KEY_DIM + kh * KD : KEY_DIM + (kh + 1) * KD]
                v = row[2 * KEY_DIM + h * VD : 2 * KEY_DIM + (h + 1) * VD]
                q = q / np.sqrt((q * q).sum() + 1e-6) / np.sqrt(KD)
                k = k / np.sqrt((k * k).sum() + 1e-6)
                st = state[slot, h] * decay[t, h]  # [KD, VD]
                delta = beta[t, h] * (v - k @ st)
                st = st + np.outer(k, delta)
                state[slot, h] = st
                out[t, h * VD : (h + 1) * VD] = q @ st
    return conv, out, window, state


def run_case(
    session, dev, lengths, pool_dtype, rng, fused=False
) -> tuple[bool, str]:
    """`fused`: the conv takes a bfloat16 input, wider than its channels, and
    applies SiLU (Qwen3.5's Intel path), its float32 output feeding the
    recurrence."""
    total = sum(lengths)
    nb = len(lengths)
    slots_n = nb + 1
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.uint32)
    slots = np.arange(nb, dtype=np.uint32)[::-1].copy()  # not the identity
    # The fused conv reads the QKV columns of a wider projection in place.
    width = CONV_DIM + (160 if fused else 0)
    x = rng.uniform(-1, 1, (total, width)).astype(np.float32)
    w = rng.uniform(-0.5, 0.5, (CONV_DIM, KS)).astype(np.float32)
    window = rng.uniform(-1, 1, (slots_n, CONV_DIM, KS - 1)).astype(np.float32)
    state = rng.uniform(-0.1, 0.1, (slots_n, NV, KD, VD)).astype(np.float32)
    decay = rng.uniform(0.55, 0.99, (total, NV)).astype(np.float32)
    beta = rng.uniform(0.1, 0.9, (total, NV)).astype(np.float32)
    if pool_dtype == DType.bfloat16:
        window = bf16_round(window)
        state = bf16_round(state)
    in_dtype = DType.bfloat16 if fused else DType.float32
    if fused:
        x = bf16_round(x)

    gpu = DeviceRef.GPU()
    with Graph(
        "gated_delta",
        input_types=[
            TensorType(in_dtype, [total, width], gpu),
            TensorType(DType.float32, [CONV_DIM, KS], gpu),
            BufferType(pool_dtype, [slots_n, CONV_DIM, KS - 1], gpu),
            BufferType(pool_dtype, [slots_n, NV, KD, VD], gpu),
            TensorType(DType.float32, [total, NV], gpu),
            TensorType(DType.float32, [total, NV], gpu),
            TensorType(DType.uint32, [nb], gpu),
            TensorType(DType.uint32, [nb + 1], gpu),
        ],
    ) as g:
        xi, wi, wpool, spool, di, bi, si, oi = g.inputs
        conv = gated_delta_conv1d_fwd(
            xi.tensor, wi.tensor, wpool.buffer, si.tensor, oi.tensor, silu=fused
        )
        out = gated_delta_recurrence_fwd(
            conv, di.tensor, bi.tensor, spool.buffer, si.tensor, oi.tensor
        )
        g.output(conv, out)
    model = session.load(g)
    args = [
        to_buffer(x, in_dtype, dev),
        Buffer.from_numpy(w).to(dev),
        to_buffer(window, pool_dtype, dev),
        to_buffer(state, pool_dtype, dev),
        Buffer.from_numpy(decay).to(dev),
        Buffer.from_numpy(beta).to(dev),
        Buffer.from_numpy(slots).to(dev),
        Buffer.from_numpy(offsets).to(dev),
    ]
    conv_g, out_g = model.execute(*args)
    conv_g = conv_g.to(CPU()).to_numpy()
    out_g = out_g.to(CPU()).to_numpy()
    window_g = from_buffer(args[2], pool_dtype)
    state_g = from_buffer(args[3], pool_dtype)
    conv_r, out_r, window_r, state_r = reference(
        x[:, :CONV_DIM],
        w,
        window,
        state,
        decay,
        beta,
        offsets,
        slots,
        silu=fused,
    )

    def rel(a, b):
        return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))

    errs = {
        "conv": rel(conv_g, conv_r),
        "out": rel(out_g, out_r),
        "state": rel(state_g, state_r),
        "window": float(np.abs(window_g - window_r).max()),
    }
    state_tol = 1e-5 if pool_dtype == DType.float32 else 4e-3
    ok = (
        errs["conv"] <= 1e-5
        and errs["out"] <= 1e-5
        and errs["state"] <= state_tol
        and errs["window"] <= (0 if pool_dtype == DType.float32 else 1e-2)
    )
    # Time executions (the pools keep evolving; decay < 1 keeps them bounded).
    for _ in range(3):
        model.execute(*args)
    dev.synchronize()
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        model.execute(*args)
        dev.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    msg = (
        f"lengths={lengths if len(lengths) < 4 else '...'} pool={pool_dtype}"
        + (" bf16 wide input, silu" if fused else "")
        + ": "
        + ", ".join(f"{k} {v:.2e}" for k, v in errs.items())
        + f"; graph {np.median(times):.1f} us per execution"
    )
    return ok, msg


def main() -> int:
    dev = Accelerator()
    session = InferenceSession(devices=[dev])
    rng = np.random.default_rng(5)
    bad = 0
    n = 0
    for lengths in (
        [1],
        [1, 1, 1, 1],
        [128],
        [512],
        [2048],
        [1, 300, 1],
        [3, 5, 2, 17, 1, 9],
    ):
        for pool_dtype in (DType.float32, DType.bfloat16):
            for fused in (False, True):
                ok, msg = run_case(
                    session, dev, lengths, pool_dtype, rng, fused
                )
                n += 1
                bad += not ok
                print(("PASS " if ok else "FAIL ") + msg, flush=True)
    print(f"RESULT {'PASS' if not bad else 'FAIL'}: {n - bad} of {n} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
