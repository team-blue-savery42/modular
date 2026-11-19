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

"""MAX_INTEL_FP8_BF16_WEIGHTS: a bf16 `max.nn.Linear` on the B70 runs as a
per-tensor FP8 weight, against NumPy.

Run on the GPU host: utils/run-max-py tests/graph/fp8_weight_matmul.py

Compiles `max.nn.Linear` (bf16, no bias) with the option on and checks its
bf16 output against a float64 NumPy reference of the same quantization:
scale = amax / 448, each weight / scale rounded to nearest-even
float8_e4m3fn (`encode_e4m3`, checked here against every code), times the
scale. Tolerance 2^-7 of sum|x w|. Shapes: Gemma-4-26B-A4B's projections
(8192 x 2816, 2816 x 4096, 4224 x 2816, 2816 x 2112) at M = 1, 7 and 300,
and the smallest the kernels take (40 x 64). A weight the FP8 kernels don't
take (N = 36) runs as bf16: it must match the bf16 reference.
"""

from __future__ import annotations

import os
import sys

os.environ["MAX_INTEL_FP8_BF16_WEIGHTS"] = "1"

import numpy as np
from max import functional as F
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.nn import Linear
from max.tensor.tensor import Tensor, TensorType, default_device, default_dtype


def bf16_round(x: np.ndarray) -> np.ndarray:
    b = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF
    return (r.astype(np.uint32) << 16).view(np.float32)


def to_bf16_buffer(x: np.ndarray) -> Buffer:
    bits = (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return Buffer.from_numpy(bits).view(DType.bfloat16, list(x.shape))


def e4m3_values() -> np.ndarray:
    c = np.arange(0x7F)
    e, m = (c >> 3) & 0xF, (c & 7).astype(np.float64)
    return np.where(e == 0, m / 8 * 2.0**-6, (1 + m / 8) * 2.0 ** (e - 7.0))


def round_e4m3(x: np.ndarray) -> np.ndarray:
    """Nearest float8_e4m3fn value, ties to the even code (brute force)."""
    vals = e4m3_values()
    a = np.abs(x.astype(np.float64)).reshape(-1)
    idx = np.searchsorted(vals, a)
    lo = np.clip(idx - 1, 0, len(vals) - 1)
    hi = np.clip(idx, 0, len(vals) - 1)
    dlo, dhi = a - vals[lo], vals[hi] - a
    pick = np.where(
        dlo < dhi, lo, np.where(dhi < dlo, hi, np.where(lo % 2 == 0, lo, hi))
    )
    return (vals[pick] * np.sign(x.reshape(-1))).reshape(x.shape)


def main() -> int:
    dev = Accelerator()
    rng = np.random.default_rng(15)
    fails = 0
    cases = [
        (n, k, m)
        for n, k in [(8192, 2816), (2816, 4096), (4224, 2816), (2816, 2112)]
        for m in (1, 7, 300)
    ] + [(40, 64, 3), (36, 64, 3)]
    for n, k, m in cases:
        # Lazily: eager initialization would run in MAX's compile pool.
        with F.lazy(), default_dtype(DType.bfloat16), default_device(dev):
            layer = Linear(k, n, bias=False)
        w = bf16_round(rng.normal(0, 0.02, (n, k)))
        w[0, 0] = 0.25  # an outlier sets the scale
        x = bf16_round(rng.uniform(-1, 1, (m, k)))
        run = layer.compile(
            TensorType(DType.bfloat16, [m, k], dev),
            weights={"weight": to_bf16_buffer(w).to(dev)},
        )
        out = run(Tensor.from_dlpack(to_bf16_buffer(x).to(dev)))
        bits = (
            Buffer.from_dlpack(out)
            .to(CPU())
            .view(DType.uint16, [m, n])
            .to_numpy()
        )
        got = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        fp8 = n % 8 == 0 and n >= 32
        if fp8:
            scale = np.float32(np.abs(w).max() / np.float32(448.0))
            wq = round_e4m3((w / scale).astype(np.float32)) * float(scale)
        else:
            wq = w.astype(np.float64)
        ref = x.astype(np.float64) @ wq.T
        mag = np.abs(x.astype(np.float64)) @ np.abs(wq).T
        err = float(np.max(np.abs(got - ref) / np.maximum(mag, 1e-30)))
        ok = err <= 2.0**-7
        fails += not ok
        print(
            f"{'PASS' if ok else 'FAIL'} {'fp8' if fp8 else 'bf16'} linear"
            f" {m}x{n}x{k}: worst |got - ref| / sum|x w| {err:.2e}",
            flush=True,
        )
    print(f"fp8_weight_matmul: failures {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
