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

"""MAX's static FP8 linear layer on the B70, against NumPy.

Run on the GPU host: utils/run-max-py tests/graph/fp8_linear.py

Builds graphs with `max.nn.legacy.quant_ops.quantized_matmul` and the
per-tensor static FP8 config Qwen3.5's checkpoints use (modelopt "FP8",
`FBGEMM_FP8`, CPU scalar scales), as `Linear` calls it, so on Intel it takes
the weight-only path (`mo.matmul.weight.only.scaled.float8.intel`). Checks
that the graph holds that op and no activation quantization, and that the
bf16 output is within 2^-7 of sum|x w| (times the scale) of a float64 NumPy
reference from the same codes, at Qwen3.5-0.8B's FP8 shapes for decode (M =
1, 4, 16: the skinny kernel) and prefill (M = 17, 512: widen and GEMM).
"""

from __future__ import annotations

import sys

import numpy as np
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn._graph_ops.quant import quantized_matmul
from max.nn.legacy.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

CONFIG = QuantConfig(
    input_scale=InputScaleSpec(
        granularity=ScaleGranularity.TENSOR,
        origin=ScaleOrigin.STATIC,
        dtype=DType.float32,
    ),
    weight_scale=WeightScaleSpec(
        granularity=ScaleGranularity.TENSOR, dtype=DType.float32
    ),
    mlp_quantized_layers=set(),
    attn_quantized_layers={0},
    embedding_output_dtype=DType.bfloat16,
    bias_dtype=None,
    format=QuantFormat.FBGEMM_FP8,
)


def e4m3(codes: np.ndarray) -> np.ndarray:
    """float8_e4m3fn values of uint8 codes (no NaNs), from the definition."""
    c = codes.astype(np.int64)
    e = (c >> 3) & 0xF
    m = (c & 7).astype(np.float64)
    v = np.where(e == 0, m / 8 * 2.0**-6, (1 + m / 8) * 2.0 ** (e - 7))
    return np.where(c & 0x80, -v, v)


def bf16_bits(x: np.ndarray) -> np.ndarray:
    bits = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    return (((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF).astype(
        np.uint16
    )


def main() -> int:
    rng = np.random.default_rng(10)
    dev = Accelerator()
    session = InferenceSession(devices=[dev])
    gpu = DeviceRef.GPU()
    fails = 0
    for m, n, k in [
        (1, 8192, 1024),
        (4, 1024, 2048),
        (16, 5120, 1024),
        (17, 8192, 1024),
        (512, 1024, 2048),
    ]:
        types = [
            TensorType(DType.bfloat16, [m, k], device=gpu),
            TensorType(DType.float8_e4m3fn, [n, k], device=gpu),
            TensorType(DType.float32, [], device=DeviceRef.CPU()),
            TensorType(DType.float32, [], device=DeviceRef.CPU()),
        ]
        with Graph(f"fp8_linear_{m}_{n}_{k}", input_types=types) as g:
            x, w, ws, xs = (v.tensor for v in g.inputs)
            g.output(quantized_matmul(x, w, ws, xs, CONFIG))
        ir = str(g)
        op_ok = (
            "mo.matmul.weight.only.scaled.float8.intel" in ir
            and "quantize_static_scaled_float8" not in ir
        )
        model = session.load(g)

        xb = bf16_bits(rng.uniform(-1, 1, (m, k)))
        codes = rng.integers(0, 256, (n, k), dtype=np.uint8)
        codes[(codes & 0x7F) == 0x7F] = 0
        scale = np.float32(0.0123)
        x_buf = Buffer.from_numpy(xb).view(DType.bfloat16, [m, k]).to(dev)
        w_buf = (
            Buffer.from_numpy(codes).view(DType.float8_e4m3fn, [n, k]).to(dev)
        )
        (out,) = model.execute(
            x_buf,
            w_buf,
            Buffer.from_numpy(np.array(scale, dtype=np.float32)),
            Buffer.from_numpy(np.array(1.0, dtype=np.float32)),
        )
        bits = out.to(CPU()).view(DType.uint16, [m, n]).to_numpy()
        got = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        xf = (xb.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        wf = e4m3(codes)
        ref = xf @ wf.T * float(scale)
        mag = np.abs(xf) @ np.abs(wf).T * float(scale)
        err = float(np.max(np.abs(got - ref) / mag))
        ok = op_ok and err <= 2.0**-7
        fails += not ok
        print(
            f"{'PASS' if ok else 'FAIL'} fp8 linear {m}x{n}x{k}: Intel "
            f"weight-only op {'used' if op_ok else 'NOT used'}, worst "
            f"|got - ref| / sum|x w| {err:.2e} (tol {2.0**-7:.2e})",
            flush=True,
        )
    print(f"fp8_linear: failures {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
