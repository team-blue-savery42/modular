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

"""The Intel weight-only NVFP4 matmul on the B70, against NumPy.

Run on the GPU host: utils/run-max-py tests/graph/fp4_linear.py

Builds graphs with `mo.matmul.weight.only.block.scaled.float4.intel`
(bf16 A, B^T as packed E2M1 codes with float8_e4m3fn block scales per 16
inputs and a float32 per-tensor scale) and checks that the bf16 output is
within 2^-7 of sum|x w| of a float64 NumPy reference from the same codes and
scales: every E2M1 code, random positive E4M3 block scales (MAX's NVFP4
dequantization uses the scale's magnitude, so negative ones are tried too),
at Qwen3.5-4B's MLP shapes (gate and up fused 18432 x 2560, down 2560 x
9216) for decode (M = 1, 4, 16: the skinny kernel with 8 and 16 sub-groups)
and prefill (M = 17, 128, 512: widen and GEMM, both tile classes), and the
smallest shape the kernels take (K = 128, N = 40).
"""

from __future__ import annotations

import sys

import numpy as np
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn._graph_ops import _intel_weight_only_block_scaled_float4_matmul

E2M1 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float64,
)


def e4m3_abs(codes: np.ndarray) -> np.ndarray:
    """|float8_e4m3fn| of uint8 codes (no NaNs), from the definition."""
    c = codes.astype(np.int64)
    e = (c >> 3) & 0xF
    m = (c & 7).astype(np.float64)
    return np.where(e == 0, m / 8 * 2.0**-6, (1 + m / 8) * 2.0 ** (e - 7))


def bf16_bits(x: np.ndarray) -> np.ndarray:
    bits = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    return (((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF).astype(
        np.uint16
    )


def main() -> int:
    rng = np.random.default_rng(13)
    dev = Accelerator()
    session = InferenceSession(devices=[dev])
    gpu = DeviceRef.GPU()
    fails = 0
    for m, n, k in [
        (1, 18432, 2560),
        (4, 2560, 9216),
        (16, 18432, 2560),
        (1, 1024, 2560),
        (17, 2560, 9216),
        (128, 18432, 2560),
        (512, 2560, 9216),
        (3, 40, 128),
        (40, 40, 128),
    ]:
        types = [
            TensorType(DType.bfloat16, [m, k], device=gpu),
            TensorType(DType.uint8, [n, k // 2], device=gpu),
            TensorType(DType.float8_e4m3fn, [n, k // 16], device=gpu),
            TensorType(DType.float32, [], device=DeviceRef.CPU()),
        ]
        with Graph(f"fp4_linear_{m}_{n}_{k}", input_types=types) as g:
            x, w, s, ws2 = (v.tensor for v in g.inputs)
            g.output(
                _intel_weight_only_block_scaled_float4_matmul(x, w, s, ws2)
            )
        model = session.load(g)

        xb = bf16_bits(rng.uniform(-1, 1, (m, k)))
        codes = rng.integers(0, 16, (n, k), dtype=np.uint8)
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
        scodes = rng.integers(0x20, 0x60, (n, k // 16), dtype=np.uint8)
        scodes[rng.random((n, k // 16)) < 0.25] |= 0x80
        ws2 = np.float32(0.0123)
        (out,) = model.execute(
            Buffer.from_numpy(xb).view(DType.bfloat16, [m, k]).to(dev),
            Buffer.from_numpy(packed).to(dev),
            Buffer.from_numpy(scodes)
            .view(DType.float8_e4m3fn, [n, k // 16])
            .to(dev),
            Buffer.from_numpy(np.array(ws2, dtype=np.float32)),
        )
        bits = out.to(CPU()).view(DType.uint16, [m, n]).to_numpy()
        got = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        xf = (xb.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        wf = E2M1[codes] * np.repeat(e4m3_abs(scodes), 16, axis=1)
        ref = xf @ wf.T * float(ws2)
        mag = np.abs(xf) @ np.abs(wf).T * float(ws2)
        err = float(np.max(np.abs(got - ref) / np.maximum(mag, 1e-30)))
        ok = err <= 2.0**-7
        fails += not ok
        print(
            f"{'PASS' if ok else 'FAIL'} fp4 linear {m}x{n}x{k}: worst"
            f" |got - ref| / sum|x w| {err:.2e} (tol {2.0**-7:.2e})",
            flush=True,
        )
    print(f"fp4_linear: failures {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
