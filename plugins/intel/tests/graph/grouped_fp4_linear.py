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

"""The Intel weight-only NVFP4 grouped (MoE) matmul on the B70, against NumPy.

Run on the GPU host: utils/run-max-py tests/graph/grouped_fp4_linear.py

Builds graphs with `mo.grouped.matmul.weight.only.block.scaled.float4.intel`
(bf16 A with rows grouped by expert, each expert's B^T as packed E2M1 codes
with float8_e4m3fn block scales per 16 inputs, a float32 scale per expert,
`moe_create_indices`-style group offsets and expert ids) and checks each
group's bf16 rows within 2^-7 of sum|x w| of a float64 NumPy reference.
Shapes are Gemma-4-26B-A4B's experts (gate and up fused 1408 x 2816, down
2816 x 704) with 16 experts in a shuffled group order: decode (one token,
8 of 16 groups one row each: the 8-row tiles), a few tokens, and prefill
with empty groups and groups longer than a tile; and the smallest shape the
kernel takes (K = 128, N = 40).
"""

from __future__ import annotations

import sys

import numpy as np
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn._graph_ops import intel_grouped_weight_only_nvfp4_matmul

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
    rng = np.random.default_rng(14)
    dev = Accelerator()
    session = InferenceSession(devices=[dev])
    gpu = DeviceRef.GPU()
    experts = 16
    fails = 0
    cases = [
        ("decode", 1408, 2816, [1] * 8 + [0] * 8),
        ("decode down", 2816, 704, [1] * 8 + [0] * 8),
        (
            "4 tokens",
            1408,
            2816,
            [2, 4, 0, 1, 3, 2, 0, 4, 2, 1, 2, 3, 2, 2, 2, 2],
        ),
        (
            "prefill",
            1408,
            2816,
            [100, 0, 33, 17, 16, 1, 0, 64, 15, 40, 9, 0, 70, 50, 47, 50],
        ),
        (
            "prefill down",
            2816,
            704,
            [100, 0, 33, 17, 16, 1, 0, 64, 15, 40, 9, 0, 70, 50, 47, 50],
        ),
        ("smallest", 40, 128, [3, 0, 20, 1] + [0] * 12),
    ]
    for name, n, k, counts in cases:
        m = sum(counts)
        types = [
            TensorType(DType.bfloat16, [m, k], device=gpu),
            TensorType(DType.uint8, [experts, n, k // 2], device=gpu),
            TensorType(DType.float8_e4m3fn, [experts, n, k // 16], device=gpu),
            TensorType(DType.uint32, [experts + 1], device=gpu),
            TensorType(DType.int32, [experts], device=gpu),
            TensorType(DType.float32, [experts], device=gpu),
        ]
        with Graph(f"grouped_fp4_{m}_{n}_{k}", input_types=types) as g:
            g.output(
                intel_grouped_weight_only_nvfp4_matmul(
                    *(v.tensor for v in g.inputs)
                )
            )
        model = session.load(g)

        xb = bf16_bits(rng.uniform(-1, 1, (m, k)))
        codes = rng.integers(0, 16, (experts, n, k), dtype=np.uint8)
        packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)
        scodes = rng.integers(0x20, 0x60, (experts, n, k // 16), dtype=np.uint8)
        scodes[rng.random(scodes.shape) < 0.25] |= 0x80
        ws2 = rng.uniform(0.005, 0.02, experts).astype(np.float32)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
        ids = rng.permutation(experts).astype(np.int32)
        (out,) = model.execute(
            Buffer.from_numpy(xb).view(DType.bfloat16, [m, k]).to(dev),
            Buffer.from_numpy(packed).to(dev),
            Buffer.from_numpy(scodes)
            .view(DType.float8_e4m3fn, [experts, n, k // 16])
            .to(dev),
            Buffer.from_numpy(offsets).to(dev),
            Buffer.from_numpy(ids).to(dev),
            Buffer.from_numpy(ws2).to(dev),
        )
        bits = out.to(CPU()).view(DType.uint16, [m, n]).to_numpy()
        got = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        xf = (xb.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        err = 0.0
        for z in range(experts):
            r0, r1 = int(offsets[z]), int(offsets[z + 1])
            if r0 == r1:
                continue
            e = int(ids[z])
            wf = E2M1[codes[e]] * np.repeat(e4m3_abs(scodes[e]), 16, axis=1)
            ref = xf[r0:r1] @ wf.T * float(ws2[e])
            mag = np.abs(xf[r0:r1]) @ np.abs(wf).T * float(ws2[e])
            err = max(
                err,
                float(
                    np.max(np.abs(got[r0:r1] - ref) / np.maximum(mag, 1e-30))
                ),
            )
        ok = err <= 2.0**-7
        fails += not ok
        print(
            f"{'PASS' if ok else 'FAIL'} grouped fp4 {name} {m}x{n}x{k}:"
            f" worst |got - ref| / sum|x w| {err:.2e} (tol {2.0**-7:.2e})",
            flush=True,
        )
    print(f"grouped_fp4_linear: failures {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
