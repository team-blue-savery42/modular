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
# ===----------------------------------------------------------------------===//

"""Launches for timing the Intel FP4-weight matmul against bf16.

Run on the GPU host with the plugin's launch profile on:
  RUN_MAX_PY_OPT=1 RUN_MAX_PY_ENV=MOJO_INTEL_PROFILE=/tmp/fp4 \
    utils/run-max-py tests/graph/fp4_bench.py [M,N,K ...]

For each M x N x K (default Qwen3.5-4B's MLP: gate and up fused 18432 x
2560, down 2560 x 9216, at M = 1 and 512) builds one graph of `COPIES`
matmuls of the same activation against distinct weights (so the weights
stream from memory, not the cache) for bf16 weights (`x @ w.T`) and for
NVFP4 weights (`mo.matmul.weight.only.block.scaled.float4.intel`), and runs
each `RUNS` times. The device times are in the profile; print nothing but
the cases, in order, for the parser.
"""

from __future__ import annotations

import sys

import numpy as np
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn._graph_ops import _intel_weight_only_block_scaled_float4_matmul

COPIES = 8
RUNS = 20


def main() -> int:
    shapes = [tuple(int(v) for v in a.split(",")) for a in sys.argv[1:]] or [
        (1, 18432, 2560),
        (1, 2560, 9216),
        (512, 18432, 2560),
        (512, 2560, 9216),
    ]
    dev = Accelerator()
    session = InferenceSession(devices=[dev])
    gpu = DeviceRef.GPU()
    rng = np.random.default_rng(5)
    for m, n, k in shapes:
        x = (
            Buffer.from_numpy(
                rng.integers(0x3C00, 0x3F80, (m, k), dtype=np.uint16)
            )
            .view(DType.bfloat16, [m, k])
            .to(dev)
        )
        for kind in ("bf16", "fp4"):
            if kind == "bf16":
                wt = [TensorType(DType.bfloat16, [n, k], device=gpu)] * COPIES
            else:
                wt = [
                    TensorType(DType.uint8, [n, k // 2], device=gpu),
                    TensorType(DType.float8_e4m3fn, [n, k // 16], device=gpu),
                ] * COPIES
            types = [TensorType(DType.bfloat16, [m, k], device=gpu)] + wt
            if kind == "fp4":
                types.append(
                    TensorType(DType.float32, [], device=DeviceRef.CPU())
                )
            with Graph(f"bench_{kind}_{m}_{n}_{k}", input_types=types) as g:
                vals = [v.tensor for v in g.inputs]
                outs = []
                for c in range(COPIES):
                    if kind == "bf16":
                        outs.append(vals[0] @ vals[1 + c].T)
                    else:
                        outs.append(
                            _intel_weight_only_block_scaled_float4_matmul(
                                vals[0],
                                vals[1 + 2 * c],
                                vals[2 + 2 * c],
                                vals[-1],
                            )
                        )
                g.output(*outs)
            model = session.load(g)
            args = [x]
            for _ in range(COPIES):
                if kind == "bf16":
                    args.append(
                        Buffer.from_numpy(
                            rng.integers(
                                0x3C00, 0x3F80, (n, k), dtype=np.uint16
                            )
                        )
                        .view(DType.bfloat16, [n, k])
                        .to(dev)
                    )
                else:
                    args.append(
                        Buffer.from_numpy(
                            rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
                        ).to(dev)
                    )
                    args.append(
                        Buffer.from_numpy(
                            rng.integers(
                                0x30, 0x40, (n, k // 16), dtype=np.uint8
                            )
                        )
                        .view(DType.float8_e4m3fn, [n, k // 16])
                        .to(dev)
                    )
            if kind == "fp4":
                args.append(Buffer.from_numpy(np.array(1.0, dtype=np.float32)))
            for _ in range(RUNS):
                model.execute(*args)
            dev.synchronize()
            print(
                f"CASE {kind} {m} {n} {k} copies {COPIES} runs {RUNS}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
