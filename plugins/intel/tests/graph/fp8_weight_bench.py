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

"""Launches for timing bf16 weights against folded per-tensor FP8 ones
(max.nn.intel_fp8) on the B70.

Run on the GPU host with the plugin's launch profile on:
  RUN_MAX_PY_OPT=1 RUN_MAX_PY_ENV=MOJO_INTEL_PROFILE=/tmp/fp8 \\
    utils/run-max-py tests/graph/fp8_weight_bench.py

For each N x K (Gemma-4-26B-A4B's projections and 4096 x 4096) builds a
graph of `COPIES` matmuls at M (argv[1], default 1) against distinct bf16 weights (so they
stream from memory), once as `x @ w.T` and once through the per-tensor FP8
quantization (graph ops on the weight, which MAX folds at load) and the
Intel weight-only FP8 op, and runs each `RUNS` times (enough for the GPU's
clocks to settle). The device times are in the profile, in case order.
"""

import sys

import numpy as np
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Weight, ops
from max.nn._graph_ops import _intel_weight_only_scaled_float8_matmul

COPIES, RUNS = 8, 60
M = int(sys.argv[1]) if len(sys.argv) > 1 else 1
shapes = [
    (8192, 2816),
    (2816, 4096),
    (4224, 2816),
    (2816, 2112),
    (9216, 2816),
    (4096, 4096),
]
dev = Accelerator()
session = InferenceSession(devices=[dev])
gpu = DeviceRef.GPU()
rng = np.random.default_rng(0)


def bf16(a):
    return Buffer.from_numpy(
        (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    ).view(DType.bfloat16, list(a.shape))


for n, k in shapes:
    for kind in ("bf16", "fp8"):
        with Graph(
            f"p_{kind}_{n}_{k}",
            input_types=[TensorType(DType.bfloat16, [M, k], device=gpu)],
        ) as g:
            x = g.inputs[0].tensor
            outs = []
            for c in range(COPIES):
                w = Weight(f"w{c}", DType.bfloat16, [n, k], device=gpu)
                if kind == "bf16":
                    outs.append(x @ ops.transpose(w, 0, 1))
                else:
                    wf = ops.cast(w, DType.float32)
                    scale = ops.max(ops.abs(wf).reshape([-1]), axis=0) / 448.0
                    q = ops.cast(wf / scale, DType.float8_e4m3fn)
                    outs.append(
                        _intel_weight_only_scaled_float8_matmul(
                            x, q, scale.reshape([1])
                        )
                    )
            g.output(*outs)
        reg = {f"w{c}": bf16(rng.uniform(-1, 1, (n, k))) for c in range(COPIES)}
        model = session.load(g, weights_registry=reg)
        xb = bf16(rng.uniform(-1, 1, (M, k))).to(dev)
        for _ in range(RUNS):
            model.execute(xb)
        dev.synchronize()
        print(f"case {kind} {n}x{k}", flush=True)
