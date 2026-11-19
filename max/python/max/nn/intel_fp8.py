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
"""Opt-in weight-only FP8 for bf16 weights on Intel Xe2.

With ``MAX_INTEL_FP8_BF16_WEIGHTS=1`` on an Intel Xe2 GPU,
:func:`weight_matmul` runs ``x @ weight.T`` for a bf16 weight as a per-tensor
float8_e4m3fn weight (scale amax / 448) with the Intel weight-only FP8
kernel, which reads half the bytes of the bf16 one. The quantization is
graph ops on the weight alone, which MAX folds when it loads the model, so
it runs once and each matmul reads the FP8 copy. It applies to any model
whose bf16 projections go through :func:`weight_matmul` (``Linear``, the
common ``MLP``). It changes the model's numbers (FP8 weights instead of
bf16), which is why it is opt-in.
"""

from __future__ import annotations

import functools
import os

from max import functional as F
from max.dtype import DType
from max.graph import Dim
from max.nn._graph_ops import _is_intel_xe2_gpu
from max.nn.kernels import intel_weight_only_fp8_matmul
from max.tensor import Tensor

FP8_ENV = "MAX_INTEL_FP8_BF16_WEIGHTS"
"""Set to ``1`` to run bf16 weights as per-tensor FP8 on Intel Xe2."""

E4M3_MAX = 448.0


@functools.cache
def intel_fp8_enabled() -> bool:
    """Whether :data:`FP8_ENV` is ``1`` and the accelerator is Intel Xe2."""
    return os.environ.get(FP8_ENV) == "1" and _is_intel_xe2_gpu()


def _static(dim: Dim) -> int | None:
    try:
        return int(dim)
    except (TypeError, ValueError):
        return None


def _eligible(x: Tensor, weight: Tensor) -> bool:
    """Whether :func:`weight_matmul` runs ``weight`` as FP8.

    That is: the option is on, the weight is a bf16 ``[N, K]`` matrix on one
    device with static dims the Intel FP8 kernels take (N a multiple of 8,
    at least 32; K a multiple of 16, at least 64), and ``x`` is bf16.
    """
    if not intel_fp8_enabled():
        return False
    if weight.dtype != DType.bfloat16 or x.dtype != DType.bfloat16:
        return False
    if weight.rank != 2 or weight.is_distributed or x.is_distributed:
        return False
    n, k = _static(weight.shape[0]), _static(weight.shape[1])
    return (
        n is not None
        and k is not None
        and n % 8 == 0
        and n >= 32
        and k % 16 == 0
        and k >= 64
    )


def quantize_per_tensor(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Splits ``weight`` into float8_e4m3fn values and one float32 scale.

    The scale is amax / 448 (at least 1e-30, so an all-zero weight gives
    zeros), shape ``[1]``; the values are ``weight / scale`` rounded to the
    nearest float8_e4m3fn.
    """
    w = weight.cast(DType.float32)
    amax = F.max(F.abs(w).reshape([-1]), axis=0)
    tiny = F.constant(1e-30, DType.float32, device=w.device)
    scale = F.max(amax / E4M3_MAX, tiny)
    return (w / scale).cast(DType.float8_e4m3fn), scale


def weight_matmul(x: Tensor, weight: Tensor) -> Tensor:
    """Returns ``x @ weight.T`` for a weight in Linear layout (``[N, K]``).

    It runs as per-tensor FP8 on Intel Xe2 when :data:`FP8_ENV` is set and
    the shapes allow (:func:`_eligible`), else as the plain matmul.
    """
    if not _eligible(x, weight):
        return x @ weight.T
    q, scale = quantize_per_tensor(weight)
    k = weight.shape[1]
    rows = x.reshape([-1, k])
    out = intel_weight_only_fp8_matmul(rows, q, scale)
    return out.reshape([*x.shape[:-1], weight.shape[0]])
