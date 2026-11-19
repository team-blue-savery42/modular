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

# ===----------------------------------------------------------------------=== #
# ===----------------------------------------------------------------------=== #
"""Intel GPU matmul kernels: DPAS with 2D block I/O on Xe2, a shared-memory
SIMT GEMM on the GPUs without DPAS."""

from .dispatch import (
    matmul_dispatch_intel,
    grouped_matmul_f4_dispatch_intel,
    matmul_f4_dispatch_intel,
    matmul_f8_dispatch_intel,
)
from .simt import (
    SIMT_THREADS,
    simt_grid,
    simt_kernel_fn,
    simt_supported,
)
from .matmul import (
    K_STEP,
    SKINNY_MAX_M,
    SKINNY_WIDE_MAX_M,
    SKINNY_SG_N,
    SMALL_WG_N,
    F8_UNIT,
    gemm_block,
    gemm_f8_supported,
    gemm_grid,
    gemm_kernel,
    F8_WIDEN_WORDS,
    f8_widen_kernel,
    gemm_kernel_f8_widened,
    gemm_kernel_fn,
    gemm_small_tiles,
    gemm_supported,
    skinny_block,
    skinny_grid,
    skinny_kernel,
    skinny_kernel_f8,
    skinny_kernel_fn,
    skinny_ks,
    skinny_prefetch,
    skinny_supported,
    skinny_wide_ks,
)
