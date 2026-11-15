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

# DPAS needs sub-group size 16 (at 32 `ocloc` crashes), so an
# `mma()` kernel compiled for the B70's sub-group size 32 variant
# must fail to compile with `_mma_intel`'s message. check-e2e expects this
# program to fail, and the message once.

from std.compile import compile_info

from max.gpu import lane_id
from max.gpu.compute.mma import mma
from max._plugin.intel.targets import ArcProB70SG32


def mma_at_32(d_ptr: MutPointer[Float32, MutAnyOrigin]):
    var a = SIMD[DType.float16, 8](1)
    var b = SIMD[DType.float16, 16](1)
    var c = SIMD[DType.float32, 8](0)
    var d = SIMD[DType.float32, 8]()
    mma(d, a, b, c)
    d_ptr[unsafe_offset=Int(lane_id())] = d[0]


def main() raises:
    var info = compile_info[
        mma_at_32, target=ArcProB70SG32.target, emission_kind="object"
    ]()
    print("compiled for sub-group size 32:", info.asm.byte_length(), "bytes")
