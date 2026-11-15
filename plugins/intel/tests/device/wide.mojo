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

# Vectors wider than SPIR-V's 16 elements on the B70: compiled for the
# Intel target, then run by utils/l0/run.sh as one work-item over a zeroed
# buffer whose first word is 7. Run with utils/check-device.

from std.compile import compile_info
from std.sys.info import CompilationTarget

from max._plugin.intel.targets import ArcProB70


def wide(p: Pointer[Int32, MutAnyOrigin]):
    var seven = p[0]
    var v = SIMD[DType.int32, 64]()
    for i in range(64):
        v[i] = Int32(i)
    # 2i + 7 in each of 64 lanes.
    var w = v * 2 + seven
    # A run-time index: lane 7, 2 * 7 + 7 = 21.
    p[1] = w[Int(seven)]
    # A 32-wide float splat multiply, 1.5 * 7 = 10.5: 105.
    var f = SIMD[DType.float32, 32](1.5) * Float32(seven)
    p[2] = Int32(f[31] * 10)
    # A 32-wide conversion to half: 10.5 is exact: 105.
    var h = f.cast[DType.float16]()
    p[3] = Int32(Float32(h[3]) * 10)
    # A 64-wide store: p[8 + i] = 2i + 7.
    (p + 8).unsafe_store(0, w)


def main() raises:
    comptime b70 = CompilationTarget[_mlir_value=ArcProB70.mlir_target]()
    with open("out/wide.spv", "w") as f:
        f.write(compile_info[wide, target=b70, emission_kind="object"]().asm)
