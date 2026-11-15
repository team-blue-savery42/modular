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

# Intel targets and the stdlib plugin: a kernel compiled for the B70
# target from MAX's Intel target table, and for a bare `spirv64` triple with no
# plugin payload, must see `is_gpu()` true and come out as a SPIR-V kernel.
# Run with utils/check-targets.

from std.compile import compile_info
from std.sys.info import CompilationTarget, is_gpu

from max._plugin.intel.targets import ArcProB70

comptime _bare_spirv = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "spirv64-unknown-unknown", arch = "bmg-g31", `,
        `data_layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1", `,
        `index_bit_width = 64, simd_bit_width = 128> : !kgen.target`,
    ]
]()


def mark(p: Pointer[Int32, MutAnyOrigin]):
    comptime if is_gpu():
        p[0] = 1111
    else:
        p[0] = 2222


def main() raises:
    comptime b70 = CompilationTarget[_mlir_value=ArcProB70.mlir_target]()
    with open("out/b70.spv", "w") as f:
        f.write(compile_info[mark, target=b70, emission_kind="object"]().asm)
    with open("out/b70.ll", "w") as f:
        f.write(compile_info[mark, target=b70, emission_kind="llvm-opt"]().asm)
    with open("out/bare.ll", "w") as f:
        f.write(
            compile_info[
                mark, target=_bare_spirv, emission_kind="llvm-opt"
            ]().asm
        )
