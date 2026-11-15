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

# Device-side printing and math on the B70: compiled for the Intel
# target, then run as one work-item by utils/l0/run.sh with a buffer holding 7.
# Run with utils/check-device, which checks the output below.

from std.compile import compile_info
from std.io.io import _printf
from std.math import exp
from std.sys.info import CompilationTarget

from max._plugin.intel.targets import ArcProB70


def io_math(p: Pointer[Int32, MutAnyOrigin]):
    var v = p[0]
    # printf with a runtime value and a string literal for %s.
    _printf["printf: %s v=%d\n"](StaticString("lit").as_c_string_span(), v)
    # print() formats on the device and writes a runtime buffer.
    print("print: v =", v, "with a tail longer than eight bytes")
    # exp(7) = 1096.63...
    p[0] = Int32(exp(Float32(v)))


def main() raises:
    comptime b70 = CompilationTarget[_mlir_value=ArcProB70.mlir_target]()
    with open("out/io_math.spv", "w") as f:
        f.write(compile_info[io_math, target=b70, emission_kind="object"]().asm)
