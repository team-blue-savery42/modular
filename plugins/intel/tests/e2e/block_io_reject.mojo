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

# Intel 2D block I/O refuses at compile time what the client
# extension (`cl_intel_subgroup_2d_block_io`) doesn't define. check-e2e runs
# this program twice and expects each to fail with the module's message:
#
#   -D CASE=shape  a 16-bit store of 16 rows (stores have 1 to 8 rows).
#   -D CASE=sg32   a valid load in a kernel built for the B70's sub-group
#                  size 32 variant (the client extension is for
#                  sub-group size 16).

from std.compile import compile_info
from std.sys.defines import get_defined_string

from max.gpu import lane_id
from max.gpu.memory.block_io_intel import (
    block_elements,
    block_load,
    block_store,
)
from max._plugin.intel.targets import ArcProB70, ArcProB70SG32

comptime CASE = get_defined_string["CASE", "shape"]()


def store_16_rows(m: MutPointer[UInt16, MutAnyOrigin]):
    var v = SIMD[DType.uint16, block_elements[16, 16]()](UInt16(lane_id()))
    block_store[16, 16](m, 64, 64, 64, 0, 0, v)


def load_at_32(m: MutPointer[UInt16, MutAnyOrigin]):
    var v = block_load[16, 8](m, 64, 64, 64, 0, 0)
    m[unsafe_offset=Int(lane_id())] = v[0]


def main() raises:
    comptime if CASE == "sg32":
        var info = compile_info[
            load_at_32, target=ArcProB70SG32.target, emission_kind="object"
        ]()
        print(
            "compiled for sub-group size 32:", info.asm.byte_length(), "bytes"
        )
    else:
        var info = compile_info[
            store_16_rows, target=ArcProB70.target, emission_kind="object"
        ]()
        print("compiled a 16-row store:", info.asm.byte_length(), "bytes")
