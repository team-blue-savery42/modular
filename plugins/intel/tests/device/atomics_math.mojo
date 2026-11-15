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

# Atomics and math on the B70: compiled for the Intel target, then run
# by utils/l0/run.sh as 256 work-items in groups of 64 over a zeroed buffer
# whose first word is 7. Run with utils/check-device, which checks the words.

from std.atomic import Atomic
from std.compile import compile_info
from std.math import exp, log, sqrt, tanh
from std.memory import AddressSpace, stack_allocation
from std.sys.info import CompilationTarget

from max._plugin.intel.targets import ArcProB70
from max.gpu import block_idx, global_idx, thread_idx
from max.gpu.sync import barrier


def atomics_math(p: Pointer[Int32, MutAnyOrigin]):
    var i = global_idx.x
    var f = p.unsafe_bitcast[Float32]()

    # p[1]: float32 atomic add, 256.0 (bits 0x43800000).
    _ = Atomic.fetch_add(f + 1, 1.0)
    # p[2]: int32 atomic max, 255.
    Atomic.max(p + 2, Int32(i))
    # p[3]: device-scope ("agent") atomic add, 256.
    _ = Atomic[Int32, scope="agent"].fetch_add(p + 3, 1)
    # p[4]: float32 atomic max, 255.0 (bits 0x437F0000).
    Atomic.max(f + 4, Float32(i))
    # p[5]: per-block counts in shared memory with work-group-scope atomics,
    # summed into global memory, 256.
    var count = stack_allocation[
        1, DType.int32, address_space=AddressSpace.SHARED
    ]()
    if thread_idx.x == 0:
        count[0] = 0
    barrier()
    _ = Atomic[Int32, scope="workgroup"].fetch_add(count, 1)
    barrier()
    if thread_idx.x == 0:
        _ = Atomic.fetch_add(p + 5, count[0])

    if i == 0:
        # bfloat16 sqrt(2) = 1.4140625, computed in float32: 1414.
        p[6] = Int32(Float32(sqrt(BFloat16(2.0))) * 1000)
        # float64 log(10) = 2.302585093: 2302585.
        p[7] = Int32(log(Float64(10.0)) * 1e6)
        # float32 tanh(0.5) = 0.46211716: 462117.
        p[8] = Int32(tanh(Float32(0.5)) * 1e6)
        # float16 exp(1): the stdlib's GPU exp is exp2(x * log2e) in float16
        # on every GPU, and log2e rounded to float16 gives 2.716797, one
        # float16 below the correctly rounded 2.71875: 2716.
        p[9] = Int32(Float32(exp(Float16(1.0))) * 1000)
        # float8_e4m3fn round trip through bfloat16 of 1.75: 175.
        var e4m3 = BFloat16(1.75).cast[DType.float8_e4m3fn]()
        p[10] = Int32(e4m3.cast[DType.float32]() * 100)
        # int32 to bfloat16 at run time of 2^24 + 2^16 + 1, just above halfway
        # between two bfloat16 values: rounds once, up, to 2^24 + 2^17 =
        # 16908288 (through float32 to nearest it would round twice, to 2^24).
        var n = Int32(16842746) + p[0]
        p[11] = Int32(n.cast[DType.bfloat16]().cast[DType.float32]())


def main() raises:
    comptime b70 = CompilationTarget[_mlir_value=ArcProB70.mlir_target]()
    with open("out/atomics_math.spv", "w") as f:
        f.write(
            compile_info[atomics_math, target=b70, emission_kind="object"]().asm
        )
