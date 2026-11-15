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

# MAX warp and block primitives on the B70: compiled for the Intel
# target, then run by utils/l0/run.sh as 256 work-items in groups of 64
# (warps of 16). Every work-item takes part in each collective, so warps stay
# converged; chosen lanes store the results. Run with utils/check-device.

from std.compile import compile_info
from std.sys.info import CompilationTarget

from max._plugin.intel.targets import ArcProB70
from max.gpu import block_idx, global_idx, lane_id, thread_idx
from max.gpu.primitives import block, warp


def warps(p: Pointer[Int32, MutAnyOrigin]):
    var gid = global_idx.x
    var lane = Int(lane_id())
    var g = Int32(gid)

    var lane_sum = warp.sum(Int32(lane))
    var from3 = warp.shuffle_idx(g, 3)
    var down = warp.shuffle_down(g, 1)
    var xor4 = warp.shuffle_xor(g, 4)
    var even = warp.vote[DType.uint32](lane % 2 == 0)
    var scan = warp.prefix_sum(Int32(1))
    var top = warp.max(Float32(lane))
    var count = block.sum[block_size=64](Int32(1))
    var groups = warp.match_any(UInt32(lane % 4))
    var half = warp.shuffle_idx(Float64(gid) * 0.5, 7)
    var first = warp.broadcast(BFloat16(Float32(lane + 1)))
    var block_scan = block.prefix_sum[block_size=64](Int32(1))

    if gid == 0:
        p[1] = lane_sum  # 0 + 1 + ... + 15 = 120
        p[2] = from3  # lane 3's id: 3
        p[5] = Int32(even)  # lanes 0, 2, ..., 14: 0x5555 = 21845
        p[7] = Int32(top)  # 15
        p[8] = count  # 64
        p[9] = Int32(groups)  # lanes 0, 4, 8, 12: 0x1111 = 4369
        p[10] = Int32(half * 10)  # lane 7's 3.5: 35
    if gid == 15:
        p[3] = down  # the warp's last lane keeps its own: 15
        p[6] = scan  # inclusive scan of 16 ones: 16
    if gid == 1:
        p[4] = xor4  # lane 1 ^ 4: 5
    if gid == 9:
        p[11] = Int32(Float32(first))  # lane 0's 1.0: 1
    if gid == 37:
        p[12] = Int32(lane)  # 37 % 16: 5
    if block_idx.x == 0 and thread_idx.x == 63:
        p[13] = block_scan  # inclusive scan of 64 ones: 64


def main() raises:
    comptime b70 = CompilationTarget[_mlir_value=ArcProB70.mlir_target]()
    with open("out/warps.spv", "w") as f:
        f.write(compile_info[warps, target=b70, emission_kind="object"]().asm)
