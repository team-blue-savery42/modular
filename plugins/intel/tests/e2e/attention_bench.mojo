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

# run-e2e: device-context
# run-e2e: packages //max/kernels/src/nn //max/kernels/src/kv_cache //max:layout
#
# MAX's ragged paged attention (`flash_attention[ragged=True]`, the
# kernel behind `mo.mha.ragged.paged`) on the B70 at Qwen3.5-0.8B's shapes:
# 8 query heads over 2 KV heads, head dim 256, bf16, page size 256, causal
# mask, scale 1/16. On Intel it dispatches to the DPAS kernels
# (`nn/attention/gpu/intel`), or with MODULAR_ENABLE_INTEL_FA=0 to MAX's
# naive three-kernel path.
# Correctness: every case against a float64 host attention over the same
# bf16 inputs (all rows up to 256 query rows, else every 5th row and the
# last ones): max |gpu - ref|, which must be <= 2e-2 (bf16 outputs of
# magnitude <= 1; MAX's attention tests use atol 1e-2 against a bf16
# reference).
# Timed (launches back to back, one synchronize, batches >= 100 ms,
# a warm-up batch, the median of 5).
#   utils/run-e2e attention_bench [-D CHECK=0] [-D TIME=0] [-D ONLY=decode|prefill|quick]
# Not in check-e2e: a measurement.

from std.math import ceildiv, exp, sqrt
from std.os.env import getenv
from std.random import rand, seed
from std.sys.defines import (
    get_defined_bool,
    get_defined_int,
    get_defined_string,
)
from std.time import perf_counter_ns, sleep

from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from max.gpu.host import DeviceContext
from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import CausalMask

comptime CHECK = get_defined_bool["CHECK", True]()
comptime TIME = get_defined_bool["TIME", True]()
comptime ONLY = get_defined_string["ONLY", ""]()
# -D NQ / NKV / HD / PAGE set the shape (Qwen3-8B is 32 / 8 / 128 /
# 128); -D PREFILL="512 4096" the prefill lengths (as a space list).
comptime NQ = get_defined_int["NQ", 8]()
comptime NKV = get_defined_int["NKV", 2]()
comptime HD = get_defined_int["HD", 256]()
comptime PAGE = get_defined_int["PAGE", 256]()
comptime PREFILL = get_defined_string["PREFILL", "128 512 2048"]()
# -D DECODE="511 4095" sets the decode cache lengths; -D BATCH=64 adds a
# decode case of that many sequences at the last length.
comptime DECODE = get_defined_string["DECODE", "511 1023 2047 4095"]()
comptime BATCH = get_defined_int["BATCH", 1]()
comptime bf16 = DType.bfloat16
comptime kv_params = KVCacheStaticParams(num_heads=NKV, head_size=HD)


def median(var v: List[Float64]) -> Float64:
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            if v[j] < v[i]:
                var t = v[i]
                v[i] = v[j]
                v[j] = t
    return v[len(v) // 2]


def run_case(
    ctx: DeviceContext,
    label: String,
    prompt_lens: List[Int],
    cache_lens: List[Int],
) raises -> Bool:
    var B = len(prompt_lens)
    var T = 0
    var max_prompt = 0
    var max_ctx = 0
    var pages = 0
    for i in range(B):
        T += prompt_lens[i]
        max_prompt = max(max_prompt, prompt_lens[i])
        max_ctx = max(max_ctx, prompt_lens[i] + cache_lens[i])
        pages += ceildiv(prompt_lens[i] + cache_lens[i], PAGE)
    var lut_cols = ((ceildiv(max_ctx, PAGE) + 7) // 8) * 8 + 16

    var q = HostDeviceTileTensor[bf16](row_major(T, Idx[NQ], Idx[HD]), ctx)
    var out = HostDeviceTileTensor[bf16](row_major(T, Idx[NQ], Idx[HD]), ctx)
    var offsets = HostDeviceTileTensor[DType.uint32](
        row_major(Int64(B + 1)), ctx
    )
    var cache_lengths = HostDeviceTileTensor[DType.uint32](
        row_major(Int64(B)), ctx
    )
    var blocks = HostDeviceTileTensor[bf16](
        row_major(
            Int64(pages + 1),
            Idx[2],
            Int64(1),
            Idx[PAGE],
            Idx[NKV],
            Idx[HD],
        ),
        ctx,
    )
    var lut = HostDeviceTileTensor[DType.uint32](
        row_major(Int64(B), Int64(lut_cols)), ctx
    )
    var qh = q.host_tensor()
    var bh = blocks.host_tensor()
    rand[bf16](qh.unsafe_ptr(), T * NQ * HD, min=-1.0, max=1.0)
    rand[bf16](
        bh.unsafe_ptr(), (pages + 1) * 2 * PAGE * NKV * HD, min=-1.0, max=1.0
    )
    var oh = offsets.host_tensor()
    var ch = cache_lengths.host_tensor()
    var lh = lut.host_tensor()
    var row = 0
    var page = 0
    for b in range(B):
        oh[b] = UInt32(row)
        ch[b] = UInt32(cache_lens[b])
        row += prompt_lens[b]
        for c in range(lut_cols):
            lh[b, c] = 0
        # Pages in reverse order, so that a sequence's pages aren't adjacent.
        for p in range(ceildiv(prompt_lens[b] + cache_lens[b], PAGE)):
            lh[b, p] = UInt32(pages - page)
            page += 1
    oh[B] = UInt32(row)
    q.to_device()
    blocks.to_device()
    offsets.to_device()
    cache_lengths.to_device()
    lut.to_device()

    comptime Collection = PagedKVCacheCollection[
        bf16,
        kv_params,
        PAGE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var coll = Collection(
        blocks.device_tensor().as_unsafe_any_origin(),
        cache_lengths.device_tensor().as_imm().as_unsafe_any_origin(),
        lut.device_tensor().as_imm().as_unsafe_any_origin(),
        UInt32(max_prompt),
        UInt32(max_ctx),
    )
    var kc = coll.get_key_cache(0)
    var vc = coll.get_value_cache(0)
    var qd = q.device_tensor()
    var od = out.device_tensor()
    var offd = offsets.device_tensor()
    var scale = Float32(1.0 / sqrt(Float64(HD)))

    @inline(.always)
    def launch() raises {
        var kc, var vc, var qd, var od, var offd, var scale, ref ctx
    }:
        flash_attention[ragged=True](
            od, qd, kc, vc, CausalMask(), offd, scale, ctx
        )

    var ok = True
    comptime if CHECK:
        launch()
        out.to_host()
        var ohost = out.host_tensor()
        var worst = Float64(0)
        var checked = 0
        var scores = List[Float64](length=max_ctx, fill=0)
        for b in range(B):
            var P = prompt_lens[b]
            var C = cache_lens[b]
            for i in range(P):
                if P > 256 and i % 5 != 0 and i < P - 8:
                    continue
                var tok = Int(oh[b]) + i
                var pos = C + i
                for h in range(NQ):
                    var kvh = h // (NQ // NKV)
                    var mx = Float64(-1e300)
                    for key in range(pos + 1):
                        var pg = Int(lh[b, key // PAGE])
                        var s = Float64(0)
                        for d in range(HD):
                            s += Float64(qh[tok, h, d]) * Float64(
                                bh[pg, 0, 0, key % PAGE, kvh, d]
                            )
                        s *= Float64(scale)
                        scores[key] = s
                        mx = max(mx, s)
                    var den = Float64(0)
                    for key in range(pos + 1):
                        scores[key] = exp(scores[key] - mx)
                        den += scores[key]
                    for d in range(HD):
                        var acc = Float64(0)
                        for key in range(pos + 1):
                            var pg = Int(lh[b, key // PAGE])
                            acc += scores[key] * Float64(
                                bh[pg, 1, 0, key % PAGE, kvh, d]
                            )
                        var err = abs(Float64(ohost[tok, h, d]) - acc / den)
                        if not (err <= worst):  # NaN too
                            worst = err
                    checked += 1
        ok = worst <= 2e-2
        print(
            "PASS" if ok else "FAIL",
            label,
            "prompts",
            String(prompt_lens),
            "cached",
            String(cache_lens),
            ": max |gpu - ref|",
            worst,
            "over",
            checked,
            "(row, head) pairs",
        )
    comptime if TIME:
        # Compile (JIT) and run once before the calibration, so that its
        # launch count isn't sized by the compile.
        launch()
        ctx.synchronize()
        var t0 = perf_counter_ns()
        var n = 0
        while perf_counter_ns() - t0 < 20_000_000:
            launch()
            n += 1
        ctx.synchronize()
        n = max(
            10, Int(Float64(n) * 0.1e9 / Float64(perf_counter_ns() - t0)) + 1
        )
        sleep(3.0)
        for _ in range(n):
            launch()
        ctx.synchronize()
        var times = List[Float64]()
        for _ in range(5):
            var t1 = perf_counter_ns()
            for _ in range(n):
                launch()
            ctx.synchronize()
            times.append(Float64(perf_counter_ns() - t1) / Float64(n) / 1e3)
        var us = median(times^)
        # Causal FLOPs (QK^T and PV) and the K/V bytes read once.
        var flops = Float64(0)
        var kv_bytes = Float64(0)
        for b in range(B):
            var P = prompt_lens[b]
            var C = cache_lens[b]
            for i in range(P):
                flops += 4.0 * Float64(NQ * HD) * Float64(C + i + 1)
            kv_bytes += Float64(2 * 2 * NKV * HD * (C + P))
        print(
            "TIME",
            label,
            "T",
            T,
            "keys",
            max_ctx,
            ":",
            us,
            "us per op (n =",
            n,
            "),",
            flops / us / 1e6,
            "TFLOP/s,",
            kv_bytes / us / 1e3,
            "GB/s of K/V",
        )
    _ = q^
    _ = out^
    _ = offsets^
    _ = cache_lengths^
    _ = blocks^
    _ = lut^
    return ok


def main() raises:
    seed(6)
    var ctx = DeviceContext()
    print("MODULAR_ENABLE_INTEL_FA =", getenv("MODULAR_ENABLE_INTEL_FA", "1"))
    var fails = 0
    comptime if ONLY == "quick":
        if not run_case(ctx, "decode", [1], [511]):
            fails += 1
        if not run_case(ctx, "prefill", [512], [0]):
            fails += 1
    comptime if ONLY == "" or ONLY == "decode":
        var last = 0
        for w in DECODE.split(" "):
            last = Int(w)
            if not run_case(ctx, "decode", [1], [last]):
                fails += 1
        if BATCH > 1:
            var ones = List[Int]()
            var caches = List[Int]()
            for _ in range(BATCH):
                ones.append(1)
                caches.append(last)
            if not run_case(ctx, "decode batch of many", ones, caches):
                fails += 1
        comptime if CHECK:
            if not run_case(ctx, "decode batch", [1, 1, 1], [300, 0, 255]):
                fails += 1
    comptime if ONLY == "" or ONLY == "prefill":
        for w in PREFILL.split(" "):
            var t = Int(w)
            if not run_case(ctx, "prefill", [t], [0]):
                fails += 1
        comptime if CHECK:
            if not run_case(ctx, "ragged", [33, 1, 300], [0, 511, 5]):
                fails += 1
    print("PASS" if fails == 0 else "FAIL", ":", fails, "failing case(s)")
    if fails:
        raise Error("attention_bench: failures")
