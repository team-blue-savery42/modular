# Performance notes

Where the time goes on the GPUs these plugins were measured on, how to
measure it, and what is left to try, ordered by the measured payoff.
Engine-against-engine results are in `benchmarks.md`. All
numbers are from an Intel Arc Pro B70 (`bmg-g31`, Xe2, 32 GB) unless a
section names an iGPU; opt build, BF16, greedy decoding.

## How to measure

- Kernel times: `MOJO_INTEL_PROFILE=<file>` makes the driver plugin write
  every kernel, copy, fill, synchronize and graph launch with host and
  device timestamps. `utils/bench/steps.py PROFILE MARKS` turns it into a
  per-step breakdown by kernel; `tests/graph/bench_model.py --marks` writes
  the step marks.
- Attention in isolation: `tests/e2e/attention_bench.mojo` (through
  `utils/run-e2e`). The shape comes from `-D NQ`, `-D NKV`, `-D HD` and
  `-D PAGE` (Qwen3-8B: 32, 8, 128, 128; Qwen3.5-0.8B: 8, 2, 256, 256), the
  prompt lengths from `-D PREFILL="..."`, the decode cache lengths from
  `-D DECODE="..."`, and `-D BATCH=n` adds a decode case of n sequences.
  Every case is checked against a float64 reference.
- Models against other engines: `benchmarks.md` has the configurations,
  scenarios, results and the `utils/bench` commands that reproduce them;
  `utils/bench/soak.sh` runs an hour of mixed serving.
- Warm-up: from idle, the clock ramps up for 75-100 ms, holds until about
  250 ms of sustained load, then the card reaches its 230 W power limit
  (`pl2` in the throttle reasons) and settles within about 1 s. A
  4096 x 4096 x 4096 GEMM runs at 140-147 TFLOP/s before the limit and
  about 117 after it. Time sustained runs past the first second, or report
  burst and sustained figures separately; a DPAS-only loop settles near
  2.45 GHz, a GEMM at 1.83-2.0 GHz.
- Compiler output: `MOJO_INTEL_PLUGIN_DUMP=<dir>` writes each module as
  bitcode (`llvm-dis` prints it) and as SPIR-V. `ocloc compile -spirv_input
  -file K.spv -device bmg-g31` with `IGC_ShaderDumpEnable=1` and
  `IGC_DumpToCustomDir=<dir>` writes IGC's ISA, which shows the SIMD width,
  the register file mode (128 or 256 GRF) and spills. Use a fresh
  `MODULAR_CACHE_DIR`, or a cached kernel dumps nothing.
  `MOJO_INTEL_IGC_OPTIONS` passes options to IGC at module creation (for
  example `-cl-opt-disable` to separate an IGC optimizer issue).
- Not available on this stack: EU stall sampling (`unitrace
  --stall-sampling` reports the metric group `EuStallSampling` missing).
  `xpu-smi` during attention prefill read EU active about 40%, EU stall
  about 58%, memory read 4.3 GB/s, power at the 230 W cap.

## Arc Pro B70 (Xe2)

### FA prefill

Prefill attention runs at 45.3 TFLOP/s at 4,096 tokens and 47.3 TFLOP/s
at 16,384 (Qwen3-8B's shape, causal), 37.9 and 39.0 TFLOP/s at head
dimension 256 (Qwen3.5-0.8B), against a DPAS peak of about 185 TFLOP/s.
Removing the exponentials, one of the two DPAS streams, or three quarters
of the K or V loads each changed the time by less than 2%, so the kernel is
bound by instruction issue and load latency per thread (4 threads per EU at
256 GRF), not by DPAS, exp or bandwidth. TTFT of a
32,768-token prompt is 0.67x of vLLM-XPU on Qwen3-8B (10.96 s against
7.37 s) and attention's share of a step grows with the square of the
prompt length.

- 16 query rows per sub-group (two DPAS row groups sharing each K and V
  fragment): 8.4 TFLOP/s as written, spilling 11,328 bytes at 256 GRF; the
  128 float accumulator registers of the O tiles plus IGC hoisting the
  loads do not fit. It needs K and V shared through SLM,
  or a way to keep IGC from hoisting the loads.
- Head dimension 256 keeps 4 rows per sub-group: 8 rows spill (8.7 TFLOP/s
  with the current load pipeline). Splitting the head dimension over two
  sub-groups is the untried route.
- Causal load balance: query blocks are launched in 64-query units, and the
  work-group of a head ends with its slowest sub-group.
- `attention_bench` hangs at 32,768 tokens (both shapes; not diagnosed).
  Servers run 32k prompts.
- Tried without gain: 64-key steps (4,651 us against 3,029 at 4,096
  tokens), 16-key steps (3,158), work-groups of 16 sub-groups (3,024), K
  loaded one step ahead (equal to unrolled in place).

### GDN recurrence (Qwen3.5)

The prefill recurrence is 30.3 ms of a 105.8 ms 4,096-token step of
Qwen3.5-0.8B (18 layers, 1,684 us each). With its global loads replaced
by constants a layer takes 1,338 us, so about 3 ms of the step is load
latency; the rest is the sequential token loop, bound by instructions per
token. The next step is a chunked (WY) formulation that puts the chunk's
work on DPAS. The conv1d before it takes 8.1 ms of the step (451 us a
layer).

### Fusions

In a 4,096-token step of Qwen3-8B (608.6 ms), SwiGLU takes 20.8 ms and
RMSNorm 17.5 ms, as graph-compiler kernels at about 270-520 GB/s. Fusing
them into the GEMM epilogues or into one kernel each is the next largest
item after attention.

### GEMMs

Dense GEMMs at Qwen3-8B's 4,096-token shapes run at 133-152 TFLOP/s, 77-82%
of the 185 TFLOP/s DPAS peak (4,096 x 6,144 x 4,096 the lowest), and are
69% of that step. Decode GEMMs at 64 sequences read weights at 565 GB/s of
about 600. Both are near their ceilings.

### Serving

Qwen3-8B serving at 64 concurrent random 512/128 requests is decode-step
bound: 1,024 steps of about 44 ms plus the prompts; the ITL
is 43.9 ms against vLLM-XPU's 44.2. Median TTFT is 2.9 s
against vLLM-XPU's 0.8 s. MAX's scheduler options did not move throughput
(`--max-batch-input-tokens` 1,024-4,096: 744-746 tok/s); in-flight batching
with 2,048-token batches gives 752 tok/s and a median TTFT
of 0.88 s.

### MoE

gpt-oss-20b (MXFP4 experts) does not load: MAX's MXFP4 MoE path has a
native grouped GEMM only for SM100 and otherwise dequantizes every expert
on each forward behind a fixed 15 GiB reservation. It needs a weight-only
MXFP4 grouped GEMM for Intel, and attention with sinks, which the Intel
FA declines (MAX's naive kernel runs them).

### Sub-group size

Kernels run at sub-group size 16; the `-sg32` targets run every kernel
at size 32. DPAS is defined for sub-group sizes 8 and 16 only, and `ocloc`
crashed on DPAS kernels at 32, so the DPAS kernels stay at 16. Measured at
both sizes: `block.sum` over 1,024 work-items takes 0.24x the time at 32,
softmax 0.75-0.82x at 4,096 x 4,096 and 32,768 x 512 but 1.10x at
262,144 x 64, copies and reads are equal. A per-kernel choice is open for
the reductions.

### Device graphs

The graph pool keeps memory that graph launches reuse until the last graph
is destroyed, including each capture's output buffers: serving Qwen3.5-0.8B
captures 64 decode graphs whose outputs were 19 distinct blocks, 564 MiB.
That memory is not available to eager allocations while graphs live.

## iGPUs (Xe-LP `rpl-p`, Xe-LPG `arl-s`)

No DPAS and no 2D block I/O: the Intel GEMM and FA are not
taken, matmuls run the shared-memory SIMT GEMM, attention runs MAX's naive
kernels. Qwen3.5-0.8B, 512-token prompt in one step: arl-s 1.16 s (441
tok/s), rpl-p 1.54 s (333 tok/s); decode arl-s 30.8 tok/s, rpl-p 19.0
tok/s.

- Decode is bound by memory bandwidth once device graphs are used.
  After a 32-token prompt: arl-s 32.7 ms a token without graphs, 23.7 ms
  with them (42.3 tok/s); rpl-p 50.8 and 47.7 ms. Without graphs arl-s
  issues the step's 407 launches in 5.0 ms, but the GPU idles 7.2 ms
  between its kernels (about 18 us each); with graphs the idle time is
  2.6 ms and the GPU is busy 20.6 ms of a 23.8 ms step. The LM-head GEMV
  reads 509 MB at 89 GB/s on arl-s and 38 GB/s on rpl-p, the measured
  memory bandwidths. The CPU core at 100% during decode is the host
  waiting for the GPU. `tests/graph/bench_model.py` runs without graphs
  unless given `--graph-capture`; `max serve` captures them by default.
- The SIMT GEMM is 878 ms of the 1.16 s arl-s prompt step, about 700
  GFLOP/s, 35% of the computed float32 peak (64 EUs x 8 lanes x 2 x
  1.95 GHz). Tile sweep (GEMM time per step): 4 x 4 accumulators with K
  slices of 16, 878 ms (kept); K slices of 32, 945; 2 x 4, 1,135; 8 x 4,
  1,375; 4 x 8, 1,658. Its ISA has not been inspected.
- MAX's naive attention (`mha_bmm0`, `mha_bmm1`) is 172 ms, 15% of the
  512-token step, and grows with the square of the prompt; the Gated
  DeltaNet recurrence is 59 ms (5%).
- Limits: allocations over `maxMemAllocSize` (4 GiB) fail with
  `ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION` on compute-runtime 26.40 with
  i915 even with the relaxed-limits request (MAX's sampling test's
  4,096-row case); a model run needs none. Xe-LP has no float64 and
  work-groups of at most 512. The FP8 weight-only op is Xe2 only; MAX's
  generic FP8 path runs instead, with errors of 5.0e-3 to 9.4e-3 in
  `tests/graph/fp8_linear.py` against its 7.8e-3 tolerance.
- Rounding: in the optimized build float64 -> BF16 conversions differ
  from the host in 66,036 of 196,608 cases (the B70's optimized build
  shows the same, its debug build none); Xe-LPG's IGC keeps denormal
  float32 inputs that the host flushes (1,528 of 393,216).
