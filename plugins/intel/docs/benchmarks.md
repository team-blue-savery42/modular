# Benchmarks

MAX with these plugins against vLLM-XPU and llama.cpp (SYCL and Vulkan) on
an Intel Arc Pro B70, and MAX alone on two iGPUs. A blank or "not run"
cell was not measured. Higher is better for tok/s, lower for times.

## Setup

| Field            | Arc Pro B70                                                     | Arrow Lake-S iGPU (`arl-s`)   | Raptor Lake-P iGPU (`rpl-p`)  |
|------------------|-----------------------------------------------------------------|-------------------------------|-------------------------------|
| GPU              | Battlemage G31, Xe2, 32 GB                                      | Xe-LPG, 64 EUs, shared memory | Xe-LP, 80 EUs, shared memory  |
| Kernel driver    | `xe`, Linux 7.2                                                 | `i915`                        | `i915`                        |
| Compute runtime  | intel-compute-runtime 26.35.39758, IGC 2.41.5                   | intel-compute-runtime 26.40   | intel-compute-runtime 26.40   |
| Level Zero       | loader 1.32.0                                                   | not recorded                  | not recorded                  |
| Vulkan           | Mesa 26.2.5 (llama.cpp Vulkan only)                             | not used                      | not used                      |
| Memory bandwidth | -                                                               | 81-86 GB/s copy (measured)    | 37 GB/s copy (measured)       |
| Engines measured | MAX, vLLM-XPU, llama.cpp SYCL, llama.cpp Vulkan                 | MAX only                      | MAX only                      |

Other engines were not measured on the iGPUs.

| Engine           | Version                                                                                      | Settings                                                                                                                                                                                                                                                                                                                                                     |
|------------------|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MAX              | this series, opt build, HAL runtime (`--//:hal_device_context=true`)                         | device graphs: on where the architecture captures (Qwen3.5), Qwen3-8B runs eager; serving: `--max-batch-size 64`, `--max-length 8192`, `--max-batch-input-tokens` and device memory utilization (KV fraction) at their defaults, prefix caching off; long prompts: `--max-batch-input-tokens 8192`, prefix caching on; Qwen3-8B with `--no-prefer-module-v3` |
| vLLM-XPU         | `intel/vllm:0.21.0-xpu` (vLLM 0.21.1.dev18+g8df6feb7d, torch 2.11.0+xpu)                     | eager unless "XPU graphs"; `--gpu-memory-utilization 0.9`, `--max-num-seqs 64`, `--max-model-len 8192`, `--max-num-batched-tokens` default, prefix caching off for serving; long prompts: `--max-num-batched-tokens 8192`, prefix caching on                                                                                                                 |
| llama.cpp SYCL   | `ba0ba54`, oneAPI 2026.1 (`ghcr.io/snailium/llama.cpp-sycl-intel-b70/llama-sycl-b70:stable`) | `llama-bench -ngl 99 -fa 1 -ub 2048 -b 2048 -r 3`                                                                                                                                                                                                                                                                                                            |
| llama.cpp Vulkan | `ba0ba54`                                                                                    | as SYCL                                                                                                                                                                                                                                                                                                                                                      |

Models: `Qwen/Qwen3-8B` (revision `b968826d`) and `Qwen/Qwen3.5-0.8B`, both
BF16 on every engine (llama.cpp: BF16 GGUF converted from the same
snapshot); greedy decoding. Every run had the GPU to itself.

Trees. MAX was measured on development commits whose code matches this
series at the points below; the hashes are of that development branch.

| Tree | Commit       | Relation to this series                                                                                                                           |
|------|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| A    | `938edf2f05` | same code as the series tip; formatting, comments and benchmark container names differ                                                            |
| B    | `5dd85d11f7` | without the last three attention prefill changes (lazy maximum, shared prefetch, loads ahead) and the graph-memory fix                            |
| C    | `bcc726c88f` | iGPUs: without the GDN conv1d and recurrence changes, the activation reserve and the graph-memory fix, which apply there but were not re-measured |

vLLM-XPU and llama.cpp don't use the plugins; their rows give the
measurement window instead: 11-08 (2026-11-08, the series' benchmark
scripts at `86f5724abe`) or 11-10 (2026-11-10, beside tree B).

## Offline, single stream, Arc Pro B70

pp*N*: prompt processing of an *N*-token prompt in one engine step,
first-step tok/s (vLLM's 32k prompt is `generate(max_tokens=1)`, which
chunks it). tg @ *N*: decode of 128 tokens at batch 1 after an *N*-token
prompt, tok/s. Three repetitions.

Qwen3.5-0.8B:

| Engine               | Tree / window | pp512  | pp4096 | pp32768    | tg @ 512 | tg @ 4096 |
|----------------------|---------------|--------|--------|------------|----------|-----------|
| MAX                  | A (B)         | 32,466 | 40,313 | 17,502 (B) | 273.4    | 262.2     |
| vLLM-XPU, XPU graphs | 11-08         | 27,948 | 47,422 | 32,895     | 254.6    | 220.6     |
| vLLM-XPU, eager      | 11-08         | 27,818 | 47,203 | 33,330     | 74.5     | 71.7      |
| llama.cpp SYCL       | 11-08         | 9,288  | 9,825  | 7,873      | 205.0    | 193.9     |
| llama.cpp Vulkan     | 11-08         | 8,539  | 7,486  | 2,332      | 193.1    | 181.1     |

Qwen3-8B:

| Engine                  | Tree / window | pp512 | pp4096 | pp32768   | tg @ 512 | tg @ 4096 |
|-------------------------|---------------|-------|--------|-----------|----------|-----------|
| MAX                     | A (B)         | 7,782 | 6,840  | 2,519 (B) | 36.0     | 34.6      |
| vLLM-XPU, eager         | 11-08         | 8,075 | 7,267  | 4,456     | 35.8     | 30.3      |
| llama.cpp SYCL          | 11-08         | 995   | 1,124  | 727       | 31.9     | 29.7      |
| llama.cpp Vulkan        | 11-08         | 2,046 | 1,280  | 284       | 34.4     | 32.3      |

MAX's pp32768 cells are from tree B (`--max-batch-input-tokens 32768`),
the rest of its row from tree A. vLLM-XPU with XPU graphs was not run on
Qwen3-8B.

## Online serving, Arc Pro B70

`vllm bench serve` as the client for every server. Random: 512 input and
128 output tokens, EOS ignored. ShareGPT: ShareGPT v3 first turns and reply
lengths, output capped at 1,024. max(32, 8 x c) requests at concurrency c,
c warm-ups, seed 0, prefix caching off. Output in tok/s.

| Model        | Dataset  | Engine          | Tree / window | c = 1 | 4     | 16      | 32    | 64      |
|--------------|----------|-----------------|---------------|-------|-------|---------|-------|---------|
| Qwen3.5-0.8B | random   | MAX             | B             | 264.2 | 904.5 | 2,349.9 |       | 3,915.7 |
|              |          | MAX             | A             |       |       | 2,345   |       | 3,904   |
|              |          | vLLM-XPU, eager | 11-08         | 70.7  | 276.7 | 1,025.2 |       | 3,017.7 |
|              | ShareGPT | MAX             | A             |       |       |         |       | 5,547   |
| Qwen3-8B     | random   | MAX             | B             | 35.4  | 131.7 | 406.6   | 621.0 | 825.9   |
|              |          | MAX             | A             |       |       | 406.7   |       | 826.1   |
|              |          | vLLM-XPU, eager | 11-08, 11-10  | 34.8  | 131.6 | 412.1   | 630.4 | 861.6   |
|              | ShareGPT | MAX             | B             | 35.9  | 139.0 | 491.5   | 852.2 | 1,276.2 |
|              |          | MAX             | A             |       |       |         |       | 1,247   |
|              |          | vLLM-XPU, eager | 11-10         |       |       | 486.2   | 838.2 | 1,324.9 |

vLLM-XPU's Qwen3-8B random row: c = 1 and 4 from the 2026-11-08 window
(11-08), c = 16-64 from the 11-10 window. vLLM-XPU with XPU
graphs was not run for serving; it failed at four or more concurrent
requests on Qwen3.5-0.8B in an earlier window.

Latency, Qwen3-8B random at c = 64 (tree B): median ITL
MAX 43.9 ms, vLLM-XPU 44.2 ms; median TTFT MAX 2.9 s,
vLLM-XPU 0.8 s.

## Long prompts and prefix caching, Arc Pro B70

TTFT of one prompt at a time (token-id prompts, 3 requests
per cell, 64 output tokens, prefix caching on, MAX
`--max-batch-input-tokens 8192`; MAX's times are the same at 32,768),
and decode tok/s after the prompt. vLLM-XPU eager; Qwen3.5-0.8B on vLLM
without prefix caching (it fails with it, below).

| Model        | Engine   | Tree / window | 8k      | 16k     | 24k     | 32k     | decode after 8k / 16k / 24k / 32k |
|--------------|----------|---------------|---------|---------|---------|---------|-----------------------------------|
| Qwen3-8B     | MAX      | A             | 1.46 s  | 3.78 s  | 6.95 s  | 10.96 s | 33.5 / 31.3 / 29.5 / 27.8         |
|              | vLLM-XPU | 11-10         | 1.18 s  | 2.83 s  | 4.89 s  | 7.37 s  | 32.9 / 31.0 / 29.0 / 27.4         |
| Qwen3.5-0.8B | MAX      | A             | 0.247 s | 0.556 s | 0.962 s | 1.46 s  | 253 / 234 / 221 / 205             |
|              | vLLM-XPU | 11-10         | 0.18 s  | 0.41 s  | 0.67 s  | 0.99 s  | 74 / 75 / 75.5 / 75.5             |

Prefix-cache follow-ups: clients sharing a 24,576-token context send four
turns of 256 new tokens each; TTFT of the follow-up turns
(medians; the first request is cold). Tree B for MAX.

| Model        | Engine                     | cold   | 1 client    | 4 clients   | 8 clients   |
|--------------|----------------------------|--------|-------------|-------------|-------------|
| Qwen3-8B     | MAX                        | 8.19 s | 0.21 s      | 0.50-0.68 s | 0.69-1.33 s |
|              | vLLM-XPU                   | 5.01 s | 0.12 s      | 0.25-0.68 s | 0.66-0.76 s |
| Qwen3.5-0.8B | MAX                        | 1.24 s | 0.11 s      | 0.20-0.24 s | 0.29-0.39 s |
|              | MAX, prefix cache off      | 1.23 s | 1.23-1.28 s | 2.0-3.8 s   | 4.4-6.1 s   |
|              | vLLM-XPU, prefix cache off | 0.71 s | 0.70 s      | 1.1-2.2 s   | 1.4-3.6 s   |

On Qwen3.5-0.8B with MAX, the first turn of the clients after the first
reuses only part of the prefix: 1.27 s with four clients (16,384 cached),
1.9 s with eight (24,064 cached). vLLM-XPU 0.21 with prefix caching on
fails at the first Qwen3.5-0.8B request (`mamba_utils.preprocess_mamba`).
Every other request succeeded.

## iGPUs, MAX only

Qwen3.5-0.8B, BF16, tree C, overlap scheduler off. Prefill: a
512-token prompt in one step. Decode at batch 1, tok/s: 8 steps after the
512-token prompt; 32 steps x 2 after a 32-token prompt (Raptor Lake-P's
figures there from the median step time).

| GPU                   | pp512              | decode after 512, eager | decode after 32, eager | decode after 32, device graphs |
|-----------------------|--------------------|-------------------------|------------------------|--------------------------------|
| Arrow Lake-S (Xe-LPG) | 441 tok/s (1.16 s) | 30.8                    | 30.6                   | 42.3                           |
| Raptor Lake-P (Xe-LP) | 333 tok/s (1.54 s) | 19.0                    | 19.7                   | 21.0                           |

`max serve` (device graphs by default) answered single requests on
Arrow Lake-S at 36-38 tok/s end to end (600-token replies; plugin at
`189e3fa5f7`, before tree C). Greedy text is identical to the B70's on
both GPUs. The Raptor Lake-P laptop was neither
clock-pinned nor cooled; its figures are single cold runs.

## Reproducing

From `plugins/intel`, on a machine with the GPU to itself (stop other
render-node users first); `MODELS_DIR` holds the Hugging Face snapshots
and `gguf/` files, `DATASETS_DIR` the ShareGPT V3 JSON.

```sh
# Offline single stream, every engine (ENGINES="sycl vulkan max vllm").
sh utils/bench/single.sh OUT/single-qwen3 Qwen3-8B Qwen3-8B-BF16.gguf \
  --extra=--no-prefer-module-v3
VLLM_GRAPH=1 ENGINES=vllm sh utils/bench/single.sh OUT/vllm-graphs \
  Qwen3.5-0.8B Qwen3.5-0.8B-BF16.gguf
# Serving: ENGINE is max, vllm, llama (Vulkan) or sycl.
WORKLOADS="random sharegpt" MAX_SERVE_ARGS=--no-prefer-module-v3 \
  sh utils/bench/serve.sh OUT/serve-qwen3-max max Qwen3-8B \
  Qwen3-8B-BF16.gguf 1 4 16 32 64
# Long prompts and prefix-cache follow-ups (ENGINE=max or vllm).
ENGINE=max SERVE_ARGS=--no-prefer-module-v3 sh utils/bench/long.sh \
  OUT/long-qwen3-max Qwen3-8B
# Integrated GPUs: prefill, then decode with device graphs.
utils/run-max-py tests/graph/bench_model.py MODEL --prompt 512 --decode 8 \
  --no-overlap --extra=--max-batch-input-tokens=8192
utils/run-max-py tests/graph/bench_model.py MODEL --prompt 32 --decode 32 \
  --reps 2 --no-overlap --graph-capture
```

`utils/bench/steps.py` turns a `MOJO_INTEL_PROFILE` launch profile into a
per-step kernel breakdown; `docs/performance.md` uses it to explain where
the time goes.
