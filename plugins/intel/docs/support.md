# Support status

What runs on Intel GPUs today, what MAX offers that has not been run on them,
how MAX tracks vLLM-XPU on the B70, what functionality is missing on Intel, and
what common use cases still need. "Measured" means run on the hardware named;
everything else is marked. Numbers are in [benchmarks.md](benchmarks.md).

## Models

| Architecture (MAX) | Model run    | Weights                     | GPUs                             | Status                                                                     |
|--------------------|--------------|-----------------------------|----------------------------------|----------------------------------------------------------------------------|
| `qwen3`            | Qwen3-8B     | BF16                        | Arc Pro B70                      | measured: offline, serving up to 64 concurrent requests, 32k-token prompts |
| `qwen3_5`          | Qwen3.5-0.8B | BF16                        | B70, Arrow Lake-S, Raptor Lake-P | measured; the regression model of the checks                               |
| `qwen3_5`          | Qwen3.5-0.8B | FP8 (per-tensor, see below) | B70                              | measured, single stream                                                    |
| `gpt_oss`          | gpt-oss-20b  | MXFP4 experts               | B70                              | does not load (below)                                                      |
| any other          | -            | -                           | -                                | not run on Intel                                                           |

Qwen3-8B, 16.4 GB of BF16 weights, is the largest model run.
Qwen3-8B needs `--no-prefer-module-v3` (MAX's module-v3 lookup resolves
`Qwen3ForCausalLM` to its embedding pipeline).

## What fits in the B70's 32 GB

Weights alone, from parameter counts (estimates, not measured). The KV
cache, activations and the runtime need the rest; Qwen3-8B's BF16
KV cache is 144 KiB per token, 4.5 GB at 32k tokens.

| Dense model size | BF16     | FP8      | 4-bit    | On the B70 today                                |
|------------------|----------|----------|----------|-------------------------------------------------|
| 8 B              | 16 GB    | 8 GB     | 4-5 GB   | BF16 measured                                   |
| 14 B             | 29 GB    | 15 GB    | 8 GB     | BF16 leaves no room; FP8 fits on paper, not run |
| 27-32 B          | 54-64 GB | 27-33 GB | 15-18 GB | needs 4-bit, which has no Intel path yet        |
| 35 B MoE         | 70 GB    | 35 GB    | 18-20 GB | needs 4-bit and MoE kernels                     |

## Weight formats

| Format                            | MAX reads it for                                                  | On Intel                                                                                                                                                                                                                                             |
|-----------------------------------|-------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| BF16                              | every architecture                                                | measured; DPAS GEMM with 2D block loads (B70), shared-memory SIMT GEMM (iGPUs)                                                                                                                                                                       |
| float32                           | most architectures                                                | GPU runs cast it to BF16; not run as float32                                                                                                                                                                                                         |
| FP8 e4m3                          | `qwen3_5` (attention and GDN projections only), `qwen3`, `llama3` | per-tensor static scales (modelopt) run on a weight-only kernel: activations stay BF16 (Xe2's DPAS has no 8-bit float type); measured only on `qwen3_5`, with a Qwen3.5-0.8B checkpoint quantized locally, since no public one in this format exists |
| FP8 with block or channel scales  | depends on the architecture                                       | the Intel kernel takes one scale per tensor; not run                                                                                                                                                                                                 |
| FP4 (NVFP4, MXFP4)                | `qwen3_5`, `llama3`, `gpt_oss`                                    | no Intel kernel                                                                                                                                                                                                                                      |
| GGUF `q4_0`, `q4_k`, `q6_k`; GPTQ | `llama3` only                                                     | not run                                                                                                                                                                                                                                              |
| any 4-bit format for `qwen3`      | none                                                              | -                                                                                                                                                                                                                                                    |

gpt-oss-20b does not load: MAX's MXFP4 MoE path has a
native grouped GEMM only for NVIDIA SM100 and elsewhere dequantizes
every expert on each forward pass behind a fixed 15 GiB reservation,
which does not fit beside the KV cache in 32 GB.

## KV cache and context

| Feature                                      | Status                                                                                                                              |
|----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Paged KV cache, BF16                         | measured                                                                                                                            |
| FP8 KV cache (`kv_cache_format`)             | in MAX, not run on Intel                                                                                                            |
| Context length                               | 32,768-token prompts measured (`--max-length 40000`); serving measured at `--max-length 8192`                                       |
| Chunked prefill (`--max-batch-input-tokens`) | measured at 8,192 and 32,768                                                                                                        |
| Prefix caching                               | measured on both models; on Qwen3.5-0.8B, clients that start while the first is still filling a shared prefix reuse only part of it |
| Batch size                                   | measured up to `--max-batch-size 64`                                                                                                |
| Device memory                                | Level Zero reserves prompt activations by request count and layer type, so a single-step 32k prompt is accepted                     |

Attention prefill above 16k tokens is the slowest part relative to
vLLM-XPU: TTFT at 24k-32k tokens is 0.67-0.70x of it.

## Serving and runtime

| Feature                                            | Status                                                                                                |
|----------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `max serve`, OpenAI-compatible `/v1/completions`   | measured with `vllm bench serve` (random and ShareGPT prompts); chat completions not benchmarked      |
| Greedy decoding                                    | measured; greedy text matches the reference on every GPU                                              |
| Sampling (temperature, top-k, top-p)               | MAX's sampling tests pass on the B70; not tuned, not benchmarked                                      |
| Device graphs (Level Zero graph capture)           | on by default in `max serve`; Qwen3.5 decode captures, Qwen3-8B runs eager                            |
| Image input (`qwen3_5` accepts images)             | not run on Intel                                                                                      |
| Speculative decoding (MTP, draft models)           | not run end to end; 14 of 16 cases of MAX's GDN verify-ring test pass, one fails past a 32-bit offset |
| LoRA adapters, embedding models, structured output | not run on Intel                                                                                      |
| Multi-GPU, TP, peer access                         | not implemented                                                                                       |

## iGPUs

Arrow Lake-S (Xe-LPG) and Raptor Lake-P (Xe-LP) run Qwen3.5-0.8B in
BF16. Their IGC rejects DPAS and 2D block I/O, so they
use MAX's generic kernels and the shared-memory GEMM, not the Intel DPAS
GEMM, FA or the FP8 kernel. With the `i915` driver and
compute-runtime 26.40 a single allocation over 4 GiB fails. Xe-LP has no
float64.

## Tracking against vLLM-XPU

MAX against vLLM-XPU 0.21 on the B70, BF16. The last column is MAX
as a fraction of vLLM-XPU: throughput divided by vLLM-XPU's, or
vLLM-XPU's time divided by MAX's; above 1.00 means MAX performs better.
vLLM-XPU runs eager except in the Qwen3.5-0.8B single-stream rows, which
use its faster mode with XPU graphs. Settings and sources are in
[benchmarks.md](benchmarks.md).

| Scenario                                     | Model        | MAX          | vLLM-XPU     | vs vLLM |
|----------------------------------------------|--------------|--------------|--------------|---------|
| Prompt processing, 512 tokens                | Qwen3-8B     | 7,782 tok/s  | 8,075 tok/s  | 0.96    |
| Prompt processing, 4,096 tokens              | Qwen3-8B     | 6,840 tok/s  | 7,267 tok/s  | 0.94    |
| Prompt processing, 32,768 tokens             | Qwen3-8B     | 2,519 tok/s  | 4,456 tok/s  | 0.57    |
| Decode at batch 1, after 4,096 tokens        | Qwen3-8B     | 34.6 tok/s   | 30.3 tok/s   | 1.14    |
| Serving, random 512/128, 16 concurrent       | Qwen3-8B     | 406.6 tok/s  | 412.1 tok/s  | 0.99    |
| Serving, random 512/128, 64 concurrent       | Qwen3-8B     | 825.9 tok/s  | 861.6 tok/s  | 0.96    |
| Serving, ShareGPT, 64 concurrent             | Qwen3-8B     | 1,276 tok/s  | 1,325 tok/s  | 0.96    |
| ITL, 64 concurrent (median)                  | Qwen3-8B     | 43.9 ms      | 44.2 ms      | 1.01    |
| TTFT, 64 concurrent (median)                 | Qwen3-8B     | 2.9 s        | 0.8 s        | 0.28    |
| TTFT, 8k-token prompt                        | Qwen3-8B     | 1.46 s       | 1.18 s       | 0.81    |
| TTFT, 32k-token prompt                       | Qwen3-8B     | 10.96 s      | 7.37 s       | 0.67    |
| Follow-up turn on a cached 24k-token context | Qwen3-8B     | 0.21 s       | 0.12 s       | 0.57    |
| Prompt processing, 4,096 tokens              | Qwen3.5-0.8B | 40,313 tok/s | 47,422 tok/s | 0.85    |
| Prompt processing, 32,768 tokens             | Qwen3.5-0.8B | 17,502 tok/s | 33,330 tok/s | 0.53    |
| Decode at batch 1, after 4,096 tokens        | Qwen3.5-0.8B | 262.2 tok/s  | 220.6 tok/s  | 1.19    |
| Serving, random 512/128, 64 concurrent       | Qwen3.5-0.8B | 3,916 tok/s  | 3,018 tok/s  | 1.30    |
| TTFT, 32k-token prompt                       | Qwen3.5-0.8B | 1.46 s       | 0.99 s       | 0.68    |

MAX's 32,768-token prompt-processing, serving and follow-up figures come
from a tree without the last three attention prefill changes
(benchmarks.md, tree B). vLLM-XPU with XPU graphs failed at four or more
concurrent requests, so its serving rows are eager. It ran Qwen3.5-0.8B's
long prompts without prefix caching, which fails on it for this model.

## Missing functionality

What stops a model or feature from running on Intel at all, apart from
speed. "Elsewhere in MAX" is what MAX's tree has for other GPUs.

| Functionality                            | Elsewhere in MAX                                                                                             | Needed on Intel                                                                                                                                                                                                                                                                                          |
|------------------------------------------|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 4-bit dense weights                      | NVFP4 on NVIDIA SM100 (activations quantized too); weight-only FP4 on Apple M5; MXFP4 on AMD and NVIDIA SM90 | a weight-only FP4 GEMM and GEMV that widen the weights to BF16 in registers for DPAS (as the FP8 kernel does), for NVFP4 (FP8 scale per 16 values) and MXFP4 (e8m0 scale per 32), and the routing to it; NVFP4 layers on Intel now reach the path written for SM100, which has no Intel kernel (not run) |
| 4-bit MoE weights                        | 4-bit grouped expert GEMMs on NVIDIA and AMD; a naive grouped GEMM for any GPU, BF16 only                    | a weight-only 4-bit grouped expert GEMM on the same building blocks; MAX's generic top-k routing and token permutation have not been run on Intel                                                                                                                                                        |
| GGUF weights (`q4_0`, `q4_k`, `q6_k`)    | CPU kernels only, on every platform; read for `llama3` only                                                  | a GPU kernel for each format, and GGUF loading for the other architectures                                                                                                                                                                                                                               |
| Speculative decoding (MTP, draft models) | `mtp_qwen3_5` and other draft architectures                                                                  | a fix for the GDN verify ring past a 32-bit offset (one failing case of MAX's test), and a check that the Intel attention kernels accept several query tokens per sequence in a decode step                                                                                                              |
| Image input (`qwen3_5`, `gemma4`)        | supported                                                                                                    | probably no new kernel: the vision encoder is expected to run on MAX's generic kernels, its non-causal attention falling back from the Intel FA; not run                                                                                                                                                 |
| Contexts beyond 32k tokens               | up to each model's limit                                                                                     | a run at full length, and a check of 32-bit offsets and 2D block I/O size limits in the Intel kernels for large KV caches                                                                                                                                                                                |
| 2-bit weights                            | none                                                                                                         | not specific to Intel                                                                                                                                                                                                                                                                                    |
| Experts kept in host memory              | none                                                                                                         | not specific to Intel: a runtime feature MAX lacks                                                                                                                                                                                                                                                       |

A 4-bit kernel also needs a checkpoint in a format MAX loads (NVFP4 or
MXFP4 safetensors); whether one is published depends on the model.

## What is still needed for common use cases

| Use case                                                  | Today                                                                   | Still needed                                                                                                                                                         |
|-----------------------------------------------------------|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Chat or API serving of an 8 B model                       | runs; throughput 0.96-1.02 of vLLM-XPU                                  | TTFT under load (0.28 at 64 concurrent): in-flight batching with 2,048-token prefill batches brought it to 0.9 s without changing throughput, but is not the default |
| Coding assistants and tool-using clients (8k-32k prompts) | runs, prefix caching works; TTFT 0.67-0.81 of vLLM-XPU, follow-ups 0.57 | faster attention prefill above 8k tokens, and cheaper reuse of a cached prefix                                                                                       |
| 14 B dense models                                         | not run; FP8 fits on paper                                              | an FP8 checkpoint with per-tensor scales (or block scales in the Intel kernel), then a measured run                                                                  |
| 27-32 B dense models, the usual size for one 32 GB card   | do not fit: no 4-bit path on Intel                                      | an Intel 4-bit weight GEMM for a format MAX reads for the architecture, and a public checkpoint other engines run at the same precision                              |
| MoE models (gpt-oss, 26-35 B MoE)                         | gpt-oss-20b does not load                                               | grouped expert GEMMs, routing and token permutation on Intel, 4-bit experts, and an MXFP4 path without the fixed 15 GiB reservation                                  |
| FP8 for whole models                                      | attention and GDN layers only                                           | FP8 MLPs: MAX wires `qwen3_5`'s MLPs for NVFP4 only                                                                                                                  |
| Long contexts with larger models                          | BF16 KV cache only                                                      | an FP8 KV cache run and tuned on Intel                                                                                                                               |
| More than one GPU                                         | not implemented                                                         | peer memory access and collectives in the driver plugin                                                                                                              |
| Vision, speculative decoding, LoRA, structured output     | not run on Intel                                                        | E2E runs and fixes for what they find                                                                                                                                |
| Laptops and desktops without a discrete GPU               | Qwen3.5-0.8B runs on Arrow Lake-S and Raptor Lake-P                     | larger models, which need the 4-bit path; a single allocation over 4 GiB also fails there                                                                            |
