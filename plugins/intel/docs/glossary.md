# Acronyms

The acronyms used in these documents. The rest of the documentation uses
only the acronym.

## Inference

| Acronym      | Meaning                                                                           |
|--------------|-----------------------------------------------------------------------------------|
| E2E          | end-to-end (the probes in `tests/e2e`)                                            |
| FA           | flash attention                                                                   |
| GDN          | Gated DeltaNet, the linear-attention layers of Qwen3.5                            |
| GEMM         | general matrix-matrix multiply                                                    |
| GEMV         | general matrix-vector multiply                                                    |
| ITL          | inter-token latency                                                               |
| KV           | key-value, as in KV cache                                                         |
| LLM          | large language model                                                              |
| LoRA         | low-rank adaptation                                                               |
| MLP          | multilayer perceptron, a transformer's feed-forward block                         |
| MoE          | mixture of experts                                                                |
| MTP          | multi-token prediction                                                            |
| NaN          | not a number                                                                      |
| OOM          | out of memory                                                                     |
| RMSNorm      | root-mean-square normalization                                                    |
| RoPE         | rotary position embedding                                                         |
| tok/s        | tokens per second                                                                 |
| TP           | tensor parallelism                                                                |
| TPOT         | time per output token                                                             |
| TTFT         | time to first token                                                               |
| pp*N*, tg*N* | llama-bench tests: prompt processing of *N* tokens, text generation of *N* tokens |

## Number formats

| Acronym | Meaning                                                                      |
|---------|------------------------------------------------------------------------------|
| BF16    | bfloat16                                                                     |
| FP4     | 4-bit floating point                                                         |
| FP8     | 8-bit floating point (E4M3 here: 4 exponent bits, 3 mantissa bits)           |
| GGUF    | llama.cpp's model file format, with quantization types such as q4_0 and q6_k |
| GPTQ    | a post-training weight quantization method                                   |
| MXFP4   | microscaling FP4 (OCP MX): one E8M0 scale per 32 values                      |
| NVFP4   | NVIDIA's FP4: one FP8 scale per 16 values and a per-tensor scale             |

## Intel GPU stack

| Acronym | Meaning                                                                                             |
|---------|-----------------------------------------------------------------------------------------------------|
| BCS     | blitter command streamer, the copy engine (`class=bcs` in kernel log lines)                         |
| CCS     | compute command streamer, the compute engine (`class=ccs`)                                          |
| DPAS    | dot product accumulate systolic, Xe2's sub-group matrix multiply-accumulate instruction, run on XMX |
| GuC     | graphics microcontroller, the firmware that schedules work on the GPU's engines                     |
| HAL     | hardware abstraction layer, MAX's driver interface (`SDK/include/HAL/`)                             |
| IGC     | Intel Graphics Compiler, which compute-runtime calls to compile SPIR-V to GPU ISA                   |
| iGPU    | integrated GPU (here Arrow Lake-S and Raptor Lake-P)                                                |
| ISA     | instruction set architecture; here, the GPU code IGC generates                                      |
| MLRT    | MAX's runtime (the `MLRT/` directory)                                                               |
| NEO     | compute-runtime's project name (`libze_intel_gpu.so`, the `NEO_*` environment variables)            |
| SIMD    | single instruction, multiple data; IGC compiles kernels SIMD16 or SIMD32                            |
| SIMT    | single instruction, multiple threads                                                                |
| SLM     | shared local memory, the memory a work-group shares (sub-group and work-group scratch)              |
| SPIR-V  | Standard Portable Intermediate Representation (Khronos), the kernels' input to IGC                  |
| UMD     | user-mode driver (compute-runtime)                                                                  |
| USM     | unified shared memory, Level Zero's host, device and shared allocations                             |
| XMX     | Xe Matrix Extensions, the matrix engines of Xe GPUs                                                 |
| XPU     | Intel's name for its GPUs in PyTorch and vLLM (vLLM-XPU)                                            |
