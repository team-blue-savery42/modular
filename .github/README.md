# Mojo and MAX on Intel GPUs

A personal project by [savery42](https://github.com/savery42): native
Intel GPU support for Mojo and MAX, built as plugins in a fork of
[modular/modular](https://github.com/modular/modular). It is not
affiliated with or endorsed by Modular or Intel.

> [!IMPORTANT]
> **Disclosure of interests**
>
> This project is not sponsored by, supported by, or endorsed by
> either Intel or Modular.

> [!WARNING]
> :construction: Under active construction.

> [!IMPORTANT]
> **AI Disclosure**
>
> All efforts of this branch have had some form of AI assistance.
> Call a spade a spade; this project is for fun and learning, AI
> tools were used heavily in areas that were not fun (e.g. thinking
> / planning how to test the system properly, and writing a few test
> harnesses? Fun! Writing 20k lines of tests? Not fun!).
>
> As well as standard rubber-ducking, second opinion on code reviews,
> running bisects overnight and after rebasing with Modular's tip, etc.
>
> Every line has had human review and undergone fairly exhaustive testing.

## Status

- **Arc Pro B70** (Battlemage, Xe2): Mojo programs and MAX models
  (Qwen3-8B, Qwen3.5-0.8B) run end to end on kernels tuned for Xe2
  (DPAS GEMM, FA, GDN). Against vLLM-XPU 0.21:
  prompt processing 0.85-1.16x up to 4,096 tokens and about 0.55x at
  32,768, single-stream decode 1.0-1.2x, serving at 64 concurrent
  requests 0.96-1.3x.
- **iGPUs** (Arrow Lake-S, Raptor Lake-P): Qwen3.5-0.8B runs
  on MAX's generic kernels and a shared-memory GEMM.
- **Not yet**: multi-GPU, MoE models, quantization below
  8 bits.

Every figure was measured; the configurations are in
[benchmarks.md](../plugins/intel/docs/benchmarks.md).

## Where to start

| Path                                                                        | What it holds                                                     |
|-----------------------------------------------------------------------------|-------------------------------------------------------------------|
| [`plugins/intel/README.md`](../plugins/intel/README.md)                     | requirements, build, run, checks, known limitations               |
| [`plugins/intel/compiler/`](../plugins/intel/compiler)                      | codegen plugin: Mojo kernels to SPIR-V for Intel's driver         |
| [`plugins/intel/driver/`](../plugins/intel/driver)                          | HAL driver plugin on Level Zero                                   |
| [`docs/benchmarks.md`](../plugins/intel/docs/benchmarks.md)                 | MAX against vLLM-XPU and llama.cpp, with reproduction commands    |
| [`docs/performance.md`](../plugins/intel/docs/performance.md)               | where the time goes and what is left to tune                      |
| [`docs/support.md`](../plugins/intel/docs/support.md)                       | what runs, MAX against vLLM-XPU, what common use cases still need |
| [`docs/glossary.md`](../plugins/intel/docs/glossary.md)                     | the acronyms used in the documentation                            |
| [`docs/intel-tools-issues.md`](../plugins/intel/docs/intel-tools-issues.md) | known bugs in Intel GPU tools, with reproductions                 |
| [`CONTRIBUTING.md`](../plugins/intel/CONTRIBUTING.md)                       | commit and style rules                                            |

Changes outside `plugins/intel/` (standard library and MAX target hooks,
Intel kernels, device-graph capture) are listed in the plugin README.

## Branches

- `intel/bringup` (default): upstream `main` plus the Intel series, one
  commit per working step, each building and passing its checks.
- Every other branch is mirrored from upstream unchanged.

## Licence

As upstream. The plugins and most changed files are under the Apache
License v2.0 with LLVM Exceptions ([LICENSE](../LICENSE)). Files in
`max_license/` directories and the MLRT and GraphCompiler sources are
under the [MAX License](../Licenses/MAX_LICENSE);
[LICENSING.md](../plugins/intel/docs/LICENSING.md) lists the ones this
work changes or adds.

## Mojo and MAX themselves

See [Modular's README](../README.md) and <https://docs.modular.com>.
Please report problems with this port here, not to Modular.
