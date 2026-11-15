# Intel GPU support for Mojo and MAX

> [!NOTE]
> Xe3 graphics, such as Panther Lake iGPUs, have not been tested.

Acronyms are listed in [docs/glossary.md](docs/glossary.md).

Two plugins and the hooks they need let Mojo programs and MAX models run on
Intel GPUs through Level Zero:

- `compiler/`: a Mojo codegen plugin (`IntelCompilerPlugin.so`) that lowers
  kernels for the `spirv64-unknown-unknown` target to OpenCL-flavoured
  SPIR-V with LLVM's SPIR-V backend, plus the IR rewrites Intel's driver
  compiler (IGC) needs.
- `driver/`: a HAL driver plugin (`LevelZeroPlugin.so`) that runs those
  binaries through Level Zero: devices, contexts, USM memory, in-order
  immediate command lists, events, module loading and kernel launch, a
  device heap for `malloc` in kernels, launch profiling, and device graphs
  (Level Zero graph capture) as entry points outside the HAL C API.
- Hooks outside `plugins/`: the `intel` target plugins of the standard
  library (`Mojo/stdlib/std/_plugin/intel/`) and of MAX
  (`max/mojo/max/_plugin/intel/`), the HAL plugin and extension
  (`max/mojo/max/_driver/max_license/hal/`), Intel kernels for MAX
  (GEMM with DPAS and 2D block I/O, FA, GDN, FP8
  weights), and device-graph capture in MLRT for the HAL build.

## Requirements

|                      | Tested                                                             |
|----------------------|--------------------------------------------------------------------|
| GPU                  | Intel Arc Pro B70 (Battlemage G31, `bmg-g31`, Xe2), 32 GB          |
| Kernel driver        | `xe` (Linux 7.2)                                                   |
| Compute runtime      | intel-compute-runtime 26.35.39758                                  |
| Driver compiler      | IGC 2.41.5                                                         |
| Level Zero loader    | 1.32.0 (headers pinned in `level_zero.MODULE.bazel`)               |
| Tools for the checks | `ocloc`, `spirv-val`, `spirv-dis`, clang with the `spirv64` target |
| Host                 | x86-64 Linux; the build host's user in the `render` group          |

Targets: `intel:bmg-g31` (sub-group size 16, DPAS, 2D block I/O) and
`intel:bmg-g31-sg32` (sub-group size 32). Other Xe2 parts should work with
the `bmg-g31` target but are untested. iGPUs: `intel:rpl-p`
(Raptor Lake-P, Xe-LP: work-groups of at most 512, no float64) and
`intel:arl-s` (Arrow Lake-S, Xe-LPG). Their IGC rejects the DPAS and 2D
block I/O extensions, so they run MAX's generic kernels and a
shared-memory SIMT GEMM, not the Intel DPAS GEMM and FA; `mma`
and the block I/O fail to compile for them. They were tested with the
`i915` kernel driver and compute-runtime 26.40.

## Build

```sh
./bazelw build //plugins/intel/... //Mojo:mojo //max/mojo/max:max
# MAX on the HAL runtime (the build Intel GPUs use):
./bazelw build --//:hal_device_context=true //max/python/max/_entrypoints:pipelines
```

Use `--compilation_mode=opt` for performance work; the default
configuration is a debug build.

## Run

Mojo programs:

```sh
export MODULAR_COMPILER_PLUGINS=$PWD/bazel-bin/plugins/intel/compiler/IntelCompilerPlugin.so
export MODULAR_DRIVER_PLUGINS="level_zero@$PWD/bazel-bin/plugins/intel/driver/LevelZeroPlugin.so"
./bazelw run //Mojo:mojo -- run --target-accelerator=intel:bmg-g31 program.mojo
```

MAX models (opt build, HAL runtime):

```sh
bin={{ opt bazel-bin }}  # ./bazelw info --compilation_mode=opt bazel-bin
export MODULAR_COMPILER_PLUGINS=$bin/plugins/intel/compiler/IntelCompilerPlugin.so
drv=$bin/plugins/intel/driver/LevelZeroPlugin.so
export MODULAR_DRIVER_PLUGINS="level_zero@$drv;CPU@$bin/Target/CPUPlugin.so"
./bazelw run --compilation_mode=opt --//:hal_device_context=true \
  //max/python/max/_entrypoints:pipelines -- \
  serve --model-path {{ dir }} --devices gpu
```

`utils/run-max-py` runs a Python script against that build.

Environment switches of the driver plugin: `MOJO_INTEL_PROFILE={{ file }}`
writes a launch profile (kernels, copies, graph launches, host and device
times; `utils/bench/steps.py` reads it); `MOJO_INTEL_COPY_OFFLOAD=1` sends
queue copies to the copy engine; `MOJO_INTEL_HEAP_SIZE` sizes the device
heap; `MOJO_INTEL_IGC_OPTIONS` passes options to IGC.

## Checks

`utils/check OUTDIR [QWEN3.5-0.8B-DIR]` builds everything and runs the
plugin unit tests, the codegen corpus (`tests/codegen`), the E2E
probes (`tests/e2e`), MAX graph tests (`tests/graph`), MAX's GPU test
suite against `tests/max-gpu-expected.txt`, and a greedy-text regression
on Qwen3.5-0.8B. Result on the B70 at this commit:

build, plugin unit tests (73), codegen corpus (23), E2E probes (43),
MAX graph tests (8) pass; MAX's GPU tests: 204 tests, 136 pass, 25 fail,
41 don't compile, 1 n/a, 1 timeout, every result as
`tests/max-gpu-expected.txt` expects; Qwen3.5-0.8B's greedy text
identical to the reference; no GPU faults in the kernel log.

## Performance

Arc Pro B70, opt build, BF16, greedy decoding; `docs/benchmarks.md` has
every configuration, scenario and engine version, and how to reproduce
them.

| Scenario                                       | Model        | MAX     | vLLM-XPU 0.21 | llama.cpp (best of SYCL, Vulkan) |
|------------------------------------------------|--------------|---------|---------------|----------------------------------|
| Prompt processing, pp4096 (tok/s)              | Qwen3.5-0.8B | 40,313  | 47,422        | 9,825                            |
|                                                | Qwen3-8B     | 6,840   | 7,267         | 1,280                            |
| Decode after 4,096 tokens, batch 1 (tok/s)     | Qwen3.5-0.8B | 262.2   | 220.6         | 193.9                            |
|                                                | Qwen3-8B     | 34.6    | 30.3          | 32.3                             |
| Serving, random 512/128, c = 64 (output tok/s) | Qwen3.5-0.8B | 3,915.7 | 3,017.7       | -                                |
|                                                | Qwen3-8B     | 825.9   | 861.6         | -                                |
| TTFT, 32k-token prompt                         | Qwen3-8B     | 10.96 s | 7.37 s        | -                                |

vLLM-XPU's best mode per cell: XPU graphs for Qwen3.5-0.8B single stream,
eager otherwise. MAX's serving cells are from a tree without the last
attention prefill changes (`docs/benchmarks.md`, tree B). llama.cpp's serving
and long prompts were not measured.

What has had tuning passes for Xe2:

- dense GEMMs (DPAS, 2D block loads and prefetch, skinny-M decode kernel)
- weight-only FP8 GEMMs, ragged paged FA (prefill with DPAS, split-K decode)
- GDN recurrence, RMSNorm and rowwise reductions with 16-lane sub-groups
- launch overhead (allocation cache, deferred frees, device graphs for decode)

What has not been tuned and/or verified:

- MoE layers
- long-context prefill (attention above 16k tokens)
- sampling kernels
- quantization below 8 bits

iGPUs, Qwen3.5-0.8B (MAX only): Arrow Lake-S 441 tok/s prompt
processing (512 tokens) and 42.3 tok/s decode with device graphs, Raptor
Lake-P 333 and 21.0 tok/s.

`docs/performance.md` describes where the time goes on each GPU and what is
left to tune, by measured payoff.

## Known limitations

`docs/support.md` lists the models, weight formats, context lengths and
serving features that run on Intel GPUs, how MAX tracks vLLM-XPU, the
functionality still missing, and what common use cases need.

- gpt-oss-20b (MXFP4 MoE) does not load: MAX's MXFP4 MoE
  path has a native grouped GEMM only for SM100; elsewhere it dequantizes
  every expert on each forward behind a fixed 15 GiB reservation, which
  the 32 GB B70 cannot hold beside the KV cache.
- TTFT at 24k-32k tokens is 0.67-0.70x of vLLM-XPU on Qwen3-8B
  and 0.68-0.70x on Qwen3.5-0.8B (attention prefill). Qwen3-8B serving at 64
  concurrent requests is 0.96x of vLLM-XPU on random 512/128 and on ShareGPT;
  TTFT at 64 requests is 2.9 s against 0.8 s (in-flight batching
  with 2,048-token batches brings it to 0.9 s without changing the throughput).
- Qwen3-8B needs `--no-prefer-module-v3`: MAX's module-v3 lookup resolves
  `Qwen3ForCausalLM` to its embedding pipeline.
- IGC 2.41/2.42 miscompiles a few math kernels (unscalarized vector
  `fabs`, some float64 paths); the affected MAX tests are marked `igc` in
  `tests/max-gpu-expected.txt` and pass with IGC master. The Bessel
  tests (`test_j1_y1_gpu`, `test_bessel_huge_args_gpu`) fail
  intermittently; they pass with `MOJO_INTEL_COPY_OFFLOAD=1`.
  `docs/intel-tools-issues.md` lists every known issue in Intel's tools.
- Copies run on the compute engine: with compute-runtime 26.35 the copy
  engine page-faulted and hung when device memory ran out, where the
  compute engine reports OOM.
- The `DeviceContext` of the HAL build lacks a few legacy APIs; the tests
  that need them are marked `compat` in `tests/max-gpu-expected.txt`.
- iGPUs: with `i915` and compute-runtime 26.40 an allocation
  over `maxMemAllocSize` (4 GiB) fails although the driver plugin asks
  for relaxed limits (MAX's sampling test's 4,096-row case; a model run
  needs none). The FP8 weight-only op needs DPAS, so FP8 linear layers
  take MAX's generic path there. Xe-LP has no float64.
- Multi-GPU, peer access and NVSHMEM-style communication are not
  implemented.

## Layout

| Path                         |                                                                                                    |
|------------------------------|----------------------------------------------------------------------------------------------------|
| `compiler/`                  | codegen plugin, its IR passes and their unit tests                                                 |
| `driver/`                    | Level Zero driver plugin, its unit tests and test kernels                                          |
| `tests/`                     | codegen corpus, device and E2E probes, MAX graph tests, expected MAX test results, regression text |
| `utils/`                     | check scripts, Level Zero helpers, benchmark scripts                                               |
| `docs/LICENSING.md`          | MAX-licensed files changed by this work                                                            |
| `docs/benchmarks.md`         | MAX against vLLM-XPU and llama.cpp: configurations, scenarios, results, reproduction               |
| `docs/intel-tools-issues.md` | known issues in Intel's GPU tools, the plugins' workarounds and a reproduction of each             |
| `docs/performance.md`        | where the time goes, what is left to tune                                                          |
| `docs/support.md`            | models, weight formats, context and features supported, and what is missing                        |
| `docs/glossary.md`           | the acronyms these docs use                                                                        |
| `CONTRIBUTING.md`            | commit and style rules                                                                             |
