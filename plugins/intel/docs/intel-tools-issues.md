# Known issues in Intel's GPU tools

Bugs in IGC, compute-runtime, the Level Zero loader
and the kernel driver found while bringing up the Intel plugins, what the
plugins do about each, and how to reproduce it without Mojo. Each reproduction
is a directory under [`tests/intel-tools/`](../tests/intel-tools/) with a
`README.md` and a `run.sh`. They need `clang` with the `spirv64` target,
`spirv-val`, `ocloc` or a C++ compiler, and most a `bmg-g31` GPU on Level Zero.

The tested stack is the one in the [README](../README.md): Linux 7.2.8
with the `xe` driver and GuC firmware 70.72.1, IGC 2.41.5,
compute-runtime 26.35.39758 and Level Zero loader 1.32.0 (Arch Linux
packages). The reproductions were also run on stacks built from Intel's
sources:

| Name in the tables | IGC                                         | compute-runtime   |
|--------------------|---------------------------------------------|-------------------|
| 2.41.5             | 2.41.5 (the tested stack)                   | 26.35.39758       |
| v2.42.14           | v2.42.14                                    | 26.40.40196.15    |
| master `bd0ad0a5`  | master of 2026-11-01, built with assertions | master `ee495cce` |
| v2.43.9            | v2.43.9                                     | 26.44.40734.8     |

Running each IGC with the other stacks' compute-runtime tells which of the
two a change comes from; for every IGC issue checked that way the result
followed the IGC. "Fixed in" names the IGC commit, found by bisecting
release builds of IGC master from `b50679576d` (where 2.41.5's release
branch forks) to `bd0ad0a5`, and the first tag that contains it.

Workarounds for bugs fixed in newer IGC stay while IGC 2.41.5 is the tested
version. None of these issues has been reported upstream yet.

## Index

### Open on the tested stack and on the newest releases

| Issue                                                                                  | Component                        | Symptom                                                                         | Affected                                                                     | Fixed in                   | What the plugins do                                                                                                    | Reproduction                                                                                                    |
|----------------------------------------------------------------------------------------|----------------------------------|---------------------------------------------------------------------------------|------------------------------------------------------------------------------|----------------------------|------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| [float16 `log2` on vectors](#igc-float16-log2-on-a-4-wide-vector-gives-wrong-lanes)    | IGC                              | 4- and 8-wide float16 `log2` gives wrong lanes                                  | every IGC tested                                                             | not fixed; candidate patch | nothing; `-cl-opt-disable` or `IGC_EnableVectorizer=0` avoids it                                                       | [igc-vector-log2-lanes](../tests/intel-tools/igc-vector-log2-lanes/)                                            |
| [Vector compare crash](#igc-the-vectorizer-crashes-on-an-extracted-vector-compare)     | IGC                              | the module build dies with a floating point exception                           | every IGC tested                                                             | not fixed; candidate patch | nothing; `-cl-opt-disable` or `IGC_EnableVectorizer=0` avoids it                                                       | [igc-vector-compare-crash](../tests/intel-tools/igc-vector-compare-crash/)                                      |
| [`SPV_KHR_abort`](#igc-rejects-spv_khr_abort)                                          | IGC                              | modules with `OpAbortKHR` are rejected                                          | every IGC tested                                                             | not fixed                  | `abort` raises a flag in host memory; the next synchronize fails                                                       | [igc-spv-khr-abort](../tests/intel-tools/igc-spv-khr-abort/)                                                    |
| [printf `%s`](#compute-runtime-reads-a-printf-s-argument-as-a-host-address)            | compute-runtime                  | a `%s` of a non-literal string segfaults the host process                       | 26.35.39758, 26.44.40734.8                                                   | not fixed                  | device messages go out as one `printf` whose format is the whole message                                               | [compute-runtime-printf-string](../tests/intel-tools/compute-runtime-printf-string/)                            |
| [Hang after a GPU fault](#after-a-gpu-page-fault-the-next-process-hangs)               | compute-runtime or kernel driver | the fault is reported as success; the next process's first wait never completes | every compute-runtime tried (26.35.39758, 26.40.40196.15, master `ee495cce`) | not fixed                  | waits check the device status, so a later wait reports the fault; the test gate runs a throwaway process after a fault | [gpu-fault-next-process-hangs](../tests/intel-tools/gpu-fault-next-process-hangs/) (do not run on a shared GPU) |
| [float64 to float32 on vectors](#igc-float64-to-float32-on-4-wide-vectors-page-faults) | IGC                              | a kernel converting `<4 x double>` page-faults                                  | 2.41.5, v2.42.14, master `bd0ad0a5`                                          | not fixed                  | nothing                                                                                                                | none (Mojo only)                                                                                                |
| [Allocations over 4 GiB on iGPUs](#igpus-allocations-over-4-gib-fail)                  | compute-runtime (`i915`)         | the relaxed-limits request is rejected                                          | 26.40 with `i915` (Raptor Lake-P, Arrow Lake-S)                              | not fixed                  | nothing; model runs need no such allocation                                                                            | none                                                                                                            |

### Fixed in newer releases

| Issue                                                                                        | Component                   | Symptom                                                      | Affected                                | Fixed in                                                                          | What the plugins do                                                         | Reproduction                                                                                 |
|----------------------------------------------------------------------------------------------|-----------------------------|--------------------------------------------------------------|-----------------------------------------|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| [Leftover vector `fabs`](#igc-a-leftover-vector-fabs-is-computed-for-its-first-lane-only)    | IGC                         | lanes 1-3 read registers nothing wrote                       | 2.41.5, v2.42.14                        | `307d24a474d5`, v2.43.0                                                           | nothing; the affected MAX math tests are expected to fail                   | [igc-vector-fabs-first-lane](../tests/intel-tools/igc-vector-fabs-first-lane/)               |
| [MAX math tests](#igc-241-and-242-fail-maxs-math-tests)                                      | IGC                         | 14 of MAX's libm tests fail with optimization on             | 2.41.5, v2.42.14                        | v2.43.9 (with the struct-argument patch) passes them                              | nothing; the tests are expected to fail                                     | none beyond the two above                                                                    |
| [BF16 selects, phis and rounding](#igc-wrong-bits-from-bf16-selects-phis-and-rounding)       | IGC                         | BF16 results get bits of neither input                       | 2.41.5                                  | `81dae221ed3a`, v2.42.0                                                           | BF16 values move as `i16`; conversions are integer operations, lane by lane | [igc-bfloat16-select-phi](../tests/intel-tools/igc-bfloat16-select-phi/)                     |
| [Aggregate phis](#igc-a-value-merged-by-an-aggregate-phi-is-lost)                            | IGC                         | a field merged by a struct phi becomes undefined             | 2.41.5, v2.42.14                        | `0221e64d2c48`, v2.43.0 (the `-cl-opt-disable` segfault: `56fc303e415f`, v2.42.0) | aggregate phis are split into one phi per field                             | [igc-aggregate-phi-index](../tests/intel-tools/igc-aggregate-phi-index/)                     |
| [Nested struct arguments](#igc-nested-struct-arguments-and-return-values-are-passed-wrongly) | IGC                         | fields of an inner struct passed by value read another field | 2.41.5, v2.42.14                        | `0221e64d2c48`, v2.43.0                                                           | nested struct arguments and results are passed as flat structs              | [igc-nested-struct-arguments](../tests/intel-tools/igc-nested-struct-arguments/)             |
| [128-bit integers](#igc-128-bit-integer-arithmetic-is-wrong)                                 | IGC                         | 128-bit constants keep 32 bits; adds are wrong               | 2.41.5; v2.42.14 (constants)            | adds `c31c33b17645`, v2.42.0; constants `cbeb3b419217`, v2.43.0                   | every 128-bit integer is split into two 64-bit halves                       | [igc-int128-arithmetic](../tests/intel-tools/igc-int128-arithmetic/)                         |
| [Validation layer crash](#level-zero-the-packaged-validation-layer-crashes-at-load)          | Level Zero loader (package) | `ZE_ENABLE_VALIDATION_LAYER=1` segfaults every program       | Arch Linux `level-zero-loader 1.32.0-1` | v1.32.0 built from source; v1.34.0 and newer                                      | nothing; use a loader and layer built from source                           | [level-zero-validation-layer-crash](../tests/intel-tools/level-zero-validation-layer-crash/) |

### Regressions in newer IGC

| Issue                                                                                                            | Component | Symptom                                                         | Affected                                                                                                                                                   | Fixed in                                                                                                           | What the plugins do                                     | Reproduction                                                                       |
|------------------------------------------------------------------------------------------------------------------|-----------|-----------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|------------------------------------------------------------------------------------|
| [NaN in paged FA](#igc-242-and-243-nan-in-paged-fa)                                                              | IGC       | MAX's paged attention returns NaN                               | v2.42.14: decode batch, prefill and ragged cases; v2.43.9 and master `bd0ad0a5`: the three decode cases (also with compute-runtime 26.35); 2.41.5 is right | decode: introduced by `e88560098464` (v2.43.0); 2.42 prefill: right again on master from `2e369af623` (2026-10-19) | nothing; stay on 2.41.5                                 | [`tests/graph/attention.py`](../tests/graph/attention.py) (MAX)                    |
| [Struct arguments segfault](#igc-243-segfaults-on-two-struct-kernel-arguments-passed-by-value)                   | IGC       | the module build segfaults on kernels with two struct arguments | v2.43.9, master `bd0ad0a5`; introduced by `0221e64d2` (v2.43.0)                                                                                            | not fixed; candidate patch                                                                                         | nothing; most MAX graphs fail on 2.43 without the patch | [igc-struct-argument-segfault](../tests/intel-tools/igc-struct-argument-segfault/) |
| [Struct kernel argument without optimization](#igc-nested-struct-arguments-and-return-values-are-passed-wrongly) | IGC       | with `-cl-opt-disable` a struct kernel argument fails to build  | v2.43.9, master `bd0ad0a5`; introduced by `0221e64d2` (v2.43.0)                                                                                            | not fixed; candidate patch                                                                                         | nothing; affects only `-cl-opt-disable` builds          | [igc-nested-struct-arguments](../tests/intel-tools/igc-nested-struct-arguments/)   |

### Resolved

| Issue                                                                    | Component                     | Symptom                                                          | Affected    | Fixed in                                       | What the plugins do                                                       | Reproduction |
|--------------------------------------------------------------------------|-------------------------------|------------------------------------------------------------------|-------------|------------------------------------------------|---------------------------------------------------------------------------|--------------|
| [Copy engine fault](#copy-engine-page-fault-when-device-memory-runs-out) | compute-runtime (copy engine) | the copy engine page-faulted and hung when device memory ran out | 26.35.39758 | not reproduced on 26.35.39758 or 26.44.40734.8 | queue copies run on the compute engine unless `MOJO_INTEL_COPY_OFFLOAD=1` | none         |

## IGC: float16 `log2` on a 4-wide vector gives wrong lanes

With IGC's default optimization, `log2` of a `<4 x half>` (widened to
float32, `log2`, narrowed back) is wrong: on 2.41.5 the result is `x` for
x >= 1 and 0 below, in every lane; on newer IGC two of the four lanes give 0.
IGC's vectorizer turns the four narrowing conversions into one vector
`fptrunc`, and `EmitPass::FPTrunc` writes a uniform vector result in the
layout of a per-lane one, past the end of its variable.

In Mojo, `log2` of `SIMD[float16, 4]` and `SIMD[float16, 8]` is wrong
(`tests/e2e/log2_widths.mojo`); widths 1 and 32 and float32 are right.

| IGC                                | Default build      | `-cl-opt-disable` | `IGC_EnableVectorizer=0` |
|------------------------------------|--------------------|-------------------|--------------------------|
| 2.41.5                             | 4 of 4 lanes wrong | right             | right                    |
| v2.42.14                           | 2 of 4 lanes wrong | right             | right                    |
| master `bd0ad0a5`                  | 2 of 4 lanes wrong | right             | right                    |
| v2.43.9                            | 2 of 4 lanes wrong | right             | right                    |
| master `bd0ad0a5` with `fix.patch` | right              | right             | -                        |

Workaround: none. `MOJO_INTEL_IGC_OPTIONS=-cl-opt-disable` avoids it at the
cost of every optimization. The candidate patch writes
element `i` of a uniform result at sub-register `i`. Reproduction and
patch: [`igc-vector-log2-lanes`](../tests/intel-tools/igc-vector-log2-lanes/).

## IGC: the vectorizer crashes on an extracted vector compare

IGC's vectorizer makes a vector `icmp` and gives the compares' scalar
users an `extractelement` of its `<4 x i1>` result. `EmitPass::emitExtract`
cannot emit that (an `i1` lane has no whole bytes): an assertion build
stops at `eltBytes`, a release build divides by zero, and `ocloc` or
`zeModuleCreate` dies with "Internal Compiler Error: Floating point
exception".

In Mojo it appeared in a kernel computing `atan2` on four lanes loaded from
memory; the module does not load. When some lanes were constants it
compiled, but `atan2(1, inf)` gave NaN instead of 0 (not reduced).

| IGC               | Default build            | `-cl-opt-disable` | `IGC_EnableVectorizer=0` |
|-------------------|--------------------------|-------------------|--------------------------|
| 2.41.5            | crash (`ocloc` exit 245) | builds            | builds                   |
| v2.42.14          | crash                    | builds            | builds                   |
| master `bd0ad0a5` | assertion `eltBytes`     | builds            | builds                   |
| v2.43.9           | crash                    | builds            | builds                   |
| with `fix.patch`  | builds                   | builds            | -                        |

Workaround: none; `-cl-opt-disable` avoids it. The
candidate patch keeps the scalar compares for their scalar users.
Reproduction and patch:
[`igc-vector-compare-crash`](../tests/intel-tools/igc-vector-compare-crash/).

## IGC rejects `SPV_KHR_abort`

`ocloc` and `zeModuleCreate` reject a module that uses `SPV_KHR_abort`
("unknown extension"). Without the extension LLVM's SPIR-V backend drops
`llvm.trap`, so a kernel cannot stop in a way Level Zero reports: an
aborting work-item prints its message and ends, the rest of
the grid finishes, and Level Zero reports success.

| IGC               | Module with `OpAbortKHR`   |
|-------------------|----------------------------|
| 2.41.5            | rejected (`ocloc` exit 10) |
| v2.42.14          | rejected (exit 10)         |
| master `bd0ad0a5` | rejected (exit 245)        |
| v2.43.9           | rejected (exit 245)        |

Workaround: the compiler plugin makes `abort` store to a flag in host
memory that the driver plugin owns, per context, and the work-item returns
to its kernel; the driver plugin checks the flag after each synchronize
and fails it, and every one after it. So an abort is reported at the next
synchronize, not at the launch, and the grid's other work-items run to
completion. Reproduction:
[`igc-spv-khr-abort`](../tests/intel-tools/igc-spv-khr-abort/).

## compute-runtime reads a printf `%s` argument as a host address

compute-runtime formats `%s` by passing the address the kernel stored to
`snprintf` as a host `char *`. That works for string literals, which IGC
places where compute-runtime maps them at the same address on the host,
and by accident for strings in shared memory; a string anywhere else,
such as private memory, segfaults the host process in
`zeCommandListHostSynchronize`.

In Mojo, `debug_assert` passes the file name and message at run time, so
a failing assertion crashed the host process this way.

| compute-runtime (IGC)   | `%s` of a private string |
|-------------------------|--------------------------|
| 26.35.39758 (2.41.5)    | host segfault            |
| 26.44.40734.8 (v2.43.9) | host segfault            |

26.40.40196.15 and master `ee495cce` have the same formatting code (read, not
run). Workaround: the standard library's Intel `debug_assert` builds the whole
line on the device, with `%` doubled, and makes one `printf` whose format is
that line; IGC copies a format string that is not a literal into the printf
buffer, where compute-runtime reads it. `print` writes runtime strings with
`%c`. Reproduction:
[`compute-runtime-printf-string`](../tests/intel-tools/compute-runtime-printf-string/).

## After a GPU page fault, the next process hangs

After a process whose kernel page-faults (the kernel log shows `Engine
memory CAT error` and an engine reset), two things go wrong. In the
faulting process `zeCommandListHostSynchronize` and `zeDeviceGetStatus`
return success right after the faulting kernel. And the first process to
use the GPU after it never sees its first command complete: its wait
returns `ZE_RESULT_NOT_READY` until it gives up. The process after that
one runs normally. A pause between them does not help.

In a test suite one faulting test costs the next test its whole timeout,
and the hang looks like a bug in that test.

Measured on 2026-11-01: the same with compute-runtime 26.35.39758,
26.40.40196.15 and master `ee495cce`, and with each of them paired with
the other IGC versions. So it is not IGC; it is compute-runtime, the `xe`
driver (Linux 7.2.8) or the GuC firmware (70.72.1).

Workaround: the driver plugin asks `zeDeviceGetStatus` when a wait fails and at
most once a millisecond otherwise, so waits fail once the device is lost.
`utils/check-max-gpu` runs tests known to fault last, and after any fault runs a
throwaway process (`utils/l0/fault-absorb`) that takes the hang. Reproduction:
[`gpu-fault-next-process-hangs`](../tests/intel-tools/gpu-fault-next-process-hangs/).
It faults the GPU on purpose: run it only on a GPU nothing else is using.

## IGC: float64 to float32 on 4-wide vectors page-faults

A Mojo kernel in which one work-item converts 196,608 float64 values to
float32 as `SIMD[float64, 4]` page-faults with IGC's default optimization
and is right with `-cl-opt-disable`, on 2.41.5, v2.42.14 and master
`bd0ad0a5` (v2.43.9 not measured). A plain kernel with one `<4 x double>`
or `<8 x double>` per work-item is right everywhere, so there is no
reproduction without Mojo yet. Workaround: none. float64
to BF16, which goes through float32, is right at widths 1, 4, 8 and
16.

## iGPUs: allocations over 4 GiB fail

On Raptor Lake-P and Arrow Lake-S with the `i915` driver and
compute-runtime 26.40, `zeMemAllocDevice` of more than `maxMemAllocSize`
(4,095 MiB) fails with `ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION`, although
the driver plugin passes `ze_relaxed_allocation_limits_ext_desc_t` with
`ZE_RELAXED_ALLOCATION_LIMITS_EXT_FLAG_MAX_SIZE`. It shows in MAX's
sampling test's 4,096-row case. A model run needs no such allocation
(Qwen3.5-0.8B's KV cache budget there is 324 MiB). No reproduction
without MAX yet.

## IGC: a leftover vector `fabs` is computed for its first lane only

IGC's final IR keeps a `<4 x float> @llvm.fabs.v4f32` beside the scalar
`fabs` calls that replaced its other uses, and the vISA emitter writes
only its first lane. Lanes 1-3 of `|x|` then come from registers the
kernel never writes: right when the GPU was idle before, wrong after a
kernel that left other values there.

In MAX, `test_j0_y0_gpu`'s float32, float16 and float64 kernels failed
depending on what ran before them, which made the result depend on
compute-runtime's persistent cache (a warm cache runs the kernels back to
back; a cold one compiles each first, leaving the GPU idle).

| IGC               | Lanes 1-3 after another kernel |
|-------------------|--------------------------------|
| 2.41.5            | wrong                          |
| v2.42.14          | wrong                          |
| master `bd0ad0a5` | right                          |
| v2.43.9           | right                          |

Fixed in IGC `307d24a474d5`, "Scalarize vector fabs/abs in ScalarizerCodeGen"
(v2.43.0). Workaround: none; the affected MAX tests are marked `igc` in
`tests/max-gpu-expected.txt`. Reproduction:
[`igc-vector-fabs-first-lane`](../tests/intel-tools/igc-vector-fabs-first-lane/).

## IGC 2.41 and 2.42 fail MAX's math tests

With IGC's default optimization, 14 of MAX's libm tests fail on 2.41.5
(acosh/asinh, atan, Bessel huge arguments, hypot, j0/y0, j1/y1, lgamma,
libm reduction, log, log1p, log ULP, remainder, which times out, scalb,
tan); those tried with `-cl-opt-disable` pass. v2.42.14 passes `test_log`
and, with a cold compute-runtime cache, `test_j0_y0_gpu`; master
`bd0ad0a5` passes the 12 of them that were in the test suite then (all
but the two Bessel tests); v2.43.9 passes all 14 (with the
struct-argument patch below).
The two issues above explain `test_j0_y0_gpu` and float16 `log2`; the
others were not reduced. The tests are marked `igc` in
`tests/max-gpu-expected.txt`. The Bessel tests (`test_j1_y1_gpu`,
`test_bessel_huge_args_gpu`) also fail intermittently on 2.41.5 and pass
with `MOJO_INTEL_COPY_OFFLOAD=1`.

## IGC: wrong bits from BF16 selects, phis and rounding

On 2.41.5, an `OpSelect` or `OpPhi` of `bfloat16` values gives bits of
neither input (`ffff`, `fe00`) wherever a loaded value is chosen, and a
value rounded to BF16 and back, then computed on again, comes out
wrong (`fe40` for `0x4058`): IGC's optimizer narrows the float32
arithmetic between the conversions to BF16 arithmetic, which the
device then computes wrongly. Some cases are wrong with `-cl-opt-disable`
too.

In Mojo this made BF16 `log`, `exp` and `**` wrong: MAX's math puts
special cases (`x == inf ? inf : ...`) in BF16 selects and phis. A
float32 to BF16 `OpFConvert` on 4- and 8-wide vectors was also wrong
for most inputs, and page-faulted at 16 lanes; a plain kernel does not
reproduce that one.

| IGC               | Selects and phis | Rounding |
|-------------------|------------------|----------|
| 2.41.5            | wrong            | wrong    |
| v2.42.14          | right            | right    |
| master `bd0ad0a5` | right            | right    |
| v2.43.9           | right            | right    |

Fixed in IGC `81dae221ed3a`, "Changes in code." (v2.42.0), in vISA's
`HWConformity::fixBFMixedMode`. Workaround: the compiler plugin moves
BF16 values as `i16`, so only conversions see the BF16 type, and
converts between BF16 and float32 with integer operations, lane by
lane (round to nearest even; NaN as the host converts it). Reproduction:
[`igc-bfloat16-select-phi`](../tests/intel-tools/igc-bfloat16-select-phi/).

## IGC: a value merged by an aggregate phi is lost

A phi of a nested struct, such as the `Optional` that
`for i, byte in enumerate(s.bytes())` iterates on, loses the integer field of
its inner struct when the other incoming value is a constant whose inner struct
is undefined. With IGC's defaults the index is undefined and its stores are
deleted; with `-cl-opt-disable` it reads whatever its private slot holds. On
2.41.5 `ocloc` with `-cl-opt-disable` also segfaults on the same module with
bounds checks on.

In MAX, `test_amd_format` wrote through an index from such a loop and
page-faulted; with bounds checks the check failed on an index that was in
bounds.

| IGC               | Indices (default / `-cl-opt-disable`) | `-cl-opt-disable` with bounds checks |
|-------------------|---------------------------------------|--------------------------------------|
| 2.41.5            | `-1 -1 -1` / `0 0 0`                  | segfault                             |
| v2.42.14          | `-1 -1 -1` / `0 0 0`                  | builds                               |
| master `bd0ad0a5` | right                                 | builds                               |
| v2.43.9           | right                                 | builds                               |

Fixed in IGC `0221e64d2c48`, "Support struct arguments passed by value in
SPIR-V" (v2.43.0); the segfault in `56fc303e415f`, "Fix no-opt compilation
failures" (v2.42.0). Workaround: the compiler plugin splits each phi of a
struct or array type with at most 64 scalar leaves into one phi per leaf
(`tests/e2e/bounds_loop.mojo` checks it); larger phis reach IGC whole.
Reproduction:
[`igc-aggregate-phi-index`](../tests/intel-tools/igc-aggregate-phi-index/).

## IGC: nested struct arguments and return values are passed wrongly

A struct like `{ i64, i64, { ptr, i64 } }` passed by value to a function,
returned from one, or passed to a kernel, arrives with the inner struct's
fields read from the wrong place: both read the outer struct's first
field, or the length reads the pointer. With `-cl-opt-disable` such
modules did not build on 2.41.5.

In Mojo, `debug_assert` passes its `SourceLocation`
(`{ line, column, { file, length } }`) this way, and the reporter read the
file name from the address 0x35, its length: a page fault. `List`'s
`_realloc` returns `{ { ptr, capacity }, length }`, and a growing list
stored through a null pointer (`tests/e2e/nested_return.mojo`).

| IGC               | Function argument, return value      | Kernel argument (default / `-cl-opt-disable`) |
|-------------------|--------------------------------------|-----------------------------------------------|
| 2.41.5            | wrong; no build without optimization | wrong / wrong                                 |
| v2.42.14          | wrong; right without optimization    | wrong / wrong                                 |
| master `bd0ad0a5` | right                                | right / IGC segfault                          |
| v2.43.9           | right                                | right / build failure                         |
| with `fix.patch`  | right                                | right / right                                 |

Fixed with IGC's defaults in `0221e64d2c48` (v2.43.0), which flattens struct
kernel arguments passed by value into the payload. The same commit broke struct
kernel arguments with `-cl-opt-disable`: `PrivateMemoryUsageAnalysis` still
counts only `byval` struct arguments, so the kernel gets no `PRIVATE_BASE` for
the `alloca` that `ResolveAggregateArguments` now makes, and
`PrivateMemoryResolution` reads a null pointer. The candidate patch counts
struct arguments passed by value too. Workaround: the compiler plugin passes
nested struct arguments of kernels and of internal functions, and the nested
struct results of internal functions, as flat structs of their scalar leaves.
Reproduction and patch:
[`igc-nested-struct-arguments`](../tests/intel-tools/igc-nested-struct-arguments/).

## IGC: 128-bit integer arithmetic is wrong

On 2.41.5 a 128-bit constant operand keeps only its low 32 bits (so `a &
(2^64 - 1)` gives `a & 0xffffffff`, and LLVM lowers `zext(trunc a to
i64)` to that `and`), and 128-bit adds are wrong, with and without
optimization.

In Mojo, float formatting computes 64 x 64 -> 128-bit products with
`UInt128`, so `print` of a float64 printed wrong digits
(`tests/e2e/fp64_steps.mojo`, `tests/e2e/mul128.mojo`).

| IGC                                | Results wrong (of 168) |
|------------------------------------|------------------------|
| 2.41.5                             | 58                     |
| v2.42.14                           | 43 (adds right)        |
| master `bd0ad0a5`, release build   | 0                      |
| master `bd0ad0a5`, with assertions | aborts on every module |
| v2.43.9                            | 0                      |

Fixed in IGC `c31c33b17645` (adds, v2.42.0) and `cbeb3b419217` (constants,
v2.43.0). An IGC built with assertions still stops at two assertions a release
build skips, in `PeepholeTypeLegalizer` and `ProgramScopeConstantAnalysis`; the
candidate patch avoids the first and drops the second, which is stricter than
the code after it. Workaround: the compiler plugin splits every 128-bit integer
into two 64-bit halves before SPIR-V emission (arithmetic, logic, shifts,
compares, casts, phis, memory, globals and internal calls); anything it cannot
split is a compile error, so no 128-bit integer reaches IGC. Reproduction and
patch:
[`igc-int128-arithmetic`](../tests/intel-tools/igc-int128-arithmetic/).

## Level Zero: the packaged validation layer crashes at load

With Arch Linux's `level-zero-loader 1.32.0-1`, setting
`ZE_ENABLE_VALIDATION_LAYER=1` makes every Level Zero program segfault:
the package's `libze_validation_layer.so.1` reads through a null global
pointer in its static initialization. The v1.32.0 tag built from source
works, as do v1.34.0, v1.35.0 and master; a loader refuses a layer older
than itself.

Two of the layer's checkers report errors that are not there for the
driver plugin, in every version from 1.32.0 to master:
`ZE_ENABLE_HANDLE_LIFETIME` registers immediate command lists as closed,
so every append fails, and `ZEL_ENABLE_EVENTS_CHECKER` does not know
counter-based events. Parameter validation and the leak, performance and
certification checkers work.

Workaround: put a source-built loader and layer first on `LD_LIBRARY_PATH`.
Reproduction:
[`level-zero-validation-layer-crash`](../tests/intel-tools/level-zero-validation-layer-crash/).

## IGC 2.42 and 2.43: NaN in paged FA

MAX's paged FA returns NaN on IGC 2.42 and 2.43, and Qwen3.5's
greedy text turns into `Thinking!!!!...`. IGC 2.41.5 is right. Two separate
IGC changes cause it, and they overlap in time, so each release shows a
different set of wrong cases in `tests/graph/attention.py`:

| IGC                    | decode, cached 511 and 255 | decode batch | prefill 128, 512, 2048 | ragged batch |
|------------------------|----------------------------|--------------|------------------------|--------------|
| 2.41.5                 | right                      | right        | right                  | right        |
| v2.42.14               | right                      | NaN          | NaN                    | NaN          |
| master `bd0ad0a5`      | NaN                        | NaN          | right                  | right        |
| v2.43.9                | NaN                        | NaN          | right                  | right        |

compute-runtime is not involved: compute-runtime 26.35.39758 with IGC
v2.43.9 gives the same NaNs as v2.43.9's own compute-runtime 26.44.40734.8.
Reverting the struct-argument commit `0221e64d2` (the segfault below) does not
change them.

**Decode (IGC 2.43).** Bisecting release builds of IGC master between
`b50679576d` (right) and `bd0ad0a5` (wrong), with the decode case at 511
cached tokens as the test, gives `e88560098464` "Generate emulation libraries
for SPIR" (2026-10-07, first in v2.43.0). The commit regenerates IGC's
precompiled emulation libraries, among them 64-bit integer division. Its
parent `9d819d85e62d` gives the decode cases right; the commit, built in the
same tree, gives them NaN. In the decode stitch kernel, which merges the
partial results of the split decode, IGC's optimized LLVM IR is identical
before and after the commit; the code differs from the point where IGC
inlines its emulation of the kernel's two `sdiv i64` (a ceiling division by
a negative divisor). The emulated division itself is right: a kernel that
divides small 64-bit integers of every sign combination, per lane and
uniform across the sub-group, gives 0 wrong quotients on 2.41.5, on
`e88560098464` and on v2.43.9. The wrong result therefore comes from a later
IGC step acting on the inlined code; it has not been reduced.

**Prefill (IGC 2.42).** The same probes show the prefill, ragged and decode
batch cases wrong from before `516b6298dc` (2026-10-02) to `3942d3465f`
(2026-10-13) and right again from `2e369af623` (2026-10-19), so IGC master
both introduced and fixed that change before 2.43. v2.42.14 branched from
master on 2026-09-28 and has it; the commits that introduced and fixed it
are not bisected.

Use IGC 2.41.5: 2.42 and 2.43 both give wrong attention results.

Reproduction:
`plugins/intel/utils/run-max-py plugins/intel/tests/graph/attention.py` under
the IGC in question, with an empty `NEO_CACHE_DIR` so that kernels are compiled
again. It prints `PASS` or `FAIL` and the largest difference from a NumPy
reference for each case.

## IGC 2.43 segfaults on two struct kernel arguments passed by value

IGC v2.43.9 segfaults in `ResolveAggregateArguments` while building a
kernel that takes two structs by value and whose entry block starts with
an `extractvalue` of the first. Commit `0221e64d2` (v2.43.0) made the pass
handle struct arguments that are values; it builds each argument's
`alloca` with one IRBuilder placed before the entry block's first
instruction and erases each `extractvalue` as it replaces it, so when the
first instruction is one of them the builder points at a freed
instruction.

In MAX on v2.43.9 most graphs, the Qwen3.5-0.8B pipeline and every
benchmark segfault in `zeModuleCreate`: Mojo kernels take several structs
by value and often start that way.

| IGC                               | Two struct arguments |
|-----------------------------------|----------------------|
| 2.41.5, v2.42.14                  | builds               |
| master `bd0ad0a5`                 | segfault             |
| v2.43.9                           | segfault             |
| v2.43.9 with `fix.patch`          | builds               |
| v2.43.9 with `0221e64d2` reverted | builds               |

The candidate patch erases the `extractvalue`s after the last argument, nested
ones first. With it MAX's graph tests, sampling and benchmarks run on v2.43.9,
and 14 more of MAX's math tests pass than on 2.41.5, but paged decode attention
returns NaN (above). Reproduction and patch:
[`igc-struct-argument-segfault`](../tests/intel-tools/igc-struct-argument-segfault/).

## Copy engine page fault when device memory runs out

With compute-runtime 26.35.39758 and the copy-offload hint on its command
lists, the copy engine page-faulted (`Engine memory CAT error ...
class=bcs`, an engine reset, a timed-out job) and hung the GPU whenever
device memory ran out while serving; with the hint off the same case
ended in a clean OOM error. On 2026-11-15 the same stress run,
with the hint on, faulted neither on compute-runtime 26.35.39758 nor on
26.44.40734.8: both reported device OOM as an error. Why it no
longer faults on 26.35.39758, with the same kernel and firmware, is not
known.

Workaround, still the default: the driver plugin's queues leave the
copy-offload hint off, so queue copies run on the compute engine
(`MOJO_INTEL_COPY_OFFLOAD=1` sets it again; synchronous copies keep it),
and MAX's memory estimate reserves prompt activations on Level Zero
devices so serving does not hit OOM. No reproduction without
MAX.

## Other tools

Two bugs in LLVM's SPIR-V backend still matter: this tree carries a patch
for one and a workaround for the other.

| Issue                                                                    | Symptom                                                                                    | Affected                                                     | What this tree does                                                                                | Reproduction                                                                   |
|--------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| A constant GEP on an address-space cast reaches the instruction selector | assertion in `SPIRVInstructionSelector`; without assertions, invalid SPIR-V for some types | LLVM 23.1.3 (invalid SPIR-V), `6941494e2ae5` with assertions | `bazel/public-patches/llvm-spirv-ptr-add-on-cast.patch` selects it as instructions in the function | [llvm-spirv-ptr-add-on-cast](../tests/intel-tools/llvm-spirv-ptr-add-on-cast/) |
| `trunc <8 x i32> to <8 x i1>` feeding a `select`                         | "VReg's low-level type and register class have different sizes"                            | `9b2201bd0e35` with assertions; 23.1.3 builds it             | the compiler plugin rewrites such truncations into `icmp ne (and x, 1), 0`                         | [llvm-spirv-vector-trunc-i1](../tests/intel-tools/llvm-spirv-vector-trunc-i1/) |
