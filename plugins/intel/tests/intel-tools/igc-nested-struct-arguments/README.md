# IGC: nested struct arguments and return values are passed wrongly

Three kernels in hand-written LLVM IR, compiled with
`clang -target spirv64`, move `{ i64 147, i64 81, { ptr name, i64 3 } }`
by value and store its four leaves to `out[0..3]`:

- `nested.ll`: the kernel passes it to a `noinline` function.
- `nested_ret.ll`: a `noinline` function returns it.
- `kernarg.ll`: the kernel takes it as an argument (an `OpTypeStruct`
  parameter).

`run-nested.cpp` runs each through Level Zero, built with the driver
plugin's options and again with `-cl-opt-disable` added. `run.sh` also
prints the vISA of `nested.ll`'s kernel up to the call, and builds
`nested_return.spv` (the module the Mojo compiler plugin built for
`tests/e2e/nested_return.mojo` before it flattened such structs) with
`ocloc`.

```sh
sh run.sh
```

It needs `clang` with the `spirv64` target, `spirv-val`, `clang++`,
`ocloc` and a `bmg-g31` GPU on Level Zero.

## Expected and actual

Each run should store 147, 81, the address of `name` and 3. With IGC
2.41.5 the inner struct's two fields both read 147 in `nested` and
`nested_ret`, and the module does not build with `-cl-opt-disable`:

```text
== run nested
options "-ze-intel-enable-auto-large-GRF-mode -cl-fp32-correctly-rounded-divide-sqrt":
  line   0x0000000000000093  expected 0x0000000000000093
  column 0x0000000000000051  expected 0x0000000000000051
  name   0x0000000000000093  expected 0xffffffffff9ffc00  <- wrong
  length 0x0000000000000093  expected 0x0000000000000003  <- wrong
zeModuleCreate failed: 0x70000004
...
== run kernarg
...
  name   0xffffffffff9ffc00  expected 0xffffffffff9ffc00
  length 0xffffffffff9ffc00  expected 0x0000000000000003  <- wrong
```

In the vISA the length 3 goes into field 1 of the outer struct's register
(then overwritten with 81), and field 3, the length, is copied from
field 1. (`0x70000004` is `ZE_RESULT_ERROR_MODULE_BUILD_FAILURE`.)

| IGC                                      | `nested`, default / `-cl-opt-disable` | `nested_ret`     | `kernarg`                       | `nested_return.spv`, `-cl-opt-disable` |
|------------------------------------------|---------------------------------------|------------------|---------------------------------|----------------------------------------|
| 2.41.5                                   | wrong / no build                      | wrong / no build | wrong / wrong                   | exit 139 (segfault)                    |
| v2.42.14                                 | wrong / right                         | wrong / right    | wrong / wrong                   | exit 0                                 |
| master `bd0ad0a5`, built with assertions | right / right                         | right / right    | right / IGC segfault            | exit 0                                 |
| v2.43.9                                  | right / right                         | right / right    | right / no build (`0x70000004`) | exit 0                                 |
| master `bd0ad0a5` with `fix.patch`       | right / right                         | right / right    | right / right                   | -                                      |

## Fixed in, and the patch

- With IGC's defaults all three are fixed by IGC commit `0221e64d2c48`,
  "Support struct arguments passed by value in SPIR-V" (2026-10-20), first
  in v2.43.0. It flattens struct kernel arguments passed by value into the
  payload, as it does `byval` struct arguments.
- With `-cl-opt-disable`, `nested` and `nested_ret` build since IGC
  commit `56fc303e415f`, "Fix no-opt compilation failures" (2026-09-10),
  first in v2.42.0.
- The same commit `0221e64d2c48` made `kernarg` fail with
  `-cl-opt-disable`. `ResolveAggregateArguments` now rebuilds such an
  argument in an `alloca`, but `PrivateMemoryUsageAnalysis`, which runs
  before it and decides whether the kernel gets the `PRIVATE_BASE`
  implicit argument, still counts only `byval` struct arguments. With
  optimization the `alloca` is promoted away; without it
  `PrivateMemoryResolution::resolveAllocaInstructions` reads a null
  `PRIVATE_BASE` (a segfault in an assertion build, a build failure in a
  release build).

`fix.patch` (against IGC master `bd0ad0a5`; MIT, like IGC) makes
`PrivateMemoryUsageAnalysis` count a struct argument passed by value as it
counts a `byval` struct. With it `kernarg` is right in both modes, in release
and assertion builds.
