# IGC: float16 `log2` on a 4-wide vector gives wrong lanes

`log2_v4half.spvasm` is one kernel as LLVM's SPIR-V backend emits Mojo's
`log2` on `SIMD[float16, 4]`: load `<4 x half>`, `OpFConvert` to
`<4 x float>`, OpenCL `log2`, `OpFConvert` back, store; the same for
`exp2`. `run-log2.cpp` runs it through Level Zero on x = 0.5, 0.75, 1,
1.25, built with the driver plugin's options and again with
`-cl-opt-disable` added, then the same with IGC's vectorizer turned off
(`IGC_EnableVectorizer=0`).

```sh
sh run.sh
```

It needs `spirv-as`, `spirv-val`, `clang++` and a `bmg-g31` GPU on Level
Zero.

## Expected and actual

The right results are -1, -0.415, 0 and 0.322. With IGC 2.41.5 and its
default optimization every lane is wrong: the result is `x` for x >= 1
and 0 below. `exp2` in the same kernel is right.

```text
== log2_v4half
lane  x       log2: host      default    -cl-opt-disable
   0  0.5               -1           0          -1  <- wrong
   1  0.75       -0.415039           0   -0.415039  <- wrong
   2  1                  0           1           0  <- wrong
   3  1.25        0.322021        1.25    0.322021  <- wrong
default build: 4 of 4 lanes wrong
== log2_v4half, IGC_EnableVectorizer=0
...
default build: 0 of 4 lanes wrong
```

| IGC (compute-runtime)                      | Default build                      | `-cl-opt-disable` | `IGC_EnableVectorizer=0` |
|--------------------------------------------|------------------------------------|-------------------|--------------------------|
| 2.41.5 (26.35.39758)                       | 4 of 4 lanes wrong                 | right             | right                    |
| v2.42.14 (26.40.40196.15)                  | 2 of 4 wrong: lanes 1 and 3 give 0 | right             | right                    |
| master `bd0ad0a5` (master `ee495cce`)      | 2 of 4 wrong                       | right             | right                    |
| v2.43.9 (26.44.40734.8)                    | 2 of 4 wrong: lanes 1 and 3 give 0 | right             | right                    |
| master `bd0ad0a5` with `fix.patch` (26.35) | right                              | right             | -                        |

Running each IGC with the other compute-runtime gives the IGC's result,
so the change is IGC's. Width 8 is wrong too; width 1 and width 32 (which
the SPIR-V path scalarizes) are right, and so is float32 `log2` at every
width.

## Cause and patch

IGC's vectorizer turns the four `fptrunc float to half` into one
`fptrunc <4 x float> to <4 x half>`. `EmitPass::FPTrunc`
(`IGC/Compiler/CISACodeGen/EmitVISAPass.cpp`) writes element `i` of the
result at row `i / 2`, column `(i % 2) * 16`, the layout of a per-lane
value, also when the result is uniform. Here it is uniform (one vector
per work-item, loaded from one address), so the vISA writes past the end
of the variable while the store reads its first four elements. Two IGC
commits made fewer lanes wrong (`f3f3dc640b76`, "IGCVectorizer: Placement
of intrinsic instructions fixed", first in v2.42.0; `8fa5a563d7f8`,
"IGCVectorizer: Cast takes proper insert point", first in v2.43.0 and
picked onto the 2.42 branch); neither fixes it.

`fix.patch` writes element `i` at sub-register `i` when the destination is
uniform. It is against IGC master `bd0ad0a5` (MIT, like IGC). With it, a release
build of `bd0ad0a5` gets every lane right. IGC's lit tests (`check-igc`) have no
new failures with it and the patches of `igc-vector-compare-crash`,
`igc-nested-struct-arguments` and `igc-int128-arithmetic` applied together. Why
`exp2` comes out right despite the same pattern was not examined.
