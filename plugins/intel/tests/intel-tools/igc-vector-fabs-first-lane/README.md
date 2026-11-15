# IGC: a leftover vector `fabs` is computed for its first lane only

`fabs4.ll` is a kernel written after the IR IGC gets from Mojo's float32
`j0` at width 4: each work-item loads a `<4 x float>`, takes `fabs` of it,
tests each lane for `|x| >= 2` and, from the bits with the sign cleared,
for "finite"; it stores `1/sqrt` of the lanes that pass and 0 for lanes
that are infinite. `run-fabs4.cpp` runs it on x = 2.5 in all four lanes of
65,536 work-items, where the right result is 0.632456 everywhere, three
ways: `alone` (the process's first kernel), `after-fill` (right after
`fill.cl`'s kernel, which leaves +inf in many registers) and `after-idle`
(the same with 100 ms of idle GPU between them). `run.sh` first builds the
kernel with IGC's dumps and prints the vector `fabs` left in IGC's final
IR and the vISA that computes `|x|`.

```sh
sh run.sh
```

It needs `clang` with the `spirv64` target, `spirv-val`, `clang++`,
`ocloc` and a `bmg-g31` GPU on Level Zero.

## Expected and actual

Every lane should be right whatever ran before. With IGC 2.41.5 lanes 1-3
read `|x|`'s bits from registers the kernel never writes, so their result
is whatever an earlier kernel left there:

```text
vector fabs left in IGC's final IR:
  %26 = call <4 x float> @llvm.fabs.v4f32(<4 x float> %17)
...
    mov (M1, 32) V0078(0,0)<1> (abs)V0069(0,0)<1;1,0>
...
    cmp.eq (M1, 32) P12 V0078(0,0)<1;1,0> 0x7f800000:f
    cmp.eq (M1, 32) P13 V0078(2,0)<1;1,0> 0x7f800000:f
    cmp.eq (M1, 32) P14 V0078(4,0)<1;1,0> 0x7f800000:f
    cmp.eq (M1, 32) P15 V0078(6,0)<1;1,0> 0x7f800000:f
...
alone                       want 0.632456; wrong of 65536 per lane: lane 0: 0 lane 1: 0 lane 2: 0 lane 3: 0
after-fill                  want 0.632456; wrong of 65536 per lane: lane 0: 0 lane 1: 65536 (the first is 0) lane 2: 65536 (the first is 0) lane 3: 65536 (the first is 0)
after-idle                  want 0.632456; wrong of 65536 per lane: lane 0: 0 lane 1: 0 lane 2: 0 lane 3: 0
after-fill -cl-opt-disable  want 0.632456; wrong of 65536 per lane: lane 0: 0 lane 1: 0 lane 2: 0 lane 3: 0
```

The `mov` writes the first lane of `V0078` (at SIMD32 a lane takes two
rows); the compares read rows 2, 4 and 6 too. With `-cl-opt-disable` no
vector `fabs` is left and the vISA has one `mov ... (abs)` per lane.

| IGC               | Vector `fabs` in the final IR | `after-fill`, default build | `-cl-opt-disable` |
|-------------------|-------------------------------|-----------------------------|-------------------|
| 2.41.5            | yes                           | lanes 1-3 wrong             | right             |
| v2.42.14          | yes                           | lanes 1-3 wrong             | right             |
| master `bd0ad0a5` | no                            | right                       | right             |
| v2.43.9           | no                            | right                       | right             |

`alone` and `after-idle` are right on every version. The result is the
same with compute-runtime's direct submission off
(`NEOReadDebugKeys=1 EnableDirectSubmission=0`). Turning IGC's vectorizer
off does not change it.

## Fixed in

IGC commit `307d24a474d5`, "Scalarize vector fabs/abs in
ScalarizerCodeGen" (2026-10-16), first in v2.43.0.
