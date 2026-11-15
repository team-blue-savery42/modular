# IGC: a value merged by an aggregate phi is lost

`bounds_loop.spv` is the module the Mojo compiler plugin built for
`tests/e2e/bounds_loop.mojo` before it split aggregate phis. The kernel
runs `for i, byte in enumerate(s.bytes())` over a 3-byte string and stores
`i` on each iteration. The iterator's next value is an `Optional`, a phi
of `{ { i64, [8 x i8] }, i8 }` (`OpPhi` of a struct type) whose "none"
side is a constant with an undefined inner struct.
`bounds_loop_split.spv` is the same kernel with the phi split into one
phi per field; `bounds_loop_assert.spv` is the first with bounds checks
on. All three pass `spirv-val`. `run.sh` builds each with `ocloc` with
IGC's defaults and with `-cl-opt-disable`, then runs the first two
through Level Zero (`run-bounds.cpp`) both ways. `NOGPU=1` skips the runs.

```sh
sh run.sh
NOGPU=1 sh run.sh
```

It needs `spirv-val`, `ocloc`, `clang++` and, for the runs, a `bmg-g31`
GPU on Level Zero.

## Expected and actual

The kernel should record the indices `0 1 2`. The buffer starts as -1, so
`-1 -1 -1` means the stores never happened. With IGC 2.41.5:

```text
== bounds_loop_assert.spv
spirv-val: valid
ocloc : exit 0
ocloc -options -cl-opt-disable : exit 139
...
== run bounds_loop.spv
options "-ze-intel-enable-auto-large-GRF-mode -cl-fp32-correctly-rounded-divide-sqrt":
  iterations 3 (expected 3)
  indices -1 -1 -1 (expected 0 1 2)  <- wrong
  bytes that differ 3 (expected 3)
options "-ze-intel-enable-auto-large-GRF-mode -cl-fp32-correctly-rounded-divide-sqrt -cl-opt-disable":
  iterations 3 (expected 3)
  indices 0 0 0 (expected 0 1 2)  <- wrong
  bytes that differ 3 (expected 3)
run-bounds: exit 1
== run bounds_loop_split.spv
...
  indices 0 1 2 (expected 0 1 2)
```

With IGC's defaults the index is already `undef` in IGC's
`afterUnification` dump and the stores of `i` are gone. With
`-cl-opt-disable` IGC keeps the phi's value in a private slot, but nothing
stores the index into it.

| IGC               | `bounds_loop`, default / `-cl-opt-disable` | `bounds_loop_split` | `ocloc -cl-opt-disable` on `bounds_loop_assert` |
|-------------------|--------------------------------------------|---------------------|-------------------------------------------------|
| 2.41.5            | `-1 -1 -1` / `0 0 0`                       | right               | exit 139 (segfault)                             |
| v2.42.14          | `-1 -1 -1` / `0 0 0`                       | right               | exit 0                                          |
| master `bd0ad0a5` | right / right                              | right               | exit 0                                          |
| v2.43.9           | right / right                              | right               | exit 0                                          |

## Fixed in

- The lost index: IGC commit `0221e64d2c48`, "Support struct arguments
  passed by value in SPIR-V" (2026-10-20), first in v2.43.0. It changes
  how IGC lowers struct values; which part fixes the phi was not
  narrowed.
- The `ocloc` segfault: IGC commit `56fc303e415f`, "Fix no-opt
  compilation failures" (2026-09-10), first in v2.42.0.
