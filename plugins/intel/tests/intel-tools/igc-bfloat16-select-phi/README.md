# IGC: wrong bits from BF16 selects, phis and rounding

`bf16_moves.spvasm` holds one small kernel per case, in hand-written SPIR-V
with `SPV_KHR_bfloat16`. Every kernel takes a buffer of 16-bit words: p[0]
= a = 1.5 (`0x3fc0`), p[1] = b = 2.5 (`0x4020`), p[2] = a flag; results go
to p[16] and p[17]. `run-words.cpp` runs one kernel on one work-item,
built with the driver plugin's options and again with `-cl-opt-disable`
added. `run.sh` runs every kernel with the flag set and clear.

```sh
sh run.sh
```

It needs `spirv-as`, `spirv-val`, `clang++` and a `bmg-g31` GPU on Level
Zero.

## Expected and actual

Selects and phis should give one of their inputs' bits; the rounding
kernels compute `bf16(bf16(1.5 * 1.5) * 1.5)` = `0x4058` (3.375), and
`0x4010` for the first rounding where it is stored too. With IGC 2.41.5:

| Kernel (flag)        | Expected    | Default build | `-cl-opt-disable` |
|----------------------|-------------|---------------|-------------------|
| `select` (0)         | `4020 3fc0` | `ffff 4020`   | right             |
| `select_const` (1)   | `3fc0 7f80` | `fe00 7f80`   | right             |
| `phi_const` (1)      | `3fc0`      | `fe00`        | right             |
| `select_convert` (1) | `3fc0 7f80` | `fe00 7f80`   | `0000 7f80`       |
| `select_convert` (0) | `7f80 3fc0` | `7f80 ffff`   | `7f80 7f80`       |
| `round_once_used`    | `4058`      | `fe40`        | right             |
| `round_twice`        | `4058 4010` | `ffff 0000`   | right             |
| `round_across` (1)   | `4058 4010` | `ffff ffff`   | right             |

Loads, stores, conversions, constants, `OpBitcast` between `ushort` and
`bfloat16`, and the same select on 16-bit integers (`copy`, `convert`,
`store_const`, `convert_const`, `bitcasts`, `select_i16`) are right.
`ffff` and `fe00` are the bits of neither input. In the rounding kernels
IGC's optimizer rewrote `fptrunc(fmul(fpext a, 1.5))` into BF16
arithmetic (`fmul bfloat a, 0xR3FC0` in its `_optimized.ll` dump), which
the device then computes wrongly; `select_convert` is wrong with
`-cl-opt-disable` too, so it is not only the optimizer.

| IGC               | Selects and phis | Rounding |
|-------------------|------------------|----------|
| 2.41.5            | wrong            | wrong    |
| v2.42.14          | right            | right    |
| master `bd0ad0a5` | right            | right    |
| v2.43.9           | right            | right    |

On the fixed versions each kernel gives the same bits with both option
sets.

## Fixed in

IGC commit `81dae221ed3a`, "Changes in code." (2026-09-14), first in
v2.42.0: vISA's `HWConformity::fixBFMixedMode` no longer leaves SIMD1
BF16 mixed-mode instructions alone, and treats an operand as packed or
unpacked only when it is GRF-aligned.
