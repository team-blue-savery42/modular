# IGC: 128-bit integer arithmetic is wrong

Three kernels in hand-written LLVM IR, compiled with
`clang -target spirv64` and `SPV_INTEL_arbitrary_precision_integers`
(`OpTypeInt 128`). Work-item i loads 128-bit integers from `in` and
stores 128-bit results to `out`:

- `mul128.ll`: multiplies, of whole operands and of operands masked to
  their low 64 bits.
- `ops.ll`: `and` and `xor` with 128-bit constants (2^64 - 1 and
  0x0123456789abcdeffedcba9876543210), shifts by 64, `zext(trunc)` and
  multiplies.
- `vars.ll`: the same `and` with the mask loaded from memory, a
  zero-extension, shifts, a multiply and two adds.

`run-mul128.cpp` launches 4 work-items, built with the driver plugin's
options and again with `-cl-opt-disable` added, and compares each result
with the host's `unsigned __int128`. clang 23 names the extension
`SPV_ALTERA_arbitrary_precision_integers`, which IGC 2.41 rejects;
`run.sh` renames it to the older `SPV_INTEL_` name (the same capability)
in the binary.

```sh
sh run.sh
```

It needs `clang` with the `spirv64` target, `python3`, `spirv-val`,
`spirv-dis`, `clang++` and a `bmg-g31` GPU on Level Zero.

## Expected and actual

Every result should match the host's. With IGC 2.41.5, in both modes:

```text
  item 0 a & (2^64-1)               got 0x00000000000000000000000006ddc391 expected 0x00000000000000003c89ed7106ddc391  <- wrong
  item 2 a ^ 0x0123..3210           got 0x00000000000000000000000076543213 expected 0x0123456789abcdeffedcba9876543213  <- wrong
...
wrong 8 of 24
wrong 35 of 80
wrong 15 of 64
```

A 128-bit constant operand keeps only its low 32 bits, and 128-bit adds
are wrong. A multiply or `and` of two loaded values, shifts by 64 and
`zext` of a loaded `i64` are right. LLVM lowers `zext(trunc a to i64)` to
the `and` with 2^64 - 1, so that is wrong too.

| IGC                                               | `mul128` | `ops`    | `vars`   |
|---------------------------------------------------|----------|----------|----------|
| 2.41.5                                            | 8 of 24  | 35 of 80 | 15 of 64 |
| v2.42.14                                          | 8 of 24  | 35 of 80 | 0 of 64  |
| master `bd0ad0a5`, release build                  | 0 of 24  | 0 of 80  | 0 of 64  |
| master `bd0ad0a5`, built with assertions          | aborts   | aborts   | aborts   |
| v2.43.9                                           | 0 of 24  | 0 of 80  | 0 of 64  |
| master `bd0ad0a5` with assertions and `fix.patch` | 0 of 24  | 0 of 80  | 0 of 64  |

## Fixed in, and the patch

- 128-bit adds: IGC commit `c31c33b17645`, "Legalize add and icmp on
  integers wider than 64 bits" (2026-09-03), first in v2.42.0.
- 128-bit constants: IGC commit `cbeb3b419217`, "Fix binary operations on
  i128 with constant in PeepholeTypeLegalizer" (2026-10-12), first in
  v2.43.0. `mul`, `and`, `or` and `xor` slice a constant operand with
  `extractBits()` instead of a constant expression that kept the illegal
  type and lost all but the low 32 bits.

A build of IGC with assertions still stops on every module with 128-bit
values: `PeepholeTypeLegalizer::cleanupBitCastInst` asserts that the
operand of `bitcast i128 %v to <4 x i32>` is a legal integer, and
`ProgramScopeConstantAnalysis::addData` asserts `bitWidth <= 64` for a
128-bit integer in a global initializer, although the code after it
copies all the bytes. A release build skips both and is right.
`fix.patch` replaces such a load with a load of the legal vector type,
folds a bitcast of a 128-bit constant, and drops the `addData` assertion.
It is against IGC master `bd0ad0a5` (MIT, like IGC) and matters for
assertion builds only.
