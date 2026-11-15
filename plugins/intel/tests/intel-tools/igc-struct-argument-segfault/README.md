# IGC 2.43: segfault on two struct kernel arguments passed by value

Three kernels in hand-written LLVM IR, compiled with
`clang -target spirv64` (each passes `spirv-val`) and built for `bmg-g31`
with `ocloc`:

- `two_structs.ll`: two `{ i64, i64 }` arguments by value (SPIR-V
  `OpTypeStruct` parameters); the entry block starts with an
  `extractvalue` of the first.
- `guarded.ll`: the same with another instruction first.
- `one_struct.ll`: one such argument, the entry block starting with its
  `extractvalue`.

```sh
sh run.sh
```

It needs `clang` with the `spirv64` target, `spirv-val` and `ocloc`; no
GPU. `run.sh` uses the `ocloc` on `PATH`, which loads its own IGC.

## Expected and actual

Every module should build (`ocloc` exit 0). IGC 2.41.5 builds all three:

```text
two_structs: ocloc exit 0
guarded: ocloc exit 0
one_struct: ocloc exit 0
```

| IGC                                      | `two_structs`                                      | `guarded` | `one_struct` |
|------------------------------------------|----------------------------------------------------|-----------|--------------|
| 2.41.5                                   | 0                                                  | 0         | 0            |
| v2.42.14                                 | 0                                                  | 0         | 0            |
| master `bd0ad0a5`, built with assertions | 134: the segfault, turned into an abort by `ocloc` | 0         | 0            |
| v2.43.9                                  | 139 (segfault)                                     | 0         | 0            |
| v2.43.9 with `fix.patch`                 | 0                                                  | 0         | 0            |
| v2.43.9 with `0221e64d2` reverted        | 0                                                  | 0         | 0            |

In MAX on IGC v2.43.9 most graphs, the Qwen3.5-0.8B pipeline and every
benchmark segfault in `zeModuleCreate`: Mojo kernels take several structs
by value and often start with such an `extractvalue`. The backtrace:

```text
#3 llvm::SymbolTableListTraits<llvm::Instruction, ...>::addNodeToList(llvm::Instruction*)
#4 llvm::Instruction::insertInto(llvm::BasicBlock*, ...)
#7 llvm::IRBuilderBase::CreateAlloca
#9 IGC::ResolveAggregateArguments::runOnFunction  AggregateArguments.cpp:312
#10 IGC::ResolveAggregateArgumentsNPM::run        AggregateArguments.cpp:395
#14 IGC::CommonOCLBasedPasses                      UnifyIROCL.cpp:710
```

## Cause and patch

IGC commit `0221e64d2`, "Support struct arguments passed by value in
SPIR-V" (first in v2.43.0), made `ResolveAggregateArguments` handle struct
arguments that are values, not only `byval` pointers. `runOnFunction`
builds every argument's `alloca` and stores with one IRBuilder placed
before the entry block's first instruction, and the new
`replaceStructValueUses` replaces each `extractvalue` of the argument with
a load and erases it at once. When the first instruction is such an
`extractvalue`, the builder is left pointing at a freed instruction, and
the next argument's `CreateAlloca` links into it.

`fix.patch` collects the `extractvalue`s and erases them after the last
argument, nested ones first. It is against IGC v2.43.9 (MIT, like IGC).
With it MAX's graph tests,
sampling and the benchmarks no longer crash on v2.43.9; paged decode
attention still returns NaN there, a separate regression
(`../../../docs/intel-tools-issues.md`).
