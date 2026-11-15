# LLVM SPIR-V backend: a constant GEP on a cast reaches the selector

Three modules in LLVM IR, each valid (the verifier accepts them):

- `sunk-addrspacecast.ll`: a load, in a second block, from
  `inttoptr (add (ptrtoint (ptr addrspace(3) @g), 24) to ptr addrspace(4))`;
  reduced from the kernel of MAX's `test_metal_memory.mojo`.
- `equal-pointee-types.ll`: the same with `@g` an `i8`, so that the cast's
  pointee type equals the load's.
- `global-initializer.ll`: a global initialized with
  `getelementptr (i8, addrspacecast (ptr addrspace(1) @g to ptr addrspace(4)), 8)`.

`run.sh` runs `llc -mtriple=spirv64-unknown-unknown` on each, then
`spirv-val` and `ocloc` (`bmg-g31`) on what it builds.

```sh
sh run.sh [LLC]
```

`LLC` defaults to the `llc` on `PATH`; an assertion build shows the
assertion. It needs `llc`, `spirv-val` and `ocloc`; no GPU.

## Expected and actual

Each module should give valid SPIR-V. With llc 23.1.3 (no assertions):

```text
== sunk-addrspacecast
llc: built
spirv-val: valid
ocloc: built
== equal-pointee-types
llc: built
error: line 26: ID '15[%15]' has not been defined
  %12 = OpSpecConstantOp %_ptr_Generic_uchar InBoundsPtrAccessChain %15 %ulong_24

ocloc: built
== global-initializer
llc: built
error: line 37: ID '25[%25]' has not been defined
  %21 = OpSpecConstantOp %_ptr_Generic_uchar InBoundsPtrAccessChain %25 %16 %ulong_8

ocloc: exit 139
```

An LLVM built with assertions (`6941494e2ae5`) stops on all three in
`SPIRVInstructionSelector.cpp` (`spvSelect`, `G_PTR_ADD`):

```text
Assertion `((*II).getOpcode() == TargetOpcode::G_GLOBAL_VALUE || (*II).getOpcode() == TargetOpcode::COPY || (*II).getOpcode() == SPIRV::OpVariable || (*II).getOpcode() == SPIRV::OpUntypedVariableKHR) && getImm(I.getOperand(2), MRI)' failed.
```

| LLVM                                             | `sunk-addrspacecast` | `equal-pointee-types` | `global-initializer`              |
|--------------------------------------------------|----------------------|-----------------------|-----------------------------------|
| 23.1.3, no assertions                            | valid                | invalid SPIR-V        | invalid SPIR-V; `ocloc` segfaults |
| `6941494e2ae5` with assertions                   | assertion            | assertion             | assertion                         |
| the same with `llvm-spirv-ptr-add-on-cast.patch` | valid                | valid                 | assertion (kept on purpose)       |

## Cause and patch

`G_PTR_ADD` reaches the SPIR-V instruction selector only from constant GEP
expressions, and the selector assumes their base is a global variable. In
`sunk-addrspacecast.ll`, SPIRVRegularizer turns the constant expression into
instructions, then CodeGenPrepare's address sinking matches them as the address
`[24 + @g]` and sinks it into the load's block as the constant

```llvm
getelementptr (i8, ptr addrspace(4)
  addrspacecast (ptr addrspace(3) @g to ptr addrspace(4)), i64 24)
```

which the IRTranslator turns into
`G_PTR_ADD (G_ADDRSPACE_CAST (G_GLOBAL_VALUE @g)), 24`. Without the branch the
address is not sunk and nothing asserts. That rewrite also changes the program's
meaning on Intel GPUs, where a Workgroup pointer's integer is an offset in its
window: `inttoptr(ptrtoint(p) + 24)` from Workgroup to Generic is not
`addrspacecast(p) + 24`.

`bazel/public-patches/llvm-spirv-ptr-add-on-cast.patch` (applied to
Modular's LLVM in this tree) selects such a `G_PTR_ADD` as instructions in
the function: an `OpBitcast` to an `i8` pointer, an `OpPtrAccessChain` by
the byte offset, and an `OpBitcast` to the result type. It keeps the check
for global initializers. The standard library no longer emits the
`inttoptr` pattern on Intel, so no Mojo program measured reaches the
patched path today; the patch stays for any other `inttoptr`/`ptrtoint`
round trip.
