# LLVM SPIR-V backend: `trunc <8 x i32> to <8 x i1>` feeding a `select`

`trunc-v8i1.ll` truncates an 8-lane vector to `i1` and uses the result as
a `select` mask. `run.sh` runs `llc -O2 -mtriple=spirv64-unknown-unknown`
on it and checks what it builds with `spirv-val`.

```sh
sh run.sh [LLC]
```

`LLC` defaults to the `llc` on `PATH`; the failure needs an assertion
build. It needs `llc` and `spirv-val`; no GPU.

## Expected and actual

`llc` should build valid SPIR-V. An LLVM built with assertions at
`9b2201bd0e35` stops:

```text
LLVM ERROR: VReg's low-level type and register class have different sizes: %21:viid(<8 x s32>) = OpBitwiseAndV %3:type(s64), %11:viid(s64), %24:viid(s64) (in function: k)
```

The `OpBitwiseAndV` that lowers the `trunc` to `and x, 1` gets 64-bit
scalar operands while its result is `<8 x s32>`.

| LLVM                           | 2 lanes | 4 lanes | 8 lanes                                |
|--------------------------------|---------|---------|----------------------------------------|
| `9b2201bd0e35` with assertions | builds  | builds  | LLVM ERROR (above)                     |
| 23.1.3, no assertions          | -       | builds  | builds; `spirv-val` accepts the output |

With llc 23.1.3 `run.sh` prints `llc: built` and `spirv-val: valid`.
Whether the bug came in after 23.1 or only an assertion build catches it
is not known. The compiler plugin rewrites such truncations into
`icmp ne (and x, 1), 0` before the backend runs.
