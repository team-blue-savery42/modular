# IGC rejects `SPV_KHR_abort`

`trap.ll` is a kernel that calls `llvm.trap` when its input is 0. With
`SPV_KHR_abort` enabled, LLVM's SPIR-V backend lowers the trap to
`OpAbortKHR`; the module passes `spirv-val`. `run.sh` builds it for
`bmg-g31` with `ocloc`.

```sh
sh run.sh
```

It needs `clang` with the `spirv64` target, `spirv-val`, `spirv-dis` and
`ocloc`; no GPU.

## Expected and actual

The module should build. Every IGC tried rejects the extension:

```text
spirv-val: valid
               OpCapability AbortKHR
               OpExtension "SPV_KHR_abort"
               OpAbortKHR %uint %uint_4294967295
ocloc: exit 10
InvalidModule: Invalid SPIR-V module: input SPIR-V module uses unknown extension 'SPV_KHR_abort'
```

| IGC               | `ocloc` exit | Message           |
|-------------------|--------------|-------------------|
| 2.41.5            | 10           | unknown extension |
| v2.42.14          | 10           | unknown extension |
| master `bd0ad0a5` | 245          | the same message  |
| v2.43.9           | 245          | the same message  |

Without the extension the backend drops `llvm.trap`, so a kernel cannot
stop in a way Level Zero reports.
