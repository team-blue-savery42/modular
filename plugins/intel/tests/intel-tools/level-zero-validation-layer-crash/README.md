# Level Zero: the packaged validation layer crashes at load

`run.sh` builds `kernels.cl` (one kernel that stores a word) and
`utils/l0/run.cpp` (a Level Zero runner for one kernel), then runs the
kernel without and with Level Zero's validation layer
(`ZE_ENABLE_VALIDATION_LAYER=1`). Put another loader and layer first on
`LD_LIBRARY_PATH` to try them.

```sh
sh run.sh
LD_LIBRARY_PATH=<level-zero install>/lib sh run.sh
```

It needs `clang` with the `spirv64` target, `c++` and a GPU on Level Zero.

## Expected and actual

Both runs should store the word. With Arch Linux's `level-zero-loader
1.32.0-1` package the second run segfaults:

```text
== without the validation layer
buffer[0] = 1
exit 0
== ZE_ENABLE_VALIDATION_LAYER=1 ZE_ENABLE_PARAMETER_VALIDATION=1
exit 139
```

The crash is in `libze_validation_layer.so.1`'s static initialization
while the loader opens it: it reads through a null global pointer. Mixing
loaders and layers shows it is that package's layer binary, not the
loader and not the source:

| Layer                      | Loaders it works with (of 1.32.0 package, 1.32.0 source, 1.34.0, 1.35.0, master) |
|----------------------------|----------------------------------------------------------------------------------|
| 1.32.0 package             | none: crashes with every one                                                     |
| v1.32.0 built from source  | 1.32.0; newer loaders refuse it                                                  |
| v1.34.0                    | 1.32.0 and 1.34.0; newer loaders refuse it                                       |
| v1.35.0, master `52d92f8b` | all                                                                              |

A loader newer than its layer refuses it: it fails `zeInitDrivers` with
`ZE_RESULT_ERROR_UNSUPPORTED_VERSION`. The same with compute-runtime
26.35.39758 and 26.40.40196.15; the v1.34.0 loader and layer also work
with compute-runtime 26.44.40734.8.

## Other validation-layer limits

Read in the layer's source from 1.32.0 to master `52d92f8b`, and measured
on 1.35.0:

- `ZE_ENABLE_HANDLE_LIFETIME` registers immediate command lists as closed
  (`zeCommandListCreateImmediate`), so every append to one fails with
  `ZE_RESULT_ERROR_INVALID_ARGUMENT`. It also does not register build
  logs from `zeModuleInspectLinkageExt` (`ZE_RESULT_ERROR_INVALID_NULL_HANDLE`).
- `ZEL_ENABLE_EVENTS_CHECKER` does not know counter-based events and
  warns on every barrier that uses them.

The driver plugin uses immediate command lists and counter-based events,
so these two checkers report errors that are not there. Parameter
validation and the leak, performance and certification checkers work.
