# compute-runtime reads a `%s` argument as a host address

`kernels.cl` has three kernels that print a string with `printf`'s `%s`:

- `literal_string`: a string literal;
- `shared_buffer_string`: a string the kernel writes into its buffer,
  which `utils/l0/run.cpp` allocates as shared memory;
- `private_string`: a string the kernel builds in private memory.

`run.sh` builds them with `clang -target spirv64` and runs each with
`utils/l0/run.cpp`, a Level Zero runner for one kernel.

```sh
sh run.sh
```

It needs `clang` with the `spirv64` target, `spirv-val`, `c++` and a
`bmg-g31` GPU on Level Zero.

## Expected and actual

Each kernel should print its string. The third crashes the host process:

```text
== literal_string
literal: hello
buffer[0] = 7
l0-run literal_string: exit 0
== shared_buffer_string
shared buffer: hi
buffer[0] = 7
l0-run shared_buffer_string: exit 0
== private_string
l0-run private_string: exit 139
```

The segfault is in `zeCommandListHostSynchronize`, where compute-runtime
formats the printf buffer: `__snprintf_chk` called from
`libze_intel_gpu.so.1` (`gdb`). IGC places string literals in a section
that compute-runtime maps at the same address on the host, and
compute-runtime's `PrintFormatter::printStringToken`
(`shared/source/program/print_formatter.cpp`) passes whatever address the
kernel stored for `%s` to `snprintf` as a host `char *`. A string in
shared memory happens to work because its device address is also a host
address; a string in private memory, as here, segfaults the process.

| compute-runtime (IGC)   | `literal_string` | `shared_buffer_string` | `private_string` |
|-------------------------|------------------|------------------------|------------------|
| 26.35.39758 (2.41.5)    | right            | right                  | segfault (139)   |
| 26.44.40734.8 (v2.43.9) | right            | right                  | segfault (139)   |

compute-runtime 26.40.40196.15 and master `ee495cce` have the same
`printStringToken` (read in the source, not run).
