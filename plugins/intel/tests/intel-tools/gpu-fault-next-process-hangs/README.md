# After a GPU page fault, the next process's first wait never completes

`run.sh` builds `utils/l0/fault-absorb` (a plain Level Zero program, also
the throwaway process `utils/check-max-gpu` runs after a faulting test)
and runs four processes in a row: `add`, `fault`, `add`, `add`. `add` adds
two 1024-float vectors; `fault` stores through an unmapped address
(`0x0000100000000000`). Each uses an in-order immediate command list on
ordinal 0 and waits with `zeCommandListHostSynchronize` for at most 10 s
(exit status 2 when the wait times out). Then it prints the kernel log's
fault lines for the run.

```sh
sh run.sh
```

It needs `clang` with the `spirv64` target, `c++`, a `bmg-g31` GPU on
Level Zero and read access to the kernel log (`journalctl -k`). The fault
resets the GPU's compute engine for every process using it: run it only
on a GPU nothing else is using.

## Expected and actual

The faulting process should get an error, and the next process should
run. Instead the fault is reported as success, and the next process's
first command never completes:

```text
add: zeCommandListHostSynchronize 0x0 after 0.000 s, zeDeviceGetStatus 0x0
== repro add: exit 0
fault: zeCommandListHostSynchronize 0x0 after 0.000 s, zeDeviceGetStatus 0x0
== repro fault: exit 0
copies in: zeCommandListHostSynchronize 0x1 after 10.000 s, zeDeviceGetStatus 0x0
HANG: copies in not done after 10 s
== repro add: exit 2
add: zeCommandListHostSynchronize 0x0 after 0.000 s, zeDeviceGetStatus 0x0
== repro add: exit 0
== kernel log since ...
Faulted Address: 0x0000100000000000  FaultType: 0  AccessType: 1
Engine memory CAT error [18]: class=ccs, logical_mask: 0x1, guc_id=3
Timedout job: seqno=4294967169, lrc_seqno=4294967169, guc_id=3, flags=0x20 in repro
exec queue reset detected
```

The hung process waits on two host-to-device copies, before any kernel;
`zeDeviceGetStatus` says success and the kernel log shows nothing for it.
A 30 s pause between the faulting process and the next one does not help.
The process after it runs normally.

Measured on 2026-11-01 with Linux 7.2.8 (`xe`) and GuC firmware 70.72.1:

| compute-runtime (IGC)                    | `fault` reported | Next `add`     |
|------------------------------------------|------------------|----------------|
| 26.35.39758 (2.41.5)                     | success          | hangs (exit 2) |
| 26.40.40196.15 (v2.42.14)                | success          | hangs (exit 2) |
| master `ee495cce` (master `bd0ad0a5`)    | success          | hangs (exit 2) |
| each compute-runtime with the other IGCs | success          | hangs (exit 2) |

IGC has no part in it. What is left is compute-runtime of every version
tried, the `xe` kernel driver or the GuC firmware. A different fault, a
store just past the end of a small device allocation, was reported
(`ZE_RESULT_ERROR_DEVICE_LOST`) and the next process did not hang.
