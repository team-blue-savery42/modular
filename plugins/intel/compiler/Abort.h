//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Sean Avery. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// abort() for Intel GPUs. IGC has no trap (it rejects SPV_KHR_abort), so a
// kernel can't stop itself, and Level Zero reports nothing. Instead an
// aborting work-item raises a flag the Level Zero driver plugin owns, and the
// plugin turns a raised flag into an error at the next synchronize. The
// stdlib's Intel `abort_fn` calls `__mojo_intel_abort`; this pass defines it.
//
// The contract with the driver plugin: a module that can abort exports the
// global `__mojo_intel_abort_flag`, a CrossWorkgroup pointer to an i32 that
// the plugin sets after loading the module. Aborting stores 1 through it.
//
// Aborting returns, so the work-item goes on past calls LLVM knows never
// return; `returnAfterAbort` turns the `unreachable` after them into returns,
// and makes callers return after a callee aborted, so it unwinds to the
// kernel and ends instead.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_COMPILER_ABORT_H
#define MOJO_INTEL_COMPILER_ABORT_H

namespace llvm {
class Module;
} // namespace llvm

namespace M::KGEN::Intel {

/// The function the stdlib's abort hook calls.
inline constexpr const char *kAbortFunction = "__mojo_intel_abort";
/// The exported global the driver plugin points at its abort flag.
inline constexpr const char *kAbortFlagGlobal = "__mojo_intel_abort_flag";

/// Defines `__mojo_intel_abort` where the module declares it, with the
/// exported `__mojo_intel_abort_flag` it stores through. Run after the address
/// spaces are in SPIR-V numbering. Returns whether the module can abort.
bool defineAbort(llvm::Module &module);

/// Removes `__mojo_intel_abort` if nothing calls it any more, then the flag if
/// nothing uses it. LLVM's optimizer may remove every abort after
/// `defineAbort` ran (an assertion it proves always holds); IGC then drops
/// the unused global, and the driver plugin, finding the export but no
/// global, fails to load the module. Run just before codegen. Returns
/// whether the flag was removed.
bool dropUnusedAbortFlag(llvm::Module &module);

/// In a module that can abort, makes a work-item that aborted return to the
/// kernel and end. Each function with an `unreachable` gets one return block
/// that its returns and its `unreachable`s branch to (with poison for the
/// value on the latter), and `noreturn` is dropped from it and the calls it
/// makes: LLVM ends each path through a call to a `noreturn` function, such
/// as `abort` or the stdlib's assertion reporter, with `unreachable`, and IGC
/// emits nothing for it, so a work-item that returned from abort ran into
/// whatever code followed (the next block's own abort, then a load
/// through a stale pointer that faulted at address 0). One return block, not
/// a `ret` per `unreachable`: with those IGC didn't finish compiling MAX's
/// `_format_float`. A caller must not go on with the poison either
/// (a reference from a failed bounds check faulted), so each work-item
/// has an aborted flag, a private byte in the kernel that every internal
/// function that can abort gets a pointer to as an added last parameter;
/// aborting sets it, and after each call that can abort the caller branches
/// to its return block if it is set. Run just before codegen, after
/// `dropUnusedAbortFlag`. Returns whether the module changed.
bool returnAfterAbort(llvm::Module &module);

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_ABORT_H
