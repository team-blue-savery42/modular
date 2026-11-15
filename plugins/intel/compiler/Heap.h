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
// malloc, free and realloc for Intel GPUs. On CUDA the driver links
// a device heap into every module that calls malloc; IGC links nothing of the
// kind, so a kernel holding a List failed to load. This pass defines them in
// the module, over a heap the Level Zero driver plugin owns.
//
// The contract with the driver plugin: a module that allocates or frees
// exports the global `__mojo_intel_heap`, a CrossWorkgroup pointer that the
// plugin sets, after loading the module, to its context's heap: a
// `kHeapHeaderBytes` header, zeroed but for its second i64, then the memory
// blocks come from. The header holds the bytes handed out so far (i64 at 0),
// the bytes after the header (i64 at 8), and from `kHeapFreeListsOffset` one
// free-list head (i64) per size class.
//
// Blocks come in size classes: a block is a power of two of
// at least 32 bytes, its first 16 bytes holding the size asked for (for
// realloc and free), so its memory is 16-byte aligned, as CUDA's malloc
// returns. free pushes a block on its class's list and malloc pops one from
// it before taking new bytes, so a long-lived context whose kernels allocate
// and free on every launch keeps reusing the same memory. A list head is a
// tagged pointer: the low 32 bits are the first block's offset after the
// header in 32-byte units plus one (0: empty), the high 32 bits a count of
// the head's changes, so a block popped and pushed again between another
// work-item's read and its compare-exchange can't be mistaken for the head it
// read (ABA). A free block's first i64 is the next block's link (low 32
// bits). Blocks past `kHeapMaxLinkedOffset` or of classes above
// `kHeapMaxClassLog2` are never linked: freeing one does nothing.
//
// When no freed block of the class is there, malloc bumps: it reserves the
// block with a device-scope atomic compare-exchange on the first i64,
// retried while other work-items move it, and returns null when the block
// doesn't fit in what is left, without using any of it, so smaller requests
// still get the rest, as with CUDA's malloc.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_COMPILER_HEAP_H
#define MOJO_INTEL_COMPILER_HEAP_H

#include <cstdint>

namespace llvm {
class Module;
} // namespace llvm

namespace M::KGEN::Intel {

/// The exported global the driver plugin points at its heap.
inline constexpr const char *kHeapGlobal = "__mojo_intel_heap";
/// Bytes of the heap's header, before the first block.
inline constexpr uint64_t kHeapHeaderBytes = 512;
/// Bytes before each block's memory, holding the size asked for; also the
/// alignment of every block's memory.
inline constexpr uint64_t kHeapBlockHeaderBytes = 16;
/// The smallest block, log2: 16 bytes of header and 16 of memory.
inline constexpr unsigned kHeapMinClassLog2 = 5;
/// The largest block that is linked into a free list when freed, log2.
inline constexpr unsigned kHeapMaxClassLog2 = 60;
/// Where in the header the free-list heads start, one i64 per class from
/// `kHeapMinClassLog2` to `kHeapMaxClassLog2`.
inline constexpr uint64_t kHeapFreeListsOffset = 64;
static_assert(kHeapFreeListsOffset +
                      8 * (kHeapMaxClassLog2 - kHeapMinClassLog2 + 1) <=
                  kHeapHeaderBytes,
              "the free-list heads must fit in the header");
/// Blocks starting this many bytes or more after the header can't be named
/// by a 32-bit link in 32-byte units, so they are never linked.
inline constexpr uint64_t kHeapMaxLinkedOffset = (uint64_t(1) << 32) * 32 - 32;

/// Defines `malloc`, `free` and `realloc` where the module declares them, as
/// internal functions named `__mojo_intel_malloc` and so on (a module
/// declaring `realloc` gets `malloc` and `free` too), with the exported
/// `__mojo_intel_heap` that they read. Run after LLVM's optimizer, so
/// that each one is still called, and after the address spaces are in SPIR-V
/// numbering. Returns whether the module changed.
bool defineHeapFunctions(llvm::Module &module);

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_HEAP_H
