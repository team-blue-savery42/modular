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
// Address spaces for Intel SPIR-V targets. Mojo numbers address spaces the way
// NVPTX and AMDGPU do (std/memory/address_space.mojo), and those numbers reach
// LLVM IR unchanged. LLVM's SPIR-V backend reads the same numbers as OpenCL
// storage classes (SPIRVUtils.h, `storageClassToAddressSpace`), where 0 is
// private (Function) and generic is 4. Every Mojo address space therefore has
// to be renumbered before SPIR-V codegen; `toSPIRVAddressSpace` is that table.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_COMPILER_INTELADDRESSSPACE_H
#define MOJO_INTEL_COMPILER_INTELADDRESSSPACE_H

#include <optional>

namespace M::KGEN::Intel {

/// Mojo's `AddressSpace` values (std/memory/address_space.mojo).
enum class MojoAddressSpace : unsigned {
  Generic = 0,
  Global = 1,
  Shared = 3,
  Constant = 4,
  Local = 5,
  SharedCluster = 7,  // NVIDIA-specific
  BufferResource = 8, // AMD-specific
};

/// LLVM address spaces as LLVM's SPIR-V backend maps them to storage classes.
enum class SPIRVAddressSpace : unsigned {
  Function = 0,        // private, per work-item
  CrossWorkgroup = 1,  // global
  UniformConstant = 2, // constant
  Workgroup = 3,       // shared local memory
  Generic = 4,
};

/// The SPIR-V address space for Mojo address space `mojo`, or nullopt when
/// Intel GPUs have no equivalent (NVIDIA clusters, AMD buffer resources, and
/// values Mojo doesn't define).
constexpr std::optional<SPIRVAddressSpace> toSPIRVAddressSpace(unsigned mojo) {
  switch (static_cast<MojoAddressSpace>(mojo)) {
  case MojoAddressSpace::Generic:
    return SPIRVAddressSpace::Generic;
  case MojoAddressSpace::Global:
    return SPIRVAddressSpace::CrossWorkgroup;
  case MojoAddressSpace::Shared:
    return SPIRVAddressSpace::Workgroup;
  case MojoAddressSpace::Constant:
    return SPIRVAddressSpace::UniformConstant;
  case MojoAddressSpace::Local:
    return SPIRVAddressSpace::Function;
  case MojoAddressSpace::SharedCluster:
  case MojoAddressSpace::BufferResource:
    return std::nullopt;
  }
  return std::nullopt;
}

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_INTELADDRESSSPACE_H
