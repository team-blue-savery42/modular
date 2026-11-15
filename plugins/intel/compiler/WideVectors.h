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
// Vectors wider than SPIR-V allows. OpenCL SPIR-V vectors have at most 16
// elements (Vector16), and LLVM's SPIR-V backend can't legalize wider ones,
// which Mojo code builds from wide SIMD values (`SIMD[DType.float32, 256]`).
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_INTEL_COMPILER_WIDEVECTORS_H
#define MOJO_INTEL_COMPILER_WIDEVECTORS_H

namespace llvm {
class Function;
class Module;
} // namespace llvm

namespace M::KGEN::Intel {

/// The widest vector OpenCL SPIR-V allows.
inline constexpr unsigned kMaxVectorElements = 16;

/// Scalarizes every function that uses a vector wider than 16 elements, with
/// LLVM's Scalarizer (into scalars, so that conversions between element
/// sizes split alike, and with loads and stores split too). Other functions
/// are left alone. Instructions the Scalarizer doesn't split stay wide.
///
/// Before it runs, the function's float vector constants move into private
/// constant-memory globals that it loads: splitting a float constant would
/// make this copy of LLVM create float constants, whose APFloat semantics the
/// host's codegen can't read (see PromoteBFloat16.cpp), and reading one's
/// value here has the same problem. A split load makes nothing new. Returns
/// whether anything changed.
bool splitWideVectors(llvm::Module &module);

} // namespace M::KGEN::Intel

#endif // MOJO_INTEL_COMPILER_WIDEVECTORS_H
