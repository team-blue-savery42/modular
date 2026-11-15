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
// Compiler plugin entry points. The Mojo compiler dlopens this library from
// `MODULAR_COMPILER_PLUGINS` (or `modular.cfg`) and resolves these symbols;
// each returns a stateless object that lives for the process. See
// Mojo/lib/Target/max_license/Plugin/PluginTraits.cpp and its lowering and
// backend counterparts for the signatures.
//
//===----------------------------------------------------------------------===//

#include "IntelBackend.h"
#include "IntelLowering.h"
#include "IntelTraits.h"

#define MOJO_INTEL_EXPORT extern "C" __attribute__((visibility("default")))

MOJO_INTEL_EXPORT const M::KGEN::TargetTraits *M_KGEN_getTargetTraits() {
  M::KGEN::Intel::trace("entry: M_KGEN_getTargetTraits");
  return &M::KGEN::Intel::IntelTraits::get();
}

MOJO_INTEL_EXPORT const M::KGEN::TargetLowering *M_KGEN_getTargetLowering() {
  M::KGEN::Intel::trace("entry: M_KGEN_getTargetLowering");
  return &M::KGEN::Intel::getLowering();
}

MOJO_INTEL_EXPORT const M::KGEN::TargetBackend *M_KGEN_getTargetBackend() {
  M::KGEN::Intel::trace("entry: M_KGEN_getTargetBackend");
  return &M::KGEN::Intel::getBackend();
}
