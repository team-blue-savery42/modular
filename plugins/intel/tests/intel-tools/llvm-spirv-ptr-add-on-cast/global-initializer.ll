; ===----------------------------------------------------------------------=== ;
; Copyright (c) 2026, Sean Avery. All rights reserved.
;
; Licensed under the Apache License v2.0 with LLVM Exceptions:
; https://llvm.org/LICENSE.txt
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS,
; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
; See the License for the specific language governing permissions and
; limitations under the License.
; ===----------------------------------------------------------------------=== ;

; The same assertion from a global variable's initializer: a constant GEP on
; an addrspacecast of a global. Hand-written; not met in a Mojo program.
;   llc -mtriple=spirv64-unknown-unknown -filetype=obj global-initializer.ll

target triple = "spirv64-unknown-unknown"

@g = addrspace(1) global [8 x float] zeroinitializer
@p = addrspace(1) global ptr addrspace(4) getelementptr (i8, ptr addrspace(4) addrspacecast (ptr addrspace(1) @g to ptr addrspace(4)), i64 8)

define spir_kernel void @k(ptr addrspace(1) %out) {
  %q = load ptr addrspace(4), ptr addrspace(1) @p
  %v = load float, ptr addrspace(4) %q, align 4
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}
