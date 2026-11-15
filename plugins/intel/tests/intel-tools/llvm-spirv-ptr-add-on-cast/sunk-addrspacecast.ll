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

; A load in a second block from
;   inttoptr (add (ptrtoint (ptr addrspace(3) @g), 24) to ptr addrspace(4))
; Reduced from the kernel of MAX's test_metal_memory.mojo. Valid IR (the
; verifier accepts it).
;   llc -mtriple=spirv64-unknown-unknown -filetype=obj sunk-addrspacecast.ll
; An LLVM built with assertions stops in SPIRVInstructionSelector.cpp
; (spvSelect, G_PTR_ADD): Assertion `((*II).getOpcode() ==
; TargetOpcode::G_GLOBAL_VALUE || ... ) && getImm(I.getOperand(2), MRI)'
; failed. Without assertions (llc 23.1.3) it builds valid SPIR-V.
;
; SPIRVRegularizer turns the constant expression into instructions in %b;
; CodeGenPrepare sinks the address [24 + @g] back as the constant
;   getelementptr (i8, ptr addrspace(4) addrspacecast (ptr addrspace(3) @g
;                  to ptr addrspace(4)), i64 24)
; which the IRTranslator turns into a G_PTR_ADD on a G_ADDRSPACE_CAST. The
; selector's G_PTR_ADD case assumes its base is a global variable.
; The branch matters: without it the address isn't sunk and nothing asserts.

target triple = "spirv64-unknown-unknown"

@g = external addrspace(3) global [8 x float]

define spir_kernel void @k(ptr addrspace(1) %out) {
  br label %b
b:
  %v = load float, ptr addrspace(4) inttoptr (i64 add (i64 ptrtoint (ptr addrspace(3) @g to i64), i64 24) to ptr addrspace(4)), align 4
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}
