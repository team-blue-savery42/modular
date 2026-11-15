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

; sunk-addrspacecast.ll with @g an i8, so that the cast's pointee type
; equals the load's.
;   llc -mtriple=spirv64-unknown-unknown -filetype=obj equal-pointee-types.ll
; With assertions it trips the same assertion as sunk-addrspacecast.ll.
; Without them (llc 23.1.3) the selector takes its OpSpecConstantOp branch
; on the cast's result, which is an instruction in the function:
;   %12 = OpSpecConstantOp %_ptr_Generic_uchar InBoundsPtrAccessChain %15 %ulong_24
;   %15 = OpPtrCastToGeneric %_ptr_Generic_uchar %g
; and spirv-val rejects it: "ID '15[%15]' has not been defined". Any other
; pointee type tried ([32 x i8] with an i8 load, float, i32, [8 x float],
; [8 x i32] with a <4 x i32> load) gets an OpBitcast and valid SPIR-V.

target triple = "spirv64-unknown-unknown"

@g = external addrspace(3) global i8

define spir_kernel void @k(ptr addrspace(1) %out) {
  br label %b
b:
  %v = load i8, ptr addrspace(4) inttoptr (i64 add (i64 ptrtoint (ptr addrspace(3) @g to i64), i64 24) to ptr addrspace(4)), align 4
  store i8 %v, ptr addrspace(1) %out, align 4
  ret void
}
