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

; Work-item i multiplies two 128-bit integers loaded from in[2i] and in[2i+1]
; (each 16 bytes) in two ways and stores the low 128 bits of each product:
;   out[3i]   = a * b                        (both operands loaded whole)
;   out[3i+1] = zext(trunc a) * zext(trunc b) (64 x 64 -> 128)
;   out[3i+2] = (a & 0xffffffffffffffff) * zext(trunc b)
; The last form is what LLVM's optimizer makes of a 64 x 64 high multiply
; written with 32-bit halves.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-unknown"

declare spir_func i64 @_Z13get_global_idj(i32)

define spir_kernel void @k(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %i = call spir_func i64 @_Z13get_global_idj(i32 0)
  %ia = mul i64 %i, 2
  %ib = add i64 %ia, 1
  %pa = getelementptr i128, ptr addrspace(1) %in, i64 %ia
  %pb = getelementptr i128, ptr addrspace(1) %in, i64 %ib
  %a = load i128, ptr addrspace(1) %pa, align 16
  %b = load i128, ptr addrspace(1) %pb, align 16
  %full = mul i128 %a, %b
  %a64 = trunc i128 %a to i64
  %b64 = trunc i128 %b to i64
  %az = zext i64 %a64 to i128
  %bz = zext i64 %b64 to i128
  %wide = mul i128 %az, %bz
  %am = and i128 %a, 18446744073709551615
  %masked = mul i128 %am, %bz
  %o0 = mul i64 %i, 3
  %o1 = add i64 %o0, 1
  %o2 = add i64 %o0, 2
  %p0 = getelementptr i128, ptr addrspace(1) %out, i64 %o0
  %p1 = getelementptr i128, ptr addrspace(1) %out, i64 %o1
  %p2 = getelementptr i128, ptr addrspace(1) %out, i64 %o2
  store i128 %full, ptr addrspace(1) %p0, align 16
  store i128 %wide, ptr addrspace(1) %p1, align 16
  store i128 %masked, ptr addrspace(1) %p2, align 16
  ret void
}
