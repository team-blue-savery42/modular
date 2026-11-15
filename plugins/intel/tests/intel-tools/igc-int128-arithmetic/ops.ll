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

; One work-item per input pair: a = in[2i], b = in[2i+1] (128-bit each); it
; stores ten 128-bit results to out[10i .. 10i+9]:
;   0  a & (2^64 - 1)
;   1  a >> 64
;   2  a << 64
;   3  zext(trunc a to i64)
;   4  (a & (2^64 - 1)) >> 64
;   5  a * b
;   6  zext(trunc a) * zext(trunc b)
;   7  zext(trunc(a * b to i64)), the low half through i64
;   8  a ^ (2^64 - 1)
;   9  a ^ 0x0123456789abcdeffedcba9876543210
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
  %r0 = and i128 %a, 18446744073709551615
  %r1 = lshr i128 %a, 64
  %r2 = shl i128 %a, 64
  %a64 = trunc i128 %a to i64
  %r3 = zext i64 %a64 to i128
  %r4 = lshr i128 %r0, 64
  %r5 = mul i128 %a, %b
  %b64 = trunc i128 %b to i64
  %bz = zext i64 %b64 to i128
  %r6 = mul i128 %r3, %bz
  %p64 = trunc i128 %r5 to i64
  %r7 = zext i64 %p64 to i128
  %r8 = xor i128 %a, 18446744073709551615
  %r9 = xor i128 %a, 1512366075204170947332355369683137040
  %o = mul i64 %i, 10
  %q0 = getelementptr i128, ptr addrspace(1) %out, i64 %o
  store i128 %r0, ptr addrspace(1) %q0, align 16
  %q1 = getelementptr i128, ptr addrspace(1) %q0, i64 1
  store i128 %r1, ptr addrspace(1) %q1, align 16
  %q2 = getelementptr i128, ptr addrspace(1) %q0, i64 2
  store i128 %r2, ptr addrspace(1) %q2, align 16
  %q3 = getelementptr i128, ptr addrspace(1) %q0, i64 3
  store i128 %r3, ptr addrspace(1) %q3, align 16
  %q4 = getelementptr i128, ptr addrspace(1) %q0, i64 4
  store i128 %r4, ptr addrspace(1) %q4, align 16
  %q5 = getelementptr i128, ptr addrspace(1) %q0, i64 5
  store i128 %r5, ptr addrspace(1) %q5, align 16
  %q6 = getelementptr i128, ptr addrspace(1) %q0, i64 6
  store i128 %r6, ptr addrspace(1) %q6, align 16
  %q7 = getelementptr i128, ptr addrspace(1) %q0, i64 7
  store i128 %r7, ptr addrspace(1) %q7, align 16
  %q8 = getelementptr i128, ptr addrspace(1) %q0, i64 8
  store i128 %r8, ptr addrspace(1) %q8, align 16
  %q9 = getelementptr i128, ptr addrspace(1) %q0, i64 9
  store i128 %r9, ptr addrspace(1) %q9, align 16
  ret void
}
