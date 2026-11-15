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

; As ops.ll, but the mask 2^64 - 1 is a third input (in[3i+2]), not a
; constant: a = in[3i], b = in[3i+1], m = in[3i+2]; work-item i stores
;   0  a & m
;   1  (a & m) >> 64
;   2  (a & m) * (b & m)
;   3  a * b
;   4  zext(x), x the low 64 bits of a loaded as an i64
;   5  a + zext(x)
;   6  zext(x) << 64
;   7  a + b
; to out[8i .. 8i+7].
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-unknown"

declare spir_func i64 @_Z13get_global_idj(i32)

define spir_kernel void @k(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %i = call spir_func i64 @_Z13get_global_idj(i32 0)
  %ia = mul i64 %i, 3
  %pa = getelementptr i128, ptr addrspace(1) %in, i64 %ia
  %pb = getelementptr i128, ptr addrspace(1) %pa, i64 1
  %pm = getelementptr i128, ptr addrspace(1) %pa, i64 2
  %a = load i128, ptr addrspace(1) %pa, align 16
  %b = load i128, ptr addrspace(1) %pb, align 16
  %m = load i128, ptr addrspace(1) %pm, align 16
  %am = and i128 %a, %m
  %bm = and i128 %b, %m
  %r1 = lshr i128 %am, 64
  %r2 = mul i128 %am, %bm
  %r3 = mul i128 %a, %b
  %x = load i64, ptr addrspace(1) %pa, align 16
  %r4 = zext i64 %x to i128
  %r5 = add i128 %a, %r4
  %r6 = shl i128 %r4, 64
  %r7 = add i128 %a, %b
  %o = mul i64 %i, 8
  %q0 = getelementptr i128, ptr addrspace(1) %out, i64 %o
  store i128 %am, ptr addrspace(1) %q0, align 16
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
  ret void
}
