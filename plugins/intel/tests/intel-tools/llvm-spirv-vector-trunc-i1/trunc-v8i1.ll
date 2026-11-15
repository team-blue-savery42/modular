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

; An 8-lane truncation to i1 that feeds a select. An LLVM built with
; assertions at 9b2201bd0e35 stops with
;   LLVM ERROR: VReg's low-level type and register class have different
;   sizes: %21:viid(<8 x s32>) = OpBitwiseAndV %3:type(s64), %11:viid(s64),
;   %24:viid(s64) (in function: k)
; The same IR at 2 and 4 lanes compiles. llc 23.1.3 (no assertions) builds
; SPIR-V that spirv-val accepts.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-unknown"

define spir_kernel void @k(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %a = load <8 x i32>, ptr addrspace(1) %p, align 4
  %m = trunc <8 x i32> %a to <8 x i1>
  %f = load <8 x float>, ptr addrspace(1) %q, align 4
  %s = select <8 x i1> %m, <8 x float> %f, <8 x float> splat (float -1.0e+03)
  store <8 x float> %s, ptr addrspace(1) %q, align 4
  ret void
}
