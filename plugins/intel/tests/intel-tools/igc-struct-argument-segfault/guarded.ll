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

; A kernel taking struct arguments by value (SPIR-V OpTypeStruct parameters).
; Build: clang -target spirv64 -c guarded.ll, then ocloc for bmg-g31.
target triple = "spirv64-unknown-unknown"
%pair = type { i64, i64 }
; Same, but the first instruction does not use a struct argument.
define spir_kernel void @two_structs_guarded(ptr addrspace(1) %out, %pair %a, %pair %b) {
entry:
  %o = getelementptr inbounds i64, ptr addrspace(1) %out, i64 0
  %x = extractvalue %pair %a, 0
  %y = extractvalue %pair %a, 1
  %z = extractvalue %pair %b, 0
  %w = extractvalue %pair %b, 1
  %s1 = add i64 %x, %y
  %s2 = add i64 %z, %w
  %s = mul i64 %s1, %s2
  store i64 %s, ptr addrspace(1) %o, align 8
  ret void
}
