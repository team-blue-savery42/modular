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
; Build: clang -target spirv64 -c one_struct.ll, then ocloc for bmg-g31.
target triple = "spirv64-unknown-unknown"
%pair = type { i64, i64 }
; One struct argument, entry block starts with its extractvalue.
define spir_kernel void @one_struct(ptr addrspace(1) %out, %pair %a) {
entry:
  %x = extractvalue %pair %a, 0
  %y = extractvalue %pair %a, 1
  %s = add i64 %x, %y
  store i64 %s, ptr addrspace(1) %out, align 8
  ret void
}
