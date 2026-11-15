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

; The kernel takes { i64, i64, { ptr, i64 } } by value and stores each field.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-unknown"

define spir_kernel void @k(ptr addrspace(1) %out, { i64, i64, { ptr addrspace(1), i64 } } %loc) {
  %line = extractvalue { i64, i64, { ptr addrspace(1), i64 } } %loc, 0
  %col = extractvalue { i64, i64, { ptr addrspace(1), i64 } } %loc, 1
  %file = extractvalue { i64, i64, { ptr addrspace(1), i64 } } %loc, 2, 0
  %len = extractvalue { i64, i64, { ptr addrspace(1), i64 } } %loc, 2, 1
  %fileint = ptrtoint ptr addrspace(1) %file to i64
  store i64 %line, ptr addrspace(1) %out, align 8
  %o1 = getelementptr i64, ptr addrspace(1) %out, i64 1
  store i64 %col, ptr addrspace(1) %o1, align 8
  %o2 = getelementptr i64, ptr addrspace(1) %out, i64 2
  store i64 %fileint, ptr addrspace(1) %o2, align 8
  %o3 = getelementptr i64, ptr addrspace(1) %out, i64 3
  store i64 %len, ptr addrspace(1) %o3, align 8
  ret void
}
