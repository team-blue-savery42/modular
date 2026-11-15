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

; A kernel that traps when its input is 0. With
;   clang -target spirv64 -mllvm --spirv-ext=+SPV_KHR_abort
; LLVM's SPIR-V backend lowers llvm.trap to OpAbortKHR (SPV_KHR_abort, the
; AbortKHR capability); without the extension it drops the trap.
target triple = "spirv64-unknown-unknown"

declare void @llvm.trap()

define spir_kernel void @k(ptr addrspace(1) %p) {
entry:
  %v = load i32, ptr addrspace(1) %p, align 4
  %c = icmp eq i32 %v, 0
  br i1 %c, label %trap, label %done

trap:
  call void @llvm.trap()
  unreachable

done:
  store i32 1, ptr addrspace(1) %p, align 4
  ret void
}
