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

; k(in, out) loads a <4 x float> per work-item, tests each lane's
; |x| >= 2 and "finite" (sign bit cleared, below 0x7f800000), takes 1/sqrt
; on the lanes that pass (of 2 on the others) and stores 0 for lanes that are
; infinite. Written by hand after the IR IGC gets from Mojo's float32 j0 at
; width 4 (MAX's test_j0_y0_gpu). In IGC's final IR (*_codegen.ll) the uses
; of |x| are scalar llvm.fabs.f32 calls, but the sign-masked bits come from a
; <4 x float> llvm.fabs.v4f32 that is left over, and the vISA emitter writes
; only the first lane of it: the kernel reads lanes 1-3 of those bits from
; registers nothing wrote.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-unknown"

declare spir_func i64 @_Z13get_global_idj(i32)
declare <4 x float> @llvm.fabs.v4f32(<4 x float>)
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)

define spir_kernel void @k(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %i = call spir_func i64 @_Z13get_global_idj(i32 0)
  %p = getelementptr <4 x float>, ptr addrspace(1) %in, i64 %i
  %x = load <4 x float>, ptr addrspace(1) %p, align 16
  %ax = call <4 x float> @llvm.fabs.v4f32(<4 x float> %x)
  %ge2 = fcmp oge <4 x float> %ax, splat (float 2.0)
  %bits = bitcast <4 x float> %x to <4 x i32>
  %mag = and <4 x i32> %bits, splat (i32 2147483647)
  %fin = icmp ult <4 x i32> %mag, splat (i32 2139095040)
  %big = and <4 x i1> %ge2, %fin
  %e0 = extractelement <4 x i1> %big, i32 0
  %e1 = extractelement <4 x i1> %big, i32 1
  %e2 = extractelement <4 x i1> %big, i32 2
  %e3 = extractelement <4 x i1> %big, i32 3
  %o01 = or i1 %e0, %e1
  %o012 = or i1 %o01, %e2
  %any = or i1 %o012, %e3
  br i1 %any, label %large, label %done

large:
  %t = select <4 x i1> %big, <4 x float> %ax, <4 x float> splat (float 2.0)
  %s = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %t)
  %r = fdiv <4 x float> splat (float 1.0), %s
  br label %done

done:
  %y = phi <4 x float> [ %r, %large ], [ %ax, %entry ]
  %bits2 = bitcast <4 x float> %x to <4 x i32>
  %mag2 = and <4 x i32> %bits2, splat (i32 2147483647)
  %inf = icmp eq <4 x i32> %mag2, splat (i32 2139095040)
  %res = select <4 x i1> %inf, <4 x float> zeroinitializer, <4 x float> %y
  %q = getelementptr <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %res, ptr addrspace(1) %q, align 16
  ret void
}
