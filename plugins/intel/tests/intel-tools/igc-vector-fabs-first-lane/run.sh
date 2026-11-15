#!/bin/sh
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Sean Avery. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# Builds fabs4.ll with IGC's dumps on, shows the vector fabs left in IGC's
# final IR and the vISA that makes |x| of the loaded vector, then runs the
# kernel alone, right after another kernel and after an idle GPU. Needs clang
# (spirv64), spirv-val, clang++, ocloc and a bmg-g31 GPU on Level Zero.
set -u
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
options="-ze-intel-enable-auto-large-GRF-mode -cl-fp32-correctly-rounded-divide-sqrt"
clang -target spirv64 -Wno-override-module -c "$here/fabs4.ll" \
  -o "$work/fabs4.spv" || exit 1
spirv-val "$work/fabs4.spv" && echo "spirv-val: valid"
clang -target spirv64 -cl-std=CL2.0 -O2 -c "$here/fill.cl" \
  -o "$work/fill.spv" || exit 1
for extra in "" " -cl-opt-disable"; do
  rm -rf "$work/dump" && mkdir "$work/dump"
  (cd "$work" && IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$work/dump" \
    ocloc compile -spirv_input -file fabs4.spv -device bmg-g31 -q \
    -options "$options$extra" -output fabs4 >/dev/null 2>&1)
  echo "ocloc -options \"$options$extra\": exit $?"
  echo "vector fabs left in IGC's final IR:"
  grep -h "call <4 x float> @llvm.fabs" "$work"/dump/*_codegen.ll
  echo "vISA moves of |x| (each lane is a row: (0,0) (1,0)... at SIMD16,"
  echo "(0,0) (2,0)... at SIMD32), then the reads of the masked bits:"
  grep -h "mov .*(abs)" "$work"/dump/*.visaasm
  grep -h "0x7f800000" "$work"/dump/*.visaasm
done
clang++ -O1 -std=c++17 "$here/run-fabs4.cpp" -o "$work/run-fabs4" \
  -lze_loader || exit 1
for mode in alone after-fill after-idle; do
  NEO_CACHE_PERSISTENT=0 "$work/run-fabs4" "$work/fabs4.spv" "$work/fill.spv" \
    $mode
done
NEO_CACHE_PERSISTENT=0 "$work/run-fabs4" "$work/fabs4.spv" "$work/fill.spv" \
  after-fill -cl-opt-disable
