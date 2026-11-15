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

# Compiles nested.ll to SPIR-V, shows the vISA IGC emits for the kernel up
# to the call, then runs it with IGC's default optimization and with
# -cl-opt-disable; then the same for a nested struct returned by a function
# (nested_ret.ll) and passed to a kernel (kernarg.ll); then ocloc on
# nested_return.spv with -cl-opt-disable. Needs clang (spirv64), spirv-val,
# clang++, ocloc and a bmg-g31 GPU on Level Zero.
set -u
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
options="-ze-intel-enable-auto-large-GRF-mode -cl-fp32-correctly-rounded-divide-sqrt"
clang -target spirv64 -Wno-override-module -c "$here/nested.ll" \
  -o "$work/nested.spv" || exit 1
spirv-val "$work/nested.spv" && echo "spirv-val: valid"
(cd "$work" && IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$work/dump" \
  ocloc compile -spirv_input -file "$work/nested.spv" -device bmg-g31 \
  -q -output nested >/dev/null 2>&1)
echo "== vISA of the kernel up to the call (ocloc, default options)"
sed -n '/^_main_0:/,/call /p' "$work"/dump/*_entry_0001.visaasm |
  sed 's| *///.*||'
clang++ -O1 -std=c++17 "$here/run-nested.cpp" -o "$work/run-nested" \
  -lze_loader || exit 1
echo "== run nested"
NEO_CACHE_PERSISTENT=0 "$work/run-nested" "$work/nested.spv"
echo "run-nested nested: exit $?"
for name in nested_ret kernarg; do
  clang -target spirv64 -Wno-override-module -c "$here/$name.ll" \
    -o "$work/$name.spv" || exit 1
  spirv-val "$work/$name.spv" || exit 1
  echo "== run $name"
  # Line-buffered: an IGC crash in the second build would lose the first's
  # lines otherwise.
  mode=
  [ "$name" = kernarg ] && mode=kernarg
  NEO_CACHE_PERSISTENT=0 stdbuf -oL "$work/run-nested" "$work/$name.spv" \
    ${mode:+"$mode"}
  echo "run-nested $name: exit $?"
done
echo "== nested_return.spv through ocloc"
spirv-val "$here/nested_return.spv" && echo "spirv-val: valid"
for extra in "" " -cl-opt-disable"; do
  (cd "$work" && ocloc compile -spirv_input -file "$here/nested_return.spv" \
    -device bmg-g31 -q -options "$options$extra" -output nr >/dev/null 2>&1)
  echo "ocloc -options \"$options$extra\": exit $?"
done
