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

# Without the GPU: bounds_loop.spv (the module before the compiler plugin
# split aggregate phis), bounds_loop_assert.spv (the same with bounds checks)
# and bounds_loop_split.spv (after the split) pass spirv-val and go through
# ocloc with IGC's defaults and with -cl-opt-disable. With the GPU (NOGPU=1
# skips it): run-bounds runs bounds_loop.spv and bounds_loop_split.spv
# through Level Zero both ways. Needs spirv-val, ocloc, clang++ and, for the
# run, a bmg-g31 GPU on Level Zero.
set -u
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
for spv in "$here/bounds_loop.spv" "$here/bounds_loop_assert.spv" \
  "$here/bounds_loop_split.spv"; do
  echo "== $(basename "$spv")"
  spirv-val "$spv" && echo "spirv-val: valid"
  for extra in "" "-cl-opt-disable"; do
    (cd "$work" && ocloc compile -spirv_input -file "$spv" -device bmg-g31 \
      -q ${extra:+-options "$extra"} -output bounds >/dev/null 2>&1)
    echo "ocloc ${extra:+-options $extra }: exit $?"
  done
done
[ "${NOGPU:-0}" = 1 ] && exit 0
clang++ -O1 -std=c++17 "$here/run-bounds.cpp" -o "$work/run-bounds" \
  -lze_loader || exit 1
for spv in "$here/bounds_loop.spv" "$here/bounds_loop_split.spv"; do
  echo "== run $(basename "$spv")"
  NEO_CACHE_PERSISTENT=0 "$work/run-bounds" "$spv"
  echo "run-bounds: exit $?"
done
