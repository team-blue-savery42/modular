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

# Compiles specials_atan2.spv for bmg-g31 with ocloc: with the Mojo driver
# plugin's options, with -cl-opt-disable added, and with IGC's vectorizer
# off. Needs spirv-val and ocloc; no GPU.
set -u
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
options="-ze-intel-enable-auto-large-GRF-mode -cl-fp32-correctly-rounded-divide-sqrt"
spirv-val "$here/specials_atan2.spv" && echo "spirv-val: valid"
for extra in "" " -cl-opt-disable"; do
  (cd "$work" && ocloc compile -spirv_input -file "$here/specials_atan2.spv" \
    -device bmg-g31 -q -options "$options$extra" -output atan2 >ocloc.log 2>&1)
  echo "ocloc -options \"$options$extra\": exit $?"
  grep -h -m1 "Internal Compiler Error\|Assertion" "$work/ocloc.log"
done
(cd "$work" && IGC_EnableVectorizer=0 ocloc compile -spirv_input \
  -file "$here/specials_atan2.spv" -device bmg-g31 -q -options "$options" \
  -output atan2 >ocloc.log 2>&1)
echo "IGC_EnableVectorizer=0 ocloc -options \"$options\": exit $?"
