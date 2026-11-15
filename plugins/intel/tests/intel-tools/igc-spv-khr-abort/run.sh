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

# Compiles trap.ll with SPV_KHR_abort enabled, checks the module with
# spirv-val, shows the abort instruction and builds it for bmg-g31 with
# ocloc. Needs clang (spirv64), spirv-val, spirv-dis and ocloc; no GPU.
set -u
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
clang -target spirv64 -Wno-override-module \
  -mllvm --spirv-ext=+SPV_KHR_abort -c "$here/trap.ll" \
  -o "$work/trap.spv" || exit 1
spirv-val "$work/trap.spv" && echo "spirv-val: valid"
spirv-dis "$work/trap.spv" | grep -E "AbortKHR|SPV_KHR_abort"
(cd "$work" && ocloc compile -spirv_input -file trap.spv -device bmg-g31 \
  -output trap >ocloc.log 2>&1)
echo "ocloc: exit $?"
grep -h "InvalidModule\|unknown extension" "$work/ocloc.log"
