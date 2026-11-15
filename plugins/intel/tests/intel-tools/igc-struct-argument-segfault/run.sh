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

# Compiles two_structs.ll, guarded.ll and one_struct.ll to SPIR-V and builds
# each for bmg-g31 with the ocloc on PATH, printing ocloc's exit status
# (139: IGC segfaulted). Needs clang (spirv64), spirv-val and ocloc; no GPU.
set -u
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
for name in two_structs guarded one_struct; do
  clang -target spirv64 -Wno-override-module -c "$here/$name.ll" \
    -o "$work/$name.spv" || exit 1
  spirv-val "$work/$name.spv" || exit 1
  ocloc compile -spirv_input -file "$work/$name.spv" -device bmg-g31 \
    -output "$work/$name" >"$work/$name.log" 2>&1
  echo "$name: ocloc exit $?"
done
