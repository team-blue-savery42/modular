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

# Runs llc for spirv64 on each .ll here, then spirv-val and ocloc (bmg-g31)
# on what it builds. Usage: sh run.sh [LLC] (default: llc on PATH). An llc
# built with assertions shows the assertion; one without shows what the
# selector builds instead. Needs llc, spirv-val and ocloc; no GPU.
set -u
here=$(cd "$(dirname "$0")" && pwd)
llc=${1:-llc}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
for name in sunk-addrspacecast equal-pointee-types global-initializer; do
  echo "== $name"
  if "$llc" -verify-machineinstrs -mtriple=spirv64-unknown-unknown \
    -filetype=obj "$here/$name.ll" -o "$work/$name.spv" \
    >"$work/$name.log" 2>&1; then
    echo "llc: built"
    if spirv-val "$work/$name.spv"; then
      echo "spirv-val: valid"
    fi
    if (cd "$work" && ocloc compile -spirv_input -file "$name.spv" \
      -device bmg-g31 -output "$name" >ocloc.log 2>&1); then
      echo "ocloc: built"
    else
      echo "ocloc: exit $?"
    fi
  else
    grep -m1 -o 'Assertion.*failed' "$work/$name.log" | cut -c1-120
    echo "llc: failed"
  fi
done
