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

# Runs llc for spirv64 on trunc-v8i1.ll and checks what it builds with
# spirv-val. Usage: sh run.sh [LLC] (default: llc on PATH); the failure
# needs an llc built with assertions. Needs llc and spirv-val; no GPU.
set -u
here=$(cd "$(dirname "$0")" && pwd)
llc=${1:-llc}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
if "$llc" -O2 -mtriple=spirv64-unknown-unknown -filetype=obj \
  "$here/trunc-v8i1.ll" -o "$work/t.spv" >"$work/llc.log" 2>&1; then
  echo "llc: built"
  spirv-val "$work/t.spv" && echo "spirv-val: valid"
else
  grep -m1 "LLVM ERROR" "$work/llc.log" | cut -c1-160
  echo "llc: failed"
fi
