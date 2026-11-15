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

# Assembles bf16_moves.spvasm and runs each kernel with a = 1.5 (0x3fc0),
# b = 2.5 (0x4020) and the flag set and clear, built with the Mojo driver
# plugin's options and with -cl-opt-disable added. Prints p[16] and p[17]
# for each. Needs spirv-as, spirv-val, clang++ and a bmg-g31 GPU on Level
# Zero.
set -u
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
spirv-as "$here/bf16_moves.spvasm" -o "$work/bf16_moves.spv" || exit 1
spirv-val "$work/bf16_moves.spv" || exit 1
clang++ -O1 -std=c++17 "$here/run-words.cpp" -o "$work/run-words" \
  -lze_loader || exit 1
for kernel in copy convert store_const convert_const bitcasts select_i16 \
  select select_const phi_const select_convert round_once_used round_twice \
  round_via_int round_across; do
  for flag in 1 0; do
    echo "== $kernel, flag $flag: p[16], p[17]"
    NEO_CACHE_PERSISTENT=0 \
      "$work/run-words" "$work/bf16_moves.spv" "$kernel" 3fc0 4020 "$flag" |
      awk '{ printf "%-16s %s %s\n", $1, $18, $19 }' | tail -2
  done
done
