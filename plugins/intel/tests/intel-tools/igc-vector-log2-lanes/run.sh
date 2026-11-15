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

# Runs log2_v4half (float16 log2 and exp2 on a 4-wide vector) through Level
# Zero, built with IGC's default optimization and with -cl-opt-disable, and
# with IGC's vectorizer off. Needs spirv-as, spirv-val, clang++ and a
# bmg-g31 GPU on Level Zero.
set -u
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
spirv-as "$here/log2_v4half.spvasm" -o "$work/log2_v4half.spv" || exit 1
spirv-val "$work/log2_v4half.spv" || exit 1
clang++ -O1 -std=c++17 "$here/run-log2.cpp" -o "$work/run-log2" -lze_loader ||
  exit 1
echo "== log2_v4half"
NEO_CACHE_PERSISTENT=0 "$work/run-log2" "$work/log2_v4half.spv" 4
echo "== log2_v4half, IGC_EnableVectorizer=0"
NEO_CACHE_PERSISTENT=0 IGC_EnableVectorizer=0 \
  "$work/run-log2" "$work/log2_v4half.spv" 4
