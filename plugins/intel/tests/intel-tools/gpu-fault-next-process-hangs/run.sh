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

# Builds utils/l0/fault-absorb (a Level Zero program with two kernels), then
# runs four processes in a row: add, fault, add, add. Prints each exit status
# (2: a wait timed out after 10 s) and the kernel log lines of the run.
# Expected on an affected system: the add right after the fault hangs.
# The fault resets the GPU's compute engine for every process using it; run
# this only on a GPU nothing else is using. Needs clang (spirv64), c++, a
# bmg-g31 GPU on Level Zero and read access to the kernel log (journalctl -k).
set -u
here=$(cd "$(dirname "$0")" && pwd)
src=$here/../../../utils/l0/fault-absorb
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
clang -target spirv64 -cl-std=CL2.0 -O2 -c "$src/kernels.cl" \
  -o "$work/kernels.spv" || exit 1
c++ -O1 -std=c++17 "$src/repro.cpp" -o "$work/repro" -lze_loader || exit 1
since=$(date '+%Y-%m-%d %H:%M:%S')
for step in add fault add add; do
  "$work/repro" "$step" "$work/kernels.spv"
  echo "== repro $step: exit $?"
done
echo "== kernel log since $since"
journalctl -k --since "$since" --no-pager |
  grep -E "CAT error|Faulted Address|Timedout|reset|ban" | head -20
