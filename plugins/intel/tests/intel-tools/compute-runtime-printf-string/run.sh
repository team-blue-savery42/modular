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

# Builds kernels.cl and utils/l0/run.cpp (a Level Zero runner for one
# kernel), then runs each kernel, which prints a string with printf's %s: a
# literal, a string in a shared-memory buffer and a string in private
# memory. Prints each run's exit status (139: the host process segfaulted).
# Needs clang (spirv64), spirv-val, c++ and a bmg-g31 GPU on Level Zero.
set -u
here=$(cd "$(dirname "$0")" && pwd)
l0=$here/../../../utils/l0
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
clang -target spirv64 -cl-std=CL2.0 -O2 -c "$here/kernels.cl" \
  -o "$work/kernels.spv" || exit 1
spirv-val "$work/kernels.spv" && echo "spirv-val: valid"
c++ -std=c++20 -O2 "$l0/run.cpp" -lze_loader -o "$work/l0-run" || exit 1
for kernel in literal_string shared_buffer_string private_string; do
  echo "== $kernel"
  (cd "$work" && exec ./l0-run kernels.spv "$kernel" 2>&1)
  echo "l0-run $kernel: exit $?"
done
