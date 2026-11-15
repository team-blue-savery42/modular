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
# kernel) and runs the kernel twice: without Level Zero's validation layer
# and with it (ZE_ENABLE_VALIDATION_LAYER=1). Prints each exit status (139:
# the process segfaulted). LD_LIBRARY_PATH picks the loader and layer, as
# for any Level Zero program. Needs clang (spirv64), c++ and a GPU on Level
# Zero.
set -u
here=$(cd "$(dirname "$0")" && pwd)
l0=$here/../../../utils/l0
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
clang -target spirv64 -cl-std=CL2.0 -O2 -c "$here/kernels.cl" \
  -o "$work/kernels.spv" || exit 1
c++ -std=c++20 -O2 "$l0/run.cpp" -lze_loader -o "$work/l0-run" || exit 1
echo "== without the validation layer"
(cd "$work" && exec ./l0-run kernels.spv store_one 2>&1)
echo "exit $?"
echo "== ZE_ENABLE_VALIDATION_LAYER=1 ZE_ENABLE_PARAMETER_VALIDATION=1"
(cd "$work" && ZE_ENABLE_VALIDATION_LAYER=1 ZE_ENABLE_PARAMETER_VALIDATION=1 \
  exec ./l0-run kernels.spv store_one 2>&1)
echo "exit $?"
