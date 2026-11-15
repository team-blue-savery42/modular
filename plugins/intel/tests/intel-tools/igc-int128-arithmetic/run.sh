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

# Compiles mul128.ll, ops.ll and vars.ll to SPIR-V, shows their 128-bit
# operations, then runs each kernel with IGC's default optimization and with
# -cl-opt-disable and compares every result with the host's. Needs clang
# (spirv64), python3, spirv-val, spirv-dis, clang++ and a bmg-g31 GPU on
# Level Zero.
set -u
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
for name in mul128 ops vars; do
  clang -target spirv64 -Wno-override-module \
    -mllvm --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers \
    -c "$here/$name.ll" -o "$work/$name-altera.spv" || exit 1
  # clang 23 names the extension SPV_ALTERA_arbitrary_precision_integers,
  # which IGC 2.41 rejects; it takes the older name
  # SPV_INTEL_arbitrary_precision_integers (the same capability, 5844).
  # Rename it in place: both strings take 10 words with their null padding.
  # (spirv-dis prints no literal for a 128-bit OpConstant, so the text form
  # can't round-trip.)
  python3 - "$work/$name-altera.spv" "$work/$name.spv" <<'PY' || exit 1
import sys
data = open(sys.argv[1], "rb").read()
old = b"SPV_ALTERA_arbitrary_precision_integers\0"
new = b"SPV_INTEL_arbitrary_precision_integers\0\0"
assert data.count(old) == 1 and len(old) == len(new) == 40
open(sys.argv[2], "wb").write(data.replace(old, new))
PY
  echo "== $name.ll"
  spirv-val "$work/$name.spv" && echo "spirv-val: valid"
  echo "128-bit operations in the SPIR-V:"
  spirv-dis "$work/$name.spv" |
    grep -E "Extension|= Op(IMul|IAdd|BitwiseAnd|BitwiseXor|UConvert|ShiftLeftLogical|ShiftRightLogical) %u128 "
done
clang++ -O1 -std=c++17 "$here/run-mul128.cpp" -o "$work/run-mul128" \
  -lze_loader || exit 1
for name in mul128 ops vars; do
  echo "== run $name"
  NEO_CACHE_PERSISTENT=0 "$work/run-mul128" "$work/$name.spv"
  echo "run-mul128: exit $?"
done
