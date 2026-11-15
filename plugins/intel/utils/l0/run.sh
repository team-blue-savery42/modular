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

# Build (once) and run run.cpp: run.sh <module.spv> <kernel-name-substring>
set -eu
here=$(cd "$(dirname "$0")" && pwd)
bin=${XDG_CACHE_HOME:-$HOME/.cache}/intel-plugin/l0-run
mkdir -p "$(dirname "$bin")"
if [ ! -x "$bin" ] || [ "$here/run.cpp" -nt "$bin" ]; then
  c++ -std=c++20 -O2 -Wall -Wextra -Werror "$here/run.cpp" -lze_loader -o "$bin"
fi
exec "$bin" "$@"
