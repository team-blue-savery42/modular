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

# download a public, ungated Hugging Face model at a pinned revision
# into ${MODELS_DIR:-~/models}/NAME with curl (no Hugging Face tooling needed).
# Skips the `original/` and `metal/` subtrees (other runtimes' copies) and
# files that already have the size the API lists. Writes REVISION.
# Usage: fetch.sh REPO REVISION NAME
set -eu
repo=$1 rev=$2 name=$3
dst=${MODELS_DIR:-$HOME/models}/$name
mkdir -p "$dst"
curl -sf "https://huggingface.co/api/models/$repo/revision/$rev?blobs=true" > "$dst/.api.json"
python3 -I -c '
import json, sys
d = json.load(open(sys.argv[1]))
assert not d.get("gated"), "gated model"
for s in d["siblings"]:
    f = s["rfilename"]
    if f.startswith(("original/", "metal/")) or f == ".gitattributes":
        continue
    print(f, s.get("size", -1))
' "$dst/.api.json" | while read -r f size; do
  mkdir -p "$dst/$(dirname "$f")"
  if [ -f "$dst/$f" ] && [ "$(stat -c %s "$dst/$f")" = "$size" ]; then continue; fi
  curl -sfL --retry 5 -C - -o "$dst/$f" "https://huggingface.co/$repo/resolve/$rev/$f"
  echo "$f $(stat -c %s "$dst/$f")"
done
echo "$rev" > "$dst/REVISION"
echo "done $repo $rev"
