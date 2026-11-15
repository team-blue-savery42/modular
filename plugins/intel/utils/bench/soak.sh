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

# Option lists ($OPT, $CONC, ...) are split into words on purpose; busy()
# lists render-node users through /proc.
# shellcheck disable=SC2086,SC2046,SC2010
# Soak test: an hour of mixed serving on one MAX server: random
# 512/128 and ShareGPT workloads at concurrencies cycling 1-64 with the same
# `vllm bench serve` client, fixed greedy completions at the start and end
# (must match), device memory sampled (Sysman) every 0.5 s, GPU faults
# counted from the kernel log. Cleans up on every exit; writes a DONE
# marker at the end of OUTDIR/summary.txt.
# Usage: sh plugins/intel/utils/bench/soak.sh OUTDIR MODEL [MINUTES] [PIPELINES ARGS...]
# MODELS_DIR (default ~/models) and DATASETS_DIR (default ~/datasets) as for
# serve.sh.
set -u
OUT=$1 MODEL=$2 MIN=${3:-60}; shift 2; [ $# -gt 0 ] && shift
mkdir -p "$OUT"
here=$(cd "$(dirname "$0")" && pwd); intel=$(cd "$here/../.." && pwd)
fork=$(cd "$intel/../.." && pwd)
M=${MODELS_DIR:-$HOME/models}/$MODEL DS=${DATASETS_DIR:-$HOME/datasets} IMG=intel/vllm:0.21.0-xpu
S=$(date "+%Y-%m-%d %H:%M:%S")
pid=
cleanup() {
  touch "$OUT/mem.stop"
  docker stop "intel-bench-soak-$$" >/dev/null 2>&1
  [ -n "$pid" ] && { kill -TERM -"$pid" 2>/dev/null; sleep 5; kill -KILL -"$pid" 2>/dev/null; }
}
trap cleanup EXIT
trap 'exit 143' INT TERM
cd "$fork" || exit 1
OPT="--compilation_mode=opt --symlink_prefix=/dev/null"
./bazelw build $OPT //plugins/intel/compiler:IntelCompilerPlugin.so //plugins/intel/driver:LevelZeroPlugin.so \
  //Target:CPUPlugin.so > "$OUT/build.log" 2>&1
BIN=$(./bazelw info $OPT output_path 2>/dev/null)/mojo_host_platform-opt/bin
HF_HUB_OFFLINE=1 MODULAR_COMPILER_PLUGINS=$BIN/plugins/intel/compiler/IntelCompilerPlugin.so \
MODULAR_DRIVER_PLUGINS="level_zero@$BIN/plugins/intel/driver/LevelZeroPlugin.so;CPU@$BIN/Target/CPUPlugin.so" \
  setsid ./bazelw run $OPT --//:hal_device_context=true //max/python/max/_entrypoints:pipelines -- serve \
  --model-path "$M" --devices gpu --max-length 8192 --max-batch-size 64 --no-enable-prefix-caching \
  --port 8000 "$@" > "$OUT/server.log" 2>&1 &
pid=$!
cd "$intel" || exit 1
python3 utils/bench/sysman_mem.py 0.5 "$OUT/mem.stop" > "$OUT/mem.txt" 2>&1 &
i=0; until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; i=$((i+1)); [ $i -gt 360 ] && break; done
python3 utils/bench/greedy_texts.py http://localhost:8000 "$M" > "$OUT/texts-start.jsonl" 2>&1
end=$(( $(date +%s) + MIN * 60 )); n=0
while [ "$(date +%s)" -lt "$end" ]; do
  for c in 1 8 64 4 32 16 2; do
    for w in random sharegpt; do
      [ "$(date +%s)" -lt "$end" ] || break 2
      case $w in
        random) ds="--dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos" ;;
        sharegpt) ds="--dataset-name sharegpt --dataset-path /ds/ShareGPT_V3_unfiltered_cleaned_split.json --sharegpt-output-len 512" ;;
      esac
      n=$((n + 1))
      # shellcheck disable=SC2086
      timeout 1800 docker run --rm --name "intel-bench-soak-$$" --network host -v "$M:/model:ro" -v "$DS:/ds:ro" $IMG \
        vllm bench serve --backend openai --endpoint /v1/completions --base-url http://localhost:8000 \
        --model "$M" --tokenizer /model $ds --num-prompts $((c * 4 > 16 ? c * 4 : 16)) \
        --max-concurrency $c --seed $n > "$OUT/bench-$n-$w-c$c.log" 2>&1
      echo "$(date -Is) $n $w c=$c rc=$? $(grep -h "Successful requests\|Failed requests" "$OUT/bench-$n-$w-c$c.log" | tr -s ' ' | tr '\n' ' ')" >> "$OUT/runs.txt"
    done
  done
done
python3 utils/bench/greedy_texts.py http://localhost:8000 "$M" > "$OUT/texts-end.jsonl" 2>&1
if cmp -s "$OUT/texts-start.jsonl" "$OUT/texts-end.jsonl" && [ -s "$OUT/texts-start.jsonl" ]; then t=same; else t=DIFFER; fi
echo "texts $t; faults $(journalctl -k --since "$S" --no-pager | grep -c 'CAT error\|Engine reset\|GPU HANG'); failed $(grep -o 'Failed requests: [0-9]*' "$OUT"/bench-*.log | awk '{s+=$3} END{print s+0}'); benches $n" >> "$OUT/summary.txt"
echo DONE >> "$OUT/summary.txt"
