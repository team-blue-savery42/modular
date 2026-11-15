#!/bin/bash
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

# shellcheck disable=SC2086,SC2046,SC2010
# Agent-sized prompts through a server, one engine and one model (stop other
# GPU users first). Starts the server (prefix caching on, context 40,000),
# then runs long_client.py: `long` cells at 8k, 16k, 24k and 32k tokens (one
# request at a time, 3 requests each, 64 output tokens, distinct prompts) and
# the `prefix` scenario (a 24k-token context shared by the agents, 4 turns of
# 256 new tokens each) at one and AGENTS (default "4 8") concurrent agents.
# The server stays up for all cells; every client call has a timeout and a
# failure is recorded as it is. Results: OUTDIR/long-N.json, prefix-cC.json,
# state.txt.
# Engines: ENGINE=max (default; `pipelines serve`, opt build) or vllm
# (intel/vllm:0.21.0-xpu). CHUNK (default 8192) is --max-batch-input-tokens /
# --max-num-batched-tokens: 8192 chunks a 24k prompt, 32768 runs it in one
# step. SERVE_ARGS adds server options (Qwen3-8B on MAX needs
# --no-prefer-module-v3); LENS, AGENTS, PREFIX override the cells;
# PREFIXCACHE=0 turns prefix caching off; CLI_ARGS=--chat sends text prompts
# through the chat endpoint (Qwen3.5 on MAX: its completions stream shows no
# tokens, the reasoning parser takes them).
# Usage: [ENGINE=max|vllm] [CHUNK=8192] sh plugins/intel/utils/bench/long.sh OUTDIR MODEL
# MODELS_DIR (default ~/models) holds the Hugging Face snapshot MODEL.
set -u
OUT=$1 MODEL=$2; mkdir -p "$OUT"
ENG=${ENGINE:-max} CHUNK=${CHUNK:-8192}
LENS=${LENS:-"8192 16384 24576 32768"} AGENTS=${AGENTS:-"4 8"} PREFIX=${PREFIX:-24576}
here=$(cd "$(dirname "$0")" && pwd)
IMG=intel/vllm:0.21.0-xpu
M=${MODELS_DIR:-$HOME/models}/$MODEL
intel=$(cd "$here/../.." && pwd)
fork=$(cd "$intel/../.." && pwd)
log() { echo "$(date -Is) $*" >> "$OUT/state.txt"; }
pid=""
cleanup() {
  touch "$OUT/mem.stop"
  [ -n "$pid" ] && { kill -TERM -"$pid" 2>/dev/null; sleep 5; kill -KILL -"$pid" 2>/dev/null; }
  pkill -f "max/_entrypoints/pipelines serve" 2>/dev/null
  docker stop intel-bench-vllm >/dev/null 2>&1; docker rm intel-bench-vllm >/dev/null 2>&1
}
trap cleanup EXIT
trap 'exit 143' INT TERM
for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q renderD && log "GPU busy before: $(cat $p/comm)"; done
PC=--enable-prefix-caching; [ "${PREFIXCACHE:-1}" = 0 ] && PC=--no-enable-prefix-caching
case $ENG in
  max)
    port=8000
    cd "$fork" || exit 1
    OPT="--compilation_mode=opt --symlink_prefix=/dev/null"
    ./bazelw build $OPT //plugins/intel/compiler:IntelCompilerPlugin.so //plugins/intel/driver:LevelZeroPlugin.so \
      //Target:CPUPlugin.so > "$OUT/build.log" 2>&1
    BIN=$(./bazelw info $OPT output_path 2>/dev/null)/mojo_host_platform-opt/bin
    HF_HUB_OFFLINE=1 MODULAR_COMPILER_PLUGINS=$BIN/plugins/intel/compiler/IntelCompilerPlugin.so \
    MODULAR_DRIVER_PLUGINS="level_zero@$BIN/plugins/intel/driver/LevelZeroPlugin.so;CPU@$BIN/Target/CPUPlugin.so" \
      setsid ./bazelw run $OPT --//:hal_device_context=true //max/python/max/_entrypoints:pipelines -- serve \
      --model-path "$M" --devices gpu --max-length 40000 --max-batch-size 64 \
      --max-batch-input-tokens "$CHUNK" $PC --port $port ${SERVE_ARGS:-} > "$OUT/server.log" 2>&1 &
    pid=$!; cd "$intel" || exit 1 ;;
  vllm)
    port=8100
    docker run -d --name intel-bench-vllm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro --ipc=host \
      --network host -e VLLM_XPU_ENABLE_XPU_GRAPH="${VLLM_GRAPH:-0}" -v "$M:/model:ro" $IMG vllm serve /model \
      --served-model-name "$M" --dtype bfloat16 --max-model-len 40000 \
      --gpu-memory-utilization 0.9 $PC --max-num-seqs 64 --max-num-batched-tokens "$CHUNK" \
      --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]}' \
      ${SERVE_ARGS:-} --port $port > "$OUT/server.id" 2>&1 ;;
esac
python3 "$here/sysman_mem.py" 0.5 "$OUT/mem.stop" > "$OUT/mem.txt" 2>&1 &
i=0; until curl -sf localhost:$port/v1/models >/dev/null; do sleep 5; i=$((i+1)); [ $i -gt 480 ] && break; done
if ! curl -sf localhost:$port/v1/models >/dev/null; then log "server did not start"; exit 1; fi
log "server up after $((i * 5)) s ($ENG $MODEL chunk=$CHUNK prefixcache=${PREFIXCACHE:-1})"
URL=http://localhost:$port
cli() { timeout -k 10 "${CLI_TIMEOUT:-3000}" python3 "$here/long_client.py" "$URL" "$M" "$@" ${CLI_ARGS:-}; }
# Warm up with a short request, then a 2k one, so that the measured cells
# don't include the server's first requests.
cli long --len 128 --n 1 --out 8 > "$OUT/warm.json" 2>&1
cli long --len 2048 --n 1 --out 8 >> "$OUT/warm.json" 2>&1
for n in $LENS; do
  cli long --len "$n" --n 3 --c 1 --out 64 --seed "$n" > "$OUT/long-$n.json" 2> "$OUT/long-$n.err"
  log "long $n rc=$? $(python3 -c "import json;d=json.load(open('$OUT/long-$n.json'));print(d['summary'])" 2>&1 | cut -c1-400)"
done
for c in 1 $AGENTS; do
  cli prefix --prefix "$PREFIX" --new 256 --c "$c" --turns 4 --out 64 --seed "$c" > "$OUT/prefix-c$c.json" 2> "$OUT/prefix-c$c.err"
  log "prefix c=$c rc=$? $(python3 -c "import json;d=json.load(open('$OUT/prefix-c$c.json'));print(d['cold'],d['turns'])" 2>&1 | cut -c1-900)"
done
curl -s localhost:8001/metrics > "$OUT/metrics.txt" 2>&1
case $ENG in vllm) docker logs intel-bench-vllm > "$OUT/server.log" 2>&1 ;; esac
log END
