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
# Serving benchmark for one engine and one model: start an
# OpenAI-compatible server, run the same client against it (`vllm bench
# serve` from intel/vllm:0.21.0-xpu) for each workload and concurrency, stop
# it. Workloads: `random` (512 in / 128 out, ignore EOS) and `sharegpt`
# (ShareGPT V3 conversations' first turns and reply lengths, output capped at
# 1024; DATASETS_DIR). max(32, 8 x c) requests, c warm-ups,
# seed 0; device memory (Sysman) sampled every 0.5 s to OUTDIR/mem.txt.
# Engines:
#   max:    `pipelines serve` (opt build, HAL runtime, our plugins), port
#           8000 (metrics 8001); MAX_BATCH requests per batch (default 64),
#           MAX_SERVE_ARGS adds options;
#   vllm:   `vllm serve` eager (VLLM_GRAPH=1: XPU graphs), port 8100;
#   llama:  llama.cpp's Vulkan llama-server ($LLAMA_VULKAN_BIN/llama-server),
#           port 8090, LLAMA_NP slots (default 64) sharing one unified f16 KV
#           cache of LLAMA_CTX tokens (default 65536);
#   sycl:   llama.cpp's SYCL llama-server ($LLAMA_SYCL_SRC/build-sycl) in
#           the llama-sycl-b70 image.
# Usage:
#   sh plugins/intel/utils/bench/serve.sh OUTDIR ENGINE MODEL GGUF [CONCURRENCY...]
# MAXLEN (default 8192) is each engine's context per request; WORKLOADS
# (default "random sharegpt").
# MODELS_DIR (default ~/models) holds Hugging Face snapshots by name and GGUF
# files in MODELS_DIR/gguf; DATASETS_DIR (default ~/datasets) holds
# ShareGPT_V3_unfiltered_cleaned_split.json.
set -u
OUT=$1 ENG=$2 MODEL=$3 GGUF=$4; shift 4; mkdir -p "$OUT"
CONC=${*:-1 2 4 8 16 32 64}
MAXLEN=${MAXLEN:-8192}
WORKLOADS=${WORKLOADS:-random sharegpt}
IMG=intel/vllm:0.21.0-xpu
MODELS=${MODELS_DIR:-$HOME/models}
M=$MODELS/$MODEL
DS=${DATASETS_DIR:-$HOME/datasets}
intel=$(cd "$(dirname "$0")/../.." && pwd)
fork=$(cd "$intel/../.." && pwd)
for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q renderD && echo "GPU busy before: $(cat $p/comm)" >> "$OUT/state.txt"; done
render=$(getent group render | cut -d: -f3)
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
      --model-path "$M" --devices gpu --max-length "$MAXLEN" --max-batch-size "${MAX_BATCH:-64}" \
      --no-enable-prefix-caching --port $port ${MAX_SERVE_ARGS:-} > "$OUT/server.log" 2>&1 &
    pid=$!; cd "$intel" || exit 1 ;;
  vllm)
    port=8100
    docker run -d --name intel-bench-vllm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro --ipc=host \
      --network host -e VLLM_XPU_ENABLE_XPU_GRAPH="${VLLM_GRAPH:-0}" -v "$M:/model:ro" $IMG vllm serve /model \
      --served-model-name "$M" --dtype "${VLLM_DTYPE:-bfloat16}" --max-model-len "$MAXLEN" \
      --gpu-memory-utilization 0.9 --no-enable-prefix-caching --max-num-seqs 64 \
      --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]}' \
      ${VLLM_SERVE_ARGS:-} --port $port > "$OUT/server.id" 2>&1 ;;
  llama|sycl)
    port=8090
    np=${LLAMA_NP:-64}
    args="-m /models/$GGUF -ngl 99 -fa on -c ${LLAMA_CTX:-65536} -np $np --kv-unified -b 2048 -ub 2048 \
      --cache-type-k f16 --cache-type-v f16 --no-cache-prompt --host 127.0.0.1 --port $port --alias $M"
    if [ "$ENG" = llama ]; then
      # shellcheck disable=SC2086
      setsid "${LLAMA_VULKAN_BIN:-$HOME/src/llama.cpp/build-vulkan/bin}/llama-server" $(echo "$args" | sed "s#/models/#$MODELS/gguf/#") \
        > "$OUT/server.log" 2>&1 &
      pid=$!
    else
      docker run -d --name intel-bench-sycl --user "$(id -u):$(id -g)" --group-add "$render" --device /dev/dri \
        --network host -v "${LLAMA_SYCL_SRC:-$HOME/src/llama.cpp}:/src:ro" -v "$MODELS/gguf:/models:ro" \
        --entrypoint /src/build-sycl/bin/llama-server ghcr.io/snailium/llama.cpp-sycl-intel-b70/llama-sycl-b70:stable \
        $args > "$OUT/server.id" 2>&1
    fi ;;
esac
python3 "$intel/utils/bench/sysman_mem.py" 0.5 "$OUT/mem.stop" > "$OUT/mem.txt" 2>&1 &
# Every exit path, a timeout's TERM included, stops the sampler, the bench
# client's container and the server's process group (or container).
cleanup() {
  touch "$OUT/mem.stop"
  docker stop "intel-bench-client-$$" >/dev/null 2>&1
  [ -n "${pid:-}" ] && { kill -TERM -"$pid" 2>/dev/null; sleep 5; kill -KILL -"$pid" 2>/dev/null; }
  docker stop intel-bench-vllm intel-bench-sycl >/dev/null 2>&1
  docker rm intel-bench-vllm intel-bench-sycl >/dev/null 2>&1
}
trap cleanup EXIT
trap 'exit 143' INT TERM
i=0; until curl -sf localhost:$port/v1/models >/dev/null; do sleep 5; i=$((i+1)); [ $i -gt 360 ] && break; done
if ! curl -sf localhost:$port/v1/models >/dev/null; then
  echo "server did not start" >> "$OUT/state.txt"
else
  echo "server up after $((i * 5)) s" >> "$OUT/state.txt"
  python3 "$intel/utils/bench/stream_probe.py" "http://localhost:$port" "$M" > "$OUT/stream.txt" 2>&1
  for w in $WORKLOADS; do
    for c in $CONC; do
      n=$((c * 8)); [ $n -lt 32 ] && n=32
      case $w in
        random) ds="--dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos" ;;
        sharegpt) ds="--dataset-name sharegpt --dataset-path /ds/ShareGPT_V3_unfiltered_cleaned_split.json --sharegpt-output-len 1024" ;;
      esac
      # shellcheck disable=SC2086
      timeout 3600 docker run --rm --name "intel-bench-client-$$" --network host -v "$M:/model:ro" -v "$DS:/ds:ro" $IMG vllm bench serve \
        --backend openai --endpoint /v1/completions --base-url "http://localhost:$port" --model "$M" --tokenizer /model \
        $ds --num-prompts $n --max-concurrency "$c" --num-warmups "$c" --seed 0 \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 > "$OUT/bench-$w-c$c.log" 2>&1
      echo "$w c=$c rc=$?" >> "$OUT/state.txt"
    done
  done
fi
case $ENG in
  max) curl -s localhost:8001/metrics > "$OUT/metrics.txt" 2>&1
       kill -TERM -"$pid" 2>/dev/null; sleep 10; kill -KILL -"$pid" 2>/dev/null
       pkill -f "max/_entrypoints/pipelines serve" 2>/dev/null ;;
  vllm) docker logs intel-bench-vllm > "$OUT/server.log" 2>&1; docker stop intel-bench-vllm >/dev/null; docker rm intel-bench-vllm >/dev/null ;;
  llama) kill -TERM -"$pid" 2>/dev/null; sleep 5; kill -KILL -"$pid" 2>/dev/null ;;
  sycl) docker logs intel-bench-sycl > "$OUT/server.log" 2>&1; docker stop intel-bench-sycl >/dev/null; docker rm intel-bench-sycl >/dev/null ;;
esac
touch "$OUT/mem.stop"
echo END >> "$OUT/state.txt"
