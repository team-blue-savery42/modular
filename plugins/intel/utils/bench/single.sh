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
# Single-stream benchmark for one model, every engine in the same machine
# state (stop other GPU users first):
#   prompt processing at 512, 4096 and LONG tokens (default 32768);
#   decode (128 tokens, batch 1) after a 512- and a 4096-token prompt.
# Engines (ENGINES, default "sycl vulkan max vllm"):
#   sycl:   llama.cpp SYCL ($LLAMA_SYCL_SRC/build-sycl, in the
#           llama-sycl-b70 image), llama-bench -r 3;
#   vulkan: llama.cpp's Vulkan build ($LLAMA_VULKAN_BIN), llama-bench -r 3;
#   max:    tests/graph/bench_model.py (opt build): prompt steps with the
#           overlap scheduler off, decode with it on; the long
#           prompt in one step (--max-batch-input-tokens; vLLM chunks it, and
#           its pp figure for it is generate(max_tokens=1));
#   vllm:   utils/bench/vllm_steps.py in intel/vllm:0.21.0-xpu, prompt with
#           async scheduling off, decode with it on (VLLM_GRAPH=1 for XPU
#           graphs; default eager).
# Usage: sh plugins/intel/utils/bench/single.sh OUTDIR MODEL GGUF [MAX_ARGS...]
#   MODEL: directory name in MODELS_DIR (default ~/models); GGUF: file name
#   in MODELS_DIR/gguf; MAX_ARGS go to bench_model.py
#   (e.g. --encoding float8_e4m3fn). LONG overrides the long prompt; REPS the
#   repetitions (default 3); VLLM_DTYPE the vLLM dtype (default bfloat16);
#   MAX_PP and MAX_TG the MAX prompt lengths to run (defaults as above).
set -u
OUT=$1 MODEL=$2 GGUF=$3; shift 3
mkdir -p "$OUT"
LONG=${LONG:-32768} REPS=${REPS:-3}
ENGINES=${ENGINES:-sycl vulkan max vllm}
here=$(cd "$(dirname "$0")" && pwd)
# utils/run-max-py and tests/graph relative to plugins/intel.
cd "$here/../.." || exit 1
M=${MODELS_DIR:-$HOME/models}/$MODEL
src=${LLAMA_SYCL_SRC:-$HOME/src/llama.cpp}
models=${MODELS_DIR:-$HOME/models}/gguf
image=ghcr.io/snailium/llama.cpp-sycl-intel-b70/llama-sycl-b70:stable
render=$(getent group render | cut -d: -f3)
busy() { for q in /proc/[0-9]*; do ls -l $q/fd 2>/dev/null | grep -q renderD && echo "GPU busy before $1: $(cat $q/comm)" >> "$OUT/state.txt"; done; sleep 3; }
log() { echo "$(date -Is) $*" >> "$OUT/run.txt"; }
{ echo "date $(date -Is)"; uptime
  git -C "$here" rev-parse --short HEAD
  d=/sys/class/drm/card0/device/tile0/gt0/freq0
  echo "freq min $(cat $d/min_freq) max $(cat $d/max_freq) act $(cat $d/act_freq)"; } > "$OUT/system.txt" 2>&1
lbench() { # ENGINE TAG ARGS...
  eng=$1 tag=$2; shift 2; busy "$eng-$tag"
  if [ "$eng" = sycl ]; then
    timeout 3600 docker run --rm --user "$(id -u):$(id -g)" --group-add "$render" --device /dev/dri \
      -v "$src:/src:ro" -v "$models:/models:ro" -w /src --entrypoint /src/build-sycl/bin/llama-bench \
      "$image" -m "/models/$GGUF" -ngl 99 -fa 1 -r "$REPS" "$@" > "$OUT/$eng-$tag.log" 2>&1
  else
    timeout 3600 "${LLAMA_VULKAN_BIN:-$HOME/src/llama.cpp/build-vulkan/bin}/llama-bench" -m "$models/$GGUF" -ngl 99 -fa 1 \
      -r "$REPS" "$@" > "$OUT/$eng-$tag.log" 2>&1
  fi
  log "$eng-$tag rc=$?"
}
llama() {
  lbench "$1" pp -p "512,4096,$LONG" -n 0 -ub 2048 -b 2048
  lbench "$1" tg -p 0 -n 128 -d 512,4096
}
max() {
  for p in ${MAX_PP-512 4096 $LONG}; do
    d=8; ov=--no-overlap
    busy "max-pp$p"
    RUN_MAX_PY_OPT=1 timeout 3600 utils/run-max-py tests/graph/bench_model.py "$M" --prompt "$p" \
      --decode "$d" --reps "$REPS" --max-length $((p + 160)) $ov "$@" \
      --extra=--max-batch-input-tokens=$((p > 8192 ? p : 8192)) > "$OUT/max-pp$p.log" 2>&1
    log "max-pp$p rc=$? $(grep RESULT "$OUT/max-pp$p.log")"
  done
  for p in ${MAX_TG-512 4096}; do
    busy "max-tg-d$p"
    RUN_MAX_PY_OPT=1 timeout 3600 utils/run-max-py tests/graph/bench_model.py "$M" --prompt "$p" \
      --decode 128 --reps "$REPS" --max-length $((p + 160)) "$@" > "$OUT/max-tg-d$p.log" 2>&1
    log "max-tg-d$p rc=$? $(grep RESULT "$OUT/max-tg-d$p.log")"
  done
}
vllm() {
  run() { # TAG SYNC PROMPT DECODE
    busy "vllm-$1"
    timeout 3600 docker run --rm --name intel-bench-steps --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro --ipc=host \
      -e STEPS_SYNC="$2" -e STEPS_PROMPT="$3" -e STEPS_DECODE="$4" -e STEPS_MAXLEN=$(($3 + 160)) \
      -e STEPS_ID_HI=100000 -e STEPS_DTYPE="${VLLM_DTYPE:-bfloat16}" -e STEPS_GPU_MEM=0.9 \
      -e VLLM_XPU_ENABLE_XPU_GRAPH="${VLLM_GRAPH:-0}" -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      -v "$M:/model:ro" -v "$here:/bench:ro" intel/vllm:0.21.0-xpu \
      python3 /bench/vllm_steps.py "$REPS" > "$OUT/vllm-$1.log" 2>&1
    rc=$?
    # A killed client leaves the container running (vLLM's XPU graphs hang
    # after UR_RESULT_ERROR_OUT_OF_RESOURCES): stop it.
    docker stop intel-bench-steps >/dev/null 2>&1
    log "vllm-$1 rc=$rc $(grep RESULT "$OUT/vllm-$1.log")"
  }
  for p in 512 4096 "$LONG"; do run "pp$p" 1 "$p" 8; done
  for p in 512 4096; do run "tg-d$p" 0 "$p" 128; done
}
for e in $ENGINES; do
  case $e in
    sycl|vulkan) llama "$e" ;;
    max) max "$@" ;;
    vllm) vllm ;;
  esac
done
log SINGLE-END
