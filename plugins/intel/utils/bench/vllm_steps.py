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

"""vLLM-XPU engine-step timings, defined as MAX's bench_model.py times them
(like-for-like with tests/graph/bench_model.py).

Runs inside intel/vllm:0.21.0-xpu with the model at /model (read-only) and
VLLM_ENABLE_V1_MULTIPROCESSING=0 (the engine core in this process):
  python3 /bench/vllm_steps.py [REPS]
Wraps the engine core's step: each call schedules, runs the model and
samples, and returns once the sampled token is on the host, as one MAX
pipeline step does. For each repetition, a fresh random 512-token prompt
(prefix caching off) generates 129 tokens greedily: the first step is the
prompt (pp = 512 / its wall time); the 128 steps after it are decode
(tg = 128 / their summed wall time). Every engine step is recorded; the
logs show 130 per repetition (the prompt, 128 decode steps, one trailing
step that only collects outputs, printed as "other steps"), so a step
without tokens inside the decode would be counted: check that count. Also
prints the time of generate(max_tokens=1) for reference.
Medians of REPS after one warm-up. STEPS_QUANT=<method> passes vLLM's
`quantization` (`fp8`, online FP8 weights from the bf16 checkpoint).
"""

import os
import statistics
import sys
import time

import numpy as np

assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", (
    "run with VLLM_ENABLE_V1_MULTIPROCESSING=0"
)

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.v1.engine.core import EngineCore

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
# STEPS_SYNC=1: async scheduling off, so each engine step returns its own
# token (as MAX's bench_model.py --no-overlap); the prompt step is then the
# whole prefill. With it on (vLLM's default), a step may return before its
# batch finishes on the device.
SYNC = os.environ.get("STEPS_SYNC") == "1"
# STEPS_PROMPT, STEPS_DECODE, STEPS_MAXLEN and STEPS_DTYPE (default
# bfloat16; "auto" for a checkpoint's own mix, e.g. gpt-oss's MXFP4 experts)
# set the shape and model.
PROMPT = int(os.environ.get("STEPS_PROMPT", "512"))
DECODE = int(os.environ.get("STEPS_DECODE", "128"))
MAXLEN = int(os.environ.get("STEPS_MAXLEN", "4096"))
DTYPE = os.environ.get("STEPS_DTYPE", "bfloat16")
# STEPS_ID_HI: random prompt ids are drawn from [1000, STEPS_ID_HI) (keep it
# under the vocabulary; 100000, as bench_model.py uses).

steps: list = []


def wrap(name):
    inner = getattr(EngineCore, name, None)
    if inner is None:
        return

    def timed(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = inner(self, *args, **kwargs)
        steps.append((name, time.perf_counter() - t0))
        return out

    setattr(EngineCore, name, timed)


for n in ("step", "step_with_batch_queue"):
    wrap(n)


def main() -> None:
    llm = LLM(
        model="/model",
        dtype=DTYPE,
        max_model_len=MAXLEN,
        gpu_memory_utilization=float(os.environ.get("STEPS_GPU_MEM", "0.8")),
        enable_prefix_caching=False,
        max_num_seqs=1,
        seed=0,
        **(
            {"quantization": os.environ["STEPS_QUANT"]}
            if os.environ.get("STEPS_QUANT")
            else {}
        ),
        **({"async_scheduling": False} if SYNC else {}),
    )
    rng = np.random.default_rng(0)

    def run(n_in, n_out):
        p = TokensPrompt(
            prompt_token_ids=rng.integers(
                1000, int(os.environ.get("STEPS_ID_HI", "200000")), n_in
            ).tolist()
        )
        sp = SamplingParams(temperature=0.0, max_tokens=n_out, ignore_eos=True)
        steps.clear()
        t0 = time.perf_counter()
        out = llm.generate([p], sp, use_tqdm=False)
        total = time.perf_counter() - t0
        assert len(out[0].outputs[0].token_ids) == n_out
        return total, list(steps)

    run(PROMPT, DECODE + 1)
    pre, dec, gen1 = [], [], []
    for r in range(REPS):
        total, st = run(PROMPT, DECODE + 1)
        names = sorted({s[0] for s in st})
        times = [s[1] for s in st]
        # The prompt is the first step; the last DECODE steps produce the
        # remaining tokens (a trailing step may only collect outputs).
        first = times[0]
        decode = times[1 : 1 + DECODE]
        pre.append(first)
        dec.append(sum(decode))
        print(
            f"rep {r}: {len(times)} engine steps ({','.join(names)}), "
            f"generate {total * 1e3:.2f} ms; prompt step {first * 1e3:.2f} ms "
            f"({PROMPT / first:.0f} tok/s); {len(decode)} decode steps "
            f"{sum(decode) * 1e3:.2f} ms = {sum(decode) / len(decode) * 1e3:.3f} "
            f"ms/token; other steps "
            f"{[round(t * 1e3, 2) for t in times[1 + DECODE :]]}",
            flush=True,
        )
        t1, _ = run(PROMPT, 1)
        gen1.append(t1)
    p, d, g = (
        statistics.median(pre),
        statistics.median(dec),
        statistics.median(gen1),
    )
    print(
        f"RESULT quantization {os.environ.get('STEPS_QUANT') or 'none'}, "
        f"async scheduling {'off' if SYNC else 'on'}, prompt {PROMPT}: engine step median {p * 1e3:.2f} ms = "
        f"{PROMPT / p:.0f} tok/s (range {PROMPT / max(pre):.0f}-"
        f"{PROMPT / min(pre):.0f}); generate(max_tokens=1) median "
        f"{g * 1e3:.2f} ms = {PROMPT / g:.0f} tok/s; decode median "
        f"{d / DECODE * 1e3:.3f} ms/token = {DECODE / d:.1f} tok/s (range "
        f"{DECODE / max(dec):.1f}-{DECODE / min(dec):.1f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
