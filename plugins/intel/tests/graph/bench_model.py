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

"""Times a model's prompt processing and decode the way llama-bench does
(`pp512` and decode tok/s after a 512-token prompt).

Builds the pipeline once, the way `pipelines generate` does, then for
each repetition takes a fresh random prompt of PROMPT token ids (no prefix
reuse: prefix caching off, a new prompt each time), and steps the pipeline
directly (`GenerateIterator.step`, one `pipeline.execute` per step, no
tokenizer or detokenizer in the timed region):

- prefill: the wall time of the first step, which processes the whole prompt
  and returns its first token (only meaningful with the overlap scheduler
  off, where a step returns its own token): pp = PROMPT / time;
- decode: the wall time of DECODE further steps, one token each, after the
  prompt (with the overlap scheduler on, steps return the previous step's
  token, so DECODE consecutive steps in the steady state are DECODE tokens):
  tg = DECODE / time.

Usage :
  RUN_MAX_PY_OPT=1 utils/run-max-py tests/graph/bench_model.py MODEL_PATH \
      [--prompt 512] [--decode 128] [--reps 5] [--no-overlap] [--marks FILE]
      [--encoding float8_e4m3fn]
Prints one line per repetition and the medians. --marks writes each step's
start and end (time.monotonic_ns(), the clock of the driver plugin's
MOJO_INTEL_PROFILE records) as `<rep> <step> <prefill|decode> <start> <end>`
for utils/bench/steps.py.
"""

import argparse
import statistics
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-overlap", action="store_true")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--marks")
    ap.add_argument(
        "--encoding", help="--quantization-encoding (e.g. float8_e4m3fn)"
    )
    ap.add_argument(
        "--graph-capture",
        action="store_true",
        help="device graphs: --device-graph-capture and"
        " warmup_graph_capture() before timing",
    )
    ap.add_argument(
        "--extra",
        action="append",
        default=[],
        help="one more `pipelines generate` option, repeatable,"
        " e.g. --extra=--no-prefer-module-v3",
    )
    a = ap.parse_args()

    from max._entrypoints import pipelines as cli
    from max._entrypoints.cli import generate as gen

    done: dict = {}

    def run(pipeline_args, sampling_params, prompt, **_ignored):
        import asyncio
        import dataclasses

        import numpy as np
        from max.pipelines import PIPELINE_REGISTRY
        from max.pipelines.lib import PipelineConfig
        from max.pipelines.lib.generate import GenerateIterator
        from max.pipelines.modeling.types import (
            RequestID,
            TextGenerationRequest,
        )

        config = PipelineConfig.from_args(
            gen._with_debug_defaults(pipeline_args)
        )
        pipeline = PIPELINE_REGISTRY.retrieve_factory(config).factory()
        if a.graph_capture:
            pipeline.warmup_graph_capture()
        print(
            f"pipeline {type(pipeline).__name__}, overlap off: {a.no_overlap}",
            flush=True,
        )
        rng = np.random.default_rng(6)
        params = dataclasses.replace(
            sampling_params, max_new_tokens=a.decode + 2
        )

        marks = open(a.marks, "w") if a.marks else None

        def step(batch, rep, i, kind):
            t0 = time.monotonic_ns()
            out = batch.step()
            if marks is not None:
                marks.write(f"{rep} {i} {kind} {t0} {time.monotonic_ns()}\n")
            return out

        def one(rep):
            ids = [int(x) for x in rng.integers(1000, 100000, a.prompt)]
            request = TextGenerationRequest(
                request_id=RequestID(),
                prompt=ids,
                model_name=gen.MODEL_NAME,
                sampling_params=params,
            )
            context = asyncio.run(pipeline.tokenizer.new_context(request))
            assert len(context.tokens) == a.prompt, len(context.tokens)
            with GenerateIterator.from_contexts(
                pipeline, [context], caller="bench_model"
            ) as batch:
                t0 = time.perf_counter()
                step(batch, rep, 0, "prefill")
                t1 = time.perf_counter()
                # first decode step (overlap: returns the prompt's token)
                step(batch, rep, 1, "decode1")
                t2 = time.perf_counter()
                got = 0
                for i in range(a.decode):
                    got += len(step(batch, rep, i + 2, "decode").get(0, []))
                t3 = time.perf_counter()
                while not batch.done:
                    batch.step()
            pre, dec = t1 - t0, t3 - t2
            print(
                f"rep {rep}: first step {pre * 1e3:.2f} ms "
                f"({a.prompt / pre:.0f} tok/s), step 2 {(t2 - t1) * 1e3:.2f} ms, "
                f"{a.decode} decode steps {dec * 1e3:.2f} ms = "
                f"{dec / a.decode * 1e3:.3f} ms/token ({a.decode / dec:.1f} tok/s), "
                f"tokens returned {got}",
                flush=True,
            )
            return pre, dec

        one("warm-up")
        res = [one(r) for r in range(a.reps)]
        pre = statistics.median(r[0] for r in res)
        dec = statistics.median(r[1] for r in res)
        print(
            f"RESULT prompt {a.prompt} decode {a.decode} overlap "
            f"{'off' if a.no_overlap else 'on'}: first step median "
            f"{pre * 1e3:.2f} ms = {a.prompt / pre:.0f} tok/s; decode median "
            f"{dec / a.decode * 1e3:.3f} ms/token = {a.decode / dec:.1f} tok/s "
            f"(range {min(a.decode / r[1] for r in res):.1f}-"
            f"{max(a.decode / r[1] for r in res):.1f})",
            flush=True,
        )
        if marks is not None:
            marks.close()
        done["ok"] = True

    gen.generate_text_for_pipeline = run
    sys.argv = [
        "pipelines",
        "generate",
        "--model-path",
        a.model,
        "--devices",
        "gpu",
        "--max-length",
        str(a.max_length),
        "--max-new-tokens",
        str(a.decode + 2),
        "--max-batch-size",
        "1",
        "--top-k",
        "1",
        "--ignore-eos",
        "--no-enable-prefix-caching",
        "--prompt",
        "unused",
        *(["--no-enable-overlap-scheduler"] if a.no_overlap else []),
        *(["--quantization-encoding", a.encoding] if a.encoding else []),
        *(["--device-graph-capture"] if a.graph_capture else []),
        *a.extra,
    ]
    try:
        cli.main(standalone_mode=False)
    except SystemExit:
        pass
    return 0 if done.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
