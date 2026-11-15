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

"""Repeats greedy generation in one process and compares the token ids.

A reproducer for nondeterministic model output on the B70: builds the
pipeline once, the way `pipelines generate` does, then runs every prompt
REPEATS times for NEW_TOKENS tokens with top-k 1, and reports each run whose
token ids differ from that prompt's first run (first differing position,
both texts).

Usage :
  utils/run-max-py tests/graph/gen_repeat.py MODEL_PATH [REPEATS] [NEW_TOKENS]
Environment: GEN_PROMPTS (a file, prompts separated by lines of `---`),
GEN_JSON (write every run's token ids there, and each prompt's token ids
after MAX's chat template as `prompt<i>`), GEN_MAX_LENGTH (default 2048),
GEN_FIRST_ONLY=1 (one run per prompt, for many-process loops), GEN_ENCODING
(--quantization-encoding, e.g. float8_e4m3fn), GEN_ARGS (more `pipelines
generate` options, split on spaces), GEN_GRAPH_CAPTURE=1 (device graphs:
`--device-graph-capture` and `warmup_graph_capture()` before the prompts).
Exit code 0 when every run matched, 1 otherwise.
"""

import asyncio
import json
import os
import sys
import time

PROMPTS = [
    "The capital of France is",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain, step by step, how a CPU executes a branch instruction.",
    "List ten prime numbers and say why each is prime.",
]


def main() -> int:
    model = sys.argv[1]
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    new_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    prompts = PROMPTS
    if os.environ.get("GEN_PROMPTS"):
        with open(os.environ["GEN_PROMPTS"]) as f:
            prompts = [
                p.strip() for p in f.read().split("\n---\n") if p.strip()
            ]
    if os.environ.get("GEN_FIRST_ONLY") == "1":
        repeats = 1

    from max._entrypoints import pipelines as cli
    from max._entrypoints.cli import generate as gen

    results: dict = {}
    prompt_ids: dict = {}

    def run(pipeline_args, sampling_params, prompt, **_ignored):
        from max.pipelines import PIPELINE_REGISTRY
        from max.pipelines.lib import PipelineConfig

        config = PipelineConfig.from_args(
            gen._with_debug_defaults(pipeline_args)
        )
        retrieved = PIPELINE_REGISTRY.retrieve_factory(config)
        tokenizer = retrieved.tokenizer
        pipeline = retrieved.factory()
        if os.environ.get("GEN_GRAPH_CAPTURE") == "1":
            # Capture the decode graphs before the prompts, as
            # `pipelines serve` does at startup.
            pipeline.warmup_graph_capture()

        async def one(text):
            from max.pipelines.modeling.types import (
                RequestID,
                TextContentPart,
                TextGenerationRequest,
                TextGenerationRequestMessage,
            )

            request = TextGenerationRequest(
                request_id=RequestID(),
                messages=[
                    TextGenerationRequestMessage(
                        role="user", content=[TextContentPart(text=text)]
                    )
                ],
                model_name=gen.MODEL_NAME,
                sampling_params=sampling_params,
            )
            context = await pipeline.tokenizer.new_context(request)
            prompt_ids[text] = [int(t) for t in context.tokens.prompt]
            out: list[int] = []

            async def collect(tokens):
                out.extend(int(t) for t in tokens)

            await gen.stream_context(pipeline, context, collect)
            return out

        async def decode(ids):
            return await tokenizer.decode(ids, skip_special_tokens=True)

        bad = 0
        total = 0
        t0 = time.time()
        for r in range(repeats):
            for i, text in enumerate(prompts):
                ids = asyncio.run(one(text))
                total += 1
                ref = results.setdefault(i, ids if r == 0 else results.get(i))
                runs = results.setdefault(f"runs{i}", [])
                runs.append(ids)
                if r == 0:
                    print(
                        f"prompt {i}: {len(ids)} tokens: "
                        f"{asyncio.run(decode(ids))[:160]!r}",
                        flush=True,
                    )
                    continue
                if ids != ref:
                    bad += 1
                    pos = next(
                        (
                            k
                            for k, (a, b) in enumerate(
                                zip(ids, ref, strict=False)
                            )
                            if a != b
                        ),
                        min(len(ids), len(ref)),
                    )
                    print(
                        f"MISMATCH run {r} prompt {i} at token {pos} "
                        f"(len {len(ids)} vs {len(ref)}): "
                        f"{asyncio.run(decode(ids))[:400]!r}",
                        flush=True,
                    )
            if r % 5 == 4:
                print(
                    f"after {r + 1} repeats: {bad} mismatches of {total} "
                    f"({time.time() - t0:.0f} s)",
                    flush=True,
                )
        print(
            f"RESULT {bad} mismatches in {total} runs "
            f"({repeats} repeats x {len(prompts)} prompts, {new_tokens} tokens)",
            flush=True,
        )
        if os.environ.get("GEN_JSON"):
            for i, text in enumerate(prompts):
                results[f"prompt{i}"] = prompt_ids.get(text, [])
                results[f"text{i}"] = text
            with open(os.environ["GEN_JSON"], "w") as f:
                json.dump({str(k): v for k, v in results.items()}, f)
        results["bad"] = bad

    gen.generate_text_for_pipeline = run
    argv = [
        "pipelines",
        "generate",
        "--model-path",
        model,
        "--devices",
        "gpu",
        "--max-length",
        os.environ.get("GEN_MAX_LENGTH", "2048"),
        "--max-new-tokens",
        str(new_tokens),
        "--max-batch-size",
        "1",
        "--top-k",
        "1",
        "--ignore-eos",
        "--prompt",
        "unused",
    ]
    if os.environ.get("GEN_ENCODING"):
        argv += ["--quantization-encoding", os.environ["GEN_ENCODING"]]
    argv += os.environ.get("GEN_ARGS", "").split()
    if os.environ.get("GEN_GRAPH_CAPTURE") == "1":
        argv += ["--device-graph-capture"]
    sys.argv = argv
    try:
        cli.main(standalone_mode=False)
    except SystemExit:
        pass
    return 1 if results.get("bad", 1) else 0


if __name__ == "__main__":
    sys.exit(main())
