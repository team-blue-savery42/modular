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

"""MAX's greedy token ids on the B70 against transformers on the CPU.

Usage (a venv with torch, transformers and safetensors; no GPU):
  VENV/bin/python -I tests/graph/hf_match.py \
      MODEL_DIR MAX_JSON [DTYPE ...]
MAX_JSON is gen_repeat.py's GEN_JSON output: for each prompt `prompt<i>`,
the token ids after MAX's chat template, and `<i>`, MAX's greedy output
ids. The same prompt ids go to transformers (so the template and thinking
setting are MAX's by construction). For each dtype (default float32 and
bfloat16): transformers' own greedy ids for as many tokens, the first
position where they differ from MAX's, and, teacher-forcing MAX's ids, the
first position where MAX's token isn't transformers' argmax, with the
logit gap there (transformers' best logit minus that of MAX's token) and
the gap between transformers' two best logits.

HF_FAKE_FP8=<dir>: after loading MODEL_DIR, every module that the FP8
checkpoint in <dir> quantizes gets that checkpoint's weight, dequantized
exactly (float8_e4m3fn code times weight_scale, in the run's dtype): the
reference for MAX's weight-only FP8 path (activations stay in the run's
dtype, as on Intel). It also prints, per
prompt, the mean negative log-likelihood of MAX's tokens under the model.
"""

import json
import os
import sys

import torch
from transformers import AutoModelForImageTextToText


def fake_fp8(model, fp8_dir: str, dtype) -> None:
    """Loads the FP8 checkpoint's quantized weights, dequantized, into
    `model` (module names as in the checkpoint)."""
    from safetensors import safe_open

    quant = json.load(open(os.path.join(fp8_dir, "hf_quant_config.json")))
    modules = set(quant["quantization"]["quantized_layers"])
    index = json.load(
        open(os.path.join(fp8_dir, "model.safetensors.index.json"))
    )
    named = dict(model.named_modules())
    done = 0
    for fname in sorted(set(index["weight_map"].values())):
        with safe_open(os.path.join(fp8_dir, fname), "pt") as f:
            for mod in modules:
                if mod + ".weight" not in f:
                    continue
                w = (
                    f.get_tensor(mod + ".weight").float()
                    * f.get_tensor(mod + ".weight_scale").float()
                )
                with torch.no_grad():
                    named[mod].weight.copy_(w.to(dtype))
                done += 1
    assert done == len(modules), (done, len(modules))
    print(f"fake FP8: {done} modules from {fp8_dir}", flush=True)


def main() -> None:
    model_dir, path = sys.argv[1], sys.argv[2]
    dtypes = sys.argv[3:] or ["float32", "bfloat16"]
    data = json.load(open(path))
    prompts = sorted(int(k[6:]) for k in data if k.startswith("prompt"))
    for name in dtypes:
        dtype = getattr(torch, name)
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=dtype
        )
        model.eval()
        if os.environ.get("HF_FAKE_FP8"):
            fake_fp8(model, os.environ["HF_FAKE_FP8"], dtype)
        for i in prompts:
            prompt = data[f"prompt{i}"]
            mx = data[str(i)]
            n = len(mx)
            ids = torch.tensor([prompt])
            with torch.no_grad():
                out = model.generate(
                    ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=n,
                    do_sample=False,
                    min_new_tokens=n,
                )
            hf = out[0, len(prompt) :].tolist()
            first = next(
                (
                    k
                    for k, (a, b) in enumerate(zip(hf, mx, strict=False))
                    if a != b
                ),
                None,
            )
            # Teacher-forced: transformers' logits after the prompt and MAX's
            # first k tokens, for every k at once.
            full = torch.tensor([prompt + mx])
            with torch.no_grad():
                logits = model(full).logits[0, len(prompt) - 1 : -1].float()
            top2 = logits.topk(2, dim=-1)
            argmax = top2.indices[:, 0].tolist()
            forced = next(
                (
                    k
                    for k, (a, b) in enumerate(zip(argmax, mx, strict=False))
                    if a != b
                ),
                None,
            )
            line = (
                f"{name} prompt {i} ({len(prompt)} prompt tokens, {n} generated): "
                f"free-running {'MATCH' if first is None else f'first difference at {first}'}"
            )
            if forced is None:
                line += (
                    "; teacher-forced: every MAX token is transformers' argmax"
                )
            else:
                gap = float(
                    logits[forced, argmax[forced]] - logits[forced, mx[forced]]
                )
                top_gap = float(top2.values[forced, 0] - top2.values[forced, 1])
                line += (
                    f"; teacher-forced: first non-argmax at {forced} (MAX {mx[forced]}, "
                    f"transformers {argmax[forced]}, logit gap {gap:.4f}, "
                    f"transformers' top-2 gap {top_gap:.4f}, "
                    f"MAX's token is transformers' #{int((logits[forced] > logits[forced, mx[forced]]).sum()) + 1})"
                )
            nll = torch.nn.functional.cross_entropy(logits, torch.tensor(mx))
            line += f"; mean NLL of MAX's tokens {float(nll):.4f}"
            worst = []
            for k, t in enumerate(mx):
                g = float(logits[k, argmax[k]] - logits[k, t])
                if g > 0:
                    worst.append((g, k))
            if worst:
                g, k = max(worst)
                line += f"; non-argmax MAX tokens {len(worst)}, largest gap {g:.4f} at {k}"
            print(line, flush=True)
            print(f"  max {mx}", flush=True)
            print(f"  hf  {hf}", flush=True)


if __name__ == "__main__":
    main()
