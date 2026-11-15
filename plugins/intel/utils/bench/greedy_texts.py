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

"""Fixed greedy completions from a running server, for
comparing a long run's start and end. Sends four fixed prompts (64 tokens,
temperature 0, not streamed) and prints one JSON line per prompt.
Usage: python3 greedy_texts.py BASE_URL MODEL"""

import json
import sys
import urllib.request

PROMPTS = [
    "The capital of France is",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain, step by step, how a CPU executes a branch instruction.",
    "List ten prime numbers and say why each is prime.",
]


def main() -> int:
    base, model = sys.argv[1], sys.argv[2]
    for p in PROMPTS:
        body = json.dumps(
            {
                "model": model,
                "prompt": p,
                "max_tokens": 64,
                "temperature": 0,
                "ignore_eos": True,
            }
        ).encode()
        req = urllib.request.Request(
            base + "/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            text = json.load(resp)["choices"][0]["text"]
        print(json.dumps({"prompt": p, "text": text}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
