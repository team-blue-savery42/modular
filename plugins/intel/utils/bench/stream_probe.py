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

"""When a server's streamed completion chunks arrive.

Sends a streamed /v1/completions request (a fixed prompt, 64 tokens,
greedy, ignore EOS) twice, without and with `stream_options.include_usage`
(which `vllm bench serve` sets), and prints each chunk's arrival time after
the request was sent, then the time to the first token and the mean gap
between later chunks. Usage: python3 stream_probe.py BASE_URL MODEL
"""

import itertools
import json
import sys
import time
import urllib.request


def probe(base: str, model: str, usage: bool) -> None:
    req_body = {
        "model": model,
        "prompt": "Explain how a CPU executes a branch instruction.",
        "max_tokens": 64,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
    }
    if usage:
        req_body["stream_options"] = {"include_usage": True}
    body = json.dumps(req_body).encode()
    req = urllib.request.Request(
        base + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    times = []
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            text = "".join(c.get("text", "") for c in chunk.get("choices", []))
            if text:
                times.append(time.perf_counter() - t0)
                print(f"{times[-1] * 1e3:9.2f} ms {text!r}", flush=True)
    gaps = [b - a for a, b in itertools.pairwise(times)]
    print(
        f"RESULT include_usage {usage}: chunks {len(times)}, first {times[0] * 1e3:.2f} ms, mean gap "
        f"{sum(gaps) / max(len(gaps), 1) * 1e3:.2f} ms, last {times[-1] * 1e3:.2f} ms"
    )


def main() -> None:
    for usage in (False, True):
        probe(sys.argv[1], sys.argv[2], usage)


if __name__ == "__main__":
    main()
