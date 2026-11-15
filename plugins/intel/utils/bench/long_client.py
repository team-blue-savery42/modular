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

"""Long-prompt serving client (standard library only).

Sends streaming /v1/completions requests with prompts of exact token counts
(token-id lists drawn from a fixed id range, so no tokenizer is needed) and
reports time to first token, decode rate and failures per request as JSON.

Modes:
  long    N requests of LEN tokens each at concurrency C (distinct prompts, so
          no prefix hits); TTFT and decode tok/s per request.
  prefix  C agents share one PREFIX-token context. Agent i's turn t sends
          prefix + the first t+1 of its own NEW-token messages. The first
          request (agent 0, turn 0) runs alone and is cold; then all agents
          run their turns 0..TURNS-1 concurrently, each turn after the
          previous reply. Reports TTFT by turn (turn 0 of agents > 0 and every
          later turn should process only the new tokens when prefix caching
          works) and the server's cached-token count when it reports one.

Usage: long_client.py URL MODEL long   --len L [--n N] [--c C] [--out T]
       long_client.py URL MODEL prefix --prefix P [--new K] [--c C] [--turns T] [--out T]
"""

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.request

SYL = ["ba", "ke", "mi", "so", "tu", "la", "re", "ni", "po", "du", "fa", "gi"]
CHAT = False


def words(rng, n):
    """n pseudo-words (two or three syllables) as one string."""
    return " ".join(
        "".join(rng.choice(SYL) for _ in range(rng.choice((2, 3))))
        for _ in range(n)
    )


def request(url, model, prompt, out_tokens, timeout):
    """One streaming request; `prompt` is a token-id list, or text with --chat
    (the chat endpoint, for models whose reasoning parser hides the completions
    endpoint's tokens: Qwen3.5 on MAX)."""
    if CHAT:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": out_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        path = "/v1/chat/completions"
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": out_tokens,
            "min_tokens": out_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
        }
        path = "/v1/completions"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url + path,
        body,
        {"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = last = None
    n = 0
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                d = line[5:].strip()
                if d == b"[DONE]":
                    break
                j = json.loads(d)
                if j.get("usage"):
                    usage = j["usage"]
                for c in j.get("choices", []):
                    if CHAT:
                        d = c.get("delta") or {}
                        t = (
                            d.get("content")
                            or d.get("reasoning")
                            or d.get("reasoning_content")
                        )
                        if t:
                            now = time.perf_counter()
                            if first is None:
                                first = now
                            last = now
                            n += 1
                        continue
                    t = c.get("text")
                    # A token whose text is empty (special tokens, a reasoning
                    # marker) is still a token: count every chunk that isn't
                    # the closing one.
                    if t or (t is not None and c.get("finish_reason") is None):
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        last = now
                        n += 1
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if first is None:
        return {"error": "no tokens", "usage": usage}
    res = {
        "ttft": first - t0,
        "tokens": n,
        "decode_tps": (n - 1) / (last - first)
        if n > 1 and last > first
        else None,
        "usage": usage,
    }
    return res


def cached(usage):
    if not usage:
        return None
    d = usage.get("prompt_tokens_details") or {}
    return d.get("cached_tokens")


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def summarize(rs):
    ok = [r for r in rs if "error" not in r]
    out = {"requests": len(rs), "failed": len(rs) - len(ok)}
    out["errors"] = sorted({r["error"] for r in rs if "error" in r})
    if ok:
        t = [r["ttft"] for r in ok]
        out["ttft_median_s"] = round(statistics.median(t), 3)
        out["ttft_p99_s"] = round(pct(t, 99), 3)
        out["ttft_min_s"] = round(min(t), 3)
        d = [r["decode_tps"] for r in ok if r["decode_tps"]]
        if d:
            out["decode_tps_median"] = round(statistics.median(d), 1)
        c = [cached(r["usage"]) for r in ok if cached(r["usage"]) is not None]
        if c:
            out["cached_tokens_median"] = statistics.median(c)
        p = [r["usage"]["prompt_tokens"] for r in ok if r.get("usage")]
        if p:
            out["prompt_tokens_median"] = statistics.median(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("mode", choices=["long", "prefix"])
    ap.add_argument("--len", type=int, default=8192)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--c", type=int, default=1)
    ap.add_argument("--out", type=int, default=64)
    ap.add_argument("--prefix", type=int, default=24576)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--chat",
        action="store_true",
        help="text prompts through /v1/chat/completions; lengths are"
        " calibrated against the server's prompt_tokens",
    )
    a = ap.parse_args()
    global CHAT
    CHAT = a.chat
    rng = random.Random(a.seed)
    if CHAT:
        # Tokens per pseudo-word and the template's fixed tokens, from two
        # short requests' prompt_tokens.
        r1 = request(a.url, a.model, words(random.Random(99), 1), 1, a.timeout)
        r2 = request(
            a.url, a.model, words(random.Random(98), 400), 1, a.timeout
        )
        t1, t2 = r1["usage"]["prompt_tokens"], r2["usage"]["prompt_tokens"]
        ratio = (t2 - t1) / 399
        calib = {"fixed_tokens": t1, "tokens_per_word": round(ratio, 3)}

        def ids(r, n, fixed=True):
            return words(r, max(1, round((n - (t1 if fixed else 0)) / ratio)))

    else:
        calib = None

        def ids(r, n, fixed=True):
            return [r.randrange(1000, 100000) for _ in range(n)]

    res = {"mode": a.mode, "calibration": calib}
    if a.mode == "long":
        prompts = [ids(rng, a.len) for _ in range(a.n)]
        results = [None] * a.n
        sem = threading.Semaphore(a.c)

        def one(i):
            with sem:
                results[i] = request(
                    a.url, a.model, prompts[i], a.out, a.timeout
                )

        ts = [threading.Thread(target=one, args=(i,)) for i in range(a.n)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        res.update(len=a.len, c=a.c, wall_s=round(time.perf_counter() - t0, 2))
        res["summary"] = summarize(results)
        res["each"] = results
    else:
        prefix = ids(rng, a.prefix)
        msgs = [
            [ids(rng, a.new, False) for _ in range(a.turns)] for _ in range(a.c)
        ]

        def prompt(agent, turn):
            if CHAT:
                return " ".join([prefix] + msgs[agent][: turn + 1])
            p = list(prefix)
            for m in msgs[agent][: turn + 1]:
                p += m
            return p

        cold = request(a.url, a.model, prompt(0, 0), a.out, a.timeout)
        res["cold"] = summarize([cold])
        res["cold_each"] = cold
        by_turn = [[] for _ in range(a.turns)]

        def agent_run(i):
            for t in range(a.turns):
                if i == 0 and t == 0:
                    continue  # the cold request
                by_turn[t].append(
                    request(a.url, a.model, prompt(i, t), a.out, a.timeout)
                )

        ts = [threading.Thread(target=agent_run, args=(i,)) for i in range(a.c)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        res.update(
            prefix=a.prefix,
            new=a.new,
            c=a.c,
            wall_s=round(time.perf_counter() - t0, 2),
        )
        res["turns"] = {
            str(t): summarize(rs) for t, rs in enumerate(by_turn) if rs
        }
        res["turns_each"] = {str(t): rs for t, rs in enumerate(by_turn)}
    json.dump(res, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
