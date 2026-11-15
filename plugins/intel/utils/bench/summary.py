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

"""Tabulates a serving run's `vllm bench serve` logs (serve.sh):
one row per workload and concurrency with successful / failed requests,
output tok/s, median and P99 TTFT, median and P99 ITL (ms).
Usage: python3 summary.py DIR [DIR...]"""

import glob
import os
import re
import sys

FIELDS = [
    ("ok", r"Successful requests:\s+(\d+)"),
    ("fail", r"Failed requests:\s+(\d+)"),
    ("out tok/s", r"Output token throughput \(tok/s\):\s+([\d.]+)"),
    ("TTFT p50", r"Median TTFT \(ms\):\s+([\d.]+)"),
    ("TTFT p99", r"P99 TTFT \(ms\):\s+([\d.]+)"),
    ("ITL p50", r"Median ITL \(ms\):\s+([\d.]+)"),
    ("ITL p99", r"P99 ITL \(ms\):\s+([\d.]+)"),
]
for d in sys.argv[1:]:
    print(f"== {d}")
    print("| workload | c | " + " | ".join(f for f, _ in FIELDS) + " |")
    rows = []
    for f in glob.glob(os.path.join(d, "bench-*-c*.log")):
        m = re.search(r"bench-(\w+)-c(\d+)\.log", f)
        text = open(f, errors="replace").read()
        vals = []
        for _, rx in FIELDS:
            mm = re.search(rx, text)
            vals.append(mm.group(1) if mm else "-")
        rows.append((m.group(1), int(m.group(2)), vals))
    for w, c, vals in sorted(rows):
        print(f"| {w} | {c} | " + " | ".join(vals) + " |")
