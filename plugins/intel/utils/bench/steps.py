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

"""Per-step kernel breakdown from the driver plugin's launch profile.

Reads a MOJO_INTEL_PROFILE file (format in plugins/intel/driver/Profile.h)
and the step marks `bench_model.py --marks` writes (one line per step:
`<rep> <step> <kind> <enter ns> <exit ns>`, CLOCK_MONOTONIC like the
profile's host times). For each selected step: the kernels, copies and
fills issued in it with their device time, the host time between launches,
the device's busy and idle time, and totals; then a table of kernels by name
summed over the steps (median per step).

Usage: python3 steps.py PROFILE MARKS [--kind decode|prefill] [--top 40]
         [--detail N] [--csv OUT]
Steps during which the plugin harvested timestamps (an `H` record) are left
out of the statistics, as the harvest delays their launches.
"""

import argparse
import bisect
import collections
import statistics
import sys


def short_name(name):
    """A readable kernel name: the function's path without its parameters."""
    cut = len(name)
    for ch in "([":
        i = name.find(ch)
        if i >= 0:
            cut = min(cut, i)
    base = name[:cut]
    parts = base.split("::")
    tail = "::".join(parts[-2:]) if len(parts) > 1 else base
    return tail[:90]


def load(path):
    hz = None
    names = {}
    kernels, copies, host, allocs, harvests, stamps = [], [], [], [], [], []
    captured = {}
    with open(path) as f:
        for line in f:
            k = line[:1]
            if k == "K":
                p = line.split()
                kernels.append(
                    (
                        int(p[9]),
                        int(p[10]),
                        int(p[11]),
                        int(p[12]),
                        int(p[2]),
                        int(p[1]),
                        tuple(map(int, p[3:6])),
                        tuple(map(int, p[6:9])),
                    )
                )
            elif k == "C":
                p = line.split()
                copies.append(
                    (
                        int(p[4]),
                        int(p[5]),
                        int(p[6]),
                        int(p[7]),
                        p[2],
                        int(p[3]),
                        int(p[1]),
                    )
                )
            elif k in "SEW":
                p = line.split()
                host.append((int(p[2]), int(p[3]), k, int(p[1])))
            elif k in "AR":
                p = line.split()
                allocs.append((int(p[3]), int(p[4]), k, int(p[1]), p[2]))
            elif k == "H":
                p = line.split()
                harvests.append((int(p[1]), int(p[2])))
            elif k == "T":
                p = line.split()
                stamps.append((int(p[1]), int(p[2])))
            elif k == "G":
                # A device graph launch: one device interval for
                # all its captured kernels (`k` records).
                p = line.split()
                gid = 1_000_000 + int(p[2])
                kernels.append(
                    (
                        int(p[3]),
                        int(p[4]),
                        int(p[5]),
                        int(p[6]),
                        gid,
                        int(p[1]),
                        (0, 0, 0),
                        (0, 0, 0),
                    )
                )
            elif k == "k":
                p = line.split()
                captured[int(p[2])] = captured.get(int(p[2]), 0) + 1
            elif k == "F":
                p = line.rstrip("\n").split(" ", 2)
                names[int(p[1])] = p[2] if len(p) > 2 else ""
            elif k == "P":
                p = line.split()
                hz, _valid, kvalid = int(p[1]), int(p[2]), int(p[3])
    for g, n in captured.items():
        names[1_000_000 + g] = f"device graph {g} ({n} kernels)"
    kernels.sort()
    copies.sort()
    host.sort()
    allocs.sort()
    return dict(
        hz=hz,
        kvalid=kvalid,
        names=names,
        kernels=kernels,
        copies=copies,
        host=host,
        allocs=allocs,
        harvests=harvests,
        stamps=stamps,
    )


def device_mapper(prof):
    """Maps raw kernel ticks to host ns, using the nearest T record (device
    global timestamps, full width) to restore the bits the kernel timestamps
    drop."""
    hz = prof["hz"]
    mask = (1 << prof["kvalid"]) - 1 if prof["kvalid"] < 64 else (1 << 64) - 1
    stamps = prof["stamps"]
    hosts = [h for h, _ in stamps]

    def to_host(ticks, near_host):
        i = bisect.bisect_left(hosts, near_host)
        cands = [j for j in (i - 1, i) if 0 <= j < len(stamps)]
        j = min(cands, key=lambda j: abs(hosts[j] - near_host))
        h0, d0 = stamps[j]
        # The device time near `near_host`, then the tick value with the
        # kernel stamp's low bits closest to it.
        d_near = d0 + (near_host - h0) * hz / 1e9
        base = int(d_near) & ~mask
        best = None
        for b in (base - (mask + 1), base, base + mask + 1):
            v = b + (ticks & mask)
            if best is None or abs(v - d_near) < abs(best - d_near):
                best = v
        return h0 + (best - d0) * 1e9 / hz

    return to_host


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("marks")
    ap.add_argument("--kind", default="decode")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument(
        "--detail",
        type=int,
        default=0,
        help="print the first N steps launch by launch",
    )
    ap.add_argument("--csv")
    ap.add_argument(
        "--keep-harvests",
        action="store_true",
        help="keep steps a harvest overlapped (long prefill steps"
        " and graph replays always overlap one)",
    )
    a = ap.parse_args()

    prof = load(a.profile)
    hz = prof["hz"]
    names = prof["names"]
    to_host = device_mapper(prof)
    marks = []
    with open(a.marks) as f:
        for line in f:
            p = line.split()
            # The warm-up repetition is left out, as bench_model.py's own
            # medians leave it out.
            if len(p) >= 5 and p[2] == a.kind and p[0] != "warm-up":
                marks.append((p[0], int(p[1]), int(p[3]), int(p[4])))
    kern = prof["kernels"]
    kstarts = [k[0] for k in kern]
    harvests = prof["harvests"]

    per_step = []
    by_name = collections.defaultdict(list)
    detail_left = a.detail
    for rep, step, t0, t1 in marks:
        if not a.keep_harvests and any(
            h0 < t1 and h1 > t0 for h0, h1 in harvests
        ):
            continue
        lo = bisect.bisect_left(kstarts, t0)
        hi = bisect.bisect_left(kstarts, t1)
        ks = kern[lo:hi]
        cs = [c for c in prof["copies"] if t0 <= c[0] < t1]
        ss = [h for h in prof["host"] if t0 <= h[0] < t1]
        al = [x for x in prof["allocs"] if t0 <= x[0] < t1]
        if not ks:
            continue
        dev = []  # (host-mapped start, end, label, device us)
        for k in ks:
            enter, exit_, s, e, fid = k[:5]
            if s < 0:
                continue
            us = ((e - s) & ((1 << prof["kvalid"]) - 1)) * 1e6 / hz
            hs = to_host(s, exit_)
            dev.append(
                (
                    hs,
                    hs + us * 1e3,
                    short_name(names.get(fid, "?")),
                    us,
                    enter,
                    exit_,
                )
            )
        for c in cs:
            enter, exit_, s, e, sub, nbytes = c[:6]
            if s < 0:
                continue
            us = ((e - s) & ((1 << prof["kvalid"]) - 1)) * 1e6 / hz
            hs = to_host(s, exit_)
            dev.append(
                (
                    hs,
                    hs + us * 1e3,
                    f"copy/fill {sub} {nbytes}B",
                    us,
                    enter,
                    exit_,
                )
            )
        dev.sort()
        busy = 0.0
        cur_s = cur_e = None
        for s, e, *_ in dev:
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    busy += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            busy += cur_e - cur_s
        dev_sum = sum(d[3] for d in dev)
        launch_host = sum(k[1] - k[0] for k in ks)
        gaps = [ks[i + 1][0] - ks[i][1] for i in range(len(ks) - 1)]
        first_start = dev[0][0] if dev else t0
        last_end = max(d[1] for d in dev) if dev else t0
        # Device idle while work was queued: gaps between consecutive device
        # intervals in the step.
        idle = 0.0
        prev_end = None
        for s, e, *_ in dev:
            if prev_end is not None and s > prev_end:
                idle += s - prev_end
            prev_end = e if prev_end is None else max(prev_end, e)
        # Launches the device waited for: the device started a kernel within
        # 2 us of its launch call returning, i.e. it was starved.
        starved = sum(1 for d in dev if d[0] - d[5] < 2000)
        row = dict(
            rep=rep,
            step=step,
            wall=(t1 - t0) / 1e3,
            n=len(ks),
            ncopy=len(cs),
            nsync=sum(1 for h in ss if h[2] == "S"),
            nalloc=sum(1 for x in al if x[2] == "A"),
            nfree=sum(1 for x in al if x[2] == "R"),
            alloc_us=sum(x[1] - x[0] for x in al) / 1e3,
            launch_host=launch_host / 1e3,
            gap_med=statistics.median(gaps) / 1e3 if gaps else 0,
            gap_sum=sum(gaps) / 1e3,
            dev_sum=dev_sum,
            dev_busy=busy / 1e3,
            dev_idle=idle / 1e3,
            first_launch=(ks[0][0] - t0) / 1e3,
            dev_first=(first_start - t0) / 1e3,
            dev_tail=(t1 - last_end) / 1e3,
            starved=starved,
            issue_span=(ks[-1][1] - ks[0][0]) / 1e3,
        )
        per_step.append(row)
        counts = collections.Counter()
        sums = collections.defaultdict(float)
        for d in dev:
            counts[d[2]] += 1
            sums[d[2]] += d[3]
        for name in counts:
            by_name[name].append((counts[name], sums[name]))
        if detail_left > 0:
            detail_left -= 1
            print(f"--- step {rep}/{step}: wall {(t1 - t0) / 1e3:.1f} us")
            for i, d in enumerate(dev):
                print(
                    f"{i:4d} launch@{(d[4] - t0) / 1e3:8.1f} "
                    f"call {(d[5] - d[4]) / 1e3:5.1f} "
                    f"dev@{(d[0] - t0) / 1e3:8.1f} {d[3]:8.2f} us  {d[2]}"
                )

    if not per_step:
        print("no steps", file=sys.stderr)
        return 1
    n = len(per_step)
    print(f"{n} {a.kind} steps (of {len(marks)}; others overlapped a harvest)")

    def med(key):
        return statistics.median(r[key] for r in per_step)

    for key, label in [
        ("wall", "step wall time (us)"),
        ("n", "kernel launches"),
        ("ncopy", "copies/fills on the queue"),
        ("nsync", "queue synchronizes"),
        ("nalloc", "allocations"),
        ("nfree", "frees"),
        ("alloc_us", "host time in alloc/free (us)"),
        ("first_launch", "step start -> first launch (us)"),
        ("issue_span", "first launch -> last launch returns (us)"),
        ("launch_host", "host time inside launch calls (us)"),
        ("gap_sum", "host time between launch calls (us)"),
        ("gap_med", "median gap between launch calls (us)"),
        ("dev_sum", "sum of device times (us)"),
        ("dev_busy", "device busy (union, us)"),
        ("dev_idle", "device idle between first and last op (us)"),
        ("dev_first", "step start -> device starts (us)"),
        ("dev_tail", "device done -> step returns (us)"),
        ("starved", "ops started < 2 us after their launch returned"),
    ]:
        print(f"  {label:48s} {med(key):10.1f}")
    print(f"\nkernels by name (median per step over {n} steps; device us)")
    rows = []
    for name, vals in by_name.items():
        cnt = statistics.median(v[0] for v in vals)
        tot = statistics.median(v[1] for v in vals)
        rows.append((tot, cnt, name, len(vals)))
    rows.sort(reverse=True)
    total = med("dev_sum")
    print(
        f"  {'device us':>10s} {'share':>6s} {'count':>6s} {'us each':>8s}  name"
    )
    for tot, cnt, name, seen in rows[: a.top]:
        print(
            f"  {tot:10.1f} {tot / total * 100:5.1f}% {cnt:6.0f} "
            f"{tot / cnt if cnt else 0:8.2f}  {name}"
            + ("" if seen == n else f"  (in {seen} steps)")
        )
    if a.csv:
        with open(a.csv, "w") as f:
            f.write("device_us,count,name\n")
            for tot, cnt, name, _ in rows:
                f.write(f"{tot:.2f},{cnt},{name!r}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
