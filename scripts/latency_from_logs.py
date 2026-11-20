"""Estimate per-study inference time from the worker's log (no code changes needed).

    docker compose logs --no-log-prefix inference > inference.log
    python scripts/latency_from_logs.py inference.log

When many studies are uploaded at once the worker analyses them back to back, so the gap between two consecutive
"analysed ..." lines approximates the time per study (Orthanc fetch + model + write-back). Gaps longer than --max-gap
seconds (idle time between uploads) are ignored. Results are reported per modality.
"""
import argparse
import re
import statistics
import sys
from datetime import datetime

LINE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}).*analysed \S+ \((\w+)\)")


def parse(lines):
    out = []
    for ln in lines:
        m = LINE.search(ln)
        if m:
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() + int(m.group(2)) / 1000
            out.append((t, m.group(3)))
    return sorted(out)


def gaps(events, max_gap):
    res = {}
    for (t0, _), (t1, mod) in zip(events, events[1:]):
        if 0 < t1 - t0 <= max_gap:
            res.setdefault(mod, []).append(t1 - t0)
    return res


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", help="log file (default: stdin)")
    ap.add_argument("--max-gap", type=float, default=300.0, help="ignore gaps longer than this many seconds")
    args = ap.parse_args()
    lines = open(args.log, encoding="utf-8", errors="ignore").read().splitlines() if args.log else sys.stdin.read().splitlines()
    events = parse(lines)
    print(f"{len(events)} 'analysed' lines found")
    g = gaps(events, args.max_gap)
    if not g:
        print("Not enough back-to-back studies to estimate (need at least two within --max-gap seconds).")
        return
    print(f"{'modality':8s} {'n':>4s} {'median s':>9s} {'mean s':>8s} {'p95 s':>7s} {'max s':>7s}")
    for mod, v in sorted(g.items()):
        print(f"{mod:8s} {len(v):4d} {statistics.median(v):9.1f} {statistics.mean(v):8.1f} {pct(v, 95):7.1f} {max(v):7.1f}")
    print("(modality is that of the later study in each pair)")


if __name__ == "__main__":
    main()
