#!/usr/bin/env python3
"""The KPI that matters: how much of our draft the native reviewers had to change, per batch, over time.

Judges rate our output above the human's on most lines and endorse only ~19% of reviewer edits, so
judge scores can't be the measure of quality. The reviewers' edits can: from review_pairs.py rows,
per batch and video — the share of lines edited and chrF(draft, final) (100 = untouched).

  PYTHONPATH=src python3 scripts/eval/vendor_edit_kpi.py $DEV_TITLES_OUTPUT/judge_calib/zh-Hans.jsonl [--out <md>]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from ab_vs_human import chrf  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rows = [
        json.loads(x) for p in a.pairs for x in Path(p).read_text(encoding="utf-8").splitlines()
    ]
    by = defaultdict(list)
    for r in rows:
        by[(r["batch"], r["video"])].append(r)
    rep = [
        "# Vendor edit KPI — share of our lines the reviewers changed",
        "",
        "| batch | video | lines | edited | chrF draft→final |",
        "|---|---|---|---|---|",
    ]
    tot = defaultdict(lambda: [0, 0, 0.0])
    for (b, v), rs in sorted(by.items()):
        n, e = len(rs), sum(r["edited"] for r in rs)
        c = sum(chrf(r["draft"], r["human"]) for r in rs) / n
        rep.append(f"| {b} | {v} | {n} | {e / n:.0%} | {c:.1f} |")
        t = tot[b]
        t[0] += n
        t[1] += e
        t[2] += c * n
    rep += ["", "| batch | lines | edited | chrF |", "|---|---|---|---|"]
    for b, (n, e, c) in sorted(tot.items()):
        rep.append(f"| **{b}** | {n} | **{e / n:.0%}** | {c / n:.1f} |")
    text = "\n".join(rep) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
