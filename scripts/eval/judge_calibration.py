#!/usr/bin/env python3
"""How often do our judges prefer what the native reviewers chose? Measured on real reviewer edits.

Rows come from review_pairs.py: lines the reviewer EDITED, as (draft they saw, their final). For each, both
judges (Claude + GPT, random A/B order, the final not marked) pick the better subtitle. A judge setup is
good when it prefers the reviewer's final — about 95% of vendor edits are right (zh vendor check) — and
rarely prefers the draft. On one reviewed batch the A/B judge prompt agreed (both judges) on 19%.

Two splits, so a prompt is never scored on the examples it was shown: dev = --dev-batch (examples are
drawn from it), test = --test-batch (held out). Batch names are the `batch` field of the pairs file.
Setups:
  current  the A/B judge prompt (scripts/eval/ab_vs_human.JUDGE)
  house    + the house style notes (glossary zh notes) + reviewer before/after examples (dev only)

  PYTHONPATH=src python3 scripts/eval/judge_calibration.py --pairs $DEV_TITLES_OUTPUT/judge_calib/zh-Hans.jsonl \
      --dev-batch <batch A> --test-batch <batch B> --out $DEV_TITLES_OUTPUT/judge_calib/results \
      [--n 120] [--setups current house]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from ab_vs_human import CATS, JUDGE  # noqa: E402

from dev_titles.cli import _qe_second_judge  # noqa: E402
from dev_titles.glossary import load  # noqa: E402
from dev_titles.translate import CliTranslator  # noqa: E402

LANG = {"zh-Hans": "Simplified Chinese", "ja": "Japanese"}

HOUSE = """You are a senior native {L} subtitle reviewer for a developer YouTube channel (interviews and
talks with developers). Two {L} subtitles for the same English sentence (with the sentences before and
after). Which one would you APPROVE for publication? Judge it as a native viewer reading at speed:
natural spoken {L}, faithful meaning, the house terminology and register below. Small wording
preferences count when they make the line read more naturally; ignore line breaks and spacing.

HOUSE STYLE (abridged):
{notes}

EDITS OUR REVIEWERS MADE (draft -> approved), to show their taste:
{examples}

Before: {prev}
ENGLISH: {en}
After: {nxt}
A: {a}
B: {b}
Answer ONLY JSON: {{"better": "A"|"B"|"equal", "gap": "<one of: {cats}>", "why": "<max 15 words>"}}"""


def judge_row(r, judges, L, template, extra, rng):
    votes = []
    for j in judges:
        flip = rng.random() < 0.5
        a, b = (r["human"], r["draft"]) if flip else (r["draft"], r["human"])
        prompt = template.format(
            L=L, prev=r["prev"], en=r["en"], nxt=r["next"], a=a, b=b, cats=", ".join(CATS), **extra
        )
        try:
            m = re.search(r"\{.*\}", j(prompt) or "", re.S)
            d = json.loads(m.group(0)) if m else {}
        except Exception:
            d = {}
        votes.append(
            {"A": "human" if flip else "draft", "B": "draft" if flip else "human"}.get(
                d.get("better"), "equal"
            )
        )
    return votes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="zh-Hans")
    ap.add_argument("--dev-batch", required=True, help="batch the examples are drawn from")
    ap.add_argument("--test-batch", required=True, help="held-out batch")
    ap.add_argument("--n", type=int, default=120, help="edited lines per split")
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--setups", nargs="+", default=["current", "house"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in Path(a.pairs).read_text(encoding="utf-8").splitlines()]
    rng = random.Random(a.seed)
    edited = [r for r in rows if r["edited"]]
    dev = [r for r in edited if r["batch"] == a.dev_batch]
    test = [r for r in edited if r["batch"] == a.test_batch]
    rng.shuffle(dev)
    # examples come from dev lines NOT scored; the test split never contributes examples
    ex_rows, dev_eval = dev[: a.examples], dev[a.examples : a.examples + a.n]
    test_eval = rng.sample(test, min(a.n, len(test)))
    L = LANG.get(a.lang, a.lang)
    st = (load(ROOT / "glossary" / "glossary.yaml").style or {}).get(a.lang, {}) or {}
    notes = (str(st.get("register_conversational", "")) + "\n" + str(st.get("notes", "")))[:3500]
    examples = "\n".join(f"- {r['en']}\n  {r['draft']} -> {r['human']}" for r in ex_rows)
    judges = [
        CliTranslator()._invoke,
        _qe_second_judge(types.SimpleNamespace(qe_judge=True, single_judge=False)),
    ]
    setups = {
        "current": (JUDGE, {}),
        "house": (HOUSE, {"notes": notes, "examples": examples}),
    }
    report = [f"# Judge calibration on reviewer edits ({a.lang})", ""]
    result = {}
    for name in a.setups:
        tpl, extra = setups[name]
        for split, rs in (("dev", dev_eval), ("test", test_eval)):

            def one(item):
                k, r = item
                return judge_row(r, judges, L, tpl, extra, random.Random(a.seed * 1000 + k))

            with ThreadPoolExecutor(8) as ex:
                votes = list(ex.map(one, enumerate(rs)))
            both = Counter(
                (
                    "human"
                    if v.count("human") == len(v)
                    else "draft" if v.count("draft") == len(v) else "split"
                )
                for v in votes
            )
            per = [Counter(v[j] for v in votes) for j in range(len(judges))]
            n = len(rs)
            res = {
                "n": n,
                "both_prefer_reviewer": both["human"] / n,
                "both_prefer_draft": both["draft"] / n,
                "per_judge_prefer_reviewer": [p["human"] / n for p in per],
            }
            result[f"{name}/{split}"] = res
            report.append(
                f"- **{name}**, {split} ({n} edited lines): both judges prefer the reviewer "
                f"{res['both_prefer_reviewer']:.0%}, both prefer the draft {res['both_prefer_draft']:.0%}; "
                f"each judge alone prefers the reviewer "
                + " / ".join(f"{x:.0%}" for x in res["per_judge_prefer_reviewer"])
            )
            (out / f"{name}_{split}.json").write_text(
                json.dumps(
                    [{**r, "votes": v} for r, v in zip(rs, votes)], ensure_ascii=False, indent=1
                ),
                encoding="utf-8",
            )
            print(report[-1], flush=True)
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
