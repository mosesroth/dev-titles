#!/usr/bin/env python3
"""Compare candidate editor-acceptance gates on the gate_benchmark data.

Input: <out>/grades.<lang>.jsonl from scripts/gate_benchmark.py. Every line there passed GPT's
same-meaning check and has a hindsight label (improved / neutral / harmed vs the human text).
This adds the reference-free verdicts the other gates need, then replays each gate:
  - Claude pairwise verdict (the same PAIR_PROMPT GPT answered, randomized order);
  - reference-free ratings (OK / NOTICEABLE / MEANING) of the current line and the suggestion,
    by Claude and by GPT, without the human text.
Gates (all within GPT same_meaning):
  strict        GPT prefers the suggestion                (current)
  loose         any                                       (option 1)
  both_prefer   GPT and Claude both prefer the suggestion
  claude_prefer Claude prefers the suggestion
  rated_better  both judges rate the suggestion strictly better (reference-free)
  rated_or_gpt  GPT prefers OR both judges rate it strictly better
Verdicts are cached in <out>/gate_verdicts.<lang>.jsonl (resumable).

  PYTHONPATH=src python3 scripts/gate_compare.py --lang ja [--out output/gate_bench]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import types
from pathlib import Path

from dev_titles.cli import _qe_second_judge
from dev_titles.editor import PAIR_PROMPT, _parse_json_array
from dev_titles import llm

LNAME = {"ja": "Japanese", "zh-Hans": "Simplified Chinese"}
RATE = """You are a native {L} speaker and professional subtitle reviewer. For each item: an English line and a
{L} subtitle of it (a whole sentence that may span several cues — don't mark it down for total length).
- OK: a native viewer finds it fine.
- NOTICEABLE: correct meaning, but unnatural/literal/odd term/wrong register — a native viewer would notice.
- MEANING: a native viewer would misunderstand or miss what was said.
Return ONLY a JSON array: [{{"id": n, "level": "OK|NOTICEABLE|MEANING"}}]

{body}"""
RANK = {"OK": 0, "NOTICEABLE": 1, "MEANING": 2}


def key(r):
    return f"{r['video']}#{r['i']}"


def rate(items, judge, L, label):
    out = {}
    for k in range(0, len(items), 10):
        ch = items[k : k + 10]
        body = "\n\n".join(f"{n + 1}. EN: {x['en']}\n   SUBTITLE: {x['cand']}" for n, x in enumerate(ch))
        try:
            js = _parse_json_array(judge(RATE.format(L=L, body=body)) or "")
        except Exception:
            js = []
        for j in js:
            try:
                out[ch[int(j["id"]) - 1]["id"]] = j.get("level")
            except (KeyError, ValueError, IndexError, TypeError):
                pass
        print(f"  rate[{label}] {min(k + 10, len(items))}/{len(items)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out", default="output/gate_bench")
    ap.add_argument("--ticket-dir", default="output/ticket")
    a = ap.parse_args()
    L = LNAME[a.lang]
    rows = [json.loads(l) for l in Path(a.out, f"grades.{a.lang}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    vp = Path(a.out, f"gate_verdicts.{a.lang}.jsonl")
    ver = {}
    if vp.exists():
        for l in vp.read_text(encoding="utf-8").splitlines():
            if l.strip():
                d = json.loads(l)
                ver[d["key"]] = d
    inv = llm.invoke()
    gpt = _qe_second_judge(types.SimpleNamespace(qe_judge=True, single_judge=False))
    todo = [r for r in rows if key(r) not in ver]
    if todo:
        srcs = {}
        for v in {r["video"] for r in todo}:
            recs = [json.loads(l) for l in open(Path(a.ticket_dir) / v / "cache.jsonl", encoding="utf-8") if l.strip()]
            srcs[v] = {x["i"]: x["src"] for x in recs if x.get("lang") == a.lang}
        rng = random.Random(f"compare:{a.lang}")
        claude_pref = {}
        for n, r in enumerate(todo, 1):
            ctx = " ".join(srcs[r["video"]][j] for j in (r["i"] - 1, r["i"] + 1) if j in srcs[r["video"]])
            swap = rng.random() < 0.5
            A, B = (r["sugg"], r["orig"]) if swap else (r["orig"], r["sugg"])
            try:
                o = inv(PAIR_PROMPT.format(L=L, en=r["src"], ctx=ctx, a=A, b=B)) or ""
                m = re.search(r"\{.*\}", o, re.S)
                d = json.loads(m.group(0)) if m else {}
            except Exception:
                d = {}
            claude_pref[key(r)] = d.get("better") == ("A" if swap else "B")
            if n % 10 == 0:
                print(f"  claude pairwise {n}/{len(todo)}", flush=True)
        items = []
        for r in todo:
            items += [{"id": key(r) + "|o", "en": r["src"], "cand": r["orig"]},
                      {"id": key(r) + "|s", "en": r["src"], "cand": r["sugg"]}]
        random.Random(9).shuffle(items)
        rc = rate(items, inv, L, "claude")
        rg = rate(items, gpt, L, "gpt")
        with vp.open("a", encoding="utf-8") as f:
            for r in todo:
                k = key(r)
                d = {"key": k, "claude_prefers": claude_pref[k],
                     "rate_claude": [rc.get(k + "|o"), rc.get(k + "|s")],
                     "rate_gpt": [rg.get(k + "|o"), rg.get(k + "|s")]}
                ver[k] = d
                f.write(json.dumps(d) + "\n")

    def rated_better(v):
        try:
            return all(RANK[s] < RANK[o] for o, s in (v["rate_claude"], v["rate_gpt"]))
        except (KeyError, TypeError):
            return False

    gates = {
        "strict (current)": lambda r, v: r["prefers_new"],
        "loose": lambda r, v: True,
        "both_prefer": lambda r, v: r["prefers_new"] and v["claude_prefers"],
        "claude_prefer": lambda r, v: v["claude_prefers"],
        "rated_better": lambda r, v: rated_better(v),
        "rated_or_gpt": lambda r, v: r["prefers_new"] or rated_better(v),
    }
    total_imp = sum(r["delta"] < 0 for r in rows)
    lines = [f"# Gate comparison — {a.lang}", "",
             f"{len(rows)} graded suggestions (all passed GPT's same-meaning check); {total_imp} of them improve the line "
             "per the hindsight grading (two graders vs the human text).", "",
             "| gate | accepted | improved | neutral | harmed | new MEANING | share of available improvements | improved per harmed |",
             "|---|---|---|---|---|---|---|---|"]
    for name, g in gates.items():
        acc = [r for r in rows if key(r) in ver and g(r, ver[key(r)])]
        imp = sum(r["delta"] < 0 for r in acc)
        harm = sum(r["delta"] > 0 for r in acc)
        mh = sum(r["meaning_harm"] for r in acc)
        ratio = f"{imp / harm:.1f}" if harm else "∞" if imp else "-"
        lines.append(f"| {name} | {len(acc)} | {imp} | {len(acc) - imp - harm} | {harm} | {mh} | "
                     f"{imp / max(total_imp, 1):.0%} | {ratio} |")
    Path(a.out, f"GATES.{a.lang}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("DONE")


if __name__ == "__main__":
    main()
