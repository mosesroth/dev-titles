#!/usr/bin/env python3
"""Grade the suggestions GPT rejected as 'meaning differs' (not covered by gate_benchmark), then
test whether a reference-free rule can rescue the real fixes among them.

Writes <out>/rescue_grades.<lang>.jsonl and <out>/RESCUE.<lang>.md. Reuses the graders and
verdict prompts of gate_benchmark.py / gate_compare.py.

  PYTHONPATH=src python3 scripts/gate_rescue.py --lang ja [--out output/gate_bench]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gate_benchmark import RANK, grade_items, human_text  # noqa: E402
from gate_compare import LNAME, rate  # noqa: E402

from dev_titles.cli import _qe_second_judge, _read_any
from dev_titles.editor import PAIR_PROMPT
from dev_titles.resegment import merge_en_to_sentences
from dev_titles.translate import CliTranslator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out", default="output/gate_bench")
    ap.add_argument("--ticket-dir", default="output/ticket")
    ap.add_argument("--human-dir", default="output/FINAL_SUBTITLES")
    a = ap.parse_args()
    L = LNAME[a.lang]
    pool = []
    for f in sorted(Path(a.out).glob(f"suggestions.*.{a.lang}.jsonl")):
        v = f.name.split(".")[1]
        s = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        s = [x for x in s if not x["same_meaning"]]
        if not s:
            continue
        en = _read_any(str(Path(a.ticket_dir) / v / "en.vtt"), "en")
        win = {x.text.strip(): (x.start, x.end) for x in merge_en_to_sentences(en)}
        ref = human_text(v, a.lang, a, win)
        recs = [json.loads(l) for l in open(Path(a.ticket_dir) / v / "cache.jsonl", encoding="utf-8") if l.strip()]
        srcs = {x["i"]: x["src"] for x in recs if x.get("lang") == a.lang}
        for x in s:
            x["ref"] = ref.get(x["src"].strip(), "")
            x["ctx"] = " ".join(srcs[j] for j in (x["i"] - 1, x["i"] + 1) if j in srcs)
        pool += [x for x in s if x["ref"]]
    print(f"{len(pool)} meaning-rejected suggestions to grade", flush=True)
    inv = CliTranslator()._invoke
    gpt = _qe_second_judge(types.SimpleNamespace(qe_judge=True, single_judge=False))
    items = []
    for n, x in enumerate(pool):
        items += [{"id": f"{n}o", "en": x["src"], "ref": x["ref"], "cand": x["orig"]},
                  {"id": f"{n}s", "en": x["src"], "ref": x["ref"], "cand": x["sugg"]}]
    random.Random(5).shuffle(items)
    gc = grade_items(items, a.lang, inv, "claude")
    gg = grade_items(items, a.lang, gpt, "gpt")
    rc = rate(items, inv, L, "rate-claude")
    rg = rate(items, gpt, L, "rate-gpt")
    rng = random.Random(f"rescue:{a.lang}")
    rows = []
    for n, x in enumerate(pool):
        o, s_ = [gc.get(f"{n}o"), gg.get(f"{n}o")], [gc.get(f"{n}s"), gg.get(f"{n}s")]
        if None in o + s_:
            continue
        swap = rng.random() < 0.5
        A, B = (x["sugg"], x["orig"]) if swap else (x["orig"], x["sugg"])
        try:
            m = re.search(r"\{.*\}", inv(PAIR_PROMPT.format(L=L, en=x["src"], ctx=x["ctx"], a=A, b=B)) or "", re.S)
            d = json.loads(m.group(0)) if m else {}
        except Exception:
            d = {}
        rows.append({**x, "orig_lv": o, "sugg_lv": s_,
                     "delta": sum(RANK[b] - RANK[c] for c, b in zip(o, s_)) / 2,
                     "meaning_harm2": all(RANK[b] == 2 and RANK[c] < 2 for c, b in zip(o, s_)),
                     "claude_same": bool(d.get("same_meaning")),
                     "claude_prefers": d.get("better") == ("A" if swap else "B"),
                     "rate_claude": [rc.get(f"{n}o"), rc.get(f"{n}s")], "rate_gpt": [rg.get(f"{n}o"), rg.get(f"{n}s")]})
        if len(rows) % 20 == 0:
            print(f"  claude pairwise {len(rows)}/{len(pool)}", flush=True)
    Path(a.out, f"rescue_grades.{a.lang}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    def rv(r):
        try:
            return [(RANK[o], RANK[s]) for o, s in (r["rate_claude"], r["rate_gpt"])]
        except (KeyError, TypeError):
            return None

    rules = {
        "all (accept every meaning-rejected suggestion)": lambda r: True,
        "Claude: same meaning + prefers": lambda r: r["claude_same"] and r["claude_prefers"],
        "both rate better": lambda r: rv(r) is not None and all(s < o for o, s in rv(r)),
        "Claude same+prefers AND both rate better": lambda r: r["claude_same"] and r["claude_prefers"]
        and rv(r) is not None and all(s < o for o, s in rv(r)),
        "Claude same+prefers AND none rates worse": lambda r: r["claude_same"] and r["claude_prefers"]
        and rv(r) is not None and not any(s > o for o, s in rv(r)),
    }
    imp_all = sum(r["delta"] < 0 for r in rows)
    out = [f"# Rescue test — {a.lang}", "",
           f"{len(rows)} suggestions GPT rejected as 'meaning differs', graded vs the human text; {imp_all} would improve the line.", "",
           "| rescue rule | accepted | improved | neutral | harmed | MEANING (both graders) |", "|---|---|---|---|---|---|"]
    for name, g in rules.items():
        acc = [r for r in rows if g(r)]
        i = sum(r["delta"] < 0 for r in acc)
        h = sum(r["delta"] > 0 for r in acc)
        out.append(f"| {name} | {len(acc)} | {i} | {len(acc) - i - h} | {h} | {sum(r['meaning_harm2'] for r in acc)} |")
    Path(a.out, f"RESCUE.{a.lang}.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print("DONE")


if __name__ == "__main__":
    main()
