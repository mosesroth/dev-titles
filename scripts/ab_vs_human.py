#!/usr/bin/env python3
"""How close is a fresh translation by the CURRENT harness to the human-reviewed final?

Re-translate an already-reviewed video (held out: SUBTITLE_HOLDOUT_VIDEOS, TM/exemplar exclusion), then
per EN sentence compare three versions:
  OLD    — the machine draft the reviewer saw (the earlier run's cache)
  NEW    — the current harness's output (the fresh run's cache)
  HUMAN  — the reviewed final, aligned to the EN sentences by CONTENT (scripts/retime_drifted.py)
Metrics: chrF (char n-gram F, 0-100) of OLD and NEW against HUMAN; then a blind two-judge comparison of
NEW vs HUMAN wherever they differ: which reads better as a subtitle (or equal), and the KIND of gap.

  PYTHONPATH=src python3 scripts/ab_vs_human.py --new-dir output/ab/MyVideo \
      --old-dir output/ticket/MyVideo --human zh-Hans=<file> --human ja=<file> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dev_titles import llm
from dev_titles.cli import _read_any
from dev_titles.resegment import merge_en_to_sentences

LANG_NAME = {"zh-Hans": "Simplified Chinese", "ja": "Japanese"}
CATS = ["meaning", "terminology", "naturalness", "register", "omission", "addition", "punctuation_only"]
JUDGE = """Two {L} subtitles for the same English sentence of a developer video (with the sentences before
and after for context). Which is the better subtitle for a native viewer — accurate, natural, idiomatic,
consistent terminology, appropriate politeness? Ignore line breaks and spacing.
Before: {prev}
ENGLISH: {en}
After: {nxt}
A: {a}
B: {b}
Answer ONLY JSON: {{"better": "A"|"B"|"equal", "gap": "<one of: {cats}>", "why": "<max 15 words>"}}
"gap" = the main kind of difference between them (for "equal", the kind of difference anyway)."""


def chrf(hyp: str, ref: str, n: int = 4, beta: float = 2.0) -> float:
    h, r = re.sub(r"\s", "", hyp or ""), re.sub(r"\s", "", ref or "")
    if not h or not r:
        return 0.0
    p_sum = r_sum = 0.0
    k = 0
    for i in range(1, n + 1):
        hg = Counter(h[j:j + i] for j in range(len(h) - i + 1))
        rg = Counter(r[j:j + i] for j in range(len(r) - i + 1))
        if not hg or not rg:
            continue
        ov = sum((hg & rg).values())
        p_sum += ov / sum(hg.values())
        r_sum += ov / sum(rg.values())
        k += 1
    if not k:
        return 0.0
    p, rc = p_sum / k, r_sum / k
    return 0.0 if p + rc == 0 else 100 * (1 + beta ** 2) * p * rc / (beta ** 2 * p + rc)


def cache_texts(d: Path, lang: str) -> dict:
    out = {}
    for line in open(d / "cache.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("lang") == lang:
            out[r["src"].strip()] = r["txt"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dir", required=True)
    ap.add_argument("--old-dir", required=True)
    ap.add_argument("--en", help="EN the new run used (default: the *.en.vtt / en.vtt in --new-dir)")
    ap.add_argument("--human", action="append", required=True, help="LANG=path to the reviewed final")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nd, od = Path(a.new_dir), Path(a.old_dir)
    en_p = a.en or str(next(iter(sorted(nd.glob("*.en.vtt")) + [nd / "en.vtt"])))
    sents = merge_en_to_sentences(_read_any(en_p, "en"))
    claude = llm.invoke()
    gpt = llm.second_judge()[0] or claude  # a different-family judge when available
    rng = random.Random(0)
    report = [f"# Fresh translation vs human final — {nd.name}", ""]
    allrows = {}
    for spec in a.human:
        lang, hp = spec.split("=", 1)
        L = LANG_NAME[lang]
        base = out / f"human_{lang}"
        if not Path(str(base) + ".alignment.json").exists():
            subprocess.run([sys.executable, "scripts/retime_drifted.py", hp, en_p, str(base)], check=True)
        human = json.loads(Path(str(base) + ".alignment.json").read_text(encoding="utf-8"))
        new, old = cache_texts(nd, lang), cache_texts(od, lang)
        rows = []
        for k, s in enumerate(sents):
            src = s.text.strip()
            rows.append({"i": k, "en": s.text, "human": human[k].replace("|", ""), "new": new.get(src, ""),
                         "old": old.get(src)})
        N = lambda x: re.sub(r"[\s，、。．・　]", "", x or "")

        def judge(r):
            if not r["new"] or not r["human"] or N(r["new"]) == N(r["human"]):
                r["verdict"] = "identical" if r["new"] and N(r["new"]) == N(r["human"]) else "n/a"
                return r
            prev = rows[r["i"] - 1]["en"] if r["i"] else ""
            nxt = rows[r["i"] + 1]["en"] if r["i"] + 1 < len(rows) else ""
            votes = []
            for j in (claude, gpt):
                flip = rng.random() < 0.5
                A, B = (r["new"], r["human"]) if flip else (r["human"], r["new"])
                try:
                    m = re.search(r"\{.*\}", j(JUDGE.format(L=L, prev=prev, en=r["en"], nxt=nxt, a=A, b=B,
                                                            cats=", ".join(CATS))) or "", re.S)
                    d = json.loads(m.group(0))
                except Exception:
                    d = {"better": "?"}
                win = {"A": "new" if flip else "human", "B": "human" if flip else "new"}.get(d.get("better"), "equal")
                votes.append({"win": win, "gap": d.get("gap"), "why": d.get("why", "")})
            w = [v["win"] for v in votes]
            r["verdict"] = ("human better" if w.count("human") == 2 else "new better" if w.count("new") == 2
                            else "equal" if w.count("equal") == 2 else "split")
            r["gaps"] = [v["gap"] for v in votes]
            r["why"] = " | ".join(v["why"] for v in votes)
            return r

        with ThreadPoolExecutor(8) as ex:
            rows = list(ex.map(judge, rows))
        allrows[lang] = rows
        both = [r for r in rows if r["old"] is not None and r["new"] and r["human"]]
        c_new = sum(chrf(r["new"], r["human"]) for r in both) / max(len(both), 1)
        c_old = sum(chrf(r["old"], r["human"]) for r in both) / max(len(both), 1)
        v = Counter(r["verdict"] for r in rows if r["verdict"] != "n/a")
        tot = sum(v.values())
        gap = Counter(r["gaps"][0] for r in rows if r.get("verdict") == "human better" and r.get("gaps"))
        report += [f"## {lang} ({len(rows)} sentences)", "",
                   f"- chrF vs human: OLD (what the reviewer saw) {c_old:.1f} -> NEW (current harness) {c_new:.1f}",
                   f"- NEW vs HUMAN, blind two-judge: " + ", ".join(f"{k} {n} ({n / tot:.0%})" for k, n in v.most_common()),
                   f"- where HUMAN is better, the gap: " + ", ".join(f"{k} {n}" for k, n in gap.most_common()), "",
                   "Human better (both judges):", ""]
        for r in rows:
            if r["verdict"] == "human better":
                report.append(f"- #{r['i']} [{'/'.join(map(str, r['gaps']))}] EN: {r['en']}\n  - NEW: {r['new']}\n"
                              f"  - HUMAN: {r['human']}\n  - {r['why']}")
        report.append("")
    (out / "ab_rows.json").write_text(json.dumps(allrows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(l for l in report if l.startswith(("#", "- chrF", "- NEW", "- where"))))


if __name__ == "__main__":
    main()
