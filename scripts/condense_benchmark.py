#!/usr/bin/env python3
"""Does ja condensation lose anything important? Measured against a native reviewer's final tracks.

Runs condensation exactly as the pipeline does (same budgets, trigger and guards; condense_lang)
on the cached machine drafts of reviewed videos, then for every line it shortens asks two graders
(Claude + GPT), with the reviewer's published text as the reference for what matters:
  lost:   none | minor | important   (a fact, step, condition, name, number, the speaker's point)
  broken: is the shortened line still grammatical and natural?
No re-translation. Writes <out>/condense_grades.jsonl and <out>/REPORT.md.

  PYTHONPATH=src python3 scripts/condense_benchmark.py --video A --video B ... \
      [--live-dir output/_live_ja_20261019] [--trigger-cps 10] [--out output/condense_bench]
"""

from __future__ import annotations

import argparse
import json
import re
import types
from pathlib import Path

from dev_titles.cli import _qe_second_judge, _read_any
from dev_titles.condense import condense_lang
from dev_titles.config import spec_for
from dev_titles.localize import SentenceRecord, _normalizer, coverage_missing
from dev_titles.resegment import merge_en_to_sentences
from dev_titles.textwidth import display_width
from dev_titles.translate import CliTranslator

GRADE = """You check a subtitle editor's work on a Japanese subtitle of a developer video. The editor shortened
the ORIGINAL so it can be read in its time on screen. The REVIEWER line is what a native reviewer
published for the same moment (guidance for what matters; wording differences are fine).

English: {en}
ORIGINAL: {orig}
SHORTENED: {new}
REVIEWER: {ref}

Answer ONLY JSON: {{"lost": "none"|"minor"|"important", "broken": true|false, "what": "<max 15 words>"}}
- "important": the shortened line drops or changes something that matters — a fact, step, condition,
  name, number, the speaker's actual point, or politeness the reviewer kept.
- "minor": only filler, hedges, intensifiers, repetition or a nuance the reviewer also dropped.
- "broken": the shortened line is ungrammatical, a fragment, or reads unnaturally."""


def _w(s):
    return display_width(re.sub(r"\s", "", s or ""), True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="append", required=True)
    ap.add_argument("--ticket-dir", default="output/ticket")
    ap.add_argument("--live-dir", default="output/_live_ja_20261019")
    ap.add_argument("--trigger-cps", type=float, default=10.0)
    ap.add_argument("--out", default="output/condense_bench")
    ap.add_argument("--grade-only", action="store_true",
                    help="reuse the saved condensation logs in --out; only (re)grade")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-width-line", action="store_true", help="production rule: skip one-line sentences")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inv = CliTranslator()._invoke
    gpt = _qe_second_judge(types.SimpleNamespace(qe_judge=True, single_judge=False))
    spec = spec_for("ja")
    rows, total_sent, total_chars = [], 0, 0
    for v in a.video:
        T = Path(a.ticket_dir) / v
        live = Path(a.live_dir) / f"{v}.ja.vtt"
        if not (T / "cache.jsonl").exists() or not live.exists():
            print("skip", v)
            continue
        sents = merge_en_to_sentences(_read_any(str(T / "en.vtt"), "en"))
        cache = {}
        for line in open(T / "cache.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r.get("lang") == "ja":
                cache.setdefault(r["src"].strip(), r["txt"])
        recs = [SentenceRecord("ja", i, s.text, cache.get(s.text.strip(), ""), "", []) for i, s in enumerate(sents)]
        recs = [r for r in recs if r.txt]
        total_sent += len(recs)
        total_chars += sum(_w(r.txt) for r in recs)
        before = {r.i: r.txt for r in recs}
        if a.grade_only:
            logp = out / f"{v}.condense_log.json"
            done = json.loads(logp.read_text(encoding="utf-8")) if logp.exists() else {}
            by_src = {e["src"].strip(): e for e in done.values()}
            changed = set()
            for r in recs:
                e = by_src.get(r.src.strip())
                if e and e.get("after") and e["after"] != before[r.i]:
                    r.txt = e["after"]
                    changed.add(r.i)
        else:
            budgets = {i: max(int(spec.max_cps * max(s.end - s.start, spec.min_dur)), 6) for i, s in enumerate(sents)}
            triggers = {i: (budgets[i] if s.text.count(",") >= 3
                            else max(int(a.trigger_cps * max(s.end - s.start, spec.min_dur)), 6))
                        for i, s in enumerate(sents)}
            changed = condense_lang(recs, inv, "ja", "Japanese", out / f"{v}.condense_state.json",
                                    coverage_missing=coverage_missing, normalize=_normalizer("ja"),
                                    logger=lambda m: None, budgets=budgets, triggers=triggers,
                                    min_width=spec.max_cpl if a.min_width_line else 0)
        ref_cues = _read_any(str(live), "ja").cues
        for r in recs:
            if r.i not in changed:
                continue
            s = sents[r.i]
            dur = max(s.end - s.start, spec.min_dur)
            ref = "".join(re.sub(r"\s", "", c.text) for c in ref_cues if s.start - 0.3 <= (c.start + c.end) / 2 <= s.end + 0.3)
            rows.append({"video": v, "i": r.i, "en": s.text, "orig": before[r.i], "new": r.txt, "ref": ref,
                         "cps_before": round(_w(before[r.i]) / dur, 1), "cps_after": round(_w(r.txt) / dur, 1)})
        print(f"{v}: {len(changed)} of {len(recs)} sentences condensed", flush=True)
    from concurrent.futures import ThreadPoolExecutor

    def grade(row):
        gs = []
        for name, judge in (("claude", inv), ("gpt", gpt)):
            try:
                m = re.search(r"\{.*\}", judge(GRADE.format(**{k: row[k] for k in ("en", "orig", "new", "ref")})) or "", re.S)
                d = json.loads(m.group(0)) if m else {}
            except Exception:
                d = {}
            gs.append({"judge": name, **d})
        row["grades"] = gs
        return row

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, _ in enumerate(ex.map(grade, rows), 1):
            if n % 50 == 0:
                print(f"  graded {n}/{len(rows)}", flush=True)
    (out / "condense_grades.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    def cnt(pred):
        return sum(1 for r in rows if pred([g.get("lost") for g in r["grades"]], [g.get("broken") for g in r["grades"]]))

    imp_both = cnt(lambda L, B: L.count("important") == 2)
    imp_any = cnt(lambda L, B: "important" in L)
    brk_both = cnt(lambda L, B: B.count(True) == 2)
    brk_any = cnt(lambda L, B: True in B)
    cut = sum(_w(r["orig"]) - _w(r["new"]) for r in rows)
    fast_fixed = sum(1 for r in rows if r["cps_after"] <= 10)
    k = 1000 / max(total_sent, 1)
    rep = [
        "# Condensation benchmark (ja) vs native reviewer's published tracks", "",
        f"{len(a.video)} videos, {total_sent} sentences, trigger {a.trigger_cps} cps (production settings).", "",
        f"- Condensed: {len(rows)} sentences ({len(rows) * k:.0f} per 1,000); characters cut: {cut} "
        f"({cut / max(total_chars, 1):.1%} of all Japanese text).",
        f"- Brought within 10 cps: {fast_fixed} of {len(rows)}.",
        f"- Important content lost: {imp_both} (both graders) / {imp_any} (either) = "
        f"{imp_both * k:.1f} / {imp_any * k:.1f} per 1,000 sentences.",
        f"- Broken or unnatural line: {brk_both} (both) / {brk_any} (either).", "",
        "Lines either grader called an important loss:", "",
    ]
    for r in rows:
        if any(g.get("lost") == "important" for g in r["grades"]):
            whats = " / ".join(g.get("what", "") for g in r["grades"] if g.get("lost") == "important")
            rep.append(f"- {r['video']} #{r['i']}: {r['orig']} -> {r['new']} ({whats})")
    (out / "REPORT.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep[:9]))
    print("DONE")


if __name__ == "__main__":
    main()
