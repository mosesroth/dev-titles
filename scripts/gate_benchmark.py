#!/usr/bin/env python3
"""Large GATE benchmark for the native-editor pass: strict vs loose acceptance, measured against the
human-reviewed subtitles of every reviewed video (no re-translation).

Phase 1 (per video+lang, resumable): run the editor's FLAGGING step on the machine draft the
reviewer saw (<ticket>/<video>/cache.jsonl), with leave-one-video-out examples (no example from
the video being evaluated) and the per-video brief. For every suggestion that passes the
token/number guard, ask the second judge (GPT) the pairwise question once and record BOTH verdicts
(same_meaning, better). -> <out>/suggestions.<video>.<lang>.jsonl

Phase 2: simulate the gates on the recorded verdicts:
    strict = same_meaning AND judge prefers the suggestion   (current production gate)
    loose  = same_meaning                                    (option 1)
and grade, against the human text, every line that either gate would change: ORIGINAL vs
SUGGESTION, each graded by TWO graders (Claude + GPT) in randomized order -> improved / neutral /
harmed per gate. -> <out>/grades.jsonl, <out>/REPORT.md

  PYTHONPATH=src python3 scripts/gate_benchmark.py --lang ja --video VideoA --video ... [--out output/gate_bench]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import types
from pathlib import Path

from dev_titles.cli import _qe_second_judge, _read_any
from dev_titles.editor import (
    EDIT_PROMPT,
    PAIR_PROMPT,
    _parse_json_array,
    deterministic_hints,
    guard_ok,
    video_brief,
    video_description,
)
from dev_titles.resegment import merge_en_to_sentences
from dev_titles.translate import CliTranslator

LNAME = {"ja": "Japanese", "zh-Hans": "Simplified Chinese"}
GRADE = """You are a native {L} speaker and professional subtitle reviewer. For each item: the English line, a
reference translation by a human reviewer, and a CANDIDATE subtitle. Judge the CANDIDATE as if published
as-is (the reference is guidance only; style differences alone are not errors; the candidate is a whole
sentence shown across one or more cues, so don't mark it down for total length).
- OK: a native viewer finds it fine.
- NOTICEABLE: correct meaning, but unnatural/literal/odd term/wrong register — a native viewer would notice.
- MEANING: a native viewer would misunderstand or miss what was said (wrong/lost fact, wrong sense, garbled).
Return ONLY a JSON array: [{{"id": n, "level": "OK|NOTICEABLE|MEANING"}}]

{body}"""
RANK = {"OK": 0, "NOTICEABLE": 1, "MEANING": 2}


def norm(s):
    return re.sub(r"\s", "", s or "")


def load_examples_excluding(lang, video, k=12):
    p = Path(f"glossary/editor_examples.{lang}.jsonl")
    if not p.exists():
        return ""
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("video") != video][:k]
    if not rows:
        return ""
    L = LNAME[lang]
    lines = "\n".join(f"- EN: {e['en']}\n  before: {e['before']}\n  after:  {e['after']}" for e in rows)
    return f"\nEdits native {L} reviewers made to earlier machine drafts (learn the KIND of fix, don't copy):\n{lines}\n"


def phase1(video, lang, a, inv, gpt):
    out = Path(a.out) / f"suggestions.{video}.{lang}.jsonl"
    if out.exists():
        return [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    tdir = Path(a.ticket_dir) / video
    recs = [json.loads(l) for l in open(tdir / "cache.jsonl", encoding="utf-8") if l.strip()]
    recs = sorted([r for r in recs if r.get("lang") == lang], key=lambda r: r["i"])
    en = _read_any(str(tdir / "en.vtt"), "en")
    bdir = Path(a.out) / "briefs"
    bdir.mkdir(parents=True, exist_ok=True)
    brief = video_brief(" ".join(c.text for c in en.cues), inv, bdir / f"{video}.txt",
                        description=video_description(tdir))
    L = LNAME[lang]
    examples = load_examples_excluding(lang, video)
    by_i = {r["i"]: r for r in recs}
    rng = random.Random(f"{video}:{lang}")
    sugg = []
    for k in range(0, len(recs), 6):
        chunk = recs[k : k + 6]
        rows = []
        for n, r in enumerate(chunk, 1):
            h = deterministic_hints(r["txt"], lang)
            rows.append(f"{n} | {r['src']} | {r['txt']}" + (f" | {', '.join(h)}" if h else ""))
        prompt = EDIT_PROMPT.format(L=L, brief=brief or "(none)", register="polite, natural spoken",
                                    terms="(product names in Latin)", examples=examples,
                                    hint_col=" | hints", items="\n".join(rows))
        try:
            flags = _parse_json_array(inv(prompt) or "")
        except Exception:
            flags = []
        for f in flags:
            try:
                r = chunk[int(f.get("id")) - 1]
            except (TypeError, ValueError, IndexError):
                continue
            new = (f.get("rewrite") or "").strip()
            if not new or new == r["txt"] or not guard_ok(r["txt"], new):
                continue
            ctx = " ".join(by_i[j]["src"] for j in (r["i"] - 1, r["i"] + 1) if j in by_i)
            swap = rng.random() < 0.5
            A, B = (new, r["txt"]) if swap else (r["txt"], new)
            try:
                o = gpt(PAIR_PROMPT.format(L=L, en=r["src"], ctx=ctx, a=A, b=B)) or ""
                m = re.search(r"\{.*\}", o, re.S)
                d = json.loads(m.group(0)) if m else {}
            except Exception:
                d = {}
            better = d.get("better")
            prefers_new = better == ("A" if swap else "B")
            sugg.append({"video": video, "lang": lang, "i": r["i"], "src": r["src"], "orig": r["txt"],
                         "sugg": new, "issue": f.get("issue", ""), "same_meaning": bool(d.get("same_meaning")),
                         "prefers_new": prefers_new, "tie": better == "tie"})
        print(f"  {video} {lang} {min(k + 6, len(recs))}/{len(recs)} suggestions={len(sugg)}", flush=True)
    out.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in sugg) + "\n", encoding="utf-8")
    return sugg


def human_text(video, lang, a, src_to_window):
    hum = _read_any(str(Path(a.human_dir) / f"{video}.{lang}.srt"), lang).cues
    return {s: "".join(norm(c.text) for c in hum if st <= (c.start + c.end) / 2 < e)
            for s, (st, e) in src_to_window.items()}


def grade_items(items, lang, grader, label):
    """items: [{id, en, ref, cand}] -> {id: level}"""
    out = {}
    for k in range(0, len(items), 10):
        ch = items[k : k + 10]
        body = "\n\n".join(f"{n + 1}. EN: {x['en']}\n   REFERENCE: {x['ref']}\n   CANDIDATE: {x['cand']}"
                           for n, x in enumerate(ch))
        try:
            o = grader(GRADE.format(L=LNAME[lang], body=body)) or ""
            js = _parse_json_array(o)
        except Exception:
            js = []
        for j in js:
            try:
                out[ch[int(j["id"]) - 1]["id"]] = j.get("level")
            except (KeyError, ValueError, IndexError, TypeError):
                pass
        print(f"    grade[{label}] {min(k + 10, len(items))}/{len(items)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--video", action="append", required=True)
    ap.add_argument("--ticket-dir", default="output/ticket")
    ap.add_argument("--human-dir", default="output/FINAL_SUBTITLES")
    ap.add_argument("--out", default="output/gate_bench")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    inv = CliTranslator()._invoke
    gpt = _qe_second_judge(types.SimpleNamespace(qe_judge=True, single_judge=False))
    if gpt is None:
        raise SystemExit("second judge (codex) unavailable")
    allsug, total = [], 0
    for v in a.video:
        try:
            s = phase1(v, a.lang, a, inv, gpt)
        except FileNotFoundError as e:
            print(f"skip {v}: {e}")
            continue
        en = _read_any(str(Path(a.ticket_dir) / v / "en.vtt"), "en")
        sents = merge_en_to_sentences(en)
        win = {x.text.strip(): (x.start, x.end) for x in sents}
        try:
            ref = human_text(v, a.lang, a, win)
        except FileNotFoundError as e:
            print(f"skip {v}: {e}")
            continue
        total += len(sents)
        for x in s:
            x["ref"] = ref.get(x["src"].strip(), "")
        allsug += [x for x in s if x["ref"]]
    strict = [x for x in allsug if x["same_meaning"] and x["prefers_new"]]
    loose_only = [x for x in allsug if x["same_meaning"] and not x["prefers_new"]]
    graded = strict + loose_only
    items = []
    for n, x in enumerate(graded):
        items.append({"id": f"{n}o", "en": x["src"], "ref": x["ref"], "cand": x["orig"]})
        items.append({"id": f"{n}s", "en": x["src"], "ref": x["ref"], "cand": x["sugg"]})
    random.Random(5).shuffle(items)  # originals and suggestions interleaved, unlabeled
    gc = grade_items(items, a.lang, inv, "claude")
    gg = grade_items(items, a.lang, gpt, "gpt")
    rows = []
    for n, x in enumerate(graded):
        lv = lambda g, t: g.get(f"{n}{t}")
        o = [lv(gc, "o"), lv(gg, "o")]
        s_ = [lv(gc, "s"), lv(gg, "s")]
        if None in o + s_:
            continue
        d = sum(RANK[b] - RANK[a_] for a_, b in zip(o, s_)) / 2  # <0 improved, >0 harmed
        rows.append({**x, "gate": "strict" if x in strict else "loose_only", "orig_lv": o, "sugg_lv": s_,
                     "delta": d, "meaning_harm": any(RANK[b] == 2 and RANK[a_] < 2 for a_, b in zip(o, s_))})
    Path(a.out, f"grades.{a.lang}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    def summ(name, R):
        if not R:
            return f"| {name} | 0 | | | | |"
        imp = sum(r["delta"] < 0 for r in R)
        harm = sum(r["delta"] > 0 for r in R)
        mh = sum(r["meaning_harm"] for r in R)
        return f"| {name} | {len(R)} | {imp} ({imp/len(R):.0%}) | {len(R)-imp-harm} | {harm} ({harm/len(R):.0%}) | {mh} |"

    lines = [f"# Gate benchmark — {a.lang}", "",
             f"{len(a.video)} videos, {total} sentences; {len(allsug)} guarded editor suggestions; "
             f"strict gate accepts {len(strict)}, loose gate accepts {len(strict) + len(loose_only)}.", "",
             "Each changed line graded ORIGINAL vs SUGGESTION by two graders (Claude + GPT), randomized, against the human text.", "",
             "| set | lines | improved | neutral | harmed | new MEANING errors |", "|---|---|---|---|---|---|",
             summ("strict-accepted (current gate)", [r for r in rows if r["gate"] == "strict"]),
             summ("loose-only (extra under option 1)", [r for r in rows if r["gate"] == "loose_only"]),
             "", f"Per 1,000 sentences: strict changes {1000*len(strict)/max(total,1):.0f} lines; "
             f"loose changes {1000*(len(strict)+len(loose_only))/max(total,1):.0f}."]
    Path(a.out, f"REPORT.{a.lang}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("DONE")


if __name__ == "__main__":
    main()
