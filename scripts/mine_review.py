#!/usr/bin/env python3
"""Learn from human reviews: mine before->after pairs (our machine sentence vs. the reviewer's
final text) and keep the ones that teach a GENERALIZABLE naturalness fix, as few-shot examples for
the native-editor pass (glossary/editor_examples.<lang>.jsonl). Also writes a report of recurring
word substitutions (candidate glossary/style rules — for a human/assistant to verify, never applied
automatically).

  PYTHONPATH=src python3 scripts/mine_review.py --lang ja --video VideoA --video VideoB ...
      [--machine-dir output/ticket] [--human-dir output/FINAL_SUBTITLES] [--exclude VideoC ...]

Machine text comes from <machine-dir>/<video>/cache.jsonl (sentence records: the draft the reviewer
saw); human text is the reviewer's subtitles grabbed by each English sentence's time window. Pairs
are pre-filtered (similar length, real but partial edit) to drop window-misalignment artifacts,
then an LLM curator keeps at most --keep pairs that illustrate distinct, reusable fixes.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
from pathlib import Path

from dev_titles import llm
from dev_titles.cli import _read_any
from dev_titles.resegment import merge_en_to_sentences

CURATE = """You curate teaching examples for a {L} subtitle editor. Each item is an English line, our MACHINE
subtitle, and the HUMAN reviewer's final subtitle. Keep ONLY items where the human edit fixes something a
native viewer would notice and that GENERALIZES to other videos: literal idiom -> natural phrase, odd or
coined term -> standard term, stiff/written -> natural spoken, wrong register, run-on -> clean, a context
misread. DROP items that are mere condensation, reviewer taste, video-specific facts, or misaligned (the
human text covers a different part of the sentence). Prefer variety; at most {keep}.
Return ONLY a JSON array of the kept item numbers with a 3-6 word label, e.g. [{{"n": 4, "label": "literal idiom"}}].

{items}"""


def norm(s):
    return re.sub(r"\s", "", s or "")


def mine(video, lang, mdir, hdir):
    rec = {
        json.loads(l)["src"].strip(): json.loads(l)["txt"]
        for l in open(Path(mdir) / video / "cache.jsonl", encoding="utf-8")
        if l.strip() and json.loads(l).get("lang") == lang
    }
    en = _read_any(str(Path(mdir) / video / "en.vtt"), "en")
    hum = _read_any(str(Path(hdir) / f"{video}.{lang}.srt"), lang).cues
    sents = merge_en_to_sentences(en)
    # <hdir>/<video>.alignment.json (scripts/retime_drifted.py): the reviewer's text per EN sentence,
    # aligned by CONTENT. Preferred over time windows, which mis-pair when the track drifted (the
    # Sept 1:1 tracks were up to two cues off, so the window grabbed the neighbouring sentence).
    ap = Path(hdir) / f"{video}.alignment.json"
    aligned = json.loads(ap.read_text(encoding="utf-8")) if ap.exists() else None
    if aligned is not None and len(aligned) != len(sents):
        aligned = None
    out = []
    for k, s in enumerate(sents):
        m = norm(rec.get(s.text.strip()))
        h = (
            norm(aligned[k].replace("|", ""))
            if aligned is not None
            else "".join(norm(c.text) for c in hum if s.start <= (c.start + c.end) / 2 < s.end)
        )
        if not m or not h or m == h:
            continue
        r = difflib.SequenceMatcher(None, m, h, autojunk=False).ratio()
        if 0.35 <= r <= 0.92 and 0.7 <= len(h) / max(len(m), 1) <= 1.35:
            out.append({"video": video, "en": s.text, "before": m, "after": h, "sim": round(r, 3)})
    return out


def substitutions(pairs):
    c = collections.Counter()
    vids = collections.defaultdict(set)
    for p in pairs:
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, p["before"], p["after"], autojunk=False
        ).get_opcodes():
            a, b = p["before"][i1:i2], p["after"][j1:j2]
            if op == "replace" and 2 <= len(a) <= 8 and 2 <= len(b) <= 8:
                c[(a, b)] += 1
                vids[(a, b)].add(p["video"])
    return [(a, b, n, len(vids[(a, b)])) for (a, b), n in c.most_common() if len(vids[(a, b)]) >= 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--video", action="append", required=True)
    ap.add_argument("--exclude", action="append", default=[], help="held-out benchmark videos")
    ap.add_argument("--machine-dir", default="output/ticket")
    ap.add_argument("--human-dir", default="output/FINAL_SUBTITLES")
    ap.add_argument("--keep", type=int, default=16)
    ap.add_argument("--out", default="glossary/editor_examples.{lang}.jsonl")
    a = ap.parse_args()
    L = {"ja": "Japanese", "zh-Hans": "Simplified Chinese"}.get(a.lang, a.lang)
    pairs = []
    for v in a.video:
        if v in a.exclude:
            continue
        try:
            pairs += mine(v, a.lang, a.machine_dir, a.human_dir)
        except FileNotFoundError as e:
            print(f"  skip {v}: {e}")
    print(f"{len(pairs)} candidate pairs from {len(a.video) - len(a.exclude)} videos")
    inv = llm.invoke()
    kept = []
    for k in range(0, len(pairs), 40):
        chunk = pairs[k : k + 40]
        body = "\n\n".join(
            f"{n + 1}. EN: {p['en']}\n   MACHINE: {p['before']}\n   HUMAN: {p['after']}"
            for n, p in enumerate(chunk)
        )
        resp = inv(CURATE.format(L=L, keep=6, items=body)) or ""
        m = re.search(r"\[.*\]", resp, re.S)
        try:
            sel = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            sel = []
        for s in sel:
            try:
                kept.append({**chunk[int(s["n"]) - 1], "label": s.get("label", "")})
            except (KeyError, ValueError, IndexError, TypeError):
                pass
        print(f"  curated {min(k + 40, len(pairs))}/{len(pairs)} -> {len(kept)} kept")
    # variety: one per label first, then fill
    seen, final = set(), []
    for p in kept:
        if p["label"] not in seen:
            final.append(p)
            seen.add(p["label"])
    final += [p for p in kept if p not in final]
    final = final[: a.keep]
    out = Path(a.out.format(lang=a.lang))
    out.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in final) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(final)} examples -> {out}")
    rep = out.with_suffix(".substitutions.txt")
    rep.write_text(
        "Recurring reviewer substitutions (>=2 videos) — candidate glossary/style rules, VERIFY before use:\n"
        + "\n".join(f"{n:3} {v}v  {a_} -> {b}" for a_, b, n, v in substitutions(pairs)[:80]),
        encoding="utf-8",
    )
    print(f"wrote substitution report -> {rep}")


if __name__ == "__main__":
    main()
