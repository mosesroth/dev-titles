#!/usr/bin/env python3
"""Score CJK segmentation against a native reviewer's hand-corrected YouTube tracks.

Reference: the reviewer's live track (same words as ours in most places; she moved cue and line
breaks). Hypothesis: our delivered text, regrouped into the EN sentence windows and re-segmented
with the CURRENT harness (build_target_cues), so harness changes can be measured without any LLM
call or re-translation.

Metrics (character positions in a whitespace-free stream, mapped between the two texts):
  cue-break F1 vs the reviewer   line-break F1 vs the reviewer
  Latin name split across a cue/line   cues under 3 s   validation err-cues

  PYTHONPATH=src python3 scripts/eval/seg_eval.py --live-dir output/_live_ja_20261019 [--lang ja]
      [--delivered output/FINAL_SUBTITLES] [--ticket output/ticket] [--baseline]
--baseline scores the delivered files as they are (no re-segmentation).
"""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path

from dev_titles.cli import _read_any
from dev_titles.config import spec_for
from dev_titles.glossary import load as load_glossary
from dev_titles.model import Subtitle
from dev_titles.resegment import build_target_cues, merge_en_to_sentences
from dev_titles.validate.rules import validity_report

WS = re.compile(r"[\s　]")
LAT_SPLIT = re.compile(r"[A-Za-z0-9]$")


def stream(cues):
    """(text, cue_breaks, line_breaks) on the whitespace-free char stream."""
    s, cb, lb = "", set(), set()
    for c in cues:
        lines = [WS.sub("", ln) for ln in c.lines if WS.sub("", ln)]
        for k, ln in enumerate(lines):
            s += ln
            if k < len(lines) - 1:
                lb.add(len(s))
        cb.add(len(s))
    cb.discard(len(s))
    return s, cb, lb


def latin_splits(cues):
    n = 0
    flat = [ln for c in cues for ln in c.lines if ln.strip()]
    for a, b in zip(flat, flat[1:]):
        if LAT_SPLIT.search(a.strip()) and re.match(r"[A-Za-z]", b.strip()):
            n += 1
    return n


def f1(hyp, ref):
    tp = len(hyp & ref)
    p = tp / len(hyp) if hyp else 1.0
    r = tp / len(ref) if ref else 1.0
    return tp, len(hyp), len(ref), (2 * p * r / (p + r) if p + r else 0.0)


def regroup(en, delivered):
    """Delivered free-seg cues -> per-EN-sentence text (each cue lies inside one window)."""
    sents = merge_en_to_sentences(en)
    texts = [[] for _ in sents]
    j = 0
    for c in delivered.cues:
        mid = (c.start + c.end) / 2
        while j + 1 < len(sents) and mid >= sents[j].end and mid >= sents[j + 1].start - 0.01:
            j += 1
        texts[j].append("".join(c.lines))  # ja/zh: a cue/line break is not a space
    return sents, ["".join(t) for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-dir", required=True)
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--delivered", default="output/FINAL_SUBTITLES")
    ap.add_argument("--ticket", default="output/ticket")
    ap.add_argument("--glossary", default="glossary/glossary.yaml")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument(
        "--from-cache", action="store_true", help="hypothesis text = cached machine translations"
    )
    a = ap.parse_args()
    spec = spec_for(a.lang)
    g = load_glossary(a.glossary)
    keep = [t.translations.get(a.lang) for t in g.terms if t.translations.get(a.lang)]
    keep += list(g.identifiers)
    tot = dict(ctp=0, ch=0, cr=0, ltp=0, lh=0, lr=0, lat=0, cues=0, short=0, err=0)
    for lv in sorted(Path(a.live_dir).glob(f"*.{a.lang}.vtt")):
        v = lv.name.split(".")[0]
        ref = _read_any(str(lv), a.lang).cues
        dl = _read_any(str(Path(a.delivered) / f"{v}.{a.lang}.srt"), a.lang)
        if a.baseline:
            hyp = dl.cues
        else:
            en = _read_any(str(Path(a.ticket) / v / "en.vtt"), "en")
            if a.from_cache:  # machine translations with their own punctuation/spaces intact
                import json

                recs = {}
                for l in open(Path(a.ticket) / v / "cache.jsonl", encoding="utf-8"):
                    if l.strip():
                        r = json.loads(l)
                        if r.get("lang") == a.lang:
                            recs[r["src"].strip()] = r["txt"]
                sents = merge_en_to_sentences(en)
                texts = [recs.get(x.text.strip(), "") for x in sents]
            else:
                sents, texts = regroup(en, dl)
            hyp = build_target_cues(en, sents, texts, spec, keep_together=keep, free_segment=True)
        hs, hcb, hlb = stream(hyp)
        rs, rcb, rlb = stream(ref)
        # map hypothesis positions onto the reference stream (texts differ where she reworded)
        m = {}
        for i1, j1, n in difflib.SequenceMatcher(
            None, hs, rs, autojunk=False
        ).get_matching_blocks():
            for k in range(n + 1):
                m[i1 + k] = j1 + k
        hcb_r = {m[p] for p in hcb if p in m}
        hlb_r = {m[p] for p in hlb if p in m}
        ctp, ch, cr, _ = f1(hcb_r, rcb)
        ltp, lh, lr, _ = f1(hlb_r, rlb)
        sub = Subtitle(a.lang, list(hyp)).renumber()
        rep = validity_report(sub, False)
        row = dict(
            ctp=ctp,
            ch=ch,
            cr=cr,
            ltp=ltp,
            lh=lh,
            lr=lr,
            lat=latin_splits(hyp),
            cues=len(hyp),
            short=sum(c.end - c.start < 3 for c in hyp),
            err=rep["cues_with_errors"],
        )
        for k in tot:
            tot[k] += row[k]

    def F(tp, h, r):
        p, q = tp / max(h, 1), tp / max(r, 1)
        return 2 * p * q / max(p + q, 1e-9)

    print(
        f"{'baseline (delivered)' if a.baseline else 'current harness'}: "
        f"cue-break F1 {F(tot['ctp'], tot['ch'], tot['cr']):.3f} (ours {tot['ch']} / hers {tot['cr']})  "
        f"line-break F1 {F(tot['ltp'], tot['lh'], tot['lr']):.3f}  "
        f"Latin splits {tot['lat']}  cues {tot['cues']}  <3s {tot['short'] / max(tot['cues'], 1):.0%}  "
        f"err-cues {tot['err']}"
    )


if __name__ == "__main__":
    main()
