#!/usr/bin/env python3
"""Fix layout violations in reviewed subtitles that are ALREADY free-segmented (the reviewer kept
our cue timings and edited text in place), without changing a character of their text.

Per cue: if a line exceeds max_cpl or the cue has too many lines, re-wrap the cue's text with the
house wrapper; if it still needs more than max_lines, split the cue across its own time window
(resegment_translation). Untouched cues are copied verbatim. Tiny sub-frame gaps are snapped.

A ja 、/。 becomes a half-/full-width space (JP subtitle style-guide rule) — the only text edit,
reported.

Run: PYTHONPATH=src python3 scripts/rewrap_reviewed.py "<dir>" --out <outdir>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from dev_titles.cli import _read_any
from dev_titles.config import spec_for
from dev_titles.formats import write_srt, write_vtt
from dev_titles.model import Cue, Subtitle
from dev_titles.resegment import resegment_translation, snap_gaps
from dev_titles.segment.cjk import wrap_lines
from dev_titles.textwidth import display_width
from dev_titles.validate.rules import validity_report


def join_lines(lines):
    """Latin-aware join: a line break between two ASCII alphanumerics was a word separator."""
    out = ""
    for p in (ln.strip() for ln in lines):
        if (
            out
            and p
            and out[-1].isascii()
            and out[-1].isalnum()
            and p[0].isascii()
            and p[0].isalnum()
        ):
            out += " "
        out += p
    return out


def balanced_split(text, start, end, spec):
    """Split an over-full cue into the fewest cues that fit, cutting near EQUAL text halves —
    preferring the reviewer's own phrase spaces (ja uses a space as a phrase separator) — so we
    never strand a 1-word orphan cue ('新しい' alone for 0.5s). Time is shared by text length."""
    n_cues = 2
    while True:
        parts = _split_n(text, n_cues)
        wrapped = [wrap_lines(p, spec) for p in parts]
        if all(len(w) <= spec.max_lines for w in wrapped) or n_cues >= 6:
            break
        n_cues += 1
    total = sum(len(p) for p in parts) or 1
    out, t = [], start
    for p, w in zip(parts, wrapped):
        dur = (end - start) * len(p) / total
        out.append(Cue(t, t + dur, w))
        t += dur
    return out


def _split_n(text, n):
    """Cut `text` into n pieces at boundaries nearest each 1/n mark: a phrase space first, else a
    wrap_lines line boundary (always a legal break). Never cuts inside a word."""
    spaces = [i for i, ch in enumerate(text) if ch in " 　"]
    # legal break offsets from the wrapper (cumulative line lengths)
    lb, acc = [], 0
    for ln in wrap_lines(text, spec_for("ja"))[:-1]:
        acc = text.find(ln, acc) + len(ln)
        lb.append(acc)
    parts, prev = [], 0
    for k in range(1, n):
        target = len(text) * k / n
        near = len(text) * 0.2  # a phrase space only wins if it's reasonably central
        cands = [s for s in spaces if prev < s < len(text) - 1 and abs(s - target) <= near] or [
            b for b in lb if prev < b < len(text)
        ]
        cut = min(cands, key=lambda c: abs(c - target)) if cands else int(target)
        parts.append(text[prev:cut].strip())
        prev = cut
    parts.append(text[prev:].strip())
    return [p for p in parts if p]


def glossary_audit(sub, lang, glossary_path="glossary/glossary.yaml"):
    from dev_titles.glossary import load

    g = load(glossary_path)
    text = "\n".join(c.text.replace("\n", " ") for c in sub.cues)
    hits = []
    for t in g.terms:
        canon = (t.translations.get(lang) or "").strip()
        if t.enforce != "hard" or not canon or canon.isascii():
            continue
        for form in t.all_source_forms:
            if not form.isascii():
                continue
            # whole Latin run, not part of a URL / file name (example.com/Store, App.js, minisim.app)
            # and not a parenthetical gloss right after the canonical (アプリ内課金（IAP）)
            pat = (
                rf"(?<![A-Za-z0-9./（(])(?<![A-Za-z0-9][ -]){re.escape(form)}"
                rf"(?![A-Za-z0-9])(?![ .-][A-Za-z0-9])(?!/)"
            )
            n = len(re.findall(pat, text, re.I))
            if n:
                hits.append(f"'{form}' x{n} left in English — glossary: {canon}")
                break
    return hits


def lang_of(name):
    if re.search(r"(ja[_-]|JP)", name):
        return "ja"
    if re.search(r"(zh[_-]|CN)", name):
        return "zh-Hans"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(a.src).glob("*.srt")):
        # '<Video>.ja_Reviewed.srt', '<Video>.ja_Reviewed_<date>.srt', '<Video>_Reviewed.ja_….srt'
        video, lang = re.split(r"[._]", f.name)[0], lang_of(f.name)
        if not lang:
            print(f"  SKIP {f.name}")
            continue
        spec = spec_for(lang)
        sub = _read_any(str(f), lang)
        cues, fixed, punct = [], 0, 0
        for c in sub.cues:
            lines = list(c.lines)
            if lang == "ja":
                # JP subtitle style guide (confirmed by a native reviewer): 読点 -> half-width space,
                # 句点 -> full-width
                new = [re.sub(r"\s*。\s*", "　", re.sub(r"\s*、\s*", " ", ln)).strip(" 　") for ln in lines]
                punct += sum(ln.count("、") + ln.count("。") for ln in lines)
                lines = new
            bad = len(lines) > spec.max_lines or any(
                display_width(ln, cjk=True) > spec.max_cpl for ln in lines
            )
            if not bad:
                cues.append(Cue(c.start, c.end, lines))
                continue
            fixed += 1
            text = join_lines(lines)
            wrapped = wrap_lines(text, spec)
            if len(wrapped) <= spec.max_lines:
                cues.append(Cue(c.start, c.end, wrapped))
            else:  # too much text for one cue: BALANCED split across the cue's own window
                cues += balanced_split(text, c.start, c.end, spec)
        # Clamp OVERLAPS (a hand-edited timestamp running into the next cue): end the earlier cue
        # where the next one starts.
        for k in range(len(cues) - 1):
            if cues[k].end > cues[k + 1].start:
                cues[k] = Cue(cues[k].start, cues[k + 1].start, cues[k].lines)
        res = Subtitle(lang, snap_gaps(cues, spec)).renumber()
        (out / f"{video}.{lang}.srt").write_text(write_srt(res), encoding="utf-8")
        (out / f"{video}.{lang}.vtt").write_text(write_vtt(res), encoding="utf-8")
        r = validity_report(res, True)
        print(
            f"  {f.name:<34} {len(sub.cues)}->{len(res.cues)} cues, re-wrapped {fixed}, "
            f"punct->space {punct}, {r['cues_with_errors']} err {r.get('violations_by_rule', {})}"
        )
        # GLOSSARY AUDIT (report only — reviewer text is never auto-changed): a hard glossary term
        # left in its English form where the glossary localizes it (e.g. a reviewed file that kept
        # 'in-app purchase' where the glossary says アプリ内課金).
        for hit in glossary_audit(res, lang):
            print(f"      glossary: {hit}")


if __name__ == "__main__":
    sys.exit(main())
