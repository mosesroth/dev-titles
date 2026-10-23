"""CLEAN-READ for unscripted English: remove stutters, false starts and filler before the EN review.

Why: an interview shipped "the standard, in the, in the, in the, in the industry" and "to, to help that,
help us". The pipeline's clean-read only ran when a caption "looked raw" (filler tokens / over-long lines),
and in Sep 2026 it had been dialed back to errors-only because it over-edited correct speech (like ->
such as, grammar restyles). This pass can't over-edit: it may only DELETE words and fix punctuation /
capitalization. Any cue whose cleaned words are not an in-order subset of its original words is rejected.

Scope (operator, 2026-10-23): everything except clearly scripted videos (tutorials, keynotes,
announcements read from a script). A cached LLM call decides; `--en-style` overrides it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

STYLE = """Is the speech in this developer video SCRIPTED or UNSCRIPTED?
- scripted: read from a prepared script — a tutorial / walkthrough narrated by one presenter, a keynote or
  product announcement, an explainer.
- unscripted: spontaneous speech — an interview, fireside chat, panel, podcast, conversation, a talk given
  off the cuff with stumbles and false starts.
Title: {title}
Transcript (opening): {text}
Answer ONLY JSON: {{"style": "scripted"|"unscripted", "why": "<max 15 words>"}}"""

CLEAN = """Clean-read these English subtitle cues of spoken (unscripted) speech, the way a professional captioner
would: remove stutters and false starts ("to, to" -> "to"; "this is, this is" -> "this is"), filler words
("like," "you know," "uh") where they carry no meaning, a phrase repeated across two cues, self-corrections
("Devices, sorry, the Expo framework" -> keep only the corrected words), and stray commas;
fix capitalization at sentence starts. You may ONLY delete words and change punctuation/capitalization —
never add, replace or reorder words, never change meaning. Keep each cue's text in its own cue (a sentence
may continue into the next cue). Leave a cue unchanged if it is already clean.
Cues (id | text):
{rows}
Return ONLY JSON: [{{"id": n, "text": "<cleaned cue>"}}] for changed cues only."""


def _words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", (s or "").lower().replace("’", "'")).split()


def deletions_only(old: str, new: str) -> bool:
    """True if `new`'s words are an in-order subset of `old`'s (only deletions; punctuation/case free)."""
    it = iter(_words(old))
    return bool(_words(new)) and all(any(w == x for x in it) for w in _words(new))


def wrap2(text: str, max_cpl: int = 42) -> list[str]:
    """One line if it fits, else the most balanced two-line split with both lines within max_cpl."""
    if len(text) <= max_cpl:
        return [text]
    best = None
    for i, ch in enumerate(text):
        if ch == " " and len(text[:i]) <= max_cpl and len(text[i + 1 :]) <= max_cpl:
            score = abs(len(text[:i]) - len(text[i + 1 :]))
            if best is None or score < best[0]:
                best = (score, i)
    return [text[: best[1]], text[best[1] + 1 :]] if best else [text]


def speech_style(en, invoke, title: str = "", cache_path=None, override: str | None = None) -> dict:
    """{'style': 'scripted'|'unscripted', 'why', 'source'}; cached in cache_path."""
    if override in ("scripted", "unscripted"):
        return {"style": override, "why": "set by --en-style", "source": "flag"}
    p = Path(cache_path) if cache_path else None
    if p and p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    text = " ".join(" ".join(c.lines) for c in en.cues)[:4000]
    try:
        m = re.search(
            r"\{.*?\}", invoke(STYLE.format(title=title or "(none)", text=text)) or "", re.S
        )
        d = json.loads(m.group(0)) if m else {}
    except Exception:
        d = {}
    style = d.get("style") if d.get("style") in ("scripted", "unscripted") else "unscripted"
    out = {
        "style": style,
        "why": d.get("why", "classifier failed — defaulting to clean-read"),
        "source": "llm",
    }
    if p:
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def clean_read(en, invoke, batch: int = 30):
    """(cleaned Subtitle, [ {i, old, new} applied ], rejected count). Cue timing is never changed."""
    import copy

    out = copy.deepcopy(en)
    cues = out.cues
    applied, rejected = [], 0
    for s in range(0, len(cues), batch):
        rows = "\n".join(
            f"{k} | {' '.join(cues[k].lines)}" for k in range(s, min(s + batch, len(cues)))
        )
        raw = invoke(CLEAN.format(rows=rows)) or ""
        try:
            edits = json.JSONDecoder().raw_decode(raw[raw.index("[") :])[0]
        except (ValueError, json.JSONDecodeError):
            continue
        for e in edits:
            try:
                k = int(e["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if not s <= k < min(s + batch, len(cues)):
                continue
            old, new = " ".join(cues[k].lines), str(e.get("text", "")).strip()
            if not new or new == old:
                continue
            if not deletions_only(old, new):
                rejected += 1
                continue
            cues[k].lines = wrap2(new)
            applied.append({"i": k, "start": round(cues[k].start, 2), "old": old, "new": new})
    return out, applied, rejected
