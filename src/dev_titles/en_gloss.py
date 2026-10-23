"""MEANING NOTES: resolve idioms and ambiguous phrases in the ENGLISH once, before any language is translated.

Why (A/B of a fresh translation vs the human-reviewed finals): the same three lines were wrong in BOTH zh and ja,
so the cause was reading the English, not writing the target:
  "Bring it home."                       = make it concrete (for developers), not "wrap up"
  "two ways of having local models"      = OBTAINING them (shipped vs downloaded), not using them
  "I was very out of date until I got it" = my old PHONE was outdated, not me
One cached LLM pass over the transcript (with the video brief) lists such traps; each note is attached
to its sentence's translation prompt and shown to the native editor next to the English.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PROMPT = """You prepare an English interview/talk transcript for translators into Chinese and Japanese.
List the phrases where a careful translator's natural, literal reading would be WRONG — only real traps,
the most dangerous first, AT MOST {cap}:
- idioms and figurative phrases ("bring it home", "boil the ocean", "a huge win") — give the meaning HERE;
- an everyday word in a less common sense ("two ways of having local models" = obtaining);
- an elliptical line whose subject / referent is not what it literally says ("I was very out of date
  until I got it" = his old phone was outdated);
- a reply whose function is easy to miss ("Okay, okay." = taking in new information).
Do NOT list: phrases any translator renders correctly anyway (common phrasal verbs like "spin up", "ran
across", "wrap up"), ordinary sentences, technical terms with one clear meaning, names, false starts.

VIDEO BRIEF: {brief}

SENTENCES (id | text):
{rows}

Return ONLY a JSON array: [{{"id": n, "phrase": "<exact words>", "meaning": "<what it means here, max 15 words>"}}]; [] if none."""


def meaning_notes(sents, invoke, cache_path=None, brief: str = "", batch: int = 60) -> dict[int, list[dict]]:
    """{sentence index: [{'phrase', 'meaning'}]} — cached by the sentence texts + brief."""
    key = hashlib.sha1(("\x00".join(s.text for s in sents) + "\x01" + brief).encode("utf-8"),
                       usedforsecurity=False).hexdigest()[:16]
    p = Path(cache_path) if cache_path else None
    if p and p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("key") == key:
                return {int(k): v for k, v in d.get("notes", {}).items()}
        except (json.JSONDecodeError, OSError):
            pass
    notes: dict[int, list[dict]] = {}
    for s0 in range(0, len(sents), batch):
        # a little overlap on each side so a phrase is read in context
        lo, hi = max(0, s0 - 3), min(len(sents), s0 + batch + 3)
        rows = "\n".join(f"{k} | {sents[k].text}" for k in range(lo, hi))
        try:
            raw = invoke(PROMPT.format(brief=brief or "(none)", rows=rows, cap=max(2, (hi - lo) // 6))) or ""
            found = json.JSONDecoder().raw_decode(raw[raw.index("["):])[0]
        except (ValueError, json.JSONDecodeError, TypeError):
            continue
        except Exception:
            continue
        for f in found:
            try:
                k = int(f["id"])
            except (KeyError, TypeError, ValueError):
                continue
            ph, me = str(f.get("phrase", "")).strip(), str(f.get("meaning", "")).strip()
            if not (s0 <= k < min(s0 + batch, len(sents)) and ph and me):
                continue
            if re.sub(r"\W", "", ph.lower()) not in re.sub(r"\W", "", sents[k].text.lower()):
                continue  # the phrase must really be in that sentence
            notes.setdefault(k, []).append({"phrase": ph, "meaning": me})
    if p:
        p.write_text(json.dumps({"key": key, "notes": {str(k): v for k, v in notes.items()}},
                                ensure_ascii=False, indent=1), encoding="utf-8")
    return notes


def meaning_note(i: int, notes) -> str:
    """Prompt suffix for sentence i ('' when it has no note)."""
    items = (notes or {}).get(i) or (notes or {}).get(str(i)) or []
    if not items:
        return ""
    body = "; ".join(f'"{n["phrase"]}" here means: {n["meaning"]}' for n in items)
    return f"\nMEANING NOTE for this line (translate the meaning, not the literal words): {body}"


def meaning_tag(i: int, notes) -> str:
    """Short inline form for the native editor's line table."""
    items = (notes or {}).get(i) or (notes or {}).get(str(i)) or []
    return " [means: " + "; ".join(f'{n["phrase"]} = {n["meaning"]}' for n in items) + "]" if items else ""
