"""SPEAKER TURNS: who says each English sentence.

Translation used to see one undifferentiated stream, so in an interview a reply came out as the
same person continuing: "Brilliant. Thank you for having me." (the guest) -> 素晴らしいです…
instead of こちらこそ…, and "We're excited to have you here" (to the guest) -> 皆さん…. Native
reviewers flag it as "the speaker was not correctly identified".

Sources, best first:
  1. ASR speaker labels (AssemblyAI diarization) saved on the words in raw.json — majority vote
     of the words inside each sentence's time window;
  2. otherwise one LLM pass over the sentences, guided by the video brief (names and roles).
A single-speaker video returns [] (nothing to do). Cached in <out>/speakers.json.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

PROMPT = """You label who is speaking each line of a video transcript, for subtitle translators.

VIDEO BRIEF (speakers, roles): {brief}

Rules:
- Use the speakers' names from the brief with their role, e.g. "Jane (guest)", "Sam (host)".
- Turns change often and mid-caption: a short reaction or reply ("Yeah.", "Sure.", "Brilliant.
  Thank you for having me.", "No problem.") is usually the OTHER person. A question is usually
  the host; the answer that follows is the guest.
- If the whole video is one person talking, give every line that one name.
{previous}
LINES:
{lines}

Return ONLY a JSON array with one speaker name per line, in order, e.g. ["Sam (host)", "Jane (guest)", ...]."""


def _key(sents, brief):
    h = hashlib.sha1(("\x00".join(s.text for s in sents) + "\x01" + (brief or "")).encode("utf-8"), usedforsecurity=False)
    return h.hexdigest()[:16]


def from_asr(sents, words) -> list[str]:
    """Majority speaker of the ASR words inside each sentence window ('' if none)."""
    out = []
    for s in sents:
        c = Counter(w.get("speaker") for w in words
                    if w.get("speaker") and s.start - 0.05 <= (w["start"] + w["end"]) / 2 <= s.end + 0.05)
        out.append(c.most_common(1)[0][0] if c else "")
    # fill gaps from the neighbor (a sentence whose window caught no word)
    for i, v in enumerate(out):
        if not v:
            out[i] = out[i - 1] if i else next((x for x in out if x), "")
    return out


def from_llm(sents, invoke, brief="", chunk=40) -> list[str]:
    labels: list[str] = []
    for k in range(0, len(sents), chunk):
        part = sents[k : k + chunk]
        prev = ""
        if labels:
            tail = list(zip(sents[max(0, k - 4) : k], labels[-4:]))
            prev = "\nPREVIOUS LINES (already labeled):\n" + "\n".join(f"[{sp}] {s.text}" for s, sp in tail) + "\n"
        lines = "\n".join(f"{n + 1}. {s.text}" for n, s in enumerate(part))
        got: list[str] = []
        for _ in range(2):
            try:
                m = re.search(r"\[.*\]", invoke(PROMPT.format(brief=brief or "(none)", previous=prev, lines=lines)) or "", re.S)
                got = [str(x).strip() for x in json.loads(m.group(0))] if m else []
            except Exception:
                got = []
            if len(got) == len(part):
                break
        if len(got) != len(part):  # unusable answer: carry the last known speaker, mark unknown
            got = [labels[-1] if labels else ""] * len(part)
        labels += got
    return labels


def label_speakers(sents, invoke=None, cache_path=None, brief="", raw_path=None, logger=print) -> list[str]:
    """One speaker label per sentence, or [] when there is only one speaker (or nothing to go on)."""
    if not sents:
        return []
    key = _key(sents, brief)
    p = Path(cache_path) if cache_path else None
    if p and p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("key") == key:
                return d.get("labels", [])
        except (json.JSONDecodeError, OSError):
            pass
    labels, source = [], ""
    words = []
    if raw_path and Path(raw_path).exists():
        try:
            words = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            words = []
    if len({w.get("speaker") for w in words if isinstance(w, dict) and w.get("speaker")}) >= 2:
        labels, source = from_asr(sents, words), "ASR diarization"
    elif invoke is not None:
        labels, source = from_llm(sents, invoke, brief), "LLM"
    if len({x for x in labels if x}) < 2:
        labels = []  # one speaker: nothing to tell the translator
    if p:
        p.write_text(json.dumps({"key": key, "source": source, "labels": labels}, ensure_ascii=False), encoding="utf-8")
    if labels:
        turns = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
        logger(f"speakers ({source}): {len(set(labels))} speakers, {turns} turn changes over {len(labels)} sentences")
    return labels


def labeled_transcript(sents, labels) -> str:
    """Transcript with a [speaker] tag at each turn change (for the translation context)."""
    if not labels or len(labels) != len(sents):
        return " ".join(s.text for s in sents)
    out, prev = [], None
    for s, sp in zip(sents, labels):
        out.append(f"[{sp}] {s.text}" if sp != prev else s.text)
        prev = sp
    return " ".join(out)


def speaker_note(i, labels) -> str:
    """Per-line note telling the translator who speaks this line and whether it's a reply."""
    if not labels or i >= len(labels) or not labels[i]:
        return ""
    note = f"\nSPEAKER OF THE LINE TO TRANSLATE: {labels[i]}."
    if i > 0 and labels[i - 1] and labels[i - 1] != labels[i]:
        note += (f" The previous line was said by {labels[i - 1]} — this line is a different person "
                 "(a reply or reaction: render who says what, and to whom, accordingly).")
    return note
