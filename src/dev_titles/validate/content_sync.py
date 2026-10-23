"""Translated (zh / ja) subtitle timing vs the SPEECH, by content.

validate/sync.py compares a translation to the EN track, so a translation re-timed to a wrong EN
track passes. Latin-name anchors against the ASR don't work either: auto-captions mis-hear exactly
the names (TanStack -> "kac"). So this anchors by MEANING:

  1. each EN sentence gets its SPOKEN time window: its words aligned to word-timed speech
     (speech_sync); only the EN text is used, never the EN track's cue timing;
  2. one batched LLM call maps each target cue to the EN sentence(s) it translates (cached);
  3. a cue is off when its whole on-screen span lies more than `tol` s away from the window in
     which its content is spoken.

Catches a whole cue (or run of cues) shown at the wrong time — e.g. a stale zh track 8 s behind. A cue that only splits a sentence differently is not flagged.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path

from .speech_sync import _norm

TOL_ERROR = 2.0
TOL_WARN = 1.0

PROMPT = """Numbered English sentences from a video, with the times they are spoken, and numbered {lang}
subtitle cues that translate them, in order. For EVERY cue, give the id of the first and the last English
sentence whose meaning that cue carries (one sentence: the same id twice). Cues are in order; a cue may carry
part of a sentence, or several. A cue with no English counterpart (an interjection): [null, null].

ENGLISH:
{en}

{lang} CUES:
{tgt}

Return ONLY JSON: {{"<cue id>": [first_sentence_id, last_sentence_id], ...}} covering every cue id."""


def sentence_speech_windows(sents, speech) -> dict[int, tuple[float, float]]:
    """EN sentence index -> (spoken start, spoken end), from words aligned to the speech (runs of
    >= 2 matching words). Sentences with no aligned word are left out."""
    toks: list[tuple[str, int]] = []
    for i, s in enumerate(sents):
        toks += [(t, i) for t in (_norm(w) for w in s.text.split()) if t]
    sm = difflib.SequenceMatcher(None, [t for t, _ in toks], [w for w, _ in speech], autojunk=False)
    times: dict[int, list[float]] = {}
    for blk in sm.get_matching_blocks():
        if blk.size < 2:
            continue
        for d in range(blk.size):
            times.setdefault(toks[blk.a + d][1], []).append(speech[blk.b + d][1])
    return {i: (min(t), max(t) + 0.4) for i, t in times.items()}


def _map_cues(
    target, sents, windows, invoke, lang_name, cache_path, chunk: int = 40
) -> dict[int, tuple]:
    key = hashlib.sha256(
        json.dumps(
            [[c.text for c in target.cues], [s.text for s in sents]], ensure_ascii=False
        ).encode()
    ).hexdigest()[:16]
    cp = Path(cache_path) if cache_path else None
    if cp and cp.exists():
        rec = json.loads(cp.read_text(encoding="utf-8"))
        if rec.get("key") == key:
            return {int(k): tuple(v) for k, v in rec["map"].items()}
    mapping: dict[int, tuple] = {}
    cues = target.cues
    for a in range(0, len(cues), chunk):
        part = cues[a : a + chunk]
        lo, hi = part[0].start - 45, part[-1].end + 45
        ids = [i for i, s in enumerate(sents) if s.end >= lo and s.start <= hi]
        if not ids:
            continue
        en = "\n".join(
            (
                f"{i}: [{windows[i][0]:.0f}s] {sents[i].text}"
                if i in windows
                else f"{i}: {sents[i].text}"
            )
            for i in ids
        )
        tgt = "\n".join(
            f"{a + k}: [{c.start:.0f}s] {' '.join(c.lines)}" for k, c in enumerate(part)
        )
        m = re.search(r"\{.*\}", invoke(PROMPT.format(lang=lang_name, en=en, tgt=tgt)) or "", re.S)
        try:
            got = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            got = {}
        for k, v in got.items():
            if isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) for x in v):
                mapping[int(k)] = (v[0], v[1])
    if cp:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps({"key": key, "map": mapping}), encoding="utf-8")
    return mapping


def content_speech_desync(target, en, speech, invoke, *, lang_name: str, cache_path=None) -> dict:
    """{'checked', 'off': [{start, gap, severity, text, en}]} for target cues shown away from when
    their content is spoken. `en` supplies the EN sentences (text); timing comes from `speech`."""
    from ..resegment import merge_en_to_sentences

    sents = merge_en_to_sentences(en)
    windows = sentence_speech_windows(sents, speech)
    mapping = _map_cues(target, sents, windows, invoke, lang_name, cache_path)
    off, checked = [], 0
    for k, c in enumerate(target.cues):
        a, b = mapping.get(k, (None, None))
        if a is None or b is None or a not in windows or b not in windows or b < a:
            continue
        s0, s1 = windows[a][0], windows[b][1]
        if s1 - s0 > 30:  # implausibly wide mapping: don't judge
            continue
        checked += 1
        gap = max(s0 - c.end, c.start - s1, 0.0)
        if gap > TOL_WARN:
            off.append(
                {
                    "cue": k + 1,
                    "start": round(c.start, 2),
                    "gap": round(gap, 2),
                    "late": c.start > s1,
                    "severity": "error" if gap > TOL_ERROR else "warn",
                    "text": " ".join(c.lines)[:40],
                    "en": sents[a].text[:60],
                    "spoken": round(s0, 2),
                }
            )
    # ERRORS come in RUNS: a stale / drifted stretch moves several consecutive cues the same way
    # (a stale zh track: 33). A single cue off is usually a mapping artefact — a dangling verb
    # (行っています) or Japanese word order spanning a sentence — so alone it is only a warning.
    by_cue = {o["cue"]: o for o in off}
    for o in off:
        if o["severity"] != "error":
            continue
        nb = [by_cue.get(o["cue"] - 1), by_cue.get(o["cue"] + 1)]
        if not any(n and n["gap"] > TOL_ERROR and n["late"] == o["late"] for n in nb):
            o["severity"] = "warn"
    return {"checked": checked, "mapped": len(mapping), "off": off}
