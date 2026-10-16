"""Cross-video term-pin ledger + auto-promotion to the shared glossary.

Per-video QE produces term-pins via `harvest_term_pins`; those are already the
SAFE class — `classify_term_fix` returns None for sense/judgment/style calls, so
only deterministic term fixes (wrong->right rendering of a name/brand/identifier/
term) ever become pins. A single video's pin is NOT promoted, though: a one-off
could be video-specific or a stochastic judge whim (the `agent`->特工 class).

We instead APPEND every applied pin to a cross-video ledger, then promote a term
to the shared glossary only when it is:
  1. CORROBORATED across >= `min_videos` DISTINCT videos, AND
  2. CONSISTENT — every video agreed on the same `correct` rendering (a conflict
     means the sense is video-dependent -> leave it for the native reviewer), AND
  3. NEW — not already owned by the curated glossary (that decision always wins).

The actual glossary append (skip-existing, contains-curated-word guard, comment-
preserving text insert) is reused from `promote_renderings_to_glossary`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def record_pins(ledger_path: str | Path, pins: list[dict], lang: str, video: str) -> int:
    """Append this video's applied term-pins to the cross-video ledger (JSONL).

    One line per pin: {source, wrong, correct, lang, video}. Deduped within this
    (video, lang) so re-running a video doesn't inflate the corroboration count.
    Returns the number of new lines written.
    """
    if not pins or not video:
        return 0
    p = Path(ledger_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple] = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            seen.add((d.get("video"), d.get("lang"), (d.get("source") or "").lower()))
    added = 0
    with p.open("a", encoding="utf-8") as fh:
        for pin in pins:
            src = (pin.get("source") or "").strip()
            correct = (pin.get("correct") or "").strip()
            if not src or not correct:
                continue
            key = (video, lang, src.lower())
            if key in seen:
                continue
            seen.add(key)
            fh.write(
                json.dumps(
                    {
                        "source": src,
                        "wrong": (pin.get("wrong") or "").strip(),
                        "correct": correct,
                        "lang": lang,
                        "video": video,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            added += 1
    return added


def promotion_renderings(ledger_path: str | Path, *, min_videos: int = 2) -> dict:
    """Read the ledger and return the `renderings` dict for terms corroborated across
    >= `min_videos` distinct videos with a CONSISTENT rendering per language:

        {lang: {source: {"canonical": correct, "avoid": [wrong...], "approved": True}}}

    A (source, lang) whose videos disagree on `correct` is dropped (sense is
    video-dependent -> reviewer's call). Each language is judged independently, so a
    term may promote its zh rendering while its ja stays pending.
    """
    p = Path(ledger_path)
    if not p.exists():
        return {}
    # (source_lower, lang) -> {"src": display, "videos": {video: correct}, "avoid": set}
    acc: dict[tuple, dict] = defaultdict(lambda: {"src": "", "videos": {}, "avoid": set()})
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        src, lang, correct = d.get("source"), d.get("lang"), d.get("correct")
        video = d.get("video")
        if not (src and lang and correct and video):
            continue
        rec = acc[(src.lower(), lang)]
        rec["src"] = rec["src"] or src
        rec["videos"][video] = correct  # last write per video wins (idempotent per video)
        if d.get("wrong"):
            rec["avoid"].add(d["wrong"])

    renderings: dict[str, dict] = {}
    for (_src_l, lang), rec in acc.items():
        votes = rec["videos"]
        if len(votes) < min_videos:
            continue  # not recurring yet
        renderings_seen = set(votes.values())
        if len(renderings_seen) != 1:
            continue  # videos disagree -> sense-dependent, leave for the reviewer
        correct = next(iter(renderings_seen))
        avoid = [a for a in rec["avoid"] if a and a != correct]
        renderings.setdefault(lang, {})[rec["src"]] = {
            "canonical": correct,
            "avoid": avoid,
            "approved": True,
        }
    return renderings


def promote_recurring_terms(
    ledger_path: str | Path,
    glossary_path: str | Path,
    *,
    min_videos: int = 2,
    date: str | None = None,
    logger=None,
) -> list[str]:
    """Promote every corroborated recurring term (see `promotion_renderings`) into the
    shared glossary. Reuses `promote_renderings_to_glossary` for the actual append (so
    the skip-existing / curated-word guards apply). Returns the promoted source terms.
    """
    from .vocab_prep import promote_renderings_to_glossary

    renderings = promotion_renderings(ledger_path, min_videos=min_videos)
    if not renderings:
        if logger:
            logger("promote: no terms corroborated across enough videos yet")
        return []
    promoted = promote_renderings_to_glossary(
        renderings,
        glossary_path,
        only_approved=True,
        source_video=f"recurring term-pins (>={min_videos} videos)",
        date=date or "",
    )
    if logger:
        logger(
            f"promote: {len(promoted)} recurring term(s) added to glossary: {', '.join(promoted)}"
            if promoted
            else "promote: corroborated terms already in glossary — nothing new"
        )
    return promoted
