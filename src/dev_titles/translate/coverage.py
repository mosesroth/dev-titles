"""Omission coverage gate + edit-based repair (pipeline step [3.5]).

Research-backed handling of source-content omission ("under-translation"):
  - LLM judges and back-translation detect ABSENCE at near-chance, so we use a
    SOURCE-anchored enumerate-then-verify check: decompose the source into atomic
    propositions and confirm each is realized in the target.
  - Under-translation is the higher-probability decode, so RE-translating the whole
    sentence keeps dropping the same clause. `edit_fix` instead EDITS the existing
    translation to splice the missing content in (validated to repair the "sticky"
    omissions that re-translation and reflect could not).

Both take an `invoke(prompt) -> str` callable (engine-agnostic + unit-testable),
mirroring `eval.qe.GembaMqmScorer`.
"""

from __future__ import annotations

_LANG = {"ja": "Japanese", "zh-Hans": "Simplified Chinese", "zh": "Simplified Chinese"}


def lang_name(code: str) -> str:
    return _LANG.get(code, code)


def coverage_prompt(src: str, tgt: str, tgt_lang_name: str) -> str:
    return (
        f"Decompose the SOURCE English sentence into its atomic propositions — list "
        f"EACH clause/condition/qualifier/fact/number/enumerated item as a separate "
        f"unit (INCLUDING any leading subordinate clause like 'When X,' / 'Unlike Y,'). "
        f"Then, for EACH unit, check whether its meaning is realized in the {tgt_lang_name} "
        f"TARGET. Output one line per unit: 'PRESENT: <unit>' or 'MISSING: <unit>'. "
        f"Only count SUBSTANTIVE content (facts, conditions, technical detail, "
        f"identifiers, enumerated items). Do NOT flag as missing: pronouns, articles, "
        f"intensifiers, greetings, or discourse filler (welcome / a little bit / "
        f"happy coding / you know) — a subtitler is expected to condense those. "
        f"After the list, "
        f"if every unit is PRESENT output a final line 'COMPLETE', else 'INCOMPLETE'.\n\n"
        f"SOURCE: {src}\n\nTARGET ({tgt_lang_name}): {tgt}"
    )


def parse_missing(verdict: str) -> str:
    """Extract the substantive missing units from a coverage verdict (empty if
    complete). Joins them with '; ' for use as an edit-fix instruction."""
    if "INCOMPLETE" not in verdict.upper() and "MISSING:" not in verdict.upper():
        return ""
    miss = [
        ln.split("MISSING:", 1)[1].strip()
        for ln in verdict.splitlines()
        if "MISSING:" in ln.upper()[:9]
    ]
    return "; ".join(m for m in miss if m)


def coverage_missing(invoke, src: str, tgt: str, tgt_lang_name: str) -> str:
    """Substantive source content NOT realized in `tgt` (empty string if complete)."""
    return parse_missing(invoke(coverage_prompt(src, tgt, tgt_lang_name)))


def edit_fix_prompt(src: str, cur: str, missing: str, tgt_lang_name: str) -> str:
    return (
        f"You are editing a {tgt_lang_name} subtitle translation that DROPPED part of "
        f"the source. Do NOT retranslate from scratch — EDIT the existing translation "
        f"to ADD the missing content, keeping what is already correct. Completeness "
        f"matters more than brevity: it is fine to make it longer (it will be split "
        f"into more cues). Output ONLY the corrected {tgt_lang_name}.\n\n"
        f"ENGLISH SOURCE: {src}\n\nCURRENT {tgt_lang_name} (incomplete): {cur}\n\n"
        f"MISSING CONTENT to add: {missing}"
    )


def edit_fix(invoke, src: str, cur: str, missing: str, tgt_lang_name: str) -> str:
    """Repair an omission by editing `cur` to splice in `missing`. Returns the
    corrected target text."""
    return invoke(edit_fix_prompt(src, cur, missing, tgt_lang_name)).strip()
