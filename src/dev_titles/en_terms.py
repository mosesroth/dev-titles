"""Approved English spelling of product and technology names (glossary/en_terms.yaml).

Built by scripts/build_en_terms.py from the glossary, an optional term base (glossary/termbase.yaml),
the documentation names harvested per video and the partner list. The English review applies it before
its own proofread: a case or spacing variant of an approved name ("Android tv", "AndroidTV", "react
native", "visual studio code") becomes the approved form, and each change is listed in en_review.md.
Only the same letters change (plus `aliases`, variant -> approved); a name glued into code
("react-native-video", "@scope/react-native") is left alone.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .model import Cue, Subtitle


def load(path=None) -> tuple[list[str], dict[str, str]]:
    from .glossary import glossary_dir

    p = Path(path) if path else glossary_dir() / "en_terms.yaml"
    if not p.exists():
        return [], {}
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    terms = [t["term"] for t in d.get("terms") or [] if t.get("term")]
    return terms, dict(d.get("aliases") or {})


def _pattern(term: str) -> re.Pattern:
    toks = [re.escape(t) for t in term.split()]
    sep = r"[\s\-]?"  # "Android TV" also matches "AndroidTV" / "Android-TV"
    return re.compile(rf"(?<![\w\-./@]){sep.join(toks)}(?![\w\-/@]|\.[A-Za-z])", re.I)


def normalize(
    text: str, terms: list[str], aliases: dict[str, str]
) -> tuple[str, list[tuple[str, str]]]:
    """(text with approved spellings, [(found, approved)])."""
    changes = []
    for variant, approved in aliases.items():
        pat = re.compile(rf"(?<![\w\-./@]){re.escape(variant)}(?![\w\-/@])", re.I)
        for m in pat.finditer(text):
            changes.append((m.group(0), approved))
        text = pat.sub(approved, text)
    for term in sorted(
        terms, key=len, reverse=True
    ):  # longest first: "Android TV Box" before "Android TV"
        pat = _pattern(term)

        def fix(m, term=term):
            if "-" in m.group(0) and m.group(0).islower():  # a package name: react-native-video
                return m.group(0)
            if m.group(0) != term:
                changes.append((m.group(0), term))
            return term

        text = pat.sub(fix, text)
    return text, changes


def apply(en: Subtitle, terms=None, aliases=None, max_cpl: int = 42) -> tuple[Subtitle, list[dict]]:
    """Normalize every cue; returns (subtitle, [{cue, found, approved}])."""
    if terms is None:
        terms, aliases = load()
    if not terms and not aliases:
        return en, []
    out, log = [], []
    for k, c in enumerate(en.cues):
        lines = []
        for ln in c.lines:
            new, ch = normalize(ln, terms, aliases or {})
            lines.append(new)
            log += [{"cue": k, "found": a, "approved": b} for a, b in ch]
        if lines != list(c.lines) and any(len(x) > max_cpl for x in lines):
            # a longer approved spelling ('Vatoo' -> 'Vantoo') can push a line over the limit:
            # re-wrap the cue (three 43-char lines once blocked the gate this way)
            from .en_cleanread import wrap2

            rewrapped = wrap2(" ".join(lines), max_cpl)
            if len(rewrapped) <= 2 and all(len(x) <= max_cpl for x in rewrapped):
                lines = rewrapped
        out.append(Cue(c.start, c.end, lines, getattr(c, "index", None)))
    return (Subtitle(en.language, out) if log else en), log
