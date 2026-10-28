"""Glossary: load, retrieve per-segment, and hard-enforce terms.

- `load()` reads the YAML schema in glossary/.
- `retrieve()` returns the terms whose source/alias appears in a segment; those
  get injected into the translation prompt.
- `enforce()` deterministically overrides the model's rendering for `hard` terms
  (used after translation, step [5] in the pipeline).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - surfaced at load time
    yaml = None


@dataclass
class Term:
    source: str
    translations: dict[str, str] = field(default_factory=dict)  # lang -> rendering
    aliases: list[str] = field(default_factory=list)
    enforce: str = "soft"  # "hard" | "soft"
    notes: str = ""
    senses: list[dict] = field(default_factory=list)  # [{when, <lang>...}] context variants
    avoid: dict[str, list[str]] = field(default_factory=dict)  # lang -> wrong forms
    case_sensitive: bool = False  # match case-sensitively — for a proper-noun DNT that is a
    # common word when lowercased ('Bolt' the product vs the verb
    # 'bolt'/'bolt-on'), so it doesn't false-match the word

    @property
    def all_source_forms(self) -> list[str]:
        return [self.source, *self.aliases]

    def sense_hint(self, lang: str) -> str:
        """Human-readable sense guidance for the prompt (empty if no senses)."""
        parts = []
        for s in self.senses:
            r = s.get(lang)
            if r and s.get("when"):
                parts.append(f'{r} (when {s["when"]})')
        return "; ".join(parts)


@dataclass
class EnforcementIssue:
    term: str  # canonical English source
    lang: str
    kind: str  # "fixed" | "missing"
    expected: str  # canonical target rendering
    detail: str = ""


@dataclass
class Glossary:
    version: str
    languages: list[str]
    terms: list[Term]
    style: dict[str, dict] = field(default_factory=dict)
    heuristics: str = ""  # localization policy prose, injected into prompts
    identifiers: list[str] = field(
        default_factory=list
    )  # DNT code/API names (ASR correction + keep-Latin)

    def retrieve(self, segment: str) -> list[Term]:
        """Terms present in `segment`, with longest-match precedence so an
        overlapping shorter term is suppressed (e.g. "Acme Cloud" wins over
        "Acme" for the same span). Latin forms match on word boundary
        (case-insensitive); CJK/other forms match as substrings.

        DNT identifiers are masked first so a term is NOT matched INSIDE one — else
        the 'Acme'->艾克米 term matches the 'Acme' in the product name 'Acme Q'
        and gets required/enforced there (false 'expected 艾克米' flags)."""
        seg = self._mask_identifiers(segment)
        return [term for term, _ in _matched_spans(seg, self.terms)]

    def _mask_identifiers(self, text: str) -> str:
        """Blank out DNT identifiers (Acme Q, Nimbus SDK, …) AND web domains so term matching
        can't fire on a term embedded in one — else the 'Acme'->艾克米 term matches the
        'acme' inside 'developer.acme.example' and demands a canonical rendering the (correctly
        verbatim) domain doesn't have. Only used for MATCHING (not output)."""
        from .qa import detect_domains

        masked, _ = _mask_identifier_spans(text, self.identifiers)
        masked, _ = _mask_protected(masked, detect_domains(masked))
        return masked

    def enforce(
        self, target: str, source: str, lang: str, placeholders: dict | None = None
    ) -> tuple[str, list[EnforcementIssue]]:
        """Deterministically enforce `hard` terms in a translated `target`,
        given the `source` segment it was translated from.

        For each hard term present in `source` (longest-match), the canonical
        `target` rendering for `lang` must appear in `target`. If a known-wrong
        form is present instead — the English source/alias left untranslated, or
        the *other* language's rendering leaking in — it is replaced with the
        canonical rendering (recorded as kind="fixed"). If the canonical rendering
        is still absent afterward, it is flagged kind="missing" (the model likely
        dropped or paraphrased the term — route to the refine loop / human).
        Replacements never fire for DNT terms whose canonical form equals the
        source, so Latin brand names are left untouched.

        `placeholders` {token: span}: the caller's own DNT masks still in `target` (the
        translate pipeline enforces BEFORE unmasking). Presence checks resolve them, else a
        masked DNT inside a canonical ([[0]]虚拟设备 for Orbit虚拟设备) reads as 'missing'.

        Returns (possibly_fixed_target, issues).
        """
        issues: list[EnforcementIssue] = []
        # PROTECT CLI commands / DNT identifiers from substring substitution: an
        # 'app'->アプリ term otherwise rewrites the 'app' inside 'create-expo-app'
        # (-> create-expo-アプリ). Mask them out of `target` for the whole enforce pass,
        # restore verbatim at the end. This shields EVERY caller (translate_segment,
        # the coverage edit-fix path, etc.), not just the main translate path.
        from .qa import protected_spans

        target, _imask = _mask_identifier_spans(target, self.identifiers)
        target, _pmask = _mask_protected(target, protected_spans(target), base=len(_imask))
        _pmask = {**_imask, **_pmask}

        def _has(canonical: str) -> bool:
            if canonical in target:
                return True
            t = _unmask_protected(target, _pmask)
            for tok, span in (placeholders or {}).items():
                t = t.replace(tok, span)
            return canonical in t

        # Avoid-form correction, ASCII forms only, is TARGET-DRIVEN: an English wrong-
        # form ('Strict Mode', 'Model Context Protocol') is safe to fix wherever it
        # appears (lookaround-bounded, can't corrupt a CJK word), and it must be — a
        # later sentence can carry it where the source used a pronoun. CJK avoid-forms
        # stay SOURCE-GATED below: applying a short/ambiguous CJK form target-wide
        # corrupted unrelated words (発生->発ネイティブ) — a per-video LLM avoid-form.
        for term in self.terms:
            canonical = term.translations.get(lang, "").strip()
            if not canonical:
                continue
            for bad in term.avoid.get(lang, []):
                if bad and bad != canonical and bad.isascii() and bad in target:
                    target = _replace_form(target, bad, canonical)
                    issues.append(
                        EnforcementIssue(
                            term.source,
                            lang,
                            "fixed",
                            canonical,
                            f"replaced avoid-form {bad!r} -> {canonical!r}",
                        )
                    )
        present_all = [t for t, _ in _matched_spans(self._mask_identifiers(source), self.terms)]
        for term in present_all:
            canonical = term.translations.get(lang, "").strip()
            # CJK avoid-forms: source-gated (only when this segment's source names the
            # term) — safe from the global-corruption failure mode above.
            if canonical:
                for bad in term.avoid.get(lang, []):
                    if bad and bad != canonical and not bad.isascii() and bad in target:
                        target = _replace_form(target, bad, canonical)
                        issues.append(
                            EnforcementIssue(
                                term.source,
                                lang,
                                "fixed",
                                canonical,
                                f"replaced avoid-form {bad!r} -> {canonical!r}",
                            )
                        )
                # Leftover ENGLISH source form in the target of a TRANSLATABLE (CJK-canonical)
                # term = untranslated -> replace with canonical. Applies to SOFT terms too (the
                # hard block below already does this, hard-only); fixes e.g. the host title
                # 'Chief Evangelist' left verbatim in English. Source-gated + word-bounded +
                # DNT-masked, so it's safe; never fires for DNT terms (their canonical == source
                # is Latin, so `not canonical.isascii()` is False).
                if not canonical.isascii() and not _has(canonical):
                    for form in sorted(term.all_source_forms, key=len, reverse=True):
                        if (
                            form
                            and form.isascii()
                            and _replace_latin_run(target, form, canonical) != target
                        ):
                            target = _replace_latin_run(target, form, canonical)
                            issues.append(
                                EnforcementIssue(
                                    term.source,
                                    lang,
                                    "fixed",
                                    canonical,
                                    f"translated leftover English {form!r} -> {canonical!r}",
                                )
                            )
                            break
            # Full canonical enforcement (force-insert / missing-flag) stays
            # HARD-only: soft terms are guidance, not mandatory renderings.
            if term.enforce != "hard" or not canonical:
                continue
            if _has(canonical):
                continue  # already correct
            # Present but with a stray Latin<->CJK boundary space -> normalize it.
            normalized = _normalize_term_spacing(target, canonical)
            if normalized != target:
                target = normalized
                issues.append(
                    EnforcementIssue(
                        term.source, lang, "fixed", canonical, "normalized boundary spacing"
                    )
                )
                continue
            # Candidate wrong forms to rewrite -> canonical.
            wrong: set[str] = set(term.all_source_forms)
            for other_lang, rendering in term.translations.items():
                if other_lang != lang and rendering:
                    wrong.add(rendering)
            wrong = {w for w in wrong if w and w != canonical}
            # Longest wrong form first so we don't partially clobber.
            for form in sorted(wrong, key=len, reverse=True):
                new_target = _replace_form(target, form, canonical)
                if new_target != target:
                    target = new_target
                    issues.append(
                        EnforcementIssue(
                            term.source,
                            lang,
                            "fixed",
                            canonical,
                            f"replaced {form!r} -> {canonical!r}",
                        )
                    )
                    if _has(canonical):
                        break
            if not _has(canonical):
                issues.append(
                    EnforcementIssue(
                        term.source,
                        lang,
                        "missing",
                        canonical,
                        "canonical rendering not found in translation",
                    )
                )
        return _unmask_protected(target, _pmask), issues


def _mask_protected(text: str, spans: list[str], base: int = 0) -> tuple[str, dict]:
    """Replace CLI-command / identifier spans with private-use sentinels so enforce's
    substring substitutions can't corrupt them (longest first). No glossary form or
    canonical rendering contains U+E000..U+E001, so the sentinels are inert."""
    mask: dict[str, str] = {}
    for i, s in enumerate(sorted(set(spans), key=len, reverse=True)):
        if s and s in text:
            token = f"{base + i}"
            text = text.replace(s, token)
            mask[token] = s
    return text, mask


def _unmask_protected(text: str, mask: dict) -> str:
    for token, s in mask.items():
        text = text.replace(token, s)
    return text


def _mask_identifier_spans(text: str, idents: list[str]) -> tuple[str, dict]:
    """Mask DNT identifiers, but only on alphanumeric boundaries so an identifier
    that is a PREFIX of a longer word isn't masked inside it — e.g. the company
    'Juli' must NOT match the 'Juli' in a person's name 'Julie Chen' (a plain
    substring replace turned 'Julie Chen' -> '<tok>e Chen' and mangled the name
    sent to the translator). Lookarounds (not \\b) so identifiers with punctuation
    edges like 'C++' or 'UIManagerBinding::dispatchEvent' still match. Longest
    first so a multi-word identifier wins over a contained shorter one."""
    mask: dict[str, str] = {}
    for s in sorted(set(x for x in idents if x), key=len, reverse=True):
        pat = r"(?<![A-Za-z0-9])" + re.escape(s) + r"(?![A-Za-z0-9])"
        if re.search(pat, text):
            token = f"{len(mask)}"
            text = re.sub(pat, token, text)
            mask[token] = s
    return text, mask


def _is_latin(form: str) -> bool:
    return bool(re.search(r"[A-Za-z]", form))


def _is_acronym(form: str) -> bool:
    """All-caps alphabetic form (ASK, IAP, API, SDK, AWS) — match case-sensitively
    so the acronym 'ASK' doesn't match the verb 'ask'."""
    return form.isupper() and any(c.isalpha() for c in form)


def _find_spans(haystack: str, needle: str, case_sensitive: bool = False) -> list[tuple[int, int]]:
    """All (start, end) occurrences of `needle` in `haystack`. Latin needles
    match on a word boundary — case-sensitive for all-caps acronyms or when
    `case_sensitive` is set (a proper-noun DNT colliding with a common word), else
    case-insensitive; non-Latin needles match as substrings."""
    if not needle:
        return []
    if _is_latin(needle):
        # Allow a trailing English inflection (plural/verb) so 'applications' and
        # 'publishes' match 'application'/'publish' — exact-form retrieval let
        # inflected words evade the glossary (audit finding). Acronyms stay exact.
        suffix = "" if _is_acronym(needle) else r"(?:s|es|ed|ing|'s)?"
        pat = r"\b" + re.escape(needle) + suffix + r"\b"
        flags = 0 if (_is_acronym(needle) or case_sensitive) else re.IGNORECASE
        return [(m.start(), m.end()) for m in re.finditer(pat, haystack, flags)]
    spans, start = [], haystack.find(needle)
    while start != -1:
        spans.append((start, start + len(needle)))
        start = haystack.find(needle, start + 1)
    return spans


def _matched_spans(segment: str, terms: list[Term]) -> list[tuple[Term, tuple[int, int]]]:
    """Match terms in `segment` with longest-match, non-overlapping precedence.

    Returns one (term, span) per accepted span, each term at most once, longer
    source forms winning any overlap (so "Acme Cloud" suppresses "Acme")."""
    candidates: list[tuple[int, int, int, Term]] = []  # (len, start, end, term)
    seg_low = segment.lower()  # cheap prefilter: with ~1.5k TB terms,
    for term in terms:  # skip the per-form regex for any form not
        for form in term.all_source_forms:  # even present as a substring (99% of terms)
            if not form or form.lower() not in seg_low:
                continue
            for s, e in _find_spans(segment, form, case_sensitive=term.case_sensitive):
                candidates.append((e - s, s, e, term))
    # Longest span first; accept if it doesn't overlap an already-accepted span.
    candidates.sort(key=lambda c: (-c[0], c[1]))
    accepted: list[tuple[Term, tuple[int, int]]] = []
    taken: list[tuple[int, int]] = []
    seen: set[int] = set()
    for _, s, e, term in candidates:
        if any(s < te and e > ts for (ts, te) in taken):
            continue
        taken.append((s, e))
        if id(term) not in seen:
            accepted.append((term, (s, e)))
            seen.add(id(term))
    return accepted


def _is_cjk(ch: str) -> bool:
    if not ch:
        return False
    import unicodedata

    return unicodedata.east_asian_width(ch) in ("W", "F")


def _boundary(a: str, b: str) -> bool:
    """True at a Latin<->CJK boundary (where models stray a space)."""
    return (_is_cjk(a) and b.isascii() and b.isalnum()) or (
        _is_cjk(b) and a.isascii() and a.isalnum()
    )


def _flexible_pattern(canonical: str) -> str:
    """Regex matching `canonical` but tolerating optional spaces at Latin<->CJK
    boundaries (e.g. 'Acme技能' also matches 'Acme 技能')."""
    out: list[str] = []
    for i, ch in enumerate(canonical):
        out.append(re.escape(ch))
        if i + 1 < len(canonical) and _boundary(ch, canonical[i + 1]):
            out.append(r"[ 　]*")
    return "".join(out)


def _normalize_term_spacing(text: str, canonical: str) -> str:
    """If `canonical` appears in `text` modulo boundary spaces, rewrite those
    occurrences to the exact canonical form. Returns text unchanged otherwise."""
    if any(_boundary(a, b) for a, b in zip(canonical, canonical[1:])):
        return re.sub(_flexible_pattern(canonical), canonical, text)
    return text


# A Latin form counts as "left untranslated" only when it is a WHOLE Latin run, not one word of
# a longer Latin name: 'TV' in 'Orbit TV' / 'Multi-TV', 'Acme' in 'Acme Developer Tools',
# 'Hello' in 'Hello World' are parts of kept-Latin names (else Multi-TV -> Multi-电视).
LATIN_RUN_EDGE_BEFORE = r"(?<![A-Za-z0-9])(?<![A-Za-z0-9][ \-])"
LATIN_RUN_EDGE_AFTER = r"(?![A-Za-z0-9])(?![ \-][A-Za-z0-9])"


def latin_run_pattern(form: str) -> str:
    return LATIN_RUN_EDGE_BEFORE + re.escape(form) + LATIN_RUN_EDGE_AFTER


def _replace_latin_run(text: str, form: str, replacement: str) -> str:
    """Translate a leftover English `form` only where it stands alone as a whole Latin run."""
    if not form or form == replacement:
        return text
    return re.sub(latin_run_pattern(form), replacement, text, flags=re.IGNORECASE)


def _replace_form(text: str, form: str, replacement: str) -> str:
    """Replace occurrences of `form` with `replacement`. PURELY-ASCII forms use a
    word-boundary, case-insensitive regex (so 'Strict Mode' doesn't match inside a
    longer word); any form containing CJK/kana uses plain substring replace — the
    `\\b` regex fails mid-CJK (e.g. 'Strictモード' sits between two word chars, so
    the boundaries never match and the avoid-form silently survived)."""
    if not form or form == replacement:
        return text
    if form.isascii() and _is_latin(form):
        # ASCII word-boundaries via lookarounds, NOT \b: \b fails at an ASCII<->CJK
        # junction ('Strict Mode' + 'は'), so an English avoid-form embedded in
        # Japanese survived. Lookarounds match there yet still won't fire inside a
        # longer Latin word (StrictModeX).
        return re.sub(
            r"(?<![A-Za-z0-9])" + re.escape(form) + r"(?![A-Za-z0-9])",
            replacement,
            text,
            flags=re.IGNORECASE,
        )
    # CJK substring replace with SEAM-DEDUP. A short avoid-form can be a prefix of a longer
    # compound whose remainder starts with the canonical's final char, so a naive replace
    # DUPLICATES that char ('网页端' + 网页->Web端  =>  'Web端端'). Collapse the duplicate at the
    # seam. A legitimate standalone fix ('网页上' -> 'Web端上') leaves no duplication and is
    # untouched, so this only removes a char the replacement itself introduced. (Rare false
    # collapse: a standalone form immediately followed by a distinct word that begins with the
    # canonical's last char — far rarer than the compound-overlap it fixes.)
    tail = replacement[-1] if replacement else ""
    out: list[str] = []
    i, n = 0, len(form)
    while True:
        j = text.find(form, i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        out.append(replacement)
        i = j + n
        if tail and i < len(text) and text[i] == tail:
            i += 1  # drop the seam-duplicated canonical tail char
    return "".join(out)


def glossary_dir() -> Path:
    """The folder holding the glossary IN USE and its companions (partners.yaml,
    editor_examples.<lang>.jsonl). Set from the CLI's --glossary (SUBTITLE_GLOSSARY_DIR), so a run on
    a code snapshot with `--glossary <snapshot>/glossary/glossary.yaml` reads AND writes the
    snapshot's copies. Before 2026-10-27 these were hard-coded to ./glossary: every A/B run read the
    live repo's editor examples and wrote partner lookups into the live partners.yaml."""
    import os

    return Path(os.environ.get("SUBTITLE_GLOSSARY_DIR", "glossary"))


def load(path: str | Path) -> Glossary:
    if yaml is None:
        raise RuntimeError("PyYAML is required: pip install pyyaml")
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    langs: list[str] = data.get("languages", [])
    terms: list[Term] = []
    for raw in data.get("terms", []):
        translations = {k: v for k, v in raw.items() if k in langs}
        terms.append(
            Term(
                source=raw["source"],
                translations=translations,
                aliases=raw.get("aliases", []),
                enforce=raw.get("enforce", "soft"),
                notes=raw.get("notes", ""),
                senses=raw.get("senses", []),
                avoid=raw.get("avoid", {}),
                case_sensitive=bool(raw.get("case_sensitive", False)),
            )
        )
    # Merge the generated term base (sibling termbase.yaml) — the authoritative ~1.4k
    # approved renderings. Hand-curated glossary.yaml WINS on any shared source form (its
    # terms carry senses/avoid/enforce tuning); TB terms fill the long tail as SOFT guidance.
    tb_path = Path(path).parent / "termbase.yaml"
    if tb_path.exists():
        have = {t.source.lower() for t in terms} | {a.lower() for t in terms for a in t.aliases}
        # glossary.yaml `termbase_exclude`: TB rows that are context-specific UI strings, not
        # terms (Hello -> 尊敬的开发者，您好！ is an email salutation; as a term it rewrote a lone
        # 'Hello' and kept flagging 'Hello World' cues as stale).
        have |= {x.lower() for x in data.get("termbase_exclude", []) or []}
        tdata = yaml.safe_load(tb_path.read_text(encoding="utf-8")) or {}
        tlangs = tdata.get("languages", langs)
        for raw in tdata.get("terms", []):
            if raw["source"].lower() in have:
                continue
            terms.append(
                Term(
                    source=raw["source"],
                    translations={k: v for k, v in raw.items() if k in tlangs},
                    aliases=raw.get("aliases", []),
                    enforce=raw.get("enforce", "soft"),
                    notes=raw.get("notes", ""),
                    avoid=raw.get("avoid", {}),
                )
            )

    return Glossary(
        version=str(data.get("version", "0")),
        languages=langs,
        terms=terms,
        style=data.get("style", {}),
        heuristics=data.get("heuristics", ""),
        identifiers=data.get("identifiers", []),
    )
