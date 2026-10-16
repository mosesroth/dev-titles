"""Deterministic QA checks that run before (and alongside) the LLM MQM judge —
cheap gates that catch the reviewer's two complaints without a model call:

- `omission_suspects`: a cue that uses far less than its available reading time
  (cps well under the cap) while translating a substantial source sentence likely
  DROPPED content — the "short + low cps = defect vs short-at-cap = OK" triage.
- `term_consistency`: one EN glossary term should map to ONE target rendering per
  document; flags a term rendered two different ways.
"""

from __future__ import annotations

import re

from .config import LangSpec
from .glossary import Glossary
from .textwidth import display_width


def omission_suspects(
    pairs: list[tuple[str, str, float]],
    spec: LangSpec,
    cps_frac: float = 0.45,
    min_src_words: int = 8,
) -> list[dict]:
    """pairs: (source_en, target_text, on_screen_seconds). Flag cues whose target
    uses < cps_frac of the reading-speed cap for a non-trivial source sentence —
    reading time is available, so shortness suggests omission, not condensation."""
    out = []
    for src, tgt, dur in pairs:
        dur = max(dur, 1e-6)
        chars = display_width(tgt, spec.cjk)
        cps = chars / dur
        if cps < cps_frac * spec.max_cps and len(src.split()) >= min_src_words:
            out.append(
                {
                    "source": src,
                    "target": tgt,
                    "cps": round(cps, 1),
                    "reason": f"cps {cps:.1f} well under cap {spec.max_cps} — "
                    "reading time unused, possible omission",
                }
            )
    return out


def is_short_of_expected(source_en: str, target: str, lang: str, min_words: int = 8) -> bool:
    """Length-ratio omission gate (Focus audit P0). The prior `omission_suspects`
    is blind to a whole sentence dropped inside a merged block: the shortfall
    shows up as INTER-cue dead air with healthy per-cue cps, so cps-based checks
    fire zero flags. Instead compare the *sentence* target length to its EN word
    count. Measured house expansion: ja ≈2.0–2.95 target chars/EN word, zh ≈1.6.
    A ratio below the floor on a non-trivial sentence means content was dropped."""
    words = len(source_en.split())
    if words < min_words:
        return False
    chars = len(target.replace(" ", "").replace("　", ""))
    floor = 1.8 if lang == "ja" else 1.4
    return chars / words < floor


_NUM = re.compile(r"\d[\d,.]*")
# EN scale words fold into the magnitude so a CJK myriad rendering matches: "50 billion"
# is realized as 500億/500亿 (=50e9), not a literal "50" — without this the bare-digit
# check flags every scaled number as dropped. See _num_present / _source_numbers.
_SCALE = {"thousand": 10**3, "million": 10**6, "billion": 10**9, "trillion": 10**12}
_NUM_SCALE = re.compile(r"(\d[\d,.]*)\s+(thousand|million|billion|trillion)\b", re.I)
_IDENT = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[._:][A-Za-z0-9]+)+\b|\b[a-z]+[A-Z][A-Za-z0-9]*\b")


# CLI commands / env vars must stay Latin verbatim — otherwise they get partially
# translated or recased ('create-expo-app'->'create-expo-アプリ', 'npm run android'->
# 'npm run Android', 'run:android'->'run: Android'), which is a real DNT failure (6yAt
# audit: most ERROR rows were command corruption). High-precision detectors:
# Leader + args, where each arg is a known subcommand OR a token bearing command
# punctuation/digits — so it stops at plain English ('npm run android and specify' ->
# 'npm run android', not '...and specify'). Trailing sentence punctuation is stripped.
_CLI_LEADER = re.compile(
    r"\b(?:npm|npx|yarn|pnpm|bun|adb|gradle|pod|pip)\b"
    # each arg: a punctuation/digit-bearing token (matched WHOLE, first), else a known
    # subcommand that ends at a boundary (so 'create' can't match inside 'create-expo-app')
    r"(?:\s+(?:[^\s]*[-:/@=\d][^\s]*|(?:run|install|start|build|init|add|remove|test|dev|"
    r"devices|expo|create|link|prebuild)(?=\s|$|[.,;:!?]))){1,5}"
)
_CLI_COLON = re.compile(r"\b[a-z][\w-]*:[a-z][\w-]+\b")  # run:android, start:dev
_CLI_HYPHEN = re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+){2,}\b")  # create-expo-app (>=2 hyphens)
_ENV_VAR = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")  # JAVA_HOME, ANDROID_HOME
# common English multi-hyphen words that are NOT commands (don't force-keep them Latin)
_HYPHEN_STOP = frozenset(
    (
        "up-to-date state-of-the-art out-of-the-box end-to-end day-to-day real-time "
        "open-source out-of-tree plug-and-play"
    ).split()
)


def detect_cli_commands(text: str) -> list[str]:
    """CLI-command / env-var spans that must stay Latin verbatim. High-precision:
    package-manager-led invocations, colon-commands, 2+-hyphen tokens (minus common
    English hyphenated words), and ALL_CAPS env vars. Deduped, order-preserving."""
    out: list[str] = []
    out += [m.group(0).strip(" .,;:!?)") for m in _CLI_LEADER.finditer(text)]  # drop trailing punct
    for rx in (_CLI_COLON, _ENV_VAR):
        out += [m.group(0).strip() for m in rx.finditer(text)]
    out += [
        m.group(0) for m in _CLI_HYPHEN.finditer(text) if m.group(0).lower() not in _HYPHEN_STOP
    ]
    seen: set[str] = set()
    res: list[str] = []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            res.append(c)
    return res


# Web domains / URLs must stay Latin verbatim — else a glossary term translates a label
# INSIDE them (developer.acme.com -> developer.艾克米.com; the QE-repair retranslate can't
# fix it because the same translate path re-corrupts it, so PREVENT it by masking). No
# leading/trailing \b: in CJK subtitles the domain is glued to CJK (…com网站), and \b fails
# between a Latin TLD and a CJK char; the label/TLD structure already bounds the match.
_DOMAIN = re.compile(
    r"(?:https?://)?"  # optional scheme
    r"(?:[a-zA-Z0-9][a-zA-Z0-9-]*\.)+"  # one+ dot-terminated labels
    r"(?:com|net|org|io|dev|co|ai|app|edu|gov|android)"  # known TLD / 2nd-level
    r"(?:\.[a-zA-Z]{2,})?"  # optional extra segment (.com.cn)
    r"(?:/[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]*)?",  # optional ASCII path (never CJK)
    re.IGNORECASE,
)


def detect_domains(text: str) -> list[str]:
    """Web domains / URLs (developer.acme.com, docs.acme.dev/foo) to keep Latin verbatim.
    Ordered, deduped; trailing sentence punctuation (ASCII + CJK) trimmed."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _DOMAIN.finditer(text):
        d = m.group(0).strip().rstrip(".,;:!?)]}。，、）】")
        if d and d.lower() not in seen:
            seen.add(d.lower())
            out.append(d)
    return out


def protected_spans(text: str) -> list[str]:
    """All DNT spans to MASK before translation/enforcement: CLI commands + web domains.
    Use this at every mask leak point so both classes survive verbatim (see translate_segment,
    glossary.enforce, localize.edit_fix)."""
    return detect_cli_commands(text) + detect_domains(text)


def _source_numbers(source_en: str) -> list[str]:
    """Numeric tokens the target must realize, with a trailing scale word folded into the
    magnitude ('50 billion' -> '50000000000') so CJK myriad forms (500億/500亿) match. A bare
    number consumed by a scale phrase is not also required on its own (avoids flagging '50'
    when only '500億' is present)."""
    nums: list[str] = []
    consumed: list[tuple[int, int]] = []
    for m in _NUM_SCALE.finditer(source_en):
        try:
            val = float(m.group(1).replace(",", "")) * _SCALE[m.group(2).lower()]
        except ValueError:
            continue
        nums.append(str(int(val)) if val == int(val) else f"{val:g}")
        consumed.append(m.span(1))
    for m in _NUM.finditer(source_en):
        if not any(a <= m.start() < b for a, b in consumed):
            nums.append(m.group(0))
    return nums


def missing_numbers_or_idents(source_en: str, target: str) -> list[str]:
    """EN numbers (58, 235) and code identifiers (performMountingInstructions,
    Systrace.capture, JSExecutor::loadBundle) that DON'T appear verbatim in the
    target. Dropping a measurement or an API name is a severe, deterministically
    detectable omission the length gate is blind to (audit A3 dropped '58 ms')."""
    want = set(_source_numbers(source_en)) | set(_IDENT.findall(source_en))
    # single-token CLI commands / env vars must survive verbatim too (multi-word
    # invocations are prevention-only via DNT; exact-match here would be too noisy)
    want |= {c for c in detect_cli_commands(source_en) if " " not in c}
    want |= set(detect_domains(source_en))  # a mangled domain (developer.艾克米.com) is a defect
    # numbers may render with a localized separator; compare on digits only
    tnum = set(_NUM.findall(target))
    tnum_digits = {n.replace(",", "").replace(".", "") for n in tnum}
    missing = []
    for tok in want:
        if _NUM.fullmatch(tok):
            if not _num_present(tok, target, tnum_digits):
                missing.append(tok)
        elif tok not in target:
            missing.append(tok)
    return missing


_INT2CJK = str.maketrans("0123456789", "〇一二三四五六七八九")


def _num_present(tok: str, target: str, tnum_digits: set[str]) -> bool:
    """Is an EN source number realized in the target — INCLUDING natural CJK forms the
    digit-only check misses: myriad 万 / 亿 (130,000 -> 13万) and half 半 (1.5 -> 一年半 / 1年半)?
    Prevents false 'dropped number' flags on every large/decimal number in CJK output."""
    if tok.replace(",", "").replace(".", "") in tnum_digits:
        return True
    raw = tok.replace(",", "")
    try:
        val = float(raw)
    except ValueError:
        return False
    if val == int(val):  # integer -> myriad forms
        v = int(val)
        # largest units first so 10^12 -> 1兆 / 1万亿, not 10000億; include ja kanji 億 and
        # 兆 (10^12) alongside simplified 亿, plus colloquial zh 万亿 (=10^12).
        for unit, div in (
            ("兆", 10**12),
            ("万亿", 10**12),
            ("億", 10**8),
            ("亿", 10**8),
            ("万", 10**4),
        ):
            if v >= div:
                q = v / div
                form = f"{int(q)}{unit}" if q == int(q) else f"{q:g}{unit}"
                if form in target or form.translate(_INT2CJK) in target:
                    return True
    if raw.endswith(".5") and "半" in target:  # 1.5 -> 一年半 / 1年半
        ip = raw[:-2]
        if not ip or ip in target or ip.translate(_INT2CJK) in target:
            return True
    return False


def term_consistency(
    sources: list[str], targets: list[str], glossary: Glossary, lang: str
) -> list[dict]:
    """Flag any glossary term whose canonical target rendering is expected (it
    appears in a source segment) but which is rendered inconsistently across the
    document — i.e. the canonical rendering is present in some target cues that
    use the term and absent in others."""
    issues = []
    for term in glossary.terms:
        canonical = term.translations.get(lang, "").strip()
        if not canonical:
            continue
        present, missing_idx = 0, []
        for i, (src, tgt) in enumerate(zip(sources, targets)):
            hits = glossary.retrieve(src)
            if hits and any(t.source == term.source for t in hits):
                if canonical in tgt:
                    present += 1
                else:
                    missing_idx.append(i)
        if present and missing_idx:  # rendered both ways across the doc
            issues.append(
                {
                    "term": term.source,
                    "expected": canonical,
                    "consistent_cues": present,
                    "inconsistent_cues": len(missing_idx),
                    "inconsistent_idx": missing_idx,
                }
            )
    return issues


# ---- MQM DQF scoring (industry MQM/DQF-style weighting) --------------------
# Severity weights (points), gate at 96% with zero criticals for CJK subtitles.
DQF_WEIGHTS = {"critical": 40.0, "major": 2.5, "minor": 0.2}
DQF_GATE = 96.0


def dqf_score(counts: dict[str, int], word_count: int) -> float:
    """MQM/DQF quality score in [0,100]. counts: {critical,major,minor}."""
    wc = max(word_count, 1)
    penalty = sum(DQF_WEIGHTS.get(sev, 0) * n for sev, n in counts.items())
    return max(0.0, 100 - penalty / wc * 1000)  # points-per-1000-words penalty


def dqf_gate(
    counts: dict[str, int], word_count: int, threshold: float = DQF_GATE
) -> tuple[bool, float]:
    """(passes, score). Fails on any critical, or score below threshold."""
    score = dqf_score(counts, word_count)
    passes = counts.get("critical", 0) == 0 and score >= threshold
    return passes, score
