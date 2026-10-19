"""LLM transcript cleanup (optional quality pass before/after cue building).

Raw ASR output is often missing punctuation/casing and full of disfluencies
("um", "you know", false starts), which hurts both the EN captions and the
translation. This runs an LLM per cue to restore punctuation & casing and drop
filler WITHOUT paraphrasing — applied AFTER cue building so word timing is
preserved (each cue's timing stays; only its display text is polished).

`invoke` is any callable taking a prompt and returning text (e.g.
CliTranslator._invoke), so cleanup is engine-agnostic and testable with a fake.
"""

from __future__ import annotations

import re

from .config import LangSpec, spec_for
from .model import Subtitle
from .segment.cjk import break_lines
from .textwidth import display_width

_LANG_NAMES = {"en": "English", "zh-Hans": "Simplified Chinese", "ja": "Japanese"}


def build_cleanup_prompt(text: str, language: str = "en") -> str:
    lang = _LANG_NAMES.get(language, language)
    return f"""Clean up this raw speech-to-text subtitle line in {lang}. Restore
correct punctuation and capitalization, remove filler words and false starts
(um, uh, you know, repeated words), and fix obvious transcription slips. Do NOT
paraphrase, translate, add, or remove real content — keep the wording and meaning.
Return ONLY the cleaned line, nothing else.

Line: {text}"""


def build_correction_prompt(
    text: str,
    terms: list[str],
    suspect_words: list[str] | None = None,
    clean_read: bool = True,
    names: list[str] | None = None,
) -> str:
    suspect = ""
    if suspect_words:
        suspect = (
            "\nThe ASR was LOW-CONFIDENCE on these words — a real English word "
            "may have been substituted for a proper noun/term; scrutinize each "
            "and fix it to the correct known term when context fits (e.g. a "
            "mis-heard 'tracker' -> 'Shaka'): " + ", ".join(sorted(set(suspect_words))) + "\n"
        )
    if names:
        # Names have no dictionary anchor, so ASR mangles them (Okafor->Oak offer/Okapho) and
        # generic term-matching won't catch them — call them out explicitly and first.
        suspect = (
            "\nSPEAKER / PROPER NAMES — these appear in the video; spell each "
            "EXACTLY as written and fix any phonetic mishearing to it (e.g. "
            "'Oak offer'/'Okapho' -> 'Okafor'): " + ", ".join(names) + "\n"
        ) + suspect
    return _build_correction_prompt(text, terms, suspect, clean_read)


# Domain acronyms the ASR reliably mangles into a DIFFERENT real-looking acronym/word.
# The generic terms list isn't enough: the mishear is itself plausible (IEP is a real
# acronym), so the model leaves it. Name each one — expansion + its common mishearings —
# so the correction restores it (context permitting). Extend as new ones surface.
_ACRONYM_HINTS: dict[str, tuple[str, list[str]]] = {
    "IAP": ("in-app purchase", ["IEP", "IAB", "I A P", "IPA"]),
}


def _acronym_hint(text: str) -> str:
    """A one-line restore hint for any domain acronym (or its known mishears) that
    appears in this chunk — keeps the prompt short by naming only what's relevant."""
    text.lower()
    parts = []
    for acr, (expansion, mishears) in _ACRONYM_HINTS.items():
        forms = [acr] + mishears
        if any(re.search(rf"(?<![A-Za-z]){re.escape(f)}(?![A-Za-z])", text, re.I) for f in forms):
            parts.append(f"{acr} ({expansion}; often mis-transcribed as " f"{'/'.join(mishears)})")
    if not parts:
        return ""
    return (
        "\nDOMAIN ACRONYMS — restore the correct one when context fits (do NOT "
        "invent it where it doesn't): " + "; ".join(parts) + "\n"
    )


def _build_correction_prompt(
    text: str, terms: list[str], suspect: str = "", clean_read: bool = True
) -> str:
    """Document-level jargon/proper-noun correction + (default) CLEAN-READ editing.
    Far more effective than Whisper `initial_prompt` biasing: the ASR baseline is
    accurate on general English but mis-hears domain terms (npm->npn, MCP->mcb,
    Claude->cloud) and doesn't re-join spelled-out identifiers. `clean_read` also
    lets it fix non-native-speaker grammar and drop filler — the house preference
    (educational clean-read captions, DCMP), matching what our human captions do."""
    lead = (
        (
            "You are producing CLEAN-READ English captions from a raw speech-to-text "
            "transcript of a technical developer video; speakers may be non-native and "
            "disfluent. Make these fixes using context. PRESERVE the technical meaning, "
            "every number, and every code/API identifier — do NOT add facts or drop "
            "substantive content:\n"
        )
        if clean_read
        else (
            "You are correcting a raw English speech-to-text transcript of a technical "
            "developer video. Make these fixes ONLY, using context; do NOT paraphrase, "
            "add, or remove content — keep everything else verbatim:\n"
        )
    )
    fixes = (
        "(1) fix misrecognized domain terms / proper nouns to their correct form "
        "(fix close phonetic matches to the known terms below) — INCLUDING a word that "
        "is not standard English and is phonetically near a known product/proper name "
        "(e.g. a mishearing 'Kuber Netty's' -> 'Kubernetes', 'Fig ma' -> 'Figma');\n"
        "(2) RE-JOIN code/API identifiers that were transcribed as separate "
        "spelled-out words into their exact camelCase / PascalCase / dotted / "
        "scope-resolved (a::b) form when context shows it's a code symbol (a prop, "
        "method, class, package): e.g. 'next focus right' -> 'nextFocusRight', "
        "'turbo module' -> 'TurboModule', 'TV focus guide view' -> 'TVFocusGuideView', "
        "'UI manager complete surface' -> 'UIManager::completeSurface', 'root shadow "
        "node layout' -> 'RootShadowNode::Layout' — PRESERVE the '::' scope operator.\n"
        "(3) format numbers (DCMP): spell out one-ten, use numerals for 11+ and "
        "for any number with a unit/percent/time/money ('58 milliseconds', '6%', "
        "'$13 million'); BUT keep code/version tokens exactly as-is (v1.2, 58ms, "
        "0.6b, IPv6).\n"
    )
    if clean_read:
        fixes += (
            "(4) LIGHT GRAMMAR FIX ONLY: fix outright grammatical ERRORS from non-"
            "native/disfluent speech — subject-verb agreement ('component mount' -> "
            "'components mount'), wrong singular/plural, clearly wrong verb tense, a "
            "missing/wrong article. Do NOT otherwise restyle: KEEP the speaker's own "
            "words and word-forms when they are already grammatical — do NOT swap "
            "synonyms (like<->such as), do NOT change gerund<->infinitive (staying<->"
            "to stay), do NOT change contractions, do NOT paraphrase. When in doubt, "
            "leave it verbatim. Remove disfluencies/false starts used as filler "
            "(um, uh, and 'like'/'you know'/'sort of'/'I mean'/'basically'/'kind of' "
            "ONLY when used as filler, repeated words, abandoned restarts). Also drop a "
            "sentence-initial backchannel/affirmation that isn't in the core message — "
            "'Yeah,'/'Yes,'/'Ah,'/'Oh,'/'Well,'/'So,'/'Right,' when it's just a spoken "
            "lead-in (keep a genuine answer: 'Yes, it does' stays; a bare opener 'Yeah, "
            "so what we built…' -> 'What we built…'). CRITICAL: "
            "do NOT drop CONTENT words — every noun, verb, adjective, adverb, number, "
            "or named thing must be kept, even short ones ('bit', 'stock', 'move', "
            "'perform', 'like' as a verb); when a word carries meaning, keep it.\n"
            "(5) Use American English spelling (color, behavior, optimize, "
            "initialize, license) — the source may be British.\n"
        )
    return (
        lead
        + fixes
        + _acronym_hint(text)
        + suspect
        + "Known domain terms: "
        + ", ".join(terms)
        + "\n\n"
        + "TRANSCRIPT:\n"
        + text
        + "\n\nReturn ONLY the corrected transcript."
    )


# en-GB -> en-US spelling fold. Parakeet tends to output British spelling
# ("favourite", "optimise", "colour"); our house style is American English (matches
# the human captions). Curated whole-word list (NOT a blanket -ise->-ize rule, which
# false-fires on exercise/promise/precise/etc.). Case of the first letter is preserved.
_UK_US = {
    "favourite": "favorite",
    "favourites": "favorites",
    "colour": "color",
    "colours": "colors",
    "coloured": "colored",
    "behaviour": "behavior",
    "behaviours": "behaviors",
    "behavioural": "behavioral",
    "optimise": "optimize",
    "optimised": "optimized",
    "optimising": "optimizing",
    "optimisation": "optimization",
    "optimisations": "optimizations",
    "initialise": "initialize",
    "initialised": "initialized",
    "initialising": "initializing",
    "initialisation": "initialization",
    "customise": "customize",
    "customised": "customized",
    "customising": "customizing",
    "customisation": "customization",
    "organise": "organize",
    "organised": "organized",
    "organising": "organizing",
    "organisation": "organization",
    "synchronise": "synchronize",
    "synchronised": "synchronized",
    "synchronisation": "synchronization",
    "serialise": "serialize",
    "serialised": "serialized",
    "deserialise": "deserialize",
    "analyse": "analyze",
    "analysed": "analyzed",
    "analysing": "analyzing",
    "prioritise": "prioritize",
    "prioritised": "prioritized",
    "summarise": "summarize",
    "summarised": "summarized",
    "recognise": "recognize",
    "recognised": "recognized",
    "minimise": "minimize",
    "minimised": "minimized",
    "maximise": "maximize",
    "maximised": "maximized",
    "utilise": "utilize",
    "utilised": "utilized",
    "normalise": "normalize",
    "normalised": "normalized",
    "capitalise": "capitalize",
    "capitalised": "capitalized",
    "visualise": "visualize",
    "visualised": "visualized",
    "authorise": "authorize",
    "authorised": "authorized",
    "authorisation": "authorization",
    "licence": "license",
    "catalogue": "catalog",
    "centre": "center",
    "labelled": "labeled",
    "labelling": "labeling",
    "cancelled": "canceled",
    "cancelling": "canceling",
    "modelling": "modeling",
    "modelled": "modeled",
    "travelling": "traveling",
    "grey": "gray",
    "defence": "defense",
    "programme": "program",
}
_UK_RE = re.compile(
    r"\b(" + "|".join(sorted(_UK_US, key=len, reverse=True)) + r")\b", re.IGNORECASE
)


# Brands whose store is spelled 'Appstore' (one word). Example values — edit for your content.
_APPSTORE_BRANDS: tuple[str, ...] = ("Acme", "Orbit")


def normalize_en_spelling(text: str) -> str:
    """Fold common British spellings to American (house style) + brand spellings. Whole-word,
    case-of-first-letter preserved. A no-op on already-normalized text (safe on any path)."""

    def repl(m):
        us = _UK_US[m.group(0).lower()]
        return us[0].upper() + us[1:] if m.group(0)[:1].isupper() else us

    text = _UK_RE.sub(repl, text)
    # Brand spelling: some vendors brand their store as 'Appstore' (one word). ONLY fold when
    # it's BRAND-QUALIFIED by one of _APPSTORE_BRANDS — 'Acme App Store' -> 'Acme Appstore'.
    # Leave a BARE / generic 'App Store' alone: in React Native / mobile dev content it usually
    # means Apple's iOS App Store or a generic store ('an app in the App Store', 'published
    # through an App Store') — a blanket fold would mislabel the brand. (Apple's is thus left
    # untouched automatically, since it isn't in _APPSTORE_BRANDS.)
    if _APPSTORE_BRANDS:
        brands = "|".join(re.escape(b) for b in _APPSTORE_BRANDS)
        text = re.sub(rf"\b({brands})\s+App\s+[Ss]tore\b", r"\1 Appstore", text)
    # Dotted names broken by a caption line: 'React. js' -> 'React.js'
    text = re.sub(r"\b([A-Za-z][A-Za-z0-9]*)\.\s+(js|ts|jsx|tsx|io)\b", r"\1.\2", text)
    return text


_SMALL = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}
# Units/suffixes after which a number stays a numeral (DCMP: numerals with units).
_UNIT_AFTER = re.compile(
    r"^(%|s|ms|fps|px|x|k|kb|mb|gb|hz|khz|p|am|pm|st|nd|rd|th|"
    r"seconds?|milliseconds?|microseconds?|minutes?|hours?|frames?|percent|"
    r"pixels?|bit|bits|byte|bytes)\b",
    re.IGNORECASE,
)


def spell_small_numbers(text: str) -> str:
    """Spell out standalone integers 1-10 (DCMP: spell one-ten). Strict guards keep
    numerals where the convention does: numbers with a unit/percent/time/money, and
    ANY code/version token (v1.2, 58ms, IPv6, 0.6b) — a number touching a letter,
    dot, %, $, :, digit, or immediately followed by a unit is left untouched."""

    def repl(m):
        num = m.group(0)
        pre, post = text[: m.start()], text[m.end() :]
        if pre.rstrip()[-1:] in "$.v" or (pre and pre[-1].isalnum()):
            return num  # $6, v6, 1.6, word-adjacent -> keep
        nxt = post.lstrip()
        if nxt[:1] in "%:." or (nxt and nxt[0].isdigit()) or _UNIT_AFTER.match(nxt):
            return num  # 6%, 6:25, 6 ms, 6.2 -> keep numeral
        return _SMALL[num]

    return re.sub(r"\b(?:10|[1-9])\b", repl, text)


def correct_transcript(
    invoke,
    text: str,
    terms: list[str],
    *,
    max_words: int = 130,
    suspect_words: list[str] | None = None,
    clean_read: bool = True,
    names: list[str] | None = None,
) -> str:
    """Glossary-aware AUTO-correction of a transcript (empty terms -> no-op).

    Chunks long transcripts (~max_words, sentence-aligned) so attention isn't
    diluted (Perfetto: whole-call missed the long idents; chunked recovered most).
    `suspect_words` = low-ASR-confidence words; each chunk is told to scrutinize the
    suspects it contains, so the reviewer AUTO-FIXES mis-heard proper nouns
    (Shaka->'tracker') instead of only flagging them. Then deterministically spells
    small numbers (DCMP). This is the ASR analogue of the translation auto-fixer."""
    if not terms and not clean_read and not names:
        return spell_small_numbers(normalize_en_spelling(text))  # deterministic passes only
    import re as _re

    sents = _re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sents:
        cur = (cur + " " + s).strip()
        if len(cur.split()) >= max_words:
            chunks.append(cur)
            cur = ""
    if cur:
        chunks.append(cur)
    susp = [w for w in (suspect_words or [])]
    parts = []
    for c in chunks:
        here = [w for w in susp if w.lower() in c.lower()] or None  # suspects in this chunk
        parts.append(_correct_chunk(invoke, c, terms, here, clean_read, names))
    return (
        spell_small_numbers(normalize_en_spelling(dedupe_adjacent_words(" ".join(parts)))) or text
    )


# Function words that never validly double in English — an immediate repeat is an ASR
# stutter/dup artifact ("improve the the software"). Deliberately EXCLUDES words that CAN
# validly double ("that that", "had had", "is is not") so the dedup can't corrupt real text.
_NEVER_DOUBLE = frozenset(
    (
        "the a an to of and or but so with in on at for from by as is are was were be been "
        "we i it he she they you your our their this these those"
    ).split()
)


def collapse_immediate_repeats(text: str, min_unit: int = 3) -> str:
    """Collapse an IMMEDIATELY-repeated substring of >= min_unit chars to a single copy — the
    target-side (CJK) analog of dedupe_adjacent_words, catching degenerate re-emits the model
    sometimes produces ('マーケティングマーケティング・ウェブサイト' -> 'マーケティング・ウェブサイト').
    min_unit=3 keeps it safe: a >=3-char block repeated back-to-back is virtually always a garble,
    while short emphatic doublings (很好很好 / そうそう, <=2 chars) are left untouched. Works on any
    script (no word boundaries needed). Idempotent."""
    # (unit of >=min_unit chars) immediately followed by >=1 identical copies -> keep one unit.
    return re.sub(rf"(.{{{min_unit},}}?)\1+", r"\1", text)


def dedupe_adjacent_words(text: str) -> str:
    """Deterministically collapse an immediately-repeated function word (case-insensitive),
    e.g. 'the the' -> 'the'. Safe subset only (see _NEVER_DOUBLE); content words and
    validly-doubling words are left untouched. Complements the LLM clean-read (which the
    chunk boundaries can make it miss)."""
    import re as _re

    toks = _re.split(r"(\s+)", text)
    out: list[str] = []
    prev = None
    for t in toks:
        if not t.strip():
            out.append(t)
            continue
        w = t.strip()
        # Only dedup a BARE alpha repeat (no attached punctuation) — so we never drop a
        # comma/period carried by the second token ("So so, it" is left alone).
        if w.isalpha() and w.lower() == prev and w.lower() in _NEVER_DOUBLE:
            if out and not out[-1].strip():
                out.pop()  # also drop the space before the removed dup
            continue
        out.append(t)
        prev = w.lower() if w.isalpha() else None
    return "".join(out)


def _correct_chunk(invoke, c, terms, here, clean_read, names, *, min_ratio: float = 0.72) -> str:
    """Correct one chunk, with a COMPLETENESS SAFEGUARD: clean-read legitimately drops
    filler and merges spelled-out identifiers (some shrinkage is fine), but it must not
    drop whole clauses. If the result is dramatically shorter than the input (< min_ratio
    of its words), the correction likely dropped content — retry once forcing completeness,
    and if it's still short, keep the ORIGINAL chunk (verbatim beats dropped)."""
    out = invoke(build_correction_prompt(c, terms, here, clean_read, names)).strip() or c
    n = len(c.split())
    if n >= 8 and len(out.split()) < min_ratio * n:
        retry = invoke(
            build_correction_prompt(c, terms, here, clean_read, names)
            + "\nIMPORTANT: your previous attempt DROPPED content. Keep EVERY "
            "sentence and clause; remove ONLY um/uh and exact repeats."
        ).strip()
        out = retry if len(retry.split()) >= min_ratio * n else c  # else keep raw (no drop)
    return out


def polish_text(invoke, text: str, language: str = "en") -> str:
    """Clean a single line via `invoke`. Falls back to the original on empty."""
    cleaned = invoke(build_cleanup_prompt(text, language)).strip()
    return cleaned or text


def cue_cps(cue, cjk: bool) -> float:
    dur = max(cue.duration, 1e-6)
    return display_width("".join(cue.lines), cjk) / dur


def condense_dense_cues(
    sub: Subtitle, invoke, *, spec: LangSpec | None = None, headroom: float = 1.0
) -> Subtitle:
    """Tighten only the cues that read too fast (cps > spec.max_cps * headroom):
    remove redundancy/filler, KEEP every fact and technical detail; timing is
    unchanged, so fewer characters in the same window lowers the reading speed.
    Cues already within cps are left untouched (no needless rewriting)."""
    spec = spec or spec_for(sub.language)
    cap = spec.max_cps * headroom
    joiner = "" if spec.cjk else " "
    lang = _LANG_NAMES.get(sub.language, sub.language)
    for cue in sub.cues:
        if cue_cps(cue, spec.cjk) <= cap:
            continue
        text = joiner.join(cue.lines)
        budget = max(int(spec.max_cps * cue.duration), 8)
        prompt = (
            f"This {lang} subtitle reads too fast. Tighten it to about "
            f"{budget} characters by removing redundancy and filler and using "
            f"more concise phrasing — but KEEP every fact, name, number, and "
            f"technical detail; do not translate or add anything. Return ONLY "
            f"the tightened {lang} text.\n\n{text}"
        )
        tightened = invoke(prompt).strip()
        if tightened and display_width(tightened, spec.cjk) < display_width(text, spec.cjk):
            cue.lines = break_lines(tightened, spec)
    return sub


def polish_subtitle(sub: Subtitle, invoke, *, spec: LangSpec | None = None) -> Subtitle:
    """Polish every cue's text in place and re-wrap lines. Timing is untouched."""
    spec = spec or spec_for(sub.language)
    joiner = " " if not spec.cjk else ""
    for cue in sub.cues:
        cleaned = polish_text(invoke, joiner.join(cue.lines), sub.language)
        cue.lines = break_lines(cleaned, spec)
    return sub
