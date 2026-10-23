"""Segment translation orchestration (pipeline steps [3] + [5]).

Composes a Translator (any engine) with the Glossary: retrieve per-segment terms,
inject register/style, draft, optionally reflect-then-refine, then deterministically
enforce hard glossary terms. Engine-agnostic — pass BedrockTranslator in prod or a
fake in tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..glossary import EnforcementIssue, Glossary
from .base import TranslationRequest, Translator
from .prompts import extract_final_translation

if TYPE_CHECKING:
    from ..model import Subtitle


def _punct_note(target_lang: str, exemplar_bank=None) -> str:
    """Language-specific punctuation-density guidance for the prompt.

    ja: subtitles carry pauses with line/cue breaks — no 、 or 。 at all (matches
        the human house files, which have ~0 of each).
    zh: a MEASURED comma target from the retrieved exemplars (the blanket
        'very few commas' over-corrected to zero, per the Focus audit), plus the
        placement rules a native writer uses. Never use 。."""
    if target_lang == "ja":
        return (
            "Do NOT use 、 or 。 anywhere. Carry every pause with the line/cue "
            "break instead (the house subtitles contain zero 、 and 。)."
        )
    if target_lang.startswith("zh"):
        rate = 2.0  # measured house rate ≈2.05 ，/100 han chars
        if exemplar_bank is not None:
            try:
                from ..exemplars import punct_profile

                rate = punct_profile(exemplar_bank.exemplars).get("，", 2.0) or 2.0
            except Exception:
                pass  # keep the default house rate if the exemplar profile is unavailable
        # Target the MEASURED house rate (~2.0). The model UNDERSHOOTS a stated target
        # (a 1.5 "ceiling" yielded only ~0.7/100 — 1/3 of house — reading as run-ons),
        # so state it as a TARGET, not a sparse ceiling; clamp to a sane band.
        rate = min(max(rate, 1.6), 2.6)
        return (
            f"Use ，for clause boundaries — after a fronted adverbial (请记住、此外、"
            f"这样一来) or between a fronted condition and its main clause — at about "
            f"{rate:.1f} per 100 characters (the human house rate). NOT after every "
            "phrase, and NOT zero. Never use 。."
        )
    return "Match the exemplars' punctuation density."


def translate_segment(
    translator: Translator,
    glossary: Glossary,
    source: str,
    target_lang: str,
    *,
    full_context: str = "",
    line_context: str = "",
    reflect: bool = True,
    char_budget: int | None = None,
    exemplar_bank=None,
    completeness_hint: str = "",
    correction_hint: str = "",
    extra_dnt: list[str] | None = None,
) -> tuple[str, list[EnforcementIssue]]:
    """Translate one source segment into `target_lang`.

    Returns (final_text, enforcement_issues). Register/style notes are read from
    the glossary's `style` block for the target language. When an `exemplar_bank`
    is given, the few most relevant human EN->target pairs are injected as few-shot
    style anchors.
    """
    style = glossary.style.get(target_lang, {})
    # MASK CLI commands / env vars before translating: a prose "keep Latin" instruction is
    # NOT enough — the model still translates the English component (create-expo-app ->
    # create-expo-アプリ) and recases (run:android -> run:Android). Replacing each with a
    # stable placeholder GUARANTEES verbatim survival; we restore after translate+reflect.
    from ..qa import protected_spans

    # Mask CLI commands + web domains AND any DNT term (glossary identifiers, product/coined
    # names, speaker names) that appears in the source — a prose "keep Latin" hint is NOT
    # enough (the model still localized the 'Acme' inside 'Acme Cloud' -> 艾克米Cloud, and
    # flip-flopped 'vibecoding' Latin/カタカナ). Masking to a placeholder GUARANTEES verbatim.
    dnt_in_source = [d for d in (extra_dnt or []) if d and len(d) >= 2 and d in source]
    masked_source, mask = _mask_spans(source, protected_spans(source) + dnt_in_source)
    # The completeness_hint echoes source English (frag list + "include X verbatim")
    # — mask commands there too, else the model renders the command from the hint and
    # translates its English component despite the placeholder in source_text.
    masked_hint = completeness_hint
    masked_correction = correction_hint
    masked_line = line_context  # the line's neighbors repeat its English: mask there too
    for token, span in mask.items():
        masked_hint = _replace_span(masked_hint, span, token)
        masked_correction = _replace_span(masked_correction, span, token)
        masked_line = _replace_span(masked_line, span, token)
    req = TranslationRequest(
        source_text=masked_source,
        source_lang="en",
        target_lang=target_lang,
        full_context=full_context,
        line_context=masked_line,
        glossary_terms=glossary.retrieve(source),  # retrieve on the REAL source
        register=style.get("register", ""),
        style_notes=style.get("notes", "")
        + ("\nTokens like [[0]] are placeholders — copy them EXACTLY, unchanged." if mask else ""),
        char_budget=char_budget,
        localization_policy=glossary.heuristics
        + (
            "\nKeep these product names / code identifiers in Latin verbatim (do not "
            "translate or transliterate): " + ", ".join(extra_dnt)
            if extra_dnt
            else ""
        ),
        exemplars=exemplar_bank.select(source) if exemplar_bank else [],
        punct_note=_punct_note(target_lang, exemplar_bank),
        completeness_hint=masked_hint,
        correction_hint=masked_correction,
    )
    draft = translator.translate(req)
    if reflect:
        # The reflect pass may wrap its answer in commentary — extract the final.
        draft = extract_final_translation(translator.reflect_and_refine(req, draft))
    # ENFORCE FIRST (while commands are still [[N]] placeholders): otherwise a glossary
    # term substitution translates a substring of a command (the 'app' in create-expo-app
    # -> create-expo-アプリ). The placeholder shields it; THEN restore verbatim.
    final, issues = glossary.enforce(draft, source, target_lang, placeholders=mask)
    return _unmask(final, mask), issues


_WORDLIKE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ]*$")


def _replace_span(text: str, span: str, token: str) -> str:
    """Replace `span` with `token`. Word-like spans (pure alphanumerics/spaces, e.g. the
    company 'Nova', 'Acme Cloud') match only on alphanumeric boundaries so an identifier
    that is a PREFIX of a longer word isn't eaten inside it ('Nova' must NOT hit the
    'Nova' in the founder 'Novak Petrov' — a plain replace left the translator '[[N]]k').
    Spans with punctuation (domains, CLI commands, 'Nimbus+', '::dispatchEvent') keep the
    substring replace they need."""
    if _WORDLIKE.match(span):
        return re.sub(r"(?<![A-Za-z0-9])" + re.escape(span) + r"(?![A-Za-z0-9])", token, text)
    return text.replace(span, token)


def _mask_spans(text: str, spans: list[str]) -> tuple[str, dict]:
    """Replace each DNT span with a stable `[[N]]` placeholder (longest first, so a
    phrase masks before its sub-token). Returns (masked_text, {placeholder: span}).
    Records a span only when it actually matched (boundary-aware for word-like spans)."""
    mask: dict[str, str] = {}
    for i, s in enumerate(sorted(set(spans), key=len, reverse=True)):
        if not s:
            continue
        token = f"[[{i}]]"
        new = _replace_span(text, s, token)
        if new != text:
            text = new
            mask[token] = s
    return text, mask


def _unmask(text: str, mask: dict) -> str:
    """Restore placeholders to their verbatim spans. Tolerates full-width brackets the
    model may emit in CJK output (［［0］］) by normalizing them first."""
    if not mask:
        return text
    text = text.replace("［［", "[[").replace("］］", "]]").replace("【", "[[").replace("】", "]]")
    for token, s in mask.items():
        text = text.replace(token, s)
        text = text.replace(
            token.replace("[[", "[").replace("]]", "]"), s
        )  # single-bracket fallback
    return text


@dataclass
class LocalizeResult:
    subtitle: "Subtitle"
    issues: list[EnforcementIssue] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # per-sentence EN
    hypotheses: list[str] = field(default_factory=list)  # per-sentence translation


def localize(
    translator: Translator,
    glossary: Glossary,
    en,
    target_lang: str,
    *,
    reflect: bool = True,
    exemplar_bank=None,
    video_renderings=None,
) -> LocalizeResult:
    """Full R3 path: merge EN cues into sentences, translate each with whole-clip
    context + glossary enforcement, then re-time into CJK-sized cues. Returns the
    target Subtitle plus per-sentence sources/hypotheses (for QE) and issues."""
    from ..config import spec_for
    from ..model import Subtitle
    from ..resegment import merge_en_to_sentences, resegment_translation, snap_gaps
    from ..segment import normalize_ja_punct, normalize_zh_punct

    spec = spec_for(target_lang)
    # Per-video pinned renderings (from vocab-prep): inject + enforce so a
    # video-specific term gets ONE consistent rendering (no MT flip-flop).
    if video_renderings:
        from ..vocab_prep import renderings_to_terms

        glossary.terms.extend(
            renderings_to_terms(
                video_renderings,
                skip_sources={f for t in glossary.terms for f in t.all_source_forms},
                curated_renderings={
                    f: t.translations for t in glossary.terms for f in t.all_source_forms
                },
            )
        )
    sentences = merge_en_to_sentences(en)
    context = " ".join(s.text for s in sentences)
    # keep-together terms: glossary targets for this language (localized multi-token
    # terms not split across a cue) — protects e.g. 開発者ポータル / React Native.
    keep = [
        t.translations.get(target_lang) for t in glossary.terms if t.translations.get(target_lang)
    ]
    cues = []
    issues: list[EnforcementIssue] = []
    sources: list[str] = []
    hyps: list[str] = []
    for s in sentences:
        # Condensation budget: chars readable in the on-screen window at target cps.
        window = max(s.end - s.start, spec.min_dur)
        budget = max(int(spec.max_cps * window), 8)
        txt, iss = translate_segment(
            translator,
            glossary,
            s.text,
            target_lang,
            full_context=context,
            reflect=reflect,
            char_budget=budget,
            exemplar_bank=exemplar_bank,
        )
        if target_lang == "zh-Hans":
            txt = normalize_zh_punct(txt)
        elif target_lang == "ja":
            txt = normalize_ja_punct(txt)
        cues += resegment_translation(txt, s.start, s.end, spec, keep_together=keep)
        issues += iss
        sources.append(s.text)
        hyps.append(txt)
    cues = snap_gaps(cues, spec)
    return LocalizeResult(Subtitle(target_lang, cues).renumber(), issues, sources, hyps)
