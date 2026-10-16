"""Translation + reflect-refine prompt builders.

These implement the R2 prompt strategy: full-transcript context, per-segment
glossary injection, register control, and a windowed <TRANSLATE_THIS> target so
the model sees surrounding text but only translates the current segment. The
reflect prompt drives the accuracy/fluency/style/terminology critique pass.

Kept provider-neutral (plain strings). The bedrock adapter wraps these in the
Messages API. Model/region are configured per deployment; see the stub.
"""

from __future__ import annotations

import re

from .base import TranslationRequest

_LANG_NAMES = {"zh-Hans": "Simplified Chinese", "ja": "Japanese", "en": "English"}

_FINAL_TAG = re.compile(r"<final>(.*?)</final>", re.IGNORECASE | re.DOTALL)
_FINAL_LABEL = re.compile(r"(?:final|final translation)\s*[:：]\s*", re.IGNORECASE)


def extract_final_translation(text: str) -> str:
    """Pull the final translation out of a reflect-pass response that may include
    commentary. Prefers <final>…</final>; else text after the last 'Final:' label;
    else the last non-empty line. Strips markdown/quote decoration."""
    m = None
    for m in _FINAL_TAG.finditer(text):
        pass  # keep the last match
    if m:
        text = m.group(1)
    else:
        parts = _FINAL_LABEL.split(text)
        if len(parts) > 1:
            text = parts[-1]
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    chosen = lines[-1] if lines else ""
    chosen = chosen.strip().strip("`").strip('"').strip("“”").strip()
    chosen = re.sub(r"^\**\s*", "", chosen).rstrip("*").strip()
    return chosen


def _glossary_block(req: TranslationRequest) -> str:
    if not req.glossary_terms:
        return "(none for this segment)"
    lines = []
    for t in req.glossary_terms:
        target = t.translations.get(req.target_lang, "")
        tag = "MUST USE" if t.enforce == "hard" else "prefer"
        note = f" — {t.notes}" if t.notes else ""
        sense = t.sense_hint(req.target_lang) if hasattr(t, "sense_hint") else ""
        if sense:
            note += f" [sense: {sense}]"
        bad = t.avoid.get(req.target_lang, []) if hasattr(t, "avoid") else []
        if bad:
            note += f" [NEVER: {', '.join(bad)}]"
        lines.append(f'  - "{t.source}" -> "{target}" [{tag}]{note}')
    return "\n".join(lines)


def _exemplar_block(req: TranslationRequest) -> str:
    """Approved human EN->target pairs from your team's own subtitles. These
    are the style anchors: the model should MATCH their register, mood, phrasing
    rhythm, and (critically) their sparse punctuation density — not copy their
    wording. Empty -> a neutral instruction so the prompt stays well-formed."""
    if not req.exemplars:
        return "(no exemplars available — follow the style notes and policy above)"
    tgt = _LANG_NAMES.get(req.target_lang, req.target_lang)
    lines = []
    for en, human in req.exemplars:
        lines.append(f"  EN: {en}\n  {tgt}: {human}")
    return "\n".join(lines)


def _budget_clause(req: TranslationRequest) -> str:
    if not req.char_budget:
        return ""
    # Keep CONTENT, condense FILLER. Earlier "never drop anything" over-corrected
    # into verbosity (52% of cues blew the cps cap on one video); professional
    # subtitles condense to read comfortably. So: fit ~budget by tightening phrasing
    # AND dropping discourse filler/greetings/redundancy the way a subtitler does —
    # but never drop a factual clause, technical detail, identifier, or enumerated
    # item. (The coverage gate separately guarantees no real content is lost.)
    return (
        f"Read comfortably in the available time (~{req.char_budget} characters): "
        f"choose tight natural phrasing and OMIT discourse filler, greetings, "
        f"hedges, and redundant restatements (e.g. 'welcome', 'a little bit', "
        f"'happy coding') as a professional subtitler would. But keep ALL "
        f"substantive content — every factual clause, condition, technical detail, "
        f"identifier, and enumerated item. Condense filler, never facts. "
    )


def build_translate_prompt(req: TranslationRequest) -> str:
    tgt = _LANG_NAMES.get(req.target_lang, req.target_lang)
    src = _LANG_NAMES.get(req.source_lang, req.source_lang)
    return f"""You are a professional subtitle translator localizing technical developer video content from {src} to {tgt}.

REGISTER: {req.register or "professional, natural spoken register"}
STYLE NOTES: {req.style_notes or "(none)"}

LOCALIZATION POLICY (follow exactly):
{req.localization_policy or "Localize descriptive terms; keep only genuine product names in the source script."}

GLOSSARY (apply exactly — including the 'prefer' terms; localize descriptive
parts of product names per the policy above, keep only the proper name in Latin):
{_glossary_block(req)}

STYLE EXEMPLARS — approved human {tgt} subtitles for this product family. MATCH
their register, mood, and phrasing rhythm. Learn the house style from them; do
NOT copy their wording:
{_exemplar_block(req)}

PUNCTUATION: {req.punct_note or "match the exemplars' punctuation density."}

Full transcript context (for reference — do NOT translate this whole thing):
<CONTEXT>
{req.full_context}
</CONTEXT>

Translate ONLY the text between <TRANSLATE_THIS> tags into {tgt}. Translate the
FULL meaning faithfully and completely (omit nothing), reading naturally as
spoken subtitle text in a professional tone for developers. {_budget_clause(req)}Do
not add notes, quotes, or explanations — return only the translation.

<TRANSLATE_THIS>
{req.source_text}
</TRANSLATE_THIS>{_completeness_block(req)}{_correction_block(req)}"""


def _completeness_block(req: TranslationRequest) -> str:
    if not req.completeness_hint:
        return ""
    return (
        "\n\nIMPORTANT — a previous attempt was too short and DROPPED content. "
        + req.completeness_hint
        + " Include EVERY clause, enumeration item "
        "(each 'or…'/'also'/'as well as'), technical identifier/API name, and "
        "the closing call-to-action. Do not summarize the tail of the sentence."
    )


def _correction_block(req: TranslationRequest) -> str:
    if not getattr(req, "correction_hint", ""):
        return ""
    return (
        "\n\nIMPORTANT — a prior review flagged specific errors in an earlier "
        "translation of THIS segment. FIX exactly these, without introducing new "
        "errors or dropping any content: " + req.correction_hint
    )


def build_reflect_prompt(req: TranslationRequest, draft: str) -> str:
    tgt = _LANG_NAMES.get(req.target_lang, req.target_lang)
    return f"""Here is a draft {tgt} subtitle translation.

Source:
<SOURCE>
{req.source_text}
</SOURCE>

Draft:
<DRAFT>
{draft}
</DRAFT>

Critique the draft, then produce a final improved translation:
1. COMPLETENESS vs CONDENSATION — go through the SOURCE and list each SUBSTANTIVE
   content unit (factual clause, condition, technical qualifier, identifier,
   enumerated item), marking it carried or DROPPED; restore every dropped one.
   BUT do NOT preserve discourse filler, greetings, hedges, or redundant
   restatements — condense those the way a professional subtitler does to read
   within the reading-speed limit. Dropping a fact is a defect; dropping filler is
   correct. Being tighter than the source is fine when only filler was trimmed.
2. ACCURACY — any addition or mistranslation?
3. FLUENCY & NATURALNESS — natural {tgt} spoken register ({req.register})? Rewrite
   any word-for-word calque as a {tgt} speaker would actually phrase it. Prefer a
   specific verb/noun over a vague hypernym. professional, developer-facing tone.
4. MOOD & VOICE — does the speech-act match the source and the STYLE EXEMPLARS?
   (calls-to-action read as invitations/instructions, not flat statements;
   automated behaviour in the natural voice; avoid over-nominalization.)
   MODALITY: a normative "should/must/needs to" in docs/specs is a requirement —
   render it as such (ja 〜する必要があります, NOT the weak expectation 〜はず;
   zh 应/需要), not a prediction.
5. PUNCTUATION — {req.punct_note or "match the exemplars' punctuation density."}
6. TERMINOLOGY — do glossary terms match, and is the localization policy applied
   (descriptive parts localized, only proper names kept in source script)?
STYLE EXEMPLARS (match their register/mood/phrasing/punctuation density):
{_exemplar_block(req)}
LOCALIZATION POLICY:
{req.localization_policy or "(none)"}
GLOSSARY:
{_glossary_block(req)}

Output the final improved translation wrapped in <final></final> tags, with
nothing else inside the tags and no commentary after them. Example:
<final>翻訳</final>{_correction_block(req)}"""
