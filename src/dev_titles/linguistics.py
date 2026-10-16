"""Per-language linguistic helpers for boundary-aware segmentation.

Ported and trimmed from MX Player's `MXVideoTranscode` `platform/ai_subtitle/
core/macro_utils.py` (author zadongqi@) — kept the dependency-free pieces we need
(word joiner, end-punctuation / comma / connector break points, repetition
detection) and reduced the language tables to the languages we localize
(en/zh/ja + the EU pilot set). Full multi-language tables live in the source.

We use these to pick *where* to break a CJK line (after a sentence-final mark,
then a comma, then a connector word) instead of a blind greedy wrap.
"""

from __future__ import annotations

import unicodedata

# Languages that put spaces between words (join with " ") vs. not (join with "").
SPLIT_WITH_SPACE = {"en", "fr", "de", "es", "it"}
SPLIT_WITHOUT_SPACE = {"zh", "zh-Hans", "zh-CN", "ja", "ja-JP", "th", "lo", "my", "km"}

# Sentence-final punctuation by language (best break points).
END_PUNCT = {
    "en": {".", "?", "!"},
    "fr": {".", "?", "!"},
    "de": {".", "?", "!"},
    "es": {".", "?", "!", "¡", "¿"},
    "it": {".", "?", "!"},
    "zh-Hans": {"。", "！", "？", "…"},
    "ja": {"。", "！", "？", "…"},
}

# In-sentence comma / pause marks (second-best break points).
COMMA_PUNCT = {
    "en": {","},
    "fr": {","},
    "de": {","},
    "es": {","},
    "it": {","},
    "zh-Hans": {"，", "、", "；", "："},
    "ja": {"、", "，", "；", "："},
}

# Conjunctions/connectors: good to break *before* (subset of the MX tables).
CONNECTORS = {
    "en": {
        "and",
        "or",
        "but",
        "so",
        "because",
        "although",
        "if",
        "when",
        "where",
        "which",
        "that",
        "though",
    },
    "fr": {"et", "ou", "mais", "car", "parce que", "quand", "où", "si", "que"},
    "de": {"und", "oder", "aber", "weil", "wenn", "wo", "dass", "denn", "obwohl"},
    "es": {"y", "o", "pero", "porque", "cuando", "donde", "si", "aunque", "que"},
    "it": {"e", "o", "ma", "perché", "quando", "dove", "se", "sebbene", "che"},
    "zh-Hans": {
        "但是",
        "当",
        "虽然",
        "如果",
        "可是",
        "因为",
        "还是",
        "而且",
        "尽管",
        "跟",
        "即使",
        "或",
        "和",
        "所以",
    },
    "ja": {
        "または",
        "ため",
        "なら",
        "だから",
        "けれども",
        "それで",
        "ので",
        "そして",
        "しかし",
        "のに",
        "が",
    },
}


def get_joiner(lang: str) -> str:
    """Return the token joiner for a language ("" for CJK, " " otherwise)."""
    if lang in SPLIT_WITHOUT_SPACE:
        return ""
    return " "


def end_punct(lang: str) -> set[str]:
    return END_PUNCT.get(lang, {".", "?", "!"})


def comma_punct(lang: str) -> set[str]:
    return COMMA_PUNCT.get(lang, {","})


def connectors(lang: str) -> set[str]:
    return CONNECTORS.get(lang, set())


def is_punctuation(char: str) -> bool:
    return unicodedata.category(char).startswith("P")


def is_repeated_text(text: str, threshold: float = 0.6, max_n: int = 10) -> tuple[bool, str | None]:
    """Detect degenerate repetition — a faithfully-transcribed speaker stutter, an ASR
    stuck-decoder loop, or an LLM repetition artifact (this only flags the pattern; it does
    not assume a cause).

    Returns (is_repeated, repeating_unit). Handles both space-delimited text and
    CJK (no spaces): single-char runs, then n-gram periodicity. Ported from MX
    `macro_utils.is_repeated_text`.
    """
    words = text.split()
    if len(words) <= 1:
        text_len = len(text)
        if text_len < 2:
            return False, None
        top_char = max(set(text), key=text.count)
        if text.count(top_char) / text_len >= threshold:
            return True, top_char
        for n in range(2, min(max_n + 1, text_len // 2 + 1)):
            ngrams = [text[i : i + n] for i in range(text_len - n + 1)]
            top = max(set(ngrams), key=ngrams.count)
            coverage = ngrams.count(top) * n / text_len
            if coverage >= threshold:
                positions = [i for i in range(text_len - n + 1) if text[i : i + n] == top]
                distances = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
                if distances:
                    period = max(set(distances), key=distances.count)
                    unit = text[:period]
                    if period and text.count(unit) * period / text_len >= threshold:
                        return True, unit
                return True, top
        return False, None
    top_word = max(set(words), key=words.count)
    if words.count(top_word) / len(words) >= threshold:
        return True, top_word
    return False, None
