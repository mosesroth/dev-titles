"""Per-language subtitle constraints and CJK punctuation rules.

Sources: Netflix Timed Text Style Guide (per-language), general broadcast norms.
Limits are enforced by validate/ and targeted by segment/. Keep this the single
source of truth so the segmenter and validator never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LangSpec:
    lang: str
    max_lines: int = 2
    max_cpl: int = 42  # characters per line (display-width based for CJK)
    max_cps: float = 17.0  # characters per second — CONDENSATION target (drives char_budget)
    warn_cps: float | None = None  # READABILITY ceiling for validation only; falls back to
    # max_cps. Decoupled because max_cps is tuned to house DENSITY
    # (used as the condensation budget), while the validator should
    # flag only cues genuinely too fast to READ — otherwise a density
    # target set at house p90 flags ~10%+ of good cues by construction.
    hard_cpl: int | None = None  # validation ERROR ceiling; between max_cpl and this is a warning.
    # max_cpl stays the SEGMENTER's target. ja: native-reviewed lines run to 21.5 (the JP SG says
    # "about 16"), so a 20-21 wide line a reviewer chose is not a defect (in one delivery 18 of 22
    # ja overruns were the reviewer's own line breaks).
    min_dur: float = 0.833  # seconds (min 20 frames @ 24fps ~ 0.833s)
    max_dur: float = 7.0
    min_gap_frames: int = 2  # min gap between adjacent cues
    fps: float = 24.0
    cjk: bool = False  # apply full-width display-width + CJK line-break rules


# Per-language specs. EU languages share the Latin defaults; CJK override CPL/cps.
SPECS: dict[str, LangSpec] = {
    # EN calibrated to the DCMP educational standard (Captioning Key): adult
    # ceiling ~160 wpm ≈ 17 cps (tighter than the 20 cps entertainment norm, since
    # dense technical vocabulary needs more reading time); cue duration ~1.5–6 s.
    "en": LangSpec("en", max_cpl=42, max_cps=17.0, min_dur=1.5, max_dur=6.0),
    "fr": LangSpec("fr", max_cpl=42, max_cps=17.0),
    "de": LangSpec("de", max_cpl=42, max_cps=17.0),
    "es": LangSpec("es", max_cpl=42, max_cps=17.0),
    "it": LangSpec("it", max_cpl=42, max_cps=17.0),
    # cps targets raised from Netflix general-audience caps (zh 9 / ja 4) to a
    # developer-conference audience reading their own domain.
    # Tune here; the pipeline also condenses translations to cps × on-screen-seconds.
    # CPL/cps calibrated to a reference set of human-made subtitles (measured across
    # the parallel corpus): ja cpl p50 15/p90 19, cps p50 6.4/p90 8.5/max 9.7.
    # The earlier ja cpl=13 was ~30% under house width -> forced mid-word cuts
    # (no 文節 boundary fit the cap) and over-condensation that drove omission.
    # Typical CJK subtitle style guides: zh "20 chars/line", ja "16文字程度" (~16,
    # approx) — but the reference human subs measure ja cpl p50 15/p90 19, so a hard 16 cap
    # over-breaks below their actual practice. zh -> 20 (SG explicit); ja -> 19 (house p90,
    # honoring the SG's tighter-than-20 direction without forcing mid-文節 cuts).
    # warn_cps = readability ceiling (validation). Set above house MAX (ja 9.7 / zh comparable)
    # so the REVIEW flags only genuinely-too-fast cues — chiefly ultra-short backchannel cues,
    # which are ALSO caught by min_dur, not the ~10% that merely sit above the density p90.
    "zh-Hans": LangSpec("zh-Hans", max_cpl=20, max_cps=12.0, warn_cps=16.0, cjk=True),
    "ja": LangSpec(
        "ja", max_cpl=19, hard_cpl=22, max_cps=8.5, warn_cps=13.0, min_dur=0.5, cjk=True
    ),
}


def spec_for(lang: str) -> LangSpec:
    if lang in SPECS:
        return SPECS[lang]
    # Fall back to Latin defaults for unlisted EU languages.
    return LangSpec(lang)


# ---- Japanese kinsoku shori (禁則処理) ----------------------------------------
# Characters forbidden at the START of a line (行頭禁則): closing marks, small
# kana, sound marks, iteration marks, the long-vowel mark.
KINSOKU_NO_LINE_START = set(
    "、。，．・：；！？）］｝」』】〕〉》〟”’"
    "ぁぃぅぇぉっゃゅょゎ"
    "ァィゥェォッャュョヮヵヶ"
    "ーゝゞ々〻"
)

# Characters forbidden at the END of a line (行末禁則): opening marks.
KINSOKU_NO_LINE_END = set("（［｛「『【〔〈《〝“‘")


# ---- Chinese (Simplified) punctuation normalization ---------------------------
# At cue boundaries, a trailing full-width comma/period is replaced by a single
# full-width space per the ZH style guide. Ellipsis is U+2026. Doubled/mixed
# terminal punctuation (！？ ／？？ ／！！) is disallowed.
ZH_CUE_BOUNDARY_STRIP = set("，。")
ZH_ELLIPSIS = "…"
ZH_DISALLOWED_RUNS = ["！？", "？？", "！！", "?!", "!!", "??"]
