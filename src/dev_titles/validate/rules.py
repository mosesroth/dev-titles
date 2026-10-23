"""Per-language validation (pipeline step [6]).

Every emitted cue is checked against the LangSpec plus CJK-specific rules. A
Violation carries a severity: "error" blocks publish; "warn" is surfaced for
human review. Timing/gap checks run at the subtitle level (they need neighbors).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import (
    KINSOKU_NO_LINE_END,
    KINSOKU_NO_LINE_START,
    ZH_DISALLOWED_RUNS,
    LangSpec,
    spec_for,
)
from ..linguistics import is_repeated_text
from ..model import Cue, Subtitle
from ..textwidth import display_width

_CJK_SOUND_TAG = re.compile(
    r"[\[［【(（]\s*(?:背景)?(?:音乐|音楽|BGM|掌声|拍手|欢呼|歓声|笑声|音效|効果音)[^\]］】)）]{0,6}[\]］】)）]"
)
_CJK_LAUGH = re.compile(r"[(（]笑[)）]")


def _sound_tag():
    from ..cue_builder import _SOUND_TAG  # the same pattern strip_sound_tags removes

    return _SOUND_TAG


@dataclass
class Violation:
    cue_index: int | None
    rule: str
    severity: str  # "error" | "warn"
    detail: str


def validate_cue(cue: Cue, spec: LangSpec, conversational: bool = False) -> list[Violation]:
    v: list[Violation] = []
    idx = cue.index

    # Line count.
    if len(cue.lines) > spec.max_lines:
        v.append(Violation(idx, "max_lines", "error", f"{len(cue.lines)} lines > {spec.max_lines}"))

    # Chars per line.
    for ln in cue.lines:
        w = display_width(ln, cjk=spec.cjk)
        if w > spec.max_cpl:
            v.append(Violation(idx, "max_cpl", "error", f"line width {w} > {spec.max_cpl}: {ln!r}"))

    # Duration. Tolerance: a cue re-timed to EXACTLY min_dur can land a hair under (0.8329999s)
    # from float rounding — that's not a real violation, so allow a 1ms epsilon (a boxed
    # interjection extended to the floor should pass, not flag).
    dur = cue.duration
    if dur < spec.min_dur - 1e-3:
        v.append(Violation(idx, "min_dur", "error", f"duration {dur:.3f}s < {spec.min_dur}s"))
    if dur > spec.max_dur:
        v.append(Violation(idx, "max_dur", "warn", f"duration {dur:.3f}s > {spec.max_dur}s"))

    # Reading speed (cps) — measured on visible chars, newlines excluded.
    chars = display_width("".join(cue.lines), cjk=spec.cjk)
    if dur > 0:
        cps = chars / dur
        # Validate against the READABILITY ceiling (warn_cps), not the condensation
        # density target (max_cps). max_cps is tuned to house density and drives the
        # char_budget; using it as the warn threshold flags ~10%+ of good cues by
        # construction. warn_cps falls back to max_cps for langs that don't set it.
        warn_cps = spec.warn_cps or spec.max_cps
        if cps > warn_cps:
            # Reading speed is a WARNING, not a hard error (Prime Video standard):
            # completeness-first output on dense technical speech legitimately
            # exceeds the cap — surface it for review, don't fail on it.
            v.append(Violation(idx, "max_cps", "warn", f"{cps:.1f} cps > {warn_cps}"))

    # Degenerate repetition (ASR / LLM hallucination) — warn.
    repeated, unit = is_repeated_text("".join(cue.lines))
    if repeated:
        v.append(
            Violation(idx, "repetition", "warn", f"looks like repeated content (unit={unit!r})")
        )

    # Non-speech SOUND TAGS ([music], (applause), ♪, [音乐] / ［音楽］): raw caption/ASR artifacts are
    # never shipped (DCMP: background sound is captioned only when essential, and then deliberately).
    # They slipped into a delivery when a fix was carried onto a live YouTube caption that still
    # had '[music]'. A Japanese （笑） is a reviewer's choice -> warn.
    text = " ".join(cue.lines)
    if _sound_tag().search(text) or _CJK_SOUND_TAG.search(text):
        v.append(Violation(idx, "sound_tag", "error", f"non-speech tag in the text: {text[:60]!r}"))
    elif _CJK_LAUGH.search(text):
        v.append(Violation(idx, "sound_tag", "warn", f"laughter marker in the text: {text[:60]!r}"))

    # CJK-specific.
    if spec.cjk:
        v.extend(_validate_cjk(cue, spec, conversational))
    return v


def _validate_cjk(cue: Cue, spec: LangSpec, conversational: bool = False) -> list[Violation]:
    v: list[Violation] = []
    idx = cue.index
    for ln in cue.lines:
        if not ln:
            continue
        if ln[0] in KINSOKU_NO_LINE_START:
            v.append(
                Violation(
                    idx, "kinsoku_line_start", "error", f"line starts with forbidden {ln[0]!r}"
                )
            )
        if ln[-1] in KINSOKU_NO_LINE_END:
            v.append(
                Violation(idx, "kinsoku_line_end", "error", f"line ends with forbidden {ln[-1]!r}")
            )
    joined = "".join(cue.lines)
    # Stray ASCII period in CJK text (not inside a Latin token like 'Next.js' / 'v1.2', not an
    # ellipsis '...'): an editing artifact — one reviewed ja file carried 8 of them
    # ('.良い質問ですね', '使えます.'). Neither ja nor zh subtitles use '.', so it's an error.
    if re.search(r"(?<![A-Za-z0-9.])\.(?![A-Za-z0-9.])", joined):
        v.append(Violation(idx, "stray_ascii_period", "error", "stray ASCII '.' in CJK text"))
    if spec.lang == "zh-Hans":
        for run in ZH_DISALLOWED_RUNS:
            if run in joined:
                v.append(
                    Violation(idx, "zh_punct_run", "warn", f"disallowed punctuation run {run!r}")
                )
        # zh second person: 您 is the house default in BOTH registers (native reviewers changed
        # our conversational 你 to 您 141:4 across four videos).
        # Only the PLURAL 你们 stays 你, since 您们 is unidiomatic.
        if re.search(r"你(?!们)", joined):
            v.append(
                Violation(
                    idx,
                    "zh_informal_you",
                    "warn",
                    "uses 你 (informal); house register expects 您 (plural 你们 is fine)",
                )
            )
    if spec.lang == "ja":
        for mark in ("、", "。"):  # JA subtitle convention: spaces, not 、。
            if mark in joined:
                v.append(
                    Violation(
                        idx,
                        "ja_punctuation",
                        "error",
                        f"JA subtitles should not contain {mark!r} " "(use a full-width space)",
                    )
                )
    return v


def validate_subtitle(sub: Subtitle, conversational: bool = False) -> list[Violation]:
    spec = spec_for(sub.language)
    out: list[Violation] = []
    sub.renumber()
    for cue in sub.cues:
        out.extend(validate_cue(cue, spec, conversational))

    # Neighbor checks: monotonic, non-overlapping, min gap.
    min_gap = spec.min_gap_frames / spec.fps
    eps = 1e-3
    for a, b in zip(sub.cues, sub.cues[1:]):
        gap = b.start - a.end
        if gap < -eps:
            out.append(
                Violation(
                    b.index, "overlap", "error", f"cue {b.index} starts before cue {a.index} ends"
                )
            )
        elif eps < gap < min_gap:
            # Only a small NON-ZERO gap can flicker; contiguous cues (gap≈0)
            # within a continuous sentence are fine.
            out.append(Violation(b.index, "min_gap", "warn", f"gap {gap:.3f}s < {min_gap:.3f}s"))
    return out


def validity_report(sub: Subtitle, conversational: bool = False) -> dict:
    """Aggregate report over a subtitle : per-rule violation counts,
    ratios, and totals. Useful as a QC summary before human review."""
    violations = validate_subtitle(sub, conversational)
    n = len(sub.cues) or 1
    by_rule: dict[str, int] = {}
    bad_cues = set()
    for viol in violations:
        by_rule[viol.rule] = by_rule.get(viol.rule, 0) + 1
        if viol.severity == "error":
            bad_cues.add(viol.cue_index)
    return {
        "language": sub.language,
        "total_cues": len(sub.cues),
        "cues_with_errors": len(bad_cues),
        "error_cue_ratio": round(len(bad_cues) / n, 3),
        "violations_by_rule": dict(sorted(by_rule.items())),
        "total_violations": len(violations),
    }
