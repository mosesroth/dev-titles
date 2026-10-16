"""Review bundle (the deliverable): subtitle files + a report for the docs team.

The pipeline's output target is a folder the docs team reviews — NOT a publish to
YouTube. For each target language we write WebVTT (canonical) + SRT, alongside the
EN master, and a human-readable REVIEW.md summarizing per-language validation,
deterministic glossary enforcement (what was auto-fixed, what's still missing),
and QE scores. Optionally the bundle can be uploaded to Frame.io as file assets
for review (see publish/frameio.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .formats import write_srt, write_vtt
from .glossary import EnforcementIssue
from .model import Subtitle
from .validate.rules import validity_report


@dataclass
class LangOutput:
    language: str
    subtitle: Subtitle
    enforcement_issues: list[EnforcementIssue] = field(default_factory=list)
    qe_metric: str | None = None
    qe_system_score: float | None = None
    qe_weak_segments: list[int] = field(default_factory=list)
    residuals: list[tuple[str, str]] = field(
        default_factory=list
    )  # (EN source, still-missing text)
    qe_suspects: list[dict] = field(default_factory=list)  # reference-free translation QE findings
    qe_pass: bool | None = None
    conversational: bool = False  # peer/interview register — validator won't flag 你 as informal
    term_pins: dict = field(
        default_factory=dict
    )  # {lang:{source:{canonical,avoid}}} auto-harvested
    # term fixes — persist to vocab + surface for promotion


def render_review_markdown(en: Subtitle, outputs: list[LangOutput]) -> str:
    lines: list[str] = ["# Subtitle review bundle", ""]
    lines.append(f"- **Source (EN):** {len(en.cues)} cues")
    for o in outputs:
        rep = validity_report(o.subtitle, o.conversational)
        missing = [i for i in o.enforcement_issues if i.kind == "missing"]
        fixed = [i for i in o.enforcement_issues if i.kind == "fixed"]
        lines += [
            "",
            f"## {o.language} — {rep['total_cues']} cues",
            "",
            f"- **Validation:** {rep['cues_with_errors']} cue(s) with errors "
            f"({rep['error_cue_ratio']:.0%}); {rep['total_violations']} total violations.",
        ]
        if rep["violations_by_rule"]:
            by_rule = ", ".join(f"{k}×{v}" for k, v in rep["violations_by_rule"].items())
            lines.append(f"  - By rule: {by_rule}")
        lines.append(
            f"- **Glossary enforcement:** {len(fixed)} auto-fixed, "
            f"{len(missing)} needs attention."
        )
        for m in missing:
            lines.append(f"  - ⚠️ `{m.term}`: expected {m.expected!r} — {m.detail}")
        _pins = (o.term_pins or {}).get(o.language, {})
        if _pins:
            lines.append(
                f"- **Term-pin auto-repair:** {len(_pins)} term(s) fixed deterministically "
                f"across all cues (from judge critiques; consider promoting recurring ones "
                f"to the shared glossary):"
            )
            for src, spec in _pins.items():
                lines.append(
                    f"  - `{src}` → {spec['canonical']!r} (was {spec.get('avoid', [''])[0]!r})"
                )
        if o.residuals:
            lines.append(
                f"- **⚠️ Coverage residuals ({len(o.residuals)}):** the completeness gate "
                f"could not splice these source clauses back — VERIFY against the source "
                f"and add the missing meaning by hand:"
            )
            for src, miss in o.residuals:
                lines.append(f"  - EN: {src!r}")
                lines.append(f"    - still missing: {miss!r}")
        if o.qe_system_score is not None:
            gate = "" if o.qe_pass is None else (" ✅ PASS" if o.qe_pass else " ❌ FAIL")
            lines.append(
                f"- **QE ({o.qe_metric}):** score {o.qe_system_score:.1f}{gate}; "
                f"{len(o.qe_weak_segments)} weak segment(s)"
                + (
                    f" (cues {', '.join(str(i + 1) for i in o.qe_weak_segments)})"
                    if o.qe_weak_segments
                    else ""
                )
                + "."
            )
        if o.qe_suspects:
            lines += _render_qe_tiers(o.qe_suspects)
        lines.append(f"- **Files:** `{o.language}.vtt`, `{o.language}.srt`")
    lines += [
        "",
        "---",
        "_Note: inferred zh product terms need native-reviewer confirmation._",
        "",
    ]
    return "\n".join(lines)


def _render_qe_tiers(suspects: list[dict]) -> list[str]:
    """Render QE findings as an ADJUDICATED sheet, not a flat dump: split into CONFIRMED
    (a judge flagged major/critical, BOTH judges flagged the cue, or a deterministic
    number/term/garble issue → fix or adjudicate) vs SINGLE-JUDGE/SUBJECTIVE (one judge, a
    minor → likely style/taste, confirm-or-dismiss). Deterministic false positives (DNT
    terms, coverage-verified omissions) are already suppressed upstream and don't appear."""
    minors_by_cue: dict = {}
    for s in suspects:
        if s["kind"].startswith("mqm-minor"):
            j = "codex" if "codex" in s["kind"] else ("claude" if "claude" in s["kind"] else "?")
            minors_by_cue.setdefault(s.get("i"), set()).add(j)

    def tier(s: dict) -> str:
        k = s["kind"]
        if k.startswith(("mqm-major", "mqm-critical")):
            return "confirmed"
        if k in ("dropped-token", "latin-typo?", "term-inconsistency"):
            return "confirmed"
        if k.startswith("mqm-minor"):  # corroborated minor = both judges
            return "confirmed" if len(minors_by_cue.get(s.get("i"), ())) >= 2 else "subjective"
        return "subjective"

    confirmed = [s for s in suspects if tier(s) == "confirmed"]
    subjective = [s for s in suspects if tier(s) == "subjective"]

    def fmt(s: dict) -> str:
        where = f"cue {s['i'] + 1}" if s.get("i") is not None else "document"
        line = f"    - [{s['kind']} @ {where}] {s['detail']}"
        return line + (f"\n      EN: {s['source'][:100]!r}" if s.get("source") else "")

    out = [
        f"- **QE review — {len(confirmed)} for human judgment, {len(subjective)} single-judge/"
        f"subjective (spot-check):** _(false positives + auto-fixable term inconsistencies "
        f"resolved upstream)_"
    ]
    if confirmed:
        out.append(
            "  - _HUMAN-JUDGMENT — auto-repair RAN on these and could not safely resolve "
            "them (elliptical/interrupted or malformed source, or the two judges disagree); "
            "a person must decide:_"
        )
        out += [fmt(s) for s in confirmed[:25]]
    if subjective:
        out.append(
            "  - _SINGLE-JUDGE / SUBJECTIVE (one judge, minor — likely style/taste; "
            "confirm or dismiss):_"
        )
        out += [fmt(s) for s in subjective[:25]]
    return out


def build_review_bundle(out_dir: str | Path, en: Subtitle, outputs: list[LangOutput]) -> list[str]:
    """Write the EN master, per-language VTT+SRT, and REVIEW.md. Returns paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def _w(name: str, text: str) -> None:
        p = out / name
        p.write_text(text, encoding="utf-8")
        written.append(str(p))

    from .resegment import normalize_en_layout

    en = normalize_en_layout(en)  # EN deliverable always meets its layout spec
    _w("en.vtt", write_vtt(en))
    for o in outputs:
        _w(f"{o.language}.vtt", write_vtt(o.subtitle))
        _w(f"{o.language}.srt", write_srt(o.subtitle))
    _w("REVIEW.md", render_review_markdown(en, outputs))
    return written
