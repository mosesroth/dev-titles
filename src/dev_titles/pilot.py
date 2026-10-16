"""Pilot runner — the repeatable, multi-clip / multi-engine harness.

Runs a fixed set of clips through one or more candidate translation engines into
each target language, scoring every cell identically so the results are
comparable: this is how we answer "which engine is superior" and how we catch
regressions after a prompt/model/glossary change.

For each (clip, engine, language) it: translates each EN cue (draft → reflect →
glossary-enforce), re-segments for the target language, validates, and optionally
QE-scores. It writes a review bundle per (clip, engine) and a top-level
PILOT_REPORT.md ranking engines per language.

Engine-agnostic: pass any Translator (CLI, Anthropic API, Bedrock, or a fake).
QE is optional — pass a QeScorer (e.g. CometKiwiScorer, or GembaMqmScorer over a
CLI judge) or leave it None to skip scoring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .eval.qe import QeResult, QeScorer, below_threshold_segments, rank_engines
from .glossary import Glossary
from .model import Subtitle
from .review import LangOutput, build_review_bundle
from .translate import Translator, localize
from .validate.rules import validity_report

DEFAULT_LANGS = ("zh-Hans", "ja")


@dataclass
class Clip:
    name: str
    en: Subtitle


@dataclass
class EngineSpec:
    name: str
    translator: Translator
    reflect: bool = True


@dataclass
class CellResult:
    clip: str
    engine: str
    language: str
    subtitle: Subtitle
    missing_terms: int
    fixed_terms: int
    error_cues: int
    total_cues: int
    qe: QeResult | None
    weak_segments: list[int]
    elapsed_s: float


@dataclass
class PilotReport:
    cells: list[CellResult]
    files: list[str] = field(default_factory=list)

    def rankings(self) -> dict[str, list[tuple[str, float]]]:
        """Per language: engines ranked by mean QE system score (if QE ran)."""
        out: dict[str, list[tuple[str, float]]] = {}
        langs = {c.language for c in self.cells if c.qe is not None}
        for lang in sorted(langs):
            by_engine: dict[str, list[float]] = {}
            for c in self.cells:
                if c.language == lang and c.qe is not None:
                    by_engine.setdefault(c.engine, []).append(c.qe.system_score)
            means = {e: QeResult([], sum(v) / len(v), "mean") for e, v in by_engine.items()}
            out[lang] = rank_engines(means)
        return out


def _translate_clip(
    clip: Clip,
    engine: EngineSpec,
    glossary: Glossary,
    lang: str,
    qe_scorer: QeScorer | None,
    qe_threshold: float,
) -> tuple[LangOutput, CellResult]:
    t0 = time.perf_counter()
    result = localize(engine.translator, glossary, clip.en, lang, reflect=engine.reflect)
    elapsed = time.perf_counter() - t0

    sub = result.subtitle
    issues = result.issues
    qe = qe_scorer.score(result.sources, result.hypotheses, "en", lang) if qe_scorer else None
    weak = below_threshold_segments(qe, qe_threshold) if qe else []
    rep = validity_report(sub)

    out = LangOutput(
        language=lang,
        subtitle=sub,
        enforcement_issues=issues,
        qe_metric=qe.metric if qe else None,
        qe_system_score=qe.system_score if qe else None,
        qe_weak_segments=weak,
    )
    cell = CellResult(
        clip=clip.name,
        engine=engine.name,
        language=lang,
        subtitle=sub,
        missing_terms=sum(1 for i in issues if i.kind == "missing"),
        fixed_terms=sum(1 for i in issues if i.kind == "fixed"),
        error_cues=rep["cues_with_errors"],
        total_cues=rep["total_cues"],
        qe=qe,
        weak_segments=weak,
        elapsed_s=elapsed,
    )
    return out, cell


def run_pilot(
    clips: list[Clip],
    engines: list[EngineSpec],
    glossary: Glossary,
    *,
    target_langs=DEFAULT_LANGS,
    qe_scorer: QeScorer | None = None,
    qe_threshold: float = 0.7,
    out_dir: str | Path = "pilot_out",
) -> PilotReport:
    out_root = Path(out_dir)
    cells: list[CellResult] = []
    files: list[str] = []
    for clip in clips:
        for engine in engines:
            lang_outputs: list[LangOutput] = []
            for lang in target_langs:
                lo, cell = _translate_clip(clip, engine, glossary, lang, qe_scorer, qe_threshold)
                lang_outputs.append(lo)
                cells.append(cell)
            bundle_dir = out_root / _slug(clip.name) / _slug(engine.name)
            files += build_review_bundle(bundle_dir, clip.en, lang_outputs)
    report = PilotReport(cells=cells, files=files)
    report_md = out_root / "PILOT_REPORT.md"
    out_root.mkdir(parents=True, exist_ok=True)
    report_md.write_text(render_pilot_report(report, engines, target_langs), encoding="utf-8")
    report.files.append(str(report_md))
    return report


def render_pilot_report(report: PilotReport, engines: list[EngineSpec], target_langs) -> str:
    lines = ["# Pilot report", ""]
    clips = sorted({c.clip for c in report.cells})
    lines.append(
        f"- **Clips:** {len(clips)} | **Engines:** "
        f"{', '.join(e.name for e in engines)} | **Languages:** "
        f"{', '.join(target_langs)}"
    )

    rankings = report.rankings()
    if rankings:
        lines += ["", "## Engine ranking (mean QE system score, higher = better)"]
        for lang, ranked in rankings.items():
            lines.append(f"- **{lang}:** " + " > ".join(f"{e} ({s:.3f})" for e, s in ranked))

    lines += [
        "",
        "## Per-cell detail",
        "",
        "| clip | engine | lang | cues | err cues | fixed | missing | QE | weak | s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in report.cells:
        qe = f"{c.qe.system_score:.3f}" if c.qe else "—"
        lines.append(
            f"| {c.clip} | {c.engine} | {c.language} | {c.total_cues} | "
            f"{c.error_cues} | {c.fixed_terms} | {c.missing_terms} | {qe} | "
            f"{len(c.weak_segments)} | {c.elapsed_s:.1f} |"
        )
    lines += ["", "_Bundles per (clip, engine) are in this directory tree._", ""]
    return "\n".join(lines)


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


# ---- clip loaders --------------------------------------------------------------
def clip_from_caption(path: str | Path, name: str | None = None) -> Clip:
    """Load an EN clip from an existing .vtt/.srt caption file."""
    from .formats import read_srt, read_vtt

    p = Path(path)
    en = read_srt(p, "en") if p.suffix.lower() == ".srt" else read_vtt(p, "en")
    return Clip(name=name or p.stem, en=en)


def clip_from_transcribe_json(path: str | Path, name: str | None = None) -> Clip:
    """Load an EN clip from AWS Transcribe output JSON (builds EN cues)."""
    import json

    from .cue_builder import build_en_cues
    from .transcribe import from_transcribe_json

    p = Path(path)
    transcript = from_transcribe_json(json.loads(p.read_text(encoding="utf-8")), "en")
    return Clip(name=name or p.stem, en=build_en_cues(transcript))


def load_clip(path: str | Path, name: str | None = None) -> Clip:
    """Dispatch by extension: .json -> Transcribe output; else caption file."""
    return (
        clip_from_transcribe_json(path, name)
        if Path(path).suffix.lower() == ".json"
        else clip_from_caption(path, name)
    )
