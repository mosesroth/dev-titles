"""Reference-free quality estimation (requirement R5) — "which engine is best".

Two scorers behind one interface, plus a pure ranking/routing layer:
  - `CometKiwiScorer` — neural QE (Unbabel/wmt22-cometkiwi-da). Lazy torch/comet
    import; the heavy model loads only when scoring is invoked.
  - `GembaMqmScorer` — LLM-as-judge (GEMBA-MQM): the model marks MQM error spans;
    we parse severities into a penalty. Takes an `invoke(prompt)->str` callable
    (reuse the Bedrock client) so it's engine-agnostic and testable.
  - `rank_engines()` / `below_threshold_segments()` — pure functions the eval
    harness uses to rank candidate engines system-level and route weak segments
    back to the translate-refine loop. No model needed — unit tested.

Score convention: higher = better for BOTH scorers. COMETKiwi returns ~0..1;
GEMBA-MQM returns a negative MQM penalty (0 = flawless, more negative = worse).
Never compare scores across scorers — only within one metric.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class QeResult:
    segment_scores: list[float]
    system_score: float
    metric: str
    details: list[dict] = field(default_factory=list)


class QeScorer(ABC):
    @abstractmethod
    def score(
        self, sources: list[str], hypotheses: list[str], src_lang: str, tgt_lang: str
    ) -> QeResult:
        raise NotImplementedError


# ---- COMETKiwi -----------------------------------------------------------------
class CometKiwiScorer(QeScorer):
    """Wraps Unbabel/wmt22-cometkiwi-da. `pip install 'dev-titles[qe]'`. The model is gated on
    Hugging Face: accept its license on the model page, then `hf auth login` once."""

    def __init__(
        self, model: str = "Unbabel/wmt22-cometkiwi-da", *, batch_size: int = 16, gpus: int = 0
    ):
        self.model_name = model
        self.batch_size = batch_size
        self.gpus = gpus
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from comet import download_model, load_from_checkpoint
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("Install QE deps: pip install 'dev-titles[qe]'") from e
            self._model = load_from_checkpoint(download_model(self.model_name))
        return self._model

    def score(self, sources, hypotheses, src_lang, tgt_lang) -> QeResult:
        if len(sources) != len(hypotheses):
            raise ValueError("sources and hypotheses must be the same length")
        data = [{"src": s, "mt": h} for s, h in zip(sources, hypotheses)]
        out = self._load().predict(data, batch_size=self.batch_size, gpus=self.gpus)
        scores = list(out["scores"] if isinstance(out, dict) else out.scores)
        system = float(out["system_score"] if isinstance(out, dict) else out.system_score)
        return QeResult(
            segment_scores=[float(s) for s in scores], system_score=system, metric="cometkiwi"
        )


# ---- GEMBA-MQM (LLM judge) -----------------------------------------------------
# MQM severity penalty weights (GEMBA-MQM style). Penalty is subtracted; the
# segment score is the negative penalty (higher/closer-to-0 = better).
MQM_WEIGHTS = {"critical": 10.0, "major": 5.0, "minor": 1.0}
_ERROR_LINE = re.compile(r"\b(critical|major|minor)\b", re.IGNORECASE)


def gemba_mqm_prompt(source: str, hypothesis: str, src_lang: str, tgt_lang: str) -> str:
    return f"""You are an expert {tgt_lang} translation evaluator using the MQM framework.
Identify error spans in the translation of the source below. For each error, output
one line: "<severity>: <short description>" where severity is one of
critical / major / minor. If there are no errors, output exactly: no-error

Source ({src_lang}):
{source}

Translation ({tgt_lang}):
{hypothesis}

Errors:"""


def parse_gemba_mqm(text: str) -> tuple[float, dict[str, int]]:
    """Parse a GEMBA-MQM judge response into (penalty, severity_counts).

    Counts one error per line that names a severity; 'no-error' -> 0 penalty.
    """
    counts = {"critical": 0, "major": 0, "minor": 0}
    if "no-error" in text.lower():
        return 0.0, counts
    for line in text.splitlines():
        m = _ERROR_LINE.search(line)
        if m:
            counts[m.group(1).lower()] += 1
    penalty = sum(MQM_WEIGHTS[sev] * n for sev, n in counts.items())
    return penalty, counts


class GembaMqmScorer(QeScorer):
    """LLM-as-judge QE. `invoke` is any callable taking a prompt -> response text
    (e.g. BedrockTranslator._invoke, or a thin lambda over messages.create)."""

    def __init__(self, invoke):
        self.invoke = invoke

    def score(self, sources, hypotheses, src_lang, tgt_lang) -> QeResult:
        seg_scores: list[float] = []
        details: list[dict] = []
        for s, h in zip(sources, hypotheses):
            resp = self.invoke(gemba_mqm_prompt(s, h, src_lang, tgt_lang))
            penalty, counts = parse_gemba_mqm(resp)
            seg_scores.append(-penalty)  # higher = better
            details.append({"penalty": penalty, **counts})
        system = sum(seg_scores) / len(seg_scores) if seg_scores else 0.0
        return QeResult(
            segment_scores=seg_scores, system_score=system, metric="gemba-mqm", details=details
        )


# ---- Pure eval-harness helpers -------------------------------------------------
def rank_engines(results_by_engine: dict[str, QeResult]) -> list[tuple[str, float]]:
    """Rank candidate engines by system-level score (higher = better)."""
    return sorted(
        ((e, r.system_score) for e, r in results_by_engine.items()),
        key=lambda x: x[1],
        reverse=True,
    )


def below_threshold_segments(result: QeResult, threshold: float) -> list[int]:
    """Indices of segments scoring below `threshold` — routed back to refine [3]."""
    return [i for i, s in enumerate(result.segment_scores) if s < threshold]
