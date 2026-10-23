"""Translator adapter interface (pipeline step [3]).

Requirement R2: full-transcript context, retrieved glossary terms, register
control (JA teineigo / ZH formal), reflect-then-refine. Concrete adapter:
bedrock (Claude, primary). A/B candidates (Qwen-MT/Hunyuan/DeepSeek/DeepL) plug
in behind the same interface for the eval harness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..glossary import Term


@dataclass
class TranslationRequest:
    source_text: str  # the segment to translate
    source_lang: str
    target_lang: str
    full_context: str = ""  # whole-transcript context window
    # the line's own surroundings + notes, shown NEXT TO the line (translate AND reflect prompts):
    # neighbors (prev/next, speaker-labeled, the line marked), speaker and meaning notes
    line_context: str = ""
    glossary_terms: list[Term] = field(default_factory=list)
    register: str = ""  # e.g. "teineigo", "formal-written"
    style_notes: str = ""
    char_budget: int | None = None  # target max display-chars (condensation)
    localization_policy: str = ""  # glossary.heuristics (localize-vs-DNT rules)
    exemplars: list[tuple[str, str]] = field(default_factory=list)  # (EN, human target) style pairs
    punct_note: str = ""  # language-specific punctuation-density guidance
    completeness_hint: str = ""  # forceful re-translate instruction when a draft dropped content
    correction_hint: str = (
        ""  # reviewer/QE-flagged errors to FIX on a repair pass (adequacy/fluency)
    )


class Translator(ABC):
    @abstractmethod
    def translate(self, req: TranslationRequest) -> str:
        raise NotImplementedError

    @abstractmethod
    def reflect_and_refine(self, req: TranslationRequest, draft: str) -> str:
        """Critique `draft` on accuracy/fluency/style/terminology, then return an
        improved translation."""
        raise NotImplementedError
