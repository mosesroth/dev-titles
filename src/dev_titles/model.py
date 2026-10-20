"""Core data model shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Word:
    """A single transcribed word with timing, in seconds."""

    text: str
    start: float
    end: float
    confidence: Optional[float] = None
    speaker: Optional[str] = None  # ASR diarization label, when the engine provides one


@dataclass
class Transcript:
    """Full-media transcript with word-level timing (from ASR)."""

    language: str
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def duration(self) -> float:
        return self.words[-1].end if self.words else 0.0


@dataclass
class Cue:
    """One subtitle event. `lines` holds already-broken display lines.

    `start`/`end` are seconds. `index` is 1-based cue number when emitted.
    """

    start: float
    end: float
    lines: list[str] = field(default_factory=list)
    index: Optional[int] = None

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Subtitle:
    """An ordered set of cues in a single language."""

    language: str
    cues: list[Cue] = field(default_factory=list)

    def renumber(self) -> "Subtitle":
        for i, cue in enumerate(self.cues, start=1):
            cue.index = i
        return self
