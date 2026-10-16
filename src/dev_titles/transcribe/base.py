"""ASR adapter interface (pipeline step [1]).

Requirement R1: EN audio -> word-level timing + jargon-aware custom vocabulary.
Concrete adapters: aws_transcribe (AWS Transcribe), assemblyai. gpt-4o-
transcribe is explicitly excluded (no word timestamps).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..model import Transcript


class Transcriber(ABC):
    @abstractmethod
    def transcribe(
        self,
        media_path: str,
        *,
        language: str = "en",
        custom_vocabulary: list[str] | None = None,
    ) -> Transcript:
        """Return a Transcript with word-level timing."""
        raise NotImplementedError
