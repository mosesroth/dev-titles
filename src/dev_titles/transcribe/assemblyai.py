"""AssemblyAI adapter (transcription A/B candidate).

STUB — implement if the WER spot-check favors AssemblyAI over AWS Transcribe
on our jargon. Word timings come from the `words` array of the transcript object.
"""

from __future__ import annotations

from ..model import Transcript
from .base import Transcriber


class AssemblyAITranscriber(Transcriber):
    def __init__(self, *, api_key: str | None = None):
        self.api_key = api_key

    def transcribe(
        self, media_path, *, language="en", custom_vocabulary=None
    ) -> Transcript:  # noqa: D401
        raise NotImplementedError(
            "AssemblyAITranscriber is a stub — enable only if WER spot-check wins."
        )
