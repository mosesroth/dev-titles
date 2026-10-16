"""Local WhisperX transcriber (no-AWS ASR path, requirement R1).

WhisperX gives forced-alignment word timestamps — the tightest timing for cue
building / CJK re-segmentation — and runs entirely locally (nothing leaves the
machine). Heavy deps (torch, whisperx, ffmpeg) are lazy-imported so the harness
runs without them.

`from_whisperx_result()` is a PURE parser (aligned WhisperX result -> Transcript)
and is unit-tested; the model load/transcribe/align path is exercised only when a
real audio file is passed.

Install: `pip install 'dev-titles[whisper]'` plus ffmpeg on PATH.
"""

from __future__ import annotations

from ..model import Transcript, Word
from .base import Transcriber


def from_whisperx_result(result: dict, language: str = "en") -> Transcript:
    """Flatten a WhisperX aligned result (segments -> words with start/end) into
    a Transcript. Words missing timing (unaligned) are skipped."""
    words: list[Word] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = (w.get("word") or "").strip()
            if not text or w.get("start") is None or w.get("end") is None:
                continue
            words.append(
                Word(
                    text=text,
                    start=float(w["start"]),
                    end=float(w["end"]),
                    confidence=w.get("score"),
                )
            )
    return Transcript(language=language, words=words)


class WhisperTranscriber(Transcriber):
    def __init__(
        self,
        *,
        model_size: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
        batch_size: int = 8,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size

    def transcribe(self, media_path, *, language="en", custom_vocabulary=None) -> Transcript:
        try:
            import whisperx
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Install Whisper deps: pip install 'dev-titles[whisper]' (needs ffmpeg)"
            ) from e

        # Jargon biasing: WhisperX accepts an initial prompt via asr_options.
        asr_options = {}
        if custom_vocabulary:
            terms = (
                custom_vocabulary
                if isinstance(custom_vocabulary, str)
                else ", ".join(custom_vocabulary)
            )
            asr_options["initial_prompt"] = f"Terms: {terms}."

        model = whisperx.load_model(
            self.model_size,
            self.device,
            compute_type=self.compute_type,
            language=language,
            asr_options=asr_options or None,
        )
        audio = whisperx.load_audio(media_path)
        result = model.transcribe(audio, batch_size=self.batch_size, language=language)
        align_model, meta = whisperx.load_align_model(language_code=language, device=self.device)
        aligned = whisperx.align(
            result["segments"], align_model, meta, audio, self.device, return_char_alignments=False
        )
        return from_whisperx_result(aligned, language=language)
