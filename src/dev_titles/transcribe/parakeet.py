"""NVIDIA Parakeet transcriber: `parakeet-mlx` on Apple Silicon, `onnx-asr` everywhere else.

A second, DIFFERENT-ARCHITECTURE ASR (FastConformer-TDT, not Whisper). Its value
is precisely that it mis-hears DIFFERENTLY than WhisperX: where Whisper silently
drops words on fast/overlapping speech, Parakeet usually emits them, so a word-diff
of the two transcripts (transcribe_qe._second_pass_disagreements) pinpoints the hard
spots with NO caption and NO cloud egress — everything runs locally on the GPU.

Runs on Apple Silicon through MLX (Metal); ~28x real-time on an M-series Mac
(validated on a 147s clip). English-only on the 0.6b-v2 model (best WER); the v3
model adds 25 European langs at higher WER. Heavy deps are lazy-imported so the
harness (and its tests) run without `parakeet-mlx` installed.

Parakeet emits SUBWORD tokens with a leading-space word-boundary convention
(' H', 'i', ',', ' I', "'", 'm', ' Pri', 'ya' -> "Hi,", "I'm", "Priya") and,
despite the README, a per-token `confidence` (0-1). `words_from_tokens()` is a PURE
reconstruction (subword tokens -> whole Words, word confidence = min over its
subwords) and is unit-tested; the model load/transcribe path runs only on real audio.

Two backends, same model weights and the same subword-token output (so `words_from_tokens` serves
both): `parakeet-mlx` (Apple Silicon, Metal, ~28x real time) and `onnx-asr` (ONNX Runtime on CPU or
GPU; Windows, Linux, Intel Macs; ~1.7x real time on an M-series CPU; Silero VAD splits long audio).
The ONNX backend made Parakeet — the different-architecture cross-check — available on Windows, where
WhisperX was otherwise the only engine. Pick with SUBTITLE_PARAKEET_BACKEND=mlx|onnx; default: mlx
if installed and on Apple Silicon, else onnx.

Install: `pip install parakeet-mlx` (Apple Silicon) or `pip install 'onnx-asr[cpu,hub]'`; ffmpeg on PATH.
"""

from __future__ import annotations

import math
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..model import Transcript, Word
from .base import Transcriber

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"  # English-only, best Open-ASR WER
# The ONNX export writes out hesitations that the MLX build drops; left in, each one is a false
# disagreement in the cross-check against Whisper (on one interview MLX and ONNX agreed on 93% of words,
# most of the rest "uh"/"um"). Subtitles drop them anyway (clean-read).
FILLERS = frozenset({"uh", "um", "uhm", "umm", "er", "erm", "ah", "hmm", "mm", "mhm"})
ONNX_MODEL = {
    "parakeet-tdt-0.6b-v2": "nemo-parakeet-tdt-0.6b-v2",
    "parakeet-tdt-0.6b-v3": "nemo-parakeet-tdt-0.6b-v3",
}


def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


def backend() -> str | None:
    """'mlx', 'onnx' or None (no Parakeet available here)."""
    forced = os.environ.get("SUBTITLE_PARAKEET_BACKEND", "").lower()
    if forced in ("mlx", "onnx"):
        return forced
    if platform.system() == "Darwin" and platform.machine() == "arm64" and _has("parakeet_mlx"):
        return "mlx"
    return "onnx" if _has("onnx_asr") else None


def onnx_tokens(segments) -> list:
    """onnx-asr segment results -> token objects (.text/.start/.end/.confidence) for
    words_from_tokens. Token timestamps are start times relative to their segment; a token ends where
    the next starts (the last at the segment end); confidence = exp(logprob)."""
    out = []
    for seg in segments:
        toks, ts = seg.tokens or [], seg.timestamps or []
        lps = seg.logprobs or [None] * len(toks)
        for i, tok in enumerate(toks):
            start = seg.start + ts[i]
            end = seg.start + ts[i + 1] if i + 1 < len(ts) else seg.end
            lp = lps[i] if i < len(lps) else None
            out.append(
                SimpleNamespace(
                    text=tok,
                    start=start,
                    end=max(end, start),
                    confidence=None if lp is None else math.exp(lp),
                )
            )
    return out


def words_from_tokens(tokens) -> list[Word]:
    """Reconstruct whole Words from Parakeet subword tokens.

    A token that starts with a space opens a new word; tokens without a leading
    space (subword continuations, attached punctuation) glue onto the current word.
    Each Word's confidence is the MIN over its subword tokens — the least-confident
    piece drives the word, so a shaky subword still flags the word for QE.
    Accepts any objects exposing `.text`/`.start`/`.end`/`.confidence`.
    """
    words: list[Word] = []
    cur_text: str = ""
    cur_start: float | None = None
    cur_end: float | None = None
    cur_confs: list[float] = []

    def _flush():
        nonlocal cur_text, cur_confs
        if cur_text and cur_start is not None and cur_end is not None:
            words.append(
                Word(
                    text=cur_text,
                    start=cur_start,
                    end=cur_end,
                    confidence=min(cur_confs) if cur_confs else None,
                )
            )
        cur_text, cur_confs = "", []

    for t in tokens:
        boundary = t.text[:1] == " "
        stripped = t.text.strip()
        if boundary and cur_text:
            _flush()
        if not stripped:  # space-only token: just a boundary, no content
            continue
        if not cur_text:
            cur_start = t.start
        cur_text += stripped
        cur_end = t.end
        if getattr(t, "confidence", None) is not None:
            cur_confs.append(t.confidence)
    _flush()
    return words


class ParakeetTranscriber(Transcriber):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        chunk_duration: float = 120.0,
        overlap_duration: float = 15.0,
    ):
        self.model_name = model
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self._model: Any = None
        self.backend: str | None = None

    def _load(self):
        if self._model is None:
            self.backend = backend()
            if self.backend == "mlx":
                from parakeet_mlx import from_pretrained

                self._model = from_pretrained(self.model_name)
            elif self.backend == "onnx":
                import onnx_asr

                name = ONNX_MODEL.get(self.model_name.split("/")[-1], self.model_name)
                self._model = (
                    onnx_asr.load_model(name)
                    .with_vad(onnx_asr.load_vad("silero"))
                    .with_timestamps()
                )
            else:  # pragma: no cover
                raise RuntimeError(
                    "Install Parakeet: pip install parakeet-mlx (Apple Silicon) or "
                    "pip install 'onnx-asr[cpu,hub]' (Windows/Linux); ffmpeg on PATH"
                )
        return self._model

    def transcribe(self, media_path, *, language="en", custom_vocabulary=None) -> Transcript:
        # Parakeet has no initial-prompt biasing; custom_vocabulary is accepted for
        # interface parity but unused (jargon is handled downstream by the
        # glossary-aware correction + the disagreement cross-check).
        model = self._load()
        if self.backend == "onnx":
            with tempfile.TemporaryDirectory() as d:  # onnx-asr reads 16 kHz mono WAV
                wav = Path(d) / "audio.wav"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(media_path),
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        str(wav),
                    ],
                    check=True,
                )
                tokens = onnx_tokens(list(model.recognize(str(wav))))
            words = [
                w
                for w in words_from_tokens(tokens)
                if re.sub(r"[^a-z]", "", w.text.lower()) not in FILLERS
            ]
            return Transcript(language=language, words=words)
        result = model.transcribe(
            media_path, chunk_duration=self.chunk_duration, overlap_duration=self.overlap_duration
        )
        return Transcript(language=language, words=words_from_tokens(result.tokens))
