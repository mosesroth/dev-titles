"""NVIDIA Parakeet transcriber via `parakeet-mlx` (Apple-Silicon / Metal ASR path).

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

Install: `pip install parakeet-mlx` (needs ffmpeg on PATH).
"""

from __future__ import annotations

from ..model import Transcript, Word
from .base import Transcriber

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"  # English-only, best Open-ASR WER


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
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from parakeet_mlx import from_pretrained
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "Install Parakeet: pip install parakeet-mlx (Apple Silicon; needs ffmpeg)"
                ) from e
            self._model = from_pretrained(self.model_name)
        return self._model

    def transcribe(self, media_path, *, language="en", custom_vocabulary=None) -> Transcript:
        # Parakeet has no initial-prompt biasing; custom_vocabulary is accepted for
        # interface parity but unused (jargon is handled downstream by the
        # glossary-aware correction + the disagreement cross-check).
        model = self._load()
        result = model.transcribe(
            media_path, chunk_duration=self.chunk_duration, overlap_duration=self.overlap_duration
        )
        return Transcript(language=language, words=words_from_tokens(result.tokens))
