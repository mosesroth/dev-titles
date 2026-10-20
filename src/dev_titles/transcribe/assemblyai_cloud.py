"""AssemblyAI cloud transcriber (Universal model) — a diarization-aware ASR for the
PUBLIC interview backlog, where cloud egress is fine. Returns word-level timestamps so
it drops into the same cue-building / clean-read-correction pipeline as WhisperX/Parakeet.

Bake-off (evi Dev Chat) found cloud ASR doesn't recover more CONTENT than Parakeet
(the word gap vs a human caption is verbatim filler), but transcribes domain JARGON
correctly at the source (React Native / agentic / boilerplate) with none of Parakeet's
conversational garbles — a cleaner input to translation. Use for interviews only if a
pilot shows Parakeet's residual garbles matter; local Parakeet remains the default.

Key from $ASSEMBLYAI_API_KEY (never hard-code / commit it). `pip install 'dev-titles[assemblyai]'`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time

from ..model import Transcript, Word
from .base import Transcriber

_BASE = "https://api.assemblyai.com/v2"


def _compress_for_asr(media_path: str) -> tuple[str, bool]:
    """Transcode to 16 kHz mono MP3 (all ASR needs) so the upload body is small.
    Large uploads to api.assemblyai.com get reset intermittently by some networks/
    middleboxes (SSLEOFError / broken pipe); a ~20-min interview shrinks from ~17MB
    to ~3-4MB, which lands far more reliably. Returns (path, is_temp); falls back to
    the original file (no temp) if ffmpeg is missing or the transcode fails."""
    if not shutil.which("ffmpeg"):
        return media_path, False
    fd, out = tempfile.mkstemp(suffix=".mp3", prefix="aai_")
    os.close(fd)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                media_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                "24k",
                out,
            ],
            check=True,
            capture_output=True,
        )
        if os.path.getsize(out) > 0:
            return out, True
    except Exception:
        pass  # no ffmpeg / transcode failed → fall back to uploading the original media
    try:
        os.unlink(out)
    except OSError:
        pass  # temp file may already be gone; cleanup is best-effort
    return media_path, False


def _upload(requests, url, headers, payload, timeout, attempts=8):
    """POST the audio body with retry+backoff. The upload endpoint is reset
    intermittently on this class of network; retrying lands it within a few tries."""
    last = None
    for i in range(attempts):
        try:
            r = requests.post(url, headers=headers, data=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # SSLError / ConnectionError / broken pipe
            last = e
            time.sleep(min(2**i, 15))
    raise RuntimeError(f"AssemblyAI upload failed after {attempts} attempts: {last}")


class AssemblyAITranscriber(Transcriber):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        speaker_labels: bool = True,
        poll_interval: float = 5.0,
        timeout: float = 1800.0,
    ):
        self.api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        self.speaker_labels = speaker_labels
        self.poll_interval = poll_interval
        self.timeout = timeout

    def transcribe(self, media_path, *, language="en", custom_vocabulary=None) -> Transcript:
        try:
            import requests
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install AssemblyAI deps: pip install 'dev-titles[assemblyai]'") from e

        if not self.api_key:
            raise RuntimeError("Set ASSEMBLYAI_API_KEY (or pass api_key=)")
        h = {"authorization": self.api_key}
        # Compress to 16 kHz mono, then upload the (small) body with retry+backoff.
        # Read the whole file so requests sends a fixed Content-Length body (not chunked).
        up_path, is_temp = _compress_for_asr(str(media_path))
        try:
            with open(up_path, "rb") as f:
                payload = f.read()
            up = _upload(requests, f"{_BASE}/upload", h, payload, self.timeout)
        finally:
            if is_temp:
                try:
                    os.unlink(up_path)
                except OSError:
                    pass  # temp upload copy; cleanup is best-effort
        body = {
            "audio_url": up["upload_url"],
            "speaker_labels": self.speaker_labels,
            "language_code": language,
        }
        if custom_vocabulary:  # jargon biasing (word_boost)
            body["word_boost"] = list(custom_vocabulary)[:1000]
        sub = requests.post(f"{_BASE}/transcript", headers=h, json=body, timeout=120).json()
        tid = sub["id"]
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            p = requests.get(f"{_BASE}/transcript/{tid}", headers=h, timeout=120).json()
            if p["status"] == "completed":
                return _to_transcript(p, language)
            if p["status"] == "error":
                raise RuntimeError(f"AssemblyAI error: {p.get('error')}")
            time.sleep(self.poll_interval)
        raise TimeoutError("AssemblyAI transcription timed out")


def _to_transcript(payload: dict, language: str = "en") -> Transcript:
    """AssemblyAI result -> Transcript. Word times are ms; confidence is 0-1."""
    words = []
    for w in payload.get("words", []):
        text = (w.get("text") or "").strip()
        if not text or w.get("start") is None or w.get("end") is None:
            continue
        words.append(
            Word(
                text=text,
                start=float(w["start"]) / 1000.0,
                end=float(w["end"]) / 1000.0,
                confidence=w.get("confidence"),
                speaker=w.get("speaker"),
            )
        )
    return Transcript(language=language, words=words)
