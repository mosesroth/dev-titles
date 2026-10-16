"""Resolve a user-supplied input into local media / subtitle files.

One place that turns "what the operator typed" — a local path or a URL — into
concrete files the pipeline can consume. Local files pass through; a YouTube URL
is fetched with yt-dlp (audio only, enough for ASR); frame.io needs account
onboarding + a token, so it raises a clear message rather than pretending.

Kept dependency-light: yt-dlp is imported lazily so the rest of the harness runs
without it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_MEDIA_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4a", ".mp3", ".wav", ".aac", ".flac"}
_SUB_EXT = {".srt", ".vtt", ".sbv", ".ttml", ".itt"}
_YT = re.compile(r"(youtube\.com/|youtu\.be/)", re.I)
_FRAMEIO = re.compile(r"(frame\.io|f\.io)/", re.I)


@dataclass
class ResolvedInput:
    kind: str  # "media" | "subtitle"
    path: str  # local path to the media or subtitle file
    title: str = ""  # human-friendly name (for output naming / vocab seed)
    source: str = ""  # original spec (path or URL)
    metadata: str = ""  # written metadata (e.g. YouTube description) — name/vocab seed
    auto_captions: str = ""  # local path to fetched YouTube auto-captions (.vtt), if any


def resolve_input(spec: str, workdir: str | Path = "output/_ingest") -> ResolvedInput:
    """Resolve `spec` to a local file. Local subtitle/media pass through; a YouTube
    URL is downloaded (audio) into `workdir`; frame.io is not yet wired."""
    if _FRAMEIO.search(spec):
        raise NotImplementedError(
            "frame.io ingestion isn't wired yet (needs account onboarding + an API "
            "token). Download the asset from frame.io manually and pass the local "
            "file, or use its review/comment features for feedback. See SKILL.md."
        )
    if _YT.search(spec) or spec.startswith(("http://", "https://")):
        return _resolve_youtube(spec, Path(workdir))
    p = Path(spec)
    if not p.exists():
        raise FileNotFoundError(f"input not found: {spec}")
    ext = p.suffix.lower()
    if ext in _SUB_EXT:
        return ResolvedInput("subtitle", str(p), title=p.stem, source=spec)
    if ext in _MEDIA_EXT:
        return ResolvedInput("media", str(p), title=p.stem, source=spec)
    raise ValueError(f"unsupported input type {ext!r}: {spec}")


def _resolve_youtube(url: str, workdir: Path) -> ResolvedInput:
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError(
            "yt-dlp is required to ingest a YouTube URL " "(`pip install yt-dlp`)."
        ) from e
    workdir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Also fetch Google's auto-captions — a free third ASR opinion used to
        # auto-correct proper-noun/product mishears (transcribe_qe.autosub_consult).
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-orig"],
        "subtitlesformat": "vtt",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
    meta = "\n".join(
        x
        for x in (info.get("title", ""), info.get("uploader", ""), info.get("description", ""))
        if x
    )
    # yt-dlp writes auto-captions as <id>.<lang>.vtt next to the media; grab the first.
    vid = info.get("id", "")
    auto_cap = ""
    for cand in (workdir / f"{vid}.en.vtt", workdir / f"{vid}.en-orig.vtt"):
        if cand.exists():
            auto_cap = str(cand)
            break
    if not auto_cap:
        hits = sorted(workdir.glob(f"{vid}*.vtt"))
        auto_cap = str(hits[0]) if hits else ""
    return ResolvedInput(
        "media",
        path,
        title=info.get("title", info.get("id", "")),
        source=url,
        metadata=meta,
        auto_captions=auto_cap,
    )
