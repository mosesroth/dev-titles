"""WebVTT reader/writer. WebVTT is the canonical format for the pipeline."""

from __future__ import annotations

import re
from pathlib import Path

from ..model import Cue, Subtitle
from .timecode import seconds_to_ts, ts_to_seconds

_CUE_TIMING = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{1,2}:\d{2}[.,]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{1,2}:\d{2}[.,]\d{3})"
)


def write_vtt(sub: Subtitle) -> str:
    sub.renumber()
    out = ["WEBVTT", ""]
    for cue in sub.cues:
        out.append(str(cue.index))
        out.append(f"{seconds_to_ts(cue.start)} --> {seconds_to_ts(cue.end)}")
        out.extend(cue.lines)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _decode(line: str) -> str:
    import html

    return re.sub(r" {2,}", " ", html.unescape(line).replace("\u00a0", " ")).rstrip()


def read_vtt(path: str | Path, language: str = "en") -> Subtitle:
    raw = Path(path).read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", raw.strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = block.splitlines()
        if not lines or lines[0].startswith("WEBVTT"):
            continue
        timing_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if timing_idx is None:
            continue
        m = _CUE_TIMING.search(lines[timing_idx])
        if not m:
            continue
        # Decode HTML entities (YouTube captions carry &nbsp; / &amp;), NBSP -> space.
        text_lines = [_decode(ln) for ln in lines[timing_idx + 1 :] if _decode(ln).strip()]
        cues.append(
            Cue(
                start=ts_to_seconds(m.group("start")),
                end=ts_to_seconds(m.group("end")),
                lines=text_lines,
            )
        )
    return Subtitle(language=language, cues=cues).renumber()
