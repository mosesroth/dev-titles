"""SubViewer/YouTube SBV reader.

SBV cues are a single timing line ``H:MM:SS.mmm,H:MM:SS.mmm`` (start,end on one
line, comma-separated, no cue index) followed by one or more text lines, blocks
separated by blank lines. YouTube's caption export uses this format.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..model import Cue, Subtitle
from .timecode import ts_to_seconds

_SBV_TIMING = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}[.,]\d{3})\s*,\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[.,]\d{3})\s*$"
)


def read_sbv(path: str | Path, language: str = "en") -> Subtitle:
    raw = Path(path).read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", raw.strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = block.splitlines()
        timing_idx = next((i for i, ln in enumerate(lines) if _SBV_TIMING.match(ln)), None)
        if timing_idx is None:
            continue
        m = _SBV_TIMING.match(lines[timing_idx])
        if m is None:  # timing_idx came from a matching line, so this is defensive
            continue
        text_lines = [ln for ln in lines[timing_idx + 1 :] if ln.strip()]
        if not text_lines:
            continue
        cues.append(
            Cue(
                start=ts_to_seconds(m.group("start")),
                end=ts_to_seconds(m.group("end")),
                lines=text_lines,
            )
        )
    return Subtitle(language=language, cues=cues).renumber()
