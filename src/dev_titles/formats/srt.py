"""SubRip (SRT) reader/writer for distribution."""

from __future__ import annotations

import re
from pathlib import Path

from ..model import Cue, Subtitle
from .timecode import seconds_to_ts, ts_to_seconds

# Hours may be 1 or 2 digits: some tools (YouTube/CaptionHub exports) emit
# "0:00:09,224". A strict \d{2} silently parsed such files as ZERO cues.
_SRT_TIMING = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2},\d{3})\s*-->\s*(?P<end>\d{1,2}:\d{2}:\d{2},\d{3})"
)


def write_srt(sub: Subtitle) -> str:
    sub.renumber()
    out: list[str] = []
    for cue in sub.cues:
        out.append(str(cue.index))
        out.append(f"{seconds_to_ts(cue.start, ',')} --> {seconds_to_ts(cue.end, ',')}")
        # A blank line is the SRT cue separator, so an empty element inside cue.lines
        # would split the cue in two — orphaning the trailing text (no timestamp) and
        # breaking 1:1 ingestion by translation-management tools. Drop empty/whitespace-only line elements.
        text = [ln for ln in cue.lines if ln.strip()]
        out.extend(text or [""])
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def read_srt(path: str | Path, language: str = "en") -> Subtitle:
    raw = Path(path).read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", raw.strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = block.splitlines()
        timing_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if timing_idx is None:
            continue
        m = _SRT_TIMING.search(lines[timing_idx])
        if not m:
            continue
        from .vtt import _decode

        text_lines = [_decode(ln) for ln in lines[timing_idx + 1 :] if _decode(ln).strip()]
        cues.append(
            Cue(
                start=ts_to_seconds(m.group("start")),
                end=ts_to_seconds(m.group("end")),
                lines=text_lines,
            )
        )
    return Subtitle(language=language, cues=cues).renumber()
