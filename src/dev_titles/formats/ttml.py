"""TTML / iTT master writer.

STUB: minimal TTML emission is implemented for archival/master use; iTT (iTunes
Timed Text) profile attributes and styling are TODO once the distribution target
is confirmed. Netflix/Apple delivery specifics belong here.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from ..model import Subtitle
from .timecode import seconds_to_ts


def write_ttml(sub: Subtitle) -> str:
    sub.renumber()
    body = []
    for cue in sub.cues:
        text = "<br/>".join(escape(ln) for ln in cue.lines)
        body.append(
            f'      <p begin="{seconds_to_ts(cue.start)}" '
            f'end="{seconds_to_ts(cue.end)}" xml:id="c{cue.index}">{text}</p>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<tt xmlns="http://www.w3.org/ns/ttml" '
        'xmlns:tts="http://www.w3.org/ns/ttml#styling" '
        f'xml:lang="{sub.language}">\n'
        "  <body>\n    <div>\n" + "\n".join(body) + "\n    </div>\n  </body>\n</tt>\n"
    )
