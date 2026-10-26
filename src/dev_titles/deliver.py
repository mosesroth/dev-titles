"""The ONE write path for a deliverable subtitle file.

Every repair / re-time / retrofit writes through `write_deliverable`, which refuses unless:
  - the cue text is unchanged from the file it replaces (unless `text_change` is given, which
    records WHY it may change — an approved EN edit);
  - the result has NO validation error at all (not just the rule being fixed);
  - the .srt and .vtt twins and the library copy are written together, so they can't diverge.
It keeps a timestamped backup and returns what it wrote.

Why: a one-off fix script once validated only the three rules it was fixing, wrote the file
anyway, and shipped an empty cue to YouTube. A repair must not be able to introduce a defect of a
kind the author wasn't thinking about.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from .formats import write_srt, write_vtt
from .model import Subtitle
from .validate import validate_subtitle


def cue_text(sub: Subtitle) -> str:
    """All cue text, whitespace removed — the identity a repair must preserve."""
    return re.sub(r"\s", "", "".join(c.text for c in sub.cues))


class DeliverRefused(Exception):
    pass


def write_deliverable(
    sub: Subtitle,
    path: str | Path,
    *,
    before: Subtitle | None = None,
    text_change: str = "",
    also: list[str | Path] | None = None,
    backup_dir: str | Path | None = None,
    ignore_rules: tuple[str, ...] = (),
) -> list[str]:
    """Write `sub` to `path` (and its .srt/.vtt twin, plus every path in `also` and their twins).

    `before`: the subtitle being replaced — its cue text must match `sub`'s unless `text_change`
    explains the edit. `ignore_rules`: pre-existing error rules this repair isn't responsible for
    (named explicitly, so ignoring one is a decision, not an oversight).
    Raises DeliverRefused on a text change without a reason, or on any validation error.
    """
    p = Path(path)
    if before is not None and not text_change and cue_text(before) != cue_text(sub):
        raise DeliverRefused(f"{p.name}: cue text changed and no `text_change` reason was given")
    errs = [
        v for v in validate_subtitle(sub) if v.severity == "error" and v.rule not in ignore_rules
    ]
    if errs:
        rules = sorted({v.rule for v in errs})
        raise DeliverRefused(
            f"{p.name}: {len(errs)} validation error(s) {rules} — e.g. cue {errs[0].cue_index}: "
            f"{errs[0].detail[:80]}"
        )
    targets: list[Path] = []
    for q in [p, *(Path(x) for x in (also or []))]:
        targets += [q.with_suffix(".srt"), q.with_suffix(".vtt")]
    bk = (
        Path(backup_dir)
        if backup_dir
        else p.parent.parent / f"_superseded_{time.strftime('%Y%m%d')}"
    )
    written = []
    for t in targets:
        if t.exists():
            bk.mkdir(parents=True, exist_ok=True)
            shutil.copy2(t, bk / f"{t.parent.name}_{t.name}")
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(write_srt(sub) if t.suffix == ".srt" else write_vtt(sub), encoding="utf-8")
        written.append(str(t))
    return written
