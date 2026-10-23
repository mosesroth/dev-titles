#!/usr/bin/env python3
"""Claude Code PostToolUse hook: validate any subtitle deliverable touched in the last few minutes.

Ad-hoc edits (a one-off script that re-times a cue, carries a fix onto a live track) bypass the
pipeline's gates; this runs the cue-level validator on every .srt/.vtt under output/UPLOAD_*/ and
output/FINAL_SUBTITLES/ modified within WINDOW seconds, plus the upload-vs-library parity check, and
reports failures back to the agent (exit 2 -> shown to Claude) so it fixes them before moving on.
Fast and offline: the EN-vs-speech check stays in scripts/check_upload.py.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(os.environ.get("DEV_TITLES_OUTPUT", "output")).resolve()  # deliverables root
WINDOW = 300  # seconds
sys.path.insert(0, str(ROOT / "src"))


def _lang(name: str) -> str:
    for code in ("zh-Hans", "ja", "en"):
        if f".{code}." in name or name.startswith(f"{code}."):
            return code
    return "en"


def main() -> int:
    now = time.time()
    files = [
        p
        for d in [OUTPUT / "FINAL_SUBTITLES", *OUTPUT.glob("UPLOAD_*")]
        if d.is_dir()
        for p in d.iterdir()
        if p.suffix in (".srt", ".vtt") and now - p.stat().st_mtime < WINDOW
    ]
    if not files:
        return 0
    from dev_titles.cli import _read_any
    from dev_titles.validate import validate_subtitle

    problems = []
    for p in sorted(files):
        try:
            errs = [
                v
                for v in validate_subtitle(_read_any(str(p), _lang(p.name)))
                if v.severity == "error"
            ]
        except Exception as e:
            problems.append(f"{p}: unreadable ({type(e).__name__})")
            continue
        for v in errs[:3]:
            problems.append(
                f"{p.parent.name}/{p.name} cue {v.cue_index} {v.rule}: {v.detail[:100]}"
            )
        if len(errs) > 3:
            problems.append(f"{p.parent.name}/{p.name}: {len(errs) - 3} more error(s)")
        if p.parent.name.startswith("UPLOAD_") and p.suffix == ".srt":
            lib = OUTPUT / "FINAL_SUBTITLES" / p.name
            if lib.exists() and lib.read_bytes() != p.read_bytes():
                problems.append(f"{p.parent.name}/{p.name} differs from FINAL_SUBTITLES/{p.name}")
    if problems:
        print(
            "Subtitle deliverables just written have errors:\n  " + "\n  ".join(problems),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
