#!/usr/bin/env python3
"""Pre-handover check for an upload folder: validate EVERY .srt/.vtt in it (sound tags, line
width, line count, durations, overlaps…) and fail on any error. Run it on each upload
folder before it goes to YouTube — files assembled outside the pipeline (fixes carried onto a
live track, hand edits) otherwise skip validation entirely.

  PYTHONPATH=src python3 scripts/check_upload.py uploads/2026-10-23
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dev_titles.cli import _read_any  # noqa: E402
from dev_titles.validate import validate_subtitle  # noqa: E402


def _lang(name: str) -> str:
    for code in ("zh-Hans", "ja", "en"):
        if f".{code}." in name or name.startswith(f"{code}."):
            return code
    return "en"


def main(folder: str) -> int:
    files = sorted(p for p in Path(folder).iterdir() if p.suffix in (".srt", ".vtt"))
    bad = 0
    for p in files:
        errs = [
            v for v in validate_subtitle(_read_any(str(p), _lang(p.name))) if v.severity == "error"
        ]
        if errs:
            bad += 1
            by_rule: dict[str, int] = {}
            for v in errs:
                by_rule[v.rule] = by_rule.get(v.rule, 0) + 1
            print(f"FAIL {p.name}: {by_rule}")
            for v in [e for e in errs if e.rule == "sound_tag"][:3]:
                print(f"     cue {v.cue_index}: {v.detail}")
    print(f"{len(files)} file(s), {bad} with errors")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
