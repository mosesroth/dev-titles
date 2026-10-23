#!/usr/bin/env python3
"""Re-time EN subtitle files to the speech (validate/speech_sync.retime_to_speech): cue boundaries
snap to the word timings of the video's YouTube auto-captions; the TEXT is never changed. A file is
written only if the result passes validation and has fewer speech-sync errors than before. The
.vtt twin and the library copy (output/FINAL_SUBTITLES) are kept identical.

  PYTHONPATH=src python3 scripts/retime_en_to_speech.py output/uploads/MyVideo.en.srt …
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dev_titles.cli import _read_any  # noqa: E402
from dev_titles.formats import write_srt, write_vtt  # noqa: E402
from dev_titles.validate import validate_subtitle  # noqa: E402
from dev_titles.validate.speech_sync import (  # noqa: E402
    retime_to_speech,
    speech_sync_report,
    speech_words_for,
    video_id_for,
)

OUTPUT = Path(os.environ.get("DEV_TITLES_OUTPUT", "output")).resolve()  # deliverables root


def _errors(rep) -> int:
    return sum(o["severity"] == "error" for o in rep["off"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="<Video>.en.srt files")
    ap.add_argument("--library", default=str(OUTPUT / "FINAL_SUBTITLES"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    backup = OUTPUT / f"_superseded_en_retime_{time.strftime('%Y%m%d')}"
    for f in map(Path, a.files):
        video = f.name.split(".")[0]
        vid = video_id_for(video, OUTPUT / "ticket")
        speech = speech_words_for(vid, OUTPUT / "_speech") if vid else []
        if not speech:
            print(f"{video}: no speech timings — skipped")
            continue
        en = _read_any(str(f), "en")
        new, moved = retime_to_speech(en, speech)
        b, n = speech_sync_report(en, speech), speech_sync_report(new, speech)
        verr = [
            v
            for v in validate_subtitle(new)
            if v.severity == "error" and v.rule != "duplicate_text"
        ]
        norm = lambda s: re.sub(r"\s", "", "".join(c.text for c in s.cues))  # noqa: E731
        ok = not verr and norm(new) == norm(en) and _errors(n) < max(_errors(b), 1)
        print(
            f"{video}: moved {moved} cue(s); sync errors {_errors(b)} -> {_errors(n)}, warnings "
            f"{len(b['off']) - _errors(b)} -> {len(n['off']) - _errors(n)}"
            + (
                ""
                if ok
                else f" — NOT written ({'validation ' + str({v.rule for v in verr}) if verr else 'no improvement'})"
            )
        )
        if not ok or a.dry_run:
            continue
        backup.mkdir(parents=True, exist_ok=True)
        targets = [f, f.with_suffix(".vtt")]
        lib = Path(a.library)
        targets += [lib / f.name, lib / f.with_suffix(".vtt").name]
        for t in targets:
            if t.exists() or t.parent == f.parent:
                if t.exists():
                    shutil.copy2(t, backup / f"{t.parent.name}_{t.name}")
                t.write_text(
                    write_srt(new) if t.suffix == ".srt" else write_vtt(new), encoding="utf-8"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
