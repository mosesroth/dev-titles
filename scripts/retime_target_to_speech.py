#!/usr/bin/env python3
"""Re-time translated (zh / ja) subtitle files to the speech by content
(validate/content_sync.retime_target_to_speech): runs of cues shown > 2 s away from when their
content is spoken are laid into that window. Text never changes; a file is written only if it
validates and has fewer errors. The .vtt twin and the library copy are kept identical.

  PYTHONPATH=src python3 scripts/retime_target_to_speech.py output/uploads/MyVideo.ja.srt
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dev_titles.cli import _read_any  # noqa: E402
from dev_titles.formats import write_srt, write_vtt  # noqa: E402
from dev_titles.translate import CliTranslator  # noqa: E402
from dev_titles.validate import validate_subtitle  # noqa: E402
from dev_titles.validate.content_sync import (  # noqa: E402
    content_speech_desync,
    retime_target_to_speech,
)
from dev_titles.validate.speech_sync import speech_words_for, video_id_for  # noqa: E402

OUTPUT = Path(os.environ.get("DEV_TITLES_OUTPUT", "output")).resolve()  # deliverables root
NAMES = {"zh-Hans": "Chinese", "ja": "Japanese"}


def main(argv=None) -> int:
    files = [Path(f) for f in (argv or sys.argv[1:])]
    backup = OUTPUT / f"_superseded_target_retime_{time.strftime('%Y%m%d')}"
    inv = CliTranslator()._invoke
    for f in files:
        video, lang = f.name.split(".")[:2]
        vid = video_id_for(video, OUTPUT / "ticket")
        speech = speech_words_for(vid, OUTPUT / "_speech") if vid else []
        enp = next(
            (
                p
                for p in (
                    f.with_name(f"{video}.en.srt"),
                    OUTPUT / "FINAL_SUBTITLES" / f"{video}.en.srt",
                )
                if p.exists()
            ),
            None,
        )
        if not speech or enp is None:
            print(f"{f.name}: no speech timings / EN — skipped")
            continue
        en, tgt = _read_any(str(enp), "en"), _read_any(str(f), lang)
        cp = OUTPUT / "_speech" / f"{vid}.{lang}.contentmap.json"
        before = content_speech_desync(tgt, en, speech, inv, lang_name=NAMES[lang], cache_path=cp)
        new, moved = retime_target_to_speech(
            tgt, en, speech, inv, lang_name=NAMES[lang], cache_path=cp
        )
        after = content_speech_desync(new, en, speech, inv, lang_name=NAMES[lang], cache_path=cp)
        nb = sum(o["severity"] == "error" for o in before["off"])
        na = sum(o["severity"] == "error" for o in after["off"])
        clean = not [v for v in validate_subtitle(new) if v.severity == "error"]
        print(
            f"{f.name}: moved {moved} cue(s); errors {nb} -> {na}"
            + ("" if clean and moved else " — not written")
        )
        if not (clean and moved):
            continue
        backup.mkdir(parents=True, exist_ok=True)
        lib = OUTPUT / "FINAL_SUBTITLES"
        for t in (f, f.with_suffix(".vtt"), lib / f.name, lib / f.with_suffix(".vtt").name):
            if t.exists() or t.parent == f.parent:
                if t.exists():
                    shutil.copy2(t, backup / f"{t.parent.name}_{t.name}")
                t.write_text(
                    write_srt(new) if t.suffix == ".srt" else write_vtt(new), encoding="utf-8"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
