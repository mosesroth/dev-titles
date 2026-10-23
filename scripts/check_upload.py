#!/usr/bin/env python3
"""Pre-handover check for an upload folder. FAILS (exit 1) on any of:

  1. a cue-level validation error in any .srt/.vtt (sound tags, '>>' markers, duplicated text, line
     width / count, durations, overlaps…);
  2. an EN track out of sync with the SPEECH (validate/speech_sync.py: a cue starting > 2 s after
     its words are spoken), measured against the video's word-timed YouTube auto-captions;
  3. a translated track drifted from the EN track (validate/sync.py), or shown away from when its
     content is SPOKEN (validate/content_sync.py: each cue mapped to the EN sentences it translates,
     those timed by the speech; a run of >= 2 cues > 2 s off is an error);
  4. a track that differs from the library copy in output/FINAL_SUBTITLES (the library must be
     what ships).

Warnings (listed, don't fail): EN cues 1-2 s late or > 3 s early; translated cues whose Latin
names sit > 5 s away from the EN cue that says them.

Files assembled outside the pipeline (a fix carried onto a live track, a hand edit) otherwise skip
every check (that is how sound tags, an EN sync slip and duplicated EN text once reached a live track).

  PYTHONPATH=src python3 scripts/check_upload.py output/uploads/2026-10-23
  PYTHONPATH=src python3 scripts/check_upload.py output/uploads/2026-10-23 --no-speech   # offline
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dev_titles.cli import _read_any  # noqa: E402
from dev_titles.validate import validate_subtitle  # noqa: E402
from dev_titles.validate.content_sync import content_speech_desync  # noqa: E402
from dev_titles.validate.speech_sync import (  # noqa: E402
    speech_sync_report,
    speech_words_for,
    video_id_for,
)
from dev_titles.validate.sync import local_desync, sync_report  # noqa: E402

OUTPUT = Path(os.environ.get("DEV_TITLES_OUTPUT", "output")).resolve()  # deliverables root


def _lang(name: str) -> str:
    for code in ("zh-Hans", "ja", "en"):
        if f".{code}." in name or name.startswith(f"{code}."):
            return code
    return "en"


def _ids_from_checklist(folder: Path) -> dict[str, str]:
    p = folder / "_UPLOAD_CHECKLIST.txt"
    if not p.exists():
        return {}
    return dict(re.findall(r"^(\S+)\s+([A-Za-z0-9_-]{11})\s", p.read_text(encoding="utf-8"), re.M))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--no-speech", action="store_true", help="skip the EN-vs-speech check")
    ap.add_argument("--library", default=str(OUTPUT / "FINAL_SUBTITLES"))
    a = ap.parse_args(argv)
    folder, library = Path(a.folder), Path(a.library)
    ids = _ids_from_checklist(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix in (".srt", ".vtt"))
    failures: list[str] = []
    warnings: list[str] = []

    for p in files:
        errs = [
            v for v in validate_subtitle(_read_any(str(p), _lang(p.name))) if v.severity == "error"
        ]
        if errs:
            rules: dict[str, int] = {}
            for v in errs:
                rules[v.rule] = rules.get(v.rule, 0) + 1
            failures.append(
                f"{p.name}: {rules} — e.g. cue {errs[0].cue_index}: {errs[0].detail[:90]}"
            )
        lib = library / p.name
        if p.suffix == ".srt" and lib.exists() and lib.read_bytes() != p.read_bytes():
            failures.append(
                f"{p.name}: differs from the library copy {lib} — update one or the other"
            )
        elif p.suffix == ".srt" and not lib.exists():
            warnings.append(f"{p.name}: no library copy in {library}")

    for p in [f for f in files if f.suffix == ".srt" and _lang(f.name) == "en"]:
        video = p.name.split(".")[0]
        en = _read_any(str(p), "en")
        speech: list = []
        if not a.no_speech:
            vid = ids.get(video) or video_id_for(video, OUTPUT / "ticket")
            speech = speech_words_for(vid, OUTPUT / "_speech") if vid else []
            if not speech:
                warnings.append(f"{p.name}: no speech timings available — EN sync not checked")
            else:
                r = speech_sync_report(en, speech)
                for o in r["off"]:
                    msg = f"{p.name} {o['start']:.1f}s: cue {o['offset']:+.1f}s from speech — {o['text']!r}"
                    (failures if o["severity"] == "error" else warnings).append(msg)
        for q in folder.glob(f"{video}.*.srt"):
            lang = _lang(q.name)
            if lang == "en":
                continue
            tgt = _read_any(str(q), lang).cues
            rep = sync_report(tgt, en.cues)
            if rep["drifted"]:
                failures.append(
                    f"{q.name}: drifted from the EN (median {rep['median']} s, {rep['far_share']:.0%} far)"
                )
            for d in local_desync(tgt, en.cues, tol=5.0):
                warnings.append(
                    f"{q.name} {d['start']:.1f}s: '{d['anchor']}' {d['gap']:.1f}s from the EN cue"
                )
            if speech:
                from dev_titles.translate import CliTranslator

                vid = ids.get(video) or video_id_for(video, OUTPUT / "ticket")
                r = content_speech_desync(
                    _read_any(str(q), lang),
                    en,
                    speech,
                    CliTranslator()._invoke,
                    lang_name="Chinese" if lang == "zh-Hans" else "Japanese",
                    cache_path=OUTPUT / "_speech" / f"{vid}.{lang}.contentmap.json",
                )
                for o in r["off"]:
                    msg = (
                        f"{q.name} {o['start']:.1f}s: shown {o['gap']:.1f}s "
                        f"{'after' if o['late'] else 'before'} its content is spoken — {o['text']!r}"
                    )
                    (failures if o["severity"] == "error" else warnings).append(msg)

    for w in warnings:
        print(f"warn  {w}")
    for f in failures:
        print(f"FAIL  {f}")
    print(f"{len(files)} file(s): {len(failures)} failure(s), {len(warnings)} warning(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
