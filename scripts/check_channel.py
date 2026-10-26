#!/usr/bin/env python3
"""Audit the LIVE YouTube tracks against our masters. Run after every upload round.

check_upload.py validates the files we are about to hand over; it cannot see what actually happened
on the channel. Three classes of defect are invisible from our side alone:

  1. a track live under the WRONG language code (a Simplified Chinese file filed as "Chinese (Hong
     Kong)", so zh-Hans viewers get nothing);
  2. a track that exists ONLY live: no master, so none of our checks ever saw it;
  3. a master that is BETTER than what is live, i.e. an upload that never happened.

For each video with a master, it fetches every live track (yt-dlp, manual subtitles only), compares
it with the master, validates it, and prints one line per finding plus a re-upload list.

  PYTHONPATH=src python3 scripts/check_channel.py [--videos A B] [--cache ../output/_live_<date>]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dev_titles.cli import _read_any  # noqa: E402
from dev_titles.deliver import cue_text  # noqa: E402
from dev_titles.validate import validate_subtitle  # noqa: E402
from dev_titles.validate.speech_sync import video_id_for  # noqa: E402

OUTPUT = Path(os.environ.get("DEV_TITLES_OUTPUT", "output")).resolve()  # deliverables root
FINAL = OUTPUT / "FINAL_SUBTITLES"
OURS = ("en", "ja", "zh-Hans")
# codes a track of ours could be filed under by mistake (zh-Hant/zh-HK/zh-TW are different languages;
# YouTube also auto-translates, but those tracks are listed as "<x> from <y>" and skipped below)
WRONG_ZH = ("zh-HK", "zh-TW", "zh-Hant", "zh-CN", "zh")


def live_langs(vid: str) -> list[str]:
    """MANUAL subtitle languages on the video, from yt-dlp's JSON (`subtitles`). Not the
    `automatic_captions` list — that holds YouTube's machine translations into ~160 languages, and
    parsing `--list-subs` text instead reported zh-Hans as present on every video."""
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "-q",
            "--no-warnings",
            "--skip-download",
            "--dump-json",
            f"https://www.youtube.com/watch?v={vid}",
        ],
        capture_output=True,
        text=True,
    ).stdout
    try:
        info = json.loads(out)
    except json.JSONDecodeError:
        return []
    return sorted((info.get("subtitles") or {}).keys())


def fetch(vid: str, lang: str, dest: Path) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "-q",
            "--no-warnings",
            "--skip-download",
            "--write-subs",
            "--sub-langs",
            lang,
            "--sub-format",
            "vtt",
            "-o",
            str(dest.with_suffix("")) + ".%(ext)s",
            f"https://www.youtube.com/watch?v={vid}",
        ],
        capture_output=True,
        text=True,
    )
    hit = sorted(dest.parent.glob(f"{dest.stem}.{lang}.vtt"))
    return hit[0] if hit else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", help="limit to these videos (default: every master)")
    ap.add_argument(
        "--cache", default=str(OUTPUT / "_live_channel"), help="where live tracks are kept"
    )
    a = ap.parse_args(argv)
    cache = Path(a.cache)
    videos = sorted({p.name.split(".")[0] for p in FINAL.glob("*.srt")})
    if a.videos:
        videos = [v for v in videos if v in set(a.videos)]
    findings, reupload = [], []
    for v in videos:
        vid = video_id_for(v, OUTPUT / "ticket")
        if not vid:
            findings.append((v, "-", "no YouTube id on file — cannot check"))
            continue
        langs = live_langs(vid)
        if not langs:
            findings.append((v, "-", f"{vid}: no live subtitle track at all"))
            continue
        for lang in OURS:
            master = FINAL / f"{v}.{lang}.srt"
            if lang in langs:
                lp = fetch(vid, lang, cache / v)
                if lp is None:
                    findings.append((v, lang, "live track listed but could not be fetched"))
                    continue
                live = _read_any(str(lp), lang)
                errs = sorted({x.rule for x in validate_subtitle(live) if x.severity == "error"})
                if not master.exists():
                    findings.append(
                        (
                            v,
                            lang,
                            f"LIVE ONLY — no master{'; live errors ' + str(errs) if errs else ''}",
                        )
                    )
                    reupload.append(
                        (v, vid, lang, "import the live track as a master, fix, re-upload")
                    )
                    continue
                m = _read_any(str(master), lang)
                same_timing = len(live.cues) == len(m.cues) and all(
                    abs(a.start - b.start) <= 0.05 and abs(a.end - b.end) <= 0.05
                    for a, b in zip(live.cues, m.cues)
                )  # YouTube re-times to its own millisecond grid: <= 50 ms is not a difference
                if cue_text(live) == cue_text(m) and same_timing:
                    if errs:
                        findings.append((v, lang, f"live == master but BOTH have errors {errs}"))
                    continue
                how = "timing differs" if cue_text(live) == cue_text(m) else "TEXT DIFFERS"
                m_errs = sorted({x.rule for x in validate_subtitle(m) if x.severity == "error"})
                findings.append(
                    (
                        v,
                        lang,
                        f"live != master ({how}); live errors {errs or 'none'}, master errors {m_errs or 'none'}",
                    )
                )
                reupload.append(
                    (v, vid, lang, "re-upload the master" if not m_errs else "fix the master first")
                )
            else:
                mis = [w for w in WRONG_ZH if w in langs] if lang == "zh-Hans" else []
                if mis and master.exists():
                    findings.append(
                        (
                            v,
                            lang,
                            f"WRONG LANGUAGE CODE: our {lang} file appears to be live as {mis} — re-upload as {lang} and delete {mis}",
                        )
                    )
                    reupload.append(
                        (v, vid, lang, f"re-upload as {lang}; delete the {','.join(mis)} track")
                    )
                elif master.exists():
                    findings.append((v, lang, "master exists, NO live track"))
                    reupload.append((v, vid, lang, "never uploaded — your call"))
    for v, lang, msg in findings:
        print(f"{v:22} {lang:8} {msg}")
    print(
        f"\n{len(videos)} video(s) checked, {len(findings)} finding(s), {len(reupload)} re-upload item(s)"
    )
    for v, vid, lang, what in reupload:
        print(f"  {v:22} {vid:12} {lang:8} {what}")
    return 1 if reupload else 0


if __name__ == "__main__":
    sys.exit(main())
