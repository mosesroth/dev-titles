"""Discover the human parallel-subtitle corpus and build exemplar banks from it.

This is the WIRING that was missing: `exemplars.py` builds/queries an `ExemplarBank`
(few-shot approved EN->target pairs — the highest-ROI parity lever), but no
production entry point ever supplied one, so every real run used an empty bank. This
module mines the human ja/zh subtitles the docs team already produced (under
`input/channel/<id>/<id>.<lang>.srt`) and builds a per-language bank, EXCLUDING the
video being processed so a video never teaches from its own reference (train/held-out).
"""

from __future__ import annotations

import re
from pathlib import Path

from .exemplars import ExemplarBank, align_by_time
from .formats import read_srt

# Target language -> human-caption filename suffixes that count as that language.
# Simplified-Chinese variants (zh-Hans / zh-CN / zh) are unified onto zh-Hans.
_LANG_SUFFIXES: dict[str, list[str]] = {
    "ja": [".ja"],
    "zh-Hans": [".zh-Hans", ".zh-CN", ".zh"],
}


def discover_channel_corpus(root: str | Path = "input/channel") -> list[tuple[str, Path, dict]]:
    """Find (video_id, en_srt_path, {lang: target_srt_path}) for every channel video
    that has a human EN caption plus at least one human target caption."""
    root = Path(root)
    out: list[tuple[str, Path, dict]] = []
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        vid = d.name
        en = d / f"{vid}.en.srt"
        if not en.exists():
            continue
        targets: dict[str, Path] = {}
        for lang, suffixes in _LANG_SUFFIXES.items():
            for suf in suffixes:
                p = d / f"{vid}{suf}.srt"
                if p.exists():
                    targets[lang] = p
                    break
        if targets:
            out.append((vid, en, targets))
    return out


def video_id_from_input(input_str: str) -> str:
    """Best-effort corpus video-id for the held-out exclusion: a YouTube id from a
    URL (?v=ID or youtu.be/ID), else the input file's stem. Returns '' if unknown."""
    m = re.search(r"(?:v=|youtu\.be/|/watch/|embed/)([A-Za-z0-9_-]{11})", input_str)
    if m:
        return m.group(1)
    stem = Path(input_str).stem
    # channel human files are '<id>.en' etc. — strip a lang suffix if present
    return re.sub(r"\.(en|ja|zh(-Hans|-CN)?)$", "", stem)


def build_exemplar_banks(
    langs, root: str | Path = "input/channel", *, exclude_video: str | None = None
) -> dict:
    """Build an ExemplarBank per target language from the human corpus, excluding
    `exclude_video` (the one being processed) so its reference can't leak. Returns
    only the banks that actually gathered exemplars."""
    banks = {lang: ExemplarBank(lang=lang) for lang in langs}
    for vid, en_path, targets in discover_channel_corpus(root):
        if exclude_video and vid == exclude_video:
            continue
        try:
            en = read_srt(str(en_path), "en")
        except Exception:
            continue
        for lang in langs:
            if lang in targets:
                try:
                    human = read_srt(str(targets[lang]), lang)
                except Exception:
                    continue
                banks[lang].add(align_by_time(en, human, lang, vid))
    return {lang: b for lang, b in banks.items() if b.exemplars}
