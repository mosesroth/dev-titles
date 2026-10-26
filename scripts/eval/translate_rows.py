#!/usr/bin/env python3
"""Worker for spot_ab.py: translate selected sentences of A/B run folders with WHICHEVER code is on
PYTHONPATH and the given glossary, using the same per-video context the pipeline uses (brief,
speakers, meaning notes, line context, fixed lines). Runs once per code version.

  PYTHONPATH=<src> python3 scripts/eval/translate_rows.py <jobs.json> <glossary.yaml> <out.json>

jobs.json: [{"dir": "<A/B run folder>", "lang": "ja", "i": 27}, ...]
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dev_titles.cli import _read_any
from dev_titles.glossary import load
from dev_titles.localize import (
    _ctx_for,
    _line_context,
    _normalizer,
    _speakers_for,
    effective_glossary,
)
from dev_titles.resegment import merge_en_to_sentences
from dev_titles.translate import CliTranslator, translate_segment

try:
    from dev_titles.localize import _fixed_line
except ImportError:  # older code without fixed lines
    _fixed_line = None
try:
    from dev_titles.localize import _translate_fresh
except ImportError:  # older code: plain translate_segment
    _translate_fresh = None
from dev_titles.config import spec_for

LANG = {"zh-Hans": "Simplified Chinese", "ja": "Japanese"}

jobs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
g0 = load(sys.argv[2])
ctxs: dict = {}


def video(d: str):
    if d not in ctxs:
        p = Path(d)
        en = next(iter(sorted(p.glob("*.en.vtt")) + [p / "en.vtt"]))
        sents = merge_en_to_sentences(_read_any(str(en), "en"))
        vv: dict = {}
        if (p / "video_brief.txt").exists():
            vv["brief"] = (p / "video_brief.txt").read_text(encoding="utf-8").strip()
        if (p / "speakers.json").exists():
            vv["speakers"] = json.loads((p / "speakers.json").read_text())["labels"]
        if (p / "meaning_notes.json").exists():
            vv["meaning_notes"] = json.loads((p / "meaning_notes.json").read_text())
        ctxs[d] = (sents, vv, _speakers_for(sents, vv), _ctx_for(sents, vv))
    return ctxs[d]


def run(job):
    sents, vv, spk, ctx = video(job["dir"])
    lang, i = job["lang"], job["i"]
    g, dnt = effective_glossary(g0, vv, True, lang=lang)
    src = sents[i].text
    if _fixed_line is not None and (fx := _fixed_line(g, lang, src)) is not None:
        return fx
    try:
        lc = _line_context(i, sents, spk, vv, lang=lang)
    except TypeError:  # older signature without lang
        lc = _line_context(i, sents, spk, vv)
    if _translate_fresh is not None:  # the pipeline's own per-sentence step: budget from the
        # sentence's timing, omission retry, coverage edit-fix — so a drop isn't judged as a change
        txt, _, _ = _translate_fresh(
            i,
            sents[i],
            CliTranslator(),
            g,
            lang,
            LANG.get(lang, lang),
            spec_for(lang),
            ctx=ctx,
            line_context=lc,
            exemplar_bank=None,
            extra_dnt=dnt,
            invoke=CliTranslator()._invoke,
            backcheck=True,
            edit_fix_rounds=2,
            logger=lambda m: None,
        )
    else:
        txt, _ = translate_segment(
            CliTranslator(),
            g,
            src,
            lang,
            full_context=ctx,
            line_context=lc,
            reflect=True,
            char_budget=60,
            extra_dnt=dnt,
        )
    return _normalizer(lang, True)(txt, src)


with ThreadPoolExecutor(6) as ex:
    out = list(ex.map(run, jobs))
Path(sys.argv[3]).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
