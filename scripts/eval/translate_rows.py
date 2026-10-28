#!/usr/bin/env python3
"""Worker for spot_ab.py: translate selected sentences of A/B run folders with WHICHEVER code is on
PYTHONPATH and the given glossary, using the same per-video context the pipeline uses (brief,
speakers, meaning notes, line context, fixed lines). Runs once per code version.

  PYTHONPATH=<src> python3 scripts/eval/translate_rows.py <jobs.json> <glossary.yaml> <out.json>

jobs.json: [{"dir": "<A/B run folder>", "lang": "ja", "i": 27}, ...]
"""

import json
import os
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
            CliTranslator(model_id=os.environ.get("SPOT_MODEL") or None),
            g,
            lang,
            LANG.get(lang, lang),
            spec_for(lang),
            ctx=ctx,
            line_context=lc,
            exemplar_bank=None,
            extra_dnt=dnt,
            invoke=CliTranslator(model_id=os.environ.get("SPOT_MODEL") or None)._invoke,
            backcheck=True,
            edit_fix_rounds=2,
            logger=lambda m: None,
        )
    else:
        txt, _ = translate_segment(
            CliTranslator(model_id=os.environ.get("SPOT_MODEL") or None),
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


with ThreadPoolExecutor(int(os.environ.get("SPOT_WORKERS", "6"))) as ex:
    out = list(ex.map(run, jobs))


def editor_pass(out: list[str]) -> list[str]:
    """SPOT_EDITOR=1: also run the pipeline's native-editor and reply passes on the translated rows, so a
    change to an editor prompt can be spot-tested (the 也谢谢您的邀请 fix needed a 4-hour full run).
    Each row is edited in its local context: its neighbours (±2) carry the A/B run's own translations
    from cache.jsonl — the same for every arm — and are marked already-reviewed so only the rows
    under test are edited."""
    import inspect
    import types

    from dev_titles import editor as ed
    from dev_titles.cli import _qe_second_judge
    from dev_titles.localize import SentenceRecord

    inv = CliTranslator(model_id=os.environ.get("SPOT_MODEL") or None)._invoke
    second = _qe_second_judge(types.SimpleNamespace(qe_judge=True, single_judge=False))
    groups: dict = {}
    for k, j in enumerate(jobs):
        groups.setdefault((j["dir"], j["lang"]), []).append(k)
    for (d, lang), ks in groups.items():
        sents, vv, spk, _ = video(d)
        g, _ = effective_glossary(g0, vv, True, lang=lang)
        cached = {}
        cp = Path(d) / "cache.jsonl"
        if cp.exists():
            for line in cp.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r.get("lang") == lang:
                    cached[r["src"].strip()] = r["txt"]
        mine = {jobs[k]["i"]: k for k in ks}
        idx = sorted({i + o for i in mine for o in range(-2, 3) if 0 <= i + o < len(sents)})
        recs = []
        for i in idx:
            txt = out[mine[i]] if i in mine else cached.get(sents[i].text.strip())
            if txt:
                recs.append(SentenceRecord(lang, i, sents[i].text, txt, "", []))
        by_i = {r.i: r for r in recs}
        tmp = Path(sys.argv[3]).parent / f"_editor_state_{abs(hash((d, lang)))}"
        tmp.mkdir(parents=True, exist_ok=True)
        st = {ed._h(r.src): ed._h(r.txt) for r in recs if r.i not in mine}
        (tmp / "editor_state.json").write_text(json.dumps(st), encoding="utf-8")
        style = (getattr(g, "style", {}) or {}).get(lang, {}) or {}
        src_all = " ".join(r.src for r in recs).lower()
        terms = [
            t.translations.get(lang, "")
            for t in g.terms
            if t.enforce == "hard" and t.source.lower() in src_all
        ]
        kw = dict(
            normalize=_normalizer(lang, True),
            pair_judge=second,
            brief=vv.get("brief", ""),
            register=str(style.get("register_conversational", ""))[:300],
            glossary_terms=terms,
            speakers=vv.get("speakers") or None,
            meaning_notes=vv.get("meaning_notes") or None,
            logger=(
                (lambda m: print(m, file=sys.stderr))
                if os.environ.get("SPOT_EDITOR_LOG")
                else (lambda m: None)
            ),
        )
        params = inspect.signature(ed.native_edit).parameters
        ed.native_edit(
            recs,
            inv,
            lang,
            LANG.get(lang, lang),
            tmp / "editor_state.json",
            **{k_: v for k_, v in kw.items() if k_ in params},
        )
        if vv.get("speakers") and hasattr(ed, "reply_edit"):
            rp = inspect.signature(ed.reply_edit).parameters
            window = rp["window"].default if "window" in rp else 2

            def key(i):
                return ed._h(
                    "|".join(
                        by_i[j].src + "\x00" + by_i[j].txt
                        for j in range(i - window, i + window)
                        if j in by_i
                    )
                )

            sp = vv["speakers"]
            turns = [i for i in range(1, len(sp)) if sp[i] and sp[i] != sp[i - 1] and i in by_i]
            (tmp / "reply_state.json").write_text(
                json.dumps({str(i): key(i) for i in turns if i not in mine}), encoding="utf-8"
            )
            rkw = {k_: v for k_, v in kw.items() if k_ in rp and k_ not in ("speakers",)}
            ed.reply_edit(
                recs,
                inv,
                lang,
                LANG.get(lang, lang),
                sp,
                state_path=tmp / "reply_state.json",
                **rkw,
            )
        for i, k in mine.items():
            if i in by_i:
                out[k] = by_i[i].txt
    return out


if os.environ.get("SPOT_EDITOR"):
    out = editor_pass(out)
Path(sys.argv[3]).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
