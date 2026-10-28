"""Leave-one-video-out check of latin_names.decide against the reviewers own choice per name.

PYTHONPATH=src python3 scripts/eval/latin_names_check.py   (2026-10-27: 400/436 = 92%)

Reads the same corpus as the pipeline: SUBTITLE_REVIEWED_GLOB (reviewed finals) and SUBTITLE_EN_DIR
(their delivered EN), defaults in dev_titles.latin_names.
"""

import collections
import os
import re
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dev_titles.cli import _qe_second_judge
from dev_titles.glossary import load
from dev_titles.latin_names import (
    _has,
    decide,
    default_en_dir,
    default_reviewed_glob,
    load_corpus,
    names_in,
    reviewed_files,
)
from dev_titles.translate import CliTranslator

g = load("glossary/glossary.yaml")
inv = CliTranslator()._invoke
second = (
    None
    if os.environ.get("ONE_JUDGE")
    else _qe_second_judge(types.SimpleNamespace(qe_judge=True, single_judge=False))
)
LN = {"zh-Hans": "Simplified Chinese", "ja": "Japanese"}


def pinned(lang):
    return [
        f.lower()
        for t in g.terms
        for f in t.all_source_forms
        if (t.translations.get(lang) or "").strip() and t.translations[lang] != t.source
    ]


jobs = []
for lang in LN:
    pin = pinned(lang)
    for tp in reviewed_files(default_reviewed_glob()):
        if lang not in re.split(r"[._-]", tp.name) and f".{lang}" not in tp.name:
            continue
        v = tp.name.split(".")[0].replace("_Reviewed", "")
        ep = Path(default_en_dir()) / f"{v}.en.srt"
        if not ep.exists():
            continue
        en, tgt = ep.read_text(), tp.read_text()
        ns = sorted(
            {
                n
                for n in names_in(re.sub(r"\n(?!\n)", " ", en))
                if n.lower() not in pin
                and not any(re.search(rf"\b{re.escape(p)}\b", n.lower()) for p in pin)
            }
        )
        flat = re.sub(r"\n(?!\n)", " ", en)
        lines = {
            n: next((l for l in re.split(r"(?<=[.?!])\s+", flat) if n in l), "")[:200] for n in ns
        }
        if ns:
            jobs.append(
                (lang, v, ns, {n: "latin" if _has(n, tgt) else "translate" for n in ns}, lines)
            )


def run(j):
    lang, v, ns, truth, lines = j
    pred = decide(
        ns, lang, LN[lang], load_corpus(lang, holdout=[v]), inv, lines=lines, second=second
    )
    return lang, v, truth, pred


res = list(ThreadPoolExecutor(6).map(run, jobs))
c = collections.Counter()
errs = []
for lang, v, truth, pred in res:
    for n, t in truth.items():
        p = pred.get(n, "?")
        p = "translate" if p == "fragment" else p
        c[(lang, t, p)] += 1
        if p != t:
            errs.append((lang, v, n, t, p))
for k in sorted(c):
    print(k, c[k])
ok = sum(n for (l, t, p), n in c.items() if t == p)
tot = sum(c.values())
print(f"agree {ok}/{tot} = {ok/tot:.0%}")
for e in errs:
    print("  ", e)
