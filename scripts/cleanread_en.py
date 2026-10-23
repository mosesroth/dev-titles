#!/usr/bin/env python3
"""Constrained clean-read of an existing EN track (e.g. a live YouTube track that never got one): deletions
(stutters, doubled words, filler), punctuation and capitalization only — see dev_titles/en_cleanread.py.
The pipeline now runs this automatically on unscripted EN before the EN review; this is for retrofits.

  PYTHONPATH=src python3 scripts/cleanread_en.py <en.srt|vtt> <out base path>   (writes .srt .vtt .log)
"""
import sys

from dev_titles.cli import _read_any
from dev_titles.en_cleanread import clean_read
from dev_titles.formats import write_srt, write_vtt
from dev_titles import llm
from dev_titles.validate.rules import validity_report

src, dst = sys.argv[1], sys.argv[2]
en = _read_any(src, "en")
out, applied, rejected = clean_read(en, llm.invoke())
open(dst + ".srt", "w", encoding="utf-8").write(write_srt(out))
open(dst + ".vtt", "w", encoding="utf-8").write(write_vtt(out))
open(dst + ".log", "w", encoding="utf-8").write("\n".join(f"{a['i']}: {a['old']!r} -> {a['new']!r}" for a in applied) + "\n")
print(f"cleaned {len(applied)} cues, rejected {rejected}; err-cues "
      f"{validity_report(en, False)['cues_with_errors']} -> {validity_report(out, False)['cues_with_errors']}")
