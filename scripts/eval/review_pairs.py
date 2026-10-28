#!/usr/bin/env python3
"""Sentence-aligned (draft the reviewer saw, reviewer's final) pairs — the ground truth for judging.

Every judge-based number in our A/Bs rests on the judges preferring what native reviewers prefer. On
one reviewed batch they endorsed only 126 of 649 reviewer edits (19%), so this builds the data to measure and
improve that: for each vendor-reviewed video whose PRE-REVIEW draft is kept (the delivered file the
review replaced), both tracks are aligned to the EN sentences by content (scripts/retime_drifted.py)
and paired line by line.

  batches: $DEV_TITLES_INPUT/<batch>/<video>.<lang>_Reviewed.srt
           + $DEV_TITLES_OUTPUT/<draft dir>/<video>.<lang>.srt
  Every batch check_vendor_review.py kept the drafts of ($DEV_TITLES_OUTPUT/review_drafts/<batch>/)
  is found automatically; older batches are passed as --batch <batch>:<draft dir>.

Output: <out>/<lang>.jsonl, one row per EN sentence: {batch, video, i, en, prev, next, draft, human,
edited}. Used by judge_calibration.py (judge agreement) and vendor_edit_kpi.py (edit rate per batch).

  PYTHONPATH=src python3 scripts/eval/review_pairs.py --out $DEV_TITLES_OUTPUT/judge_calib \
      [--batch <batch>:<draft dir> ...]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dev_titles.cli import _read_any  # noqa: E402
from dev_titles.resegment import merge_en_to_sentences  # noqa: E402

OUT = Path(os.environ.get("DEV_TITLES_OUTPUT", "output")).resolve()  # deliverables root
INP = Path(os.environ.get("DEV_TITLES_INPUT", "input")).resolve()  # reviewed batches


def norm(s: str) -> str:
    return re.sub(r"[\s，、。．・　|？?！!]", "", s or "")


def align(track: Path, en: Path, base: Path) -> list[str] | None:
    if not Path(f"{base}.alignment.json").exists():
        base.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "retime_drifted.py"),
                str(track),
                str(en),
                str(base),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if r.returncode:
            print(f"  align {track.name} FAILED: {r.stderr.strip()[-160:]}")
            return None
    return json.loads(Path(f"{base}.alignment.json").read_text(encoding="utf-8"))


def batches(lang: str, extra=()) -> list[tuple[str, str]]:
    """(batch folder under the input root, folder under the output root holding the drafts that batch
    reviewed — the files the reviewed finals replaced): the `extra` pairs (--batch), plus every batch
    check_vendor_review.py kept the drafts of (review_drafts/<batch>/), so a new reviewed batch is
    included without any flag.
    """
    out = list(extra)
    have = {b for b, _ in out}
    for d in sorted((OUT / "review_drafts").glob("*")):
        if d.is_dir() and d.name not in have and any(d.glob(f"*.{lang}.srt")):
            out.append((d.name, f"review_drafts/{d.name}"))
    return out


def build(lang: str, out: Path, extra=()) -> list[dict]:
    rows = []
    for batch, draft_dir in batches(lang, extra):
        for human in sorted((INP / batch).glob(f"*{lang}*Reviewed*.srt")):
            v = human.name.split(".")[0].replace("_Reviewed", "")
            draft = OUT / draft_dir / f"{v}.{lang}.srt"
            run = OUT / "ticket" / v
            en = run / "en.vtt"
            if not draft.exists() or not en.exists():
                print(f"  skip {v}: no draft or EN")
                continue
            h = align(human, en, out / "align" / f"{v}.{lang}.human")
            d = align(draft, en, out / "align" / f"{v}.{lang}.draft")
            sents = merge_en_to_sentences(_read_any(str(en), "en"))
            if h is None or d is None or not (len(h) == len(d) == len(sents)):
                print(f"  skip {v}: alignment length mismatch")
                continue
            for k, s in enumerate(sents):
                dd, hh = d[k].replace("|", "").strip(), h[k].replace("|", "").strip()
                if not dd or not hh:
                    continue
                rows.append(
                    {
                        "batch": batch,
                        "video": v,
                        "i": k,
                        "en": s.text,
                        "prev": sents[k - 1].text if k else "",
                        "next": sents[k + 1].text if k + 1 < len(sents) else "",
                        "draft": dd,
                        "human": hh,
                        "edited": norm(dd) != norm(hh),
                    }
                )
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="zh-Hans")
    ap.add_argument(
        "--batch",
        action="append",
        default=[],
        metavar="BATCH:DRAFT_DIR",
        help="a reviewed batch whose drafts are not in review_drafts/ (repeatable)",
    )
    a = ap.parse_args(argv)
    out = Path(a.out)
    extra = [tuple(b.split(":", 1)) for b in a.batch if ":" in b]
    rows = build(a.lang, out, extra)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.lang}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    ed = sum(r["edited"] for r in rows)
    print(f"{len(rows)} aligned lines, {ed} edited by the reviewer -> {out / (a.lang + '.jsonl')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
