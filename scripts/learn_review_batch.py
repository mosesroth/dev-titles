#!/usr/bin/env python3
"""Learn from every reviewed delivery, automatically — the step that used to need a prompt.

Runs by itself at the end of scripts/check_vendor_review.py (vendor batches) and
scripts/rewrap_reviewed.py (reviews that come back as files), detached, so a delivery never waits
on it. By hand:

  PYTHONPATH=src python3 scripts/learn_review_batch.py --missing        # every review not yet learned
  PYTHONPATH=src python3 scripts/learn_review_batch.py input/my_batch [--no-apply]

Per reviewed video (<reviewed>/<batch>/<video>.<lang>_Reviewed*.srt):
  1. scripts/learn_from_review.py against output/ticket/<video> (judge-filtered lessons, validated
     with vs without; --apply commits them). The reviewer's file is the CHECKED final when one exists
     (output/FINAL_SUBTITLES_<batch>/): edits the vendor check reverted were vendor errors, not lessons.
  2. its report, lessons and aligned rows are kept in output/learn/<batch>/<video>_<lang>/ and the
     run is recorded in output/learn/ledger.json — so a missed review is visible (--missing), and a
     report can't be lost in a cleanup of the run folders.
Then, over the aligned rows of EVERY learned video in the language (the cross-video corpus):
  3. consistent word changes (learn.conventions): the reviewers' house terms, which judges can't know
     — glossary lessons when established (learn.convention_lessons), applied with --apply;
  4. recurring punctuation/spacing edits: reported in output/learn/conventions.<lang>.md for a code
     change (they need one: line breaks, ？, quotes are the finalizer's job, not the glossary's).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dev_titles.learn import (  # noqa: E402
    LANG_NAME,
    apply_lessons,
    convention_lessons,
    conventions,
    triage,
)

OUTPUT = Path(os.environ.get("DEV_TITLES_OUTPUT", "output")).resolve()  # deliverables root
# the reviewers' own finals (the same corpus latin_names.default_reviewed_glob reads)
REVIEWED = Path(os.environ.get("DEV_TITLES_INPUT", "input")).resolve()
LEARN = OUTPUT / "learn"
LEDGER = LEARN / "ledger.json"


def lang_of(name: str) -> str | None:
    for lang in LANG_NAME:
        if f".{lang}" in name or f"_{lang}" in name:
            return lang
    return None


def video_of(name: str) -> str:
    return re.split(r"[._]", name)[0]


def reviews(batch_dir: Path):
    """(video, lang, path) of each reviewed subtitle in a batch folder."""
    for p in sorted(batch_dir.glob("*.srt")):
        lang = lang_of(p.name)
        if lang:
            yield video_of(p.name), lang, p


def ledger() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {}


def _save_ledger(d: dict) -> None:
    LEARN.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(d, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def keep_in_corpus(src_dir: Path) -> Path:
    """The batch's reviewed files under <reviewed>/<batch>/ (the reviewers' corpus)."""
    dst = REVIEWED / src_dir.name
    if src_dir.resolve() != dst.resolve():
        for _, _, p in reviews(src_dir):
            if not (dst / p.name).exists():
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst / p.name)
    return dst


def checked_final(batch: str, video: str, lang: str, raw: Path) -> Path:
    p = OUTPUT / f"FINAL_SUBTITLES_{batch}" / f"{video}.{lang}.srt"
    return p if p.exists() else raw


def _git(*args) -> None:
    subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True)


def learn_video(batch: str, video: str, lang: str, path: Path, apply: bool) -> dict:
    run = OUTPUT / "ticket" / video
    if not (run / "cache.jsonl").exists():
        return {"status": "skipped: no machine run (output/ticket/<video>/cache.jsonl)"}
    out = LEARN / batch / f"{video}_{lang}"
    out.mkdir(parents=True, exist_ok=True)
    human = checked_final(batch, video, lang, path)
    cmd = [sys.executable, str(ROOT / "scripts" / "learn_from_review.py"), "--lang", lang]
    cmd += ["--run-dir", str(run), "--human", str(human), "--out", str(out)]
    cmd += ["--apply"] if apply else []
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}
    with open(out / "run.log", "w", encoding="utf-8") as log:
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    rep = out / "REPORT.md"
    head = (
        [ln for ln in rep.read_text(encoding="utf-8").splitlines() if ln.startswith("- ")][:2]
        if rep.exists()
        else []
    )
    return {
        "status": "done" if r.returncode == 0 else f"FAILED (exit {r.returncode}, see run.log)",
        "human": os.path.relpath(human),
        "summary": head,
    }


def corpus_pass(lang: str, apply: bool, invoke) -> list[str]:
    rows = []
    for p in sorted(LEARN.glob(f"*/*_{lang}/rows.json")):
        rows += json.loads(p.read_text(encoding="utf-8"))
    if not rows:
        return []
    words, punct = conventions(rows, lang, REVIEWED)
    lessons, rest = convention_lessons(words, invoke, lang)
    from dev_titles.glossary import load

    gp = ROOT / "glossary" / "glossary.yaml"
    todo, person = triage(lessons, load(str(gp)))
    tag = f"reviewer conventions {lang} {time.strftime('%Y-%m-%d')}"
    changes = apply_lessons(todo, gp, gp.parent, tag) if apply and todo else []
    vids = len({r["video"] for r in rows})
    rep = [
        f"# Reviewer conventions ({lang}) — {len(rows)} aligned lines, {vids} videos",
        "",
        "Consistent changes across videos (learn.conventions). Word conventions that are established",
        "become soft glossary terms (applied: below); punctuation conventions need a code change.",
        "",
        f"## Applied as glossary terms ({len(changes)})",
        "",
        *[f"- {c}" for c in changes],
        *(
            [f"- (not applied: --no-apply) {x['source']} -> {x['target']}" for x in todo]
            if not apply
            else []
        ),
        "",
        f"## Conflicts with the glossary — a person decides ({len(person)})",
        "",
        *[f"- {x['source']} -> {x['target']}: {x.get('conflict')}" for x in person],
        "",
        f"## Word changes that are not (yet) a rule ({len(rest)})",
        "",
        *[
            f"- {c['from']} -> {c['to']} ({len(c['videos'])} videos): {c['not_a_lesson']}"
            for c in rest
        ],
        "",
        f"## Punctuation / spacing conventions — need code ({len(punct)})",
        "",
    ]
    for c in punct:
        ctx = {"Q": " after a question particle", "^": " at line start"}.get(c["after"], "")
        rep.append(f"- {c['from']!r} -> {c['to']!r}{ctx}: {c['n']}x in {len(c['videos'])} videos")
        for e in c["examples"][:2]:
            rep.append(f"  - ours: {e['ours']}  \n    reviewer: {e['human']}")
    (LEARN / f"conventions.{lang}.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    return changes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "batches",
        nargs="*",
        help="reviewed batch folders (<reviewed>/<batch>)",
    )
    ap.add_argument("--missing", action="store_true", help="every reviewed video not in the ledger")
    ap.add_argument("--no-apply", action="store_true", help="report only; commit nothing")
    a = ap.parse_args(argv)
    apply = not a.no_apply
    from dev_titles.localize import _lock

    LEARN.mkdir(parents=True, exist_ok=True)
    # one job at a time: a second delivery arriving mid-run waits, then finds the first one's work done
    lock = open(LEARN / ".lock", "a+", encoding="utf-8")
    _lock(lock)
    dirs = [keep_in_corpus(Path(b)) for b in a.batches]
    if a.missing:
        dirs += [d for d in sorted(REVIEWED.iterdir()) if d.is_dir() and d not in dirs]
    led = ledger()
    langs = set()
    for d in dirs:
        for video, lang, p in reviews(d):
            key = f"{d.name}/{video}/{lang}"
            if led.get(key, {}).get("status") == "done":
                continue
            print(f"[{time.strftime('%H:%M')}] learn {key}", flush=True)
            res = learn_video(d.name, video, lang, p, apply)
            led[key] = {**res, "date": time.strftime("%Y-%m-%d")}
            _save_ledger(led)
            print(f"  {res['status']}", flush=True)
            langs.add(lang)
            if apply:
                _git("add", str(LEARN))
                _git("commit", "-qm", f"learn: {key} — {res['status']}")
    from dev_titles.translate import CliTranslator

    for lang in sorted(langs):
        changes = corpus_pass(lang, apply, CliTranslator()._invoke)
        print(f"conventions {lang}: {len(changes)} applied -> {LEARN / f'conventions.{lang}.md'}")
        if apply:
            _git("add", str(LEARN), "glossary")
            msg = f"learn: reviewer conventions ({lang}) — {len(changes)} glossary change(s)"
            _git("commit", "-qm", msg + "".join(f"\n- {c}" for c in changes))
    return 0


def launch_detached(batch_dir: str) -> None:
    """Start the learn job for a delivery in the background (called by the intake scripts)."""
    LEARN.mkdir(parents=True, exist_ok=True)
    log = open(LEARN / f"auto_{Path(batch_dir).name}.log", "a", encoding="utf-8")
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), batch_dir, "--missing"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"},
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"learn_from_review started in the background -> {log.name}")


if __name__ == "__main__":
    sys.exit(main())
