#!/usr/bin/env python3
"""Learn from a human review (dev_titles/learn.py): reviewer improvements -> validated harness changes.

  # from a finished A/B (rows already aligned and judged):
  PYTHONPATH=src python3 scripts/learn_from_review.py --lang ja \
      --rows output/ab/MyVideo/vs_human/ab_rows.json \
      --run-dir output/ab/MyVideo --out output/learn/MyVideo_ja [--apply]

  # from a delivery that came back reviewed (aligns the reviewer's file to the EN by content):
  PYTHONPATH=src python3 scripts/learn_from_review.py --lang zh-Hans --run-dir output/ticket/MyVideo \
      --human reviewed/MyVideo.zh-Hans.srt --out output/learn/MyVideo_zh [--apply]

--run-dir holds the machine draft the reviewer saw (cache.jsonl) and the EN (en.vtt / *.en.vtt).
Writes <out>/REPORT.md and <out>/lessons.json. --apply writes the ACCEPTED lessons into
glossary/glossary.yaml and glossary/editor_examples.<lang>.jsonl and commits them (local commit; a
person reviews it). Lessons needing a decision (glossary conflicts) are only reported.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dev_titles.cli import _qe_second_judge, _read_any  # noqa: E402
from dev_titles.glossary import load  # noqa: E402
from dev_titles.learn import (  # noqa: E402
    LANG_NAME,
    apply_lessons,
    human_better_rows,
    propose,
    reviewer_counterevidence,
    term_evidence_elsewhere,
    triage,
    validate,
    with_lesson,
)
from dev_titles.localize import (  # noqa: E402
    _ctx_for,
    _fixed_line,
    _line_context,
    _normalizer,
    _speakers_for,
    effective_glossary,
)
from dev_titles.resegment import merge_en_to_sentences  # noqa: E402
from dev_titles.translate import CliTranslator, translate_segment  # noqa: E402

GLOSSARY = ROOT / "glossary" / "glossary.yaml"
OUTPUT = Path(os.environ.get("DEV_TITLES_OUTPUT", "output")).resolve()  # deliverables root


def _rows(a, run: Path, sents) -> list[dict]:
    if a.rows:
        d = json.loads(Path(a.rows).read_text(encoding="utf-8"))[a.lang]
        return [{**r, "ours": r.get("new") or r.get("ours") or ""} for r in d]
    out = Path(a.out)
    base = out / f"human_{a.lang}"
    en_p = _en_path(run)
    if not Path(str(base) + ".alignment.json").exists():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "retime_drifted.py"),
                a.human,
                str(en_p),
                str(base),
            ],
            check=True,
            cwd=ROOT,
        )
    human = json.loads(Path(str(base) + ".alignment.json").read_text(encoding="utf-8"))
    ours = {}
    for line in open(run / "cache.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("lang") == a.lang:
            ours[r["src"].strip()] = r["txt"]
    return [
        {
            "i": k,
            "en": s.text,
            "human": human[k].replace("|", ""),
            "ours": ours.get(s.text.strip(), ""),
        }
        for k, s in enumerate(sents)
    ]


def _en_path(run: Path) -> Path:
    c = sorted(run.glob("*.en.vtt")) + [run / "en.vtt"]
    return next(p for p in c if p.exists())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(LANG_NAME))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--rows", help="ab_rows.json from scripts/eval/ab_vs_human.py (skips align + judge)"
    )
    ap.add_argument("--human", help="the reviewer's final subtitle file (when no --rows)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--max-validate", type=int, default=12)
    a = ap.parse_args(argv)
    if not a.rows and not a.human:
        ap.error("--rows or --human is required")
    run, out = Path(a.run_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sents = merge_en_to_sentences(_read_any(str(_en_path(run)), "en"))
    rows = _rows(a, run, sents)
    claude = CliTranslator()._invoke
    gpt = _qe_second_judge(types.SimpleNamespace(qe_judge=True, single_judge=False))
    judges = [j for j in (claude, gpt) if j]
    L = LANG_NAME[a.lang]
    g0 = load(str(GLOSSARY))
    kept = human_better_rows(rows, judges, L)
    print(f"{len(kept)} row(s) where the reviewer's line is better (both judges)")
    st = (g0.style or {}).get(a.lang, {}) or {}
    notes = (st.get("notes_conversational") or "") + "\n" + (st.get("notes") or "")
    lessons = propose(kept, claude, a.lang, notes)
    todo, person = triage(lessons, g0)
    print(
        f"{len(lessons)} lesson(s) proposed; {len(todo)} to validate, {len(person)} need a person"
    )

    # the harness's translate call, with this video's context (brief, speakers, meaning notes)
    vv: dict = {}
    for name, key in (("video_brief.txt", "brief"),):
        if (run / name).exists():
            vv[key] = (run / name).read_text(encoding="utf-8").strip()
    if (run / "speakers.json").exists():
        vv["speakers"] = json.loads((run / "speakers.json").read_text())["labels"]
    if (run / "meaning_notes.json").exists():
        vv["meaning_notes"] = json.loads((run / "meaning_notes.json").read_text())
    spk, ctx = _speakers_for(sents, vv), _ctx_for(sents, vv)
    norm = _normalizer(a.lang, True)
    memo: dict = {}

    def translate(g, r):
        key = (id(g), r["i"])
        if key not in memo:
            gg, dnt = effective_glossary(g or g0, vv if r["i"] >= 0 else None, True, lang=a.lang)
            src = sents[r["i"]].text if r["i"] >= 0 else r["en"]
            fixed = _fixed_line(gg, a.lang, src)
            if fixed is not None:
                memo[key] = fixed
            else:
                here = r["i"] >= 0  # cross-video evidence: no context from this video
                txt, _ = translate_segment(
                    CliTranslator(),
                    gg,
                    src,
                    a.lang,
                    full_context=ctx if here else "",
                    line_context=_line_context(r["i"], sents, spk, vv, lang=a.lang) if here else "",
                    reflect=True,
                    char_budget=60,
                    extra_dnt=dnt,
                )
                memo[key] = norm(txt, src)
        return memo[key]

    accepted, rejected = [], []
    for x in todo[: a.max_validate]:
        if x["kind"] == "editor_example":  # examples are gated where they're used (editor pass)
            accepted.append({**x, "validation": "n/a (editor pass gates examples)"})
            continue
        if x["kind"] == "glossary_term":
            counter = reviewer_counterevidence(x, OUTPUT / "FINAL_SUBTITLES", run.name)
            if sum(int(h.rsplit("x", 1)[1]) for h in counter) >= 2:
                rejected.append(
                    {**x, "validation": {"score": {"reviewers_keep_avoided_form": counter}}}
                )
                print(
                    f"  reject {x['kind']}: {_label(x)} — reviewers keep it elsewhere: {counter[:3]}"
                )
                continue
        if x["kind"] == "glossary_term":  # a term applies everywhere: test it on other videos too
            x["extra_evidence"] = term_evidence_elsewhere(
                x["source"],
                a.lang,
                sorted((OUTPUT / "ticket").glob("*/cache.jsonl")),
                skip=run.name,
            )
        x["_glossary"] = with_lesson(g0, x)
        res = validate(x, translate, judges, rows)
        x.pop("_glossary")
        (accepted if res["accepted"] else rejected).append({**x, "validation": res})
        print(
            f"  {'ACCEPT' if res['accepted'] else 'reject'} {x['kind']}: {_label(x)} {res['score']}"
        )

    tag = f"{run.name} {a.lang} {time.strftime('%Y-%m-%d')}"
    rep = [
        f"# Learned from review — {tag}",
        "",
        f"- reviewer better (both judges): {len(kept)} row(s)",
        f"- lessons proposed: {len(lessons)}; accepted {len(accepted)}, rejected {len(rejected)}, "
        f"need a person {len(person)}",
        "",
    ]
    for title, xs in (
        ("Accepted", accepted),
        ("Rejected by validation", rejected),
        ("Needs a person", person),
    ):
        rep += [f"## {title}", ""]
        for x in xs:
            ev = x["evidence"][0]
            rep.append(
                f"- **{x['kind']}** {_label(x)}"
                + (f" — {x['conflict']}" if x.get("conflict") else "")
            )
            rep.append(
                f"  - e.g. EN: {ev['en']}  \n    ours: {ev['ours']}  \n    reviewer: {ev['human']}"
            )
            v = x.get("validation")
            if isinstance(v, dict):
                rep.append(f"  - validation: {v['score']}")
        rep.append("")
    (out / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")
    (out / "lessons.json").write_text(
        json.dumps(
            {"accepted": accepted, "rejected": rejected, "person": person},
            ensure_ascii=False,
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"report -> {out / 'REPORT.md'}")
    if a.apply and accepted:
        changes = apply_lessons(accepted, GLOSSARY, GLOSSARY.parent, tag)
        load(str(GLOSSARY))  # still parses
        subprocess.run(["git", "add", "glossary"], cwd=ROOT, check=True)
        msg = f"learned from review ({tag}): {len(changes)} change(s)\n\n" + "\n".join(
            f"- {c}" for c in changes
        )
        subprocess.run(
            ["git", "commit", "-qm", msg + f"\n\nReport: {out / 'REPORT.md'}"], cwd=ROOT, check=True
        )
        print("applied + committed:\n  " + "\n  ".join(changes))
    return 0


def _label(x) -> str:
    if x["kind"] == "glossary_term":
        return f"{x.get('source')!r} -> {x.get('target')!r}"
    if x["kind"] == "fixed_line":
        return f"{x.get('source')!r} -> {x.get('target')!r}"
    if x["kind"] == "style_note":
        return re.sub(r"\s+", " ", x.get("text", ""))[:100]
    return x.get("label", "")


if __name__ == "__main__":
    sys.exit(main())
