#!/usr/bin/env python3
"""Mine the reviewers' CONSISTENT word changes across videos -> glossary proposals with evidence.

A judge can't know the house terminology (油漆工 -> 粉刷工, 仓库 -> 存储库, 应用 -> 应用程序), and the learn
job looks at one video at a time, so it never sees enough evidence. This looks at every reviewed line
pair (review_pairs.py: the draft the reviewer saw vs their final), at word level:

  1. each short word replacement draft -> final (jieba tokens, difflib), counted per video;
  2. for each, how often the reviewers KEPT the draft word unchanged (it stayed in the final) — a
     change made in 3 lines but kept in 30 is a contextual choice, not a rule;
  3. counter-evidence: the draft word in OTHER reviewed finals (all reviewed batches, also those
     without a kept draft);
  4. candidates (>= --min-videos videos, change rate >= --min-rate) go to one LLM call with their
     example lines: which English term do they render, and is the change a general terminology
     preference or a context-specific choice?

Output: <out>/REPORT.md + candidates.json. Nothing is applied — a person approves each proposal.

  PYTHONPATH=src python3 scripts/eval/mine_term_changes.py --pairs $DEV_TITLES_OUTPUT/judge_calib/zh-Hans.jsonl \
      --out $DEV_TITLES_OUTPUT/term_mining
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# reviewed finals (*Reviewed*.srt, any depth) live under the input root
INPUT = Path(os.environ.get("DEV_TITLES_INPUT", "input")).resolve()

from dev_titles.glossary import load  # noqa: E402
from dev_titles.learn import _json  # noqa: E402
from dev_titles.translate import CliTranslator  # noqa: E402

PUNCT = re.compile(r"^[\s，、。．・　|？?！!：:；;“”\"'（）()…—-]+$")
CJK = re.compile(r"[一-鿿]")

ASK = """Native {L} subtitle reviewers of a developer YouTube channel changed our machine subtitles.
Below, word changes they made CONSISTENTLY across several videos, each with example lines (English, our
draft, their final). For each change decide:
- "term": the English word or phrase both words render (as it appears in the English lines);
- "kind": "terminology" if the reviewers' word is the house rendering of that English term in general
  (apply it everywhere), "context" if it depends on the sentence, "grammar" if it is a grammatical or
  stylistic adjustment rather than a term (particles, measure words, verb forms);
- "why": max 15 words.
CHANGES:
{items}
Return ONLY a JSON array: [{{"id": <id>, "term": "...", "kind": "terminology|context|grammar", "why": "..."}}]"""


def tokens(s: str) -> list[str]:
    import jieba

    return [t for t in jieba.lcut(re.sub(r"\s+", "", s or "")) if t]


def replacements(draft: str, human: str):
    """Short word replacements draft -> final (1-3 tokens a side, CJK, <= 6 chars)."""
    a, b = tokens(draft), tokens(human)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "replace" or not (1 <= i2 - i1 <= 3 and 1 <= j2 - j1 <= 3):
            continue
        x, y = "".join(a[i1:i2]), "".join(b[j1:j2])
        if PUNCT.match(x) or PUNCT.match(y) or len(x) > 6 or len(y) > 6:
            continue
        if CJK.search(x) and CJK.search(y) and x != y:
            yield x, y


def reviewed_finals(lang: str) -> dict[str, str]:
    out = {}
    for p in sorted(INPUT.rglob("*Reviewed*.srt")):
        if lang in re.split(r"[._-]", p.name) or f".{lang}" in p.name:
            out[p.name.split(".")[0].replace("_Reviewed", "")] = p.read_text(encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="zh-Hans")
    ap.add_argument("--min-videos", type=int, default=2)
    ap.add_argument("--min-rate", type=float, default=0.5)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in Path(a.pairs).read_text(encoding="utf-8").splitlines()]
    changes: dict = defaultdict(lambda: {"videos": set(), "n": 0, "ex": []})
    for r in rows:
        if not r["edited"]:
            continue
        for x, y in replacements(r["draft"], r["human"]):
            c = changes[(x, y)]
            c["videos"].add(r["video"])
            c["n"] += 1
            if len(c["ex"]) < 4:
                c["ex"].append({k: r[k] for k in ("video", "en", "draft", "human")})
    finals = reviewed_finals(a.lang)
    g = load(ROOT / "glossary" / "glossary.yaml")
    gl = {}
    for t in g.terms:
        tr = (t.translations.get(a.lang) or "").strip()
        if tr:
            gl[tr] = t.source
    cands = []
    for (x, y), c in changes.items():
        if len(c["videos"]) < a.min_videos:
            continue
        drafts_with = [r for r in rows if x in re.sub(r"\s", "", r["draft"])]
        kept = sum(1 for r in drafts_with if x in re.sub(r"\s", "", r["human"]))
        changed = sum(
            1
            for r in drafts_with
            if x not in re.sub(r"\s", "", r["human"]) and y in re.sub(r"\s", "", r["human"])
        )
        rate = changed / max(changed + kept, 1)
        if rate < a.min_rate:
            continue
        # counter-evidence: the draft word in reviewed finals of videos where no change was recorded
        elsewhere = {
            v: t.count(x) for v, t in finals.items() if v not in c["videos"] and t.count(x)
        }
        cands.append(
            {
                "from": x,
                "to": y,
                "videos": sorted(c["videos"]),
                "changes": c["n"],
                "kept": kept,
                "rate": round(rate, 2),
                "reviewers_use_from_elsewhere": elsewhere,
                "glossary_has_from": gl.get(x),
                "glossary_has_to": gl.get(y),
                "examples": c["ex"],
            }
        )
    cands.sort(key=lambda c: (-len(c["videos"]), -c["changes"]))
    if cands:
        items = "\n".join(
            json.dumps(
                {
                    "id": k,
                    "from": c["from"],
                    "to": c["to"],
                    "examples": [
                        {k2: e[k2] for k2 in ("en", "draft", "human")} for e in c["examples"][:3]
                    ],
                },
                ensure_ascii=False,
            )
            for k, c in enumerate(cands)
        )
        got = _json(
            CliTranslator()._invoke(
                ASK.format(
                    L="Simplified Chinese" if a.lang == "zh-Hans" else "Japanese", items=items
                )
            )
            or ""
        )
        for x in got if isinstance(got, list) else []:
            if isinstance(x, dict) and isinstance(x.get("id"), int) and 0 <= x["id"] < len(cands):
                cands[x["id"]].update({k: x.get(k) for k in ("term", "kind", "why")})
    (out / "candidates.json").write_text(
        json.dumps(cands, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    rep = [
        f"# Consistent reviewer word changes ({a.lang}) — {len(rows)} line pairs, "
        f"{len({r['video'] for r in rows})} videos",
        "",
        "| draft → reviewer | EN term | kind | videos | changed / kept | reviewers keep the draft word elsewhere | glossary |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cands:
        gl_note = (
            f"has {c['glossary_has_from']} = {c['from']}" if c["glossary_has_from"] else ""
        ) or (f"has {c['glossary_has_to']} = {c['to']}" if c["glossary_has_to"] else "")
        ew = ", ".join(
            f"{v} ×{n}"
            for v, n in sorted(c["reviewers_use_from_elsewhere"].items(), key=lambda kv: -kv[1])[:3]
        )
        rep.append(
            f"| {c['from']} → {c['to']} | {c.get('term', '')} | {c.get('kind', '')} | {len(c['videos'])} | "
            f"{c['changes']} / {c['kept']} | {ew} | {gl_note} |"
        )
    rep.append("")
    for c in cands:
        rep.append(f"## {c['from']} → {c['to']} ({c.get('kind', '')}: {c.get('why', '')})")
        for e in c["examples"]:
            rep.append(
                f"- {e['video']}: {e['en'][:100]}\n  - draft: {e['draft']}\n  - final: {e['human']}"
            )
        rep.append("")
    (out / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")
    print(f"{len(cands)} candidate(s) -> {out / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
