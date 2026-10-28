#!/usr/bin/env python3
"""How do the HUMAN zh translators render English it / this / that / they? Measure it on the corpus.

The zh pronoun rule (localize._PRONOUN_RULE) was written from a handful of A/B rows; an out-of-sample
check could not tell it from noise. This counts, over every approved zh line we have, how each
English pronoun occurrence is rendered by role:

  corpus: the vendor-reviewed zh finals (--reviewed, a glob under $DEV_TITLES_INPUT), aligned to our
          EN sentences by content (scripts/retime_drifted.py; cached alignments reused), each with OUR
          draft from the video's cache.jsonl; plus the TM's human zh lines (tm/tm.jsonl, no draft).
  per sentence with a candidate word, one LLM call (batched) tags each occurrence: role (subject,
  object, possessive, determiner, dummy, relative/complementizer...), what it refers to (thing /
  people / situation), and how the HUMAN line renders it (dropped, 它/它们, 这/那…, 他们, the referent
  named, restructured) — and, where we have it, how OUR draft does.

Output: <out>/rows.jsonl (every tagged occurrence) and <out>/REPORT.md (role x rendering tables,
human vs ours, examples). Videos are kept on each row so a rule derived from it can be held out.

  PYTHONPATH=src python3 scripts/eval/pronoun_corpus.py --out $DEV_TITLES_OUTPUT/pronoun_corpus \
      [--reviewed "*/*zh*Reviewed*.srt"]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dev_titles.cli import _read_any  # noqa: E402
from dev_titles.learn import _json  # noqa: E402
from dev_titles.resegment import merge_en_to_sentences  # noqa: E402
from dev_titles.translate import CliTranslator  # noqa: E402

OUTPUT = Path(os.environ.get("DEV_TITLES_OUTPUT", "output")).resolve()  # deliverables root
INPUT = Path(os.environ.get("DEV_TITLES_INPUT", "input")).resolve()  # reviewed batches
WORDS = re.compile(r"\b(it|its|it's|itself|they|them|their|they're|this|that|these|those)\b", re.I)
RENDER = (
    "dropped",
    "ta",
    "demonstrative",
    "people_pronoun",
    "referent_named",
    "restructured",
    "other",
)

TAG = """For each row: an English subtitle sentence from a developer video, and one or two Simplified
Chinese translations (HUMAN = approved by a native reviewer; OURS = machine draft, may be absent).
For EVERY occurrence of it / its / itself / they / them / their / this / that / these / those in the
English (including inside it's / they're), give:
  "word": the English word,
  "role": subject | object (incl. after a preposition) | possessive | determiner ("that model") |
          dummy ("it's easy to…", "it's raining") | relative_or_complementizer ("the app that…",
          "I think that…") | intensifier ("that big") | other,
  "refers_to": thing | people | situation (an idea, event, or what was just said) | none,
  "human": how the HUMAN line renders that occurrence — dropped (nothing stands for it) | ta (它 / 它们
          / 其) | demonstrative (这 / 那 / 这个 / 那个 / 这些 / 这样 / 此 …) | people_pronoun (他们 /
          她们 / 他 / 她) | referent_named (the thing or people named outright) | restructured (the
          sentence is reworded so the question doesn't apply) | other,
  "human_text": the Chinese words that render it ("" if dropped),
  "ours", "ours_text": the same for OURS (omit both if there is no OURS line).
ROWS:
{rows}
Return ONLY a JSON array: [{{"id": <row id>, "occ": [{{...}}, ...]}}, ...]"""


def _ours(video: str) -> dict[str, str]:
    p = OUTPUT / "ticket" / video / "cache.jsonl"
    out: dict[str, str] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("lang") == "zh-Hans":
                out[r["src"].strip()] = r["txt"]
    return out


def _alignment(video: str, human: Path, work: Path) -> list[str] | None:
    """Reuse any alignment already made for this video's review; else make one."""
    for c in (
        list((OUTPUT / "learn").glob(f"*_{video}/human_zh-Hans.alignment.json"))
        + list((OUTPUT / "learn").glob(f"*{video}_zh-Hans/human_zh-Hans.alignment.json"))
        + list(OUTPUT.glob(f"ab*/{video}/vs_human/human_zh-Hans.alignment.json"))
    ):
        return json.loads(c.read_text(encoding="utf-8"))
    run = OUTPUT / "ticket" / video
    en = (sorted(run.glob("*.en.vtt")) + [run / "en.vtt"])[0]
    base = work / "align" / video
    base.parent.mkdir(parents=True, exist_ok=True)
    if not Path(f"{base}.alignment.json").exists():
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "retime_drifted.py"),
                str(human),
                str(en),
                str(base),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if r.returncode:
            print(f"  align {video} FAILED: {r.stderr.strip()[-200:]}")
            return None
    return json.loads(Path(f"{base}.alignment.json").read_text(encoding="utf-8"))


def reviewed_rows(work: Path, reviewed: str = "*/*zh*Reviewed*.srt") -> list[dict]:
    rows = []
    for human in sorted(INPUT.glob(reviewed)):
        video = human.name.split(".")[0].replace("_Reviewed", "")
        run = OUTPUT / "ticket" / video
        if not run.exists():
            print(f"  skip {video}: no run folder")
            continue
        al = _alignment(video, human, work)
        if al is None:
            continue
        en = (sorted(run.glob("*.en.vtt")) + [run / "en.vtt"])[0]
        sents = merge_en_to_sentences(_read_any(str(en), "en"))
        if len(sents) != len(al):
            print(f"  skip {video}: {len(sents)} sentences vs {len(al)} aligned")
            continue
        ours = _ours(video)
        for s, h in zip(sents, al):
            h = h.replace("|", "").strip()
            if h and WORDS.search(s.text):
                rows.append(
                    {
                        "src": "review",
                        "video": video,
                        "en": s.text,
                        "human": h,
                        "ours": ours.get(s.text.strip(), ""),
                    }
                )
    return rows


def tm_rows() -> list[dict]:
    rows = []
    for line in (ROOT / "tm" / "tm.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("lang") == "zh-Hans" and WORDS.search(r["source"]):
            rows.append(
                {
                    "src": "tm",
                    "video": r.get("video", ""),
                    "en": r["source"],
                    "human": r["target"],
                    "ours": "",
                }
            )
    return rows


def tag(rows: list[dict], batch: int = 15) -> list[dict]:
    inv = CliTranslator()._invoke

    def one(chunk):
        body = "\n".join(
            json.dumps(
                {
                    "id": k,
                    "en": r["en"],
                    "HUMAN": r["human"],
                    **({"OURS": r["ours"]} if r["ours"] else {}),
                },
                ensure_ascii=False,
            )
            for k, r in chunk
        )
        got = _json(inv(TAG.format(rows=body)) or "")
        by = {k: r for k, r in chunk}
        out = []
        for x in got if isinstance(got, list) else []:
            r = by.get(x.get("id")) if isinstance(x, dict) else None
            if r is None:
                continue
            for o in x.get("occ") or []:
                if isinstance(o, dict) and o.get("human") in RENDER:
                    out.append(
                        {**{k: r[k] for k in ("src", "video", "en", "human", "ours")}, "tag": o}
                    )
        return out

    items = list(enumerate(rows))
    chunks = [items[a : a + batch] for a in range(0, len(items), batch)]
    with ThreadPoolExecutor(6) as ex:
        return [o for part in ex.map(one, chunks) for o in part]


def report(occ: list[dict]) -> str:
    rep = ["# How human zh translators render it / this / that / they", ""]
    n_v = len({o["video"] for o in occ})
    rep.append(
        f"{len(occ)} tagged occurrences from {n_v} videos "
        f"({sum(o['src'] == 'review' for o in occ)} vendor-reviewed, {sum(o['src'] == 'tm' for o in occ)} TM)."
    )

    def table(title, key_fn, rend_key):
        rep.extend(
            [
                "",
                f"## {title}",
                "",
                "| role / refers to | n | " + " | ".join(RENDER) + " |",
                "|---|---|" + "---|" * len(RENDER),
            ]
        )
        groups = collections.defaultdict(collections.Counter)
        for o in occ:
            r = o["tag"].get(rend_key)
            if r in RENDER:
                groups[key_fn(o)][r] += 1
        for k, c in sorted(groups.items(), key=lambda kv: -sum(kv[1].values())):
            n = sum(c.values())
            if n < 5:
                continue
            rep.append(
                f"| {k} | {n} | "
                + " | ".join(f"{100 * c[x] / n:.0f}%" if c[x] else "" for x in RENDER)
                + " |"
            )

    def role(o):
        t = o["tag"]
        return f"{t.get('role')} / {t.get('refers_to')}"

    keep = occ
    occ = [o for o in keep if o["src"] == "review"]
    table("HUMAN rendering by role — vendor-reviewed interviews/talks", role, "human")
    occ = [o for o in keep if o["src"] == "tm"]
    table("HUMAN rendering by role — TM (scripted tutorials)", role, "human")
    occ = keep
    with_ours = [o for o in occ if o["ours"] and o["tag"].get("ours") in RENDER]
    occ = with_ours
    table("OUR draft's rendering by role (vendor-reviewed videos)", role, "ours")
    table("HUMAN rendering, same rows (vendor-reviewed videos)", role, "human")
    occ = keep
    # where we and the reviewer differ: the confusion pairs
    pairs = collections.Counter(
        (role(o), o["tag"]["ours"], o["tag"]["human"])
        for o in with_ours
        if o["tag"]["ours"] != o["tag"]["human"]
    )
    rep.extend(["", "## Where our draft and the reviewer differ (ours -> human)", ""])
    for (r, a, b), n in pairs.most_common(15):
        rep.append(f"- {r}: {a} -> {b}: {n}")
        ex = [
            o
            for o in with_ours
            if role(o) == r and o["tag"]["ours"] == a and o["tag"]["human"] == b
        ][:3]
        for o in ex:
            rep.append(
                f"  - {o['video']}: {o['en'][:90]}\n    - ours:  {o['ours']}\n    - human: {o['human']}"
            )
    return "\n".join(rep) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--reviewed",
        default="*/*zh*Reviewed*.srt",
        help="glob (under $DEV_TITLES_INPUT) of the vendor-reviewed zh finals",
    )
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_p = out / "rows.jsonl"
    if rows_p.exists():
        occ = [json.loads(x) for x in rows_p.read_text(encoding="utf-8").splitlines()]
    else:
        rows = reviewed_rows(out, a.reviewed) + tm_rows()
        print(f"{len(rows)} sentence(s) with a candidate word; tagging")
        occ = tag(rows)
        rows_p.write_text(
            "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in occ), encoding="utf-8"
        )
    (out / "REPORT.md").write_text(report(occ), encoding="utf-8")
    print(f"{len(occ)} occurrence(s) -> {out / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
