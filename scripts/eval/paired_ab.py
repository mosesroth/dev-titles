#!/usr/bin/env python3
"""Paired A/B of two FULL pipeline runs of the same video, judged line by line against the human final.

Both runs translate the video fresh (old code, new code), so neither is the draft the reviewer edited:
comparing our delivered draft with the human final is biased toward the draft, because the final IS
that draft plus edits (one video, zh: chrF 85 for the delivered draft vs 61 for a fresh run).

For each EN sentence where the two runs differ, both judges (Claude + GPT, random A/B order) say which
line is better, with the human final as the reference (spot_ab.ref_votes). A line counts as a win or a
loss only when both judges agree. Verdict: one-sided sign test on wins vs losses, per language and
pooled over every --pair given.

Run each arm with scripts/eval/localize_arm.py (isolated: the snapshot's own glossary, partners and
editor examples).

  PYTHONPATH=src python3 scripts/eval/paired_ab.py --out <dir> \
      --pair <video> <old run dir> <new run dir> zh-Hans=<human.srt> ja=<human.srt> [--pair ...]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from ab_vs_human import chrf  # noqa: E402
from spot_ab import ref_votes, sign_p  # noqa: E402

from dev_titles.cli import _qe_second_judge, _read_any  # noqa: E402
from dev_titles.learn import LANG_NAME  # noqa: E402
from dev_titles.resegment import merge_en_to_sentences  # noqa: E402
from dev_titles.translate import CliTranslator  # noqa: E402


def cache_texts(d: Path, lang: str) -> dict[str, str]:
    out = {}
    p = d / "cache.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("lang") == lang:
                out[r["src"].strip()] = r["txt"]
    return out


def _n(s: str) -> str:
    return re.sub(r"[\s，、。．・　|]", "", s or "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs="+", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    judges = [
        j
        for j in (
            CliTranslator()._invoke,
            _qe_second_judge(types.SimpleNamespace(qe_judge=True, single_judge=False)),
        )
        if j
    ]
    rows, tally = [], {}
    for spec in a.pair:
        video, od, nd, *humans = spec
        od, nd = Path(od), Path(nd)
        en_p = next(iter(sorted(nd.glob("*.en.vtt")) + [nd / "en.vtt"]))
        sents = merge_en_to_sentences(_read_any(str(en_p), "en"))
        for h in humans:
            lang, hp = h.split("=", 1)
            base = out / f"human_{video}_{lang}"
            if not Path(f"{base}.alignment.json").exists():
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "retime_drifted.py"),
                        hp,
                        str(en_p),
                        str(base),
                    ],
                    check=True,
                    cwd=ROOT,
                )
            human = json.loads(Path(f"{base}.alignment.json").read_text(encoding="utf-8"))
            old, new = cache_texts(od, lang), cache_texts(nd, lang)
            for k, s in enumerate(sents):
                rows.append(
                    {
                        "video": video,
                        "lang": lang,
                        "i": k,
                        "en": s.text,
                        "human": human[k].replace("|", "").strip(),
                        "old": old.get(s.text.strip(), ""),
                        "new": new.get(s.text.strip(), ""),
                    }
                )

    def judge(item):
        k, r = item
        if not (r["old"] and r["new"] and r["human"]):
            return {**r, "verdict": "n/a"}
        if _n(r["old"]) == _n(r["new"]):
            return {**r, "verdict": "same"}
        v = ref_votes(
            judges, LANG_NAME[r["lang"]], r["en"], r["human"], r["new"], r["old"], random.Random(k)
        )
        key = "win" if v.count("x") == len(v) else "loss" if v.count("y") == len(v) else "tie"
        return {**r, "verdict": key}

    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(judge, enumerate(rows)))
    (out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    rep = ["# Paired A/B — new code vs old code, both fresh, judged against the human final", ""]
    groups = (
        [("pooled", rows)]
        + [
            (lang, [r for r in rows if r["lang"] == lang])
            for lang in sorted({r["lang"] for r in rows})
        ]
        + [
            (f"{v} {lang}", [r for r in rows if r["video"] == v and r["lang"] == lang])
            for v, lang in sorted({(r["video"], r["lang"]) for r in rows})
        ]
    )
    for name, rs in groups:
        c = Counter(r["verdict"] for r in rs)
        p = sign_p(c["win"], c["loss"])
        p_worse = sign_p(c["loss"], c["win"])
        # judge-free second opinion: chrF to the human final, line by line (valid here because both
        # versions are fresh translations; the judges endorse only ~19% of reviewer edits)
        d = [
            chrf(r["new"], r["human"]) - chrf(r["old"], r["human"])
            for r in rs
            if r["old"] and r["new"] and r["human"] and r["verdict"] != "same"
        ]
        up, down = sum(x > 0.5 for x in d), sum(x < -0.5 for x in d)
        p_c = sign_p(up, down)
        tally[name] = {
            **c,
            "p_better": p,
            "p_worse": p_worse,
            "chrf_closer": up,
            "chrf_farther": down,
            "chrf_mean_delta": round(sum(d) / len(d), 2) if d else 0.0,
            "p_chrf_better": p_c,
        }
        rep.append(
            f"- **{name}**: new wins {c['win']}, loses {c['loss']}, ties {c['tie']}, same {c['same']}"
            f" — p(new better) = {p:.3f}, p(new worse) = {p_worse:.3f}; chrF to the human: closer on {up},"
            f" farther on {down} (mean {tally[name]['chrf_mean_delta']:+.1f}), p(closer) = {p_c:.3f}"
        )
    for key, title in (
        ("loss", "Lines where OLD was better (both judges)"),
        ("win", "Lines where NEW was better (both judges)"),
    ):
        rep += ["", f"## {title}", ""]
        for r in [r for r in rows if r["verdict"] == key][:60]:
            rep.append(
                f"- {r['video']} {r['lang']} #{r['i']}: {r['en'][:100]}\n  - old:   {r['old']}\n"
                f"  - new:   {r['new']}\n  - human: {r['human']}"
            )
    (out / "REPORT.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    (out / "result.json").write_text(json.dumps(tally, indent=1), encoding="utf-8")
    print("\n".join(rep[: 2 + len(groups)]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
