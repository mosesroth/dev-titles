#!/usr/bin/env python3
"""Spot A/B of a harness change: minutes, not the 10 h of a full panel.

Picks, from finished A/B run folders (vs_human/ab_rows.json), the TARGET rows (where the human final was
better: what the change is meant to fix) and CONTROL rows (a random sample of the rest: what it must not
break). Translates each with the BASELINE code + glossary and with the CANDIDATE code + glossary
(scripts/eval/translate_rows.py, one process per version), then blind-judges candidate vs baseline
(Claude + GPT, random A/B order) WITH the human final as the reference — without it, judges grade two
plausible variants as a coin flip (a first test: 3 win / 3 loss / 8 tie where a reference-based look
showed 9 of 11 moving toward the reviewer). Reports wins / losses on targets and harm on controls.

  PYTHONPATH=src python3 scripts/eval/spot_ab.py --runs output/ab/* \
      --baseline <repo-or-snapshot root> --candidate <repo root> [--lang ja] [--controls 10] --out <dir>

A root is a folder holding src/ and glossary/glossary.yaml (a git worktree, or the repo itself).
Verdict: PROMISING when target wins exceed losses by >= 2 and the controls are not net-worse beyond
noise (more losses than wins by at most one line or 10% of the controls).
"""

import argparse
import json
import os
import random
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dev_titles.cli import _qe_second_judge  # noqa: E402
from dev_titles.learn import LANG_NAME, _json  # noqa: E402

REF = """Two {L} subtitles (A, B) for the same English line of a developer video, and the REFERENCE: the
line a native subtitle reviewer approved. Which of A and B is the better subtitle, judged with the
reference as the guide to what a native reviewer wants (meaning, register, naturalness, terminology)?
It does not have to copy the reference word for word; ignore line breaks and spacing.
ENGLISH: {en}
REFERENCE: {ref}
A: {a}
B: {b}
Answer ONLY JSON: {{"better": "A"|"B"|"equal"}}"""


def ref_votes(judges, L, en, ref, x, y, rng):
    """Blind, reference-guided votes for x vs y ('x' | 'y' | 'equal'), random A/B order per judge."""
    out = []
    for j in judges:
        flip = rng.random() < 0.5
        a, b = (y, x) if flip else (x, y)
        d = _json(j(REF.format(L=L, en=en, ref=ref, a=a, b=b)) or "", "{")
        out.append(
            {"A": "y" if flip else "x", "B": "x" if flip else "y"}.get(d.get("better"), "equal")
        )
    return out


from dev_titles.translate import CliTranslator  # noqa: E402


def _translate(root: Path, jobs, out: Path):
    jp, op = out.with_suffix(".jobs.json"), out
    jp.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval" / "translate_rows.py"),
            str(jp),
            str(root / "glossary" / "glossary.yaml"),
            str(op),
        ],
        check=True,
        env=env,
        cwd=root,
    )
    return json.loads(op.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--lang", choices=sorted(LANG_NAME))
    ap.add_argument("--controls", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    targets, pool = [], []
    for d in a.runs:
        rows = json.loads((Path(d) / "vs_human" / "ab_rows.json").read_text(encoding="utf-8"))
        for lang, rs in rows.items():
            if a.lang and lang != a.lang:
                continue
            for r in rs:
                job = {
                    "dir": str(Path(d).resolve()),
                    "lang": lang,
                    "i": r["i"],
                    "en": r["en"],
                    "human": r.get("human", ""),
                }
                (targets if r.get("verdict") == "human better" else pool).append(job)
    controls = rng.sample(pool, min(a.controls * max(1, len(a.runs)), len(pool)))
    jobs = targets + controls
    print(f"{len(targets)} target row(s), {len(controls)} control row(s)")
    base = _translate(Path(a.baseline).resolve(), jobs, out / "baseline.json")
    cand = _translate(Path(a.candidate).resolve(), jobs, out / "candidate.json")
    judges = [
        j
        for j in (
            CliTranslator()._invoke,
            _qe_second_judge(types.SimpleNamespace(qe_judge=True, single_judge=False)),
        )
        if j
    ]
    res = {
        "targets": {"win": 0, "loss": 0, "tie": 0, "same": 0},
        "controls": {"win": 0, "loss": 0, "tie": 0, "same": 0},
    }
    detail = []

    def judge(item):
        k, (j, b, c) = item
        bucket = "targets" if k < len(targets) else "controls"
        if "".join(b.split()) == "".join(c.split()) or not j.get("human"):
            return bucket, "same", None
        v = ref_votes(
            judges, LANG_NAME[j["lang"]], j["en"], j["human"], c, b, random.Random(a.seed + k)
        )
        key = "win" if v.count("x") == len(v) else "loss" if v.count("y") == len(v) else "tie"
        return bucket, key, {**j, "bucket": bucket, "baseline": b, "candidate": c, "verdict": key}

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(8) as ex:  # independent judge calls
        for bucket, key, d in ex.map(judge, enumerate(zip(jobs, base, cand))):
            res[bucket][key] += 1
            if d:
                detail.append(d)
    t, c_ = res["targets"], res["controls"]
    # net effect: targets clearly better; controls not worse beyond noise (one line or 10%)
    n_ctrl = sum(c_.values())
    ok_ctrl = c_["loss"] - c_["win"] <= max(1, round(0.1 * n_ctrl))
    verdict = "PROMISING" if t["win"] - t["loss"] >= 2 and ok_ctrl else "NOT PROMISING"
    rep = [
        f"# Spot A/B — {verdict}",
        "",
        f"- targets (human was better): {t}",
        f"- controls: {c_}",
        "",
    ]
    for d in detail:
        rep.append(
            f"- [{d['bucket']}/{d['verdict']}] {Path(d['dir']).name} {d['lang']} #{d['i']}: {d['en'][:80]}\n"
            f"  - baseline:  {d['baseline']}\n  - candidate: {d['candidate']}\n  - human:     {d['human']}"
        )
    (out / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")
    (out / "result.json").write_text(
        json.dumps({"verdict": verdict, **res}, indent=1), encoding="utf-8"
    )
    print(f"{verdict}: targets {t}, controls {c_} -> {out / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
