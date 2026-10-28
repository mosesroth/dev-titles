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
Verdict: PROMISING when target wins beat losses in a sign test (p < 0.1) and the controls are not
net-worse (sign test the other way not p < 0.1).

Why a sign test and --samples: one translation per line is too noisy. On one video the
BASELINE judged against ITSELF came out 13 win / 10 loss on targets and 3 / 7 on controls (2026-10-26)
— the old "net >= 2" threshold sat inside that noise. --samples k translates every line k times per
version and judges sample s against sample s; a line is a win when its candidate wins more of the k
comparisons than it loses.

--editor: also run the native-editor + reply passes on each row (in its local context, neighbours
from the A/B run's cache) — needed to test an editor-prompt change; validated on one video's
"Thank you for having me": the old reply prompt proposes 也谢谢您的邀请 there, as in the full run.

--target-re: targets are the lines whose English matches the pattern (a rule that touches a kind of
line — pronouns — is tested on every such line, not only where the human was better); controls are the
rest. --cache-dir: translations are reused across invocations (same code root + same rows + sample),
so comparing one candidate against two baselines translates it once.
"""

import argparse
import hashlib
import json
import math
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


def sign_p(wins: int, losses: int) -> float:
    """One-sided exact sign test: P(>= wins of wins+losses | p = 0.5)."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n


def _code_hash(root: Path) -> str:
    h = hashlib.sha1(usedforsecurity=False)
    for p in sorted(list((root / "src").rglob("*.py")) + list((root / "glossary").rglob("*"))):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def _translate(root: Path, jobs, out: Path, cache_dir: Path | None = None, sample: int = 0):
    editor = bool(os.environ.get("SPOT_EDITOR"))
    if cache_dir is not None:
        # keyed on the CODE + GLOSSARY CONTENT, not the root's path: a candidate revised in place
        # (same "." root) must not reuse the first version's translations (agent cycle 2026-10-27)
        key = hashlib.sha1(
            json.dumps(
                [_code_hash(root), sample, editor, os.environ.get("SPOT_MODEL", ""), jobs],
                ensure_ascii=False,
            ).encode(),
            usedforsecurity=False,
        ).hexdigest()[:16]
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = cache_dir / f"{root.name}_{key}.json"
        if cp.exists():
            return json.loads(cp.read_text(encoding="utf-8"))
        res = _translate(root, jobs, out)
        cp.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        return res
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
    ap.add_argument("--samples", type=int, default=1, help="translations per line per version")
    ap.add_argument("--target-re", help="targets = lines whose English matches (case-insensitive)")
    ap.add_argument("--max-targets", type=int, default=0, help="per run folder (random sample)")
    ap.add_argument("--cache-dir", help="reuse translations across invocations")
    ap.add_argument("--baseline-model", default="", help="pin the baseline's translation model")
    ap.add_argument("--candidate-model", default="", help="pin the candidate's translation model")
    ap.add_argument(
        "--editor",
        action="store_true",
        help="also run the native-editor + reply passes on each row (tests editor-prompt changes)",
    )
    a = ap.parse_args(argv)
    if a.editor:
        os.environ["SPOT_EDITOR"] = "1"  # read by translate_rows.py (inherited env)
    rng = random.Random(a.seed)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    import re

    tre = re.compile(a.target_re, re.I) if a.target_re else None
    targets, pool = [], []
    for d in a.runs:
        rows = json.loads((Path(d) / "vs_human" / "ab_rows.json").read_text(encoding="utf-8"))
        run_t = []
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
                if not job["human"]:
                    continue
                hit = tre.search(r["en"]) if tre else r.get("verdict") == "human better"
                (run_t if hit else pool).append(job)
        if a.max_targets and len(run_t) > a.max_targets:
            run_t = rng.sample(run_t, a.max_targets)
        targets += run_t
    controls = rng.sample(pool, min(a.controls * max(1, len(a.runs)), len(pool)))
    jobs = targets + controls
    print(f"{len(targets)} target row(s), {len(controls)} control row(s)")
    cdir = Path(a.cache_dir).resolve() if a.cache_dir else None
    bases, cands = [], []

    def _model(m):
        if m:
            os.environ["SPOT_MODEL"] = m
        else:
            os.environ.pop("SPOT_MODEL", None)

    for s_ in range(a.samples):
        _model(a.baseline_model)
        bases.append(
            _translate(Path(a.baseline).resolve(), jobs, out / f"baseline{s_}.json", cdir, s_)
        )
        _model(a.candidate_model)
        cands.append(
            _translate(Path(a.candidate).resolve(), jobs, out / f"candidate{s_}.json", cdir, s_)
        )
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
        k, j = item
        bucket = "targets" if k < len(targets) else "controls"
        bs = [bases[s_][k] for s_ in range(a.samples)]
        cs = [cands[s_][k] for s_ in range(a.samples)]
        if all("".join(b.split()) == "".join(c.split()) for b, c in zip(bs, cs)):
            return bucket, "same", None
        xs = ys = 0
        for s_, (b, c) in enumerate(zip(bs, cs)):
            if "".join(b.split()) == "".join(c.split()):
                continue
            v = ref_votes(
                judges,
                LANG_NAME[j["lang"]],
                j["en"],
                j["human"],
                c,
                b,
                random.Random(a.seed + 1000 * s_ + k),
            )
            xs += v.count("x") == len(v)
            ys += v.count("y") == len(v)
        key = "win" if xs > ys else "loss" if ys > xs else "tie"
        return (
            bucket,
            key,
            {**j, "bucket": bucket, "baseline": bs[0], "candidate": cs[0], "verdict": key},
        )

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(8) as ex:  # independent judge calls
        for bucket, key, d in ex.map(judge, enumerate(jobs)):
            res[bucket][key] += 1
            if d:
                detail.append(d)
    t, c_ = res["targets"], res["controls"]
    p_t = sign_p(t["win"], t["loss"])
    p_c = sign_p(c_["loss"], c_["win"])  # evidence the controls got WORSE
    verdict = "PROMISING" if p_t < 0.1 and p_c >= 0.1 else "NOT PROMISING"
    rep = [
        f"# Spot A/B — {verdict}",
        "",
        f"- samples per line: {a.samples}",
        f"- targets: {t} (sign test p = {p_t:.3f})",
        f"- controls: {c_} (p controls worse = {p_c:.3f})",
        "",
    ]
    for d in detail:
        rep.append(
            f"- [{d['bucket']}/{d['verdict']}] {Path(d['dir']).name} {d['lang']} #{d['i']}: {d['en'][:80]}\n"
            f"  - baseline:  {d['baseline']}\n  - candidate: {d['candidate']}\n  - human:     {d['human']}"
        )
    (out / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")
    (out / "result.json").write_text(
        json.dumps(
            {"verdict": verdict, "p_targets": p_t, "p_controls_worse": p_c, **res}, indent=1
        ),
        encoding="utf-8",
    )
    print(f"{verdict}: targets {t}, controls {c_} -> {out / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
