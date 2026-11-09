#!/usr/bin/env python3
"""Check a vendor's reviewed subtitles for errors the review itself introduced, and fix only the ones
that hold up in a blind check. For deliveries we can't send back to the vendor.

Why: two zh vendor reviews were ~90% real improvements, but ~4% of edits broke a line
that was right: a technical term swapped for its everyday sense (server function -> 服务器功能,
rebuild -> 重构), casual English read literally (Oh my God -> 太好了), or debris from a rewrite
(a doubled 您, a stray 、 inside "React Native Web").

Steps, per video:
  1. structure: cue count / timing vs. the version the vendor reviewed (ours)
  2. protected terms (no model calls): a glossary avoid-variant in the reviewed text, or a DNT
     identifier / hard-term rendering that was in our line but is gone from theirs
  3. proofread (Claude): clear errors only, with the English and our line alongside
  4. blind check: for every candidate, GPT and Claude each pick between the vendor's line and the
     fix, in both A/B orders, without knowing which is which. A fix is applied only if it wins
     >= 3 of the 4 votes and the vendor's line wins none ("both fine" counts for neither).
  5. apply, then re-wrap layout (scripts/rewrap_reviewed.py)

Writes <out>/REPORT.md, <out>/_last_fixes_applied.txt, <out>/check_state.json and the fixed files.

  PYTHONPATH=src python3 scripts/check_vendor_review.py input/vendor_batch --lang zh-Hans \
      --out output/vendor_batch_fixed [--ours-dir output/final] [--dry-run]
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dev_titles import llm
from dev_titles.cli import _read_any
from dev_titles.formats import write_srt
from dev_titles.glossary import load
from dev_titles.model import Subtitle
from dev_titles.validate.sync import sync_report

LANG_NAME = {"zh-Hans": "Simplified Chinese", "ja": "Japanese"}

PROOF = """You proofread FINAL {L} subtitles (already reviewed by a native translator) of a developer video,
with the English alongside and our earlier MACHINE line (the version the reviewer edited). Report ONLY clear
errors a careful proofreader must fix: typos / wrong characters, garbled or scrambled text, missing or
duplicated characters/words, wrong numbers, misspelled product/company/person names, a technical term
rendered in its everyday (non-software) sense, a line that contradicts the English, or a reviewer edit
that introduced a meaning error the MACHINE line did not have.
Do NOT report style, wording preferences, condensation, punctuation style or line breaks.
Lines (id | English nearby | MACHINE | REVIEWED):
{rows}
Return ONLY a JSON array: [{{"id": n, "error": "<what is wrong, in English>", "fix": "<corrected REVIEWED line>"}}]; [] if none."""

BLIND = """Two candidate {L} subtitles for the same moment of a developer video. Which one is a CORRECT
translation of what is said? Judge meaning and terminology against the English; ignore punctuation,
spacing and line breaks. If both are acceptable, say "both".
English (the full surrounding speech): {en}
Previous subtitle: {prev}
Next subtitle: {nxt}
A: {a}
B: {b}
Answer ONLY JSON: {{"choice": "A"|"B"|"both", "why": "<max 15 words>"}}"""


def _norm(s):
    return re.sub(r"[\s　，、。]", "", s or "")


def _flat(c):
    # a name broken across the two lines ("React Native" / "DevTools") keeps its space: joined
    # without one, the proofreader reported "React NativeDevTools" / "forWindows" as misspellings
    return re.sub(r"(?<=[A-Za-z0-9])\n(?=[A-Za-z0-9])", " ", c.text).replace("\n", "")


_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’+./\-]*(?: [A-Za-z0-9][A-Za-z0-9'’+./\-]*)*")


def _key(s):
    # only case, spacing, apostrophes, hyphens and "Plus" for "+" may differ: "Expo Dev Awards" is
    # not a misspelling of expo.dev/awards
    return re.sub(r"[\s'’\-]", "", re.sub(r"\s+plus\b", "+", s.lower()))


def rule_fixes(rev, en, lang, applied_variants):
    """Errors a meaning judge can't see, fixed without a vote (the blind check asks "which is a correct
    translation?" and answers "both" for them). On one batch all of these survived it:
      - a name spelled differently from the English: "Wits End" for Wit's End, "Nimbus Plus" for Nimbus+;
      - an enumeration that lost its separator: "Expo Expo的工具" for "Expo, Expo's Tools";
      - a glossary avoid-variant left on some lines of a track after the same fix won the vote on
        others (挂钩 -> Hook fixed 3x, one 挂钩 left) — unless the vendor's form won a vote there.
    `applied_variants` = {(bad, good)} of glossary fixes already applied in this track.
    Returns {cue_no: (reason, fixed line)}."""
    out = {}
    for k, c in enumerate(rev, 1):
        txt = _flat(c)
        ew = " ".join(e.text for e in en if e.start < c.end + 1.5 and e.end > c.start - 1.5).split()
        # English names: 1-4 word spans that don't cross punctuation ("Apple TV, tvOS" is two names)
        spans: list[list[str]] = [[]]
        for w in ew:
            t = w.strip(',.?!;:"()—–')
            if t:
                spans[-1].append(t)
            if not t or w[-1] in ",.?!;:":
                spans.append([])
        # name-like only: an apostrophe, +, / or a capital after the first letter (Wit's End, Nimbus+,
        # CI/CD, SAM); "Hi there" or a sentence-start "Hooks" is not a name
        names = sorted(
            {
                g
                for sp in spans
                for n in (4, 3, 2, 1)
                for i in range(len(sp) - n + 1)
                for g in [" ".join(sp[i : i + n])]
                if re.search(r"['’+/]|.[A-Z]", g)
            },
            key=len,
            reverse=True,
        )
        why, new = [], txt
        for run in _RUN.findall(txt):
            words = run.rstrip(".,").split()
            for n in range(len(words), 0, -1):
                for i in range(len(words) - n + 1):
                    sub = " ".join(words[i : i + n])
                    hit = next(
                        (
                            g
                            for g in names
                            if g != sub and _key(g) == _key(sub) and len(_key(g)) >= 3
                        ),
                        None,
                    )
                    if (
                        hit
                        and sub in new
                        and not re.search(rf"(?<!\w){re.escape(sub)}(?!\w)", " ".join(ew))
                    ):
                        new = new.replace(sub, hit)
                        why.append(f"name spelled as in the English: '{sub}' -> '{hit}'")
        for m in re.finditer(r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9]+) \1(?![A-Za-z0-9])", new):
            w = m.group(1)
            if re.search(rf"\b{re.escape(w)}, {re.escape(w)}", " ".join(ew)):
                new = new.replace(m.group(0), f"{w}、{w}", 1)
                why.append(f"separator restored: '{w} {w}' -> '{w}、{w}' (English: '{w}, {w}')")
        for bad, good in applied_variants:
            if not re.fullmatch(r"[\x20-\x7e]+", bad):
                if bad in new:
                    new = new.replace(bad, good)
                    why.append(
                        f"'{bad}' -> {good}: the same glossary fix won the vote in this track"
                    )
                continue
            # a Latin variant: whole words only, and never where the English has it verbatim — the
            # command `npm run android` is not a misspelling of Android
            pat = rf"(?<![A-Za-z0-9]){re.escape(bad)}(?![A-Za-z0-9])"
            if re.search(pat, new) and not re.search(pat, " ".join(ew)):
                new = re.sub(pat, good, new)
                why.append(f"'{bad}' -> {good}: glossary spelling of the name")
        if why and new != txt:
            out[k] = ("; ".join(why), new)
    return out


def _json(s, kind="["):
    m = re.search(r"\[.*\]" if kind == "[" else r"\{.*\}", s or "", re.S)
    try:
        return json.loads(m.group(0)) if m else ([] if kind == "[" else {})
    except Exception:
        return [] if kind == "[" else {}


def _variant(error):
    """("挂钩", "Hook") from a protected_flags avoid-variant message, else (None, None)."""
    m = re.match(r"'(.+?)' is a glossary avoid-variant of '.+?' -> (.+)$", error or "")
    return (m.group(1), m.group(2)) if m else (None, None)


def video_name(path: Path) -> str:
    """'MyVideo_Reviewed.zh-Hans.srt' / 'MyVideo.ja_Reviewed_20260101.srt' -> the video."""
    return re.split(r"[._]", path.name, maxsplit=1)[0]


def _window(cues, start, end, pad=0.3):
    return "".join(_flat(c) for c in cues if c.start < end + pad and c.end > start - pad)


def protected_flags(rev, ours, lang, gl):
    """Deterministic checks (no model calls). Returns [(cue_no, reason, proposed_fix_line or None)]."""
    out = []
    terms = [t for t in gl.terms if t.translations.get(lang)]
    for k, c in enumerate(rev, 1):
        txt, mine = _flat(c), _window(ours, c.start, c.end)
        for t in terms:
            for bad in (getattr(t, "avoid", {}) or {}).get(lang, []) or []:
                if bad and bad in txt and bad not in mine:
                    out.append(
                        (
                            k,
                            f"'{bad}' is a glossary avoid-variant of '{t.source}' -> "
                            f"{t.translations[lang]}",
                            txt.replace(bad, t.translations[lang]),
                        )
                    )
            want = t.translations[lang]
            if (
                t.enforce == "hard"
                and want in mine
                and want not in txt
                and _norm(want) not in _norm(_window(rev, c.start, c.end))
            ):
                out.append(
                    (
                        k,
                        f"hard term '{t.source}' -> {want} was in our line, gone from the review",
                        None,
                    )
                )
        for ident in gl.identifiers:
            if (
                re.search(r"[A-Za-z]", ident)
                and ident in mine
                and ident not in _window(rev, c.start, c.end)
            ):
                out.append((k, f"DNT name '{ident}' was in our line, gone from the review", None))
    return out


def proofread(rev, ours, en, L, invoke, workers=8):
    rows = []
    for k, c in enumerate(rev, 1):
        e = " ".join(
            " ".join(q.lines) for q in en if q.start < c.end + 0.3 and q.end > c.start - 0.3
        )
        rows.append(f"{k} | {e[:220]} | {_window(ours, c.start, c.end, 0.0)} | {_flat(c)}")
    chunks = [rows[i : i + 50] for i in range(0, len(rows), 50)]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda ch: _json(invoke(PROOF.format(L=L, rows="\n".join(ch)))), chunks))
    return [f for r in res for f in r if isinstance(f, dict) and f.get("fix")]


def blind_votes(cand, rev, en, L, judges, rng):
    k, fix = cand["id"], cand["fix"]
    c = rev[k - 1]
    e = " ".join(" ".join(q.lines) for q in en if q.start < c.end + 2.5 and q.end > c.start - 2.5)
    prev = _flat(rev[k - 2]) if k > 1 else ""
    nxt = _flat(rev[k]) if k < len(rev) else ""
    votes = []
    for j in judges:
        for flip in (False, True):
            a, b = (fix, _flat(c)) if flip else (_flat(c), fix)
            ch = _json(j(BLIND.format(L=L, en=e, prev=prev, nxt=nxt, a=a, b=b)), "{").get(
                "choice", "?"
            )
            votes.append(
                {"A": "fix" if flip else "vendor", "B": "vendor" if flip else "fix"}.get(ch, ch)
            )
    return votes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reviewed_dir")
    ap.add_argument("--lang", required=True, choices=sorted(LANG_NAME))
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--ours-dir",
        default="output/final",
        help="the version the vendor reviewed: <video>.<lang>.srt",
    )
    ap.add_argument("--ticket-dir", default="output/ticket", help="<video>/en.vtt for the English")
    ap.add_argument(
        "--glossary",
        default=str(Path(__file__).resolve().parents[1] / "glossary" / "glossary.yaml"),
    )
    ap.add_argument(
        "--reuse",
        action="store_true",
        help="reuse the proofread findings and votes in <out>/check_state.json (re-run after a "
        "checker fix without new model calls for unchanged candidates)",
    )
    ap.add_argument("--no-learn", action="store_true", help="don't start learn_review_batch.py")
    ap.add_argument("--dry-run", action="store_true", help="report only; write no subtitle files")
    a = ap.parse_args()
    lang, L, out = a.lang, LANG_NAME[a.lang], Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gl = load(a.glossary)
    claude = llm.invoke()
    second, _why = llm.second_judge()
    judges = (
        (second, claude) if second else (claude,)
    )  # different-family judge first when available
    rng = random.Random(0)
    prior = (
        json.loads((out / "check_state.json").read_text(encoding="utf-8"))
        if a.reuse and (out / "check_state.json").exists()
        else {}
    )
    state, report, applied = {}, [f"# Vendor review check ({lang}): {a.reviewed_dir}", ""], []
    stage = out / "_stage"
    stage.mkdir(exist_ok=True)
    for f in sorted(Path(a.reviewed_dir).glob("*.srt")):
        v = video_name(f)
        rev = _read_any(str(f), lang).cues
        ours_p = Path(a.ours_dir) / f"{v}.{lang}.srt"
        en_p = Path(a.ticket_dir) / v / "en.vtt"
        if not ours_p.exists() or not en_p.exists():
            report.append(f"## {v}\n\nSKIPPED: missing {ours_p if not ours_p.exists() else en_p}\n")
            continue
        # keep the exact draft this batch reviewed: it is the ground truth for the judge calibration and
        # the vendor-edit KPI (scripts/eval/review_pairs.py), and FINAL_SUBTITLES gets replaced
        keep = Path(a.ours_dir).parent / "review_drafts" / Path(a.reviewed_dir).name / ours_p.name
        if not keep.exists():
            keep.parent.mkdir(parents=True, exist_ok=True)
            keep.write_bytes(ours_p.read_bytes())
        ours, en = _read_any(str(ours_p), lang).cues, _read_any(str(en_p), "en").cues
        same = len(ours) == len(rev) and all(
            abs(x.start - y.start) < 0.05 for x, y in zip(ours, rev)
        )
        changed = sum(_norm(x.text) != _norm(y.text) for x, y in zip(ours, rev)) if same else None
        flags = protected_flags(rev, ours, lang, gl)
        cands: dict = {}
        for k, why, fix in flags:
            if fix:
                cands.setdefault(k, {"id": k, "error": why, "fix": fix, "source": "glossary"})
        old = [c for c in (prior.get(v) or {}).get("candidates", []) if c.get("source") != "rule"]
        found = (
            [c for c in old if c.get("source") == "proofread"]
            if v in prior
            else proofread(rev, ours, en, L, claude)
        )
        for p in found:
            k = p.get("id")
            if not isinstance(k, int) or not 1 <= k <= len(rev) or k in cands:
                continue
            # a fix must be an edit of the vendor's line, not a rewrite
            if difflib.SequenceMatcher(None, _flat(rev[k - 1]), p["fix"]).ratio() < 0.5:
                continue
            cands[k] = {"id": k, "error": p["error"], "fix": p["fix"], "source": "proofread"}
        seen = {(c["id"], c["fix"]): c.get("votes") for c in old if c.get("votes")}
        todo = {k: c for k, c in cands.items() if (k, c["fix"]) not in seen}
        with ThreadPoolExecutor(8) as ex:
            votes = dict(
                zip(
                    todo,
                    ex.map(lambda c: blind_votes(c, rev, en, L, judges, rng), todo.values()),
                )
            )
        votes.update({k: seen[(k, c["fix"])] for k, c in cands.items() if k not in todo})
        ok = []
        for k, c in sorted(cands.items()):
            vs = votes[k]
            c["votes"] = vs
            # every vote but at most one for the fix (3 of 4 with two judges; 2 of 2 with one), none for the vendor
            c["applied"] = (
                vs.count("fix") >= len(vs) - (1 if len(vs) >= 4 else 0) and vs.count("vendor") == 0
            )
            c["kind"] = (
                "revert to our original"
                if _norm(c["fix"]) in _norm(_window(ours, rev[k - 1].start - 3, rev[k - 1].end + 3))
                else "new fix"
            )
            if c["applied"]:
                ok.append(c)
                rev[k - 1].lines = [c["fix"]]
        bad_good = {
            (b, g) for c in ok if c["source"] == "glossary" for b, g in [_variant(c["error"])] if b
        }
        # the vendor's form won a vote on these lines: context decides there (令牌 for an auth token)
        vendor_won = {k for k, c in cands.items() if "vendor" in c["votes"]}
        # a Latin avoid-variant of a Latin name ("Nimbus Plus" for Nimbus+) is a misspelling anywhere
        bad_good |= {
            (b, t.translations[lang])
            for t in gl.terms
            if re.fullmatch(r"[\x20-\x7e]+", t.translations.get(lang) or "")
            for b in (getattr(t, "avoid", {}) or {}).get(lang, []) or []
            if b and re.fullmatch(r"[\x20-\x7e]+", b)
        }
        for k, (why, fix) in rule_fixes(rev, en, lang, bad_good).items():
            if k in vendor_won and not why.startswith("name") and "separator" not in why:
                continue
            c = cands.get(k) or {"id": k, "source": "rule"}
            c.update(error=why, fix=fix, votes=["rule"], applied=True, kind="rule (no vote)")
            cands[k] = c
            if c not in ok:
                ok.append(c)
            rev[k - 1].lines = [fix]
        sync = sync_report(rev, getattr(en, "cues", en))
        state[v] = {
            "sync": sync,
            "cues": len(rev),
            "timing_same_as_ours": same,
            "changed_by_vendor": changed,
            "protected_flags": flags,
            "candidates": list(cands.values()),
        }
        (stage / f"{v}.{lang}.srt").write_text(write_srt(Subtitle(lang, rev)), encoding="utf-8")
        pct = f"{len(ok)}/{changed} ({len(ok) / changed:.1%})" if changed else str(len(ok))
        report += [
            f"## {v}",
            "",
            f"- cues {len(rev)}; same timing as ours: {same}; vendor changed {changed}",
            f"- sync with the speech: median {sync['median']} s, {sync['far_share']} of anchors >3 s off"
            + (" — OUT OF SYNC: re-time with scripts/retime_drifted.py" if sync["drifted"] else ""),
            f"- protected-term flags: {len(flags)}; candidates: {len(cands)}; fixes applied: {pct}",
            "",
        ]
        for k, c in sorted(cands.items()):
            mark = "APPLIED" if c["applied"] else "kept vendor"
            report.append(
                f"- #{k} [{mark}, {c['kind']}, votes {'/'.join(c['votes'])}] {c['error']}"
            )
        for k, why, _ in flags:
            if k not in cands:
                report.append(f"- #{k} [flag only] {why}")
        report.append("")
        applied += [f"- {v} #{c['id']} ({c['kind']}): {c['error']}" for c in ok]
    (out / "check_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "_last_fixes_applied.txt").write_text(
        f"Applied on top of the vendor's reviewed files ({Path(a.reviewed_dir).name}); each fix won a blind "
        "two-judge check (Claude + GPT, both A/B orders):\n"
        + "\n".join(applied)
        + "\nLayout: cues over the line limit re-wrapped (scripts/rewrap_reviewed.py).\n",
        encoding="utf-8",
    )
    if not a.dry_run:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("rewrap_reviewed.py")),
                str(stage),
                "--out",
                str(out),
                "--no-learn",
            ],
            check=True,
        )
    print((out / "REPORT.md").read_text(encoding="utf-8"))
    if not a.dry_run and not a.no_learn:
        # learn from the review: automatic, never a separate step (learn_review_batch.py)
        sys.path.insert(0, str(Path(__file__).parent))
        from learn_review_batch import launch_detached

        launch_detached(a.reviewed_dir)


if __name__ == "__main__":
    main()
