"""Review-and-fix: the last pass a person used to do by hand on REVIEW.md.

After translation, QE-repair and the editor, three kinds of real problems still survived to the
deliverable (one video), each visible in the review but fixed by nobody:
  - one-judge findings that were right: an added "with the remote" (遥控器 / リモコン) the speaker
    never said, "build faster" read as faster compiling (加快构建速度), two clauses run together;
  - one English term rendered two ways in a track ("tiles": 磁贴 4x / 图块 6x);
  - the two languages disagreeing on the meaning of one line ("Take the jump from 0.72 to 0.83":
    zh "taking it as an example", ja "let's look at the migration").

This pass collects those candidates — the QE findings left after repair, a whole-track term
consistency check, and a zh-vs-ja meaning check — and for each line asks the model for a minimal
correction or NO_CHANGE (a finding can be wrong: a judge misread "Take the jump" as a command). A
correction is applied only if it passes the same gates as every repair (no Latin term or number
lost, nothing omitted) AND the judges prefer it: no judge for the old line, or, on a 1-1 split, at
least 3 of 4 votes after both judges vote again. Lines a person locked are never touched.
"""

from __future__ import annotations

import json
import random
import re

from .localize import coverage_missing, model_line

TERMS = """Below is a {L} subtitle track of a developer video, one line per English sentence (id | English | {L}).
List the English TERMS (technical words or recurring nouns) that this track translates with DIFFERENT {L}
words in different lines, where one rendering should be used throughout. Ignore pronouns, ordinary
variation that reads naturally, and terms the track already renders consistently.
{rows}
Return ONLY JSON: [{{"term": "<English term>", "use": "<the {L} rendering to use everywhere>",
"ids": [<ids of the lines that use another rendering>]}}]  ([] if none)"""

CROSS = """Each row gives an English subtitle sentence from a developer video and its Simplified Chinese and
Japanese translations (id | English | zh | ja). List the rows where the two translations convey a
DIFFERENT MEANING of the English (not wording or style), and say which one is faithful to the English.
{rows}
Return ONLY JSON: [{{"id": <id>, "wrong": "zh" | "ja", "why": "<max 20 words>"}}]  ([] if none)"""

FIX = """You are a senior native {L} subtitle editor for a developer video. A reviewer raised the issue(s) below
about one subtitle line. Decide whether each issue is REAL. If at least one is, rewrite the line with
the smallest change that fixes it, keeping everything else (wording, Latin terms, numbers). If none
is real (the reviewer misread the English), answer NO_CHANGE.
Before: {prev}
ENGLISH: {en}{meaning}
After: {nxt}
CURRENT {L}: {cur}
ISSUES:
{issues}
Output ONLY the corrected {L} line, or NO_CHANGE."""

MEANING = """
INTENDED MEANING (verified by the video's operator or two models): {note}. A line that conveys it is
correct; an issue that asks for a different reading of the English is NOT real."""


def _json(text: str):
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def _latin(s: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9.+#_-]*", s or "")}


def _nums(s: str) -> list[str]:
    return sorted(re.findall(r"\d+(?:\.\d+)?", s or ""))


def candidates(lang, L, records, suspects, other=None, invoke=None) -> dict[int, list[str]]:
    """{record index: [issue text, ...]} from the leftover QE findings, the term-consistency check
    and (with `other` = {"lang": records}) the cross-language meaning check."""
    by_i = {r.i: r for r in records}
    out: dict[int, list[str]] = {}

    def add(i, why):
        if i in by_i and not by_i[i].locked and why not in out.setdefault(i, []):
            out[i].append(why)

    for s in suspects or []:
        if str(s.get("kind", "")).startswith("mqm-") and s.get("i") is not None:
            add(s["i"], str(s.get("detail", ""))[:300])
    if invoke is None:
        return out
    rows = "\n".join(f"{r.i} | {r.src} | {r.txt}" for r in sorted(records, key=lambda r: r.i))
    try:
        for t in _json(invoke(TERMS.format(L=L, rows=rows))):
            if isinstance(t, dict) and t.get("use"):
                ids = [i for i in t.get("ids") or [] if isinstance(i, int)]
                # unify toward the MAJORITY rendering only: in a test the model picked the minority
                # form (磁贴 3x over 图块 7x) and the track ended up flipping one line the other way
                if sum(t["use"] in r.txt for r in records) < len(ids):
                    continue
                for i in ids:
                    if isinstance(i, int):
                        add(
                            i,
                            f"term consistency: render '{t.get('term')}' as {t['use']} (used elsewhere in this track)",
                        )
    except Exception:  # never sink a run on a review pass
        pass
    if other:
        ((olang, orecs),) = other.items()
        ob = {r.i: r for r in orecs}
        pairs = [(r, ob[r.i]) for r in sorted(records, key=lambda r: r.i) if r.i in ob]
        zh_ja = (lambda a, b: (a, b)) if lang == "zh-Hans" else (lambda a, b: (b, a))
        rows = "\n".join(
            f"{r.i} | {r.src} | {zh_ja(r.txt, o.txt)[0]} | {zh_ja(r.txt, o.txt)[1]}"
            for r, o in pairs
        )
        try:
            for x in _json(invoke(CROSS.format(rows=rows))):
                if isinstance(x, dict) and isinstance(x.get("id"), int):
                    if (x.get("wrong") == "zh") == (lang == "zh-Hans"):
                        add(
                            x["id"],
                            f"meaning differs from the {olang} translation: {x.get('why', '')}",
                        )
        except Exception:
            pass
    return out


def review_and_fix(
    lang, L, records, suspects, invoke, judges, other=None, logger=print, seed=7, notes=None
):
    """Apply gated corrections in place. Returns [{i, before, after, issues, status}].
    `notes`: the video's verified meaning notes ({sentence index: [{phrase, meaning}]}, operator
    answers included). The fix and its judges see a line's note: without it a line that followed
    the operator's answer ('scratch the surface' = 'start looking into it') was 'fixed' back to a
    literal reading ('a brief look')."""
    from .learn import pairwise

    cands = candidates(lang, L, records, suspects, other, invoke)
    by_i = {r.i: r for r in records}
    rng = random.Random(seed)
    log = []
    for i, issues in sorted(cands.items()):
        r = by_i[i]
        prev = by_i[i - 1].src if i - 1 in by_i else ""
        nxt = by_i[i + 1].src if i + 1 in by_i else ""
        items = (notes or {}).get(i) or (notes or {}).get(str(i)) or []
        note = "; ".join(f'"{n["phrase"]}" = {n["meaning"]}' for n in items)
        meaning = MEANING.format(note=note) if note else ""
        try:
            raw = invoke(
                FIX.format(
                    L=L,
                    prev=prev,
                    en=r.src,
                    meaning=meaning,
                    nxt=nxt,
                    cur=r.txt,
                    issues="\n".join(f"- {x}" for x in issues),
                )
            )
        except Exception:
            continue
        raw = (raw or "").strip()
        entry = {"i": i, "en": r.src, "before": r.txt, "issues": issues}
        if raw.upper().startswith("NO_CHANGE"):
            log.append({**entry, "status": "not real"})
            continue
        new = model_line(raw, r.txt)
        if re.sub(r"\s", "", new) == re.sub(r"\s", "", r.txt):
            log.append({**entry, "status": "unchanged"})
            continue
        if not _latin(r.txt) <= _latin(new) or _nums(r.txt) != _nums(new):
            log.append({**entry, "after": new, "status": "rejected: Latin term or number changed"})
            continue
        if coverage_missing(invoke, r.src, new, L, note):
            log.append({**entry, "after": new, "status": "rejected: omission"})
            continue
        en_j = r.src + (f" (INTENDED MEANING, verified: {note})" if note else "")
        votes = pairwise(judges, L, en_j, prev, nxt, new, r.txt, rng)
        if "x" in votes and "y" in votes:
            # a 1-1 split: both judges vote again (fresh A/B order, so position bias evens out); the
            # fix wins with >= 3 of 4 votes and at most 1 against — the rule check_vendor_review
            # uses. In the first test most real fixes were blocked by exactly this split.
            votes = votes + pairwise(judges, L, en_j, prev, nxt, new, r.txt, rng)
            ok = votes.count("x") >= 3 and votes.count("y") <= 1
        else:
            ok = "x" in votes and "y" not in votes
        if not ok:
            log.append({**entry, "after": new, "status": f"rejected: judges {votes}"})
            continue
        r.txt = new
        log.append({**entry, "after": new, "status": "fixed"})
    n = sum(e["status"] == "fixed" for e in log)
    logger(f"{lang}: review-and-fix — {len(cands)} line(s) reviewed, {n} fixed")
    return log
