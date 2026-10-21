"""Subtitle CONDENSATION pass (ja, ON by default): tighten finished translations the way
a native subtitle editor does, with guardrails so it can't drop real content.

Why a separate pass: native ja reviewers keep ~88% of our length. Only ~8% of what they cut is
rule-able (filler/intensifiers/pronouns/hedges); the rest is editing judgment — dropping back-channel
reactions and redundant restatements, merging clauses. Asking the translator for a tighter budget did
nothing, because the completeness gates (omission retry + coverage edit-fix) treat any condensation
as a possible omission and push the length back up. So condense AFTER translation, then verify.

Guardrails, per sentence (any failure keeps the original):
  1. never longer than the original;
  2. every Latin token and every number in the original survives (names, products, code, figures);
  3. a coverage check against the EN finds no missing SUBSTANTIVE point (coverage_missing already
     ignores filler/pronouns/intensifiers/greetings).
  4. after QE, a condensed line either judge flagged is offered back in its original form
     (accept-only-if-better) — the coverage check can't see broken grammar or lost politeness.
Only lines clearly too fast to read are condensed (trigger ~10 cps), toward the 8.5 cps budget.
"""

from __future__ import annotations

import json
import re
from collections import Counter

_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9'’.+#/-]*")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")

_PROMPT = """You are a native {lang_name} subtitle editor tightening translated subtitles for a published
developer video (interview, talk or tutorial). A line reaches you only because it is too long to read
in the time it is on screen. Make the SMALLEST edit that brings it within its budget:
- drop content-free filler and reflex reactions (e.g. "That's a great question", "Interesting", "Yeah",
  "Exactly" used as a reflex), intensifiers (とても/非常に/本当に) and hedges (と思います/基本的に/実際に)
  unless they change the meaning;
- drop redundant restatements and pronouns that are obvious from context; merge clauses;
- KEEP every substantive fact, claim, step, condition, name, product/technical term and number.
- KEEP the speaker's greeting and self-introduction (こんにちは…です), sign-offs, thanks and calls to
  action ("check out…", "try…", "let us know…"): they are content, not filler. A thank-you keeps what
  it is for (ご参加いただき / お時間をいただき / ご紹介), and a welcome keeps 本日は…お迎えしました.
- KEEP the politeness of every predicate: です・ます endings, honorific / humble forms (いただく, 伺う,
  お迎えする); a polite question stays a full polite question (〜ありますか, not 〜は？ or 〜か).
- KEEP who / whose when it says whose thing it is (弊社は, 私たちの, あなたの, 御社の) and time words that
  carry the point (今こそ, 本日は).
- KEEP who does what: never change the subject or agent (businesses recommending to each other is not
  customers recommending), and never swap a specific term for a broader one (GitHub Discussions is not
  議論; software is not 製品).
- KEEP focus words that carry the point: だけ / のみ / も / さえ / just / only / also / even ("it just
  needs to be a live app" must still say ONLY that), and the object of a verb (示せるのは… keeps what
  is shown).
- Every line stays a complete, grammatical sentence with a natural predicate. Never cut a request or
  closing down to a fragment ending in へ/に/を/で, and never drop the subject a predicate needs
  (「…など多数です」 is broken; 「…などに対応しています」 is fine). Keep polite request forms
  (〜ください/〜ましょう).
- Keep the register and house style (です・ます; product names in Latin). Do not add anything.
- The budget is a CEILING, not a target: stop as soon as the line fits; do not squeeze further. If it
  cannot fit without losing a fact or breaking the grammar, shorten it only as far as stays natural —
  a natural line slightly over budget is better than a broken one that fits.

Examples of edits native reviewers made (ORIGINAL -> EDITED):
{examples}

Do NOT make edits like these (natural meaning or politeness lost):
- ご質問がある場合は開発者フォーラムに投稿してください  ->  質問は開発者フォーラムへ
  (better: 質問は開発者フォーラムに投稿してください)
- こんにちは 田中です 今日は新しいSDKをご紹介します  ->  田中です 新しいSDKを紹介します  (greeting dropped)

Now edit these. Return ONLY a JSON array of strings, one per input line, same order and count.
{items}
"""


def _toks(s, rx):
    return Counter(rx.findall(s or ""))


def guard_ok(original: str, condensed: str) -> bool:
    """Deterministic guardrails 1-2: not longer, and no Latin token / number lost."""
    if not condensed or len(condensed) > len(original):
        return False
    for rx in (_LATIN, _NUM):
        if _toks(original, rx) - _toks(condensed, rx):
            return False
    return True


def condense_batch(invoke, items, lang_name, examples):
    """items: [(en_src, target[, char_budget])] -> condensed targets (None where the model failed)."""
    ex = "\n".join(f"- {a}  ->  {b}" for a, b in examples)
    body = "\n".join(
        f"{i + 1}. EN: {it[0]}\n   {lang_name.upper()}: {it[1]}"
        + (f"\n   BUDGET: <= {it[2]} characters (currently {len(it[1])})" if len(it) > 2 else "")
        for i, it in enumerate(items)
    )
    resp = invoke(_PROMPT.format(lang_name=lang_name, examples=ex, items=body)) or ""
    m = re.search(r"\[.*\]", resp, re.S)
    try:
        out = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        out = None
    if not isinstance(out, list) or len(out) != len(items):
        return [None] * len(items)
    return [o if isinstance(o, str) else None for o in out]


def condense_records(
    records,
    invoke,
    lang,
    lang_name,
    examples,
    *,
    coverage=None,
    batch=8,
    logger=print,
    budgets=None,
):
    """Condense SentenceRecords in place. `coverage(src, tgt) -> missing_str` is guardrail 3 (empty
    string = complete). Returns stats {tried, accepted, rejected_guard, rejected_coverage}."""
    st = Counter()
    for k in range(0, len(records), batch):
        chunk = records[k : k + batch]
        items = [(r.src, r.txt, budgets[r.i]) if budgets else (r.src, r.txt) for r in chunk]
        outs = condense_batch(invoke, items, lang_name, examples)
        for r, new in zip(chunk, outs):
            st["tried"] += 1
            if new is None or new.strip() == r.txt.strip():
                st["unchanged"] += 1
                continue
            new = new.strip()
            if not guard_ok(r.txt, new):
                st["rejected_guard"] += 1
                continue
            if coverage is not None and coverage(r.src, new):
                st["rejected_coverage"] += 1
                continue
            r.txt = new
            st["accepted"] += 1
        logger(f"{lang}: condense {min(k + batch, len(records))}/{len(records)} {dict(st)}")
    return dict(st)


# Illustrative native-style tightenings (filler / intensifier / hedge removal, verbose
# constructions shortened). Hand-written rather than auto-mined: mined pairs kept slipping in
# cue-boundary artifacts (text that MOVED to the next subtitle looks deleted).
DEFAULT_EXAMPLES_JA = [
    (
        "えーと、実はこのライブラリは本当にとても使いやすいんですよね",
        "このライブラリはとても使いやすいんです",
    ),
    ("設定画面からオプションを選択することができます", "設定画面からオプションを選べます"),
    ("一見するとかなり複雑そうに見えました", "一見複雑そうに見えました"),
    (
        "そうですね、基本的にはまずプロジェクトを作成するところから始めることになると思います",
        "まずプロジェクトを作成するところから始めます",
    ),
    (
        "正直に言うと、最初はパフォーマンスの問題がかなり大きかったんじゃないかなと思います",
        "最初はパフォーマンスの問題が大きかったと思います",
    ),
    (
        "なので、私たちはテストを自動化するためのツールを独自に作成したというわけです",
        "そこでテスト自動化ツールを独自に作成しました",
    ),
    (
        "はいもちろんです、当時ちょうど新しいバージョンをテストしていたところだったと思います",
        "もちろんです、当時新しいバージョンを検証していました",
    ),
    (
        "だからこそ、コミュニティの動向を常にきちんと把握しておきたいと考えているんです",
        "だからコミュニティの動向を常に把握していたいんです",
    ),
    ("これはいわゆるオープンソースのソリューションです", "オープンソースのソリューションです"),
    (
        "これらすべてによって実際の本番環境でそうしたメトリクスの統計を取得することが可能になります",
        "これらすべてで本番環境のメトリクスの統計を取得できます",
    ),
]


def _h(s: str) -> str:
    import hashlib

    return hashlib.sha1((s or "").encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def condense_lang(
    records,
    invoke,
    lang,
    lang_name,
    state_path,
    *,
    coverage_missing,
    normalize,
    logger=print,
    examples=None,
    budgets=None,
    triggers=None,
    min_width=0,
):
    """Production entry point (in A/B testing: ~12% shorter, matching reviewers; DQF flat-to-up).
    IDEMPOTENT: a sidecar <state_path> maps src-hash -> condensed-text-hash, so a re-run (which
    reuses already-condensed cached text) does NOT condense again and shave more off each run.
    `budgets` {i: chars} is the TARGET (reading-speed limit x speech window); `triggers` {i: chars}
    (default = budgets) is the looser threshold a line must exceed to be condensed at all, so lines
    only slightly over the limit (which native reviewers routinely ship) are left as translated.
    Every edit is logged original->condensed in a sidecar (see load_originals) for audit + revert.
    Records are edited in place; returns the set of changed record indices."""
    import json as _json
    from pathlib import Path as _P

    sp = _P(state_path) if state_path else None
    state = _json.loads(sp.read_text(encoding="utf-8")) if sp and sp.exists() else {}
    pending = [r for r in records if state.get(_h(r.src)) != _h(r.txt)]
    # A ONE-LINE sentence is never condensed: it gets reading time from merging with its neighbours
    # and lingering into silence, while trimming it can only cut politeness or thanks (benchmark
    # 2026-10-20 vs a native reviewer's published tracks: half the condensed lines, half the
    # important losses, 72% of the reading-speed gain kept).
    if min_width:
        from .textwidth import display_width as _dw

        pending = [r for r in pending if _dw(re.sub(r"\s", "", r.txt), True) > min_width]
    # READING-SPEED GATE (all content types): only lines that cannot be read in their time on screen
    # are condensed — budget = limit x speech window. A slow tutorial line is left exactly as
    # translated; a dense interview line is fitted to its budget. No genre detection needed.
    gate = triggers if triggers is not None else budgets
    if gate is not None:
        todo = [r for r in pending if len(re.sub(r"\s", "", r.txt)) > gate.get(r.i, 10**9)]
        skipped = len(pending) - len(todo)
    else:
        todo, skipped = pending, 0
    if not todo:
        logger(f"{lang}: condense — nothing over the reading-speed budget ({len(pending)} checked)")
        for r in pending:
            state[_h(r.src)] = _h(r.txt)
        if sp:
            sp.write_text(_json.dumps(state), encoding="utf-8")
        return set()
    logger(
        f"{lang}: condense — {len(todo)} line(s) over the reading-speed budget, {skipped} within it (untouched)"
    )
    before = {r.i: r.txt for r in todo}

    def _lost(src, new):
        miss_new = coverage_missing(invoke, src, new, lang_name)
        if not miss_new:
            return False
        r = next(x for x in todo if x.src == src)
        miss_orig = coverage_missing(invoke, src, before[r.i], lang_name)
        return not miss_orig or len(miss_new) > len(miss_orig) * 1.2

    condense_records(
        todo,
        invoke,
        lang,
        lang_name,
        examples or DEFAULT_EXAMPLES_JA,
        coverage=_lost,
        logger=logger,
        budgets=budgets,
    )
    changed = set()
    for r in pending:
        if r.i not in before:
            state[_h(r.src)] = _h(r.txt)  # within budget: checked, deliberately untouched
            continue
        if r.txt != before[r.i]:
            r.txt = normalize(r.txt, r.src)
            changed.add(r.i)
        state[_h(r.src)] = _h(r.txt)  # mark processed (changed or deliberately left as-is)
    if sp:
        sp.write_text(_json.dumps(state), encoding="utf-8")
        if changed:
            lp = _log_path(sp)
            log = _json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {}
            for r in todo:
                if r.i in changed:
                    log[_h(r.src)] = {"src": r.src, "before": before[r.i], "after": r.txt}
            lp.write_text(_json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def _log_path(state_path):
    from pathlib import Path as _P

    sp = _P(state_path)
    return sp.with_name(sp.name.replace("condense_state", "condense_log"))


def load_originals(state_path, records) -> dict:
    """{record index: pre-condensation text} for records whose CURRENT text is still the condensed
    edit (a later repair that changed the text drops out). Used to offer the original back to the
    QE panel when a condensed line is flagged — condensation can break grammar the coverage check
    can't see."""
    import json as _json

    if not state_path:
        return {}
    lp = _log_path(state_path)
    if not lp.exists():
        return {}
    log = _json.loads(lp.read_text(encoding="utf-8"))
    out = {}
    for r in records:
        e = log.get(_h(r.src))
        if e and e.get("after") == r.txt and e.get("before") and e["before"] != r.txt:
            out[r.i] = e["before"]
    return out


def mark_processed(state_path, records) -> None:
    """Record `records`' current text as already handled, so a reverted line is not re-condensed."""
    import json as _json
    from pathlib import Path as _P

    if not state_path:
        return
    sp = _P(state_path)
    state = _json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    for r in records:
        state[_h(r.src)] = _h(r.txt)
    sp.write_text(_json.dumps(state), encoding="utf-8")
