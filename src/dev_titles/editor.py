"""NATIVE-EDITOR pass: the final self-review before a human reviewer sees the subtitles.

The QE panel hunts for ERRORS and auto-repairs majors; naturalness problems (literal idioms,
calqued phrasing, odd/coined terms, run-on lists, stiff register) were logged as 'minor,
subjective' and shipped — they were most of what a native viewer still noticed in the
2026-10-17 re-run eval (ja 5%, zh 6% of sentences NOTICEABLE). This pass asks a native subtitle
editor (reference-free) to flag ONLY lines a native viewer would notice, with a minimal rewrite,
then accepts a rewrite only if it passes:
  1. deterministic guards — every Latin token and number survives; not much longer;
  2. a PAIRWISE check by a judge (the different-family second judge when available, else
     Claude) — same meaning AND B preferred, with A/B order randomized against position bias;
  3. ratings — both judges rate current vs suggested line (no reference): a pairwise-accepted
     suggestion either rates worse is vetoed; a pairwise-rejected one every judge rates better is
     rescued (scripts/eval/gate_benchmark.py, gate_compare.py, gate_rescue.py).
Idempotent via <out>/editor_state.<lang>.json; every accepted edit is logged (before/after/issue)
to editor_log.<lang>.json. Few-shot examples come from glossary/editor_examples.<lang>.jsonl —
before->after pairs mined from human reviews (scripts/mine_review.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path

_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9'’.+#/-]*")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")

EDIT_PROMPT = """You are a native {L} subtitle editor doing the FINAL pass on machine-translated subtitles of a
developer video, before a human reviewer sees them. Your goal: the reviewer should find nothing a
careful native editor could have fixed.

VIDEO BRIEF: {brief}

Flag ONLY lines a native viewer would notice as machine-made or wrong:
- literal / word-for-word renderings of idioms, tautologies or set phrases (use the natural {L} equivalent);
- unnatural or coined word choices, odd terms, wrong register (keep the house register: {register});
- awkward word order, run-on sentences, list items without separators;
- a meaning misread in context (tech-industry jargon, tag questions like "…, right?", negation, sarcasm);
- the wrong SPEAKER or addressee: when the English is tagged [speaker], a reply or reaction from
  the other person must read as theirs ("Thank you for having me" from a guest -> ja こちらこそ…,
  zh 也谢谢您的邀请), and "you" said to one guest is that person, not the audience;
- ADDED material the English doesn't have: a filler (ja ええ, zh 嗯) where the speaker said nothing like it,
  an "also" (ja も, zh 也) the English lacks, or an explicit noun for "it / them" when the previous line
  already names it — remove it;
- a line marked [means: …] in the English column that doesn't convey that meaning (those notes resolve
  idioms and ambiguous phrases: "Bring it home" = make it concrete, not "wrap up");
- stray or unbalanced quotation marks, leftover characters;
- the AUDIENCE addressed as a single person: when the speaker talks to viewers / the room
  ("Are you interested in…?"), zh uses 大家 (not 您), ja uses 皆さん;
- rhetorical questions and talk openers rendered as literal questions ("What are we going to talk
  about today?" -> ja 今日は…についてお話しします / 今日お話しするのは…, zh 今天我们来聊聊…);
- a word used in the wrong SENSE for this context ("this has huge upside" = potential, not a literal
  "upper side"; "the talent" = performers, not ability), and stiff or rare words where an everyday one
  exists (household/everyday objects: use the word people actually say).
Do NOT flag lines that are already correct and natural, mere style preferences, or condensation choices.
Keep EXACTLY: product/brand names, code, numbers, and these glossary renderings: {terms}
A rewrite must keep every fact, name and number, read as natural spoken subtitle text, and not be
longer than needed (shorter is welcome). Follow the punctuation convention of the current line.
{examples}
LINES (id | English | current {L}{hint_col}):
{items}

Return ONLY a JSON array of the flagged lines, e.g. [{{"id": 3, "issue": "literal idiom", "rewrite": "…"}}];
return [] if every line is fine."""

PAIR_PROMPT = """You compare two {L} subtitle renderings of the same English line from a developer video.
English: {en}
Context (neighbouring English): {ctx}
A: {a}
B: {b}
Answer ONLY JSON: {{"same_meaning": true|false, "better": "A"|"B"|"tie", "why": "<max 12 words>"}}.
"same_meaning" = both convey what the SPEAKER MEANT with no lost or added fact. A natural rendering of an
idiom or figure of speech ("move the needle" -> make a real difference; "boil the ocean" -> try to do
everything at once; a rare dictionary word -> the everyday word people say) conveys the same meaning — it is NOT less faithful than a word-for-word
one. "better" = what a professional native subtitler would put on screen: natural, idiomatic, correct in
context. Prefer B only if a native viewer would genuinely find it better."""


def _h(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def _toks(s, rx):
    from collections import Counter

    return Counter(rx.findall(s or ""))


def guard_ok(old: str, new: str, max_growth: float = 1.15, slack: int = 3) -> bool:
    if not new or not new.strip():
        return False
    if len(re.sub(r"\s", "", new)) > max(
        len(re.sub(r"\s", "", old)) * max_growth, len(old) + slack
    ):
        return False
    for rx in (_LATIN, _NUM):
        if _toks(old, rx) - _toks(new, rx):
            return False
    return True


def load_examples(lang: str, path_tpl: str = "glossary/editor_examples.{lang}.jsonl", k: int = 12):
    p = Path(path_tpl.format(lang=lang))
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    # evaluation hold-out: when re-translating an already-reviewed video to measure the harness, its
    # own reviewed lines must not be in the prompt (SUBTITLE_HOLDOUT_VIDEOS=VideoA,VideoB)
    held = {
        v.strip() for v in os.environ.get("SUBTITLE_HOLDOUT_VIDEOS", "").split(",") if v.strip()
    }
    rows = [r for r in rows if r.get("video") not in held]
    return rows[:k]


def _examples_block(examples, L):
    if not examples:
        return ""
    lines = "\n".join(
        f"- EN: {e['en']}\n  before: {e['before']}\n  after:  {e['after']}" for e in examples
    )
    return f"\nEdits native {L} reviewers made to earlier machine drafts (learn the KIND of fix, don't copy):\n{lines}\n"


def deterministic_hints(txt: str, lang: str, src: str = "") -> list[str]:
    """Cheap flags fed to the editor (and fixed when safe): unbalanced quotes, run-ons."""
    hints = []
    if txt.count("“") != txt.count("”") or txt.count("「") != txt.count("」"):
        hints.append("unbalanced quotes")
    if lang.startswith("zh") and re.search(r"[一-鿿]{28,}", txt):
        hints.append("long run without punctuation")
    if lang == "ja" and re.search(r"[ぁ-んァ-ヶ一-龯ー]{40,}", txt):
        hints.append("long run without a break")
    if src:  # an English word in the line that the English sentence doesn't contain: a stray
        # fragment (a reviewer deleted a dangling 'LLM' left at a line end)
        low = src.lower()  # substring match: plurals (LLMs), compounds (website ⊃ web) are fine
        stray = [
            w for w in _LATIN.findall(txt) if len(w) > 1 and w.lower().rstrip(".'’") not in low
        ]
        if stray:
            hints.append(
                "English word(s) not in the English line: " + ", ".join(dict.fromkeys(stray))
            )
    return hints


def fix_unbalanced_quotes(txt: str) -> str:
    """A lone closing/opening curly quote with no partner in the same sentence is a stray — drop it."""
    for o, c in (("“", "”"), ("「", "」")):
        if txt.count(o) == 0 and txt.count(c) == 1:
            txt = txt.replace(c, "", 1)
        elif txt.count(c) == 0 and txt.count(o) == 1:
            txt = txt.replace(o, "", 1)
    return txt


def _parse_json_array(text):
    m = re.search(r"\[.*\]", text or "", re.S)
    try:
        v = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        v = []
    return [x for x in v if isinstance(x, dict)]


def _pairwise(judge, L, en, ctx, old, new, rng):
    swap = rng.random() < 0.5
    a, b = (new, old) if swap else (old, new)
    out = judge(PAIR_PROMPT.format(L=L, en=en, ctx=ctx, a=a, b=b)) or ""
    m = re.search(r"\{.*\}", out, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    better = d.get("better")
    new_wins = (better == "A") if swap else (better == "B")
    if better == "tie" and d.get("same_meaning"):
        return False, "__tie__"
    # report the verdict in OUR terms (the judge's A/B labels were randomized)
    verdict = (
        "prefers the suggestion"
        if new_wins
        else "prefers the current line" if better in ("A", "B") else "tie"
    )
    if not d.get("same_meaning"):
        verdict += "; says the meaning differs"
    return bool(d.get("same_meaning")) and new_wins, verdict


RATE_PROMPT = """You are a native {L} speaker and professional subtitle reviewer. For each item: an English line and a
{L} subtitle of it (a whole sentence that may span several cues — don't mark it down for total length).
- OK: a native viewer finds it fine.
- NOTICEABLE: correct meaning, but unnatural/literal/odd term/wrong register — a native viewer would notice.
- MEANING: a native viewer would misunderstand or miss what was said.
Return ONLY a JSON array: [{{"id": n, "level": "OK|NOTICEABLE|MEANING"}}]

{body}"""
_RANK = {"OK": 0, "NOTICEABLE": 1, "MEANING": 2}


def rate_pairs(cands, lang_name, judges, seed=29) -> dict:
    """{i: [(current_level, suggested_level) per judge]} for candidates (rec, new, ...), rated
    reference-free by each judge with current and suggested lines shuffled together, unlabeled.
    A judge whose rating failed or didn't cover both lines is simply absent from the list."""
    if not cands:
        return {}
    items = []
    for c in cands:
        items += [(c[0].i, "o", c[0].src, c[0].txt), (c[0].i, "s", c[0].src, c[1])]
    random.Random(seed + len(items)).shuffle(items)
    body = "\n\n".join(
        f"{n + 1}. EN: {en}\n   SUBTITLE: {t}" for n, (_, _, en, t) in enumerate(items)
    )
    out: dict = {c[0].i: [] for c in cands}
    for judge in judges:
        try:
            js = _parse_json_array(judge(RATE_PROMPT.format(L=lang_name, body=body)) or "")
        except Exception:
            continue
        lv = {}
        for j in js:
            try:
                it = items[int(j["id"]) - 1]
                lv[(it[0], it[1])] = _RANK[j.get("level")]
            except (KeyError, ValueError, IndexError, TypeError):
                pass
        for i in out:
            if (i, "o") in lv and (i, "s") in lv:
                out[i].append((lv[(i, "o")], lv[(i, "s")]))
    return out


def _gate(
    items,
    *,
    invoke,
    judge,
    lang_name,
    by_i,
    normalize,
    rng,
    stats,
    log,
    changed,
    ctx_fn=None,
    slack=3,
):
    """Accept editor suggestions [(rec, rewrite, issue)] in place through the measured gate: token/
    number guard -> pairwise second judge (randomized, tie -> other-family tie-break) -> both judges
    rate current vs suggested (no reference): VETO an accepted one either rates worse, RESCUE a
    rejected one every judge rates better (gate benchmark 2026-10-18, scripts/eval/gate_*.py)."""
    cands: list = []
    rejected: list = []
    for r, new, issue in items:
        if not new or new == r.txt:
            continue
        stats["flagged"] += 1
        if not guard_ok(r.txt, new, slack=slack):
            stats["guard"] += 1
            continue
        ctx = (
            ctx_fn(r) if ctx_fn else " ".join(by_i[j].src for j in (r.i - 1, r.i + 1) if j in by_i)
        )
        try:
            ok, why = _pairwise(judge, lang_name, r.src, ctx, r.txt, new, rng)
            if not ok and why == "__tie__" and judge is not invoke:
                ok, why = _pairwise(invoke, lang_name, r.src, ctx, r.txt, new, rng)
                why = f"tie-break: {why}"
        except Exception:
            ok, why = False, "pairwise failed"
        (cands if ok else rejected).append((r, normalize(new, r.src), issue, why))
    judges = (invoke, judge) if judge is not invoke else (invoke,)
    lv = rate_pairs(cands + rejected, lang_name, judges)
    for passed, (r, new, issue, why) in [(True, c) for c in cands] + [(False, c) for c in rejected]:
        pairs = lv.get(r.i, [])
        if passed:
            ok = not any(s_ > o for o, s_ in pairs)
            if not ok:
                stats["veto"] += 1
                why = f"{why}; a judge rated it worse"
        else:
            ok = len(pairs) == len(judges) and all(s_ < o for o, s_ in pairs)
            if ok:
                stats["rescued"] += 1
                why = f"{why}; rescued: every judge rated it better"
            else:
                stats["pair"] += 1
        log.append(
            {
                "i": r.i,
                "src": r.src,
                "before": r.txt,
                "after": new,
                "issue": issue,
                "why": why,
                "status": "accepted" if ok else "rejected",
            }
        )
        if ok:
            r.txt = new
            changed.add(r.i)
            stats["accepted"] += 1


def native_edit(
    records,
    invoke,
    lang,
    lang_name,
    state_path,
    *,
    normalize,
    pair_judge=None,
    brief="",
    register="",
    glossary_terms=(),
    logger=print,
    batch=6,
    seed=17,
    speakers=None,
    meaning_notes=None,
):
    """Run the native-editor pass over SentenceRecords in place. Returns the set of changed indices."""
    sp = Path(state_path) if state_path else None
    state = json.loads(sp.read_text(encoding="utf-8")) if sp and sp.exists() else {}
    recs = sorted(records, key=lambda r: r.i)
    by_i = {r.i: r for r in recs}
    pending = [r for r in recs if state.get(_h(r.src)) != _h(r.txt)]
    if not pending:
        logger(f"{lang}: native editor — nothing new to review")
        return set()
    # deterministic fixes first (safe, local)
    for r in pending:
        fixed = fix_unbalanced_quotes(r.txt)
        if fixed != r.txt:
            r.txt = fixed
    examples = _examples_block(load_examples(lang), lang_name)
    terms = ", ".join(sorted({t for t in glossary_terms if t}))[:1500] or "(none)"
    judge = pair_judge or invoke
    rng = random.Random(seed)
    changed: set = set()
    log: list = []
    stats = {
        "reviewed": 0,
        "flagged": 0,
        "accepted": 0,
        "guard": 0,
        "pair": 0,
        "veto": 0,
        "rescued": 0,
    }
    for k in range(0, len(pending), batch):
        chunk = pending[k : k + batch]
        rows = []
        for n, r in enumerate(chunk, 1):
            hints = deterministic_hints(r.txt, lang, r.src)
            who = (
                f"[{speakers[r.i]}] " if speakers and r.i < len(speakers) and speakers[r.i] else ""
            )
            from .en_gloss import meaning_tag

            src = r.src + meaning_tag(r.i, meaning_notes)
            rows.append(f"{n} | {who}{src} | {r.txt}" + (f" | {', '.join(hints)}" if hints else ""))
        prompt = EDIT_PROMPT.format(
            L=lang_name,
            brief=brief or "(none)",
            register=register or "polite, natural spoken",
            terms=terms,
            examples=examples,
            hint_col=" | hints",
            items="\n".join(rows),
        )
        try:
            flags = _parse_json_array(invoke(prompt) or "")
        except Exception as e:  # never sink the run on an editor failure
            logger(f"{lang}: native editor batch failed ({type(e).__name__}) — skipped")
            flags = []
        stats["reviewed"] += len(chunk)
        items = []
        for f in flags:
            try:
                r = chunk[int(f.get("id")) - 1]
            except (TypeError, ValueError, IndexError):
                continue
            items.append((r, (f.get("rewrite") or "").strip(), f.get("issue", "")))
        _gate(
            items,
            invoke=invoke,
            judge=judge,
            lang_name=lang_name,
            by_i=by_i,
            normalize=normalize,
            rng=rng,
            stats=stats,
            log=log,
            changed=changed,
        )
        logger(f"{lang}: native editor {min(k + batch, len(pending))}/{len(pending)} {stats}")
    for r in pending:
        state[_h(r.src)] = _h(r.txt)
    if sp:
        sp.write_text(json.dumps(state), encoding="utf-8")
        lp = sp.with_name(sp.name.replace("editor_state", "editor_log"))
        old = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else []
        lp.write_text(json.dumps(old + log, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


REPLY_PROMPT = """You are a native {L} subtitle editor checking SPEAKER TURNS in an interview / talk. Each block
shows the end of one person's turn, then (marked >>) the start of the next person's turn, with the
current {L} subtitles.

VIDEO BRIEF: {brief}

For the >> lines only, flag a line whose {L} does not read as THAT person replying to what was just
said, and give a minimal rewrite. Typical problems:
- reciprocal thanks rendered as fresh thanks: after the host thanks the guest, "Thank you for having
  me" / "Thanks for having me" is a reply (ja こちらこそ…ありがとうございました, zh 也谢谢您的邀请);
- a one-word reaction used as a reply rendered literally: "Brilliant." / "Cheers." / "Great." / "Sure." /
  "Absolutely." answering thanks or a request means thanks / agreement / of course, not the adjective
  (ja 素晴らしいです for a reply "Brilliant." is wrong; ありがとうございます / ええ / もちろんです);
- yes/no answers that contradict the question's polarity; agreement words (同感です / そうですね);
- "you" meaning the other speaker rendered as the audience (皆さん / 大家), or the reverse;
- a welcome or thank-you addressed to the guest rendered as addressed to viewers.
Flag nothing else: lines that already read naturally as the reply stay as they are. Keep every fact,
name and number; keep it short.
{examples}
{items}

Return ONLY a JSON array, e.g. [{{"id": 57, "issue": "reply to host's thanks", "rewrite": "…"}}]; [] if all fine."""


REPLY_PAIR_PROMPT = """An exchange from a developer video, with two {L} subtitle renderings of the LAST line.
{exchange}
LAST LINE ({who}): "{en}"
A: {a}
B: {b}
Which is the better {L} subtitle for THIS person saying this line IN THIS EXCHANGE — conveying what they
mean as a reply (thanks returned, agreement, an answer), to the right person, with no fact lost or
invented? Translating a reply-word literally ("Brilliant." as 素晴らしい) is worse than its function
in context. Answer ONLY JSON: {{"better": "A"|"B"|"tie", "why": "<max 12 words>"}}"""


def _reply_accept(judges, lang_name, exchange, who, en, old, new, rng) -> tuple[bool, str]:
    """Accept only if EVERY judge, seeing the exchange, prefers the rewrite (order randomized)."""
    whys = []
    for judge in judges:
        swap = rng.random() < 0.5
        a, b = (new, old) if swap else (old, new)
        try:
            out = (
                judge(
                    REPLY_PAIR_PROMPT.format(
                        L=lang_name, exchange=exchange, who=who, en=en, a=a, b=b
                    )
                )
                or ""
            )
            m = re.search(r"\{.*\}", out, re.S)
            d = json.loads(m.group(0)) if m else {}
        except Exception:
            return False, "reply judge failed"
        if d.get("better") != ("A" if swap else "B"):
            return False, f"a judge kept the current line: {d.get('why', '')}"
        whys.append(d.get("why", ""))
    return True, "every judge prefers it as the reply: " + " / ".join(whys)


def reply_edit(
    records,
    invoke,
    lang,
    lang_name,
    speakers,
    *,
    normalize,
    pair_judge=None,
    brief="",
    state_path=None,
    logger=print,
    window=2,
    per_call=8,
    seed=31,
):
    """Turn-boundary editor: at each speaker change, check that the new speaker's first lines read
    as a reply to the previous speaker (こちらこそ for reciprocal thanks, a reply "Brilliant." is not
    素晴らしい). Sentence-by-sentence translation and the general editor miss this (A/B 2026-10-20).
    Suggestions go through the same measured gate as native_edit. Returns the set of changed indices.
    """
    if not speakers:
        return set()
    recs = sorted(records, key=lambda r: r.i)
    by_i = {r.i: r for r in recs}
    turns = [
        i
        for i in range(1, len(speakers))
        if speakers[i] and speakers[i] != speakers[i - 1] and i in by_i
    ]
    sp = Path(state_path) if state_path else None
    state = json.loads(sp.read_text(encoding="utf-8")) if sp and sp.exists() else {}

    def key(i):
        return _h(
            "|".join(
                by_i[j].src + "\x00" + by_i[j].txt
                for j in range(i - window, i + window)
                if j in by_i
            )
        )

    todo = [i for i in turns if state.get(str(i)) != key(i)]
    if not todo:
        return set()
    judge = pair_judge or invoke
    rng = random.Random(seed)
    changed: set = set()
    log: list = []
    stats = {
        "reviewed": 0,
        "flagged": 0,
        "accepted": 0,
        "guard": 0,
        "pair": 0,
        "veto": 0,
        "rescued": 0,
    }

    def tag(j):
        return f"[{speakers[j]}]" if j < len(speakers) and speakers[j] else ""

    def ctx_fn(r):  # the exchange before the line, with who says what
        return "\n".join(
            f'{tag(j)}: "{by_i[j].src}"' for j in range(r.i - window - 1, r.i) if j in by_i
        )

    for k in range(0, len(todo), per_call):
        part = todo[k : k + per_call]
        blocks, allowed = [], {}
        for n, i in enumerate(part, 1):
            lines = [f"BLOCK {n}"]
            for j in range(i - window, i + window):
                if j not in by_i:
                    continue
                mark = ">> " if j >= i else "   "
                lines.append(
                    f"{mark}{tag(j)} (id {j}) EN: {by_i[j].src} | {lang_name}: {by_i[j].txt}"
                )
                if j >= i and speakers[j] == speakers[i]:
                    allowed[j] = by_i[j]
            blocks.append("\n".join(lines))
        prompt = REPLY_PROMPT.format(
            L=lang_name, brief=brief or "(none)", examples="", items="\n\n".join(blocks)
        )
        try:
            flags = _parse_json_array(invoke(prompt) or "")
        except Exception as e:
            logger(f"{lang}: reply check batch failed ({type(e).__name__}) — skipped")
            flags = []
        stats["reviewed"] += len(part)
        judges = (invoke, judge) if judge is not invoke else (invoke,)
        for f in flags:
            try:
                r = allowed[int(f.get("id"))]
            except (TypeError, ValueError, KeyError):
                continue
            new = (f.get("rewrite") or "").strip()
            if not new or new == r.txt:
                continue
            stats["flagged"] += 1
            # a reply marker (こちらこそ / ええ / いいですね) adds a few characters
            if not guard_ok(r.txt, new, slack=7):
                stats["guard"] += 1
                continue
            # The general gate judges a line in isolation and calls a reply rendering ("Brilliant."
            # -> ありがとうございます) a meaning change; here every judge sees the exchange.
            ok, why = _reply_accept(judges, lang_name, ctx_fn(r), tag(r.i), r.src, r.txt, new, rng)
            new = normalize(new, r.src)
            log.append(
                {
                    "i": r.i,
                    "src": r.src,
                    "before": r.txt,
                    "after": new,
                    "issue": "reply: " + f.get("issue", ""),
                    "why": why,
                    "status": "accepted" if ok else "rejected",
                }
            )
            if ok:
                r.txt = new
                changed.add(r.i)
                stats["accepted"] += 1
            else:
                stats["pair"] += 1
    for i in todo:
        state[str(i)] = key(i)
    if sp:
        sp.write_text(json.dumps(state), encoding="utf-8")
        lp = sp.with_name(sp.name.replace("reply_state", "editor_log"))
        old = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else []
        lp.write_text(json.dumps(old + log, ensure_ascii=False, indent=1), encoding="utf-8")
    logger(f"{lang}: reply check over {len(todo)} speaker turns {stats}")
    return changed


BRIEF_PROMPT = """Write a 3-5 sentence brief of this video for a subtitle translator: who is speaking (names,
roles/job titles, companies — the DESCRIPTION is authoritative for these), the topic, the audience, the
tone (interview / talk / tutorial), and any jargon whose meaning depends on context (e.g. "product"
meaning a product-management team). Plain text, no lists.

VIDEO DESCRIPTION (may be empty):
{description}

TRANSCRIPT NOTES / TEXT:
{text}"""

NOTES_PROMPT = """Summarize this part of a video transcript in 3-6 short bullet notes for a subtitle translator:
speakers and their roles, topics, products/companies named, context-dependent jargon. No preamble.

{text}"""


def video_description(out_dir) -> str:
    """The video's written description: <out>/description.txt if present (saved at ingest, or
    supplied with --description-file); otherwise, if <out>/SOURCE-URL.txt names a YouTube video,
    fetch it once with yt-dlp and cache it. Best-effort: '' on any failure (unpublished videos)."""
    if not out_dir:
        return ""
    d = Path(out_dir)
    f = d / "description.txt"
    if f.exists():
        return f.read_text(encoding="utf-8").strip()
    src = d / "SOURCE-URL.txt"
    if not src.exists():
        return ""
    m = re.search(r"https?://\S*(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]{11}", src.read_text())
    if not m:
        return ""
    try:
        import subprocess

        out = subprocess.run(
            ["yt-dlp", "--skip-download", "--print", "description", m.group(0)],
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout.strip()
    except Exception:
        out = ""
    if out:
        f.write_text(out, encoding="utf-8")
    return out


def video_brief(
    en_text: str, invoke, cache_path=None, description: str = "", chunk: int = 12000
) -> str:
    """One paragraph of context shared by the translator and the editor. Built from the video's
    DESCRIPTION (authoritative for names/roles) plus the WHOLE transcript — long transcripts are
    summarized chunk by chunk first (map-reduce) so a late-introduced guest or topic isn't missed.
    Cached per out dir; regenerated when the description or transcript changes."""
    p = Path(cache_path) if cache_path else None
    key = hashlib.sha1(
        (description + "\x00" + en_text).encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:16]
    if p and p.exists():
        txt = p.read_text(encoding="utf-8")
        if txt.startswith(f"# {key}\n"):
            return txt.split("\n", 1)[1].strip()
    try:
        if len(en_text) > chunk:
            parts = [en_text[k : k + chunk] for k in range(0, len(en_text), chunk)]
            notes = "\n".join((invoke(NOTES_PROMPT.format(text=t)) or "").strip() for t in parts)
        else:
            notes = en_text
        brief = (
            invoke(BRIEF_PROMPT.format(description=description[:4000] or "(none)", text=notes))
            or ""
        ).strip()
    except Exception:
        brief = ""
    if p and brief:
        p.write_text(f"# {key}\n{brief}", encoding="utf-8")
    return brief


REPAIR_PROMPT = """Two {L} subtitle renderings of the same English line from a developer video. A is the current
subtitle; B is a proposed correction. Judge as a native viewer: ACCURACY to what the speaker meant in
context first, then natural spoken {L}. A literal rendering of a word in the wrong sense is less accurate.
English: {en}
Context (neighbouring English): {ctx}
A: {a}
B: {b}
Answer ONLY JSON: {{"better": "A"|"B"|"tie", "why": "<max 12 words>"}}"""


def repair_check(judge, lang_name, records, seed=23):
    """accept_check for QE-repair: accept a repair only if a native-viewer judge prefers it to the
    current line (accuracy first, then naturalness). Order randomized against position bias."""
    by_i = {r.i: r for r in records}
    rng = random.Random(seed)

    def check(rec, old, new):
        swap = rng.random() < 0.5
        a, b = (new, old) if swap else (old, new)
        ctx = " ".join(by_i[j].src for j in (rec.i - 1, rec.i + 1) if j in by_i)
        try:
            out = judge(REPAIR_PROMPT.format(L=lang_name, en=rec.src, ctx=ctx, a=a, b=b)) or ""
            m = re.search(r"\{.*\}", out, re.S)
            d = json.loads(m.group(0)) if m else {}
        except Exception:
            return True  # never block a repair because the check itself failed
        better = d.get("better")
        return better == ("A" if swap else "B")

    return check


def editor_rewritten(state_path, records) -> set:
    """Indices whose CURRENT text is a native-editor rewrite (from editor_log.<lang>.json)."""
    if not state_path:
        return set()
    lp = Path(state_path).with_name(Path(state_path).name.replace("editor_state", "editor_log"))
    if not lp.exists():
        return set()
    after = {
        (e["src"].strip(), e["after"])
        for e in json.loads(lp.read_text(encoding="utf-8"))
        if e.get("status", "accepted") == "accepted"
    }
    return {r.i for r in records if (r.src.strip(), r.txt) in after}


def review_section(out_dir, lang) -> str:
    """REVIEW.md block: editor suggestions the second judge rejected (reviewer decides), and a count
    of accepted rewrites."""
    lp = Path(out_dir) / f"editor_log.{lang}.json"
    if not lp.exists():
        return ""
    log = json.loads(lp.read_text(encoding="utf-8"))
    acc = [e for e in log if e.get("status", "accepted") == "accepted"]
    rej = [e for e in log if e.get("status") == "rejected"]
    out = [
        f"- **Native editor:** {len(acc)} line(s) rewritten for naturalness"
        + (
            f"; {len(rej)} suggestion(s) the second judge did not accept — your call:"
            if rej
            else "."
        )
    ]
    for e in rej[:30]:
        out.append(
            "    - EN: "
            + e["src"][:140]
            + "\n      current: "
            + e["before"]
            + "\n      suggested: "
            + e["after"]
            + "\n      editor: "
            + e.get("issue", "")[:120]
            + " | judge: "
            + e.get("why", "")[:80]
        )
    return "\n".join(out) + "\n"
