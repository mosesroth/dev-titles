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
     rescued (scripts/gate_benchmark.py, gate_compare.py, gate_rescue.py).
Idempotent via <out>/editor_state.<lang>.json; every accepted edit is logged (before/after/issue)
to editor_log.<lang>.json. Few-shot examples come from glossary/editor_examples.<lang>.jsonl —
before->after pairs mined from human reviews (scripts/mine_review.py).
"""

from __future__ import annotations

import hashlib
import json
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


def guard_ok(old: str, new: str, max_growth: float = 1.15) -> bool:
    if not new or not new.strip():
        return False
    if len(re.sub(r"\s", "", new)) > max(len(re.sub(r"\s", "", old)) * max_growth, len(old) + 3):
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
    return rows[:k]


def _examples_block(examples, L):
    if not examples:
        return ""
    lines = "\n".join(
        f"- EN: {e['en']}\n  before: {e['before']}\n  after:  {e['after']}" for e in examples
    )
    return f"\nEdits native {L} reviewers made to earlier machine drafts (learn the KIND of fix, don't copy):\n{lines}\n"


def deterministic_hints(txt: str, lang: str) -> list[str]:
    """Cheap flags fed to the editor (and fixed when safe): unbalanced quotes, run-ons."""
    hints = []
    if txt.count("“") != txt.count("”") or txt.count("「") != txt.count("」"):
        hints.append("unbalanced quotes")
    if lang.startswith("zh") and re.search(r"[一-鿿]{28,}", txt):
        hints.append("long run without punctuation")
    if lang == "ja" and re.search(r"[ぁ-んァ-ヶ一-龯ー]{40,}", txt):
        hints.append("long run without a break")
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
    verdict = ("prefers the suggestion" if new_wins else "prefers the current line" if better in ("A", "B") else "tie")
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
    body = "\n\n".join(f"{n + 1}. EN: {en}\n   SUBTITLE: {t}" for n, (_, _, en, t) in enumerate(items))
    out = {c[0].i: [] for c in cands}
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
    changed, log = set(), []
    stats = {"reviewed": 0, "flagged": 0, "accepted": 0, "guard": 0, "pair": 0, "veto": 0, "rescued": 0}
    for k in range(0, len(pending), batch):
        chunk = pending[k : k + batch]
        rows = []
        for n, r in enumerate(chunk, 1):
            hints = deterministic_hints(r.txt, lang)
            rows.append(f"{n} | {r.src} | {r.txt}" + (f" | {', '.join(hints)}" if hints else ""))
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
        cands, rejected = [], []
        for f in flags:
            try:
                r = chunk[int(f.get("id")) - 1]
            except (TypeError, ValueError, IndexError):
                continue
            new = (f.get("rewrite") or "").strip()
            if not new or new == r.txt:
                continue
            stats["flagged"] += 1
            if not guard_ok(r.txt, new):
                stats["guard"] += 1
                continue
            ctx = " ".join(by_i[j].src for j in (r.i - 1, r.i + 1) if j in by_i)
            try:
                ok, why = _pairwise(judge, lang_name, r.src, ctx, r.txt, new, rng)
                if not ok and why == "__tie__" and judge is not invoke:
                    # the second judge called it a tie (same meaning, neither better): ask the other
                    # family for a tie-break instead of discarding a plausible improvement
                    ok, why = _pairwise(invoke, lang_name, r.src, ctx, r.txt, new, rng)
                    why = f"tie-break: {why}"
            except Exception:
                ok, why = False, "pairwise failed"
            (cands if ok else rejected).append((r, normalize(new, r.src), f.get("issue", ""), why))
        # Both judges rate current vs suggested line, reference-free (gate benchmark 2026-10-18,
        # 4,282 ja / 1,737 zh sentences vs human reviews, scripts/gate_*.py):
        #  - VETO a pairwise-accepted suggestion either judge rates worse: harmful edits ja 53->34,
        #    zh 14->10, for 4% fewer improvements;
        #  - RESCUE a pairwise-rejected one (incl. "meaning differs") that EVERY judge rates better:
        #    ja 42 improved / 0 harmed, zh 17 / 2.
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
            log.append({"i": r.i, "src": r.src, "before": r.txt, "after": new, "issue": issue,
                        "why": why, "status": "accepted" if ok else "rejected"})
            if ok:
                r.txt = new
                changed.add(r.i)
                stats["accepted"] += 1
        logger(f"{lang}: native editor {min(k + batch, len(pending))}/{len(pending)} {stats}")
    for r in pending:
        state[_h(r.src)] = _h(r.txt)
    if sp:
        sp.write_text(json.dumps(state), encoding="utf-8")
        lp = sp.with_name(sp.name.replace("editor_state", "editor_log"))
        old = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else []
        lp.write_text(json.dumps(old + log, ensure_ascii=False, indent=1), encoding="utf-8")
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


def video_brief(en_text: str, invoke, cache_path=None, description: str = "", chunk: int = 12000) -> str:
    """One paragraph of context shared by the translator and the editor. Built from the video's
    DESCRIPTION (authoritative for names/roles) plus the WHOLE transcript — long transcripts are
    summarized chunk by chunk first (map-reduce) so a late-introduced guest or topic isn't missed.
    Cached per out dir; regenerated when the description or transcript changes."""
    p = Path(cache_path) if cache_path else None
    key = hashlib.sha1((description + "\x00" + en_text).encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
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
        brief = (invoke(BRIEF_PROMPT.format(description=description[:4000] or "(none)", text=notes)) or "").strip()
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
    out = [f"- **Native editor:** {len(acc)} line(s) rewritten for naturalness"
           + (f"; {len(rej)} suggestion(s) the second judge did not accept — your call:" if rej else ".")]
    for e in rej[:30]:
        out.append(
            "    - EN: " + e["src"][:140]
            + "\n      current: " + e["before"]
            + "\n      suggested: " + e["after"]
            + "\n      editor: " + e.get("issue", "")[:120] + " | judge: " + e.get("why", "")[:80]
        )
    return "\n".join(out) + "\n"
