"""Learn from a human review: turn the reviewer's real improvements into harness changes.

What a person (or an assistant) did by hand after each review — compare our draft with the
reviewed final, keep what the reviewer improved, generalize it into a rule, check the rule on the
evidence before adopting it — as a job:

  1. rows: per EN sentence (en, ours, human) — aligned by content upstream (scripts/eval/ab_vs_human.py
     or scripts/learn_from_review.py);
  2. keep the rows where a blind two-judge vote prefers the HUMAN line (a vendor's own mistakes —
     ~5% of edits — must not be learned);
  3. one LLM pass proposes at most one GENERALIZABLE lesson per kept row: a glossary term, a style
     note, a fixed rendering for a whole short line, an editor example — or nothing (a one-off);
  4. lessons that duplicate or contradict the glossary are set aside for a person;
  5. each remaining glossary / style / fixed-line lesson is VALIDATED: its evidence sentences, plus
     unrelated control sentences, are re-translated with and without it, 3 times each; it is
     accepted only if the evidence wins in a sign test and the controls show no significant harm
     (blind judges again);
  6. accepted lessons are written into glossary.yaml / editor_examples.<lang>.jsonl (text insertion,
     comments kept) by apply_lessons — the caller commits; a person reviews the commit.
"""

from __future__ import annotations

import copy
import difflib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LANG_NAME = {"zh-Hans": "Simplified Chinese", "ja": "Japanese"}
KINDS = ("glossary_term", "style_note", "fixed_line", "editor_example")

PROPOSE = """You maintain the house style for {L} subtitles of developer videos (interviews and talks).
For each row: the English sentence, OUR machine subtitle, and the REVIEWER's final, which blind judges
preferred; "why" is their reason. Decide what, if anything, the harness should LEARN so it writes lines
like the reviewer's in OTHER videos too. Propose AT MOST ONE lesson per row, and only if it generalizes:

- "glossary_term": a term with one right rendering everywhere — {{"source": "<English term>",
  "target": "<{L} rendering>", "avoid": ["<our wrong rendering>"], "notes": "<why>"}}
- "style_note": a short, general rule about phrasing/register (not about this one sentence) —
  {{"text": "<rule, with a short example>"}}
- "fixed_line": a WHOLE line whose rendering NEVER depends on context (rare — NOT conversational replies or
  closings, whose rendering depends on who speaks to whom; those are style notes) —
  {{"source": "<exact English line>", "target": "<{L}>"}}
- "editor_example": a reusable naturalness fix shown by example — {{"label": "<3-6 words>"}}
- "none": a one-off choice, context-specific, or reviewer preference with no general rule

Do NOT turn a context-dependent wording into a glossary term, and do not propose a rule our existing
rules already state (EXISTING STYLE NOTES below).

EXISTING STYLE NOTES (abridged):
{notes}

ROWS:
{rows}

Return ONLY a JSON array, one object per row id:
[{{"id": <id>, "kind": "glossary_term|style_note|fixed_line|editor_example|none", ...fields}}]"""

PAIR = """Two {L} subtitles for the same English line of a developer video. Which is the better subtitle for
a native viewer — accurate, natural, idiomatic, appropriate politeness? Ignore line breaks and spacing.
Before: {prev}
ENGLISH: {en}
After: {nxt}
A: {a}
B: {b}
Answer ONLY JSON: {{"better": "A"|"B"|"equal"}}"""


def _json(text: str, kind: str = "["):
    pat = r"\[.*\]" if kind == "[" else r"\{.*\}"
    m = re.search(pat, text or "", re.S)
    if not m:
        return [] if kind == "[" else {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return [] if kind == "[" else {}


def _n(s: str) -> str:
    return re.sub(r"[\s，、。．・　|]", "", s or "")


def pairwise(judges, L: str, en: str, prev: str, nxt: str, x: str, y: str, rng) -> list[str]:
    """Blind votes for x vs y ('x' | 'y' | 'equal'), each judge once in a random A/B order."""
    out = []
    for j in judges:
        flip = rng.random() < 0.5
        a, b = (y, x) if flip else (x, y)
        d = _json(j(PAIR.format(L=L, en=en, prev=prev, nxt=nxt, a=a, b=b)) or "", "{")
        out.append(
            {"A": "y" if flip else "x", "B": "x" if flip else "y"}.get(d.get("better"), "equal")
        )
    return out


def human_better_rows(rows, judges, L: str, seed: int = 0) -> list[dict]:
    """Rows where the reviewer's line differs from ours and BOTH judges prefer it. Rows that already
    carry a verdict (from scripts/eval/ab_vs_human.py) are reused as is."""
    rng = random.Random(seed)
    todo = [r for r in rows if r.get("human") and r.get("ours") and _n(r["human"]) != _n(r["ours"])]

    def judge(r):
        if r.get("verdict"):
            return r
        k = r["i"]
        prev = rows[k - 1]["en"] if k else ""
        nxt = rows[k + 1]["en"] if k + 1 < len(rows) else ""
        v = pairwise(judges, L, r["en"], prev, nxt, r["human"], r["ours"], rng)
        r["verdict"] = "human better" if v.count("x") == len(v) else "other"
        return r

    with ThreadPoolExecutor(8) as ex:
        done = list(ex.map(judge, todo))
    return [r for r in done if r["verdict"] == "human better"]


def propose(kept, invoke, lang: str, notes: str, batch: int = 20) -> list[dict]:
    L = LANG_NAME.get(lang, lang)
    out: list[dict] = []
    for a in range(0, len(kept), batch):
        chunk = kept[a : a + batch]
        rows = "\n".join(
            json.dumps(
                {
                    "id": r["i"],
                    "en": r["en"],
                    "ours": r["ours"],
                    "reviewer": r["human"],
                    "why": r.get("why", ""),
                },
                ensure_ascii=False,
            )
            for r in chunk
        )
        got = _json(invoke(PROPOSE.format(L=L, notes=notes[:6000], rows=rows)))
        by = {r["i"]: r for r in chunk}
        for x in got:
            if isinstance(x, dict) and x.get("kind") in KINDS and x.get("id") in by:
                r = by[x["id"]]
                out.append({**x, "lang": lang, "evidence": [r]})
    return _merge(out)


def _merge(lessons: list[dict]) -> list[dict]:
    """One lesson per glossary term / fixed line (evidence pooled); style notes and examples as is."""
    merged: dict = {}
    rest = []
    for x in lessons:
        key = None
        if x["kind"] == "glossary_term" and x.get("source") and x.get("target"):
            key = ("term", x["source"].strip().lower(), x["target"].strip())
        elif x["kind"] == "fixed_line" and x.get("source") and x.get("target"):
            key = ("fixed", " ".join(x["source"].split()).lower(), x["target"].strip())
        if key is None:
            rest.append(x)
        elif key in merged:
            merged[key]["evidence"] += x["evidence"]
        else:
            merged[key] = x
    return list(merged.values()) + rest


def triage(lessons, glossary) -> tuple[list[dict], list[dict]]:
    """(to_validate, needs_a_person). A glossary term that exists with the same rendering is dropped
    (known: an enforcement gap, not a missing rule); with a DIFFERENT rendering for this language it
    is a conflict for a person; when the term exists but has no rendering for this language, the
    lesson adds one (validated like a new term). A fixed line that already exists is dropped."""
    keep, person = [], []
    terms = {f.lower(): t for t in glossary.terms for f in t.all_source_forms}
    for x in lessons:
        if x["kind"] == "glossary_term":
            t = terms.get(x["source"].strip().lower())
            if t is not None:
                have = t.translations.get(x["lang"])
                if have and _n(have) == _n(x["target"]):
                    continue
                if have:
                    person.append({**x, "conflict": f"glossary has {x['source']!r} -> {have!r}"})
                    continue
                x = {**x, "adds_rendering_to": t.source}
        elif x["kind"] == "fixed_line":
            fixed = ((glossary.style or {}).get(x["lang"], {}) or {}).get("fixed_lines") or {}
            src = " ".join(x["source"].split()).lower()
            if any(" ".join(k.split()).lower() == src for k in fixed):
                continue
        keep.append(x)
    return keep, person


def term_evidence_elsewhere(term: str, lang: str, caches, skip: str, k: int = 3) -> list[dict]:
    """Up to k sentences from OTHER videos' sentence caches (<runs>/<video>/cache.jsonl) that contain
    `term` — so a glossary lesson is validated beyond the one review it came from. Rows carry their own
    (en, ours); validation is with-vs-without, so no human reference is needed."""
    pat = re.compile(rf"\b{re.escape(term)}\b", re.I)
    out: list[dict] = []
    for cp in caches:
        if skip and Path(cp).parent.name == skip:
            continue
        try:
            lines = Path(cp).read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            r = json.loads(line)
            if r.get("lang") == lang and pat.search(r.get("src", "")):
                out.append(
                    {
                        "i": -1 - len(out),
                        "en": r["src"],
                        "ours": r["txt"],
                        "video": Path(cp).parent.name,
                    }
                )
                break
        if len(out) >= k:
            break
    return out


def reviewer_files(reviewed_dir, lang: str) -> dict[str, str]:
    """{video: text} of the reviewers' own finals (<reviewed>/<batch>/<video>.<lang>_Reviewed*.srt,
    every batch; a later review of a video replaces an earlier one)."""
    out = {}
    for p in sorted(Path(reviewed_dir).rglob("*.srt")):
        if f".{lang}" in p.name or f"_{lang}" in p.name:
            out[re.split(r"[._]", p.name)[0]] = p.read_text(encoding="utf-8")
    return out


def reviewer_counterevidence(x, reviewed_dir, skip: str) -> list[str]:
    """For a glossary lesson: where the REVIEWERS' own finals of OTHER videos keep the form the lesson
    would avoid — evidence the reviewer's choice was contextual, not a rule (a reviewer's 'side
    project' -> 個人プロジェクト in one video, while the same reviewer kept サイドプロジェクト 9x elsewhere).
    Reads the reviewer corpus only: the deliverables root also holds our own unreviewed tracks,
    which are no evidence of what a reviewer keeps."""
    hits = []
    avoid = [a for a in (x.get("avoid") or []) if a and len(a) >= 2]
    for video, text in reviewer_files(reviewed_dir, x["lang"]).items():
        if video == skip:
            continue
        for a in avoid:
            n = text.count(a)
            if n:
                hits.append(f"{video}: {a!r} x{n}")
    return hits


# ---- consistent reviewer conventions across videos (judges can't know house terminology) ----

PUNCT = re.compile(r"^[\s，、。．・　|？?！!：:；;“”\"'（）()…—\-「」『』]*$")
_CJK = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


def tokens(s: str, lang: str) -> list[str]:
    s = re.sub(r"\s+", "", s or "")
    if lang.startswith("zh"):
        import jieba

        return [t for t in jieba.lcut(s) if t]
    # ja: script runs (kanji / katakana / hiragana / Latin) — coarse, but a term is one run
    return re.findall(
        r"[A-Za-z0-9.+#\-]+|[\u4e00-\u9fff々]+|[\u30a0-\u30ff]+|[\u3040-\u309f]+|.", s
    )


def replacements(draft: str, human: str, lang: str):
    """Short word replacements draft -> final (1-3 tokens a side, <= 8 chars, the final side CJK);
    the draft side may be Latin (Web -> 网页)."""
    a, b = tokens(draft, lang), tokens(human, lang)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "replace" or not (1 <= i2 - i1 <= 3 and 1 <= j2 - j1 <= 3):
            continue
        x, y = "".join(a[i1:i2]), "".join(b[j1:j2])
        # the tokenizer may cut the two sides differently (Web|开发 vs 网页开发): trim what they share
        while x and y and x[-1] == y[-1]:
            x, y = x[:-1], y[:-1]
        while x and y and x[0] == y[0]:
            x, y = x[1:], y[1:]
        if not x or not y or PUNCT.match(x) or PUNCT.match(y) or len(x) > 8 or len(y) > 8 or x == y:
            continue
        if _CJK.search(y) and (_CJK.search(x) or re.fullmatch(r"[A-Za-z][A-Za-z0-9.+#\-]*", x)):
            yield x, y


def punct_edits(draft: str, human: str):
    """Punctuation / spacing edits draft -> final, with the character before them for context:
    ('吗', '', '？') = a ？ added after 吗. Line breaks are ignored."""
    a, b = (draft or "").replace("\n", ""), (human or "").replace("\n", "")
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        x, y = a[i1:i2], b[j1:j2]
        if op != "equal" and PUNCT.match(x) and PUNCT.match(y) and (x.strip() or y.strip()):
            yield (a[i1 - 1] if i1 else "^"), x, y


def conventions(rows, lang: str, reviewed_dir, min_videos: int = 2, min_rate: float = 0.7):
    """Consistent reviewer changes over `rows` ({video, en, ours, human}, any number of videos):
    (word_changes, punct_changes). A word change counts when made in >= min_videos videos and the
    reviewers changed the draft word in >= min_rate of the lines that had it (kept elsewhere = a
    contextual choice). Each carries its counter-evidence: the draft word in reviewer finals of
    videos where it was never changed."""
    from collections import defaultdict

    ch: dict = defaultdict(lambda: {"videos": set(), "n": 0, "ex": []})
    pu: dict = defaultdict(lambda: {"videos": set(), "n": 0, "ex": []})
    for r in rows:
        if not r.get("ours") or not r.get("human"):
            continue
        if _n(r["ours"]) != _n(r["human"]):
            for x, y in set(replacements(r["ours"], r["human"], lang)):
                c = ch[(x, y)]
                c["videos"].add(r["video"])
                c["n"] += 1
                if len(c["ex"]) < 4:
                    c["ex"].append({k: r[k] for k in ("video", "en", "ours", "human")})
        for before, x, y in set(punct_edits(r["ours"], r["human"])):
            key = (("Q" if before in "吗呢吧么" else "") + ("^" if before == "^" else ""), x, y)
            c = pu[key]
            c["videos"].add(r["video"])
            c["n"] += 1
            if len(c["ex"]) < 3:
                c["ex"].append({k: r[k] for k in ("video", "en", "ours", "human")})
    finals = reviewer_files(reviewed_dir, lang)
    words = []
    for (x, y), c in ch.items():
        if len(c["videos"]) < min_videos:
            continue
        has = [r for r in rows if r.get("ours") and x in _n(r["ours"]) and r.get("human")]
        kept = sum(x in _n(r["human"]) for r in has)
        changed = sum(x not in _n(r["human"]) and y in _n(r["human"]) for r in has)
        rate = changed / max(changed + kept, 1)
        if rate < min_rate:
            continue
        elsewhere = {v: t.count(x) for v, t in finals.items() if v not in c["videos"] and x in t}
        words.append(
            {
                "from": x,
                "to": y,
                "videos": sorted(c["videos"]),
                "changes": c["n"],
                "kept": kept,
                "rate": round(rate, 2),
                "reviewers_keep_elsewhere": elsewhere,
                "examples": c["ex"],
            }
        )
    punct = [
        {
            "after": k[0],
            "from": k[1],
            "to": k[2],
            "videos": sorted(c["videos"]),
            "n": c["n"],
            "examples": c["ex"],
        }
        for k, c in pu.items()
        if len(c["videos"]) >= min_videos and c["n"] >= 3
    ]
    words.sort(key=lambda c: (-len(c["videos"]), -c["changes"]))
    punct.sort(key=lambda c: -c["n"])
    return words, punct


CLASSIFY = """Native {L} subtitle reviewers of a developer YouTube channel changed our machine subtitles. Below
are word changes they made CONSISTENTLY across several videos, each with example lines (English, our
draft, their final). For each change decide:
- "term": the English word or phrase both words render (exactly as it appears in the English lines);
- "kind": "terminology" if the reviewers' word is the house rendering of that English term in general
  (apply it everywhere), "context" if it depends on the sentence, "grammar" if it is a grammatical or
  stylistic adjustment rather than a term (particles, measure words, verb forms);
- "why": max 15 words.
CHANGES:
{items}
Return ONLY a JSON array: [{{"id": <id>, "term": "...", "kind": "terminology|context|grammar", "why": "..."}}]"""


def convention_lessons(words, invoke, lang: str, max_elsewhere: int = 1) -> tuple[list, list]:
    """Word conventions -> (glossary_term lessons, the rest with the reason). A lesson needs
    kind == terminology, an English term found in its example lines, >= 3 changes, and reviewers
    keeping the draft word in at most `max_elsewhere` lines of other videos. Validated by the corpus,
    not by judges: judges endorse only ~1 in 4 reviewer edits and cannot know house terms
    (memory: judge calibration) — the same reason a fixed line must be established in the corpus."""
    if not words:
        return [], []
    L = LANG_NAME.get(lang, lang)
    items = "\n".join(
        json.dumps(
            {
                "id": k,
                "from": c["from"],
                "to": c["to"],
                "examples": [
                    {k2: e[k2] for k2 in ("en", "ours", "human")} for e in c["examples"][:3]
                ],
            },
            ensure_ascii=False,
        )
        for k, c in enumerate(words)
    )
    got = _json(invoke(CLASSIFY.format(L=L, items=items)) or "")
    for x in got if isinstance(got, list) else []:
        if isinstance(x, dict) and isinstance(x.get("id"), int) and 0 <= x["id"] < len(words):
            words[x["id"]].update({k: x.get(k) for k in ("term", "kind", "why")})
    lessons, rest = [], []
    for c in words:
        term = (c.get("term") or "").strip()
        in_en = term and all(re.search(re.escape(term), e["en"], re.I) for e in c["examples"])
        n_else = sum(c["reviewers_keep_elsewhere"].values())
        why = (
            None
            if c.get("kind") == "terminology"
            and in_en
            and c["changes"] >= 3
            and n_else <= max_elsewhere
            else f"kind={c.get('kind')}, term in English={bool(in_en)}, changes={c['changes']}, "
            f"reviewers keep it elsewhere={n_else}"
        )
        if why:
            rest.append({**c, "not_a_lesson": why})
            continue
        lessons.append(
            {
                "kind": "glossary_term",
                "lang": lang,
                "source": term,
                "target": c["to"],
                "avoid": [c["from"]],
                "notes": f"reviewers changed {c['from']} -> {c['to']} {c['changes']}x in "
                f"{len(c['videos'])} videos ({', '.join(c['videos'][:4])}); {c.get('why') or ''}",
                "evidence": [
                    {
                        "i": -1,
                        "en": e["en"],
                        "ours": e["ours"],
                        "human": e["human"],
                        "video": e["video"],
                    }
                    for e in c["examples"]
                ],
                "validation": {
                    "score": {
                        "corpus": f"{c['changes']} changes, rate {c['rate']}, "
                        f"kept elsewhere {n_else}"
                    }
                },
            }
        )
    return lessons, rest


def corpus_support(source: str, lang: str, pairs) -> dict:
    """How the reviewers' own finals translate this EXACT English line: `pairs` = (en, human target,
    video) from the approved corpus (the TM, the A/B alignments). {'n', 'videos', 'top', 'top_n',
    'top_videos'}: the most frequent rendering and its support."""
    key = " ".join(source.split()).lower()
    hits = [(_n(t), t, v) for e, t, v in pairs if " ".join(e.split()).lower() == key and t]
    counts: dict = {}
    for k, t, v in hits:
        counts.setdefault(k, {"t": t, "n": 0, "v": set()})
        counts[k]["n"] += 1
        counts[k]["v"].add(v)
    top = max(counts.values(), key=lambda c: c["n"]) if counts else None
    return {
        "n": len(hits),
        "videos": len({v for _, _, v in hits}),
        "top": top["t"] if top else None,
        "top_n": top["n"] if top else 0,
        "top_videos": len(top["v"]) if top else 0,
    }


def fixed_line_established(x, pairs, min_n: int = 3, min_videos: int = 2, share: float = 0.8):
    """A fixed rendering bypasses the model for every occurrence of the line, and validation can't
    test it (control lines never match the exact sentence) — so it must be ESTABLISHED in the
    reviewers' corpus: rendered this way >= min_n times in >= min_videos videos and in >= share of
    all their renderings of the line. Returns (ok, reason)."""
    c = corpus_support(x["source"], x["lang"], pairs)
    if c["top"] is None:
        return False, "the reviewers' corpus never has this line"
    if _n(c["top"]) != _n(x["target"]):
        return False, f"reviewers mostly render it {c['top']!r} ({c['top_n']}/{c['n']})"
    if c["top_n"] < min_n or c["top_videos"] < min_videos or c["top_n"] < share * c["n"]:
        return False, f"not established: {c['top_n']}/{c['n']} in {c['top_videos']} video(s)"
    return True, f"established: {c['top_n']}/{c['n']} in {c['top_videos']} videos"


def with_lesson(glossary, x):
    """A copy of the glossary with lesson x applied in memory (for validation)."""
    from .glossary import Term

    g = copy.copy(glossary)
    g.terms = list(glossary.terms)
    g.style = {k: dict(v) for k, v in (glossary.style or {}).items()}
    st = g.style.setdefault(x["lang"], {})
    if x["kind"] == "glossary_term":
        g.terms.append(
            Term(
                source=x["source"],
                translations={x["lang"]: x["target"]},
                enforce="soft",
                avoid={x["lang"]: list(x.get("avoid") or [])},
            )
        )
    elif x["kind"] == "style_note":
        for key in ("notes", "notes_conversational"):
            if st.get(key):
                st[key] = st[key] + "\n" + x["text"]
    elif x["kind"] == "fixed_line":
        st["fixed_lines"] = {**(st.get("fixed_lines") or {}), x["source"]: x["target"]}
    return g


def sign_p(wins: int, losses: int) -> float:
    """One-sided exact sign test: P(>= wins of wins+losses | p = 0.5)."""
    import math

    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n


def validate(
    x, translate, judges, rows, *, n_controls: int = 3, seed: int = 0, samples: int = 3, alpha=0.1
) -> dict:
    """Re-translate the lesson's evidence (and `n_controls` unrelated rows) with and without it,
    `samples` times each. `translate(glossary_or_None, row, sample) -> text` (None = the current
    glossary). Sample s with the lesson is judged against sample s without it (both judges, random
    order); a comparison counts as a win/loss only when both judges agree.

    Accepted when the evidence wins beat its losses in a sign test (p < alpha) and the controls show
    no significant harm (the sign test the other way not < alpha). Why: one translation per row is
    noise — a baseline judged against ITSELF swings 3-4 lines (spot A/B, 2026-10-26) — and the old
    rule accepted a lesson on a single winning line. A lesson with one evidence row now needs all of
    its samples to win AND more evidence (glossary terms bring rows from other videos)."""
    rng = random.Random(seed)
    L = LANG_NAME.get(x["lang"], x["lang"])
    ev = (x["evidence"] + x.get("extra_evidence", []))[:6]
    ids = {r["i"] for r in ev}
    pool = [r for r in rows if r["i"] not in ids and r.get("ours")]
    controls = rng.sample(pool, min(n_controls, len(pool)))
    g = x["_glossary"]
    res: dict = {"evidence": [], "controls": []}
    items_all = ev + controls
    jobs = [(r, s_) for s_ in range(samples) for r in items_all]
    with ThreadPoolExecutor(6) as ex:  # the translations are independent model calls
        news = list(ex.map(lambda rs: translate(g, rs[0], rs[1]), jobs))
        olds = list(ex.map(lambda rs: translate(None, rs[0], rs[1]), jobs))
    done: dict = {}
    for (r, s_), n_, o_ in zip(jobs, news, olds):
        done.setdefault(r["i"], []).append((n_, o_))
    tally = {"evidence": [0, 0], "controls": [0, 0]}
    for bucket, items in (("evidence", ev), ("controls", controls)):
        for r in items:
            k = r["i"]
            here = 0 <= k < len(rows)  # cross-video evidence (k < 0) has no neighbours here
            prev = rows[k - 1]["en"] if here and k else ""
            nxt = rows[k + 1]["en"] if here and k + 1 < len(rows) else ""
            xs = ys = 0
            pairs: list[dict] = []
            for new, old in done[k]:
                if _n(new) == _n(old):
                    pairs.append({"same": True})
                    continue
                v = pairwise(judges, L, r["en"], prev, nxt, new, old, rng)
                xs += v.count("x") == len(v)
                ys += v.count("y") == len(v)
                pairs.append({"with": new, "without": old, "votes": v})
            tally[bucket][0] += xs
            tally[bucket][1] += ys
            res[bucket].append({"i": k, "wins": xs, "losses": ys, "samples": pairs})
    (w, l_), (cw, cl) = tally["evidence"], tally["controls"]
    p_ev, p_harm = sign_p(w, l_), sign_p(cl, cw)
    res["accepted"] = p_ev < alpha and p_harm >= alpha
    res["score"] = {
        "wins": w,
        "losses": l_,
        "p": round(p_ev, 3),
        "control_wins": cw,
        "control_losses": cl,
        "p_controls_worse": round(p_harm, 3),
        "samples": samples,
    }
    return res


# ---- writing accepted lessons (text insertion: the glossary's comments are part of its value) ----


def _yq(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def apply_lessons(accepted, glossary_path, examples_dir, tag: str) -> list[str]:
    """Insert accepted lessons into glossary.yaml (terms before `style:`, style notes at the end of the
    language's `notes`, fixed lines under `fixed_lines`) and editor examples into
    editor_examples.<lang>.jsonl. Returns a description of each change."""
    p = Path(glossary_path)
    s = p.read_text(encoding="utf-8")
    done = []
    for x in [x for x in accepted if x.get("adds_rendering_to")]:
        m = re.search(rf'^  - source: "?{re.escape(x["adds_rendering_to"])}"?\n', s, re.M)
        if m:
            s = (
                s[: m.end()]
                + f"    {x['lang']}: {_yq(x['target'])}   # learned {tag}\n"
                + s[m.end() :]
            )
            done.append(f"term {x['adds_rendering_to']!r} + {x['lang']} {x['target']!r}")
    terms = [x for x in accepted if x["kind"] == "glossary_term" and not x.get("adds_rendering_to")]
    if terms:
        block = f"  # --- learned from review ({tag}) ---\n"
        for x in terms:
            block += f"  - source: {_yq(x['source'])}\n    {x['lang']}: {_yq(x['target'])}\n    enforce: soft\n"
            if x.get("avoid"):
                block += (
                    f"    avoid:\n      {x['lang']}: {json.dumps(x['avoid'], ensure_ascii=False)}\n"
                )
            block += f"    notes: {_yq((x.get('notes') or '') + f' (learned {tag})')}\n"
            done.append(f"term {x['source']!r} -> {x['target']!r} [{x['lang']}]")
        i = s.index("\nstyle:")
        s = s[:i] + "\n" + block.rstrip("\n") + s[i:]
    for x in [x for x in accepted if x["kind"] == "style_note"]:
        m = re.search(
            rf"^  {re.escape(x['lang'])}:\n(?:    .*\n|\n)*?    notes: >-\n((?:      .*\n)+)",
            s,
            re.M,
        )
        if not m:
            continue
        end = m.end(1)
        s = s[:end] + f"      LEARNED ({tag}): {x['text']}\n" + s[end:]
        done.append(f"style note [{x['lang']}]: {x['text'][:80]}")
    for x in [x for x in accepted if x["kind"] == "fixed_line"]:
        m = re.search(
            rf"^  {re.escape(x['lang'])}:\n((?:    .*\n|\n)*?)    fixed_lines:\n", s, re.M
        )
        line = f"      {_yq(x['source'])}: {_yq(x['target'])}   # learned {tag}\n"
        if m:
            s = s[: m.end()] + line + s[m.end() :]
        else:
            m2 = re.search(rf"^  {re.escape(x['lang'])}:\n", s, re.M)
            if not m2:
                continue
            s = s[: m2.end()] + "    fixed_lines:\n" + line + s[m2.end() :]
        done.append(f"fixed line [{x['lang']}] {x['source']!r} -> {x['target']!r}")
    p.write_text(s, encoding="utf-8")
    for x in [x for x in accepted if x["kind"] == "editor_example"]:
        r = x["evidence"][0]
        ep = Path(examples_dir) / f"editor_examples.{x['lang']}.jsonl"
        rec = {
            "video": r.get("video", tag),
            "en": r["en"],
            "before": r["ours"],
            "after": r["human"],
            "label": x.get("label", ""),
        }
        with ep.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done.append(f"editor example [{x['lang']}]: {x.get('label', '')}")
    return done
