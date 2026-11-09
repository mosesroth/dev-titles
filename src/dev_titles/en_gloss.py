"""MEANING NOTES: resolve idioms and ambiguous phrases in the ENGLISH once, before any language is translated.

Why (A/B of a fresh translation vs the human-reviewed finals): the same three lines were wrong in BOTH zh and ja,
so the cause was reading the English, not writing the target:
  "Bring it home."                       = make it concrete (for developers), not "wrap up"
  "two ways of having local models"      = OBTAINING them (shipped vs downloaded), not using them
  "I was very out of date until I got it" = my old PHONE was outdated, not me
One cached LLM pass over the transcript (with the video brief) lists such traps; each note is attached
to its sentence's translation prompt and shown to the native editor next to the English.

A note is an instruction both languages follow, so a WRONG note is worse than none. On one
video one model's notes caused three defects in both languages: "Take the jump from 0.72 to
0.83" glossed as "for example, not an instruction" (it means: make the upgrade), and "stays navigable"
glossed as "usable with the remote" (both translations then added "the remote"). So, with `verify`, a
model of another family checks every note: agreed notes are kept; DISPUTED ones are dropped and held
for the operator at the English sign-off (cli._en_review), whose answer (`dev_titles set-meaning`)
then overrides any model note for that sentence.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PROMPT = """You prepare an English interview/talk transcript for translators into Chinese and Japanese.
List the phrases where a careful translator's natural, literal reading would be WRONG — only real traps,
the most dangerous first, AT MOST {cap}:
- idioms and figurative phrases ("bring it home", "boil the ocean", "a huge win") — give the meaning HERE;
- an everyday word in a less common sense ("two ways of having local models" = obtaining);
- an elliptical line whose subject / referent is not what it literally says ("I was very out of date
  until I got it" = his old phone was outdated);
- a reply whose function is easy to miss ("Okay, okay." = taking in new information).
The meaning must say only what the speaker means: spell out the sense the channel's domain implies, but
never add a fact the speaker did not say.
Do NOT list: phrases any translator renders correctly anyway (common phrasal verbs like "spin up", "ran
across", "wrap up"), ordinary sentences, technical terms with one clear meaning, names, false starts.

VIDEO BRIEF: {brief}

SENTENCES (id | text):
{rows}

Return ONLY a JSON array: [{{"id": n, "phrase": "<exact words>", "meaning": "<what it means here, max 15 words>"}}]; [] if none."""


VERIFY = """What does a phrase in an English developer-video transcript mean? Read it in context.
VIDEO BRIEF: {brief}
Before: {prev}
SENTENCE: {sent}
After: {nxt}
PHRASE: {phrase}
Answer ONLY JSON: {{"meaning": "<what the speaker means here, max 15 words>"}}"""

SAME = """Two readings of the phrase "{phrase}" in: "{sent}"
Before: {prev}
After: {nxt}
A: {a}
B: {b}
How do they relate? "same": they say the same thing (wording aside). "a_adds" / "b_adds": they agree,
but one adds detail from the context (a reason, a referent from a nearby sentence). "conflict": they
give DIFFERENT meanings, so a translator would write a different line depending on which is right.
If the speaker explains the phrase in the next sentence, read it that way: that is not ambiguity.
Answer ONLY JSON: {{"relation": "same" | "a_adds" | "b_adds" | "conflict"}}"""


def _relation(judges, phrase, sent, prev, nxt, a, b):
    """'same' | 'a_adds' | 'b_adds' | 'conflict' | None. A conflict needs every judge that answers
    (both model families) to call it one; one model's strictness sent agreeing readings to the
    operator (phrases like 'universal shipping' or 'too much token burning', where the two
    readings in fact agreed)."""
    rels = []
    for j in judges:
        if j is None:
            continue
        try:
            raw = j(SAME.format(phrase=phrase, sent=sent, prev=prev, nxt=nxt, a=a, b=b)) or ""
            m = re.search(r"\{.*\}", raw, re.S)
            r = (json.loads(m.group(0)) if m else {}).get("relation")
        except Exception:
            r = None
        if r in ("same", "a_adds", "b_adds"):
            return r
        rels.append(r)
    # a judge with no usable answer counts for neither side: a conflict stands unless a judge
    # positively says the readings agree
    return "conflict" if "conflict" in rels else None


def _apply_operator(notes: dict, d: dict, sents) -> dict:
    """The operator's answers override every model note for their sentence. Keyed by the sentence
    TEXT, so an EN edit that renumbers sentences doesn't move an answer to the wrong line."""
    out = dict(notes)
    ops = d.get("operator") or {}
    for k, s in enumerate(sents):
        if s.text in ops:
            out[k] = [{"phrase": s.text, "meaning": ops[s.text], "by": "operator"}]
    return out


def disputed(cache_path, sentences=None) -> list[dict]:
    """Notes the two models disagreed on that the operator has not answered yet. With `sentences`
    (the current EN's), a question about a sentence the EN no longer has is moot: the line was
    corrected since ("getting stuff done into NWA" -> "in an end-to-end way").
    """
    p = Path(cache_path) if cache_path else None
    if not (p and p.exists()):
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    done = set(d.get("operator") or {})
    out = [x for x in d.get("disputed", []) if x.get("sentence") not in done]
    if sentences is not None:
        have = {" ".join(str(s).split()) for s in sentences}
        out = [x for x in out if " ".join(str(x.get("sentence", "")).split()) in have]
    return out


def recheck_disputes(cache_path, verify, sents=None, first=None) -> list[str]:
    """Re-judge stored disputes with the current comparison (SAME): readings that agree — one
    adding context — become a kept note with the plainer reading; only conflicts stay questions.
    For disputes recorded under the old same/different comparison. Returns the settled sentences."""
    p = Path(cache_path)
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    keep, settled = [], []
    for x in d.get("disputed", []):
        texts = [str(getattr(t, "text", t)) for t in (sents or [])]
        k = texts.index(x["sentence"]) if x["sentence"] in texts else None
        rel = _relation(
            [verify, first],
            x["phrase"],
            x["sentence"],
            texts[k - 1] if k else "",
            texts[k + 1] if k is not None and k + 1 < len(texts) else "",
            x["a"],
            x["b"],
        )
        if rel in ("same", "a_adds", "b_adds"):
            plain = x["b"] if rel == "a_adds" else x["a"]
            d.setdefault("notes", {}).setdefault(str(x["id"]), []).append(
                {"phrase": x["phrase"], "meaning": plain}
            )
            settled.append(x["sentence"])
        else:
            keep.append(x)
    if settled:
        d["disputed"] = keep
        p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return settled


def set_operator_meaning(cache_path, sentence: str, meaning: str) -> None:
    p = Path(cache_path)
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    d.setdefault("operator", {})[sentence] = meaning
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def _verify(notes, sents, verify, brief, first=None):
    kept: dict[int, list[dict]] = {}
    disputes: list[dict] = []
    for k, items in sorted(notes.items()):
        for n in items:
            # BLIND: the second model gives its own reading without seeing the first one's (shown the
            # first reading, it agreed with "for example" for "Take the jump…" in one run of two), then
            # a separate call compares the two readings
            v: dict = {}
            try:
                raw = (
                    verify(
                        VERIFY.format(
                            brief=brief or "(none)",
                            prev=sents[k - 1].text if k else "",
                            sent=sents[k].text,
                            nxt=sents[k + 1].text if k + 1 < len(sents) else "",
                            phrase=n["phrase"],
                        )
                    )
                    or ""
                )
                m = re.search(r"\{.*\}", raw, re.S)
                other = str((json.loads(m.group(0)) if m else {}).get("meaning", "")).strip()
                if other:
                    rel = _relation(
                        [verify, first],
                        n["phrase"],
                        sents[k].text,
                        sents[k - 1].text if k else "",
                        sents[k + 1].text if k + 1 < len(sents) else "",
                        n["meaning"],
                        other,
                    )
                    # only a real conflict goes to the operator: readings that agree, one with
                    # extra context ("the biggest obstacle" vs "...to letting AI agents buy"), keep
                    # the reading that adds nothing — the translator sees the line's neighbours
                    # anyway, and a detailed note invites the line to say more than the speaker did
                    # (on one video both of the questions raised were this kind)
                    if rel in ("same", "a_adds", "b_adds", "conflict"):
                        v = {"agree": rel != "conflict", "meaning": other}
                        if rel == "a_adds":
                            n = {**n, "meaning": other}
            except Exception:
                v = {}
            if v.get("agree") is True:
                kept.setdefault(k, []).append(n)
            elif v.get("agree") is False:
                disputes.append(
                    {
                        "id": k,
                        "sentence": sents[k].text,
                        "start": getattr(sents[k], "start", 0.0),
                        "phrase": n["phrase"],
                        "a": n["meaning"],
                        "b": str(v.get("meaning", "")),
                    }
                )
            # no verdict (judge unavailable / unparsable): keep the note, as before verification
            else:
                kept.setdefault(k, []).append(n)
    return kept, disputes


def meaning_notes(
    sents, invoke, cache_path=None, brief: str = "", batch: int = 60, verify=None
) -> dict[int, list[dict]]:
    """{sentence index: [{'phrase', 'meaning'}]} — cached by the sentence texts + brief. With `verify`
    (a judge of another model family), only notes both models agree on are returned; disputed ones
    are stored in the cache file's "disputed" list for the operator. Operator answers ("operator")
    always win."""
    key = hashlib.sha1(
        ("\x00".join(s.text for s in sents) + "\x01" + brief + ("\x02v" if verify else "")).encode(
            "utf-8"
        ),
        usedforsecurity=False,
    ).hexdigest()[:16]
    p = Path(cache_path) if cache_path else None
    prev_d: dict = {}
    if p and p.exists():
        try:
            prev_d = json.loads(p.read_text(encoding="utf-8"))
            if prev_d.get("key") == key:
                return _apply_operator(
                    {int(k): v for k, v in prev_d.get("notes", {}).items()}, prev_d, sents
                )
        except (json.JSONDecodeError, OSError):
            prev_d = {}
    notes: dict[int, list[dict]] = {}
    # REUSE: a sentence whose text is unchanged keeps its verified notes and its open dispute, so a
    # small EN edit (four name fixes in one video) doesn't re-ask every sentence and turn up
    # new disagreements on lines nobody touched. Only changed sentences go back to the models.
    side = hashlib.sha1(
        (brief + ("\x02v" if verify else "")).encode("utf-8"), usedforsecurity=False
    )
    side_key = side.hexdigest()[:16]
    old_texts = prev_d.get("texts") if prev_d.get("side") == side_key else None
    reused_disputes: list[dict] = []
    todo = set(range(len(sents)))
    if isinstance(old_texts, list):
        # duplicate lines ('Okay.') are matched in order: the n-th occurrence of a text now is the
        # n-th one before, so each keeps its own note and disputes
        where: dict = {}
        for j, t in enumerate(old_texts):
            where.setdefault(t, []).append(j)
        old_notes = prev_d.get("notes", {}) or {}
        old_disp: dict = {}
        for d in prev_d.get("disputed", []) or []:  # a sentence can hold several disputes
            old_disp.setdefault(d.get("id"), []).append(d)
        seen: dict = {}
        for k, snt in enumerate(sents):
            n = seen.get(snt.text, 0)
            seen[snt.text] = n + 1
            olds = where.get(snt.text) or []
            if n >= len(olds):
                continue
            j = olds[n]
            todo.discard(k)
            if str(j) in old_notes:
                notes[k] = old_notes[str(j)]
            reused_disputes += [{**d, "id": k} for d in old_disp.get(j, [])]
    for s0 in range(0, len(sents), batch):
        if not any(s0 <= k < s0 + batch for k in todo):
            continue
        # a little overlap on each side so a phrase is read in context
        lo, hi = max(0, s0 - 3), min(len(sents), s0 + batch + 3)
        rows = "\n".join(f"{k} | {sents[k].text}" for k in range(lo, hi))
        try:
            raw = (
                invoke(
                    PROMPT.format(brief=brief or "(none)", rows=rows, cap=max(2, (hi - lo) // 6))
                )
                or ""
            )
            found = json.JSONDecoder().raw_decode(raw[raw.index("[") :])[0]
        except (ValueError, json.JSONDecodeError, TypeError):
            continue
        except Exception:
            continue
        for f in found:
            try:
                k = int(f["id"])
            except (KeyError, TypeError, ValueError):
                continue
            ph, me = str(f.get("phrase", "")).strip(), str(f.get("meaning", "")).strip()
            if not (s0 <= k < min(s0 + batch, len(sents)) and ph and me) or k not in todo:
                continue
            if re.sub(r"\W", "", ph.lower()) not in re.sub(r"\W", "", sents[k].text.lower()):
                continue  # the phrase must really be in that sentence
            notes.setdefault(k, []).append({"phrase": ph, "meaning": me})
    disputes: list[dict] = list(reused_disputes)
    fresh = {k: v for k, v in notes.items() if k in todo}
    if verify is not None and fresh:
        kept, new_disputes = _verify(fresh, sents, verify, brief, first=invoke)
        notes = {**{k: v for k, v in notes.items() if k not in todo}, **kept}
        disputes += new_disputes
    if p:
        p.write_text(
            json.dumps(
                {
                    "key": key,
                    "side": side_key,
                    "texts": [snt.text for snt in sents],
                    "notes": {str(k): v for k, v in notes.items()},
                    "disputed": disputes,
                    "operator": prev_d.get("operator") or {},  # answers survive a re-generation
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
    return _apply_operator(notes, {"operator": prev_d.get("operator") or {}}, sents)


def meaning_note(i: int, notes) -> str:
    """Prompt suffix for sentence i ('' when it has no note)."""
    items = (notes or {}).get(i) or (notes or {}).get(str(i)) or []
    if not items:
        return ""
    body = "; ".join(f'"{n["phrase"]}" here means: {n["meaning"]}' for n in items)
    return f"\nMEANING NOTE for this line (translate the meaning, not the literal words): {body}"


def meaning_tag(i: int, notes) -> str:
    """Short inline form for the native editor's line table."""
    items = (notes or {}).get(i) or (notes or {}).get(str(i)) or []
    return (
        " [means: " + "; ".join(f'{n["phrase"]} = {n["meaning"]}' for n in items) + "]"
        if items
        else ""
    )
