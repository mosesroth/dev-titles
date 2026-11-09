"""Names and slips in the EN, checked against what the video itself says about them.

The EN proofread (transcribe_qe.en_review) sees only numbered cue text, so it can't know that a guest
spelled 'Naeem' in the description was heard 'Nahim', that 'Thanks, Cutter' was meant for the host
Chris, or that 'Gazelle Bardio' is the guest 'Ghizlan Bardiowil' — 'Gazelle' is a real word. Nor can it
tell that 'that local model clip' was the speaker's slip for her own term 'local model cliff' (the
video's title); the meaning notes knew ("mis-said 'local model cliff'") but only the translators
read them. All four kinds have shipped in signed-off EN.

Two kinds, each applied only through a gate in code, and only when a model of the other family
agrees; anything else is held for the operator with the proposed fix:
  - name: a person / company / product name spelled differently from the title, description, brief
    or known names. Gate: every word of the fix is written in that evidence.
  - slip: the speaker said a wrong word for a term of their own. Gate: the fix sounds like what was
    said (caption_tiebreak.sounds_like) and the intended form is in the title / description or in
    another line of the video. A factual misspeak ('Q division of MI5') stays as said: its 'fix' is
    not established anywhere in the video.
"""

from __future__ import annotations

import json
import re

ASK = """You check an English subtitle script of a video against what the video itself says about its
names and terms. Find ONLY:
- "name": a person, company or product name written differently from how the title, description or
  brief spell it, or one the brief says was mis-transcribed (give the correct name);
- "slip": the speaker clearly said a wrong word for a term of their own that the video uses correctly
  elsewhere or in its title ("that local model clip" when the talk is about "the local model cliff").
Never flag ordinary words, grammar, style, or a fact the speaker got wrong.
Title: {title}
Description: {description}
Brief: {brief}
Known names: {known}
Lines (id | text):
{rows}
Answer ONLY JSON: [{{"id": <line id>, "type": "name"|"slip", "old": "<exact words in the line>",
"new": "<the correction>", "evidence": "<where the correct form appears>"}}] — [] if none."""

VERIFY = """A subtitle line of a video: {line!r}
Proposed correction: {old!r} -> {new!r} ({kind}; evidence: {evidence})
Title: {title}
Description: {description}
Is the correction certainly right — the speaker meant {new!r}, and {old!r} is a mis-transcription or a
slip of the tongue, not what they intended? Answer ONLY JSON: {{"ok": true|false}}"""

_MISSAID = re.compile(r"mis-?s(?:aid|poke)[^'\"“‘]*['\"“‘]([^'\"”’]{2,60})['\"”’]", re.I)


def _w(s: str) -> list[str]:
    """Lowercase words; a quote or possessive doesn't make a different word ('kakar', jon's -> jon)."""
    out = []
    for t in re.findall(r"[a-z0-9][a-z0-9'’-]*", (s or "").lower()):
        t = re.sub(r"['’]s$", "", t).strip("'’-")
        if t:
            out.append(t)
    return out


def _has_words(words: list[str], text: str) -> bool:
    t = " " + " ".join(_w(text)) + " "
    return bool(words) and f" {' '.join(words)} " in t


def _json_list(raw: str) -> list:
    s, e = (raw or "").find("["), (raw or "").rfind("]")
    try:
        v = json.loads(raw[s : e + 1]) if s != -1 and e != -1 else []
    except Exception:
        return []
    return v if isinstance(v, list) else []


def _ok(raw: str) -> bool:
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        return bool(m and json.loads(m.group(0)).get("ok") is True)
    except Exception:
        return False


def _said_mistranscribed(old: str, new: str, brief: str) -> bool:
    """The brief says, in one sentence, that `old` is a mis-transcription of `new` ("the 'Otto' in the
    closing thanks is a transcription error for 'Idris'"; "'Cutter' is a mis-transcription of his
    name")."""
    for sent in re.split(r"(?<=[.;])\s+", brief or ""):
        low = sent.lower()
        # a hedged claim ('likely a mistranscription of James') is a guess, not evidence
        hedged = re.search(r"\b(likely|probably|possibly|perhaps|may|might|could|seems?)\b", low)
        if (
            not hedged
            and re.search(r"mis-?transcri|transcription error|misheard", low)
            and _has_words(_w(old), sent)
        ):
            if _has_words(_w(new), sent) or any(_has_words([w], sent) for w in _w(new)):
                return True
    return False


def gate(
    kind: str,
    old: str,
    new: str,
    line: str,
    *,
    evidence: str,
    others: str,
    brief: str = "",
    next_line: str = "",  # (reserved: the cue after; a name split across it is finished in check)
) -> str:
    """'' when the fix passes its gate, else why not. `evidence`: title + description + brief + known
    names; `others`: the video's other lines; `next_line`: the cue after this one."""
    from .caption_tiebreak import sounds_like

    ow, nw = _w(old), _w(new)
    if not ow or not nw or old.split() == new.split():
        return "no change"
    if not _has_words(ow, line):
        return f"{old!r} is not in the line"
    if len(ow) > 4 or len(nw) > 4:
        return "too long for a name or slip fix"
    if kind == "name":
        if ow == nw:  # capitalization only ('Jetflix' -> 'JetFlix'): the evidence must spell it
            return (
                ""
                if new in evidence
                else f"{new!r} isn't spelled that way in the title/description/brief"
            )
        changed = [w for w in nw if w not in ow]
        missing = [w for w in changed if not _has_words([w], evidence)]
        if missing:
            return f"not written in the title/description/brief: {' '.join(missing)}"
        # a fix that doesn't sound like what was heard isn't a transcription error but a different
        # referent or a misspeak ('Expo Dev Awards' for the Expo App Awards at expo.dev/awards;
        # 'Bumbling Barci' vs the chapter's 'Bumbling Bastard') — unless the brief says it is one
        if not sounds_like(old, new) and not _said_mistranscribed(old, new, brief):
            return "doesn't sound like what was heard — a different name or a misspeak? the operator decides"
        return ""
    if kind == "slip":
        if not sounds_like(old, new):
            return "doesn't sound like what was said"
        if not (_has_words(nw, evidence) or _has_words(nw, others)):
            return "the intended form isn't used anywhere else in the video"
        return ""
    return f"unknown kind {kind!r}"


def note_slips(notes: dict, sentences) -> list[dict]:
    """Slip candidates from verified meaning notes ('mis-said "local model cliff"'), as
    {sentence, old, new}."""
    out = []
    for k, items in (notes or {}).items():
        for n in items or []:
            m = _MISSAID.search(str(n.get("meaning", "")))
            if m and n.get("phrase"):
                i = int(k)
                if 0 <= i < len(sentences):
                    out.append(
                        {"sentence": sentences[i].text, "old": n["phrase"], "new": m.group(1)}
                    )
    return out


def _replace(line: str, old: str, new: str) -> str:
    """`old` -> `new` in `line`, whole words, case-insensitive, keeping the rest verbatim."""
    pat = r"(?<![\w'’-])" + r"[\s,]*".join(re.escape(w) for w in old.split()) + r"(?![\w'’-])"
    return re.sub(pat, lambda _m: new, line, count=1, flags=re.I)  # `new` literally (no \\1)


def check(
    en,
    judges,
    *,
    title: str = "",
    description: str = "",
    brief: str = "",
    known: str = "",
    notes: dict | None = None,
    batch: int = 60,
):
    """(en, applied, held). `judges`: [finder, verifier] — model callables of different families (the
    finder proposes, the verifier must agree); with one, nothing is auto-applied. Changes `en` in
    place (a cue that overflows is split by caption_tiebreak._place)."""
    from .caption_tiebreak import _place
    from .resegment import merge_en_to_sentences

    find, verify = judges[0], (judges[1] if len(judges) > 1 else None)
    evidence = " ".join((title, description, brief, known))
    cands: list[dict] = []
    for s in range(0, len(en.cues), batch):
        rows = "\n".join(
            f"{k} | {' '.join(en.cues[k].lines)}" for k in range(s, min(s + batch, len(en.cues)))
        )
        try:
            got = _json_list(
                find(
                    ASK.format(
                        title=title or "(none)",
                        description=description[:1500] or "(none)",
                        brief=brief[:1200] or "(none)",
                        known=known[:800] or "(none)",
                        rows=rows,
                    )
                )
            )
        except Exception:
            got = []
        cands += [c for c in got if isinstance(c, dict) and isinstance(c.get("id"), int)]
    # slips the meaning notes found: located in the cue that holds the mis-said phrase
    for n in note_slips(notes or {}, merge_en_to_sentences(en)):
        for k, c in enumerate(en.cues):
            if _has_words(_w(n["old"]), " ".join(c.lines)):
                cands.append(
                    {
                        "id": k,
                        "type": "slip",
                        "old": n["old"],
                        "new": n["new"],
                        "evidence": "meaning note",
                    }
                )
                break
    applied, held, seen = [], [], set()
    applied_at: dict = {}  # cue -> the name written there (for a name split across cues)
    current: dict = {}  # cue -> its text with the fixes so far
    # top-down, so a split name's head is fixed before its tail; deletions after the fixes
    # (a meaning-note slip first, so the finder's copy of the same slip is the duplicate)
    cands.sort(
        key=lambda c: (
            c["id"],
            not str(c.get("new", "")).strip(),
            c.get("evidence") != "meaning note",
        )
    )
    for c in cands:
        k, kind = c["id"], str(c.get("type", ""))
        old, new = str(c.get("old", "")).strip(), str(c.get("new", "")).strip()
        if not (0 <= k < len(en.cues)) or (k, old.lower()) in seen:
            continue
        seen.add((k, old.lower()))
        # a second fix in the same cue applies to the first one's result
        line = current.get(k) or " ".join(" ".join(en.cues[k].lines).split())
        others = " ".join(" ".join(x.lines) for j, x in enumerate(en.cues) if j != k)
        m, sec = divmod(int(en.cues[k].start), 60)
        item = {
            "ts": f"{m:02d}:{sec:02d}",
            "kind": f"en-{kind}",
            "was": line,
            "old": old,
            "new": new,
            "evidence": str(c.get("evidence", ""))[:160],
        }
        prev = applied_at.get(k - 1, "")
        if kind == "name" and not new and prev:
            # the rest of a name split across cues ('…by Gazelle Badio' / 'Zamani, the SVP'): drop it
            # when the name corrected in the cue before ends with it ('Ghizlan Bardiowil')
            ok_tail = any(w.endswith("".join(_w(old))) for w in _w(prev)) and len(_w(old)) == 1
            why = "" if ok_tail else "a deletion that doesn't finish the previous cue's name"
        else:
            why = gate(
                kind,
                old,
                new,
                line,
                evidence=evidence,
                others=others,
                brief=brief,
                next_line=" ".join(en.cues[k + 1].lines) if k + 1 < len(en.cues) else "",
            )
        if not why and kind == "slip" and c.get("evidence") != "meaning note":
            # a slip needs the meaning notes' two-family "mis-said" too: a finder + verifier pair took
            # 'AI-generated and human-refined' for a slip of 'human-verified' — a word
            # choice, not a slip
            why = "a slip the meaning notes didn't flag — the operator decides"
        tail = kind == "name" and not new  # completes a name already verified in the cue before
        if not why and verify is None and not tail:
            why = "no second model to confirm it"
        if not why and verify is not None and not tail:
            try:
                ok = _ok(
                    verify(
                        VERIFY.format(
                            line=line,
                            old=old,
                            new=new,
                            kind=kind,
                            evidence=item["evidence"],
                            title=title or "(none)",
                            description=description[:800] or "(none)",
                        )
                    )
                )
            except Exception:
                ok = False
            why = "" if ok else "the second model didn't confirm it"
        fixed = _replace(line, old, new) if not why else line
        if not new and fixed != line:
            fixed = re.sub(r"^[\s,;]+", "", " ".join(fixed.split()))
        if not why and fixed == line:
            why = f"{old!r} not found as whole words"
        if not why and not fixed.strip():
            why = "it would leave the cue empty"
        if why:
            held.append(
                {
                    "ts": item["ts"],
                    "kind": item["kind"],
                    "detail": f"[cue {k}] {old!r} -> {new!r}? ({item['evidence']}) — not applied: {why}",
                }
            )
            continue
        current[k] = fixed
        applied.append({**item, "now": fixed})
        applied_at[k] = new
        # a name that runs into the next cue ('the Acme App' / 'Store, why now?' -> 'Acme Appstore'):
        # its tail there is dropped, as a split name's is ('Gazelle Badio' / 'Zamani, the SVP')
        nxt = _w(" ".join(en.cues[k + 1].lines))[:1] if k + 1 < len(en.cues) else []
        last_new, last_old = (_w(new) or [""])[-1], (_w(old) or [""])[-1]
        if (
            kind == "name"
            and nxt
            and last_new != last_old
            and last_new.endswith(nxt[0])
            and len(nxt[0]) >= 3
            and _w(line)[-1:] == [last_old]
            and not any(c2["id"] == k + 1 and not str(c2.get("new", "")).strip() for c2 in cands)
        ):
            # the validated word itself, as written (any line of the cue may hold it)
            nl = " ".join(en.cues[k + 1].lines).split()
            first = nl[0].strip(",.;:!?") if nl else nxt[0]
            cands.append(
                {
                    "id": k + 1,
                    "type": "name",
                    "old": first,
                    "new": "",
                    "evidence": "rest of the name",
                }
            )
    for k, fixed in sorted(current.items(), key=lambda x: -x[0]):
        _place(en, k, fixed)
    if current:
        en.renumber()
    return en, applied, held


def section(applied: list[dict]) -> list[str]:
    """en_review.md lines for the fixes applied (they are part of the EN sign-off)."""
    if not applied:
        return []
    return [
        f"## Names and slips corrected ({len(applied)}; the video's title/description/brief as evidence, "
        "two model families agreed)",
        "",
        "| Time | Kind | Was | Now | Evidence |",
        "|---|---|---|---|---|",
        *[
            f"| {a['ts']} | {a['kind']} | {a['was']!r} | {a['now']!r} | {a['evidence']} |"
            for a in applied
        ],
        "",
    ]
