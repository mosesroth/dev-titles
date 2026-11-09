"""YouTube's auto-captions as a tiebreaker for the EN lines the review held as likely mishears.

The EN review flags lines that look misheard ("getting stuff done into NWA for customers") and holds
them for the operator. For those lines only, two models of different families each read our line next
to what YouTube's auto-captions and the 2nd ASR heard at that moment, with the neighbouring lines and
the video's brief, and say what the speaker most likely said. The fix is applied only when both
return the same line, every new word is one the captions or the 2nd ASR actually heard, and nothing
clean-read removed (fillers, stutters) comes back. Anything else stays held, with the caption reading
added for the operator.

Why not the old consult (transcribe_qe.autosub_consult, now suggestions only): it let the captions
and the 2nd ASR OUTVOTE the primary wherever they agreed — two weaker recognizers that share mishears
('Vantoo' -> 'the view', 'Flow Writer' -> 'floor writer') and keep the fillers clean-read deletes. Here
the captions are evidence a model weighs in context, on lines already suspected, not a vote.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

KINDS = (
    "en-garbled",
    "en-mishear",
    "en-grammar",
    "en-terminology",
    "en-spelling",
    "en-punctuation",
)
FILLERS = {"um", "uh", "er", "ah", "hmm", "like", "mm"}
# a phrase starts at these: a cue split before one reads naturally ("…in an end-to-end way / for customers…")
BREAK_BEFORE = {
    "for",
    "and",
    "but",
    "so",
    "because",
    "which",
    "that",
    "when",
    "where",
    "with",
    "or",
    "if",
}

ASK = """You check a single subtitle line of an English video transcript that a reviewer flagged as a likely
speech-recognition error. Our transcript is a CLEAN READ: fillers, stutters and false starts were
removed on purpose. YouTube's auto-captions and a second recognizer heard the same moment; both are
verbatim and weaker (they mishear names too), so treat them as evidence, not as the answer.
Video: {brief}
Known names and terms in this video: {known}
Previous line: {prev}
OUR LINE: {line}
Next line: {nxt}
Reviewer's note: {note}
YouTube auto-captions around this moment: {caps}
Second recognizer around this moment: {second}
What did the speaker most likely say in OUR LINE? Correct only misheard words, keep everything else
exactly (no fillers, no stutters, names as the video spells them, no words from the neighbouring
lines). If our line is right, keep it. Many speakers are non-native: when a recognizer heard a word
that makes sense in context, even as unusual phrasing ("we ship in smaller pace"), prefer it to a
more idiomatic word nobody heard ("smaller parts") — the most sensible reading of what was HEARD.
If the reviewer's note is about punctuation, fix only the punctuation, the way the captions' phrasing
splits the sentences. Write version numbers the way the video's brief writes them ("0.83").
Answer ONLY JSON: {{"line": "<our line, corrected or unchanged>", "changed": true|false}}"""


def _k(w: str) -> str:
    """A word's comparison key; a number by its digits without leading zeros, so the written version
    '0.83' matches the spoken 'React Native 83' the captions heard."""
    k = re.sub(r"[^a-z0-9]", "", w.lower())
    return (k.lstrip("0") or "0") if k.isdigit() else k


def caption_window(sub, s: float, e: float, pad: float = 1.0) -> str:
    """Caption text around [s, e], rolling repeats and caption markers ('>>', '[music]') dropped."""
    lines: list[str] = []
    for c in sub.cues:
        if c.end > s - pad and c.start < e + pad:
            for ln in c.lines:
                ln = re.sub(r"\[[^\]]*\]|>>|&nbsp;", " ", ln)
                ln = " ".join(ln.split())
                if ln and (not lines or lines[-1] != ln):
                    lines.append(ln)
    return " ".join(lines)


def asr_window(transcript, s: float, e: float, pad: float = 1.0) -> str:
    if transcript is None:
        return ""
    return " ".join(w.text for w in transcript.words if s - pad <= w.start <= e + pad).strip()


def _cue_of(en, item: dict, window: float = 8.0) -> int | None:
    """The cue a held item is about. The review's '[cue N]' and mm:ss drift from the cues (cues
    are split / re-timed after the review; 'One thing that brought up' was held at 06:20 for a
    cue starting 06:23 and never checked, 2026-11-05): prefer a cue near the mm:ss that contains
    the words the note quotes, else the '[cue N]' if it is near, else the nearest cue start."""
    m = re.match(r"(\d+):(\d\d)", item.get("ts", ""))
    t = int(m.group(1)) * 60 + int(m.group(2)) if m else None
    detail = item.get("detail", "")
    quotes = [q for q in re.findall(r"['\"‘“]([^'\"’”]{6,}?)['\"’”]", detail)]
    near = (
        [k for k, c in enumerate(en.cues) if abs(c.start - t) <= window]
        if t is not None
        else list(range(len(en.cues)))
    )

    def text(k):
        return " ".join(" ".join(en.cues[k].lines).split()).lower()

    for q in quotes:
        hit = [k for k in near if q.lower() in text(k)]
        if hit:
            return min(hit, key=lambda k: abs(en.cues[k].start - (t or 0)))
    n = re.search(r"\[cue (\d+)\]", detail)
    if n and int(n.group(1)) in near:
        return int(n.group(1))
    if t is None or not near:
        return None
    return min(near, key=lambda k: abs(en.cues[k].start - t))


def _verdict(judge, prompt: str) -> str | None:
    try:
        m = re.search(r"\{.*\}", judge(prompt) or "", re.S)
        d = json.loads(m.group(0)) if m else {}
    except Exception:
        return None
    if not isinstance(d, dict) or not d.get("changed") or not isinstance(d.get("line"), str):
        return None
    return " ".join(d["line"].split())


def acceptable(old: str, new: str, evidence: str, kind: str = "") -> str:
    """'' if `new` may replace `old`, else why not. A punctuation finding may change only punctuation
    and case ('It has been so. I'm glad' -> 'It has been. So I'm glad', as the captions heard it).
    """
    ow, nw = old.split(), new.split()
    if kind == "en-punctuation":

        def raw(w):  # digits kept as written: '0.83' -> '83' is not a punctuation fix
            return re.sub(r"[^a-z0-9]", "", w.lower())

        same_words = [raw(w) for w in ow if raw(w)] == [raw(w) for w in nw if raw(w)]
        return "" if same_words and old != new else "a punctuation fix may only change punctuation"
    heard = {_k(w) for w in evidence.split()}
    added = [w for w in nw if _k(w) and _k(w) not in {_k(x) for x in ow}]
    if not added and len(nw) >= len(ow):
        return "no misheard word changed"
    if any(_k(w) in FILLERS for w in added):
        return "adds a filler"
    if any(a and a == b for a, b in zip(map(_k, nw), map(_k, nw[1:]))) and not any(
        a and a == b for a, b in zip(map(_k, ow), map(_k, ow[1:]))
    ):
        return "adds a stutter"
    unheard = [w for w in added if _k(w) not in heard]
    if unheard and not sounds_like(old, new):
        return f"not heard by the captions or the 2nd ASR: {' '.join(unheard)}"
    return ""


def sounds_like(old: str, new: str, max_words: int = 3, min_sim: float = 0.5) -> bool:
    """A small correction whose new words sound like the misheard ones ('tire' / 'type to',
    'Lukash' / 'Łukasz', 'logo' / 'flow'): every recognizer misheard the same way, so the audio
    can't confirm it; two models agreeing from context, on a sound-alike, can."""
    import difflib
    import unicodedata

    def letters(x):
        x = unicodedata.normalize("NFKD", x)
        return "".join(ch for ch in x.lower() if ch.isalpha())

    ow, nw = old.split(), new.split()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, [_k(w) for w in ow], [_k(w) for w in nw], autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        if tag != "replace" or (i2 - i1) > max_words or (j2 - j1) > max_words:
            return False
        a, b = letters(" ".join(ow[i1:i2])), letters(" ".join(nw[j1:j2]))
        if difflib.SequenceMatcher(None, a, b).ratio() < min_sim:
            return False
    return True


def _place(en, k: int, text: str, words=None, max_cpl: int = 42) -> None:
    """Put `text` into cue k; when it doesn't fit two lines, split the cue in two at the line break
    nearest the middle, timed at the second part's first word in the primary transcript (else by
    length) — never by rewording (EN caption rule: a clean read only deletes)."""
    from .en_cleanread import wrap2
    from .model import Cue

    c = en.cues[k]
    lines = wrap2(text, max_cpl)
    if len(lines) <= 2 and all(len(x) <= max_cpl for x in lines):
        c.lines = lines
        return
    toks = text.split()
    # the speaker's pauses in this cue (primary ASR word timings), to break where they paused
    gap: dict[str, float] = {}
    if words is not None:
        ws = [w for w in words if c.start - 0.2 <= w.start <= c.end + 0.2]
        for a, b in zip(ws, ws[1:]):
            gap.setdefault(_k(b.text), max(b.start - a.end, 0.0))
    best, cut = None, len(toks) // 2
    for i in range(1, len(toks)):
        a, b = " ".join(toks[:i]), " ".join(toks[i:])
        if not all(len(w) <= 2 and all(len(x) <= max_cpl for x in w) for w in (wrap2(a), wrap2(b))):
            continue
        natural = toks[i - 1][-1] in ",.?!;:" or _k(toks[i]) in BREAK_BEFORE
        score = abs(len(a) - len(b)) - (30 if natural else 0) - 40 * gap.get(_k(toks[i]), 0.0)
        if best is None or score < best:
            best, cut = score, i
    first, second = " ".join(toks[:cut]), " ".join(toks[cut:])
    t = c.start + (c.end - c.start) * len(first) / max(len(text), 1)
    if words is not None:
        hit = [
            w.start for w in words if c.start <= w.start <= c.end and _k(w.text) == _k(toks[cut])
        ]
        if hit:
            t = min(hit, key=lambda x: abs(x - t))
    en.cues[k : k + 1] = [
        Cue(c.start, t, wrap2(first, max_cpl)),
        Cue(t, c.end, wrap2(second, max_cpl)),
    ]


def heard_as_written(line: str, caps: str, sec: str) -> bool:
    """Every content word of our line was heard by BOTH the captions and the 2nd ASR: three
    recognizers agree, so the speaker said it — the EN keeps the speaker's words (unusual
    phrasing like 'doing your own experience' was held, then kept)."""
    from .transcribe_qe import _is_content

    content = [_k(w) for w in line.split() if _is_content(w)]
    if not content or not sec:
        return False
    a, b = {_k(w) for w in caps.split()}, {_k(w) for w in sec.split()}
    # the 2nd ASR's word timings can fall just outside the cue: a miss on the cue's first or last
    # content word doesn't count ('Over here is sometimes…')
    missed = [i for i, w in enumerate(content) if not (w in a and w in b)]
    return all(i in (0, len(content) - 1) and content[i] in a for i in missed)


def _sibling(line: str, done: list[dict]) -> dict | None:
    """A line already corrected in this video that this one nearly repeats ('So I think getting
    stuff done into NWA for customers' / 'I think getting stuff done into NBA for customers')."""
    import difflib

    best = None
    for r in done:
        ratio = difflib.SequenceMatcher(
            None, line.lower().split(), r["was"].lower().split()
        ).ratio()
        if ratio >= 0.75 and (best is None or ratio > best[0]):
            best = (ratio, r)
    return best[1] if best else None


def resolve(
    en,
    held: list[dict],
    autosub,
    second,
    judges,
    *,
    brief: str = "",
    words=None,
    log=None,
    known: str = "",
):
    """(en, resolved, still_held). `judges`: two model callables (prompt -> text). Changes `en` in
    place (it may gain cues from a split); held items it can't resolve get the caption reading.
    `log` (a list) receives every attempt: line, captions, both answers, the outcome.

    Outcomes: 'confirmed' (all three recognizers heard the line as written: kept, released),
    'resolved' (both models agree on a correction the captions support), else still held. Lines are
    decided top-down so a near-repeat of a line corrected earlier sees that correction."""
    resolved, still, placements = [], [], []
    log = log if log is not None else []
    todo = []
    for item in held:
        k = _cue_of(en, item) if item.get("kind") in KINDS and autosub is not None else None
        if k is None:
            still.append(item)
            continue
        todo.append((k, item))
    for k, item in sorted(todo, key=lambda x: x[0]):
        c = en.cues[k]
        line = " ".join(" ".join(c.lines).split())
        caps = caption_window(autosub, c.start, c.end)
        sec = asr_window(second, c.start, c.end)
        if not caps:
            still.append(item)
            continue
        mishear_kind = item.get("kind") in ("en-garbled", "en-mishear", "en-grammar")
        # a terminology/spelling question is about the written form ('LM' for LLM), which all
        # the recognizers can share — 'heard as written' settles only mishear questions
        if mishear_kind and heard_as_written(
            line, caption_window(autosub, c.start, c.end, pad=0.3), sec
        ):
            log.append(
                {"ts": item.get("ts"), "line": line, "captions": caps, "outcome": "confirmed"}
            )
            resolved.append({**item, "was": line, "now": line, "captions": caps, "confirmed": True})
            continue
        sib = _sibling(line, [r for r in resolved if not r.get("confirmed")])
        note = re.sub(r"^\[cue \d+\]\s*", "", item.get("detail", ""))
        if sib:
            note += f" | Earlier in this video a near-identical line was corrected: {sib['was']!r} -> {sib['now']!r}"
        prompt = ASK.format(
            brief=brief[:600] or "(none)",
            prev=" ".join(en.cues[k - 1].lines) if k else "",
            line=line,
            nxt=" ".join(en.cues[k + 1].lines) if k + 1 < len(en.cues) else "",
            note=note,
            caps=caps,
            second=sec or "(none)",
            known=known[:800] or "(none)",
        )
        answers = [_verdict(j, prompt) for j in judges]
        agreed = (
            len(judges) >= 2
            and all(answers)
            and len({re.sub(r"[^\w]", "", a.lower()) for a in answers if a}) == 1
        )
        why = "" if agreed else "the two models didn't agree on a correction"
        new = answers[0] if agreed and answers[0] else ""
        # a correction mirrored from a sibling is supported by the sibling's captions too
        # `known`: the video's names and glossary terms — a name the recognizers can't spell
        # ('Lukash' for Łukasz) is evidence from the speaker list, not from the audio
        evidence = (
            caps
            + " "
            + sec
            + " "
            + known
            + (" " + sib["now"] + " " + sib["captions"] if sib else "")
        )
        why = why or acceptable(line, new, evidence, item.get("kind", ""))
        log.append(
            {
                "ts": item.get("ts"),
                "line": line,
                "captions": caps,
                "answers": answers,
                "sibling": sib["was"] if sib else None,
                "outcome": why or "resolved",
            }
        )
        if why:
            base = item.get("detail", "").split(" | YouTube captions: ")[0]
            still.append({**item, "detail": f"{base} | YouTube captions: {caps!r}"})
            continue
        placements.append((k, new))
        resolved.append({**item, "was": line, "now": new, "captions": caps})
    # bottom-up, so a split cue doesn't shift the indices of the ones still to place
    for k, new in sorted(placements, key=lambda x: -x[0]):
        _place(en, k, new, words)
    en.renumber()
    order = {id(x): i for i, x in enumerate(held)}
    still.sort(key=lambda x: order.get(id(x), 0))
    return en, resolved, still


def drop_moot_meaning(held: list[dict], resolved: list[dict]) -> list[dict]:
    """A meaning question about a line the tiebreaker just corrected is moot ('What does "getting
    stuff done into NWA" mean?')."""
    gone = [r["was"] for r in resolved if not r.get("confirmed")]

    def moot(x):
        if x.get("kind") != "en-meaning":
            return False
        m = re.search(r'What does "(.+?)" mean', x.get("detail", ""))
        return m is not None and any(m.group(1) in g for g in gone)

    def moot_name(x):
        # a held name/slip question about words the tiebreaker just replaced ('React Native AD3')
        if x.get("kind") not in ("en-name", "en-slip"):
            return False
        # repr() quotes: '...' normally, "..." when the words hold an apostrophe ("O'Brien")
        m = re.search(r"\] (['\"])(.+?)\1 -> ", x.get("detail", ""))
        return m is not None and any(
            x.get("ts")  # no timestamp, no proof it's the same line
            and r.get("ts") == x.get("ts")  # the same line, not any line that holds a short name
            and m.group(2) in r["was"]
            and m.group(2) not in r["now"]
            for r in resolved
            if not r.get("confirmed")
        )

    return [x for x in held if not moot(x) and not moot_name(x)]


def section(resolved: list[dict]) -> str:
    fixed = [r for r in resolved if not r.get("confirmed")]
    kept = [r for r in resolved if r.get("confirmed")]
    out = ""
    if fixed:
        rows = "\n".join(
            f"| {r['ts']} | {r['was']} | {r['now']} | {r['captions'][:120]} |" for r in fixed
        )
        out += (
            f"\n## Resolved with the YouTube captions ({len(fixed)}; both models agreed, words heard)\n\n"
            f"| Time | Was | Now | Captions |\n|---|---|---|---|\n{rows}\n"
        )
    if kept:
        rows = "\n".join(f"| {r['ts']} | {r['was']} |" for r in kept)
        out += (
            f"\n## Confirmed as said ({len(kept)}; the captions and the 2nd ASR heard it as written — kept)\n\n"
            f"| Time | Line |\n|---|---|\n{rows}\n"
        )
    return out


def captions_file(out) -> str:
    hits = sorted(Path(out, "_ingest").glob("*.en.vtt")) + sorted(
        Path(out, "_ingest").glob("*.vtt")
    )
    return str(hits[0]) if hits else ""
