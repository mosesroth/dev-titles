"""EN subtitle timing vs the SPEECH itself.

validate/sync.py checks a translation against the EN track, so a wrong EN track passes and so does
every translation timed to it. This checks the EN against an independent speech clock: word-level
timestamps from an ASR pass (our raw.json) or YouTube's auto-captions (json3 / karaoke-tagged
vtt). The EN words are aligned to the speech words by content (difflib on normalized tokens); a
cue is OFF when the speech time of its first matched word is far from where the cue starts.

Why it exists: an EN track once shipped with "in IDE features into a terminal" ~4 s after it was
said (a caption cue spanning a pause; caption_to_transcript spreads words evenly across a cue), and
the zh re-timed to that EN inherited it. Nothing compared EN to the audio.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

# A cue may lead its first word a little (reading lead-in) and auto-caption word times lag the
# audio slightly; beyond these it is a visible sync error.
LATE_S = 1.0  # WARN: the cue starts this much AFTER its first word is spoken (text arrives late)
LATE_ERROR_S = 2.0  # ERROR beyond this. 1-2 s late is typically a sentence's tail word carried
# into the next cue ("…finds it / interesting."); the defect that motivated this was 4.2 s.
EARLY_ERROR_S = 5.0  # ERROR: text on screen this long before it is said (a 6 s "Hi" opening cue)
PAUSE_S = (
    3.0  # ERROR: a silence this long INSIDE one cue — the text after it is on screen that much
)
# before it is said. The cue-start check can't see it: the cue starts on time.
EARLY_S = 3.0  # WARN: the cue starts this much BEFORE its words (e.g. it kept the time span of
# filler that clean-read deleted, so the text sits on screen while "um, you know" is said)

_TOK = re.compile(r"[a-z0-9']+")


def _norm(word: str) -> str:
    return "".join(_TOK.findall(word.lower().replace("’", "'")))


def load_speech_words(path: str | Path) -> list[tuple[str, float]]:
    """(normalized word, start seconds) from a YouTube json3 auto-caption, a karaoke-tagged
    auto-caption .vtt (<00:00:01.234><c> word</c>), or our raw.json (ASR words)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    out: list[tuple[str, float]] = []
    if p.suffix == ".json3":
        for ev in json.loads(text).get("events", []):
            t0 = ev.get("tStartMs", 0)
            for sg in ev.get("segs", []) or []:
                w = _norm(sg.get("utf8", ""))
                if w:
                    out.append((w, (t0 + sg.get("tOffsetMs", 0)) / 1000))
        return out
    if p.suffix == ".json":  # raw.json: [{"text","start","end",…}]
        return [(_norm(w["text"]), float(w["start"])) for w in json.loads(text) if _norm(w["text"])]
    # karaoke vtt: the cue start belongs to the first word, then <ts><c> word</c> pairs
    ts = r"(\d+):(\d\d):(\d\d)\.(\d{3})"

    def sec(m) -> float:
        h, mi, s, ms = (int(x) for x in m.groups()[:4])
        return h * 3600 + mi * 60 + s + ms / 1000

    seen: set[tuple[str, float]] = set()
    for block in text.split("\n\n"):
        m = re.search(ts + r" --> ", block)
        if not m or "<c>" not in block:
            continue  # rolling duplicates of the previous line carry no <c> tags
        start = sec(m)
        body = block.split("\n", 1)[1] if "\n" in block else ""
        line = [ln for ln in body.split("\n") if "<c>" in ln]
        if not line:
            continue
        head = line[0].split("<", 1)[0]  # the words before the first timestamp tag
        for w in head.split():
            if _norm(w) and (_norm(w), start) not in seen:
                seen.add((_norm(w), start))
                out.append((_norm(w), start))
        for mm in re.finditer(r"<" + ts + r"><c>([^<]*)</c>", line[0]):
            t = sec(mm)
            for w in mm.group(5).split():
                if _norm(w):
                    out.append((_norm(w), t))
    return out


def speech_sync_report(en, speech: list[tuple[str, float]]) -> dict:
    """Per-cue offset (speech time of the cue's first matched word - cue start) for EN cues whose
    first words are found in the speech. `off`: cues starting > LATE_S after or > EARLY_S before
    their words. `coverage`: share of EN words matched (low = the speech source isn't this track).
    """
    en_tok: list[tuple[str, int]] = []
    for k, c in enumerate(en.cues):
        for w in " ".join(c.lines).split():
            if _norm(w):
                en_tok.append((_norm(w), k))
    sm = difflib.SequenceMatcher(
        None, [t for t, _ in en_tok], [w for w, _ in speech], autojunk=False
    )
    first_hit: dict[int, tuple[int, float]] = {}  # cue -> (word position in cue, speech time)
    pos_in_cue: list[int] = []
    last, n = -1, 0
    for _, k in en_tok:
        n = n + 1 if k == last else 0
        pos_in_cue.append(n)
        last = k
    matched = 0
    in_cue: dict[int, list[tuple[int, float]]] = {}  # cue -> [(word position, speech time)]
    for blk in sm.get_matching_blocks():
        if blk.size < 2:  # a lone common word ("the") is not evidence of alignment
            continue
        for d in range(blk.size):
            i = blk.a + d
            matched += 1
            k = en_tok[i][1]
            in_cue.setdefault(k, []).append((pos_in_cue[i], speech[blk.b + d][1]))
            if k not in first_hit or pos_in_cue[i] < first_hit[k][0]:
                first_hit[k] = (pos_in_cue[i], speech[blk.b + d][1])
    rows, off = [], []
    for k, (pos, t) in sorted(first_hit.items()):
        if pos > 0:  # the cue's FIRST word wasn't matched; its start can't be judged
            continue
        c = en.cues[k]
        delta = round(t - c.start, 2)
        rows.append((k, c.start, delta))
        if delta < -LATE_S or delta > EARLY_S:
            off.append(
                {
                    "cue": k + 1,
                    "start": round(c.start, 2),
                    "offset": delta,
                    "severity": (
                        "error" if delta < -LATE_ERROR_S or delta > EARLY_ERROR_S else "warn"
                    ),
                    "text": c.text[:60],
                }
            )
    for k, hits in sorted(in_cue.items()):
        hits.sort()
        gaps = [(b[1] - a[1], a[0]) for a, b in zip(hits, hits[1:]) if b[0] == a[0] + 1]
        if gaps and max(gaps)[0] > PAUSE_S:
            g, pos = max(gaps)
            c = en.cues[k]
            off.append(
                {
                    "cue": k + 1,
                    "start": round(c.start, 2),
                    "offset": round(g, 2),
                    "kind": "pause",
                    "after_word": pos,
                    "severity": "error",
                    "text": c.text[:60],
                }
            )
    off.sort(key=lambda o: o["start"])
    ds = sorted(d for _, _, d in rows)
    return {
        "checked": len(rows),
        "coverage": round(matched / max(len(en_tok), 1), 2),
        "median": ds[len(ds) // 2] if ds else None,
        "off": off,
    }


def video_id_for(name: str, ticket_root: str | Path = "../output/ticket") -> str | None:
    """YouTube id of a delivered video: its ticket folder's SOURCE-URL.txt / TICKET-DRAFT.md, else
    an `_ingest/<id>.*` file name."""
    d = Path(ticket_root) / name
    for f in (d / "SOURCE-URL.txt", d / "TICKET-DRAFT.md"):
        if f.exists():
            m = re.search(r"v=([A-Za-z0-9_-]{11})", f.read_text(encoding="utf-8"))
            if m:
                return m.group(1)
    if (d / "_ingest").is_dir():
        for f in sorted((d / "_ingest").iterdir()):
            m = re.match(r"([A-Za-z0-9_-]{11})\.", f.name)
            if m:
                return m.group(1)
    return None


def speech_words_for(video_id: str, cache_dir: str | Path) -> list[tuple[str, float]]:
    """Word-timed YouTube auto-captions for a video (downloaded once into cache_dir with yt-dlp),
    or [] when unavailable."""
    cache = Path(cache_dir)
    p = cache / f"{video_id}.en.json3"
    if not p.exists():
        try:
            import yt_dlp
        except ImportError:
            return []
        cache.mkdir(parents=True, exist_ok=True)
        opts = {
            "skip_download": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "subtitlesformat": "json3",
            "outtmpl": str(cache / f"{video_id}.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as y:
                y.download([f"https://www.youtube.com/watch?v={video_id}"])
        except Exception:
            return []
    return load_speech_words(p) if p.exists() else []


def _cue_word_times(en, speech) -> dict[int, tuple[float, float]]:
    """cue index -> (speech time of its first word, of its last word), for cues whose first AND
    last words are aligned (blocks of >= 2 matching words)."""
    en_tok: list[tuple[str, int, int]] = []  # (token, cue, position in cue)
    for k, c in enumerate(en.cues):
        toks = [t for t in (_norm(w) for w in " ".join(c.lines).split()) if t]
        en_tok += [(t, k, i) for i, t in enumerate(toks)]
    sizes = {k: sum(1 for _, kk, _ in en_tok if kk == k) for k in range(len(en.cues))}
    sm = difflib.SequenceMatcher(
        None, [t for t, _, _ in en_tok], [w for w, _ in speech], autojunk=False
    )
    first: dict[int, float] = {}
    last: dict[int, float] = {}
    for blk in sm.get_matching_blocks():
        if blk.size < 2:
            continue
        for d in range(blk.size):
            _, k, pos = en_tok[blk.a + d]
            t = speech[blk.b + d][1]
            if pos == 0:
                first[k] = t
            if pos == sizes[k] - 1:
                last[k] = t
    return {k: (first[k], last[k]) for k in first if k in last and last[k] >= first[k]}


def _split_at_pauses(en, speech, spec):
    """Split every cue that holds a long silence (speech_sync_report kind "pause") at that silence;
    text unchanged, each part re-broken into lines. A fragment of <= 2 words joins the neighbouring
    cue on ITS side of the silence (before it: the previous cue; after it: the next one) when that
    still fits; a split that can't be laid out legally is not made. Times are rough here —
    retime_to_speech snaps them to the words right after."""
    import copy

    from ..model import Cue, Subtitle
    from ..segment import break_lines

    def fits(text: str) -> bool:
        ls = break_lines(text, spec)
        return len(ls) <= spec.max_lines and all(len(x) <= spec.max_cpl for x in ls)

    pauses = {
        o["cue"] - 1: o for o in speech_sync_report(en, speech)["off"] if o.get("kind") == "pause"
    }
    if not pauses:
        return en
    cues = [copy.deepcopy(c) for c in en.cues]
    out: list = []
    carry = ""  # a <= 2-word fragment waiting to be prefixed to the next cue
    for k, c in enumerate(cues):
        raw = (carry + " " + " ".join(c.lines)).split() if carry else " ".join(c.lines).split()
        shift = len(carry.split()) if carry else 0
        carry = ""
        o = pauses.get(k)
        idx, n = None, -1
        if o is not None:
            for jj, w in enumerate(raw[shift:]):  # positions count only tokens that normalize
                if _norm(w):
                    n += 1
                    if n == o["after_word"]:
                        idx = jj + shift
                        break
        if idx is None or idx + 1 >= len(raw):
            # untouched cue keeps its own lines; only a cue that received a carried fragment is re-broken
            lines = break_lines(" ".join(raw), spec) if shift else list(c.lines)
            out.append(Cue(c.start, c.end, lines))
            continue
        a, b = raw[: idx + 1], raw[idx + 1 :]
        ta, tb = " ".join(a), " ".join(b)
        mid = c.start + (c.end - c.start) * len(ta) / max(len(ta) + len(tb), 1)
        if len(a) <= 2 and out and fits(out[-1].text.replace("\n", " ") + " " + ta):
            prev = out[-1]
            out[-1] = Cue(
                prev.start, mid, break_lines(prev.text.replace("\n", " ") + " " + ta, spec)
            )
            out.append(Cue(mid, c.end, break_lines(tb, spec)))
        elif (
            len(b) <= 2
            and k + 1 < len(cues)
            and fits(tb + " " + cues[k + 1].text.replace("\n", " "))
        ):
            out.append(Cue(c.start, mid, break_lines(ta, spec)))
            carry = tb
        elif fits(ta) and fits(tb):  # a short part stands alone: the silence gives it reading time
            out += [
                Cue(c.start, mid, break_lines(ta, spec)),
                Cue(mid, c.end, break_lines(tb, spec)),
            ]
        else:
            out.append(Cue(c.start, c.end, break_lines(" ".join(raw), spec)))
    return Subtitle(en.language, out).renumber()


def retime_to_speech(en, speech, spec=None, lead: float = 0.1, tail: float = 0.6):
    """Snap EN cue boundaries to the speech: a cue starts `lead` s before its first word and runs
    to its last word + `tail` (or the next cue's start). Text is never touched; a cue whose words
    can't be aligned, or whose snapped timing would break min/max duration or overlap, keeps its
    old timing. Returns (new subtitle, number of cues moved)."""
    import copy

    from ..config import spec_for

    spec = spec or spec_for("en")
    gap = 0.0  # contiguous cues are fine (the validator only flags real overlaps / tiny gaps)
    en = _split_at_pauses(en, speech, spec)
    wt = _cue_word_times(en, speech)
    out = copy.deepcopy(en)
    cues = out.cues
    moved = 0
    for k, c in enumerate(cues):
        if k not in wt:
            continue
        s = max(wt[k][0] - lead, (cues[k - 1].end + gap) if k else 0.0)
        nxt = cues[k + 1].start if k + 1 < len(cues) else None
        if k + 1 < len(cues) and (k + 1) in wt:
            nxt = wt[k + 1][0] - lead
        e = wt[k][1] + tail
        if nxt is not None:
            e = min(e, nxt - gap)
        e = max(e, s + spec.min_dur)  # hold a short cue long enough to read
        if nxt is not None and e > nxt - gap and (k + 1) not in wt:
            continue  # would run into a cue that keeps its old timing: keep this one's too
        # (when the next cue is itself being snapped, it simply starts after this one ends)
        if e - s > spec.max_dur:
            e = s + spec.max_dur
        if abs(s - c.start) > 0.25 or abs(e - c.end) > 0.25:
            c.start, c.end = round(s, 3), round(e, 3)
            moved += 1
    # REPAIR (one forward sweep, never stuck): an overlap trims the earlier cue when it stays
    # readable; a cue left too short borrows time from the silence after it, else before it.
    for k in range(len(cues)):
        c = cues[k]
        nxt = cues[k + 1].start if k + 1 < len(cues) else None
        prv = cues[k - 1].end if k else 0.0
        if nxt is not None and c.end > nxt:
            c.end = max(nxt, c.start)
        if c.end - c.start < spec.min_dur:
            want = c.start + spec.min_dur
            c.end = min(want, nxt) if nxt is not None else want
            if c.end - c.start < spec.min_dur:
                c.start = max(prv, c.end - spec.min_dur)
            if (
                c.end - c.start < spec.min_dur and k
            ):  # still short: borrow from a roomy previous cue
                p_ = cues[k - 1]
                need = spec.min_dur - (c.end - c.start)
                if p_.end - p_.start - need >= spec.min_dur:
                    p_.end = round(p_.end - need, 3)
                    c.start = p_.end
    return out, moved


def target_speech_desync(
    target_cues, speech, *, tol: float = 3.0, unique_window: float = 30.0
) -> list[dict]:
    """Translated cues out of step with the SPEECH: a Latin name/product in the cue (it survives
    translation verbatim) is spoken more than `tol` s outside the cue's span. Only names spoken ONCE
    within ±unique_window s count. Compared to the speech, not the EN track, because a translation
    re-timed to a wrong EN track looks perfectly consistent with it."""
    anchor = re.compile(r"[A-Za-z][A-Za-z0-9.+-]{3,}")
    out = []
    for c in target_cues:
        mid = (c.start + c.end) / 2
        for tok in sorted(set(anchor.findall(" ".join(c.lines)))):
            key = _norm(tok)
            hits = [t for w, t in speech if w == key and abs(t - mid) <= unique_window]
            if len(hits) != 1:
                continue
            gap = max(hits[0] - c.end, c.start - hits[0], 0.0)
            if gap > tol:
                out.append(
                    {
                        "start": round(c.start, 2),
                        "anchor": tok,
                        "gap": round(gap, 2),
                        "spoken": round(hits[0], 2),
                        "text": " ".join(c.lines)[:40],
                    }
                )
    return out
