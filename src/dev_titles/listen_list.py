"""LISTEN LIST: the human review of held English lines, as one clickable HTML page.

SOP (operator, 2026-10-23): every EN line the EN review holds is resolved by a person LISTENING to it.
The page lists each held line, numbered, with
  - a YouTube link that starts 2 s before the line (opens in a new tab),
  - our English sentence with the flagged words highlighted,
  - the reviewer's note (what it suspects),
  - optionally what an independent ASR (Parakeet) heard for the same moment,
and greys out answered items. The operator replies "<n>: <what's said>" or "<n> ok"; each answer is
applied to EN and carried into ja + zh. Written automatically by the EN sign-off gate when it stops a
run (<out>/EN_LISTEN.html) and by scripts/triage_held_en.py for delivered videos.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

LEAD_S = 2  # links open this many seconds before the line

_Q = re.compile(r'"([^"]{4,}?)"|“([^”]{4,}?)”|(?<![A-Za-z])\'(.{4,}?)\'(?![A-Za-z])')


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def quoted(detail: str) -> str:
    """The first phrase the review quoted ('...' / "..." / “...”), apostrophes inside words kept."""
    d = re.sub(r"^\[cue \d+\]\s*", "", detail or "")
    m = _Q.search(d)
    return next(g for g in m.groups() if g) if m else ""


def locate(cues, quote: str):
    """Index of the cue holding the quoted phrase (preferring one that holds it whole); None if gone."""
    q = norm(quote)
    if len(q) < 6:
        return None
    for k, c in enumerate(cues):
        if q in norm(" ".join(c.lines)):
            return k
    for k in range(len(cues) - 1):
        if q in norm(" ".join(" ".join(c.lines) for c in cues[k:k + 2])):
            return k
    return None


def sentence(cues, i: int) -> tuple[int, int]:
    """(first, last) cue index of the sentence around cue i (at most 3 cues either way)."""
    end = lambda c: re.search(r"[.?!…]['\"”]?$", " ".join(c.lines).strip())
    lo = i
    while lo > 0 and not end(cues[lo - 1]) and i - lo < 3:
        lo -= 1
    hi = i
    while hi < len(cues) - 1 and not end(cues[hi]) and hi - i < 3:
        hi += 1
    return lo, hi


def _ts(s):
    m = re.match(r"(?:(\d+):)?(\d+):(\d+)", s or "")
    return None if not m else int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def build_items(en_cues, held: list[dict], video: str, youtube_id: str = "", asr_words=None) -> list[dict]:
    """One item per held line: video, link id, start, sentence, flagged phrase, note, ASR text."""
    items = []
    for h in held:
        det = re.sub(r"^\[cue \d+\]\s*", "", h.get("detail", ""))
        q = quoted(det)
        k = locate(en_cues, q) if q else None
        found = k is not None
        if not found:
            t0 = _ts(h.get("ts"))
            if t0 is None or not en_cues:
                continue
            k = min(range(len(en_cues)), key=lambda j: abs(en_cues[j].start - t0))
        lo, hi = sentence(en_cues, k)
        ctx = " ".join(" ".join(c.lines) for c in en_cues[lo:hi + 1])
        if q and found and norm(q) not in norm(ctx):
            lo, hi = min(lo, k), min(max(hi, k + 1), len(en_cues) - 1)
            ctx = " ".join(" ".join(c.lines) for c in en_cues[lo:hi + 1])
        a0, a1 = en_cues[lo].start, en_cues[hi].end
        asr = " ".join(w for w, s in (asr_words or []) if a0 - 1 <= s <= a1 + 1)
        items.append({"video": video, "id": youtube_id, "start": int(en_cues[k].start), "ctx": ctx,
                      "q": q, "found": found, "note": det, "asr": asr})
    return items


def write_html(items: list[dict], path, title: str = "English lines to listen to", answers: dict | None = None) -> Path:
    e = html.escape
    answers = answers or {}
    out = [
        f"<!doctype html><meta charset='utf-8'><title>{e(title)}</title>",
        "<style>body{font:15px/1.5 -apple-system,sans-serif;max-width:920px;margin:2em auto;padding:0 1em}"
        "li{margin:0 0 1.3em}a{font-weight:600}.q{background:#fff3b0;padding:0 .2em}"
        "td{vertical-align:top;padding:1px 8px 1px 0}td.k{color:#666;white-space:nowrap}.w{color:#a40}"
        ".done{opacity:.45}</style>",
        f"<h1>{e(title)}</h1><p>Reply with the number and what's said, or “ok” if our line is right. Each fix "
        f"goes into EN + ja + zh. Links open {LEAD_S} s before the line. Greyed = answered.</p><ol>",
    ]
    for n, x in enumerate(items, 1):
        t = max(x["start"] - LEAD_S, 0)
        link = f"https://www.youtube.com/watch?v={x['id']}&t={t}s" if x.get("id") else ""
        ctx = e(x["ctx"])
        if x.get("found") and x.get("q"):
            m = re.search(re.escape(e(x["q"])).replace(r"\ ", r"\s+"), ctx, re.I)
            if m:
                ctx = ctx[:m.start()] + f"<span class='q'>{ctx[m.start():m.end()]}</span>" + ctx[m.end():]
        if x.get("q") and not x.get("found"):
            ctx += "<br><span class='w'>The flagged words aren't in our English any more — just check that this line is right.</span>"
        when = f"{x['start'] // 60:02d}:{x['start'] % 60:02d}"
        head = (f"<a href='{link}' target='_blank'>{e(x['video'])} — {when}</a>" if link
                else f"<b>{e(x['video'])} — {when}</b> <span class='w'>(no YouTube link: play the file at {when})</span>")
        rows = [("Our English", ctx), ("Reviewer's note", e(x["note"]))]
        if "asr" in x:
            rows.append(("Parakeet heard", e(x["asr"]) or "(no transcript for this moment)"))
        a = answers.get(str(n))
        if a:
            rows.append(("Answered", e(a.get("answer", "") if isinstance(a, dict) else str(a))))
        cls = " class='done'" if a else ""
        table = "".join(f"<tr><td class='k'>{k}</td><td>{v}</td></tr>" for k, v in rows)
        out.append(f"<li{cls}>{head}<table>{table}</table></li>")
    out.append("</ol>")
    p = Path(path)
    p.write_text("\n".join(out), encoding="utf-8")
    return p


def youtube_id(out_dir) -> str:
    p = Path(out_dir) / "SOURCE-URL.txt"
    if p.exists():
        m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", p.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    return ""


def for_out_dir(en, out_dir, held: list[dict]) -> Path | None:
    """Write <out>/EN_LISTEN.html for a run's held EN lines (used by the EN sign-off gate)."""
    if not held:
        return None
    out_dir = Path(out_dir)
    asr = out_dir / "parakeet.json"
    words = json.loads(asr.read_text(encoding="utf-8")) if asr.exists() else None
    items = build_items(en.cues, held, out_dir.name, youtube_id(out_dir), words)
    return write_html(items, out_dir / "EN_LISTEN.html", f"English lines to listen to — {out_dir.name}")
