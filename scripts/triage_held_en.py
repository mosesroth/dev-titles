#!/usr/bin/env python3
"""Triage EN-review lines that were HELD for a person but never resolved, on already-delivered videos.

For each held line: the EN cue as delivered, the live YouTube EN track, an independent ASR of the
audio (Parakeet) and YouTube's own auto-captions for the same moment. Two ASRs that agree with each
other but not with our EN are strong evidence of a mishear; a judge then sorts each line into
  ok      — our EN is right (the review's flag was a false alarm)
  fix     — our EN is wrong; a minimal corrected line is proposed
  listen  — the evidence disagrees; a person must listen (timestamped link)
Writes <out>/TRIAGE.md and <out>/triage.json. Needs yt-dlp (audio + auto-captions) and parakeet-mlx.

  PYTHONPATH=src python3 scripts/triage_held_en.py --manifest output/final/MANIFEST.md \
      --out scratch/triage [--video NAME ...]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dev_titles import en_gate, listen_list
from dev_titles import llm
from dev_titles.cli import _read_any
from dev_titles.transcribe_qe import load_autosub

JUDGE = """An English subtitle line from a developer video was flagged by a proofreader. Decide whether the
line is actually wrong, using two INDEPENDENT machine transcripts of the same audio as evidence (they make
their own mistakes, especially on names and product terms, and include filler our clean-read removes).

Proofreader's flag: {flag}
FLAGGED PHRASE: {quote}
OUR SENTENCE: {ctx}
LIVE YOUTUBE ENGLISH for that moment: {live}
ASR 1 (Parakeet): {asr1}
ASR 2 (YouTube auto-captions): {asr2}

Answer ONLY JSON: {{"verdict": "ok"|"fix"|"listen", "replacement": "<what the flagged phrase should say, only for fix>",
"why": "<max 20 words>"}}
- ok: the flagged phrase matches what is said (clean-read: fillers/stutters removed is fine) — a false alarm.
- fix: the flagged phrase is clearly wrong AND the transcripts agree on the right wording. Replacement keeps the
  clean-read style and replaces ONLY the flagged phrase.
- listen: the transcripts disagree with each other or are unclear, so a person has to listen."""


def _ts(s):
    m = re.match(r"(?:(\d+):)?(\d+):(\d+)", s or "")
    if not m:
        return None
    h, mi, se = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(se)


def _text(cues, a, b):
    return " ".join(" ".join(c.lines) for c in cues if c.start < b and c.end > a)


def _norm(x):
    return re.sub(r"[^a-z0-9]", "", (x or "").lower())


def _locate(en, quote):
    """Index of the cue where the quoted phrase sits (it may span two cues); None if it is gone."""
    qn = _norm(quote)
    if len(qn) < 6:
        return None
    for k in range(len(en)):
        if qn in _norm(" ".join(en[k].lines)):  # wholly inside one cue
            return k
    for k in range(len(en) - 1):
        if qn in _norm(" ".join(" ".join(c.lines) for c in en[k:k + 2])):  # straddles a cue break
            return k + 1 if _norm(" ".join(en[k + 1].lines)).startswith(qn[-6:]) is False and qn[:6] not in _norm(" ".join(en[k].lines)) else k
    return None


def _sentence(en, i):
    """(first, last) cue index of the sentence(s) around cue i: back/forward to sentence-final punctuation."""
    end = lambda c: re.search(r"[.?!…]['\"”]?$", " ".join(c.lines).strip())
    lo = i
    while lo > 0 and not end(en[lo - 1]) and i - lo < 3:
        lo -= 1
    hi = i
    while hi < len(en) - 1 and not end(en[hi]) and hi - i < 3:
        hi += 1
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="output/final/MANIFEST.md")
    ap.add_argument("--ticket-dir", default="output/ticket")
    ap.add_argument("--out", default="scratch/triage")
    ap.add_argument("--video", action="append")
    ap.add_argument("--extra", action="append", default=[], help="NAME=youtube_id for videos not in the manifest")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "media").mkdir(parents=True, exist_ok=True)
    ids = dict(re.findall(r"\|\s*(\w+)\s*\|\s*https://www\.youtube\.com/watch\?v=([\w-]+)",
                          Path(a.manifest).read_text(encoding="utf-8")))
    ids.update(dict(x.split("=", 1) for x in a.extra))
    from dev_titles.transcribe.parakeet import DEFAULT_MODEL, ParakeetTranscriber

    asr = ParakeetTranscriber(model=DEFAULT_MODEL)
    claude = llm.invoke()
    gpt = llm.second_judge()[0] or claude  # a different-family judge when available
    items = []
    for d in sorted(Path(a.ticket_dir).iterdir()):
        v = d.name
        if not d.is_dir() or (a.video and v not in a.video) or not (d / "en_review.md").exists():
            continue
        _, held = en_gate.review_state(d)
        if not held:
            continue
        vid = ids.get(v)
        if not vid:
            print(f"{v}: no YouTube id — skipped ({len(held)} held)")
            continue
        m = out / "media"
        if not (m / f"{v}.mp4").exists():
            subprocess.run(["yt-dlp", "-q", "-f", "233/140/bestaudio", "-o", str(m / f"{v}.%(ext)s"), "--", vid],
                           check=False)
            subprocess.run(["yt-dlp", "-q", "--skip-download", "--write-auto-subs", "--write-subs",
                            "--sub-langs", "en,en-orig", "--sub-format", "vtt", "-o", str(m / f"{v}"), "--", vid],
                           check=False)
        audio = next(iter(sorted(m.glob(f"{v}.mp4")) + sorted(m.glob(f"{v}.m4a")) + sorted(m.glob(f"{v}.webm"))), None)
        pj = m / f"{v}.parakeet.json"
        if audio and not pj.exists():
            t = asr.transcribe(str(audio))
            pj.write_text(json.dumps([[w.text, w.start] for w in t.words]), encoding="utf-8")
        words = json.loads(pj.read_text()) if pj.exists() else []
        auto_p = next(iter(sorted(m.glob(f"{v}.en-orig.vtt")) + sorted(m.glob(f"{v}.en.vtt"))), None)
        auto = load_autosub(str(auto_p)).cues if auto_p else []
        live_p = m / f"{v}.en.vtt"
        live = _read_any(str(live_p), "en").cues if live_p.exists() and auto_p != live_p else []
        en = _read_any(str(d / "en.vtt"), "en").cues
        for h in held:
            det = h.get("detail", "")
            q = re.findall(r"[\"“'‘]([^\"”'’]{6,})[\"”'’]", det)
            quote = q[0] if q else ""
            i = _locate(en, quote) if quote else None
            if i is None:  # quote gone (fixed since) or no quote: fall back to the timestamp
                t0 = _ts(h.get("ts"))
                if t0 is None:
                    continue
                i = min(range(len(en)), key=lambda j: abs(en[j].start - t0))
            lo, hi = _sentence(en, i)
            a0, a1 = en[lo].start, en[hi].end
            items.append({"video": v, "id": vid, "ts": h.get("ts"), "start": round(en[i].start, 1),
                          "flag": f"{h.get('kind')}: {det}", "quote": quote or "(none quoted)",
                          "still_in_ours": bool(quote) and _norm(quote) in _norm(_text(en, a0 - 8, a1 + 8)),
                          "still_live": bool(quote) and bool(live) and _norm(quote) in _norm(_text(live, a0 - 8, a1 + 8)),
                          "line": " ".join(en[i].lines), "ctx": " ".join(" ".join(x.lines) for x in en[lo:hi + 1]),
                          "live": _text(live, a0 - 0.5, a1 + 0.5) if live else "(no live EN found)",
                          "asr1": " ".join(w for w, st in words if a0 - 1.5 <= st <= a1 + 1.5),
                          "asr2": _text(auto, a0 - 1.5, a1 + 1.5)})
        print(f"{v}: {len(held)} held line(s) gathered", flush=True)

    def judge(it):
        vs = []
        for j in (claude, gpt):
            try:
                m = re.search(r"\{.*\}", j(JUDGE.format(**it)) or "", re.S)
                vs.append(json.loads(m.group(0)))
            except Exception:
                vs.append({"verdict": "listen", "why": "judge failed"})
        a_, b_ = vs[0].get("verdict"), vs[1].get("verdict")
        it["verdict"] = a_ if a_ == b_ else "listen"
        it["fixed_line"] = vs[0].get("replacement") if it["verdict"] == "fix" else ""
        if not it["still_in_ours"] and it["quote"] != "(none quoted)":
            it["verdict"] = "already fixed" if not it["still_live"] else it["verdict"]
        it["why"] = " | ".join(x.get("why", "") for x in vs)
        return it

    with ThreadPoolExecutor(8) as ex:
        items = list(ex.map(judge, items))
    (out / "triage.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter

    cnt = Counter(i["verdict"] for i in items)
    rep = [f"# Held EN lines on delivered videos — triage", "",
           f"{len(items)} lines: {cnt.get('fix', 0)} fix, {cnt.get('listen', 0)} listen, {cnt.get('ok', 0)} ok, "
           f"{cnt.get('already fixed', 0)} already fixed "
           "(both judges must agree on ok/fix, else listen).", ""]
    for verdict in ("fix", "listen", "already fixed", "ok"):
        rep += [f"## {verdict} ({cnt.get(verdict, 0)})", ""]
        for i in (x for x in items if x["verdict"] == verdict):
            link = f"https://www.youtube.com/watch?v={i['id']}&t={int(i['start'])}"
            rep.append(f"- **{i['video']} {i['ts']}** ([listen]({link})): {i['line']}")
            rep.append(f"  - flagged: {i['quote']}  (still in our EN: {i['still_in_ours']}, live: {i['still_live']})")
            if verdict == "fix":
                rep.append(f"  - fix: {i['quote']} -> {i['fixed_line']}")
            rep.append(f"  - {i['why']}")
        rep.append("")
    (out / "TRIAGE.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep[:3]))


if __name__ == "__main__":
    main()
