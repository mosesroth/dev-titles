"""ENGLISH SIGN-OFF GATE: nothing is translated from an English script that hasn't been fully cleaned
and reviewed.

Why: a video went to translation with three lines the EN review had flagged and HELD for a human
(a mis-heard word, a garbled half-sentence, a stumble). The review printed them to en_review.md and
translation ran anyway, so every language inherited the errors. Likewise a crashed review printed
"skipped" and carried on.

Rule, enforced before any translation call:
  * the EN review must have RUN (a failed review blocks), and
  * the exact EN being translated must be signed off: `en_approved.json` holds a hash of that EN.
Sign-off is automatic only when the review held nothing back; otherwise a person resolves the held
items (edit en.vtt), then signs off with `dev_titles approve-en <out>`. Any later edit to the EN
changes the hash and needs a new sign-off. There is deliberately no bypass flag.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

APPROVAL = "en_approved.json"
STATE = "en_review.json"


def en_hash(en) -> str:
    """Content hash of an EN subtitle: cue timings (10 ms) + whitespace-normalized text."""
    rows = [f"{c.start:.2f}|{c.end:.2f}|{' '.join(' '.join(c.lines).split())}" for c in en.cues]
    return hashlib.sha1("\n".join(rows).encode("utf-8"), usedforsecurity=False).hexdigest()


def save_review_state(out, status: str, held: list[dict], applied: int = 0) -> None:
    """status: 'ok' (review ran) or 'failed' (review crashed — blocks translation)."""
    Path(out, STATE).write_text(
        json.dumps(
            {"status": status, "applied": applied, "held": held}, ensure_ascii=False, indent=1
        ),
        encoding="utf-8",
    )


def review_state(out) -> tuple[str, list[dict]]:
    """(status, held items). Falls back to a legacy en_review.md (no json) by counting its held rows."""
    p = Path(out, STATE)
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return d.get("status", "failed"), d.get("held", [])
    md = Path(out, "en_review.md")
    if not md.exists():
        return "missing", []
    text = md.read_text(encoding="utf-8")
    if "No editorial issues flagged" in text:
        return "ok", []
    sec = text.split("## Needs your adjudication", 1)
    held = []
    if len(sec) == 2:
        for line in sec[1].splitlines():
            m = re.match(r"\|\s*([\d:]+)\s*\|\s*([^|]+)\|\s*(.+?)\s*\|\s*$", line)
            if m:
                held.append({"ts": m.group(1), "kind": m.group(2).strip(), "detail": m.group(3)})
    return "ok", held


def text_hash(en) -> str:
    """Hash of the EN TEXT only (whitespace-normalized). A sign-off is about the words; re-timing
    the cues to the speech (validate/speech_sync.py) must not void it."""
    t = " ".join(" ".join(" ".join(c.lines) for c in en.cues).split())
    return hashlib.sha1(t.encode("utf-8"), usedforsecurity=False).hexdigest()


def approve(out, en, by: str, note: str = "") -> dict:
    rec = {
        "hash": en_hash(en),
        "text_hash": text_hash(en),
        "by": by,
        "at": time.strftime("%Y-%m-%d %H:%M"),
        "note": note,
    }
    Path(out, APPROVAL).write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


SPEECH_SYNC = (
    "speech_sync.json"  # written by cli._speech_align: EN cues out of sync with the speech
)
_MECHANICAL = ("sound_tag", "caption_artifact", "duplicate_text", "overlap", "max_lines", "max_cpl")


def mechanical_issues(en, out) -> list[dict]:
    """Defects no sign-off can waive, as held-line items (they show in EN_LISTEN.html): validation
    errors (sound tags, '>>' markers, duplicated text, overlaps, line count/width) and EN cues > 2 s
    out of sync with the speech (speech_sync.json, when it was measured for THIS EN)."""
    from .validate import validate_subtitle

    def ts(t: float) -> str:
        return f"{int(t) // 60:02d}:{int(t) % 60:02d}"

    items = []
    for v in validate_subtitle(en):
        if v.severity == "error" and v.rule in _MECHANICAL and v.cue_index:
            c = en.cues[v.cue_index - 1]
            items.append({"ts": ts(c.start), "kind": v.rule, "detail": v.detail, "quote": c.text})
    p = Path(out, SPEECH_SYNC)
    if p.exists():
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("hash") == en_hash(en):
            for o in rec.get("off", []):
                if o.get("severity") == "error":
                    items.append(
                        {
                            "ts": ts(o["start"]),
                            "kind": "speech_sync",
                            "detail": (
                                f"{o['offset']:.1f} s silence inside the cue (text shown early): "
                                if o.get("kind") == "pause"
                                else f"cue starts {-o['offset']:+.1f} s from its words: "
                            )
                            + repr(" ".join(o.get("text", "").split())[:40]),
                            "quote": o.get("text", ""),
                        }
                    )
    return items


def check(en, out) -> tuple[bool, str, list[dict]]:
    """(ok, reason, open held items). Writes an automatic sign-off when the review held nothing.
    Mechanical defects (mechanical_issues) block even a signed-off EN."""
    mech = mechanical_issues(en, out)
    if mech:
        return (
            False,
            f"{len(mech)} mechanical defect(s) in the EN (fix them; no sign-off waives these)",
            mech,
        )
    p = Path(out, APPROVAL)
    if p.exists():
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("hash") == en_hash(en):
            return True, f"EN signed off ({rec.get('by')}, {rec.get('at')})", []
        if rec.get("text_hash") and rec["text_hash"] == text_hash(en):
            return (
                True,
                f"EN signed off ({rec.get('by')}, {rec.get('at')}; re-timed since, text unchanged)",
                [],
            )
        stale = "the EN changed since it was signed off"
    else:
        stale = ""
    status, held = review_state(out)
    if status != "ok":
        return False, f"the EN review did not run successfully (status: {status})", held
    if held:
        return (
            False,
            (stale + "; " if stale else "") + f"{len(held)} EN line(s) held for review",
            held,
        )
    if stale:
        return False, stale, []
    approve(out, en, by="auto: EN review held nothing")
    return True, "EN signed off automatically (review held nothing)", []


def require(en, out, en_path: str = "") -> bool:
    """Print the verdict; False (caller must stop) unless the EN is signed off."""
    ok, why, held = check(en, out)
    if ok:
        print(why)
        return True
    print(f"\nSTOP — not translating: {why}.")
    for h in held:
        print(f"  {h.get('ts', '?'):>6}  {h.get('kind', '')}: {h.get('detail', '')}")
    if (
        held
    ):  # SOP: the person resolves held lines by LISTENING — one clickable page, links 2 s early
        from .listen_list import for_out_dir

        try:
            page = for_out_dir(en, out, held)
            print(f"\nListen list (open in a browser, links start 2 s before each line): {page}")
        except Exception as e:  # never let the page block the message itself
            print(f"(listen list not written: {type(e).__name__})")
    target = en_path or str(Path(out, "en.vtt"))
    print(
        f"\nResolve each held line in {target} (or confirm it is right), then sign off:\n"
        f"  PYTHONPATH=src python3 -m dev_titles.cli approve-en {out} --en {target}\n"
    )
    return False
