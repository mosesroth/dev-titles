"""Carry fixes made on DELIVERED files back into the video's working copy, so a re-run keeps them.

A fix made at handover (a name corrected in the EN and the ja/zh, a slip, a line break) is written
to the deliverable and the library (deliver.write_deliverable), not to the run folder's en.vtt and
cache.jsonl that a re-run rebuilds from. Nine name fixes and a slip once existed only in the
delivered files; `run` now stops when the delivered EN is newer (cli._delivered_en_drift), and this
is the way past that stop:

  - EN: the delivered text becomes en.vtt (the previous one kept as a backup); approve-en is needed,
    since the signed-off text changed.
  - ja / zh: each changed span between the run's own track (<out>/<lang>.srt) and the delivered one
    is replaced in the ONE cache line that holds it, and that line is locked (as fix-line does).
    Layout-only changes (line breaks, spaces) are skipped; a span found in no line or in several is
    reported, never guessed.
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
import time
from pathlib import Path

CONTEXT = 3  # characters of unchanged text kept around a change so it is found in one line


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _joined(sub) -> str:
    """The track's text with cue/line breaks as '\n' (a break inside a word vs between Latin words is
    decided when a fragment is taken out)."""
    return "\n".join(ln for c in sub.cues for ln in c.lines)


def _index(text: str) -> list[int]:
    """Positions of the non-whitespace characters of `text`."""
    return [k for k, ch in enumerate(text) if not ch.isspace()]


def _fragment(text: str, idx: list[int], lo: int, hi: int) -> str:
    """text between compact positions [lo, hi) with its own spacing; a line break becomes a space
    only between two Latin letters/digits ('Ghizlan\nZaheer' -> 'Ghizlan Bardiowil')."""
    lo, hi = max(0, lo), min(len(idx), hi)
    if lo >= hi:
        return ""
    raw = text[idx[lo] : idx[hi - 1] + 1]
    return re.sub(
        r"(?<=[A-Za-z0-9])\n+(?=[A-Za-z0-9])",
        " ",
        re.sub(r"\n+(?![A-Za-z0-9])|(?<![A-Za-z0-9])\n+", "", raw),
    )


def changed_spans(old_sub, new_sub) -> list[dict]:
    """One {a: (lo, hi), b: (lo, hi)} per edit between two tracks, in compact (whitespace-free)
    positions, plus the compact strings and the original texts to cut fragments from. Character
    changes a few characters apart are one edit ('Cutter' -> 'Chris' is five little diff ops), and an
    edit touching a Latin word covers the whole word."""
    ta, tb = _joined(old_sub), _joined(new_sub)
    ia, ib = _index(ta), _index(tb)
    a, b = "".join(ta[k] for k in ia), "".join(tb[k] for k in ib)
    merged: list[list[int]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if merged and i1 - merged[-1][1] <= CONTEXT:
            merged[-1][1], merged[-1][3] = i2, j2
        else:
            merged.append([i1, i2, j1, j2])

    def latin(ch: str) -> bool:
        return ch.isascii() and ch.isalnum()

    wide: list[list[int]] = []
    for i1, i2, j1, j2 in merged:
        left = right = 0
        while left < min(i1, j1) and (latin(a[i1 - left - 1]) or latin(b[j1 - left - 1])):
            left += 1
        while (
            i2 + right < len(a)
            and j2 + right < len(b)
            and (latin(a[i2 + right]) or latin(b[j2 + right]))
        ):
            right += 1
        span = [i1 - left, i2 + right, j1 - left, j2 + right]
        # widened edits (plus their context) that reach each other are ONE edit: the context of one
        # must never run into the next, where old and new differ (a ja line was once cut to
        # 'React Compilerが こうド時に' — a deletion's context reached an insertion)
        if wide and span[0] - wide[-1][1] <= 2 * CONTEXT:
            wide[-1][1], wide[-1][3] = max(wide[-1][1], span[1]), max(wide[-1][3], span[3])
        else:
            wide.append(span)
    out = []
    for i1, i2, j1, j2 in wide:
        out.append({"a": (i1, i2), "b": (j1, j2), "ca": a, "tb": tb, "ib": ib})
    return out


def _variants(sp: dict):
    """(old compact fragment, new spaced fragment) with CONTEXT characters on both sides, then
    one side, then none — the first that sits inside a single cache line wins."""
    (i1, i2), (j1, j2), n = sp["a"], sp["b"], len(sp["ca"])
    # (merging edits closer than 2 x CONTEXT means this context never reaches a neighbouring edit)
    for left, right in ((CONTEXT, CONTEXT), (0, CONTEXT), (CONTEXT, 0)):
        lo, hi = max(0, i1 - left), min(n, i2 + right)
        dl, dr = i1 - lo, hi - i2
        old = sp["ca"][lo:hi]
        new = _fragment(sp["tb"], sp["ib"], j1 - dl, j2 + dr)
        if len(old) >= 4:  # a shorter fragment matches by accident ('r' in 'Chris')
            yield old, new


def _replace_compact(txt: str, old: str, new: str) -> str | None:
    """Replace the whitespace-free `old` inside `txt` (which may carry spaces) with `new`; None when
    `old` doesn't occur exactly once."""
    pos = _index(txt)
    comp = "".join(txt[k] for k in pos)
    if comp.count(old) != 1:
        return None
    s = comp.index(old)
    start, end = pos[s], pos[s + len(old) - 1] + 1
    return txt[:start] + new + txt[end:]


def sync_track(rows: list[dict], lang: str, old_sub, new_sub) -> tuple[list[dict], list[str]]:
    """Apply the delivered changes to the cache rows of `lang` (in place). Returns (done, failed):
    done = edit records for locked_edits.jsonl; failed = descriptions of spans not placed."""
    done, failed = [], []
    mine = [r for r in rows if r.get("lang") == lang]
    for sp in changed_spans(old_sub, new_sub):
        placed = False
        for old, new in _variants(sp):
            if not mine:
                break
            hits = [r for r in mine if old in _compact(r.get("txt", ""))]
            if len(hits) != 1:
                continue
            r = hits[0]
            if _compact(new) in _compact(r.get("txt", "")):
                # carried already: an insertion's old side is just its context, which survives the
                # insertion, so a second sync would insert it again ('メリットがが')
                placed = True
                break
            after = _replace_compact(r["txt"], old, new)
            if after is None:
                continue
            done.append({"i": r["i"], "en": r.get("src", ""), "before": r["txt"], "after": after})
            r["txt"], r["locked"] = after, True
            placed = True
            break
        if not placed:
            if any(
                len(_compact(new)) >= 4  # a short fragment is found by accident, like a short old
                and not any(old in _compact(r.get("txt", "")) for r in mine)
                and any(_compact(new) in _compact(r.get("txt", "")) for r in mine)
                for old, new in _variants(sp)
            ):
                continue  # carried already (the old text is gone and the new one is there)
            old, new = next(_variants(sp), ("", ""))
            failed.append(f"{old!r} -> {new!r}: not in exactly one cache line")

    # every line now matching the delivered text is the delivered line: lock it, so a re-run doesn't
    # let the native editor / condense / review-and-fix re-word lines nobody changed (a ja
    # '1列' -> '1行', zh 您 -> 大家 on a re-check)
    # punctuation ignored: the zh layout drops a comma at a line break, so a cache line and its
    # delivered line can differ by one '，' (on one zh track 24 of 75 lines didn't match)
    def bare(x: str) -> str:
        return re.sub(r"[\s，、。,.!?！？；;：:]+", "", x or "")

    shipped = bare(_joined(new_sub))
    for r in mine:
        t = bare(r.get("txt", ""))
        # a short line ('はい', 'Okay') is found inside the long joined track by accident
        if len(t) >= 6 and t in shipped and not r.get("locked"):
            r["locked"] = True
    return done, failed


MAX_SPANS = 25  # more than this is a reviewed track, not a few handover fixes


def sync(out, library, read, write_vtt, user: str = "", allow_many: bool = False) -> dict:
    """Sync one run folder from the library. `read(path, lang)` / `write_vtt(sub)` are the caller's
    subtitle I/O. Returns {"en": bool, "<lang>": {"locked": n, "failed": [...]}}."""
    out, library = Path(out), Path(library)
    v = out.name
    report: dict = {}
    lib_en, wc = library / f"{v}.en.srt", out / "en.vtt"
    report["en"] = False
    if lib_en.exists() and wc.exists():
        shipped, work = read(str(lib_en), "en"), read(str(wc), "en")
        if _compact("".join(c.text for c in shipped.cues)) != _compact(
            "".join(c.text for c in work.cues)
        ):
            shutil.copy2(wc, out / f"en.pre_sync_{time.strftime('%Y%m%d_%H%M%S')}.vtt")
            wc.write_text(write_vtt(shipped), encoding="utf-8")
            report["en"] = True
    cp = out / "cache.jsonl"
    if not cp.exists():
        return report
    rows = [json.loads(x) for x in cp.read_text(encoding="utf-8").splitlines() if x.strip()]
    locked_before = sum(bool(r.get("locked")) for r in rows)
    edits = []
    for lang in ("ja", "zh-Hans"):
        lib, mine = library / f"{v}.{lang}.srt", out / f"{lang}.srt"
        if not (lib.exists() and mine.exists()):
            continue
        old_sub, new_sub = read(str(mine), lang), read(str(lib), lang)
        n = len(changed_spans(old_sub, new_sub))
        if n > MAX_SPANS and not allow_many:
            msg = (
                f"{n} changes: a reviewed track, not handover fixes — not synced (--all forces it)"
            )
            report[lang] = {"locked": 0, "failed": [msg]}
            continue
        done, failed = sync_track(rows, lang, old_sub, new_sub)
        report[lang] = {"locked": len(done), "failed": failed}
        edits += [{**d, "lang": lang} for d in done]
    # written also when only lock flags changed (every span carried already, unchanged lines locked)
    if edits or sum(bool(r.get("locked")) for r in rows) != locked_before:
        cp.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
    if edits:
        with (out / "locked_edits.jsonl").open("a", encoding="utf-8") as fh:
            stamp = time.strftime("%Y-%m-%d %H:%M")
            for e in edits:
                rec = {**e, "by": user, "at": stamp, "note": "sync-delivered"}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return report
