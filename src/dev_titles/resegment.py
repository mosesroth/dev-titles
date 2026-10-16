"""Sentence merge + CJK re-timing (requirement R3).

English caption timing splits sentences across short cues; CJK reads much slower
(ja ≤4 cps, zh ≤9 vs EN ≤20), so translating per-EN-cue and reusing EN timing
overflows the cues. Instead we:

  1. `merge_en_to_sentences` — stitch the fragmented EN cues back into whole
     sentences, each with the combined [start, end] time window.
  2. (translation happens per sentence — better context + fluency)
  3. `resegment_translation` — re-split the translated sentence into cues sized to
     the target language's CPL/lines, distributing the sentence's time window
     proportionally to each cue's character width so reading speed is balanced.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Sequence

from .config import LangSpec
from .model import Cue, Subtitle
from .segment.cjk import wrap_lines
from .textwidth import display_width

_SENT_END = tuple(".!?。！？…")


@dataclass
class Sentence:
    text: str
    start: float
    end: float


# Split AFTER a sentence-final mark that is followed by whitespace. LOSSLESS:
# every character is kept (concatenating the parts reproduces the text), and a
# mid-token dot like "skills.sh" or "v1.2" is NOT a split point (its dot isn't
# followed by whitespace). The earlier findall-based splitter DROPPED the text
# before such a dot ("...from skills.sh." collapsed to "sh."), losing whole
# clauses before translation — an omission the coverage gate couldn't catch
# because the content never reached the translator.
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？…])\s+")


def _split_into_sentences(text: str, start: float, end: float) -> list[Sentence]:
    """Split a merged block that contains MULTIPLE sentences into one Sentence each
    (lossless), distributing [start, end] proportionally to char count.

    Leading-sentence loss (audit): the model, handed a 2-3 sentence blob, keeps
    the last sentence and silently summarizes the earlier ones. Translating one
    sentence at a time removes the blob, so there is no tail to drop."""
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    if len(parts) <= 1:
        return [Sentence(text, start, end)]
    total = sum(len(p) for p in parts) or 1
    out: list[Sentence] = []
    t = start
    for p in parts:
        dur = (end - start) * (len(p) / total)
        out.append(Sentence(p, t, min(t + dur, end)))
        t += dur
    out[-1] = Sentence(out[-1].text, out[-1].start, end)
    return out


def merge_en_to_sentences(en: Subtitle) -> list[Sentence]:
    """Merge EN cues into sentences. Accumulates cue text (whitespace-normalized,
    placeholder '_'/empty cues dropped) until a cue ends with sentence-final
    punctuation; the sentence spans the first cue's start to the last cue's end.
    A flushed block that still holds several sentences (cue breaks didn't align to
    sentence ends) is split one-per-sentence with proportional timing."""
    sents: list[Sentence] = []
    buf: list[str] = []
    start: float | None = None
    last_end: float = 0.0
    for c in en.cues:
        txt = re.sub(r"\s+", " ", " ".join(c.lines)).strip()
        if not txt or txt == "_":
            continue
        if start is None:
            start = c.start
        buf.append(txt)
        last_end = c.end
        if txt.endswith(_SENT_END):
            sents += _split_into_sentences(" ".join(buf).strip(), start, last_end)
            buf, start = [], None
    if buf and start is not None:
        sents += _split_into_sentences(" ".join(buf).strip(), start, last_end)
    return sents


_CJK_BREAKS = set("、。，,．；;：:！!？?…　 ")
_LATTOK = re.compile(r"[A-Za-z0-9]")


def _in_latin_token(text: str, pos: int) -> bool:
    """Would a split at boundary `pos` fall INSIDE a Latin/number token (both neighbours are
    ASCII letters/digits)? Used to avoid mid-token cuts like Nimbus->'Nim'|'bus'."""
    return (
        0 < pos < len(text)
        and bool(_LATTOK.match(text[pos - 1]))
        and bool(_LATTOK.match(text[pos]))
    )


def _split_cjk_proportional(text: str, weights: "Sequence[float]") -> list[str]:
    """Split `text` into len(weights) pieces sized proportionally to `weights`, snapping each
    split to the nearest CJK/clause break within a small window (so pieces end at punctuation
    or a space, not mid-word). Every character is kept (concatenation reproduces `text`)."""
    n = len(weights)
    if n <= 1 or not text:
        return [text] + [""] * (n - 1)
    total = sum(weights) or 1.0
    L = len(text)
    pieces: list[str] = []
    prev = 0
    acc = 0.0
    for j, w in enumerate(weights[:-1]):
        acc += w
        remaining = n - 1 - j  # pieces AFTER this boundary (must each get >=1)
        # clamp so this piece is >=1 char AND every remaining piece can still get >=1 char
        lo, hi = prev + 1, max(prev + 1, L - remaining)
        tgt = min(max(int(round(L * acc / total)), lo), hi)
        best = tgt
        for d in range(0, 9):  # search ±8 chars for a break boundary
            for p in (tgt + d, tgt - d):
                if lo <= p <= hi and (p == L or text[p - 1] in _CJK_BREAKS):
                    best = p
                    break
            else:
                continue
            break
        # TOKEN-SAFE: never split INSIDE a Latin/number token (e.g. Nimbus -> "Nim"|"bus",
        # React Native -> "Reac"|"t"). If `best` lands mid-token, snap to the nearer token
        # edge that stays within [lo, hi]; if neither edge fits (token longer than the clamp
        # window — rare), keep `best`. CJK has no such tokens so this only affects Latin/digits.
        if 0 < best < L and _in_latin_token(text, best):
            left = best
            while left > lo and _in_latin_token(text, left):
                left -= 1
            right = best
            while right < hi and _in_latin_token(text, right):
                right += 1
            cands = [
                p
                for p in (
                    right if (right - best) <= (best - left) else left,
                    left if (right - best) <= (best - left) else right,
                )
                if lo <= p <= hi and not _in_latin_token(text, p)
            ]
            if cands:
                best = cands[0]
        pieces.append(text[prev:best])
        prev = best
    pieces.append(text[prev:])
    return pieces


def _nows(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _reflow_key(encs: list[str], cjk: str) -> str:
    """Cache key for a re-flow item: the split depends only on the EN cue boundaries + the
    translation text, so identical (cues, translation) always yields the same chunks."""
    return "\x00".join(encs) + "\x01" + (cjk or "")


def reflow_llm_batch(
    items: list[tuple[list[str], str]],
    invoke,
    lang_name: str,
    group: int = 8,
    cache: dict | None = None,
) -> list[list[str] | None]:
    """BATCHED meaning-aligned re-flow. `items` = [(en_cue_texts, cjk), ...] for multi-cue
    sentences. Handles `group` sentences per LLM call (so a video is ~10 calls, not ~100).
    Returns [chunks | None] aligned with `items` (None → caller falls back to proportional for
    that item). Re-SEGMENTS only (never re-translates), so translation quality is unchanged; the
    caller's per-item lossless guard rejects any paraphrase/drop.

    `cache` (optional dict, keyed by `_reflow_key`) memoizes chunk-splits across runs: an unchanged
    (cues, translation) reuses its result with NO LLM call, so re-aligning after a small EN edit
    re-flows only the changed sentences (seconds, not ~40 min). Newly computed splits are written
    back into `cache` for the caller to persist."""
    import json as _json

    out: list[list[str] | None] = [None] * len(items)
    todo: list[int] = []
    for k, (encs, cjk) in enumerate(items):
        cached = cache.get(_reflow_key(encs, cjk)) if cache is not None else None
        if isinstance(cached, list) and len(cached) == len(encs):
            out[k] = cached
        else:
            todo.append(k)
    for g0 in range(0, len(todo), group):
        idxs = todo[g0 : g0 + group]
        blocks = []
        for local, k in enumerate(idxs):
            encs, cjk = items[k]
            cues = "\n".join(f"    cue {j + 1}: {t}" for j, t in enumerate(encs))
            blocks.append(
                f"### Sentence {local} ({len(encs)} cues)\n{cues}\n    translation: {cjk}"
            )
        prompt = (
            f"For EACH sentence below, split its {lang_name} translation into EXACTLY as many "
            f"consecutive chunks as it has cues, so chunk i carries the MEANING of cue i "
            f"(accounting for word-order differences vs English). Keep every character in order; "
            f"ONLY insert split points — do NOT rephrase, add, or drop anything. Output ONLY a "
            f"JSON object mapping each sentence number (as a string) to its array of chunk "
            f'strings, e.g. {{"0": ["...","..."], "1": ["...","...","..."]}}.\n\n'
            + "\n\n".join(blocks)
        )
        try:
            raw = invoke(prompt) or ""
            s, e = raw.find("{"), raw.rfind("}")
            data = _json.loads(raw[s : e + 1]) if s != -1 and e != -1 else {}
        except Exception:
            data = {}
        for local, k in enumerate(idxs):
            encs, cjk = items[k]
            ch = data.get(str(local), data.get(local))
            if (
                isinstance(ch, list)
                and len(ch) == len(encs)
                and all(isinstance(x, str) for x in ch)
            ):
                out[k] = ch
                if cache is not None:
                    cache[_reflow_key(encs, cjk)] = ch
    return out


def _reflow_pieces(
    content: list[Cue], sentences: list[Sentence], cjk_texts: list[str], reflow
) -> list[str]:
    """Assign each EN content cue its meaning-aligned CJK chunk. Each cue maps to the sentence it
    overlaps most; multi-cue sentences are collected and re-flowed in ONE batched call
    (`reflow(items) -> [chunks|None]`), with a per-sentence lossless guard (else proportional).
    A sentence that won no cue has its text carried into the next cue-bearing group (no loss)."""

    def _ov(a0, a1, b0, b1):
        return max(0.0, min(a1, b1) - max(a0, b0))

    cue_sent = (
        [
            max(
                range(len(sentences)),
                key=lambda si: _ov(c.start, c.end, sentences[si].start, sentences[si].end),
            )
            for c in content
        ]
        if sentences
        else [0] * len(content)
    )
    from collections import defaultdict

    sent_cues: dict[int, list[int]] = defaultdict(list)
    for ci, si in enumerate(cue_sent):
        sent_cues[si].append(ci)
    # build an ordered plan; collect multi-cue items for a single batched re-flow
    plan: list = []  # ('single', cue, txt) | ('multi', cues, txt, item_idx)
    items: list[tuple[list[str], str]] = []
    pending = ""
    for si in range(len(sentences)):
        txt = pending + (cjk_texts[si] if si < len(cjk_texts) else "")
        cues = sent_cues.get(si, [])
        if not cues:
            pending = txt
            continue
        pending = ""
        if len(cues) == 1:
            plan.append(("single", cues[0], txt))
        else:
            items.append(([" ".join(content[ci].lines) for ci in cues], txt))
            plan.append(("multi", cues, txt, len(items) - 1))
    results = reflow(items) if (reflow and items) else [None] * len(items)
    pieces: list[str] = [""] * len(content)
    for entry in plan:
        if entry[0] == "single":
            pieces[entry[1]] = entry[2]
            continue
        _, cues, txt, idx = entry
        ch = results[idx] if idx < len(results) else None
        if not ch or len(ch) != len(cues) or _nows("".join(ch)) != _nows(txt):
            weights = [max(len(_nows(" ".join(content[ci].lines))), 1) for ci in cues]
            ch = _split_cjk_proportional(txt, weights)  # guard → proportional (lossless)
        for ci, c in zip(cues, ch):
            pieces[ci] = c
    if pending:  # trailing no-cue text → last assigned cue
        idx = max((i for i, p in enumerate(pieces) if p), default=len(pieces) - 1)
        pieces[idx] = pieces[idx] + pending
    return pieces


def align_translation_to_en(
    en: Subtitle,
    sentences: list[Sentence],
    cjk_texts: list[str],
    spec: LangSpec,
    keep_together=(),
    reflow=None,
    lang_name: str = "",
) -> list[Cue]:
    """STRICT 1:1 EN-cue alignment: emit exactly one target cue per (non-empty) EN cue, at the
    SAME start/end, so EN cue N ↔ ja/zh cue N (for translation-management tools that ingest cue-by-cue).

    LOSSLESS: the FULL translation (all sentences, in order) is concatenated into one stream and
    distributed across the EN cues in proportion to each cue's SOURCE character count, split at
    CJK break points. Concatenating the cues reproduces the whole translation — no sentence can be
    dropped (an earlier per-sentence-bucket version dropped sentences that shared one EN cue's
    time window). `sentences`/`cjk_texts` are paired but only the ordered text stream is used."""
    content = [
        c for c in en.cues if re.sub(r"\s+", " ", " ".join(c.lines)).strip() not in ("", "_")
    ]
    if not content:
        return []
    full = "".join((t or "") for t in cjk_texts)  # ordered translation stream
    # weight each EN cue by its EN character count (translation length tracks source length),
    # so each cue gets a proportional slice near its own content.
    weights = [max(len(re.sub(r"\s+", " ", " ".join(c.lines)).strip()), 1) for c in content]
    # MEANING-aligned re-flow when an LLM is provided (distributes each sentence's translation
    # across its cues by meaning); else proportional. WHOLE-VIDEO lossless guard: if the re-flow
    # result doesn't concatenate back to the full translation, fall back to the proportional split
    # (which is provably lossless) — so quality is never worse than the proportional baseline.
    pieces = None
    if reflow is not None:
        rp = _reflow_pieces(content, sentences, cjk_texts, reflow)
        # Accept the reflow ONLY if it is lossless AND leaves no content cue empty. The reflow can
        # legitimately concatenate back to the full text while assigning "" to some cues (all the
        # text piled into neighbors) — that produces empty deliverable cues, breaking 1:1 ingestion. The
        # proportional fallback has a no-starve clamp (>=1 char/cue), so reject empties -> proportional.
        if (
            len(rp) == len(content)
            and _nows("".join(rp)) == _nows(full)
            and all(_nows(p) for p in rp)
        ):
            pieces = rp
    if pieces is None:
        pieces = _split_cjk_proportional(full, weights)
    out: list[Cue] = []
    for c, piece in zip(content, pieces):
        # A reflow chunk can carry an embedded newline (LLM JSON); left in, wrap_lines
        # yields an empty line element that write_srt turns into a mid-cue blank line —
        # a malformed cue that orphans the trailing text and breaks 1:1 ingestion. Collapse
        # any internal whitespace run to a single space before wrapping.
        piece = re.sub(r"\s+", " ", piece).strip()
        lines = [ln for ln in wrap_lines(piece, spec, keep_together) if ln.strip()] if piece else []
        out.append(Cue(start=c.start, end=c.end, lines=lines or [""]))
    return out


def free_segment_default() -> bool:
    """Shipping default: FREE independent-timed CJK segmentation — metrics showed it beats or
    ties strict 1:1 EN-cue alignment on every arm. Escape hatch: SUBTITLE_FREE_SEGMENT=0
    forces legacy 1:1 (e.g. a deliverable for a tool that requires 1:1 cues); any other/unset
    value = free-seg."""
    return os.environ.get("SUBTITLE_FREE_SEGMENT", "1") != "0"


def build_target_cues(
    en: Subtitle,
    sents: list["Sentence"],
    cjk_texts: list[str],
    spec: LangSpec,
    keep_together=(),
    reflow=None,
    free_segment: bool | None = None,
) -> list[Cue]:
    """Build the shipping target cues from per-sentence translations.

    Two modes:
      * 1:1 (legacy/strict-ingestion): one target cue per EN cue, same timings (align_translation_to_en).
      * FREE (free_segment=True): re-split each sentence's translation within its OWN speech
        window [sent.start, sent.end], sized to the language's cps/max_lines — no EN-cue-count
        constraint. This is the original design (see module docstring); it removes the
        structural cps/max_lines violations that strict 1:1 forced onto CJK (esp. ja).

    `free_segment=None` resolves from free_segment_default() (the SUBTITLE_FREE_SEGMENT env)."""
    if free_segment is None:
        free_segment = free_segment_default()
    if not free_segment:
        return align_translation_to_en(
            en, sents, cjk_texts, spec, keep_together=keep_together, reflow=reflow
        )
    cues: list[Cue] = []
    for sent, cjk in zip(sents, cjk_texts):
        txt = re.sub(r"\s+", " ", (cjk or "")).strip()
        if not txt:
            continue
        cues += resegment_translation(txt, sent.start, sent.end, spec, keep_together)
    # resegment_translation enforces min_dur WITHIN a sentence but can't reach across sentence
    # boundaries, so a short sentence leaves a sub-min_dur cue. With independent timing we can
    # fix it globally: extend the short cue into the following silence, then merge if adjacent.
    cues = enforce_min_dur(cues, spec)
    cues = snap_gaps(cues, spec)
    # Cap display at max_dur — a sentence over a long/paused speech window (or a within-sentence
    # merge) can leave a cue longer than max_dur; trim the END (trailing silence is fine with
    # independent timing). Last so nothing downstream re-stretches it.
    cues = [
        (
            Cue(start=c.start, end=c.start + spec.max_dur, lines=c.lines)
            if c.duration > spec.max_dur
            else c
        )
        for c in cues
    ]
    return _move_dangling_open_brackets(cues)


_OPEN_BRACKETS = "「『（(【〔《〈［[｛{“‘"


def _move_dangling_open_brackets(cues: list[Cue]) -> list[Cue]:
    """A cue must not END with an OPENING bracket/quote — it binds to the quoted text that starts
    the NEXT cue (kinsoku_line_end). wrap_lines can't fix this because the partner text lives in a
    different cue, so move the dangling opener to the front of the next cue (dropped if it's the
    last cue, where it's an unmatched orphan)."""
    for i, c in enumerate(cues):
        if not c.lines:
            continue
        last = c.lines[-1].rstrip()
        if not last or last[-1] not in _OPEN_BRACKETS:
            continue
        ch, trimmed = last[-1], last[:-1].rstrip()
        new_lines = (c.lines[:-1] + [trimmed]) if trimmed else c.lines[:-1]
        cues[i] = Cue(start=c.start, end=c.end, lines=new_lines or [""])
        if i + 1 < len(cues):
            nxt = cues[i + 1]
            nl = list(nxt.lines) or [""]
            nl[0] = ch + nl[0]
            cues[i + 1] = Cue(start=nxt.start, end=nxt.end, lines=nl)
    return cues


def normalize_en_layout(sub: Subtitle) -> Subtitle:
    """Guarantee the EN deliverable meets its layout spec. EN corrections (ASR run-recovery, the
    auto-caption cross-check, the editorial en_autofix) write a fixed cue back as ONE unwrapped
    line, and a correction that adds words can spill onto a 3rd line — 26/28 delivered EN files had
    such cues (lines to 100 chars). Re-wrap only offending cues; a cue that needs more than
    max_lines is split across its own time window; then the min-duration floor is enforced."""
    from .config import spec_for
    from .textwidth import display_width

    spec = spec_for(sub.language or "en")
    out: list[Cue] = []
    for c in sub.cues:
        bad = len(c.lines) > (spec.max_lines or 2) or any(
            display_width(ln, cjk=spec.cjk) > spec.max_cpl for ln in c.lines
        )
        if not bad:
            out.append(c)
            continue
        text = re.sub(r"\s+", " ", " ".join(c.lines)).strip()
        out += resegment_translation(text, c.start, c.end, spec) or [c]
    return Subtitle(sub.language, snap_gaps(enforce_min_dur(out, spec), spec)).renumber()


def enforce_min_dur(cues: list[Cue], spec: LangSpec) -> list[Cue]:
    """Global (cross-sentence) min-duration cleanup for free segmentation. A cue shorter than
    spec.min_dur first extends its END into the gap before the next cue (independent timing lets
    a subtitle linger over the following silence); if still short and butted against the next cue
    with a compatible line budget, it MERGES into that neighbor. Never overlaps or eats a cue."""
    if not cues:
        return cues
    md = spec.min_dur
    # Target a hair ABOVE min_dur: extending to exactly min_dur leaves end-start at 0.8329999s
    # under float rounding, which the validator's strict `< min_dur` still flags. 20ms (~0.5
    # frame) is imperceptible and safely clears the fp boundary.
    tgt = md + 0.02
    out: list[Cue] = list(cues)
    # Pass 1: extend short cues into the following gap (or trailing time for the last cue). If a
    # cue is butted against the next with no gap, BORROW time by pushing the next cue's start
    # later — but only while the next cue itself stays >= min_dur (never starve a neighbor).
    for i, c in enumerate(out):
        if c.duration >= md:
            continue
        if i + 1 < len(out):
            nxt = out[i + 1]
            new_end = min(c.start + tgt, nxt.start)
            if new_end > c.end:
                out[i] = c = Cue(start=c.start, end=new_end, lines=c.lines)
            need = tgt - c.duration
            if need > 1e-6:  # still short and butted -> borrow from next's slack
                give = min(need, max(nxt.duration - tgt, 0.0))
                if give > 0:
                    b = c.end + give
                    out[i] = Cue(start=c.start, end=b, lines=c.lines)
                    out[i + 1] = Cue(start=b, end=nxt.end, lines=nxt.lines)
        else:
            out[i] = Cue(start=c.start, end=c.start + tgt, lines=c.lines)
        # Still short? Symmetric backward fill: extend the START earlier into the PRECEDING gap,
        # then borrow from the previous cue's slack (keeping prev >= target). Handles interjections
        # that sit before a long cue or after a preceding silence (no room on the next side).
        c = out[i]
        if c.duration < md and i > 0:
            prev = out[i - 1]
            new_start = max(c.start - (tgt - c.duration), prev.end)
            if new_start < c.start:
                out[i] = c = Cue(start=new_start, end=c.end, lines=c.lines)
            need = tgt - c.duration
            if need > 1e-6:  # butted to prev -> borrow from prev's slack
                give = min(need, max(prev.duration - tgt, 0.0))
                if give > 0:
                    b = c.start - give
                    out[i - 1] = Cue(start=prev.start, end=b, lines=prev.lines)
                    out[i] = Cue(start=b, end=c.end, lines=c.lines)
    # Pass 2: any residual sub-min_dur cue butted against its neighbor -> merge if lines fit.
    merged: list[Cue] = []
    max_lines = spec.max_lines or 2
    for c in out:
        if (
            merged
            and c.duration < md
            and len(merged[-1].lines) + len(c.lines) <= max_lines
            and (c.end - merged[-1].start) <= spec.max_dur  # don't create a too-long cue
        ):
            p = merged[-1]
            merged[-1] = Cue(start=p.start, end=c.end, lines=p.lines + c.lines)
        else:
            merged.append(c)
    return merged


def snap_gaps(cues: list[Cue], spec: LangSpec) -> list[Cue]:
    """Close tiny sub-min_gap gaps between adjacent cues (flicker) by extending
    the earlier cue's end to the next cue's start. Larger gaps are left intact.
    Applied across the full subtitle (gaps arise at sentence boundaries)."""
    if len(cues) < 2:
        return cues
    min_gap = spec.min_gap_frames / spec.fps
    out = [cues[0]]
    for c in cues[1:]:
        p = out[-1]
        gap = c.start - p.end
        if 0 < gap < min_gap:
            out[-1] = Cue(start=p.start, end=c.start, lines=p.lines)
        out.append(c)
    return out


def resegment_translation(
    text: str, start: float, end: float, spec: LangSpec, keep_together=()
) -> list[Cue]:
    """Re-split `text` into cues within [start, end], sized to the language's
    CPL/lines, with time distributed proportionally to each cue's char width.

    `keep_together` = terms (glossary targets / DNT product names) that must not
    be split across a line OR cue boundary — the cue breaks fall at the line
    breaks, so protecting them in `wrap_lines` protects them across cues too."""
    lines = wrap_lines(text, spec, keep_together)
    if not lines:
        return []
    max_lines = spec.max_lines or 2
    groups = [lines[i : i + max_lines] for i in range(0, len(lines), max_lines)]
    widths = [max(sum(display_width(ln, spec.cjk) for ln in g), 0.5) for g in groups]
    total = sum(widths)
    span = max(end - start, 0.0)
    cues: list[Cue] = []
    t = start
    for g, w in zip(groups, widths):
        share = span * (w / total) if total else span
        # Display no longer than max_dur (leave a trailing gap); advance by the
        # full share so cues stay aligned to the speech window.
        dur = min(share, spec.max_dur)
        cues.append(Cue(start=t, end=t + dur, lines=g))
        t += share

    # Merge a too-short cue (< min_dur, a brief flash) into the previous one when
    # their combined lines still fit; extends the previous cue's end.
    merged: list[Cue] = []
    for c in cues:
        if (
            merged
            and c.duration < spec.min_dur
            and len(merged[-1].lines) + len(c.lines) <= (spec.max_lines or 2)
        ):
            p = merged[-1]
            merged[-1] = Cue(start=p.start, end=c.end, lines=p.lines + c.lines)
        else:
            merged.append(c)

    # For any residual sub-min_dur cue (couldn't merge — lines full), borrow time
    # from the previous cue by shifting the boundary earlier, keeping both cues
    # within the window and >= min_dur where possible (no overlap, no gap-eating).
    for i in range(len(merged) - 1, 0, -1):
        c, p = merged[i], merged[i - 1]
        if c.duration < spec.min_dur:
            take = min(spec.min_dur - c.duration, max(p.duration - spec.min_dur, 0.0))
            if take > 0:
                b = c.start - take
                merged[i - 1] = Cue(start=p.start, end=b, lines=p.lines)
                merged[i] = Cue(start=b, end=c.end, lines=c.lines)
    return merged
