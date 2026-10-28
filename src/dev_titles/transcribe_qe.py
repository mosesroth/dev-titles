"""Reference-free quality estimation for a transcript — find problem areas WITHOUT
a human caption to diff against (the "without a net" case for a brand-new video).

Combines three signals into one timestamped suspect list a human can spot-check:
  1. ASR word CONFIDENCE — WhisperX's own low-confidence words (jargon/names hide here).
  2. transcription GAPS — long silences where speech may have been dropped.
  3. LLM PLAUSIBILITY — a reference-free reader that flags garbled/implausible spans,
     broken code identifiers, and sentences that don't parse (text-only; it cannot
     hear the audio, so it misses plausible-sounding mishears — pair with 1 + a 2nd ASR).
Optionally a SECOND ASR pass: disagreements between two independent transcribers
pinpoint acoustic uncertainty with no human involved.
"""

from __future__ import annotations

import json
import re

from .cue_builder import low_confidence_spans, transcription_gaps


def _fmt(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


# ASR assigns low confidence to many common short/function words — that's noise, not
# error. Real mishears hide in low-confidence CONTENT words, so we flag only those.
_FUNC = frozenset(
    (
        "a an the this that these those it its there here is are was were be been being am "
        "s re ll m ve d t do does did will would can could shall should may might must have "
        "has had we you i they he she him her them us my your our their of in on at to into "
        "onto by with for from and or but so nor yet as if then than just now some any no "
        "not who which what when where how i'm it's that's we're you're don't we'll to're "
        "uh um oh yeah okay ok so well right like"
    ).split()
)


def _is_content(word: str) -> bool:
    k = re.sub(r"[^a-z0-9']", "", word.lower())
    return bool(k) and k not in _FUNC and len(re.sub(r"[^a-z0-9]", "", k)) >= 3


def plausibility_suspects(invoke, text: str, *, max_words: int = 140) -> list[dict]:
    """LLM reads the transcript (no reference) and flags spans that look like
    transcription errors. Chunked. Returns [{phrase, reason}]."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sents:
        cur = (cur + " " + s).strip()
        if len(cur.split()) >= max_words:
            chunks.append(cur)
            cur = ""
    if cur:
        chunks.append(cur)
    out: list[dict] = []
    for c in chunks:
        prompt = (
            "This is an automatic speech-to-text transcript of a technical developer video. "
            "With NO reference, flag ONLY spans that look like TRANSCRIPTION ERRORS: garbled "
            "or nonsensical phrasing, non-words, a code/API identifier split or mangled, a "
            "domain term that doesn't fit, or an obviously dropped word that breaks the "
            "sentence. Do NOT flag correct-but-informal speech. Quote the exact span. Output "
            'ONLY a JSON array: [{"phrase":"...","reason":"<=8 words"}].\n\nTRANSCRIPT:\n' + c
        )
        try:
            raw = invoke(prompt) or ""
            s, e = raw.find("["), raw.rfind("]")
            data = json.loads(raw[s : e + 1]) if s != -1 else []
        except Exception:
            data = []
        for d in data:
            if isinstance(d, dict) and d.get("phrase"):
                out.append(
                    {"phrase": str(d["phrase"]).strip(), "reason": str(d.get("reason", "")).strip()}
                )
    return out


def _locate(transcript, phrase: str) -> float | None:
    """Approx start time of `phrase` in the word stream (match first 1-2 words)."""
    key = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in phrase.split() if w]
    if not key:
        return None
    ws = transcript.words
    kws = [re.sub(r"[^a-z0-9]", "", w.text.lower()) for w in ws]
    for i in range(len(kws)):
        if kws[i] == key[0] and (len(key) < 2 or (i + 1 < len(kws) and kws[i + 1] == key[1])):
            return ws[i].start
    return None


def confidence_clusters(
    transcript, *, threshold: float = 0.35, window: int = 6, min_lowconf: int = 4
) -> list[dict]:
    """Windows where several low-confidence words CLUSTER — a stronger 'garbled/uncertain
    region' signal than isolated low-confidence words (a run of unsure words usually
    means a mistranscribed patch). Does NOT catch silent DROPS (no words emitted there —
    only a second ASR or a human catches those)."""
    ws = transcript.words
    out, i = [], 0
    while i < len(ws):
        w = ws[i]
        lc = [
            j
            for j in range(i, min(i + window, len(ws)))
            if ws[j].confidence is not None
            and ws[j].confidence < threshold
            and _is_content(ws[j].text)
        ]
        if len(lc) >= min_lowconf:
            end = lc[-1]
            span = " ".join(x.text for x in ws[i : end + 1])
            out.append({"start": w.start, "span": span, "n": len(lc)})
            i = end + 1
        else:
            i += 1
    return out


def _second_pass_disagreements(a, b) -> list[dict]:
    """Word-diff two transcripts; each disagreement span (with a's timestamp) is an
    acoustic-uncertainty suspect — no human needed.

    Casing is already folded (`.lower()`). Pure SEGMENTATION differences — two engines
    splitting the same letters differently ('Dark Mode'↔'DarkMode', 'web view'↔'WebView',
    'TV-specific'↔'TV specific') — are cosmetic noise, not disagreements: a diff span is
    DROPPED when its two sides are identical after removing word boundaries."""
    import difflib

    aw = [(w.text, re.sub(r"[^a-z0-9]", "", w.text.lower()), w.start) for w in a.words]
    bw = [re.sub(r"[^a-z0-9]", "", w.text.lower()) for w in b.words]
    ak = [x[1] for x in aw]
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ak, bw, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if "".join(ak[i1:i2]) == "".join(bw[j1:j2]):  # same letters, only segmentation differs
            continue
        span = " ".join(x[0] for x in aw[i1:i2]).strip() or "(nothing)"
        ts = aw[i1][2] if i1 < i2 and i1 < len(aw) else (aw[i1 - 1][2] if i1 > 0 else 0)
        out.append({"ts": _fmt(ts), "kind": "asr-disagreement", "detail": f"pass-1 heard {span!r}"})
    return out


def recover_dropped_spans(primary, second, *, min_run: int = 4):
    """SURGICALLY splice back spans the PRIMARY engine dropped but the SECOND heard.

    Parakeet (RNNT/TDT) is known to silently delete whole multi-word spans on
    music/non-speech (NeMo #14430, open) — a completeness failure with no internal
    signal. This diffs the two word streams and, for each run of >= `min_run`
    CONSECUTIVE words present ONLY in `second` (a real dropped segment, not a 1-2
    word substitution), inserts `second`'s words WITH THEIR TIMESTAMPS into `primary`,
    re-sorted by time. It touches ONLY the gaps — everything the primary got right
    (its full-context decode) is left untouched, so unlike a VAD-resegment front-end
    it introduces no context-loss regressions. `second` must carry word timing.

    Returns (merged_transcript, recovered_word_count)."""
    import difflib

    from .model import Transcript

    pk = [re.sub(r"[^a-z0-9]", "", w.text.lower()) for w in primary.words]
    sk = [re.sub(r"[^a-z0-9]", "", w.text.lower()) for w in second.words]
    adds = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, pk, sk, autojunk=False).get_opcodes():
        if tag == "insert" and (j2 - j1) >= min_run:  # primary emitted nothing here
            adds.extend(second.words[j1:j2])
    if not adds:
        return primary, 0
    merged = sorted(list(primary.words) + adds, key=lambda w: (w.start, w.end))
    return Transcript(language=primary.language, words=merged), len(adds)


def recover_repeated_runs(transcript, autosub, *, min_repeat: int = 5):
    """Collapse a REPEATED-TOKEN run ('the the the the the the') in the RAW transcript by
    replacing it with the auto-caption's text for that time span — BEFORE clean-read. The run
    can be EITHER a genuine speaker stutter (the ASR faithfully transcribed a disfluency the
    speaker actually said 5-6x) OR a stuck-decoder loop; either way it reads badly and the
    auto-caption is the cleaner source for that window. (If left, the clean-read rewrites the
    run into duplicated real text that the cue-level consult can no longer detect.) Word-level: find a dense run where ONE token repeats >= `min_repeat`
    times (small gaps allowed), replace it with the auto-caption words for its time span,
    distributed evenly. Returns (transcript, n_runs_fixed)."""
    from .model import Transcript, Word

    ws = transcript.words
    if not ws or autosub is None:
        return transcript, 0
    norm = [re.sub(r"[^a-z0-9]", "", w.text.lower()) for w in ws]
    out: list = []
    i = 0
    fixed = 0
    while i < len(ws):
        key = norm[i]
        if key and len(key) >= 2:
            j, last, cnt, gap = i, i, 1, 0
            while j + 1 < len(ws) and gap <= 2:  # extend the run, tolerating <=2-word gaps
                j += 1
                if norm[j] == key:
                    last, cnt, gap = j, cnt + 1, 0
                else:
                    gap += 1
            if cnt >= min_repeat and cnt >= 0.5 * (last - i + 1):  # a dense loop of `key`
                s, e = ws[i].start, ws[last].end
                toks = _cue_window_text(autosub, s, e).split()
                # collapse any residual >=3-in-a-row token loop in the auto-caption block
                clean: list[str] = []
                for t in toks:
                    if len(clean) >= 2 and clean[-1] == clean[-2] == t:
                        continue
                    clean.append(t)
                if clean:
                    span = max(e - s, 0.1) / len(clean)
                    out.extend(
                        Word(text=t, start=s + k * span, end=s + (k + 1) * span)
                        for k, t in enumerate(clean)
                    )
                fixed += 1
                i = last + 1
                continue
        out.append(ws[i])
        i += 1
    if fixed:
        return Transcript(language=transcript.language, words=out), fixed
    return transcript, 0


def _locate_sub(sub, phrase: str):
    """Timestamp of the cue containing `phrase` (or its first few words) in a Subtitle."""
    needle = phrase.strip().lower()
    head = " ".join(needle.split()[:4])
    for c in sub.cues:
        t = " ".join(c.lines).lower()
        if needle in t or (head and head in t):
            return c.start
    return None


def _normkey(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", x.lower())


def load_autosub(path):
    """Parse a YouTube auto-caption .vtt into a Subtitle, stripping inline <...> karaoke
    tags (auto-captions carry per-word <00:00:01.234><c> spans) so cue lines are plain text."""
    from .formats import read_vtt

    sub = read_vtt(path, "en")
    for c in sub.cues:
        c.lines = [ln for ln in (re.sub(r"<[^>]+>", "", x).strip() for x in c.lines) if ln]
    return sub


def _cue_window_text(sub, s: float, e: float) -> str:
    """Auto-caption text overlapping [s, e], with rolling-repeat dedup: YouTube auto-captions
    scroll, so cue N shows lines [A, B] and cue N+1 shows [B, C] — naively concatenating
    duplicates every line. Drop a line identical to the one just emitted."""
    lines: list[str] = []
    for c in sub.cues:
        if c.end > s and c.start < e:
            for ln in c.lines:
                ln = ln.strip()
                if ln and (not lines or lines[-1] != ln):
                    lines.append(ln)
    return " ".join(lines)


def autosub_consult(corrected, second_transcript, autosub, flagged_phrases, *, logger=None):
    """Phrase-level, agreement-gated auto-correction using the YouTube auto-caption as a
    THIRD ASR opinion. For each cue flagged implausible, align the primary text against the
    auto-caption text in that time window; for each differing span, splice in the auto-
    caption's tokens ONLY when the 2nd ASR pass ALSO has them (two independent sources
    agree the primary is the outlier). Fixes OOV/proper-noun mishears AND garbled multi-
    word spans (e.g. 'let's lower new tasks' -> 'lets them learn new tasks'). Never deletes
    primary content on its own. Mutates `corrected`; returns fixes [{cue, ts, was, now}].
    """
    import difflib

    from .linguistics import is_repeated_text

    if autosub is None:
        return []
    flags = [f.strip().lower() for f in flagged_phrases if f]
    fixes: list[dict] = []

    # PASS 1 — REPEATED-TOKEN RUNS: a run of identical cues ('the, the, the …'), whether a genuine
    # speaker stutter faithfully transcribed by BOTH ASRs (so the two-source gate can't help) or a
    # stuck-decoder loop. The auto-caption is a different engine and the cleaner read here. Find
    # each CONTIGUOUS run of repeated cues and DISTRIBUTE the auto-caption's text for the run's
    # whole span across its cues by duration — gathering the block ONCE and splitting it avoids the
    # cross-cue duplication a per-cue window causes.
    cues = corrected.cues
    i = 0
    while i < len(cues):
        if not is_repeated_text(" ".join(cues[i].lines))[0]:
            i += 1
            continue
        j = i
        while j < len(cues) and is_repeated_text(" ".join(cues[j].lines))[0]:
            j += 1
        run = cues[i:j]
        block = _cue_window_text(autosub, run[0].start, run[-1].end).strip()
        if block and not is_repeated_text(block)[0]:
            words = block.split()
            durs = [max(c.end - c.start, 0.1) for c in run]
            tot = sum(durs) or 1.0
            k = 0
            for idx, c in enumerate(run):
                take = (
                    len(words) - k
                    if idx == len(run) - 1
                    else max(round(len(words) * sum(durs[: idx + 1]) / tot) - k, 0)
                )
                seg = words[k : k + take]
                k += take
                was = " ".join(c.lines)
                c.lines = [" ".join(seg)] if seg else [" "]
                fixes.append(
                    {"cue": c.index, "ts": _fmt(c.start), "was": was, "now": " ".join(seg)}
                )
            if logger:
                logger(
                    f"autosub-consult: repeated-token run cues {run[0].index}-{run[-1].index} "
                    f"replaced from auto-caption ({len(words)} words distributed by duration)"
                )
        i = j

    # PASS 2 — phrase-level, agreement-gated consult on flagged cues (needs the 2nd ASR).
    if second_transcript is None:
        return fixes
    for c in corrected.cues:
        ctext = " ".join(c.lines)
        if is_repeated_text(ctext)[0]:
            continue  # a degenerate cue with no usable auto-caption block — leave for review
        low = ctext.lower()
        if not any(f and (f in low or " ".join(f.split()[:4]) in low) for f in flags):
            continue
        a_raw = _cue_window_text(autosub, c.start, c.end).split()
        if not a_raw:
            continue
        b_set = {
            _normkey(w.text)
            for w in second_transcript.words
            if c.start - 0.8 <= w.start <= c.end + 0.8
        }
        p_raw = ctext.split()
        p_norm = [_normkey(t) for t in p_raw]
        a_norm = [_normkey(t) for t in a_raw]
        out: list[str] = []
        changed = False
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, p_norm, a_norm, autojunk=False
        ).get_opcodes():
            if tag in ("replace", "insert"):
                repl, repl_norm = a_raw[j1:j2], a_norm[j1:j2]
                # gate: EVERY auto-caption replacement token must also be in the 2nd-ASR
                # window (two-source agreement) before it may overwrite the primary.
                if repl and all(n in b_set for n in repl_norm if n):
                    out.extend(repl)
                    changed = True
                    continue
            out.extend(p_raw[i1:i2])  # equal, or unconfirmed diff -> keep primary
        new_text = " ".join(out)
        if changed and new_text and new_text != ctext:
            c.lines = [new_text]
            fixes.append({"cue": c.index, "ts": _fmt(c.start), "was": ctext, "now": new_text})
            if logger:
                logger(
                    f"autosub-consult: cue {c.index} @ {_fmt(c.start)} corrected "
                    f"(auto-caption + 2nd-ASR agree): {ctext!r} -> {new_text!r}"
                )
    return fixes


def transcription_suspects(
    transcript,
    invoke=None,
    *,
    second_transcript=None,
    corrected=None,
    conf_threshold: float = 0.5,
    max_gap: float = 12.0,
    max_low_conf: int = 30,
) -> list[dict]:
    """Unified, timestamped, reference-free suspect list from confidence + gaps +
    (optional) LLM plausibility + (optional) a second ASR pass. Sorted by time.

    Low-confidence content words are CAPPED to the `max_low_conf` least-confident
    (a long/unscripted video yields hundreds — the implausibility flags are the
    high-value signal; keep the confidence list short and scannable)."""
    suspects: list[dict] = []
    lc = [
        s
        for s in low_confidence_spans(transcript, threshold=conf_threshold)
        if _is_content(s["word"])
    ]
    lc.sort(key=lambda s: s["confidence"])  # least confident first
    for s in lc[:max_low_conf]:
        suspects.append(
            {
                "ts": _fmt(s["start"]),
                "kind": "low-confidence",
                "detail": f"{s['word']!r} (conf {s['confidence']})",
            }
        )
    for g in transcription_gaps(transcript, max_gap=max_gap):
        suspects.append(
            {
                "ts": _fmt(g["start"]),
                "kind": "gap",
                "detail": f"{g['gap']}s silence after {g['after']!r} — dropped speech?",
            }
        )
    for cl in confidence_clusters(transcript, threshold=conf_threshold):
        suspects.append(
            {
                "ts": _fmt(cl["start"]),
                "kind": "uncertain-region",
                "detail": f"{cl['n']} unsure words: {cl['span'][:60]!r} — likely garbled",
            }
        )
    if invoke is not None:
        # Run PLAUSIBILITY on the corrected/delivered EN when given — else it flags spans the
        # clean-read correction already rewrote (stale false alarms). Confidence/gap above stay
        # on the raw transcript (they're genuine ASR-uncertainty pointers to the audio).
        ptext = (
            " ".join(ln for c in corrected.cues for ln in c.lines)
            if corrected is not None
            else transcript.text
        )
        for p in plausibility_suspects(invoke, ptext):
            ts = (
                _locate_sub(corrected, p["phrase"])
                if corrected is not None
                else _locate(transcript, p["phrase"])
            )
            suspects.append(
                {
                    "ts": _fmt(ts) if ts is not None else "?",
                    "kind": "implausible",
                    "detail": f"{p['phrase']!r} — {p['reason']}",
                }
            )
    if second_transcript is not None:
        suspects += _second_pass_disagreements(transcript, second_transcript)
    return sorted(suspects, key=lambda s: s["ts"])


def en_consistency_suspects(text: str) -> list[dict]:
    """Deterministic EN self-consistency check for SINGLE-token terms written with inconsistent
    casing — 'Fuel iX'/'Fuel IX' (the iX/IX token), camelCase/PascalCase or acronym variants.
    Only tokens with an INTERNAL capital (camel/Pascal, e.g. OpenTelemetry, iX) or all-caps
    acronyms are considered, so ordinary sentence-initial capitals ('The') aren't flagged.
    Multi-word / compound inconsistencies (e.g. 'OpenTelemetry' vs 'open telemetry') are left to
    the LLM proofread. Groups by lowercase; flags a key with >1 distinct surface form."""
    from collections import defaultdict

    forms: dict[str, set] = defaultdict(set)
    for m in re.finditer(r"\b[A-Za-z][A-Za-z0-9]{1,}\b", text):
        s = m.group(0)
        internal_cap = any(c.isupper() for c in s[1:])  # camelCase / PascalCase / iX
        all_caps = s.isupper() and 2 <= len(s) <= 6  # acronym
        if internal_cap or all_caps:
            forms[s.lower()].add(s)
    return [
        {"kind": "en-inconsistency", "ts": "—", "detail": f"'{k}' inconsistent casing: {sorted(v)}"}
        for k, v in sorted(forms.items())
        if len(v) > 1
    ]


def en_review(en, invoke, *, batch: int = 25) -> list[dict]:
    """A GENERAL editorial proofread of the EN subtitle/script — spelling, grammar, punctuation,
    terminology/product-name consistency, garbled/nonsensical phrasing — the standard review any
    script gets before delivery to translators, run REGARDLESS of whether the harness transcribed
    it (human transcripts carry errors too). LLM pass over numbered cues (batched) + the
    deterministic consistency check. Conservative: genuine errors only, not style/tone/natural
    speech fragments. Returns [{ts, kind, detail}] with cue references."""
    import json as _json

    cues = en.cues
    out: list[dict] = []
    for start in range(0, len(cues), batch):
        chunk = cues[start : start + batch]
        numbered = "\n".join(f"{start + j}: {' '.join(c.lines)}" for j, c in enumerate(chunk))
        prompt = (
            "You are PROOFREADING an English video subtitle script before it goes to translators. "
            "Flag ONLY genuine ERRORS in the numbered lines: spelling mistakes, grammar errors, "
            "wrong or INCONSISTENT terminology / product-name casing, garbled or nonsensical "
            "phrasing, and punctuation that changes meaning. Do NOT flag: informal tone, natural "
            "spoken fragments, or acceptable stylistic variation. Output ONLY a JSON array of "
            '{"cue": <line number>, "type": "spelling|grammar|terminology|garbled|punctuation", '
            '"detail": "<brief, name the fix>"}. Empty array if the lines are clean.\n\n' + numbered
        )
        try:
            raw = invoke(prompt) or ""
            s, e = raw.find("["), raw.rfind("]")
            items = _json.loads(raw[s : e + 1]) if s != -1 and e != -1 else []
        except Exception:
            items = []
        for it in items:
            if not isinstance(it, dict):
                continue
            i = it.get("cue")
            ts = _fmt(cues[i].start) if isinstance(i, int) and 0 <= i < len(cues) else "—"
            out.append(
                {
                    "ts": ts,
                    "kind": f"en-{it.get('type', 'issue')}",
                    "detail": f"[cue {i}] {str(it.get('detail', '')).strip()}",
                }
            )
    # dedupe consistency findings against the LLM ones by term, then append
    out += en_consistency_suspects(" ".join(" ".join(c.lines) for c in cues))
    return out


def en_autofix(en, findings: list[dict], invoke) -> tuple:
    """STANDARD EN-review procedure (mirrors QE auto-repair of majors): auto-apply the CLEAR-ERROR
    findings to the EN, HOLD the judgment calls for the operator.

    A finding is CLEAR when the fix is MECHANICAL — spelling, grammar, punctuation, casing, or a
    well-known terminology / product-name normalization — and does NOT guess a word from a garble,
    change a fact, or alter meaning. It is HELD (judgment) when the fix would GUESS garbled content
    (e.g. a mis-transcribed brand/word), correct a FACT, or change meaning; those stay flagged so
    the operator adjudicates (policy: auto-apply clear-error fixes, a human adjudicates the rest).

    Only cue-referenced findings ('[cue N]') are auto-fixable; global consistency findings are held.
    Returns (new_en, applied, held): `new_en` is the EN with clear fixes applied (a fresh Subtitle,
    original untouched); applied/held are finding dicts (applied annotated with `new_text`)."""
    import json as _json

    from .model import Cue, Subtitle

    cue_findings, held = [], []
    for f in findings:
        m = re.search(r"\[cue (\d+)\]", f.get("detail", ""))
        if m and 0 <= int(m.group(1)) < len(en.cues):
            cue_findings.append((int(m.group(1)), f))
        else:
            held.append(f)  # global/consistency finding → operator
    if not cue_findings:
        return en, [], held
    blocks = []
    for k, (ci, f) in enumerate(cue_findings):
        blocks.append(
            f"{k}. cue {ci} | current: {' '.join(en.cues[ci].lines)!r}\n"
            f"   finding: {f.get('detail', '')}"
        )
    prompt = (
        "You are triaging proofreading findings on an English subtitle before translation. For EACH "
        "finding decide:\n"
        '- "clear": a MECHANICAL fix — spelling, grammar, punctuation, casing, or a well-known '
        "terminology/product-name normalization — that does NOT guess a garbled word, change a fact, "
        "or alter meaning. Give the FULL corrected cue text.\n"
        '- "judgment": the fix would GUESS a word from a garble, correct a FACT, or change meaning. '
        "Do NOT rewrite it; leave for a human.\n"
        "Keep corrections minimal (fix only what the finding names; preserve the rest verbatim). "
        'Output ONLY a JSON array of {"i": <number>, "verdict": "clear"|"judgment", '
        '"new_text": "<full corrected cue, only if clear>"}.\n\n' + "\n".join(blocks)
    )
    try:
        raw = invoke(prompt) or ""
        s, e = raw.find("["), raw.rfind("]")
        verdicts = _json.loads(raw[s : e + 1]) if s != -1 and e != -1 else []
    except Exception:
        verdicts = []
    vmap = {v["i"]: v for v in verdicts if isinstance(v, dict) and "i" in v}
    cues = list(en.cues)
    applied = []
    for k, (ci, f) in enumerate(cue_findings):
        v = vmap.get(k)
        new_text = (v or {}).get("new_text", "")
        if (
            v
            and v.get("verdict") == "clear"
            and isinstance(new_text, str)
            and new_text.strip()
            and new_text.strip() != " ".join(en.cues[ci].lines).strip()
        ):
            # re-wrapped: a corrected cue written as one line broke max_cpl and the EN gate blocked the
            # run (one video, 2026-10-26, 73- and 78-char lines)
            from .en_cleanread import wrap2

            cues[ci] = Cue(
                start=en.cues[ci].start, end=en.cues[ci].end, lines=wrap2(new_text.strip())
            )
            applied.append({**f, "new_text": new_text.strip()})
        else:
            held.append(f)
    new_en = Subtitle(en.language, cues) if applied else en
    return new_en, applied, held


def write_suspects_md(suspects: list[dict], path, video: str = "") -> None:
    from pathlib import Path

    lines = [
        f"# Transcription suspects (reference-free) — {video}".rstrip(),
        "",
        "No human caption was used. These are spots to spot-check against the AUDIO — "
        "the harness is unsure, or they read as likely errors. Kinds: low-confidence "
        "(ASR unsure), gap (possible dropped speech), implausible (reads wrong), "
        "asr-disagreement (two ASR passes differ).",
        "",
        "| Time | Kind | Detail |",
        "|---|---|---|",
    ]
    for s in suspects:
        lines.append(f"| {s['ts']} | {s['kind']} | {s['detail']} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
