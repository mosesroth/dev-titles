"""EN cue builder (pipeline step [2]): word-timed transcript -> WebVTT master.

Groups ASR words into subtitle cues honoring the English LangSpec (max chars/
line, reading speed, <=2 lines, duration, gaps) and breaks at natural boundaries
(sentence-final punctuation, then pauses). The EN master is the timing/reference
source; CJK cues are RE-DERIVED later (segment/), never reused from here.
"""

from __future__ import annotations

import re

from .config import LangSpec, spec_for
from .linguistics import end_punct
from .model import Cue, Subtitle, Transcript, Word
from .segment.cjk import break_lines

# Standalone speech disfluencies to drop unconditionally. These are technical
# talks, not dramatic dialogue, so filler interjections carry no content and only
# waste cue space / inflate reading speed. Kept to UNAMBIGUOUS interjections —
# false starts and word repetitions are left to the LLM cleanup (context needed).
FILLER_WORDS = frozenset(
    {
        "um",
        "umm",
        "ummm",
        "uh",
        "uhh",
        "uhhh",
        "uhm",
        "erm",
        "er",
        "err",
        "mm",
        "mmm",
        "mhm",
        "hmm",
        "hmmm",
        "uh-huh",
        "mm-hmm",
        "huh",
    }
)


def strip_disfluencies(transcript: Transcript) -> Transcript:
    """Drop standalone filler interjections (um/uh/er/mm/hmm…) from the word
    stream, with their timing, so they never occupy a cue. Deterministic."""
    kept = [
        w for w in transcript.words if re.sub(r"[^a-z-]", "", w.text.lower()) not in FILLER_WORDS
    ]
    return Transcript(language=transcript.language, words=kept)


_NONSPEECH = re.compile(r"^\s*[\[(].*?[\])]\s*$|^♪+$")


def strip_nonspeech_cues(transcript: Transcript) -> Transcript:
    """Drop WhisperX non-speech sound tokens ([music], (applause), ♪) from the word
    stream. Per the DCMP Captioning Key, background music/sound is captioned ONLY
    when essential to understanding (and never if <5s) — our technical tutorials
    essentially never have essential non-speech audio, and Whisper's tokens are
    unformatted artifacts. If a video ever needs an essential sound captioned, it
    should be added deliberately as a lowercase descriptive bracket ([upbeat music])
    in its own cue, not left as a raw ASR token."""
    kept = [w for w in transcript.words if not _NONSPEECH.match(w.text.strip())]
    return Transcript(language=transcript.language, words=kept)


# Cue-level sound tags in an imported caption. Restricted to NON-SPEECH words so a real
# parenthetical in speech ("Single Sign-On (SSO)") is never removed.
_SOUND_TAG = re.compile(
    r"\s*[\[(](?:upbeat |soft |gentle |background |light |dramatic )?"
    r"(?:music|applause|laughter|laughs|laughing|cheering|cheers|inaudible|silence|crosstalk|"
    r"chuckles|chuckling|sighs|coughs|beep|bell|noise|sound|instrumental)[\])]\s*|\s*♪+[^♪]*♪*\s*",
    re.IGNORECASE,
)


def strip_sound_tags(sub: Subtitle) -> Subtitle:
    """Remove non-speech sound tags ([music], (applause), [laughter], ♪…) from caption cues — the
    cue-level twin of strip_nonspeech_cues (which only runs on ASR word streams inside
    build_en_cues). Without it, `localize` on a clean human caption passed '[music]' straight into
    translation (BuilderToolsAI ja got '[音楽]'). A cue left empty is dropped."""
    out = []
    for c in sub.cues:
        lines = [re.sub(r" {2,}", " ", _SOUND_TAG.sub(" ", ln)).strip() for ln in c.lines]
        lines = [ln for ln in lines if ln]
        if lines:
            out.append(Cue(start=c.start, end=c.end, lines=lines))
    return Subtitle(sub.language, out)


def low_confidence_spans(transcript: Transcript, threshold: float = 0.5) -> list[dict]:
    """Words the ASR was unsure about — flag them for the mandatory human review
    pass (W3C WAI: raw auto-captions must be human-confirmed). Technical proper
    nouns / APIs are exactly where confidence dips AND where errors hide, so this
    is the highest-leverage automatable review signal. Words with no confidence
    score are skipped (nothing to judge)."""
    return [
        {"word": w.text, "start": round(w.start, 2), "confidence": round(w.confidence, 3)}
        for w in transcript.words
        if w.confidence is not None and w.confidence < threshold
    ]


def transcription_gaps(transcript: Transcript, max_gap: float = 12.0) -> list[dict]:
    """Large silent stretches between consecutive transcribed words — a caption
    completeness check (WCAG/FCC: captions must be complete). ASR can silently drop
    low-energy speech; an unexpectedly long gap is where that would show, so flag
    each for a human to confirm it's real silence, not dropped audio."""
    ws = transcript.words
    gaps = []
    for a, b in zip(ws, ws[1:]):
        if b.start - a.end >= max_gap:
            gaps.append(
                {
                    "after": a.text,
                    "start": round(a.end, 2),
                    "end": round(b.start, 2),
                    "gap": round(b.start - a.end, 1),
                }
            )
    return gaps


def build_en_cues(
    transcript: Transcript,
    spec: LangSpec | None = None,
    *,
    max_pause: float = 0.7,
    drop_disfluencies: bool = True,
) -> Subtitle:
    """Build EN cues from a word-timed transcript.

    Flush the current cue when: a pause >= `max_pause` precedes the next word;
    the text would exceed the cue char budget (max_cpl * max_lines); the cue
    reaches max duration; or a sentence-final mark lands past the half-budget.
    Cues shorter than min_dur are extended to it.
    """
    spec = spec or spec_for("en")
    transcript = strip_nonspeech_cues(transcript)  # DCMP: no bare [music] artifacts
    if drop_disfluencies:
        transcript = strip_disfluencies(transcript)
    max_lines = spec.max_lines or 2
    half_budget = 0.5 * spec.max_cpl * max_lines
    ends = end_punct(spec.lang)

    # 2 lines is the standard; a 3rd is a documented exception (DCMP/BBC) used here
    # only to absorb a would-be orphan rather than strand it as its own cue.
    hard_lines = max_lines + 1

    def fits(words: list[Word], limit: int | None = None) -> bool:
        limit = limit or max_lines
        return len(break_lines(" ".join(w.text for w in words), spec)) <= limit

    # Pass 1: group words into cue-sized word lists (line-count aware).
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for w in transcript.words:
        if cur and (w.start - cur[-1].end) >= max_pause:
            groups.append(cur)
            cur = []
        if cur and not fits(cur + [w]):  # adding w would overflow max_lines
            groups.append(cur)
            cur = []
        cur.append(w)
        text = " ".join(x.text for x in cur)
        if (w.end - cur[0].start) >= spec.max_dur:
            groups.append(cur)
            cur = []
        elif w.text and w.text[-1] in ends and len(text) >= half_budget:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)

    # Pass 2: repair tiny orphans (<=2 words). Prefer merging into the previous
    # group; else rebalance by pulling its last word down so no 1-word cue remains.
    i = 1
    while i < len(groups):
        g = groups[i]
        if len(g) <= 2:
            prev = groups[i - 1]
            dur_ok = (g[-1].end - prev[0].start) <= spec.max_dur
            if dur_ok and fits(prev + g):  # merge within 2 lines
                groups[i - 1] = prev + g
                groups.pop(i)
                continue
            if dur_ok and fits(prev + g, hard_lines):  # 3-line exception to absorb orphan
                groups[i - 1] = prev + g
                groups.pop(i)
                continue
            if len(g) == 1 and len(prev) >= 3:  # else rebalance: no 1-word cue
                groups[i - 1], groups[i] = prev[:-1], [prev[-1]] + g
        i += 1

    def _cue(words: list[Word]) -> Cue:
        start, end = words[0].start, words[-1].end
        if end - start < spec.min_dur:
            end = start + spec.min_dur
        return Cue(start=start, end=end, lines=break_lines(" ".join(w.text for w in words), spec))

    cues = [_cue(g) for g in groups]
    # Clamp a min_dur-extended cue so it can't overlap the next one on dense speech
    # (leave a min_gap where possible; never invert start/end).
    min_gap = spec.min_gap_frames / spec.fps
    for i in range(len(cues) - 1):
        limit = cues[i + 1].start - min_gap
        if cues[i].end > limit:
            cues[i] = Cue(
                start=cues[i].start,
                end=max(limit, cues[i].start),
                lines=cues[i].lines,
                index=cues[i].index,
            )
    return Subtitle(language=spec.lang, cues=cues).renumber()


def dedup_rolling_cues(sub: Subtitle) -> Subtitle:
    """Merge consecutive cues with identical text — the YouTube rolling-caption artifact where
    the same line is emitted twice with split timings (cue N shows [A], cue N+1 re-shows [A]
    with a later end). Keeps the first cue's start and extends its end over the duplicate.
    Returns a new renumbered Subtitle. A no-op on captions that don't roll."""
    out: list[Cue] = []
    for c in sub.cues:
        t = " ".join(c.lines).strip()
        if out and " ".join(out[-1].lines).strip() == t:
            out[-1].end = c.end  # absorb the duplicate's time span
            continue
        out.append(Cue(start=c.start, end=c.end, lines=[t] if t else list(c.lines)))
    return Subtitle(language=sub.language, cues=out).renumber()


def caption_to_transcript(sub: Subtitle) -> Transcript:
    """Adapt a caption (cue-level) into a word-level Transcript so a human/imported caption gets
    the SAME clean-read correction path as an ASR transcript. Each cue's text is split into words
    with timings distributed evenly across the cue span; no per-word confidence (so nothing is
    flagged low-confidence). Pair with build_en_cues to re-segment the corrected words into cues."""
    words: list[Word] = []
    for c in sub.cues:
        toks = " ".join(c.lines).split()
        if not toks:
            continue
        span = max(c.end - c.start, 1e-3) / len(toks)
        for i, tok in enumerate(toks):
            words.append(Word(text=tok, start=c.start + i * span, end=c.start + (i + 1) * span))
    return Transcript(language=sub.language, words=words)
