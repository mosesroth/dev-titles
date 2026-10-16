"""Re-time a corrected transcript onto the original WhisperX word timestamps.

The glossary-aware correction pass (`cleanup.correct_transcript`) rewrites words
(jargon fixes, and re-joining spelled-out code identifiers like
'next focus right' -> 'nextFocusRight'), so the corrected text no longer aligns
1:1 with the ASR word timings. Rather than a second forced-alignment audio pass,
we align the corrected tokens to the original word-timed tokens (edit-distance
backtrace) and inherit timing:
  - match / substitution -> take the aligned original word's [start, end];
  - MERGE (a corrected token that swallowed several original words, e.g. an
    identifier) -> span from the first to the last consumed original word;
  - insertion (no original) -> zero-width at the previous end (rare — correction
    is not supposed to add content).
Deterministic, no audio required, unit-tested.
"""

from __future__ import annotations

import re

from ..model import Transcript, Word

_TOK = re.compile(r"\S+")


def _norm(tok: str) -> str:
    return re.sub(r"[^a-z0-9]", "", tok.lower())


def retime_corrected(orig: Transcript, corrected_text: str) -> Transcript:
    """Return a Transcript whose words are the corrected tokens, each carrying a
    [start, end] inherited from the original word timings it aligns to."""
    ow = orig.words
    ct = _TOK.findall(corrected_text)
    on = [_norm(w.text) for w in ow]
    cn = [_norm(t) for t in ct]
    n, m = len(on), len(cn)
    if not m:
        return Transcript(language=orig.language, words=[])
    if not n:
        return Transcript(language=orig.language, words=[Word(t, 0.0, 0.0) for t in ct])
    # edit-distance DP over normalized tokens
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if on[i - 1] == cn[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    # backtrace: assign each corrected token j the original words it consumes
    i, j = n, m
    consumed: list[list[int]] = [[] for _ in range(m)]  # corrected idx -> orig idxs
    unmatched: list[int] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if on[i - 1] == cn[j - 1] else 1):
            consumed[j - 1].append(i - 1)
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            # a dropped original word: attach it to the corrected token to its
            # right (the one we most recently stepped to) so a merge spans it
            (consumed[j] if j < m else consumed[m - 1]).append(i - 1)
            i -= 1
        else:
            unmatched.append(j - 1)
            j -= 1
    words: list[Word] = []
    last_end = ow[0].start
    for jj, tok in enumerate(ct):
        idxs = sorted(consumed[jj])
        if idxs:
            start, end = ow[idxs[0]].start, ow[idxs[-1]].end
            last_end = end
        else:  # insertion — zero-width at the running position
            start = end = last_end
        words.append(Word(text=tok, start=start, end=max(end, start)))
    return Transcript(language=orig.language, words=words)
