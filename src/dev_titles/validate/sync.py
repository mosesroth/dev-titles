"""SYNC CHECK: is a translated track still in step with the speech?

Why: the retired 1:1 "reflow" step (Sept 2026) left some zh tracks one or two cues behind the English
(one line still on screen while the next sentence was spoken; median 5 s off),
and every later step, including a vendor review that kept our timing, preserved it. The cue-level
validator can't see this: each cue is well-formed, just showing the wrong sentence.

Method (LLM-free): Latin anchors — names, products, numbers-with-letters — appear verbatim in both
tracks. For each anchor in a target cue, find the nearest EN cue that contains it and take the
time offset between the two cue midpoints. In-step tracks measure ~0.7-1.4 s median (long merged
cues add noise); drifted ones 2.2-5 s, with 30-70% of anchors more than 3 s off.
"""

from __future__ import annotations

import re
import statistics

_ANCHOR = re.compile(r"[A-Za-z][A-Za-z0-9.+-]{3,}")


def sync_offsets(target_cues, en_cues) -> list[float]:
    """Absolute offsets (s) between each Latin anchor in the target and its nearest EN occurrence."""
    en = [((e.start + e.end) / 2, " ".join(e.lines)) for e in en_cues]
    out = []
    for c in target_cues:
        mid = (c.start + c.end) / 2
        for tok in set(_ANCHOR.findall(" ".join(c.lines))):
            pat = re.compile(r"\b" + re.escape(tok) + r"\b", re.I)
            hits = [m for m, t in en if pat.search(t)]
            if hits:
                out.append(abs(mid - min(hits, key=lambda m: abs(m - mid))))
    return out


def sync_report(target_cues, en_cues, *, max_median: float = 1.8, max_far_share: float = 0.30,
                min_anchors: int = 15) -> dict:
    """{'anchors', 'median', 'far_share', 'drifted'}; drifted is None when there are too few anchors."""
    d = sync_offsets(target_cues, en_cues)
    if len(d) < min_anchors:
        return {"anchors": len(d), "median": None, "far_share": None, "drifted": None}
    med = statistics.median(d)
    far = sum(x > 3 for x in d) / len(d)
    return {"anchors": len(d), "median": round(med, 2), "far_share": round(far, 2),
            "drifted": med > max_median or far > max_far_share}
