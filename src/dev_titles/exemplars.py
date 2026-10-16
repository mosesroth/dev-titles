"""Few-shot style exemplars mined from human parallel subtitles.

The single highest-ROI lever toward human parity (per the reviewer audit): prose
style directives under-deliver, but showing the model a few *approved* EN->target
cue pairs teaches house register, mood, phrasing rhythm, and punctuation density
directly. We mine those pairs from the human ja/zh subtitles the docs team already
produced, align them to the EN source by time overlap, and retrieve the most
lexically-relevant few per segment at translate time.

Mining EXEMPLARS from a video's human subs is only sound when that video is NOT
the one being evaluated — otherwise the reference leaks into the hypothesis. The
eval harness enforces the train/held-out split; this module just builds/queries
the bank.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .model import Subtitle
from .resegment import merge_en_to_sentences


@dataclass
class Exemplar:
    source: str  # EN sentence
    target: str  # human translation (target-language span aligned by time)
    lang: str  # target language code
    video: str  # provenance (for the train/held-out split)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def align_by_time(
    en: Subtitle, human: Subtitle, lang: str, video: str, *, min_overlap: float = 0.2
) -> list[Exemplar]:
    """Align EN *sentences* to the human target by time overlap.

    Each EN sentence [s,e] is paired with the concatenation of the human cues that
    overlap it (overlap fraction of the human cue >= `min_overlap`, so a cue that
    merely grazes the boundary isn't double-counted into both neighbours). Yields
    one Exemplar per EN sentence that gets a non-empty target span."""
    sentences = merge_en_to_sentences(en)
    out: list[Exemplar] = []
    for sent in sentences:
        parts: list[str] = []
        for cue in human.cues:
            ov = min(sent.end, cue.end) - max(sent.start, cue.start)
            if ov <= 0:
                continue
            dur = max(cue.duration, 1e-6)
            if ov / dur >= min_overlap:
                parts.append(_clean(cue.text))
        tgt = _clean(" ".join(p for p in parts if p))
        src = _clean(sent.text)
        if src and tgt:
            out.append(Exemplar(src, tgt, lang, video))
    return out


def _tokens(text: str) -> set[str]:
    """Lowercased alphanumeric word tokens for lexical overlap scoring (EN side)."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass
class ExemplarBank:
    """Retrieval over mined EN->target exemplars for one target language."""

    lang: str
    exemplars: list[Exemplar] = field(default_factory=list)
    max_src_chars: int = 160  # skip over-long pairs (keep the prompt compact)

    def add(self, exemplars: list[Exemplar]) -> "ExemplarBank":
        for ex in exemplars:
            if ex.lang == self.lang and len(ex.source) <= self.max_src_chars:
                self.exemplars.append(ex)
        return self

    def select(self, query: str, k: int = 4) -> list[tuple[str, str]]:
        """Top-`k` (source, target) pairs most lexically relevant to `query`.

        Ranked by Jaccard token overlap with the query EN, then:
          - pad (when overlap is sparse) with pairs whose LENGTH is closest to the
            query — NOT the shortest, which are the least-punctuated, most
            fragmentary cues and were actively teaching a comma-free, syntax-free
            style (audit finding);
          - dedupe by source video so one video can't dominate the k window;
          - guarantee at least one pair that actually contains a target comma, so
            the model always sees where the house style DOES place ，/、.
        Skips a pair whose source is (nearly) the query itself (no answer leak)."""
        q = _tokens(query)
        qnorm = _clean(query).lower()
        qlen = len(_clean(query))
        scored: list[tuple[float, Exemplar]] = []
        for ex in self.exemplars:
            if _clean(ex.source).lower() == qnorm:
                continue
            t = _tokens(ex.source)
            if not t:
                continue
            union = len(q | t) or 1
            scored.append((len(q & t) / union, ex))
        # relevant first; length-closest as the tiebreak/padding order
        scored.sort(key=lambda x: (-x[0], abs(len(x[1].source) - qlen)))
        chosen: list[Exemplar] = []
        seen_videos: set[str] = set()
        for score, ex in scored:
            if len(chosen) >= k:
                break
            if ex.video in seen_videos and len(scored) > k * 3:
                continue  # one-per-video while the bank is rich enough
            chosen.append(ex)
            seen_videos.add(ex.video)
        # ensure >=1 comma-bearing exemplar so sparse-comma style isn't over-taught
        if chosen and not any(("，" in e.target or "、" in e.target) for e in chosen):
            for score, ex in scored:
                if "，" in ex.target or "、" in ex.target:
                    chosen[-1] = ex
                    break
        return [(ex.source, ex.target) for ex in chosen]


def punct_profile(exemplars: list[Exemplar]) -> dict[str, float]:
    """Punctuation density of the human targets, per 100 Han/kana characters.

    Feeds the prompt a MEASURED comma target instead of a vague 'few commas' —
    the audit found the zero-comma directive over-corrected (zh house rate is
    ~1.2 ，/100 chars, not 0), forcing 那么/就 padding and unparsable runs."""
    import re as _re

    body = "".join(e.target for e in exemplars)
    han = len(_re.findall(r"[぀-ヿ㐀-鿿]", body)) or 1

    def per100(ch):
        return round(body.count(ch) / han * 100, 2)

    return {"，": per100("，"), "、": per100("、"), "！": per100("！"), "chars": han}


def mine_corpus(
    specs: list[tuple[str, Subtitle, Subtitle]],
    lang: str,
    *,
    exclude_video: str | None = None,
) -> ExemplarBank:
    """Build a bank for `lang` from parallel `(video, en_subtitle, human_subtitle)`
    specs (the subtitles already loaded), excluding `exclude_video` (the held-out
    eval video) so its reference never leaks into exemplars. Most callers build banks
    directly via `align_by_time` + `ExemplarBank.add`; this is a harness convenience."""
    bank = ExemplarBank(lang=lang)
    for video, en, human in specs:
        if exclude_video and video == exclude_video:
            continue
        bank.add(align_by_time(en, human, lang, video))
    return bank
