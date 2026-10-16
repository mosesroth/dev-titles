"""Translation memory (TM): a persistent, GROWING store of human-approved source->target
segments, retrieved at translate time.

Upgrades the (static, corpus-mined) exemplar bank in the two ways the research flagged as
the highest-evidence consistency win for a video SERIES:

  - EXACT-MATCH REUSE: a segment whose source matches a prior approved entry (whitespace/
    case-normalized) reuses that target VERBATIM — identical rendering of repeated
    boilerplate (intros, CTAs, recurring product lines) across the series, which a
    glossary + generic exemplars cannot guarantee.
  - GROWTH: seeded from the human parallel corpus AND appended with each approved batch,
    so later videos benefit from earlier approved translations (batch N reuses batch N-1).

Fuzzy retrieval reuses `ExemplarBank` (Jaccard token overlap) so near-duplicate segments
inject the most relevant approved pairs as in-context examples — Moslem et al. (EAMT 2023)
found fuzzy TM in-context took GPT en->zh 38.7 -> 49.1 spBLEU, beating Google/NLLB.

Stored as JSONL (portable, appendable); `export_tmx` writes the industry-standard TMX so a
CAT tool / the localization team can consume it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .exemplars import Exemplar, ExemplarBank


def _norm(s: str) -> str:
    """Normalization key for EXACT match: case-folded, whitespace-collapsed. Kept strict
    (keeps content punctuation) so we only reuse a target for a genuinely-identical line."""
    return re.sub(r"\s+", " ", s.strip().lower())


@dataclass
class TMEntry:
    source: str
    target: str
    lang: str
    video: str = ""


class TranslationMemory:
    def __init__(self):
        self.entries: list[TMEntry] = []
        self._exact: dict[tuple[str, str], str] = {}  # (lang, norm-source) -> target

    def add(self, source: str, target: str, lang: str, video: str = "") -> None:
        s, t = source.strip(), target.strip()
        if not s or not t:
            return
        self.entries.append(TMEntry(s, t, lang, video))
        self._exact[(lang, _norm(s))] = t  # last write wins (newest approval)

    def exact(self, source: str, lang: str) -> str | None:
        """The approved target for an identical (normalized) source, else None."""
        return self._exact.get((lang, _norm(source)))

    def import_tmx(self, path, langs=("zh-Hans", "ja"), max_pairs=None) -> int:
        """Stream an industry TMX export and add EN->target pairs. Memory-safe via iterparse
        (vendor Gold/Silver TMs can be GB-scale). Maps tmx codes en-us/zh-cn/ja-jp to ours.
        Import Gold AFTER Silver so Gold (LQA'ed) wins the last-write exact key. Deferred/
        opt-in: such TMs are typically UI/doc strings, so subtitle exact-match hits are rare."""
        import xml.etree.ElementTree as ET

        XLANG = "{http://www.w3.org/XML/1998/namespace}lang"

        def norm_lang(code: str) -> str:
            c = (code or "").lower()
            return (
                "en"
                if c.startswith("en")
                else "zh-Hans" if c.startswith("zh") else "ja" if c.startswith("ja") else c
            )

        n = 0
        for _, elem in ET.iterparse(str(path), events=("end",)):
            if elem.tag.split("}")[-1] != "tu":
                continue
            by: dict[str, str] = {}
            for tuv in elem.iter():
                if tuv.tag.split("}")[-1] != "tuv":
                    continue
                lg = norm_lang(tuv.get(XLANG) or tuv.get("lang") or "")
                seg = next(
                    (
                        ("".join(ch.itertext())).strip()
                        for ch in tuv
                        if ch.tag.split("}")[-1] == "seg"
                    ),
                    None,
                )
                if seg:
                    by[lg] = seg
            s = by.get("en")
            if s:
                for lg in langs:
                    if by.get(lg):
                        self.add(s, by[lg], lg)
                        n += 1
            elem.clear()
            if max_pairs and n >= max_pairs:
                break
        return n

    def bank(self, lang: str, **kwargs) -> ExemplarBank:
        """An ExemplarBank over this TM's entries for `lang` (fuzzy retrieval at
        translate time). `kwargs` pass through to ExemplarBank (e.g. max_src_chars)."""
        b = ExemplarBank(lang=lang, **kwargs)
        b.add([Exemplar(e.source, e.target, e.lang, e.video) for e in self.entries])
        return b

    def exclude(self, video: str) -> "TranslationMemory":
        """Drop entries from `video` (held-out: don't let a video reuse its OWN approved
        translation when re-processing / measuring quality). Returns a new TM."""
        if not video:
            return self
        out = TranslationMemory()
        for e in self.entries:
            if e.video != video:
                out.add(e.source, e.target, e.lang, e.video)
        return out

    def __len__(self) -> int:
        return len(self.entries)

    def counts(self) -> dict:
        from collections import Counter

        return dict(Counter(e.lang for e in self.entries))

    # ---- persistence ----------------------------------------------------------
    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for e in self.entries:
                fh.write(json.dumps(e.__dict__, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path) -> "TranslationMemory":
        tm = cls()
        p = Path(path)
        if p.exists():
            for line in p.open(encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    tm.add(d["source"], d["target"], d["lang"], d.get("video", ""))
        return tm

    # ---- seeding / growth -----------------------------------------------------
    def seed_from_corpus(
        self, root="input/channel", langs=("ja", "zh-Hans")
    ) -> "TranslationMemory":
        """Populate from the human parallel corpus (time-aligned EN sentence -> human
        target). The starting TM before any batch approvals."""
        from .corpus import discover_channel_corpus
        from .exemplars import align_by_time
        from .formats import read_srt

        for vid, en_path, targets in discover_channel_corpus(root):
            try:
                en = read_srt(str(en_path), "en")
            except Exception:
                continue
            for lang in langs:
                if lang in targets:
                    try:
                        human = read_srt(str(targets[lang]), lang)
                    except Exception:
                        continue
                    for ex in align_by_time(en, human, lang, vid):
                        self.add(ex.source, ex.target, lang, vid)
        return self

    def add_from_subtitles(self, en_sub, target_sub, lang: str, video: str = "") -> int:
        """Append an APPROVED (EN, target) pair to the TM by time-aligning EN sentences
        to the target — how a reviewed batch grows the memory. Returns entries added."""
        from .exemplars import align_by_time

        before = len(self.entries)
        for ex in align_by_time(en_sub, target_sub, lang, video):
            self.add(ex.source, ex.target, lang, video)
        return len(self.entries) - before

    def export_tmx(self, path, src_lang: str = "en") -> None:
        """Write TMX 1.4 (one <tu> per entry) — the portable interchange format."""
        from xml.sax.saxutils import escape

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<tmx version="1.4">',
            '<header creationtool="dev_titles" segtype="sentence" '
            f'srclang="{src_lang}" datatype="plaintext"/>',
            "<body>",
        ]
        for e in self.entries:
            lines.append(
                f'<tu><tuv xml:lang="{src_lang}"><seg>{escape(e.source)}</seg></tuv>'
                f'<tuv xml:lang="{e.lang}"><seg>{escape(e.target)}</seg></tuv></tu>'
            )
        lines += ["</body>", "</tmx>"]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
