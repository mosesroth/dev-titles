"""Which capitalized names stay Latin in a track — decided per name, not left to the translator.

Reviewers keep technology, product and company names in Latin and translate titles, places,
events and departments. Measured on the 19 vendor-reviewed zh videos (2026-10-27): of 143
capitalized multi-word names not in the glossary, 75% of uses stayed Latin (Universal React 8/8,
VS Code 8/8); the translated ones were job titles, places, events and departments (Vice President,
Product Manager, San Francisco, Dev Meetup...).

The names policy (entities.py) asks for these up front, but as one long list it misses some — one
run wrote 通用React all 8 times. So after translation, every capitalized name the track never keeps
Latin gets a second look:
  1. the reviewers' corpus (reviewed finals + TM): their majority choice for this exact name;
  2. else one short classification call for the unseen names: technology/product/company -> Latin;
     title/place/event/department/ordinary phrase -> translate.
The pipeline (localize._latin_name_consistency) re-translates the sentences that lost a Latin
name and keeps a re-translation only if no judge finds the sentence worse, then lists the name in
the review. Leave-one-out against the reviewers (scripts/eval/latin_names_check.py, 436 name
decisions): 92% agree; 11 would force Latin where reviewers translated (e.g. a company or service
name the reviewers rendered in Chinese), which is what the sentence check guards. Requiring a second model family to agree (`second`)
removed only 2 of those 11 and lost 9 correct calls, so the pipeline doesn't use it.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_LEAD_WORDS = frozenset(
    "so and but or the a an i we our my you your this that it its if when then now yes yeah "
    "okay ok well also just because what how why where who today for with their both as can does do "
    "is are was hi hey like".split()
)
_NAME = re.compile(r"\b[A-Z][A-Za-z0-9]+(?: [A-Z][A-Za-z0-9]+)+\b")


def names_in(text: str) -> list[str]:
    """Capitalized multi-word names in an English line, sentence-initial filler stripped
    ("So Universal React" -> "Universal React")."""
    out = []
    for m in _NAME.finditer(text or ""):
        words = m.group(0).split()
        while words and words[0].lower() in _LEAD_WORDS:
            words = words[1:]
        if len(words) >= 2:
            out.append(" ".join(words))
    return out


def _has(name: str, text: str) -> bool:
    return name.replace(" ", "").lower() in re.sub(r"\s", "", text or "").lower()


def never_latin(records, exclude=()) -> list[str]:
    """Names that occur in the EN of the track but never in Latin in its target."""
    ex = {e.lower() for e in exclude}
    seen: dict[str, bool] = {}
    for r in records:
        for n in names_in(r.src):
            if n.lower() in ex or any(re.search(rf"\b{re.escape(e)}\b", n.lower()) for e in ex):
                continue
            seen[n] = seen.get(n, False) or _has(n, r.txt)
    return sorted(n for n, latin in seen.items() if not latin)


def reviewer_votes(name: str, lang: str, corpus) -> tuple[int, int]:
    """(kept Latin, not Latin) over the corpus `[(en_text, target_text), ...]` units that contain
    `name` in the English."""
    lat = tr = 0
    for en, tgt in corpus:
        if name in en:
            if _has(name, tgt):
                lat += 1
            else:
                tr += 1
    return lat, tr


def default_reviewed_glob() -> str:
    """SUBTITLE_REVIEWED_GLOB, else <DEV_TITLES_INPUT or input>/**/*Reviewed*.srt."""
    return os.environ.get("SUBTITLE_REVIEWED_GLOB") or str(
        Path(os.environ.get("DEV_TITLES_INPUT", "input")) / "**" / "*Reviewed*.srt"
    )


def default_en_dir() -> str:
    """SUBTITLE_EN_DIR, else <DEV_TITLES_OUTPUT or output>/FINAL_SUBTITLES (the delivered EN)."""
    return os.environ.get("SUBTITLE_EN_DIR") or str(
        Path(os.environ.get("DEV_TITLES_OUTPUT", "output")) / "FINAL_SUBTITLES"
    )


def reviewed_files(pat: str) -> list[Path]:
    """The files matching `pat`; a '**/' in it matches at any depth below the part before it."""
    root, _, rest = pat.partition("**/")
    return sorted(Path(root or ".").glob("**/" + rest) if rest else Path(".").glob(pat))


def load_corpus(
    lang: str, reviewed=None, en_dir=None, tm_path="tm/tm.jsonl", holdout=(), holdout_ids=()
):
    """[(en, target)]: one unit per REVIEWER-edited track (the files that came back from review —
    not our own delivered tracks, which would vote for whatever we did) against its delivered EN,
    plus one unit per TM line. Videos in `holdout` (names) / `holdout_ids` (TM video ids) are left
    out, so a video never decides its own names (an A/B on a reviewed video would copy the answer).
    Paths: SUBTITLE_REVIEWED_GLOB / SUBTITLE_EN_DIR (defaults: default_reviewed_glob /
    default_en_dir); either missing -> just the TM."""
    pat = reviewed or default_reviewed_glob()
    en_d = Path(en_dir or default_en_dir())
    hold = set(holdout) | {
        v.strip() for v in os.environ.get("SUBTITLE_HOLDOUT_VIDEOS", "").split(",") if v.strip()
    }
    out, seen = [], set()
    for tp in reviewed_files(pat):
        name = tp.name
        if lang not in re.split(r"[._-]", name) and f".{lang}" not in name:
            continue
        v = name.split(".")[0].replace("_Reviewed", "")
        ep = en_d / f"{v}.en.srt"
        if v in hold or v in seen or not ep.exists():
            continue
        seen.add(v)
        out.append((ep.read_text(encoding="utf-8"), tp.read_text(encoding="utf-8")))
    tp = Path(tm_path)
    if tp.exists():
        for line in tp.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("lang") == lang and r.get("video") not in set(holdout_ids):
                out.append((r.get("source", ""), r.get("target", "")))
    return out


CLASSIFY = """Names from a developer video's English transcript, each with a line where it occurs. For
each, would a professional {L} subtitler keep it in its Latin spelling, or translate it?
- "latin": the name of a technology, framework, library, tool, feature, API, product, app, book,
  podcast, company, organization, open-source project or a person — including a name built from a
  technology name plus ordinary words ("Universal React", "React Native Performance", "AI Gateway").
- "translate": a job title, a place, a generic event or meetup ("a dev meetup"), a program or
  series named with ordinary words ("Speaker Series"), a team or department, a group of people
  ("Google Developer Experts"), a famous fictional character, a product whose official {L} name is
  localized (Google Assistant -> Googleアシスタント), or an ordinary phrase that is only capitalized.
- "fragment": not one name but a name plus ordinary words or a grammatical neighbour, picked up
  by capitalization ("GitHub URL", "Android TVs", "Using Java", "Not Figma", "DOM APIs", "ChatGPT Mac").
When unsure between "latin" and "translate", answer "latin": reviewers keep about 3 of every 4 such
names in Latin.
VIDEO BRIEF: {brief}
NAMES: {names}
Return ONLY JSON: {{"<name>": "latin" | "translate" | "fragment", ...}}"""


def classify(names, invoke, lang_name: str, brief: str = "", lines=None) -> dict[str, str]:
    """`lines`: {name: an EN line where it occurs} — context for the call."""
    if not names:
        return {}
    shown = {n: (lines or {}).get(n, "") for n in names}
    try:
        out = (
            invoke(
                CLASSIFY.format(
                    L=lang_name,
                    brief=brief or "(none)",
                    names=json.dumps(shown, ensure_ascii=False),
                )
            )
            or ""
        )
        m = re.search(r"\{.*\}", out, re.S)
        d = json.loads(m.group(0)) if m else {}
    except Exception:
        return {}
    return {n: str(d[n]) for n in names if d.get(n) in ("latin", "translate", "fragment")}


def decide(
    names,
    lang: str,
    lang_name: str,
    corpus,
    invoke=None,
    brief: str = "",
    lines=None,
    second=None,
    sources=None,
) -> dict[str, str]:
    """{name: 'latin' | 'translate' | 'fragment'}; the reviewers' corpus wins over the model. Only
    'latin' is acted on. For a name the corpus hasn't seen, `second` (a judge of another model
    family) must ALSO say 'latin' — a lone model's 'latin' becomes 'translate' (no action).
    `sources`, if a dict, gets {name: 'corpus' | 'model'}."""
    out, unseen = {}, []
    for n in names:
        lat, tr = reviewer_votes(n, lang, corpus)
        if lat + tr:
            out[n] = "latin" if lat > tr else "translate"
            if sources is not None:
                sources[n] = "corpus"
        else:
            unseen.append(n)
    if invoke is not None and unseen:
        a = classify(unseen, invoke, lang_name, brief, lines)
        b = classify(unseen, second, lang_name, brief, lines) if second is not None else a
        for n in unseen:
            if n in a:
                out[n] = (
                    "latin"
                    if a[n] == "latin" and b.get(n) == "latin"
                    else (a[n] if a[n] != "latin" else "translate")
                )
                if sources is not None:
                    sources[n] = "model"
    return out
