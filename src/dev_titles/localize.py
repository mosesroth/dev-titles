"""End-to-end localization of an English subtitle into CJK targets.

This is the production orchestration that used to live in the `/tmp` driver: for
each target language it merges the EN cues into sentences, translates each with
whole-clip context + glossary + register + reflect, runs the completeness gates
(omission retry + source-anchored coverage edit-fix), then re-segments into
CJK-sized cues (never reusing EN line breaks) and snaps gaps. Per-video vocab
(doc-harvested identifiers + pinned renderings) feeds both DNT and consistency.

Engine-agnostic: pass any `Translator` (CliTranslator in prod, a fake in tests).
The completeness gates need a raw-prompt `invoke` callable; it defaults to the
translator's `_invoke` and is skipped (with a warning) if neither is available.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import spec_for
from .glossary import EnforcementIssue, Glossary
from .model import Subtitle
from .qa import is_short_of_expected, missing_numbers_or_idents
from .resegment import build_target_cues, merge_en_to_sentences, reflow_llm_batch
from .review import LangOutput
from .segment import normalize_ja_punct, normalize_zh_punct
from .translate import translate_segment
from .validate.rules import validity_report

_LANG_NAME = {"ja": "Japanese", "zh-Hans": "Simplified Chinese"}


def _write_status(cache_path, **fields) -> None:
    """Observability (backlog #2): merge `fields` into output/<video>/status.json next to the
    cache, so a monitor reads one file (phase, per-lang DQF, updated_at) instead of grepping
    pids/cache. Best-effort — never raises into the run."""
    if not cache_path:
        return
    try:
        import time as _t

        p = Path(cache_path).parent / "status.json"
        cur = {}
        if p.exists():
            try:
                cur = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                cur = {}
        cur.update(fields)
        cur["updated_at"] = _t.strftime("%Y-%m-%d %H:%M:%S")
        p.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass  # status.json is best-effort telemetry — never let it break the run


def _noop(_msg: str) -> None:
    pass


@contextlib.contextmanager
def llm_serialize_lock(logger=_noop):
    """Serialize heavy-LLM jobs (translation, QE re-score) via a BLOCKING advisory file lock
    so two don't hammer Bedrock at once and cause sustained throttling -> dropped/untranslated
    segments. A second job WAITS for the first instead of
    competing. Best-effort: if flock is unavailable, runs unguarded. Ad-hoc LLM scripts should
    acquire this too (`with llm_serialize_lock(print): ...`)."""
    import os

    try:
        import fcntl
    except ImportError:  # Windows: no flock -> run unguarded
        yield
        return
    path = os.environ.get("DEV_TITLES_LLM_LOCK") or os.path.expanduser("~/.dev_titles_llm.lock")
    fh = None
    try:
        fh = open(path, "w")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            logger("LLM serialize lock: another translation/QE job is running — waiting…")
            fcntl.flock(fh, fcntl.LOCK_EX)  # block until the other finishes
        yield
    except OSError:
        yield  # can't create/lock the file -> run unguarded (best-effort)
    finally:
        if fh is not None:
            try:
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
            except OSError:
                pass


def _normalizer(lang: str, conversational: bool = False):
    """Punctuation normalizer for a language, plus the zh second-person register fold.

    zh REGISTER (reversed by native-reviewer verdict): the fold used to go
    您 -> 你 for conversational content. Four independently human-reviewed videos changed our 你
    back to 您 almost everywhere (141 您 vs 4 你; all four 你 were the PLURAL 你们), so the fold
    now runs the other way: standalone 你 -> 您, while 你们 (plural "you all") is preserved
    because 您们 is unnatural. Applies to both registers — docs-formal already wanted 您."""
    base = normalize_ja_punct if lang == "ja" else normalize_zh_punct
    if lang == "zh-Hans":
        return lambda t, src=None: restore_question_mark(
            src, zh_list_separators(_zh_polite_you(base(t)), src), lang
        )
    return lambda t, src=None: restore_question_mark(src, base(t), lang)


_EN_NAME_TOKEN = r"(?:[A-Z][A-Za-z0-9.+#-]*|[a-z]+[A-Z][A-Za-z0-9.+#-]*|[A-Z0-9][A-Za-z0-9]*\+\+)"
_EN_NAME_RUN = re.compile(rf"{_EN_NAME_TOKEN}(?: {_EN_NAME_TOKEN})*")


def zh_list_separators(txt: str, src: str | None) -> str:
    """zh: an English LIST of Latin names ("JavaScript, TypeScript, ReScript, Java…") that the
    target ran together with spaces or nothing ('JavaScriptTypeScriptReScript') gets 、 between the
    items — a run-on list read as one name and was flagged by native reviewers. Only items that
    appear in the English list, only when >=3 of them sit adjacent with nothing but spaces between.
    """
    if not src or not txt or src.count(",") < 2:
        return txt
    items = []
    for part in re.split(r",\s*|\s+(?:and|or)\s+", src):
        items += [m.group(0) for m in _EN_NAME_RUN.finditer(part) if len(m.group(0)) >= 2]
    items = sorted(set(items), key=len, reverse=True)
    if len(items) < 3:
        return txt
    alt = "|".join(re.escape(x) for x in items)
    run = re.compile(rf"(?:{alt})(?:\s*(?:{alt})){{2,}}")

    def fix(m):
        parts = re.findall(alt, m.group(0))
        return (
            "、".join(parts)
            if "".join(parts).replace(" ", "") == m.group(0).replace(" ", "")
            else m.group(0)
        )

    return run.sub(fix, txt)


# Question-final forms: a ja/zh rendering of an English question that ends in one of these but has
# no ？ lost its mark. In one measured batch most English questions came out of our ja WITHOUT
# ？ — the style note says keep ？ for genuine questions (the subtitle style guide allows ？ at line end),
# but the model under-applies prose rules, so restore it deterministically.
_Q_FINAL = {
    # ja intentionally EMPTY: native reviewers REMOVED ？ in several videos while others added it;
    # the JP subtitle style guide is silent. Reviewer-split -> do not force it either way.
    "ja": (),
    "zh-Hans": ("吗", "呢", "吧", "么", "嘛"),
}


def restore_question_mark(src, txt: str, lang: str) -> str:
    """Append ？ when the English source is a question and the target ends in a question-final
    form but carries no question mark. Conservative: only the final sentence, only question-final
    endings (a statement-shaped answer to a rhetorical '?' is left alone)."""
    if not src or not txt or not src.rstrip().endswith("?"):
        return txt
    body = txt.rstrip()
    if body.endswith(("？", "?", "！", "!")):
        return txt
    if body.endswith(_Q_FINAL.get(lang, ())):
        return body + "？"
    return txt


def _budget_scale(lang: str) -> float:
    """Condensation-budget multiplier for A/B tests: SUBTITLE_BUDGET_SCALE_JA / _ZH (default 1.0).
    <1.0 asks the model for tighter renderings. Production default is unchanged."""
    import os

    key = "SUBTITLE_BUDGET_SCALE_" + ("JA" if lang == "ja" else "ZH")
    try:
        return float(os.environ.get(key, "1.0"))
    except ValueError:
        return 1.0


def _zh_polite_you(text: str) -> str:
    """你 -> 您 except the plural 你们 (您们 is not idiomatic; the reviewers kept 你们)."""
    import re as _re

    return _re.sub(r"(?<![迷])你(?!们)(?!好(?=[，,。！!？?」』\s]|$))", "您", text)


def coverage_missing(invoke, src: str, tgt: str, tgt_lang_name: str) -> str:
    """Source-anchored enumerate-then-verify omission gate: decompose the SOURCE
    into atomic propositions and check each is realized in the TARGET. Returns the
    substantive content NOT realized (empty if complete). Ignores pronoun/article/
    filler nuances (a subtitler is expected to condense those)."""
    verdict = invoke(
        f"Decompose the SOURCE English sentence into its atomic propositions — list "
        f"EACH clause/condition/qualifier/fact/number/enumerated item as a separate "
        f"unit (INCLUDING any leading subordinate clause like 'When X,' / 'Unlike Y,'). "
        f"Then, for EACH unit, check whether its meaning is realized in the {tgt_lang_name} "
        f"TARGET. Output one line per unit: 'PRESENT: <unit>' or 'MISSING: <unit>'. "
        f"Only count SUBSTANTIVE content (facts, conditions, technical detail, "
        f"identifiers, enumerated items). Do NOT flag as missing: pronouns, articles, "
        f"intensifiers, greetings, or discourse filler (welcome / a little bit / "
        f"happy coding / you know) — a subtitler is expected to condense those. "
        f"After the list, if every unit is PRESENT output a final line 'COMPLETE', "
        f"else 'INCOMPLETE'.\n\nSOURCE: {src}\n\nTARGET ({tgt_lang_name}): {tgt}"
    )
    if "INCOMPLETE" not in verdict.upper() and "MISSING:" not in verdict.upper():
        return ""
    miss = [
        ln.split("MISSING:", 1)[1].strip()
        for ln in verdict.splitlines()
        if "MISSING:" in ln.upper()[:9]
    ]
    return "; ".join(m for m in miss if m)


def edit_fix(invoke, src: str, cur: str, miss: str, tgt_lang_name: str) -> str:
    """Repair an omission by EDITING the existing translation to splice in the
    missing content — NOT retranslating from scratch (which falls back into the
    same truncation) and NOT under the tight cps budget. It may grow; resegmentation
    splits it into more cues.

    CLI commands in the source/missing-content are MASKED before the edit (this is a
    raw LLM call that otherwise translates 'app' inside create-expo-app while splicing
    it back), then restored verbatim."""
    from .qa import protected_spans
    from .translate.pipeline import _mask_spans, _unmask

    _src, mask = _mask_spans(src, protected_spans(src))
    _miss = miss
    _cur = cur
    for token, span in mask.items():  # apply the SAME tokens to miss/cur
        _miss = _miss.replace(span, token)
        _cur = _cur.replace(span, token)
    note = "\nTokens like [[0]] are placeholders — copy them EXACTLY, unchanged." if mask else ""
    out = invoke(
        f"You are editing a {tgt_lang_name} subtitle translation that DROPPED part of "
        f"the source. Do NOT retranslate from scratch — EDIT the existing translation "
        f"to ADD the missing content, keeping what is already correct. Completeness "
        f"matters more than brevity: it is fine to make it longer (it will be split "
        f"into more cues).{note} Output ONLY the corrected {tgt_lang_name}.\n\n"
        f"ENGLISH SOURCE: {_src}\n\nCURRENT {tgt_lang_name} (incomplete): {_cur}\n\n"
        f"MISSING CONTENT to add: {_miss}"
    ).strip()
    return _unmask(out, mask)


def _with_brief(transcript: str, video_vocab: dict | None) -> str:
    """Prefix the full-transcript context with the per-video brief (speakers, roles, topic,
    context-dependent jargon) so a line like "we don't have product" is read as the tech-industry
    'product (management) team', not 'no product'. The brief is runtime-only (video_vocab['brief']).
    """
    brief = (video_vocab or {}).get("brief", "")
    return f"VIDEO BRIEF: {brief}\n\nTRANSCRIPT: {transcript}" if brief else transcript


def _speakers_for(sents, video_vocab) -> list[str]:
    """Per-sentence speaker labels (video_vocab['speakers'], from speakers.label_speakers) when
    they match this sentence list, else [] (single speaker / unknown)."""
    sp = (video_vocab or {}).get("speakers") or []
    return list(sp) if len(sp) == len(sents) else []


def _ctx_for(sents, video_vocab) -> str:
    from .speakers import labeled_transcript

    return _with_brief(labeled_transcript(sents, _speakers_for(sents, video_vocab)), video_vocab)


def _line_context(i: int, sents, spk: list[str], video_vocab, k: int = 2) -> str:
    """The line in its local context — the k lines before and after (speaker-labeled, the line
    marked >>) — plus its speaker and meaning notes, for the prompt block beside the line
    (TranslationRequest.line_context). On A/B lines against a human final the old placement (notes
    appended to the whole-transcript context, no position marker, nothing in the reflect prompt)
    got wrong — "Bring it home", "Okay, okay", "wrap up with a couple things" before questions —
    came out right with this block."""
    from .en_gloss import meaning_note
    from .speakers import speaker_note

    def lab(j):
        return (f"{spk[j]}: " if spk else "") + sents[j].text

    rows = [
        (">> " if j == i else "   ") + lab(j)
        for j in range(max(0, i - k), min(len(sents), i + k + 1))
    ]
    return (
        "THE LINE IN CONTEXT (>> marks the line you are translating; the others are for "
        "understanding only — do not translate them):\n"
        + "\n".join(rows)
        + speaker_note(i, spk)
        + meaning_note(i, (video_vocab or {}).get("meaning_notes"))
    )


def keep_latin_terms(glossary, video_vocab, lang: str = "") -> list[str]:
    """SINGLE source of truth for 'keep this Latin' (DNT) terms, unified across EVERY source
    so none is a separate crack (a term slipping through one source while masked by another). Collects:
      - glossary identifiers (code/API/DNT names);
      - glossary terms whose target rendering == source for `lang` (Orbit TV, C++ — PER-LANG:
        Acme is Latin in ja but 艾克米 in zh);
      - per-video vocab identifiers / terms / speaker names;
      - per-video renderings whose canonical == source (auto-proposed keep-Latin, e.g. Pixel Quest).
    Deduped, order-preserving. Used BOTH to mask at translation AND to audit at finalize
    (dnt_survival_violations), so prevention and detection can never drift apart."""
    dnt: list[str] = list(getattr(glossary, "identifiers", []))
    # Glossary sources that the glossary TRANSLATES for this lang (target != source). These WIN over
    # any per-video vocab term/pin that would keep them Latin: a vocab auto-pin must never override
    # an approved glossary rendering (Acme Store -> Acme应用商店, not Latin).
    translated: set[str] = set()
    if lang:
        for t in getattr(glossary, "terms", []):
            tr = (t.translations.get(lang) or "").strip()
            src = (t.source or "").strip()
            if tr and tr == src:
                dnt.append(t.source)
            elif tr and tr != src:
                for f in getattr(t, "all_source_forms", [src]):
                    translated.add(f.strip().lower())

    def _glossary_translates(term: str) -> bool:
        return term.strip().lower() in translated

    if video_vocab:
        # A per-video vocab term is NOT keep-Latin if the video's OWN renderings TRANSLATE it for
        # this lang (canonical != source) — e.g. October->10月/十月, Software->ソフトウェア,
        # Bluetooth->蓝牙. Without this, such a term (extracted into vocab['terms']) was masked as
        # DNT and shipped Latin, overriding its own translating rendering (ja
        # 'October6日'). Fold those sources into `translated` so the exclusion below covers them.
        _lang_r = (video_vocab.get("renderings", {}) or {}).get(lang, {}) if lang else {}
        for _s, _sp in (_lang_r or {}).items():
            _canon = (_sp or {}).get("canonical", "").strip()
            if _canon and _canon != (_s or "").strip():
                translated.add(_s.strip().lower())
        dnt += [
            t
            for t in (
                list(video_vocab.get("identifiers", []))
                + list(video_vocab.get("terms", []))
                + list(video_vocab.get("names", []))
                # speakers' names from the speaker labels ('Moses (host)' -> 'Moses'): a host
                # named only in the brief was auto-pinned to katakana (モーゼス) by the term-pin pass
                + [
                    re.sub(r"\s*\(.*?\)", "", x).strip()
                    for x in set(video_vocab.get("speakers") or [])
                    if x
                ]
            )
            if not _glossary_translates(t)
        ]
        for _m in video_vocab.get("renderings", {}).values():
            for _src, _spec in (_m or {}).items():
                if (_spec or {}).get("canonical", "").strip() == (_src or "").strip() and not (
                    _glossary_translates(_src)
                ):
                    dnt.append(_src)
    return list(dict.fromkeys(d for d in dnt if d and d.strip()))


def stale_untranslated_reuse(target: str, en_source: str, glossary, lang: str) -> bool:
    """True if a CACHED target left a glossary term in Latin that the glossary now TRANSLATES for `lang`
    (translation != source) — e.g. 'App Store' Latin in a zh target that should be 应用商店.
    `enforce` normalizes wrong CJK avoid-forms but CANNOT reverse a free Latin rendering back to
    CJK, so such a reused record is STALE and must re-translate. This makes the cache self-healing
    against a glossary fix: correct the glossary + re-run, and ONLY the cues that left the term
    Latin re-translate (the rest reuse; incremental re-review re-judges only the changed) — a
    fix-and-rescore, not a full re-translate. Gated on the term appearing in the EN source so an
    incidental Latin token elsewhere isn't flagged."""
    import re as _re

    src_low = (en_source or "").lower()
    for t in getattr(glossary, "terms", []):
        tr = (t.translations.get(lang) or "").strip()
        src = (t.source or "").strip()
        if not tr or tr == src:
            continue  # keep-Latin (translation==source) or no rendering — not applicable
        for f in getattr(t, "all_source_forms", [src]):
            fl = f.strip().lower()
            if not fl or fl not in src_low:
                continue  # this term isn't in the cue's English source
            from .glossary import latin_run_pattern

            # whole Latin run only: the 'TV' of 'Orbit TV' / 'Acme' of 'Acme Developer Tools' is part
            # of a kept-Latin name, not a stale untranslated term (else a quarter of cues re-translate).
            if _re.search(latin_run_pattern(f), target or "", _re.I):
                return True  # its Latin form survived in the target -> stale, re-translate
    return False


def adjacent_duplicate_clauses(target, min_len: int = 6):
    """Detect a CLAUSE/line duplicated across ADJACENT cues — the translation-side analog of the
    rolling-caption dup: cue N+1 repeats a whole line from cue N because the model re-emitted the
    previous rendering instead of translating the next source (e.g.
    'excited about the future' ×2). Returns [{cue, dup}] (1-based cue index of the repeat) for the
    review. `min_len` skips short shared function-word lines so a legitimate brief echo isn't
    flagged. Deterministic; complements is_repeated_text (which only sees WITHIN one cue)."""
    out = []
    cues = target.cues
    for i in range(1, len(cues)):
        prev = {ln.strip() for ln in cues[i - 1].lines if len(ln.strip()) >= min_len}
        for ln in cues[i].lines:
            ls = ln.strip()
            if ls in prev:
                out.append({"cue": i + 1, "dup": ls})
    return out


def dnt_survival_violations(en, target, glossary, video_vocab, lang: str = "", min_len: int = 3):
    """FINALIZE BACKSTOP (detection, independent of masking). For each keep-Latin term present
    in EN, verify it survived verbatim in the target. Any that didn't were translated/garbled
    (Pixel Quest->ピクセルクエスト, Orbit TV->轨道电视) and are returned as [{cue, term, survived}]. This
    catches slips regardless of prevention gaps — the safety net we lacked. `min_len` skips
    short tokens.

    Segmentation-agnostic: under strict 1:1 (target cue count == EN) we still report per-cue and
    mark `survived` for a term that merely moved to an adjacent cue. Under FREE segmentation
    (counts differ) there is no cue-N correspondence, so a term is a garble ONLY if it is absent
    from the WHOLE target; presence anywhere = fine (reported once, keyed to the EN cue)."""
    dnt = [t for t in keep_latin_terms(glossary, video_vocab, lang) if len(t) >= min_len]
    if not dnt:
        return []
    full_tt = " ".join(" ".join(tc.lines) for tc in target.cues)
    out = []
    if len(en.cues) == len(target.cues):
        # 1:1: a term missing from its ALIGNED cue but present elsewhere merely moved (survived).
        for i, (ec, tc) in enumerate(zip(en.cues, target.cues), 1):
            et, tt = " ".join(ec.lines), " ".join(tc.lines)
            for term in dnt:
                if term in et and term not in tt:
                    out.append({"cue": i, "term": term, "survived": term in full_tt})
    else:
        # Free segmentation: whole-target presence is the only meaningful signal.
        seen: set[str] = set()
        for i, ec in enumerate(en.cues, 1):
            et = " ".join(ec.lines)
            for term in dnt:
                if term in et and term not in full_tt and term not in seen:
                    out.append({"cue": i, "term": term, "survived": False})
                    seen.add(term)
    return out


def effective_glossary(
    glossary: Glossary, video_vocab: dict | None, conversational: bool = False, lang: str = ""
) -> tuple[Glossary, list[str]]:
    """A shallow copy of `glossary` extended with the per-video pinned renderings
    (static glossary wins via skip_sources), plus the DNT term list (identifiers +
    proper terms to keep Latin). Does not mutate the caller's glossary.

    When `conversational` (interview/peer-dialogue content), the zh register is swapped
    from formal 您 to casual 你 (see the glossary's register_conversational /
    pronoun_conversational); ja teineigo is unchanged (polite spoken JA suits peers)."""
    g = copy.copy(glossary)
    g.terms = list(glossary.terms)
    if conversational and "zh-Hans" in getattr(glossary, "style", {}):
        zh = dict(glossary.style["zh-Hans"])
        if zh.get("register_conversational"):
            zh["register"] = zh["register_conversational"]
        if zh.get("pronoun_conversational"):
            zh["notes"] = zh["pronoun_conversational"] + "\n" + zh.get("notes", "")
        g.style = {**glossary.style, "zh-Hans": zh}  # don't mutate the caller's style dict
    if conversational:
        # any language's interview-only notes (ja: spoken register over the presentation rules)
        for code, st in list((getattr(g, "style", None) or {}).items()):
            if isinstance(st, dict) and st.get("notes_conversational"):
                st = dict(st)
                st["notes"] = st["notes_conversational"] + "\n" + st.get("notes", "")
                g.style = {**g.style, code: st}
    # DNT / keep-Latin is resolved from ONE unified source (see keep_latin_terms) so no
    # individual source is a separate crack (a term kept Latin by one source but not another).
    extra_dnt = keep_latin_terms(glossary, video_vocab, lang)
    if video_vocab:
        from .vocab_prep import renderings_to_terms

        skip = {f for t in glossary.terms for f in t.all_source_forms}
        curated = {f: t.translations for t in glossary.terms for f in t.all_source_forms}
        g.terms.extend(
            renderings_to_terms(
                video_vocab.get("renderings", {}), skip_sources=skip, curated_renderings=curated
            )
        )
    return g, extra_dnt


# Backoff (seconds) between per-segment translate retries — rides out a transient
# outage before falling back to the EN placeholder. Overridable in tests.
_SEGMENT_RETRY_BACKOFF = (5, 20)


@dataclass
class SentenceRecord:
    lang: str
    i: int
    src: str
    txt: str
    residual_miss: str
    issues: list[dict]


def _load_cache(
    cache_path, lang: str
) -> tuple[dict[int, SentenceRecord], dict[str, SentenceRecord]]:
    """The resumable cache for `lang`, indexed by sentence position AND by source text. The by-text
    index makes reuse robust to SEGMENTATION SHIFTS: an EN edit that adds/removes a sentence
    renumbers every later sentence, so an index-only lookup would miss all of them
    (src != cache[i].src) and re-translate the whole tail. Keying unchanged source text by content
    lets them reuse their existing translation regardless of their new index — only genuinely
    new/edited text re-translates."""
    cache: dict[int, SentenceRecord] = {}
    cache_by_src: dict[str, SentenceRecord] = {}
    p = Path(cache_path) if cache_path else None
    if p is not None and p.exists():
        for line in p.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r["lang"] == lang:
                    rec0 = SentenceRecord(**r)
                    cache[r["i"]] = rec0
                    cache_by_src.setdefault(rec0.src, rec0)  # first wins (stable for dup lines)
    return cache, cache_by_src


def _reuse_record(i, s, cache, cache_by_src, g, lang, normalize, logger):
    """A record rebuilt from the cache for this UNCHANGED source, or None to (re-)translate.

    The cache is index-keyed, but REUSE ONLY IF THE SOURCE IS UNCHANGED. An operator EN correction
    (e.g. 'Chile'->'LG') keeps the same sentence index, so an index-only reuse would ship the STALE
    translation of the OLD text and never re-translate the fix; on any change, fall back to the
    by-source index (a segmentation shift), else re-translate."""
    reuse = cache[i] if (i in cache and cache[i].src == s.text) else cache_by_src.get(s.text)
    if reuse is None:
        return None
    if stale_untranslated_reuse(reuse.txt, s.text, g, lang):
        # A glossary fix made a term that this cached target left Latin now TRANSLATABLE
        # (App Store -> 应用商店); enforce can't reverse it, so drop the reuse and re-translate
        # THIS cue only. The cache self-heals against glossary changes without a full clear.
        logger(
            f"{lang}: cache-reuse stale #{i} (glossary now translates a Latin-left term) — re-translating"
        )
        return None
    # Re-enforce against the CURRENT glossary (deterministic, no LLM) and APPLY its fixes —
    # refreshes stale flags (pre-DNT 'Acme missing' FPs on 'Acme Q') AND normalizes same-cue
    # avoid-forms to canonical (指导文件->引导文档), so a glossary change lands on cache reuse
    # without a full re-translation.
    rec = SentenceRecord(lang, i, s.text, reuse.txt, reuse.residual_miss, list(reuse.issues))
    txt, iss = g.enforce(rec.txt, s.text, lang)
    # Normalize like the fresh/TM paths — folds the register + punct on a REUSED translation;
    # without it a segment cached under another register kept that register forever on reuse.
    rec.txt = normalize(txt, s.text)
    return rec, iss


def _translate_fresh(
    i,
    s,
    translator,
    g,
    lang,
    lang_name,
    spec,
    *,
    ctx,
    line_context="",
    exemplar_bank,
    extra_dnt,
    invoke,
    backcheck,
    edit_fix_rounds,
    logger,
):
    """Translate one sentence: draft + reflect, omission gate, coverage edit-fix. Returns
    (txt, issues, residual).

    RESILIENCE: a single segment's LLM call can hang/time out (Bedrock under load, or a dropped
    internet connection). RETRY with backoff first — a transient outage that recovers within ~30s
    is survived here. Only after the retries are exhausted do we keep the EN placeholder + flag
    it, rather than abort the whole multi-hour run (a crash here once lost a full ja pass)."""
    import time as _time

    # Soft budget with completeness headroom (1.15x): the tight cps budget pressured the model to
    # DROP clauses; cps stays a review-only warning.
    budget = max(
        int(spec.max_cps * 1.15 * _budget_scale(lang) * max(s.end - s.start, spec.min_dur)), 12
    )
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            txt, iss = translate_segment(
                translator,
                g,
                s.text,
                lang,
                full_context=ctx,
                line_context=line_context,
                reflect=True,
                char_budget=budget,
                exemplar_bank=exemplar_bank,
                extra_dnt=extra_dnt,
            )
            txt, iss = _omission_gate(
                translator,
                g,
                s,
                lang,
                txt,
                iss,
                ctx,
                budget,
                exemplar_bank,
                extra_dnt,
                logger,
                i,
                line_context=line_context,
            )
            residual = ""
            if backcheck:
                txt, iss, residual = _coverage_editfix(
                    invoke, g, s, lang, lang_name, txt, iss, edit_fix_rounds, logger, i
                )
            return txt, iss, residual
        except Exception as e:
            last_exc = e
            if attempt < 2:
                logger(
                    f"{lang}: segment #{i} attempt {attempt + 1} failed "
                    f"({type(e).__name__}); retrying"
                )
                _time.sleep(_SEGMENT_RETRY_BACKOFF[attempt])  # ride out a transient drop
    name = type(last_exc).__name__
    logger(
        f"{lang}: segment #{i} translation FAILED ({name}) after retries — keeping EN, "
        f"flagged for review"
    )
    return s.text, [], f"translation failed ({name}) — needs manual translation"


def _build_lang_subtitle(
    en, sents, records, spec, keep_together, speakers, conversational, issues, lang, logger
):
    """Shipping cues from the per-sentence records (free CJK segmentation or 1:1, see
    resegment.build_target_cues), with the sync and validity reports logged."""
    from .validate.sync import sync_report

    cues = build_target_cues(
        en, sents, [r.txt for r in records], spec, keep_together=keep_together, speakers=speakers
    )
    sub = Subtitle(lang, cues).renumber()
    rep = validity_report(sub, conversational)
    sync = sync_report(cues, en.cues)
    if sync["drifted"]:
        logger(
            f"{lang}: WARNING out of sync with the speech — median {sync['median']} s, "
            f"{sync['far_share']:.0%} of name anchors >3 s off (see validate/sync.py)"
        )
    logger(
        f"{lang}: DONE {len(cues)} cues, {rep['cues_with_errors']} err-cues, "
        f"{sum(1 for x in issues if x.kind == 'missing')} missing"
    )
    return sub


def translate_subtitle(
    en: Subtitle,
    translator,
    glossary: Glossary,
    lang: str,
    *,
    exemplar_bank=None,
    video_vocab: dict | None = None,
    tm=None,
    invoke=None,
    backcheck: bool = True,
    edit_fix_rounds: int = 4,
    cache_path: str | Path | None = None,
    conversational: bool = False,
    logger=_noop,
) -> tuple[Subtitle, list[EnforcementIssue], list[SentenceRecord]]:
    """Translate `en` into one `lang`, returning (Subtitle, issues, per-sentence records).

    Completeness gates (omission retry + coverage edit-fix) run when `invoke` (or
    `translator._invoke`) is available and `backcheck` is on. A `cache_path` JSONL
    makes the run resumable (per-(lang, sentence) records)."""
    from .qa import protected_spans

    spec = spec_for(lang)
    normalize = _normalizer(lang, conversational)
    g, extra_dnt = effective_glossary(glossary, video_vocab, conversational, lang=lang)
    # Auto-protect CLI commands / env vars found anywhere in this video as DNT (doc-global)
    # — they get partially translated/recased otherwise (create-expo-app -> create-expo-アプリ).
    cmds = protected_spans(" ".join(c.text for c in en.cues))  # CLI commands + web domains
    extra_dnt = list(dict.fromkeys(list(extra_dnt) + cmds))
    keep_together = [
        t.translations.get(lang) for t in g.terms if t.translations.get(lang)
    ] + extra_dnt
    if invoke is None:
        invoke = getattr(translator, "_invoke", None)
    if backcheck and invoke is None:
        logger(f"{lang}: no raw-invoke available — skipping completeness gates")
        backcheck = False

    cache, cache_by_src = _load_cache(cache_path, lang)
    cf = Path(cache_path).open("a", encoding="utf-8") if cache_path else None
    sents = merge_en_to_sentences(en)
    ctx = _ctx_for(sents, video_vocab)
    spk = _speakers_for(sents, video_vocab)
    logger(f"{lang}: {len(en.cues)} EN cues -> {len(sents)} sentences")
    issues: list[EnforcementIssue] = []
    records: list[SentenceRecord] = []

    for i, s in enumerate(sents):
        reused = _reuse_record(i, s, cache, cache_by_src, g, lang, normalize, logger)
        if reused is not None:
            # NOT re-appended to the append-only cache: the translation is already persisted
            # (under its original index) and the by-source index re-finds it on resume.
            rec, iss = reused
        else:
            if tm is not None and (hit := tm.exact(s.text, lang)) is not None:
                # TM EXACT-MATCH REUSE: an identical source line already has an approved
                # target — reuse it verbatim for cross-video consistency (no LLM, no gates).
                logger(f"{lang}: TM exact-match reuse #{i}")
                txt, iss, residual = hit, [], ""
            else:
                txt, iss, residual = _translate_fresh(
                    i,
                    s,
                    translator,
                    g,
                    lang,
                    _LANG_NAME.get(lang, lang),
                    spec,
                    ctx=ctx,
                    line_context=_line_context(i, sents, spk, video_vocab),
                    exemplar_bank=exemplar_bank,
                    extra_dnt=extra_dnt,
                    invoke=invoke,
                    backcheck=backcheck,
                    edit_fix_rounds=edit_fix_rounds,
                    logger=logger,
                )
            rec = SentenceRecord(
                lang,
                i,
                s.text,
                normalize(txt, s.text),
                residual,
                [
                    {"term": x.term, "kind": x.kind, "expected": x.expected, "detail": x.detail}
                    for x in iss
                ],
            )
            if cf:
                cf.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
                cf.flush()
        issues += iss
        records.append(rec)

    if cf:
        cf.close()
        # Compact the append-only cache to EXACTLY the current sentence set (one record per
        # sentence, this lang only). Without this, runs across a changed segmentation accumulate
        # stale/duplicate (src) entries — e.g. 'Good, thank you.' translated twice at two indices —
        # which later trips the lossless audit even though the delivered file is correct.
        _rewrite_cache(cache_path, lang, records)
    sub = _build_lang_subtitle(
        en, sents, records, spec, keep_together, spk, conversational, issues, lang, logger
    )
    return sub, issues, records


def _omission_gate(
    translator,
    g,
    s,
    lang,
    txt,
    iss,
    ctx,
    budget,
    exemplar_bank,
    extra_dnt,
    logger,
    i,
    line_context="",
):
    """One completeness retry when the draft is far shorter than its EN word count
    OR a hard term is missing OR a number/identifier was dropped. Keeps the attempt
    that covers the most required tokens."""
    import re as _re

    frags = [
        f.strip()
        for f in _re.split(r",|;|\band\b|\bor\b|\bas well as\b", s.text)
        if len(f.split()) >= 2
    ]
    hint = (
        (
            "Convey EACH of these source parts — omit NONE: "
            + " | ".join(f"({k+1}) {f}" for k, f in enumerate(frags))
        )
        if len(frags) > 1
        else "Your prior draft was too short."
    )

    def incomplete(t, isu):
        return (
            is_short_of_expected(s.text, t, lang)
            or [x for x in isu if x.kind == "missing"]
            or missing_numbers_or_idents(s.text, t)
        )

    if not incomplete(txt, iss):
        return txt, iss
    miss_nums = missing_numbers_or_idents(s.text, txt)
    logger(
        f"{lang}: omission-gate retry #{i} ({len(txt)}c/{len(s.text.split())}w"
        + (f", missing-num/id {miss_nums}" if miss_nums else "")
        + ")"
    )
    h = hint + (
        f" You also dropped these EXACT tokens — include each verbatim: {miss_nums}."
        if miss_nums
        else ""
    )
    cand, ci = translate_segment(
        translator,
        g,
        s.text,
        lang,
        full_context=ctx,
        line_context=line_context,
        reflect=True,
        char_budget=budget,
        exemplar_bank=exemplar_bank,
        completeness_hint=h,
        extra_dnt=extra_dnt,
    )

    def _incompleteness(t, isu):
        # rank by (short-of-expected, #missing hard terms, #missing numbers/idents)
        return (
            int(is_short_of_expected(s.text, t, lang)),
            len([x for x in isu if x.kind == "missing"]),
            len(missing_numbers_or_idents(s.text, t)),
        )

    # Keep the retry ONLY when it's genuinely MORE COMPLETE — not merely longer
    # (the old `len(cand) > len(txt)` rewarded verbosity, re-inflating cues a
    # subtitler intentionally condensed and pushing cps over cap).
    if _incompleteness(cand, ci) < _incompleteness(txt, iss):
        return cand, ci
    return txt, iss


def _coverage_editfix(invoke, g, s, lang, lang_name, txt, iss, rounds, logger, i):
    """Source-anchored coverage gate + edit-in-place repair. Bounded; anything still
    missing is returned as `residual` (flagged for the editor, never silently dropped)."""
    cmiss = coverage_missing(invoke, s.text, txt, lang_name)
    bt = 0
    while cmiss and bt < rounds:
        bt += 1
        logger(f"{lang}: coverage EDIT-fix {bt} #{i} missing: {cmiss[:80]}")
        cand = edit_fix(invoke, s.text, txt, cmiss, lang_name)
        cand, ci = g.enforce(cand, s.text, lang)
        nc = coverage_missing(invoke, s.text, cand, lang_name)
        if len(nc) < len(cmiss):
            txt, iss, cmiss = cand, ci, nc
        else:
            break
    if cmiss:
        logger(f"{lang}: RESIDUAL omission #{i} (flagged for review): {cmiss[:80]}")
    return txt, iss, cmiss


def _lang_setup(
    en: Subtitle,
    glossary: Glossary,
    lang: str,
    video_vocab: dict | None,
    conversational: bool = False,
):
    """Recompute the per-language translation context (effective glossary, DNT list,
    keep-together terms, sentences, full context) — the same setup `translate_subtitle`
    does — so the QE-repair closure and the post-repair rebuild stay in lockstep with it."""
    from .qa import protected_spans

    spec = spec_for(lang)
    g, extra_dnt = effective_glossary(glossary, video_vocab, conversational, lang=lang)
    cmds = protected_spans(" ".join(c.text for c in en.cues))  # CLI commands + web domains
    extra_dnt = list(dict.fromkeys(list(extra_dnt) + cmds))
    keep_together = [
        t.translations.get(lang) for t in g.terms if t.translations.get(lang)
    ] + extra_dnt
    sents = merge_en_to_sentences(en)
    ctx = _ctx_for(sents, video_vocab)
    return spec, g, extra_dnt, keep_together, sents, ctx


def make_retranslate(
    en,
    translator,
    glossary,
    lang,
    *,
    exemplar_bank=None,
    video_vocab=None,
    invoke=None,
    backcheck=True,
    conversational=False,
):
    """Build a `retranslate(rec, critique) -> str` for QE-repair: re-translate ONE
    sentence with the QE critique as a corrective constraint, then re-run the SAME
    completeness/coverage gates as the main pass (so a repair can't silently drop content)."""
    spec, g, extra_dnt, _keep, sents, ctx = _lang_setup(
        en, glossary, lang, video_vocab, conversational
    )
    lang_name = _LANG_NAME.get(lang, lang)
    normalize = _normalizer(lang, conversational)
    inv = invoke if invoke is not None else getattr(translator, "_invoke", None)

    def retranslate(rec, critique):
        s = sents[rec.i]
        budget = max(
            int(spec.max_cps * 1.15 * _budget_scale(lang) * max(s.end - s.start, spec.min_dur)), 12
        )
        # LOCAL DISCOURSE CONTEXT: the flagged cue is fragmentary, so re-translating it in
        # isolation reproduces the same misread (a turn-handoff 'and Łukasz', an ambiguous
        # referent 'on some of them'). Give the model the IMMEDIATE neighbors so it can resolve
        # who's speaking / what a pronoun points to — this is exactly the context the whole-doc
        # blob dilutes. It's for disambiguation only; only the target line is translated.
        _line = _line_context(rec.i, sents, _speakers_for(sents, video_vocab), video_vocab)
        txt, iss = translate_segment(
            translator,
            g,
            s.text,
            lang,
            full_context=ctx,
            line_context=_line,
            reflect=True,
            char_budget=budget,
            exemplar_bank=exemplar_bank,
            correction_hint=critique,
            extra_dnt=extra_dnt,
        )
        txt, iss = _omission_gate(
            translator,
            g,
            s,
            lang,
            txt,
            iss,
            ctx,
            budget,
            exemplar_bank,
            extra_dnt,
            _noop,
            rec.i,
            line_context=_line,
        )
        if backcheck and inv is not None:
            txt, iss, _ = _coverage_editfix(inv, g, s, lang, lang_name, txt, iss, 4, _noop, rec.i)
        return normalize(txt, s.text)

    return retranslate


def _subtitle_from_records(
    en,
    records,
    glossary,
    lang,
    video_vocab,
    conversational: bool = False,
    invoke=None,
    reflow_cache_path=None,
) -> Subtitle:
    """Rebuild the target Subtitle from (possibly repaired) per-sentence records —
    same re-segmentation + gap-snapping as `translate_subtitle`'s cue-building tail.
    MUST use the same (conversational) normalizer: normalize_zh_punct re-formalizes 你->您,
    so a base normalizer here would undo the conversational register on the delivered file."""
    spec, _g, _dnt, keep_together, sents, _ctx = _lang_setup(
        en, glossary, lang, video_vocab, conversational
    )
    normalize = _normalizer(lang, conversational)
    by_i = {r.i: r for r in records}
    cjk_texts = [
        normalize(by_i[i].txt, by_i[i].src) if i in by_i else "" for i in range(len(sents))
    ]
    # meaning-aligned re-flow when an LLM is available (batched, lossless-guarded); else proportional.
    # A persisted reflow cache (keyed by cues+translation) reuses unchanged splits — re-aligning
    # after a small edit re-flows only the changed sentences instead of the whole video.
    reflow: "Callable[[list], list] | None"
    rc = None
    rc_path = Path(reflow_cache_path) if reflow_cache_path else None
    if invoke is not None:
        _lname = _LANG_NAME.get(lang, lang)
        rc = {}
        if rc_path and rc_path.exists():
            try:
                rc = json.loads(rc_path.read_text(encoding="utf-8"))
            except Exception:
                rc = {}

        def reflow(items):
            return reflow_llm_batch(items, invoke, _lname, cache=rc)

    else:
        reflow = None

    cues = build_target_cues(
        en,
        sents,
        cjk_texts,
        spec,
        keep_together=keep_together,
        reflow=reflow,
        speakers=_speakers_for(sents, video_vocab),
    )
    if rc_path is not None and rc:
        rc_path.write_text(json.dumps(rc, ensure_ascii=False), encoding="utf-8")
    return Subtitle(lang, cues).renumber()


def _rewrite_cache(cache_path, lang, records) -> None:
    """Rewrite the resumable cache so repaired text survives a resume: keep every OTHER
    language's lines verbatim, replace this language's with the current records."""
    if not cache_path:
        return
    p = Path(cache_path)
    keep = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("lang") != lang:
                keep.append(line)
    new = [json.dumps(r.__dict__, ensure_ascii=False) for r in records]
    p.write_text("\n".join(keep + new) + "\n", encoding="utf-8")


def invalidate_cache(
    cache_path,
    lang: str,
    *,
    src_contains=None,
    tgt_contains=None,
    indices=None,
    predicate=None,
    logger=_noop,
) -> int:
    """SURGICAL cache invalidation — the ONLY correct way to remediate already-translated cues
    (a glossary/vocab fix, a term-pin corruption, a hand-fix). Drops ONLY the matching `lang`
    records from cache.jsonl so a re-localize re-translates JUST those; every other cue reuses its
    cached translation. Do NOT clear a whole language or the whole cache — and do NOT touch
    qe_state.*/reflow_cache.*: incremental_review re-judges exactly the cues whose target-text hash
    changed (so the re-translated ones re-judge automatically, the rest reuse), and reflow_cache is
    content-keyed so it regenerates only their splits. See feedback_surgical-cache-invalidation.

    A record (lang-matched) is invalidated if ANY selector matches:
      src_contains: str or iterable of substrings to look for in the EN source (r['src'])
      tgt_contains: str or iterable of substrings in the target text (r['txt']) — a bad rendering
      indices:      iterable of cue indices (r['i'])
      predicate:    callable(record_dict) -> bool
    Returns the number of records invalidated (logs it + a red-flag warning if it's most of them).
    """
    if not cache_path:
        return 0
    p = Path(cache_path)
    if not p.exists():
        return 0

    def _subs(x):
        return [x] if isinstance(x, str) else list(x or [])

    src_subs, tgt_subs = _subs(src_contains), _subs(tgt_contains)
    idx = set(indices or ())
    kept, dropped, total_lang = [], 0, 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("lang") != lang:
            kept.append(line)
            continue
        total_lang += 1
        hit = (
            (r.get("i") in idx)
            or any(s in (r.get("src") or "") for s in src_subs)
            or any(s in (r.get("txt") or "") for s in tgt_subs)
            or (predicate(r) if predicate else False)
        )
        if hit:
            dropped += 1  # drop -> re-translates on next localize
        else:
            kept.append(line)
    p.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    logger(f"invalidate_cache[{lang}]: dropped {dropped}/{total_lang} record(s) (reuse the rest)")
    if total_lang and dropped > total_lang * 0.6:
        logger(
            f"invalidate_cache[{lang}]: RED FLAG — dropped {dropped}/{total_lang} (>60%); a "
            f"remediation should hit a small subset. Double-check the selector isn't over-broad."
        )
    return dropped


def localize_en(*args, logger=_noop, **kwargs):
    """Localize an EN subtitle into each target language, SERIALIZED behind the LLM lock
    (llm_serialize_lock) so concurrent runs queue instead of throttling Bedrock. Thin wrapper over _localize_en_impl; see it for the full pipeline.
    """
    with llm_serialize_lock(logger):
        return _localize_en_impl(*args, logger=logger, **kwargs)


@dataclass
class _Run:
    """What every stage of one localize run shares (the old 17-parameter function's arguments)."""

    en: Subtitle
    translator: Any
    glossary: Glossary
    langs: tuple
    video_vocab: dict | None
    invoke: Callable[[str], str] | None
    qe_invoke: Callable[[str], str] | None
    backcheck: bool
    exemplar_bank_for: Callable[[str], Any] | None
    tm: Any
    qe_judge: bool
    qe_second_judge: Callable[[str], str] | None
    qe_repair: bool
    conversational: bool
    cache_path: str | Path | None
    condense: bool
    native_editor: bool
    logger: Callable[[str], None]

    def path(self, name: str) -> Path | None:
        """A file next to the cache (the video's output dir), or None without a cache."""
        return (Path(self.cache_path).parent / name) if self.cache_path else None


@dataclass
class _LangRun:
    """One target language's state as it moves through the stages."""

    lang: str
    bank: Any = None
    rcp: Path | None = None
    sub: Subtitle | None = None
    issues: list = field(default_factory=list)
    records: list = field(default_factory=list)
    suspects: list = field(default_factory=list)
    dqf: dict | None = None
    prior: Any = None
    qe_state_path: Path | None = None
    has_major: bool = False
    judges: Any = None
    term_pins: dict = field(default_factory=dict)
    pins: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return _LANG_NAME.get(self.lang, self.lang)


def _has_major(suspects) -> bool:
    return any(
        "major" in str(s.get("kind", "")) or "critical" in str(s.get("kind", "")) for s in suspects
    )


def _dqf_note(dqf, prefix: str = ", ") -> str:
    return (f"{prefix}DQF {dqf['score']} {'PASS' if dqf['pass'] else 'FAIL'}") if dqf else ""


def _localize_en_impl(
    en: Subtitle,
    translator,
    glossary: Glossary,
    *,
    langs=("ja", "zh-Hans"),
    video_vocab: dict | None = None,
    invoke=None,
    backcheck: bool = True,
    exemplar_bank_for=None,
    tm=None,
    qe_judge: bool = False,
    qe_second_judge=None,
    qe_repair: bool = True,
    conversational: bool = False,
    cache_path: str | Path | None = None,
    condense: bool = True,
    native_editor: bool = True,
    logger=_noop,
) -> list[LangOutput]:
    """Localize an EN subtitle into each target language. Returns LangOutputs ready
    for `review.build_review_bundle`. `exemplar_bank_for(lang) -> bank` optionally
    supplies few-shot style anchors per language. A `tm` (TranslationMemory) adds
    exact-match verbatim reuse + series-specific fuzzy exemplars (falls back to it for
    the exemplar bank when `exemplar_bank_for` isn't given). Reference-free translation
    QE runs per language (deterministic always; a calibrated-DQF LLM MQM judge when `qe_judge`).

    Stages: video-level preparation once (brief, speakers, meaning notes, names policy), then per
    language: translate -> pronoun referents (zh) -> condense (ja) -> native editor + reply check -> QE review ->
    condensation revert -> QE repair -> term pins -> save QE state -> DNT auto-correct -> finalize.
    """
    run = _Run(
        en=en,
        translator=translator,
        glossary=glossary,
        langs=tuple(langs),
        video_vocab=video_vocab,
        invoke=invoke,
        qe_invoke=invoke or getattr(translator, "_invoke", None),
        backcheck=backcheck,
        exemplar_bank_for=exemplar_bank_for,
        tm=tm,
        qe_judge=qe_judge,
        qe_second_judge=qe_second_judge,
        qe_repair=qe_repair,
        conversational=conversational,
        cache_path=cache_path,
        condense=condense,
        native_editor=native_editor,
        logger=logger,
    )
    if run.qe_invoke is not None and native_editor:
        _prepare_video(run)
    outputs: list[LangOutput] = []
    for lang in run.langs:
        L = _translate_lang(run, lang)
        _pronoun_pass(run, L)
        _condense_pass(run, L)
        _editor_pass(run, L)
        _review_pass(run, L)
        _condensation_revert(run, L)
        _qe_repair_pass(run, L)
        _term_pin_pass(run, L)
        if L.dqf is not None and L.qe_state_path is not None:
            from .translation_qe import save_qe_state

            # the final, post-repair panel, so the NEXT run can re-review incrementally
            save_qe_state(L.qe_state_path, L.records, L.dqf, L.suspects)
        _dnt_autocorrect(run, L)
        outputs.append(_finalize_lang(run, L))
    _write_status(cache_path, phase="complete")
    return outputs


# ---- video-level preparation (once, shared by every language) ----


def _prepare_video(run: _Run) -> None:
    """Video brief, speaker turns, meaning notes and the names policy -> run.video_vocab."""
    from .editor import video_brief, video_description

    od = Path(run.cache_path).parent if run.cache_path else None
    brief = video_brief(
        " ".join(c.text for c in run.en.cues),
        run.qe_invoke,
        run.path("video_brief.txt"),
        description=video_description(od),
    )
    if brief:
        run.video_vocab = {**(run.video_vocab or {}), "brief": brief}
        run.logger(f"video brief: {brief[:160]}…")
    _add_speakers(run, brief or "")
    _add_meaning_notes(run, brief or "")
    _apply_names_policy(run, brief or "")


def _add_speakers(run: _Run, brief: str) -> None:
    """Who says each sentence (interviews): ASR diarization if saved, else one LLM pass."""
    from .speakers import label_speakers

    try:
        spk = label_speakers(
            merge_en_to_sentences(run.en),
            run.qe_invoke,
            run.path("speakers.json"),
            brief=brief,
            raw_path=run.path("raw.json"),
            logger=run.logger,
        )
    except Exception as e:  # never sink a run on speaker labeling
        run.logger(f"speaker labeling skipped ({type(e).__name__})")
        spk = []
    if spk:
        run.video_vocab = {**(run.video_vocab or {}), "speakers": spk}


def _add_meaning_notes(run: _Run, brief: str) -> None:
    """Idioms / ambiguous phrases resolved ONCE in the English, shared by every language."""
    from .en_gloss import meaning_notes

    try:
        mn = meaning_notes(
            merge_en_to_sentences(run.en),
            run.qe_invoke,
            run.path("meaning_notes.json"),
            brief=brief,
        )
    except Exception as e:  # never sink a run on it
        run.logger(f"meaning notes skipped ({type(e).__name__})")
        mn = {}
    if mn:
        run.video_vocab = {**(run.video_vocab or {}), "meaning_notes": mn}
        run.logger(
            f"meaning notes: {sum(len(v) for v in mn.values())} phrase(s) resolved for the translators"
        )


def _apply_names_policy(run: _Run, brief: str) -> None:
    """Partner companies stay Latin (or their own official localized name); ordinary-word products
    get ONE rendering — the partner's official one when its site has it (locked in partners.yaml).
    """
    from .entities import apply_policy, names_policy

    try:
        pol = names_policy(
            " ".join(c.text for c in run.en.cues),
            run.qe_invoke,
            run.path("entities.json"),
            brief=brief,
        )
    except Exception as e:  # never sink a run on it
        run.logger(f"names policy skipped ({type(e).__name__})")
        pol = {}
    if pol.get("ordinary_products"):
        try:
            from .entities import resolve_official
            from .search import web_search
            from .vocab_prep import http_fetch

            pol = resolve_official(
                pol,
                web_search=web_search,
                web_fetch=http_fetch,
                invoke=run.qe_invoke,
                langs=run.langs,
                logger=run.logger,
            )
        except Exception as e:
            run.logger(f"official-name lookup skipped ({type(e).__name__})")
    if pol.get("companies") or pol.get("ordinary_products"):
        run.video_vocab = apply_policy(run.video_vocab, pol, run.langs)
        run.logger(
            f"names policy: keep Latin {pol.get('companies')}; translate "
            f"{[(x.get('name'), x.get('ja'), x.get('zh-Hans')) for x in pol.get('ordinary_products', [])]}"
        )


# ---- per-language stages ----


def _translate_lang(run: _Run, lang: str) -> _LangRun:
    L = _LangRun(lang)
    L.bank = (
        run.exemplar_bank_for(lang)
        if run.exemplar_bank_for
        else (run.tm.bank(lang) if run.tm is not None else None)
    )
    L.rcp = run.path(f"reflow_cache.{lang}.json")
    _write_status(run.cache_path, phase="translate", lang=lang)
    L.sub, L.issues, L.records = translate_subtitle(
        run.en,
        run.translator,
        run.glossary,
        lang,
        exemplar_bank=L.bank,
        video_vocab=run.video_vocab,
        tm=run.tm,
        invoke=run.invoke,
        backcheck=run.backcheck,
        cache_path=run.cache_path,
        conversational=run.conversational,
        logger=run.logger,
    )
    return L


def _pronoun_pass(run: _Run, L: _LangRun) -> None:
    """zh: who each English "you" means — generic / the addressee's team / the viewers get no
    pronoun / 你们 / 大家 instead of the 您 reserved for one person. See dev_titles/zh_pronoun.py."""
    if not (L.lang == "zh-Hans" and run.qe_invoke is not None and L.records):
        return
    from .zh_pronoun import pronoun_pass

    chg = pronoun_pass(
        L.records,
        run.qe_invoke,
        run.path(f"pronoun_state.{L.lang}.json"),
        normalize=_normalizer(L.lang, run.conversational),
        conversational=run.conversational,
        logger=run.logger,
    )
    if chg:
        _rewrite_cache(run.cache_path, L.lang, L.records)
        kinds = sorted({c["referent"] for c in chg})
        run.logger(f"{L.lang}: pronoun referents fixed in {len(chg)} line(s) ({', '.join(kinds)})")


def _condense_pass(run: _Run, L: _LangRun) -> None:
    """CONDENSATION (ja, ON by default; --no-condense): tighten finished translations like a
    subtitle editor before QE, so the panel judges what ships. See dev_titles/condense.py."""
    if not (run.condense and L.lang == "ja" and run.qe_invoke is not None and L.records):
        return
    from .condense import condense_lang

    spec = spec_for(L.lang)
    sents = merge_en_to_sentences(run.en)
    # per-sentence character budget = reading-speed limit x the sentence's speech window
    budgets = {
        i: max(int(spec.max_cps * max(s.end - s.start, spec.min_dur)), 6)
        for i, s in enumerate(sents)
    }
    # ...but only lines CLEARLY too fast are touched: native-reviewed ja ships 20% of cues
    # above 8.5 cps (p90 ~10.2), and squeezing near-limit lines to the exact budget is what
    # produced broken grammar / dropped greetings on a tutorial test.
    trig_cps = float(os.environ.get("SUBTITLE_CONDENSE_TRIGGER_CPS", "10.0"))
    # LONG LISTS ("JavaScript, TypeScript, ReScript, Java, Kotlin…") are hard to read even
    # just over the limit — native graders flagged them — so a list sentence (>=3 commas)
    # triggers at the target budget itself, not the looser 10 cps.
    triggers = {
        i: (
            budgets[i]
            if s.text.count(",") >= 3
            else max(int(trig_cps * max(s.end - s.start, spec.min_dur)), 6)
        )
        for i, s in enumerate(sents)
    }
    chg = condense_lang(
        L.records,
        run.qe_invoke,
        L.lang,
        L.name,
        run.path(f"condense_state.{L.lang}.json"),
        coverage_missing=coverage_missing,
        normalize=_normalizer(L.lang, run.conversational),
        logger=run.logger,
        budgets=budgets,
        triggers=triggers,
        min_width=spec.max_cpl,
    )
    if chg:
        _rewrite_cache(run.cache_path, L.lang, L.records)
        run.logger(f"{L.lang}: condensed {len(chg)} line(s)")


def _editor_pass(run: _Run, L: _LangRun) -> None:
    """NATIVE-EDITOR pass (ON by default; --no-editor): a native subtitle editor flags only lines a
    native viewer would notice and proposes minimal rewrites, accepted past the token/number guards
    AND a pairwise check by the second-family judge; then the REPLY check at speaker turns. Runs
    before QE so the panel scores what ships. See dev_titles/editor.py."""
    if not (run.native_editor and run.qe_invoke is not None and L.records):
        return
    from .editor import native_edit

    vv = run.video_vocab or {}
    st = (getattr(run.glossary, "style", {}) or {}).get(L.lang, {}) or {}
    reg = st.get("register_conversational" if run.conversational else "register", "")
    src_all = " ".join(r.src for r in L.records).lower()
    terms = [
        t.translations.get(L.lang, "")
        for t in effective_glossary(run.glossary, run.video_vocab, run.conversational, lang=L.lang)[
            0
        ].terms
        if t.enforce == "hard" and t.source.lower() in src_all
    ]
    ech = native_edit(
        L.records,
        run.qe_invoke,
        L.lang,
        L.name,
        run.path(f"editor_state.{L.lang}.json"),
        normalize=_normalizer(L.lang, run.conversational),
        pair_judge=run.qe_second_judge,
        brief=vv.get("brief", ""),
        register=str(reg)[:300],
        glossary_terms=terms,
        speakers=vv.get("speakers") or None,
        meaning_notes=vv.get("meaning_notes") or None,
        logger=run.logger,
    )
    if ech:
        _rewrite_cache(run.cache_path, L.lang, L.records)
        run.logger(f"{L.lang}: native editor rewrote {len(ech)} line(s)")
    if vv.get("speakers"):
        from .editor import reply_edit

        rch = reply_edit(
            L.records,
            run.qe_invoke,
            L.lang,
            L.name,
            vv["speakers"],
            normalize=_normalizer(L.lang, run.conversational),
            pair_judge=run.qe_second_judge,
            brief=vv.get("brief", ""),
            state_path=run.path(f"reply_state.{L.lang}.json"),
            logger=run.logger,
        )
        if rch:
            _rewrite_cache(run.cache_path, L.lang, L.records)
            run.logger(f"{L.lang}: reply check rewrote {len(rch)} line(s)")


def _review_pass(run: _Run, L: _LangRun) -> None:
    """QE review. If a prior QE state exists for this out dir, re-judge ONLY the segments whose
    translation changed and reuse the stored panel for the rest (loaded REGARDLESS of --qe-judge, so
    any later run refreshes the REVIEW incrementally and the DQF + MQM tiers are never silently
    dropped); else the full first review. The repair passes then run whenever a major/critical is
    flagged, regardless of the DQF pass bar."""
    from .translation_qe import (
        build_judges,
        incremental_review,
        load_qe_state,
        translation_suspects,
    )

    L.qe_state_path = run.path(f"qe_state.{L.lang}.json")
    L.prior = load_qe_state(L.qe_state_path) if L.qe_state_path else None
    if L.prior is not None and run.qe_invoke is not None:
        (L.suspects, L.dqf), chg = incremental_review(
            L.records,
            run.glossary,
            L.lang,
            judges=build_judges(run.qe_invoke, run.qe_second_judge),
            lang_name=L.name,
            prior_state=L.prior,
            coverage_ran=run.backcheck,
        )
        run.logger(
            f"{L.lang}: incremental re-review — re-judged {len(chg)} changed segment(s)"
            + _dqf_note(L.dqf)
        )
    else:
        L.suspects, L.dqf = translation_suspects(
            L.records,
            run.glossary,
            L.lang,
            invoke=run.qe_invoke,
            judge=run.qe_judge,
            second_judge=run.qe_second_judge,
            coverage_ran=run.backcheck,
        )
        run.logger(f"{L.lang}: translation QE — {len(L.suspects)} suspect(s)" + _dqf_note(L.dqf))
    # gate the repair passes on the PRESENCE of a fixable defect, not the DQF pass threshold:
    # accept-only-if-better can't regress, and a clean panel is a fast no-op
    L.has_major = _has_major(L.suspects)


def _condensation_revert(run: _Run, L: _LangRun) -> None:
    """Condensation is checked for lost meaning, not broken grammar or tone: any condensed line a
    judge flagged (major OR minor, either judge) is offered its pre-condensation text back,
    accept-only-if-better on the same panel."""
    from .condense import load_originals, mark_processed
    from .translation_qe import build_judges
    from .translation_qe import qe_repair as _qe_repair
    from .translation_qe import rescore_after_repair

    cond_sp = run.path(f"condense_state.{L.lang}.json")
    orig = load_originals(cond_sp, L.records) if (run.qe_repair and L.dqf is not None) else {}
    flagged = {
        s["i"] for s in L.suspects if s.get("i") is not None and s["kind"].startswith("mqm-")
    }
    revert = sorted(i for i in orig if i in flagged)
    if not revert:
        return
    assert L.dqf is not None
    L.judges = build_judges(run.qe_invoke, run.qe_second_judge)
    mqm = [s for s in L.suspects if s["kind"].startswith("mqm-")]
    reverted = _qe_repair(
        L.records,
        L.lang,
        judges=L.judges,
        lang_name=L.name,
        retranslate=lambda rec, _crit: orig.get(rec.i, ""),
        per_seg={i: v for i, v in (L.dqf.get("per_seg") or {}).items() if i in revert},
        judge_suspects=[s for s in mqm if s["i"] in revert],
        compare_judges=L.judges,
        extra_flags=revert,
        cmp_samples=3,
        logger=run.logger,
    )
    if reverted:
        mark_processed(cond_sp, [r for r in L.records if r.i in reverted])
        _rewrite_cache(run.cache_path, L.lang, L.records)
        L.suspects, L.dqf = rescore_after_repair(
            L.records,
            run.glossary,
            L.lang,
            judges=L.judges,
            lang_name=L.name,
            prior_per_seg=L.dqf.get("per_seg"),
            prior_mqm_suspects=mqm,
            changed=reverted,
            coverage_ran=run.backcheck,
        )
        L.has_major = _has_major(L.suspects)
    run.logger(
        f"{L.lang}: condensation revert — {len(reverted)}/{len(revert)} flagged condensed "
        f"line(s) restored" + _dqf_note(L.dqf, " -> ")
    )


def _qe_repair_pass(run: _Run, L: _LangRun) -> None:
    """QE-guided auto-repair (Layer B): re-translate judge-flagged majors + deterministic garbles
    with the critique as a constraint; accept only if the full panel's re-judge improves. Native-
    editor rewrites are PROTECTED, and every repair must also win a native-viewer pairwise check."""
    if not (run.qe_repair and L.dqf is not None and (L.prior is None or L.has_major)):
        return
    from .editor import editor_rewritten, repair_check
    from .translation_qe import build_judges
    from .translation_qe import qe_repair as _qe_repair
    from .translation_qe import rescore_after_repair

    protected = (
        editor_rewritten(run.path(f"editor_state.{L.lang}.json"), L.records)
        if (run.cache_path and run.native_editor)
        else set()
    )
    accept_check = (
        repair_check(run.qe_invoke, L.name, L.records)
        if (run.native_editor and run.qe_invoke is not None)
        else None
    )
    L.judges = build_judges(run.qe_invoke, run.qe_second_judge)
    mqm = [s for s in L.suspects if s["kind"].startswith("mqm-")]
    extra = sorted(
        {
            s["i"]
            for s in L.suspects
            if s["i"] is not None and s["kind"] in ("dropped-token", "latin-typo?")
        }
    )
    # term-inconsistency -> repair the offending cues toward the canonical rendering
    extra_crit: dict[int, str] = {}
    for s in L.suspects:
        if s["kind"] == "term-inconsistency":
            note = (
                f"render '{s['source']}' as '{s['target']}' — its canonical rendering "
                f"used elsewhere in this video (consistency)"
            )
            for ci in s.get("fix_idx", []):
                extra_crit[ci] = (extra_crit[ci] + "; " + note) if ci in extra_crit else note
    retrans = make_retranslate(
        run.en,
        run.translator,
        run.glossary,
        L.lang,
        exemplar_bank=L.bank,
        video_vocab=run.video_vocab,
        invoke=run.invoke,
        backcheck=run.backcheck,
        conversational=run.conversational,
    )
    # Accept/reject uses the FULL panel (both judges): a second-judge-caught major is invisible to a
    # one-judge compare, so a genuine fix could never register; a fabrication still fails.
    fixed = _qe_repair(
        L.records,
        L.lang,
        judges=L.judges,
        lang_name=L.name,
        retranslate=retrans,
        per_seg=L.dqf.get("per_seg"),
        judge_suspects=mqm,
        compare_judges=L.judges,
        include_minor=True,
        extra_flags=extra,
        extra_critiques=extra_crit,
        cmp_samples=3,
        logger=run.logger,
        protected=protected,
        accept_check=accept_check,
    )
    if fixed:
        _rewrite_cache(run.cache_path, L.lang, L.records)
        # re-score ONLY the changed segments against the prior panel
        L.suspects, L.dqf = rescore_after_repair(
            L.records,
            run.glossary,
            L.lang,
            judges=L.judges,
            lang_name=L.name,
            prior_per_seg=L.dqf.get("per_seg"),
            prior_mqm_suspects=mqm,
            changed=fixed,
            coverage_ran=run.backcheck,
        )
        run.logger(
            f"{L.lang}: QE-repair fixed {len(fixed)} segment(s) -> "
            + (_dqf_note(L.dqf, "") or "(no re-score)")
        )


def _term_pin_pass(run: _Run, L: _LangRun) -> None:
    """TERM-PIN auto-repair (Layer C): a wrong term rendering often scores ~equal, so accept-if-
    better rejects it. Classify each remaining critique as a term fix vs a judgment call; pin term
    fixes source->correct and enforce them deterministically across all cues. The whole batch is
    rolled back if it regresses DQF (a common-word pin once mass-flipped we->you, 96.5 -> 46.9)."""
    if not (
        run.qe_repair
        and L.dqf is not None
        and run.qe_invoke is not None
        and (L.prior is None or L.has_major)
    ):
        return
    from .translation_qe import (
        apply_term_pins,
        build_judges,
        classify_term_fix,
        harvest_term_pins,
        make_override_adjudicator,
        pins_to_renderings,
        rescore_after_repair,
    )

    ln = L.name
    mqm_now = [s for s in L.suspects if str(s["kind"]).startswith("mqm-")]
    # adjudicate uncorroborated single-judge CJK->CJK flips with the DIFFERENT-family judge
    adj = make_override_adjudicator(run.qe_second_judge or run.qe_invoke, ln)
    pins, resolved = harvest_term_pins(
        L.records,
        L.lang,
        mqm_now,
        logger=run.logger,
        classify=lambda c, s, t: classify_term_fix(c, s, t, ln, run.qe_invoke),
        keep_latin=keep_latin_terms(run.glossary, run.video_vocab, L.lang),
        adjudicate=adj,
        curated={
            f: t.translations.get(L.lang, "")
            for t in run.glossary.terms
            if t.enforce == "hard"
            for f in t.all_source_forms
        },
    )
    if not pins:
        return
    L.pins = pins
    if run.cache_path:
        # cross-video ledger: recurring deterministic fixes can be promoted to the shared glossary
        from .promote import record_pins

        ticket = Path(run.cache_path).parent
        record_pins(ticket.parent / "term_ledger.jsonl", pins, L.lang, ticket.name)
    geff, _ = effective_glossary(run.glossary, run.video_vocab, run.conversational, lang=L.lang)
    pre_txt = {r.i: r.txt for r in L.records}
    pre_dqf, pre_suspects = L.dqf, L.suspects
    changed2 = apply_term_pins(
        L.records,
        pins,
        geff,
        L.lang,
        protect=keep_latin_terms(run.glossary, run.video_vocab, L.lang),
    )
    if not changed2:
        return
    _rewrite_cache(run.cache_path, L.lang, L.records)
    # re-judge the pin-changed segments: every critique on those cues is now stale
    judges = L.judges or build_judges(run.qe_invoke, run.qe_second_judge)
    L.suspects, L.dqf = rescore_after_repair(
        L.records,
        run.glossary,
        L.lang,
        judges=judges,
        lang_name=L.name,
        prior_per_seg=L.dqf.get("per_seg"),
        prior_mqm_suspects=[s for s in L.suspects if str(s["kind"]).startswith("mqm-")],
        changed=list(changed2),
        coverage_ran=run.backcheck,
    )
    if pre_dqf is not None and L.dqf is not None and L.dqf["score"] + 0.5 < pre_dqf["score"]:
        # ACCEPT-ONLY-IF-BETTER: revert the whole batch to the pre-pin text/score
        for r in L.records:
            if r.i in pre_txt:
                r.txt = pre_txt[r.i]
        _rewrite_cache(run.cache_path, L.lang, L.records)
        run.logger(
            f"{L.lang}: term-pins REVERTED — batch regressed DQF "
            f"{pre_dqf['score']}->{L.dqf['score']} ({len(pins)} pin(s), "
            f"{len(changed2)} cue(s)); kept pre-pin output. Pins left as review "
            f"suggestions: {[p['source'] for p in pins]}"
        )
        L.suspects, L.dqf, L.term_pins, L.pins = pre_suspects, pre_dqf, {}, []
    else:
        L.term_pins = pins_to_renderings(pins, L.lang)
        run.logger(
            f"{L.lang}: term-pin auto-repair — {len(pins)} pin(s) across "
            f"{len(changed2)} cue(s) re-judged; {len(resolved)} critique(s) resolved"
        )


def _dnt_autocorrect(run: _Run, L: _LangRun) -> None:
    """Self-heal: re-translate any sentence where a keep-Latin source term didn't survive in the
    target (masking now protects it); accept only a re-translation that restores the term(s)."""
    kl = [t for t in keep_latin_terms(run.glossary, run.video_vocab, L.lang) if len(t) >= 3]
    need = [(r, [t for t in kl if t in r.src and t not in r.txt]) for r in L.records]
    need = [(r, m) for r, m in need if m]
    if not (need and run.qe_invoke is not None):
        return
    rt = make_retranslate(
        run.en,
        run.translator,
        run.glossary,
        L.lang,
        exemplar_bank=L.bank,
        video_vocab=run.video_vocab,
        invoke=run.qe_invoke,
        backcheck=run.backcheck,
        conversational=run.conversational,
    )
    for r, miss in need:
        nt = (rt(r, f"Keep verbatim in Latin, do NOT translate: {', '.join(miss)}") or "").strip()
        if nt and all(t in nt for t in miss):
            r.txt = nt
            run.logger(f"{L.lang}: DNT auto-correct — re-translated #{r.i} to restore {miss}")


def _finalize_lang(run: _Run, L: _LangRun) -> LangOutput:
    """Final enforcement issues, the deliverable subtitle, deterministic last fixes and audits."""
    # enforcement issues recomputed from the FINAL records (the first-pass ones go stale)
    geff2, _ = effective_glossary(run.glossary, run.video_vocab, run.conversational, lang=L.lang)
    issues = [x for r in L.records for x in geff2.enforce(r.txt, r.src, L.lang)[1]]
    sub = _subtitle_from_records(
        run.en,
        L.records,
        run.glossary,
        L.lang,
        run.video_vocab,
        run.conversational,
        invoke=run.qe_invoke,
        reflow_cache_path=L.rcp,
    )
    if L.pins:
        # DURABLE PIN GUARD: reflow can re-emit a pinned avoid-form; re-apply on the final cue text
        from .translation_qe import pin_substr_fix

        n = 0
        for cue in sub.cues:
            new_lines = [pin_substr_fix(ln, L.pins, L.lang) for ln in cue.lines]
            if new_lines != cue.lines:
                cue.lines = new_lines
                n += 1
        if n:
            run.logger(f"{L.lang}: durable pin guard — re-applied pins on {n} reflowed cue(s)")
    # DEGENERATE-REPEAT COLLAPSE ('マーケティングマーケティング・…' -> 'マーケティング・…'), >=3-char units
    from .cleanup import collapse_immediate_repeats

    rc = 0
    for cue in sub.cues:
        new_lines = [collapse_immediate_repeats(ln) for ln in cue.lines]
        if new_lines != cue.lines:
            cue.lines = new_lines
            rc += 1
    if rc:
        run.logger(f"{L.lang}: collapsed degenerate immediate-repeats in {rc} cue(s)")
    suspects = L.suspects
    # DNT-SURVIVAL AUDIT: a keep-Latin term ABSENT from the whole target is a real garble; one that
    # merely moved to an adjacent cue on re-segmentation is not
    dnt_viol = dnt_survival_violations(run.en, sub, run.glossary, run.video_vocab, L.lang)
    garbled = [v for v in dnt_viol if not v.get("survived")]
    moved = [v for v in dnt_viol if v.get("survived")]
    if moved:
        run.logger(
            f"{L.lang}: DNT-survival — {len(moved)} keep-Latin term(s) present but in an adjacent "
            f"cue (re-segmentation, not garbled): {sorted({v['term'] for v in moved})}"
        )
    if garbled:
        run.logger(
            f"{L.lang}: DNT-survival AUDIT — {len(garbled)} keep-Latin term(s) NOT in the target "
            f"at all (translated/garbled): {sorted({v['term'] for v in garbled})}"
        )
        for v in garbled:
            suspects.append(
                {
                    "kind": "dnt-not-survived",
                    "ts": "",
                    "detail": f"cue {v['cue']}: keep-Latin term {v['term']!r} was "
                    "translated/garbled — should stay Latin verbatim",
                }
            )
    # ADJACENT-DUPLICATE CLAUSE audit: usually a re-emit that displaced real content
    for d in adjacent_duplicate_clauses(sub):
        suspects.append(
            {
                "kind": "adjacent-duplicate",
                "ts": "",
                "detail": f"cue {d['cue']}: repeats a line from the previous cue "
                f"({d['dup'][:40]!r}) — likely a re-emit that dropped this cue's content",
            }
        )
    lo = LangOutput(
        L.lang,
        sub,
        enforcement_issues=issues,
        residuals=[(r.src, r.residual_miss) for r in L.records if r.residual_miss],
        qe_suspects=suspects,
        conversational=run.conversational,
        term_pins=L.term_pins,
    )
    if L.dqf is not None:
        lo.qe_metric = "DQF"
        lo.qe_system_score = L.dqf["score"]
        lo.qe_weak_segments = L.dqf["weak"]
        lo.qe_pass = L.dqf["pass"]
    _write_status(
        run.cache_path,
        phase="done",
        lang=L.lang,
        cues=len(sub.cues),
        **{f"dqf_{L.lang}": (L.dqf["score"] if L.dqf is not None else None)},
    )
    return lo
