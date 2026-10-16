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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
    import fcntl
    import os

    path = os.path.expanduser("~/.dev_titles_llm.lock")
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
        return lambda t, src=None: restore_question_mark(src, _zh_polite_you(base(t)), lang)
    return lambda t, src=None: restore_question_mark(src, base(t), lang)


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
    spec = spec_for(lang)
    lang_name = _LANG_NAME.get(lang, lang)
    normalize = _normalizer(lang, conversational)
    g, extra_dnt = effective_glossary(glossary, video_vocab, conversational, lang=lang)
    # Auto-protect CLI commands / env vars found anywhere in this video as DNT (doc-global)
    # — they get partially translated/recased otherwise (create-expo-app -> create-expo-アプリ).
    from .qa import protected_spans

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

    # resumable cache — indexed by sentence position AND by source text. The by-text index makes
    # reuse robust to SEGMENTATION SHIFTS: an EN edit that adds/removes a sentence renumbers every
    # later sentence, so an index-only lookup would miss all of them (src != cache[i].src) and
    # re-translate the whole tail. Keying unchanged source text by content lets them reuse their
    # existing translation regardless of their new index — only genuinely new/edited text re-translates.
    cache: dict[int, SentenceRecord] = {}
    cache_by_src: dict[str, SentenceRecord] = {}
    cf = None
    if cache_path:
        p = Path(cache_path)
        if p.exists():
            for line in p.open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    if r["lang"] == lang:
                        rec0 = SentenceRecord(**r)
                        cache[r["i"]] = rec0
                        cache_by_src.setdefault(rec0.src, rec0)  # first wins (stable for dup lines)
        cf = p.open("a", encoding="utf-8")

    sents = merge_en_to_sentences(en)
    ctx = " ".join(s.text for s in sents)
    logger(f"{lang}: {len(en.cues)} EN cues -> {len(sents)} sentences")
    cues: list = []
    issues: list[EnforcementIssue] = []
    records: list[SentenceRecord] = []

    for i, s in enumerate(sents):
        # Cache is index-keyed, but REUSE ONLY IF THE SOURCE IS UNCHANGED. An operator EN
        # correction (e.g. 'Chile'->'LG') keeps the same sentence index, so an index-only reuse
        # would ship the STALE translation of the OLD text and never re-translate the fix. Compare
        # the cached source to the current one; on any change, fall through to re-translation.
        reuse = cache[i] if (i in cache and cache[i].src == s.text) else cache_by_src.get(s.text)
        if reuse is not None and stale_untranslated_reuse(reuse.txt, s.text, g, lang):
            # A glossary fix made a term that this cached target left Latin now TRANSLATABLE
            # (App Store -> 应用商店); enforce can't reverse it, so drop the reuse and re-translate
            # THIS cue only. The cache self-heals against glossary changes without a full clear.
            logger(
                f"{lang}: cache-reuse stale #{i} (glossary now translates a Latin-left term) — re-translating"
            )
            reuse = None
        if reuse is not None:
            # Reuse the cached translation for this UNCHANGED source (same index, or moved by a
            # segmentation shift). Re-enforce against the CURRENT glossary (deterministic, no LLM)
            # and APPLY its fixes — refreshes stale flags (pre-DNT 'Acme missing' FPs on 'Acme
            # Q') AND normalizes same-cue avoid-forms to canonical (指导文件->引导文档), so a glossary
            # change lands on cache reuse without a full re-translation.
            rec = SentenceRecord(
                lang, i, s.text, reuse.txt, reuse.residual_miss, list(reuse.issues)
            )
            txt, iss = g.enforce(rec.txt, s.text, lang)
            # Normalize like the fresh/TM paths (they call normalize; only this reuse branch used
            # to skip it) — folds conversational 您->你 + punct on a REUSED translation. Without
            # this, a segment cached under a non-conversational run keeps its formal 您 forever on
            # reuse, so re-running --conversational never actually converted the register.
            rec.txt = normalize(
                txt, s.text
            )  # keep records/QE/cues in sync with the normalized text
            # Do NOT append content-reused records to the append-only cache: the translation is
            # already persisted (under its original index) and on resume the by-source index
            # re-finds it — re-appending would accumulate duplicate (src) entries across runs.
        elif tm is not None and (hit := tm.exact(s.text, lang)) is not None:
            # TM EXACT-MATCH REUSE: an identical source line already has an approved
            # target — reuse it verbatim for cross-video consistency (no LLM, no gates).
            txt, iss, residual = hit, [], ""
            logger(f"{lang}: TM exact-match reuse #{i}")
            txt = normalize(txt, s.text)
            rec = SentenceRecord(lang, i, s.text, txt, residual, [])
            if cf:
                cf.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
                cf.flush()
        else:
            # Soft budget with completeness headroom (1.15x): the tight cps budget
            # pressured the model to DROP clauses; cps stays a review-only warning.
            budget = max(
                int(spec.max_cps * 1.15 * _budget_scale(lang) * max(s.end - s.start, spec.min_dur)),
                12,
            )
            # RESILIENCE: a single segment's LLM call can hang/time out (Bedrock under load,
            # or a dropped internet connection). RETRY with backoff first — a transient outage
            # that recovers within ~30s is survived here (so the segment gets a full-quality
            # translation regardless of whether the QE-repair pass runs later). Only after the
            # retries are exhausted do we keep the EN placeholder + flag it, rather than abort the
            # whole multi-hour run. (A crash here previously lost a full ja pass at segment #4.)
            import time as _time

            txt, iss, residual = "", [], ""
            last_exc: Exception | None = None
            for _attempt in range(3):
                try:
                    txt, iss = translate_segment(
                        translator,
                        g,
                        s.text,
                        lang,
                        full_context=ctx,
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
                    )
                    residual = ""
                    if backcheck:
                        txt, iss, residual = _coverage_editfix(
                            invoke, g, s, lang, lang_name, txt, iss, edit_fix_rounds, logger, i
                        )
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    if _attempt < 2:
                        logger(
                            f"{lang}: segment #{i} attempt {_attempt + 1} failed "
                            f"({type(e).__name__}); retrying"
                        )
                        _time.sleep(_SEGMENT_RETRY_BACKOFF[_attempt])  # ride out a transient drop
            if last_exc is not None:
                logger(
                    f"{lang}: segment #{i} translation FAILED ({type(last_exc).__name__}) after "
                    f"retries — keeping EN, flagged for review"
                )
                txt, iss, residual = (
                    s.text,
                    [],
                    f"translation failed ({type(last_exc).__name__}) — needs manual translation",
                )
            txt = normalize(txt, s.text)
            rec = SentenceRecord(
                lang,
                i,
                s.text,
                txt,
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
    # Build shipping cues: FREE independent-timed CJK segmentation when opted in (default OFF
    # during the 1:1 migration — see resegment.free_segment_default), else strict 1:1 EN-cue
    # alignment (legacy: one target cue per EN cue, same timestamps).
    cjk_texts = [r.txt for r in records]  # already normalized when built/cached
    cues = build_target_cues(en, sents, cjk_texts, spec, keep_together=keep_together)
    sub = Subtitle(lang, cues).renumber()
    rep = validity_report(sub, conversational)
    logger(
        f"{lang}: DONE {len(cues)} cues, {rep['cues_with_errors']} err-cues, "
        f"{sum(1 for x in issues if x.kind == 'missing')} missing"
    )
    return sub, issues, records


def _omission_gate(
    translator, g, s, lang, txt, iss, ctx, budget, exemplar_bank, extra_dnt, logger, i
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
    ctx = " ".join(s.text for s in sents)
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
        prev = sents[rec.i - 1].text if rec.i > 0 else ""
        nxt = sents[rec.i + 1].text if rec.i + 1 < len(sents) else ""
        local = ""
        if prev or nxt:
            local = (
                "\nLOCAL CONTEXT (for disambiguation only — do NOT translate these, only the "
                f"line above): [previous] {prev!r}  [next] {nxt!r}"
            )
        txt, iss = translate_segment(
            translator,
            g,
            s.text,
            lang,
            full_context=ctx,
            reflect=True,
            char_budget=budget,
            exemplar_bank=exemplar_bank,
            correction_hint=critique + local,
            extra_dnt=extra_dnt,
        )
        txt, iss = _omission_gate(
            translator, g, s, lang, txt, iss, ctx, budget, exemplar_bank, extra_dnt, _noop, rec.i
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

    cues = build_target_cues(en, sents, cjk_texts, spec, keep_together=keep_together, reflow=reflow)
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
    (llm_serialize_lock) so concurrent runs queue instead of throttling Bedrock. Thin wrapper over _localize_en_impl; see it for the full pipeline."""
    with llm_serialize_lock(logger):
        return _localize_en_impl(*args, logger=logger, **kwargs)


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
    logger=_noop,
) -> list[LangOutput]:
    """Localize an EN subtitle into each target language. Returns LangOutputs ready
    for `review.build_review_bundle`. `exemplar_bank_for(lang) -> bank` optionally
    supplies few-shot style anchors per language. A `tm` (TranslationMemory) adds
    exact-match verbatim reuse + series-specific fuzzy exemplars (falls back to it for
    the exemplar bank when `exemplar_bank_for` isn't given). Reference-free translation
    QE runs per language (deterministic always; a calibrated-DQF LLM MQM judge when `qe_judge`)."""
    from .translation_qe import build_judges, incremental_review, load_qe_state
    from .condense import load_originals, mark_processed
    from .translation_qe import qe_repair as _qe_repair
    from .translation_qe import rescore_after_repair, save_qe_state, translation_suspects

    qe_invoke = invoke or getattr(translator, "_invoke", None)
    outputs: list[LangOutput] = []
    for lang in langs:
        bank = (
            exemplar_bank_for(lang)
            if exemplar_bank_for
            else (tm.bank(lang) if tm is not None else None)
        )
        rcp = (Path(cache_path).parent / f"reflow_cache.{lang}.json") if cache_path else None
        _write_status(cache_path, phase="translate", lang=lang)
        sub, issues, records = translate_subtitle(
            en,
            translator,
            glossary,
            lang,
            exemplar_bank=bank,
            video_vocab=video_vocab,
            tm=tm,
            invoke=invoke,
            backcheck=backcheck,
            cache_path=cache_path,
            conversational=conversational,
            logger=logger,
        )
        # CONDENSATION (ON by default, ja; disable with --no-condense): tighten finished translations like a subtitle editor before
        # QE, so the panel judges what ships. See dev_titles/condense.py and its A/B notes.
        if condense and lang == "ja" and qe_invoke is not None and records:
            from .condense import condense_lang

            _sp = (Path(cache_path).parent / f"condense_state.{lang}.json") if cache_path else None
            # per-sentence character budget = reading-speed limit x the sentence's speech window
            _spec = spec_for(lang)
            _sents = merge_en_to_sentences(en)
            _budgets = {
                i: max(int(_spec.max_cps * max(s.end - s.start, _spec.min_dur)), 6)
                for i, s in enumerate(_sents)
            }
            # ...but only lines CLEARLY too fast are touched: native-reviewed ja ships 20% of cues
            # above 8.5 cps (p90 ~10.2), and squeezing near-limit lines to the exact budget is what
            # produced broken grammar / dropped greetings on a tutorial test.
            _trig_cps = float(os.environ.get("SUBTITLE_CONDENSE_TRIGGER_CPS", "10.0"))
            _triggers = {
                i: max(int(_trig_cps * max(s.end - s.start, _spec.min_dur)), 6)
                for i, s in enumerate(_sents)
            }
            _chg = condense_lang(
                records,
                qe_invoke,
                lang,
                _LANG_NAME.get(lang, lang),
                _sp,
                coverage_missing=coverage_missing,
                normalize=_normalizer(lang, conversational),
                logger=logger,
                budgets=_budgets,
                triggers=_triggers,
            )
            if _chg:
                _rewrite_cache(cache_path, lang, records)
                logger(f"{lang}: condensed {len(_chg)} line(s)")
        # AUTOMATIC INCREMENTAL RE-REVIEW: if a prior QE state exists for this out dir (a
        # previous --qe-judge run), re-judge ONLY the segments whose translation changed since
        # then and reuse the stored panel for the rest — a cheap, correct refresh of the full
        # REVIEW (DQF + MQM tiers) after any cache mutation (QE-repair recovery, term-pins,
        # a manual fix). First review (no state) runs the full panel. On a re-review, the panel
        # is refreshed incrementally; the repair/term-pin passes below then run too whenever a
        # major/critical is flagged (regardless of the DQF pass bar), and no-op when the panel
        # is clean.
        # Load a prior QE state REGARDLESS of the --qe-judge flag: once a video has been
        # reviewed, ANY later run (even enforce-only, e.g. after a term-pin/manual cache edit)
        # AUTO-refreshes the full REVIEW incrementally — re-judging only changed segments — so
        # the DQF + MQM tiers are never silently dropped and always reflect the current cache.
        _qe_state_path = (Path(cache_path).parent / f"qe_state.{lang}.json") if cache_path else None
        _prior = load_qe_state(_qe_state_path) if _qe_state_path else None
        if _prior is not None and qe_invoke is not None:
            _judges = build_judges(qe_invoke, qe_second_judge)
            (suspects, dqf), _chg = incremental_review(
                records,
                glossary,
                lang,
                judges=_judges,
                lang_name=_LANG_NAME.get(lang, lang),
                prior_state=_prior,
                coverage_ran=backcheck,
            )
            logger(
                f"{lang}: incremental re-review — re-judged {len(_chg)} changed segment(s)"
                + (f", DQF {dqf['score']} {'PASS' if dqf['pass'] else 'FAIL'}" if dqf else "")
            )
        else:
            suspects, dqf = translation_suspects(
                records,
                glossary,
                lang,
                invoke=qe_invoke,
                judge=qe_judge,
                second_judge=qe_second_judge,
                coverage_ran=backcheck,
            )
            logger(
                f"{lang}: translation QE — {len(suspects)} suspect(s)"
                + (f", DQF {dqf['score']} {'PASS' if dqf['pass'] else 'FAIL'}" if dqf else "")
            )

        # Any judge-flagged major/critical to fix? Gate the repair passes on the PRESENCE of a
        # fixable defect, NOT on the (arbitrary) DQF pass threshold — a video can clear the bar
        # yet still carry a fixable major (wrong term, dropped clause), and there's no reason to
        # leave it. accept-only-if-better + targeted accepts can't regress, so running it on a
        # passing video is safe; a clean panel (no majors) is a fast no-op.
        _has_major = any(
            "major" in str(s.get("kind", "")) or "critical" in str(s.get("kind", ""))
            for s in suspects
        )
        # CONDENSATION REVERT: condensation is checked for lost meaning, not for broken grammar or
        # tone. Any condensed line a judge flagged (major OR minor, either judge) gets its
        # pre-condensation text offered back — accept-only-if-better on the same panel, so a
        # revert lands only when the original genuinely scores better.
        _cond_sp = (Path(cache_path).parent / f"condense_state.{lang}.json") if cache_path else None
        _orig = load_originals(_cond_sp, records) if (qe_repair and dqf is not None) else {}
        _flagged = {
            s["i"] for s in suspects if s.get("i") is not None and s["kind"].startswith("mqm-")
        }
        _revert = sorted(i for i in _orig if i in _flagged)
        if _revert:
            judges = build_judges(qe_invoke, qe_second_judge)
            _mqm = [s for s in suspects if s["kind"].startswith("mqm-")]
            _reverted = _qe_repair(
                records,
                lang,
                judges=judges,
                lang_name=_LANG_NAME.get(lang, lang),
                retranslate=lambda rec, _crit: _orig.get(rec.i, ""),
                per_seg={i: v for i, v in (dqf.get("per_seg") or {}).items() if i in _revert},
                judge_suspects=[s for s in _mqm if s["i"] in _revert],
                compare_judges=judges,
                extra_flags=_revert,
                cmp_samples=3,
                logger=logger,
            )
            if _reverted:
                mark_processed(_cond_sp, [r for r in records if r.i in _reverted])
                _rewrite_cache(cache_path, lang, records)
                sub = _subtitle_from_records(
                    en, records, glossary, lang, video_vocab, conversational, reflow_cache_path=rcp
                )
                suspects, dqf = rescore_after_repair(
                    records,
                    glossary,
                    lang,
                    judges=judges,
                    lang_name=_LANG_NAME.get(lang, lang),
                    prior_per_seg=dqf.get("per_seg"),
                    prior_mqm_suspects=_mqm,
                    changed=_reverted,
                    coverage_ran=backcheck,
                )
                _has_major = any(
                    "major" in str(s.get("kind", "")) or "critical" in str(s.get("kind", ""))
                    for s in suspects
                )
            logger(
                f"{lang}: condensation revert — {len(_reverted)}/{len(_revert)} flagged condensed "
                f"line(s) restored"
                + (f" -> DQF {dqf['score']} {'PASS' if dqf['pass'] else 'FAIL'}" if dqf else "")
            )
        # QE-guided auto-repair (Layer B): re-translate judge-flagged majors + deterministic
        # garbles with the critique as a constraint; accept only if the re-judge improves.
        if qe_repair and dqf is not None and (_prior is None or _has_major):
            judges = build_judges(qe_invoke, qe_second_judge)
            mqm = [s for s in suspects if s["kind"].startswith("mqm-")]
            extra = sorted(
                {
                    s["i"]
                    for s in suspects
                    if s["i"] is not None and s["kind"] in ("dropped-token", "latin-typo?")
                }
            )
            # term-inconsistency -> repair the offending cues toward the canonical rendering
            # (closes the gap where inconsistencies were detected but wired to no fixer).
            extra_crit: dict[int, str] = {}
            for s in suspects:
                if s["kind"] == "term-inconsistency":
                    note = (
                        f"render '{s['source']}' as '{s['target']}' — its canonical rendering "
                        f"used elsewhere in this video (consistency)"
                    )
                    for ci in s.get("fix_idx", []):
                        extra_crit[ci] = (
                            (extra_crit[ci] + "; " + note) if ci in extra_crit else note
                        )
            retrans = make_retranslate(
                en,
                translator,
                glossary,
                lang,
                exemplar_bank=bank,
                video_vocab=video_vocab,
                invoke=invoke,
                backcheck=backcheck,
                conversational=conversational,
            )
            # Accept/reject uses the FULL panel (both judges, corroboration aggregation), NOT
            # Claude-only: a codex-CAUGHT major (Claude didn't flag — e.g. "Because Nimbus does"
            # = supports MCPs) is invisible to a Claude-only compare, so a genuine fix could
            # never register as an improvement. Full panel lets a fix that satisfies the
            # flagging judge (without a new disagreement) land, while a fabrication that breaks
            # the text (penalty rises) still fails the strict-drop test. Only major-flagged
            # segments are retried (~4-5/lang), so the extra second-judge cost is bounded.
            fixed = _qe_repair(
                records,
                lang,
                judges=judges,
                lang_name=_LANG_NAME.get(lang, lang),
                retranslate=retrans,
                per_seg=dqf.get("per_seg"),
                judge_suspects=mqm,
                compare_judges=judges,
                include_minor=True,
                extra_flags=extra,
                extra_critiques=extra_crit,
                cmp_samples=3,
                logger=logger,
            )
            if fixed:
                _rewrite_cache(cache_path, lang, records)
                sub = _subtitle_from_records(
                    en, records, glossary, lang, video_vocab, conversational, reflow_cache_path=rcp
                )
                # Re-score ONLY the changed segments against the prior panel (low-noise +
                # avoids a second full Codex panel) rather than re-drawing the whole panel.
                suspects, dqf = rescore_after_repair(
                    records,
                    glossary,
                    lang,
                    judges=judges,
                    lang_name=_LANG_NAME.get(lang, lang),
                    prior_per_seg=dqf.get("per_seg"),
                    prior_mqm_suspects=mqm,
                    changed=fixed,
                    coverage_ran=backcheck,
                )
                logger(
                    f"{lang}: QE-repair fixed {len(fixed)} segment(s) -> "
                    + (
                        f"DQF {dqf['score']} {'PASS' if dqf['pass'] else 'FAIL'}"
                        if dqf
                        else "(no re-score)"
                    )
                )

        # TERM-PIN auto-repair (Layer C): mechanical term-substitution fixes the retranslate
        # loop misses — a wrong term rendering often scores ~equal, so accept-if-better rejects
        # it and it falls through to 'human judgment'. Classify each remaining mqm critique as a
        # term fix vs a judgment call; for term fixes, pin source->correct (avoid wrong) and
        # enforce DETERMINISTICALLY across ALL cues (every occurrence + inconsistency, no
        # re-translation). Judgment calls (synonym choice, idiom, rephrase) stay for the human.
        term_pins: dict = {}
        pins: list = []
        if (
            qe_repair
            and dqf is not None
            and qe_invoke is not None
            and (_prior is None or _has_major)
        ):
            from .translation_qe import (
                apply_term_pins,
                classify_term_fix,
                harvest_term_pins,
                make_override_adjudicator,
                pins_to_renderings,
            )

            _ln = _LANG_NAME.get(lang, lang)
            _mqm_now = [s for s in suspects if str(s["kind"]).startswith("mqm-")]
            # Adjudicate uncorroborated single-judge CJK->CJK flips with the DIFFERENT-family judge
            # (independent of the primary that usually raises them) — verifies the specific
            # correction against the source so a right flip (promise->前景) applies while a
            # wrong-sense one (agent->特工) stays held. Falls back to the primary if no 2nd judge.
            _adj = make_override_adjudicator(qe_second_judge or qe_invoke, _ln)
            pins, resolved = harvest_term_pins(
                records,
                lang,
                _mqm_now,
                logger=logger,
                classify=lambda c, s, t: classify_term_fix(c, s, t, _ln, qe_invoke),
                keep_latin=keep_latin_terms(glossary, video_vocab, lang),
                adjudicate=_adj,
                curated={
                    f: t.translations.get(lang, "")
                    for t in glossary.terms
                    if t.enforce == "hard"
                    for f in t.all_source_forms
                },
            )
            if pins:
                # Cross-video ledger: record these deterministic term fixes so recurring
                # ones (>= min_videos, consistent rendering) can be auto-promoted to the
                # shared glossary later (see promote.promote_recurring_terms).
                if cache_path:
                    from .promote import record_pins

                    _ticket = Path(cache_path).parent
                    record_pins(_ticket.parent / "term_ledger.jsonl", pins, lang, _ticket.name)
                _geff, _ = effective_glossary(glossary, video_vocab, conversational, lang=lang)
                # Protect keep-Latin identifiers (react-native-video, Acme Q…) from a single-word
                # pin firing INSIDE them (the 'video'->视频 corrupting 'react-native-video' bug).
                # SNAPSHOT the pre-pin text + score so the batch can be ROLLED BACK if it
                # regresses (accept-only-if-better, like qe_repair). A pin harvested from ONE cue
                # is applied source-gated across ALL cues with that term; for a common word (the
                # 'you' 我们->你 disaster: fired in ~260 cues, mass-flipping we->you, DQF 96.5->46.9)
                # that can crater quality. Unconditional application shipped it — now we revert.
                _pre_txt = {r.i: r.txt for r in records}
                _pre_dqf, _pre_suspects = dqf, suspects
                changed2 = apply_term_pins(
                    records,
                    pins,
                    _geff,
                    lang,
                    protect=keep_latin_terms(glossary, video_vocab, lang),
                )
                if changed2:
                    _rewrite_cache(cache_path, lang, records)
                    sub = _subtitle_from_records(
                        en,
                        records,
                        glossary,
                        lang,
                        video_vocab,
                        conversational,
                        reflow_cache_path=rcp,
                    )
                    # RE-JUDGE the pin-changed segments (not just drop the explicitly-resolved
                    # critiques): a pin fixes the TEXT, so every critique on those cues is now
                    # stale — re-scoring refreshes them all + keeps the saved qe_state accurate
                    # (else a stale critique, e.g. 'China left untranslated', survives into the
                    # next re-review even though the text is fixed).
                    suspects, dqf = rescore_after_repair(
                        records,
                        glossary,
                        lang,
                        judges=judges,
                        lang_name=_LANG_NAME.get(lang, lang),
                        prior_per_seg=dqf.get("per_seg"),
                        prior_mqm_suspects=[
                            s for s in suspects if str(s["kind"]).startswith("mqm-")
                        ],
                        changed=list(changed2),
                        coverage_ran=backcheck,
                    )
                    # ACCEPT-ONLY-IF-BETTER: if the pin batch regressed the DQF (a common-word pin
                    # mass-corrupted the target), REVERT the whole batch to the pre-pin text/score.
                    # Losing a few good product-name pins (App.tsx…) is far better than shipping a
                    # crater; the dropped pins still surface as review suggestions.
                    if (
                        _pre_dqf is not None
                        and dqf is not None
                        and dqf["score"] + 0.5 < _pre_dqf["score"]
                    ):
                        for r in records:
                            if r.i in _pre_txt:
                                r.txt = _pre_txt[r.i]
                        _rewrite_cache(cache_path, lang, records)
                        sub = _subtitle_from_records(
                            en,
                            records,
                            glossary,
                            lang,
                            video_vocab,
                            conversational,
                            reflow_cache_path=rcp,
                        )
                        logger(
                            f"{lang}: term-pins REVERTED — batch regressed DQF "
                            f"{_pre_dqf['score']}->{dqf['score']} ({len(pins)} pin(s), "
                            f"{len(changed2)} cue(s)); kept pre-pin output. Pins left as review "
                            f"suggestions: {[p['source'] for p in pins]}"
                        )
                        suspects, dqf, term_pins, pins = _pre_suspects, _pre_dqf, {}, []
                    else:
                        term_pins = pins_to_renderings(pins, lang)
                        logger(
                            f"{lang}: term-pin auto-repair — {len(pins)} pin(s) across "
                            f"{len(changed2)} cue(s) re-judged; {len(resolved)} critique(s) resolved"
                        )

        # Persist the (final, post-repair) QE panel so the NEXT run can re-review incrementally
        # — whenever we have a panel (a --qe-judge run, or an auto-refresh from prior state).
        if dqf is not None and _qe_state_path is not None:
            save_qe_state(_qe_state_path, records, dqf, suspects)

        # DNT AUTO-CORRECT (self-heal, not just flag): re-translate any sentence where a
        # keep-Latin term in the source didn't survive in the target. Masking now protects the
        # term (it's in extra_dnt), so the re-translate is guaranteed to keep it Latin. Only
        # accepts a re-translation that actually restores the term(s). On a normal run masking
        # already kept them, so `need` is empty and this costs nothing.
        _kl = [t for t in keep_latin_terms(glossary, video_vocab, lang) if len(t) >= 3]
        need = [(r, [t for t in _kl if t in r.src and t not in r.txt]) for r in records]
        need = [(r, m) for r, m in need if m]
        if need and qe_invoke is not None:
            _rt = make_retranslate(
                en,
                translator,
                glossary,
                lang,
                exemplar_bank=bank,
                video_vocab=video_vocab,
                invoke=qe_invoke,
                backcheck=backcheck,
                conversational=conversational,
            )
            for r, miss in need:
                nt = (
                    _rt(r, f"Keep verbatim in Latin, do NOT translate: {', '.join(miss)}") or ""
                ).strip()
                if nt and all(t in nt for t in miss):
                    r.txt = nt
                    logger(f"{lang}: DNT auto-correct — re-translated #{r.i} to restore {miss}")

        # RECOMPUTE enforcement issues from the FINAL records — the first-pass `issues` go stale
        # after qe_repair/term-pin/enforce change a cue (e.g. a transliterated 'Nimbus' since
        # normalized to Latin still showed as 'expected Nimbus — not found'). Re-enforce each final
        # record against the effective glossary so the REVIEW's enforcement section is accurate.
        _geff2, _ = effective_glossary(glossary, video_vocab, conversational, lang=lang)
        issues = [x for r in records for x in _geff2.enforce(r.txt, r.src, lang)[1]]
        # FINAL deliverable: rebuild the sub with meaning-aligned LLM RE-FLOW (batched, lossless-
        # guarded) — the shipping 1:1 format. Runs once per language, using the final records.
        sub = _subtitle_from_records(
            en,
            records,
            glossary,
            lang,
            video_vocab,
            conversational,
            invoke=qe_invoke,
            reflow_cache_path=rcp,
        )
        # DURABLE PIN GUARD: reflow (LLM cue-split) can re-emit a harvested pin's avoid-form or
        # the raw source term even after the record was fixed (memory: reflow-bypasses-term-pins).
        # Re-apply the run's harvested pins on the FINAL reflowed cue text so no pinned avoid-form
        # ships. String-level + distinctive-source-gated (see pin_substr_fix).
        if pins:
            from .translation_qe import pin_substr_fix

            _n = 0
            for cue in sub.cues:
                new_lines = [pin_substr_fix(ln, pins, lang) for ln in cue.lines]
                if new_lines != cue.lines:
                    cue.lines = new_lines
                    _n += 1
            if _n:
                logger(f"{lang}: durable pin guard — re-applied pins on {_n} reflowed cue(s)")
        # DEGENERATE-REPEAT COLLAPSE: auto-fix an immediately-repeated substring the model
        # sometimes re-emits ('マーケティングマーケティング・…' -> 'マーケティング・…'). Deterministic,
        # target-side analog of the EN dedupe; >=3-char units only so emphatic short doublings are
        # safe. (is_repeated_text only FLAGGED this; here it's repaired.)
        from .cleanup import collapse_immediate_repeats

        _rc = 0
        for cue in sub.cues:
            new_lines = [collapse_immediate_repeats(ln) for ln in cue.lines]
            if new_lines != cue.lines:
                cue.lines = new_lines
                _rc += 1
        if _rc:
            logger(f"{lang}: collapsed degenerate immediate-repeats in {_rc} cue(s)")
        # DNT-SURVIVAL AUDIT (backstop, detection independent of masking): each keep-Latin term
        # in EN must survive verbatim in the target (segmentation-agnostic — see
        # dnt_survival_violations). Any that don't were translated/garbled despite masking —
        # surface them so a slip can't ship silently again.
        dnt_viol = dnt_survival_violations(en, sub, glossary, video_vocab, lang)
        # Only a term ABSENT from the whole target is a real garble; one that "survived" merely
        # moved to an adjacent cue on re-segmentation — don't flag it as translated/garbled (that
        # over-stated the review, sending reviewers to hunt for errors that aren't there).
        garbled = [v for v in dnt_viol if not v.get("survived")]
        moved = [v for v in dnt_viol if v.get("survived")]
        if moved:
            logger(
                f"{lang}: DNT-survival — {len(moved)} keep-Latin term(s) present but in an adjacent "
                f"cue (re-segmentation, not garbled): {sorted({v['term'] for v in moved})}"
            )
        if garbled:
            _terms = sorted({v["term"] for v in garbled})
            logger(
                f"{lang}: DNT-survival AUDIT — {len(garbled)} keep-Latin term(s) NOT in the target "
                f"at all (translated/garbled): {_terms}"
            )
            for _v in garbled:
                suspects.append(
                    {
                        "kind": "dnt-not-survived",
                        "ts": "",
                        "detail": f"cue {_v['cue']}: keep-Latin term {_v['term']!r} was "
                        "translated/garbled — should stay Latin verbatim",
                    }
                )
        # ADJACENT-DUPLICATE CLAUSE audit: a cue that repeats a whole line from its neighbor is
        # usually a re-emit that DISPLACED real content —
        # surface it for the reviewer (deterministic; not auto-repaired since which cue is correct
        # needs judgment).
        for _d in adjacent_duplicate_clauses(sub):
            suspects.append(
                {
                    "kind": "adjacent-duplicate",
                    "ts": "",
                    "detail": f"cue {_d['cue']}: repeats a line from the previous cue "
                    f"({_d['dup'][:40]!r}) — likely a re-emit that dropped this cue's content",
                }
            )
        residuals = [(r.src, r.residual_miss) for r in records if r.residual_miss]
        lo = LangOutput(
            lang,
            sub,
            enforcement_issues=issues,
            residuals=residuals,
            qe_suspects=suspects,
            conversational=conversational,
            term_pins=term_pins,
        )
        if dqf is not None:
            lo.qe_metric = "DQF"
            lo.qe_system_score = dqf["score"]
            lo.qe_weak_segments = dqf["weak"]
            lo.qe_pass = dqf["pass"]
        _write_status(
            cache_path,
            phase="done",
            lang=lang,
            cues=len(sub.cues),
            **{f"dqf_{lang}": (dqf["score"] if dqf is not None else None)},
        )
        outputs.append(lo)
    _write_status(cache_path, phase="complete")
    return outputs
