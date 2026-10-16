"""Reference-free translation QE for the RUN flow — the translation analog of
`transcribe_qe` (which auto-produces suspects for a new video with no reference).

Before this, a brand-new video shipped from `run`/`localize` with ZERO automatic
translation-quality signal beyond the deterministic glossary/validate checks; the
GEMBA/DQF/consistency machinery existed but was wired into nothing. This assembles
a reference-free suspect list a reviewer can scan, from signals we already have:

  DETERMINISTIC (free, always on):
    - OMISSION: target far shorter than the source sentence expects (dropped content).
    - DROPPED-TOKEN: a source number / code identifier not present verbatim in the target.
    - TERM-INCONSISTENCY: one EN glossary term rendered two different ways across the doc.
  OPTIONAL LLM MQM JUDGE (`judge=True`):
    - a batched GEMBA-MQM pass marks per-segment error spans; counts are scored via
      DQF (0-100, calibrated, gate 96 + zero-criticals) — NOT the old raw-GEMBA 0.7
      threshold, which was meaningless (a GEMBA penalty is <=0, so <0.7 was ~always true).
"""

from __future__ import annotations

import re

from .qa import (
    dqf_gate,
    is_short_of_expected,
    missing_numbers_or_idents,
    term_consistency,
)

_LANG_NAME = {"ja": "Japanese", "zh-Hans": "Simplified Chinese"}
_JUDGE_LINE = re.compile(
    r"^\s*\[?(\d+)\]?[\s.):-]+(critical|major|minor|no-?error)\b(.*)", re.IGNORECASE
)


def _judge_batch(invoke, pairs, lang_name):
    """One batched MQM call over numbered (i, src, tgt) pairs. Returns
    {i: [(severity, desc), ...]} — segments with no error simply don't appear."""
    body = "\n".join(f"[{n}] SOURCE: {src}\n    TARGET: {tgt}" for n, src, tgt in pairs)
    prompt = (
        f"You are an expert {lang_name} subtitle-translation evaluator using MQM. For "
        f"EACH numbered segment, list translation errors, one per line, as "
        f"'<n> <severity>: <short description>' where severity is critical / major / "
        f"minor (adequacy = dropped/added/mistranslated meaning, a wrong term or number; "
        f"fluency = grammar/register/naturalness). If a segment has no error output "
        f"'<n> no-error'. Judge the TRANSLATION against its SOURCE only. Do NOT penalize "
        f"a subtitler's legitimate condensation of filler/greetings/pronouns.\n\n"
        f"{body}\n\nErrors:"
    )
    resp = invoke(prompt) or ""
    out: dict[int, list[tuple[str, str]]] = {}
    for line in resp.splitlines():
        m = _JUDGE_LINE.match(line)
        if not m:
            continue
        n, sev, rest = (
            int(m.group(1)),
            m.group(2).lower().replace("-", ""),
            m.group(3).strip(" :.-"),
        )
        if sev == "noerror":
            continue
        out.setdefault(n, []).append((sev, rest))
    return out


# No \b: in CJK output Latin is glued to CJK (USPデバッグ) and CJK counts as a word char,
# so \b fails there. The ASCII char-class already bounds a Latin run against CJK.
_LATIN_TOK = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[._:/-][A-Za-z0-9]+)*")


def _latin_typo_suspects(records, glossary):
    """A Latin acronym / code-identifier in the TARGET that is NOT in the SOURCE and is
    NOT a known glossary term is a likely TYPO or garble (e.g. USB mistyped as 'USP',
    which the MQM judge missed). Deterministic; complements the LLM judge."""
    known = set()
    for t in getattr(glossary, "terms", []):
        for f in getattr(t, "all_source_forms", []):
            known.add(f.lower())
        for r in getattr(t, "translations", {}).values():
            for m in _LATIN_TOK.findall(r or ""):
                known.add(m.lower())
    known |= {i.lower() for i in getattr(glossary, "identifiers", [])}

    def _sing(w):  # crude singular: LLMs->llm, IDEs->ide (so a plural source token
        return w[:-1] if w.endswith("s") and len(w) > 2 else w  # matches its singular target)

    out = []
    for rec in records:
        src_toks = {m.lower() for m in _LATIN_TOK.findall(rec.src)}
        src_norm = src_toks | {_sing(w) for w in src_toks}  # inflection-tolerant
        for tok in _LATIN_TOK.findall(rec.txt):
            low = tok.lower()
            # flag only acronym-like (>=2 letters, all-caps) or code-ident-like tokens
            # >=3 letters: 2-letter all-caps are mostly benign loanwords/noise (OK, TV, AI)
            is_acr = tok.isupper() and len(tok) >= 3 and tok.isalpha()
            is_ident = bool(re.search(r"[._:/-]|[a-z][A-Z]|[0-9]", tok)) and len(tok) >= 3
            # match ignoring plural/inflection BOTH ways ('LLM' target vs 'LLMs' source, or vice
            # versa) — a plural mismatch was flagging correct acronyms as typos.
            if (
                (is_acr or is_ident)
                and low not in src_norm
                and _sing(low) not in src_norm
                and low not in known
            ):
                out.append(
                    {
                        "i": rec.i,
                        "kind": "latin-typo?",
                        "source": rec.src,
                        "target": rec.txt,
                        "detail": f"target has {tok!r} but the source doesn't — possible typo/garble",
                    }
                )
    return out


def translation_suspects(
    records,
    glossary,
    lang,
    *,
    spec=None,
    invoke=None,
    judge: bool = False,
    second_judge=None,
    batch: int = 8,
    coverage_ran: bool = False,
):
    """Reference-free QE over per-sentence `records` (each with .i/.src/.txt).

    `invoke` is the primary MQM judge (Claude). `second_judge` (optional) is a
    DIFFERENT-FAMILY judge callable (e.g. CodexCliJudge().invoke, GPT-5.x) — the
    two-judge panel: our default judge is Claude grading Claude output, which
    self-preference bias inflates, so a non-Claude judge de-biases it. Per-segment
    aggregation (see `_agg`): major/critical = MAX across judges (UNION — never miss a
    real defect); minor = MIN across judges (CORROBORATION — a lone-judge minor on
    casual speech is usually subjective taste and adds stochastic noise, so a minor is
    only counted when BOTH judges see a problem in that segment). A segment is weak if
    either judge flags major/critical.

    Returns (suspects, dqf) where `suspects` is a scannable list of dicts
    {i, kind, source, target, detail} and `dqf` is None unless the LLM judge ran,
    in which case {'score','pass','counts','weak'} (weak = segment indices with a
    major/critical error)."""
    lang_name = _LANG_NAME.get(lang, lang)
    suspects = _deterministic_suspects(records, glossary, lang, coverage_ran=coverage_ran)

    dqf = None
    if judge and invoke and records:
        judges = build_judges(invoke, second_judge)
        pairs = [(r.i, r.src, r.txt) for r in records]
        per_seg, judge_suspects = _run_panel(pairs, judges, lang_name, batch)
        suspects += judge_suspects
        wc = sum(len(r.src.split()) for r in records) or 1
        counts, weak, passes, score = _score_panel(per_seg, wc)
        dqf = {
            "score": round(score, 1),
            "pass": passes,
            "counts": counts,
            "weak": sorted(set(weak)),
            "judges": [n for n, _ in judges],
            "per_seg": per_seg,
        }

    suspects.sort(key=lambda s: (s["i"] is None, s["i"] if s["i"] is not None else 0))
    return suspects, dqf


def _deterministic_suspects(records, glossary, lang, *, coverage_ran: bool = False):
    """Reference-free deterministic suspects (no LLM): omission (length), dropped
    number/identifier, target-side Latin typo/garble, and doc-level term inconsistency.

    When `coverage_ran` (the source-anchored LLM coverage gate ran in this build, so a real
    omission is already surfaced as a coverage residual + caught by the MQM judge), the
    length-ratio omission flag is SUPPRESSED: on conversational English it fires on complete
    but concise CJK (function-word-dense EN → low chars/word even when nothing is dropped),
    producing ~20 false positives per interview that only add noise to the review sheet."""
    suspects: list[dict] = []
    # Whole delivered target — a token/term absent from THIS record but present elsewhere just
    # moved to an adjacent cue during re-segmentation (not a real drop); don't flag those.
    full_txt = " ".join(r.txt for r in records)
    for r in records:
        if not coverage_ran and is_short_of_expected(r.src, r.txt, lang):
            suspects.append(
                {
                    "i": r.i,
                    "kind": "omission",
                    "source": r.src,
                    "target": r.txt,
                    "detail": "target far shorter than the source expects — likely dropped content",
                }
            )
        miss = [m for m in missing_numbers_or_idents(r.src, r.txt) if m not in full_txt]
        if miss:
            suspects.append(
                {
                    "i": r.i,
                    "kind": "dropped-token",
                    "source": r.src,
                    "target": r.txt,
                    "detail": f"source number/identifier not in target: {', '.join(miss)}",
                }
            )
    suspects += _latin_typo_suspects(records, glossary)
    srcs = [r.src for r in records]
    tgts = [r.txt for r in records]
    for issue in term_consistency(srcs, tgts, glossary, lang):
        suspects.append(
            {
                "i": None,
                "kind": "term-inconsistency",
                "source": issue["term"],
                "target": issue["expected"],
                "detail": f"{issue['inconsistent_cues']} cue(s) don't use the "
                f"canonical rendering {issue['expected']!r}",
                "fix_idx": issue.get("inconsistent_idx", []),
            }
        )
    return suspects


def build_judges(invoke, second_judge=None):
    """The MQM panel: Claude always, plus a different-family judge (Codex/GPT-5.x) when
    given — see the self-preference-bias note on `translation_suspects`."""
    return [("claude", invoke)] + ([("codex", second_judge)] if second_judge else [])


def _run_panel(pairs, judges, lang_name, batch=8):
    """Run the MQM judge panel over (i, src, tgt) pairs. Returns (per_seg, suspects):
    per_seg[i] = {judge_name: {critical,major,minor}} with EVERY panel judge zero-filled
    for each flagged segment (so the MIN-across-judges minor corroboration in `_agg` sees
    a 0 for a judge that raised nothing); suspects = per-finding dicts (judge-tagged when
    the panel has >1). Segments no judge flagged simply don't appear (zero penalty)."""
    names = [n for n, _ in judges]
    per_seg: dict[int, dict[str, dict]] = {}
    suspects: list[dict] = []
    multi = len(judges) > 1
    for name, jinvoke in judges:
        for k in range(0, len(pairs), batch):
            try:
                found = _judge_batch(jinvoke, pairs[k : k + batch], lang_name)
            except Exception:
                found = {}  # one judge failing must not sink the panel
            for i, errs in found.items():
                jseg = per_seg.setdefault(i, {}).setdefault(
                    name, {"critical": 0, "major": 0, "minor": 0}
                )
                for sev, desc in errs:
                    jseg[sev] += 1
                    tag = f"mqm-{sev}" + (f"[{name}]" if multi else "")
                    suspects.append(
                        {
                            "i": i,
                            "kind": tag,
                            "source": "",
                            "target": "",
                            "detail": desc or f"{sev} error",
                        }
                    )
    for seg in per_seg.values():  # zero-fill judges that flagged nothing here
        for name in names:
            seg.setdefault(name, {"critical": 0, "major": 0, "minor": 0})
    return per_seg, suspects


_DQF_W = {"critical": 40.0, "major": 2.5, "minor": 0.2}


def _agg(seg: dict) -> dict:
    """Aggregate one segment's per-judge counts into {critical,major,minor}. major/critical
    = MAX across judges (union: a defect either judge sees counts); minor = MIN across
    judges (corroboration: only counted when every judge saw a problem — de-noises the
    subjective-minor mass the two-judge union used to over-count on casual speech). Tolerant
    of a FLAT {sev:int} seg (single-judge / test shape) — returns it unchanged."""
    if not seg:
        return {"critical": 0, "major": 0, "minor": 0}
    if all(isinstance(v, int) for v in seg.values()):  # flat shape
        return {s: seg.get(s, 0) for s in _DQF_W}
    vals = list(seg.values())
    return {
        "critical": max(v["critical"] for v in vals),
        "major": max(v["major"] for v in vals),
        "minor": min(v["minor"] for v in vals),
    }


def _score_panel(per_seg, wc):
    """(counts, weak, passes, score) from a per_seg map — weak = major/critical segments."""
    counts = {"critical": 0, "major": 0, "minor": 0}
    weak = []
    for i, seg in per_seg.items():
        a = _agg(seg)
        for sev in counts:
            counts[sev] += a[sev]
        if a["critical"] or a["major"]:
            weak.append(i)
    passes, score = dqf_gate(counts, wc)
    return counts, weak, passes, score


def _seg_penalty(seg: dict) -> float:
    a = _agg(seg)
    return sum(_DQF_W[s] * a[s] for s in _DQF_W)


# A rejected repair is "borderline" (worth re-sampling to de-noise) when NEW is within this
# many penalty points of OLD — ~one major (2.5). A larger gap is a decisive rejection: no
# re-sampling needed. This bounds the codex cost of cmp_samples to genuinely-close calls.
_BORDERLINE = 2.5


def qe_repair(
    records,
    lang,
    *,
    judges,
    lang_name,
    retranslate,
    per_seg=None,
    judge_suspects=None,
    compare_judges=None,
    include_minor=False,
    logger=None,
    batch=8,
    max_repairs=12,
    extra_flags=None,
    extra_critiques=None,
    cmp_samples=1,
):
    """QE-guided auto-repair (Layer B). For each segment a judge flagged major/critical
    (plus any deterministic `extra_flags` indices — dropped-token / latin-typo garbles),
    re-translate it with the judges' own critique as a corrective constraint, then
    re-judge OLD vs NEW in ONE panel call and ACCEPT only if the new weighted MQM penalty
    STRICTLY drops. This accept-only-if-better gate is what keeps a bad auto-edit (or a
    judge false-positive) from ever shipping — a rejected repair leaves the original.

    `judges` selects/targets segments (use the full two-judge panel — its de-biasing
    matters for WHICH segments to fix). `compare_judges` (default = `judges`) runs the
    per-candidate OLD-vs-NEW decision; pass a lighter single-judge panel here to keep the
    inner loop fast — the accept/reject signal doesn't need the slow second judge, only the
    authoritative reported DQF does.

    `retranslate(rec, critique) -> str` is supplied by the caller (it re-runs the full
    translate+enforce+completeness gates). Mutates `records[i].txt` in place on accept;
    returns the list of repaired indices."""
    compare_judges = compare_judges or judges
    log = logger or (lambda _m: None)
    if not judges or not records:
        return []
    by_i = {r.i: r for r in records}
    if per_seg is None:  # reuse the caller's first-pass panel if given
        per_seg, judge_suspects = _run_panel(
            [(r.i, r.src, r.txt) for r in records], judges, lang_name, batch
        )
    judge_suspects = judge_suspects or []
    crit: dict[int, list[str]] = {}
    for s in judge_suspects:
        if s["kind"].startswith("mqm-"):
            crit.setdefault(s["i"], []).append(s["detail"])
    # Always repair major/critical. With include_minor, also repair CORROBORATED minors
    # (agg minor >= 1 means BOTH judges flagged that segment — a clear agreed slip like
    # today->今すぐ; the accept-only-if-better guardrail still blocks a non-improvement).
    extra_critiques = extra_critiques or {}
    targets = {
        i
        for i, seg in per_seg.items()
        if (_a := _agg(seg))["critical"] or _a["major"] or (include_minor and _a["minor"])
    }
    targets |= set(extra_flags or [])
    targets |= set(extra_critiques)  # term-consistency cues (canonical-rendering fix)
    order = sorted(targets, key=lambda i: -_seg_penalty(per_seg.get(i, {})))[:max_repairs]
    repaired = []
    for i in order:
        rec = by_i.get(i)
        if rec is None:
            continue
        # RESILIENCE: a retranslate or re-judge LLM call can fail mid-loop (expired creds,
        # timeout). Don't crash + lose every accepted repair so far — skip this candidate and
        # continue, so qe_repair returns normally and the caller can persist what landed.
        try:
            parts = ([extra_critiques[i]] if i in extra_critiques else []) + crit.get(i, [])
            critique = (
                "; ".join(dict.fromkeys(parts))
                or "fix any mistranslation, dropped meaning, wrong term, or unnatural phrasing"
            )
            new_txt = (retranslate(rec, critique) or "").strip()
            if not new_txt or new_txt == rec.txt:
                log(f"{lang}: QE-repair #{i} skip (no change)")
                continue
            # (1) TARGETED accept: the critique quotes a WRONG rendering (present in old) and a
            # CORRECT one; if the retranslate removed the wrong term AND added the right one, the
            # SPECIFIC flagged error is resolved — accept even if the holistic panel score TIES
            # (the judge may flag the segment for unrelated fragment reasons, which the penalty
            # gate can't distinguish — that's why cue-93 conflicts->競合 stuck at 2.5->2.5).
            if _targeted_fix_ok(critique, rec.txt, new_txt):
                log(
                    f"{lang}: QE-repair #{i} ACCEPT (targeted: quoted wrong->right term): {critique[:60]}"
                )
                rec.txt = new_txt
                rec.residual_miss = ""  # fresh retranslate — any prior miss/failed-flag is obsolete
                repaired.append(i)
                continue
            # (1b) OMISSION-TARGETED accept: the OLD text was short-of-expected (a dropped
            # clause/sentence) and the retranslate is no longer short — the omission is resolved.
            # Accept even on a holistic penalty tie: a completed long sentence often scores ~equal
            # overall yet fixes a critical drop, so the strict-drop gate would wrongly reject it
            # (e.g. a cue with nearly the whole sentence dropped kept getting rejected 'no gain').
            if is_short_of_expected(rec.src, rec.txt, lang) and not is_short_of_expected(
                rec.src, new_txt, lang
            ):
                log(
                    f"{lang}: QE-repair #{i} ACCEPT (omission resolved: short-of-expected -> complete)"
                )
                rec.txt = new_txt
                rec.residual_miss = ""
                repaired.append(i)
                continue

            # (2) ADAPTIVE penalty comparison: one re-judge draw; RE-SAMPLE to de-noise ONLY a
            # BORDERLINE rejection — new within `_BORDERLINE` penalty of old, i.e. plausibly a
            # stochastic mis-score. A CLEAR rejection (new decisively worse) or an accept resolves
            # in ONE draw, so the extra codex samples are spent only where the call is close.
            # (Previously EVERY reject re-sampled cmp_samples×, which made a full re-run ~2h.)
            def _pen():
                cp, _ = _run_panel(
                    [(0, rec.src, rec.txt), (1, rec.src, new_txt)], compare_judges, lang_name, batch
                )
                return _seg_penalty(cp.get(0, {})), _seg_penalty(cp.get(1, {}))

            old_p, new_p = _pen()
            if cmp_samples > 1 and 0 <= new_p - old_p <= _BORDERLINE:  # borderline reject only
                olds, news = [old_p], [new_p]
                for _s in range(cmp_samples - 1):
                    oo, nn = _pen()
                    olds.append(oo)
                    news.append(nn)
                old_p, new_p = sum(olds) / len(olds), sum(news) / len(news)
        except Exception as e:
            log(f"{lang}: QE-repair #{i} skipped ({type(e).__name__}) — keeping original")
            continue
        if new_p < old_p:
            log(f"{lang}: QE-repair #{i} ACCEPT ({old_p:.1f}->{new_p:.1f}): {critique[:70]}")
            rec.txt = new_txt
            rec.residual_miss = ""  # fresh retranslate — any prior miss/failed-flag is obsolete
            repaired.append(i)
        else:
            log(f"{lang}: QE-repair #{i} reject ({old_p:.1f}->{new_p:.1f}, no gain)")
    return repaired


# Han runs OR katakana runs (>=2). Hiragana particles (の/を/し/は…) act as separators, so a
# quoted term like 予定の重複 splits into 予定/重複 (the meaningful cores) rather than gluing into
# 予定の重複の意味を外している — which would never substring-match the target.
_CJK_TERM = __import__("re").compile(r"[一-鿿]{2,}|[ァ-ヴ][ァ-ヴー]+")


def _targeted_fix_ok(critique: str, old: str, new: str) -> bool:
    """True when the retranslate performed the SPECIFIC wrong->right term substitution the
    critique named: a CJK term quoted in the critique that was in OLD is GONE in NEW (the wrong
    rendering removed), AND a CJK term quoted in the critique that was NOT in OLD is now in NEW
    (the correct rendering added). Confirms the exact flagged defect is fixed without relying on
    the (tie-prone) holistic panel score. Low false-accept risk — requires both directions."""
    quoted = set(_CJK_TERM.findall(critique))
    wrong = {t for t in quoted if t in old and t not in new}  # named, was in old, now gone
    right = {t for t in quoted if t not in old and t in new}  # named, not in old, now present
    return bool(wrong and right)


def classify_term_fix(critique: str, src: str, tgt: str, lang_name: str, invoke) -> dict | None:
    """Decide whether ONE judge critique is a mechanical TERM-rendering fix (a specific term
    rendered with the wrong target wording, a wrong technical term, DNT left in English, or an
    inconsistent term — with a CLEAR correct rendering) vs a JUDGMENT call (rephrase, fluency,
    added/dropped nuance, idiom, or a synonym choice where several are acceptable). Returns
    {"source","wrong","correct"} for a term fix, else None. Conservative by design: only a
    term fix when the correct rendering is unambiguous — judgment calls are left for the human
    reviewer (this is the line the whole auto-fix design defends)."""
    import json as _json

    prompt = (
        "You are auditing ONE subtitle QE critique to decide if it is a MECHANICAL term-"
        "rendering fix or a JUDGMENT call.\n"
        f"Critique: {critique}\nEnglish source: {src}\nCurrent {lang_name} translation: {tgt}\n\n"
        "If the critique identifies a specific TERM/phrase rendered with the WRONG target "
        "wording AND there is ONE clear correct target rendering (a terminology error, wrong "
        "technical term, a do-not-translate name left in English, or an inconsistent term), "
        'output JSON: {"kind":"term","source":"<English term>","wrong":"<wrong target '
        'wording, must appear verbatim in the current translation>","correct":"<correct '
        'target rendering>"}.\n'
        "Otherwise — a rephrase, awkward fluency, added/dropped nuance, an idiom, or a synonym "
        'choice where several renderings are acceptable — output {"kind":"judgment"}.\n'
        "POLYSEME CAUTION: a word can have different correct renderings by DOMAIN — e.g. 'agent' = "
        "AI agent 智能体 in a tech/AI talk, but travel/service agent 代理/客服 for an airline; "
        "'controller', 'lane', 'stick' similarly. Choose the sense that fits THIS English sentence. "
        "If the correct sense is NOT clear from the sentence, output judgment — do NOT guess.\n"
        'Be conservative: only "term" when the correct rendering is unambiguous. Output ONLY JSON.'
    )
    try:
        raw = invoke(prompt) or ""
        s, e = raw.find("{"), raw.rfind("}")
        d = _json.loads(raw[s : e + 1]) if s != -1 and e != -1 else {}
    except Exception:
        return None
    if d.get("kind") != "term":
        return None
    source, wrong, correct = (str(d.get(k, "")).strip() for k in ("source", "wrong", "correct"))
    # Guard: wrong must be PRESENT in the current target (so enforce can remove it), and the
    # correct rendering must differ from the wrong one. Reject anything that doesn't hold.
    if not (source and wrong and correct) or wrong == correct or wrong not in tgt:
        return None
    return {"source": source, "wrong": wrong, "correct": correct}


def make_override_adjudicator(invoke, lang_name: str = ""):
    """Verifier for an uncorroborated single-judge CJK->CJK term flip: does the proposed
    correction actually fit the SOURCE meaning? Counting judges can't tell a right flip
    (promise->前景) from a wrong-sense one (agent->特工/spy); this asks an LLM to check the
    specific substitution against the source — the step a human reviewer performs. Prefer a
    DIFFERENT-family model than the judge that raised the flag, for independence. Returns True
    only on an explicit VERIFIED verdict; every other outcome (REJECT, ambiguity, error) returns
    False so the pin stays held for a human — this can only ADD verified fixes, never weaken the
    hold."""

    def adjudicate(src: str, tgt: str, wrong: str, correct: str, critique: str = "") -> bool:
        if not invoke or not (wrong and correct and src):
            return False
        prompt = (
            f"Verify ONE proposed {lang_name} subtitle-translation correction.\n"
            f"SOURCE (English): {src!r}\n"
            f"A reviewer claims the current rendering uses {wrong!r} but it should be {correct!r}."
            + (f" Reviewer note: {critique}" if critique else "")
            + "\n\nFrom the SOURCE MEANING ALONE, is "
            + f"{correct!r} CLEARLY a more accurate rendering than {wrong!r} in this context? "
            "Reject false 'corrections' that pick a WRONG SENSE of a polysemous word (e.g. "
            "software 'agent' -> 特工/spy, or 'promise' -> a pledge when it means potential). "
            "Answer with exactly one word: VERIFIED (clearly correct) or REJECT (wrong or unsure)."
        )
        try:
            r = (invoke(prompt) or "").strip().upper()
        except Exception:
            return False
        return "VERIFIED" in r and "REJECT" not in r

    return adjudicate


# Ultra-common source words a term-pin must never be harvested for: they appear in most cues, so a
# source-gated pin fires video-wide, and their CJK renderings are legitimate words elsewhere ->
# mass mis-substitution. (Pronouns/articles/copulas; the 'you' 我们->你 crater.)
_PIN_STOP_SOURCES = {
    "you",
    "we",
    "i",
    "it",
    "he",
    "she",
    "they",
    "them",
    "us",
    "me",
    "him",
    "her",
    "this",
    "that",
    "these",
    "those",
    "the",
    "a",
    "an",
    "is",
    "are",
    "be",
    "to",
    "of",
    "and",
    "or",
    "your",
    "our",
    "their",
    "my",
    "its",
}
# Bare common CJK pronouns: a pin whose avoid- OR correct-form is exactly one of these flips a
# high-frequency pronoun across the video.
_PIN_STOP_CJK = {
    "我们",
    "你",
    "你们",
    "他们",
    "她们",
    "它们",
    "我",
    "他",
    "她",
    "它",
    "咱们",
    "您",
    "您们",
}


def harvest_term_pins(
    records,
    lang,
    judge_suspects,
    *,
    classify,
    logger=None,
    max_pins=15,
    skip_cues=(),
    keep_latin=(),
    adjudicate=None,
    curated=None,
):
    """Extract mechanical TERM-substitution pins from the judge critiques (the class that
    `qe_repair`'s retranslate-accept-if-better misses — a wrong term rendering often scores
    ~equal so the retranslate is rejected and it falls through to 'human judgment'). Returns
    a deduped list of pins [{"source","wrong","correct","cues":[i,...]}] AND the list of
    resolved suspects (the critiques classified as term fixes — the caller drops them from the
    review so they no longer read as 'human judgment'). `classify(critique, src, tgt) -> fix|
    None` is the LLM term/judgment classifier. Cues in `skip_cues` (already repaired by
    qe_repair) are skipped. Pure w.r.t. records — application is separate."""
    log = logger or (lambda _m: None)
    import re as _re

    def _nrm(x: str) -> str:
        return _re.sub(r"\s+", "", x or "").lower()

    kl = {t.strip().lower() for t in (keep_latin or ()) if t and t.strip()}
    by_i = {r.i: r for r in records}
    skip = set(skip_cues)
    # distinct judges per cue (from kinds like "mqm-major[claude]" / "mqm-minor[codex]") — used
    # to CORROBORATE overrides: a single judge's whim should not flip an already-plausible rendering.
    judges_by_cue: dict = {}
    for s in judge_suspects or []:
        m = _re.search(r"\[([^\]]+)\]", str(s.get("kind", "")))
        if m and s.get("i") is not None:
            judges_by_cue.setdefault(s["i"], set()).add(m.group(1))
    pins: dict[tuple, dict] = {}  # (source.lower, correct) -> pin
    for s in judge_suspects or []:
        if not str(s.get("kind", "")).startswith("mqm-"):
            continue
        i = s.get("i")
        rec = by_i.get(i)
        if rec is None or i in skip:
            continue
        try:
            fix = classify(s["detail"], rec.src, rec.txt)
        except Exception:
            fix = None
        if not fix:
            continue
        # COMMON-WORD GUARD: never harvest a pin whose source is an ultra-common function word/
        # pronoun, or whose avoid-form is a bare common CJK pronoun. Such a pin is source-gated on
        # a term that appears in most cues (so it fires video-wide) and its avoid-form (我们/你…) is
        # a legitimate word elsewhere — mass-substituting it craters quality (the 'you' 我们->你
        # disaster: ~260 cues flipped we->you, DQF 96.5->46.9). These are per-cue judgment calls,
        # not mechanical term fixes — leave them for the human.
        _src_l = fix["source"].strip().lower()
        if (
            _src_l in _PIN_STOP_SOURCES
            or _nrm(fix.get("wrong", "")) in _PIN_STOP_CJK
            or _nrm(fix.get("correct", "")) in _PIN_STOP_CJK
        ):
            log(f"{lang}: term-pin REJECTED (common pronoun/function word) {fix['source']!r}")
            continue
        # KEEP-LATIN GUARD: never harvest a pin that TRANSLATES a protected keep-Latin/DNT term
        # (name/identifier). A judge sometimes flags the correct Latin form as 'wrong' and proposes
        # a CJK rendering (Nickel->镍币); persisting that inverts the DNT contract and the pin then
        # ping-pongs run-to-run. The only valid rendering of a keep-Latin term is the term itself,
        # so a pin whose `correct` isn't the Latin source is rejected (the restorative CJK->Latin
        # direction, correct==source, is kept).
        if (
            kl
            and fix["source"].strip().lower() in kl
            and _nrm(fix["correct"]) != _nrm(fix["source"])
        ):
            log(
                f"{lang}: term-pin REJECTED (keep-Latin/DNT) {fix['source']!r} must stay Latin, "
                f"not {fix['correct']!r}"
            )
            continue
        # CURATED-GLOSSARY GUARD: `curated` {source_lower: canonical} for this lang. A judge may
        # prefer another rendering of an approved glossary term (ja Orbit仮想デバイス -> Latin); as a
        # durable pin that flips EVERY occurrence and overrides the glossary (every cue in the
        # video). The glossary owns these — leave the critique for the human.
        _cur = {k.lower(): v for k, v in (curated or {}).items() if v}
        if _cur and (
            (_src_l in _cur and _nrm(_cur[_src_l]) != _nrm(fix["correct"]))
            or _nrm(fix.get("wrong", "")) in {_nrm(v) for v in _cur.values()}
        ):
            log(f"{lang}: term-pin REJECTED (overrides curated glossary) {fix['source']!r}")
            continue
        # COMMON-PHRASE LATIN GUARD: restoring plain lowercase English (home screen) as Latin is a
        # per-context call (a folder name in a path), but a pin applies it video-wide — it turned
        # 'From the home screen' into 在home screen上. Rejected when the phrase also occurs in OTHER
        # sentences (where it is ordinary English); a one-off (a spoken CLI command) still pins.
        if (
            _nrm(fix["correct"]) == _nrm(fix["source"])
            and _re.fullmatch(r"[a-z]+(?: [a-z]+)*", fix["source"].strip())
            and _src_l not in kl
            and sum(_src_l in (r.src or "").lower() for r in records) > 1
        ):
            log(f"{lang}: term-pin REJECTED (common English phrase kept Latin) {fix['source']!r}")
            continue
        if len(pins) >= max_pins and (fix["source"].lower(), fix["correct"]) not in pins:
            continue
        key = (fix["source"].lower(), fix["correct"])
        pin = pins.setdefault(key, {**fix, "cues": [], "_suspects": [], "_corrob": False})
        if i not in pin["cues"]:
            pin["cues"].append(i)
        pin["_suspects"].append(s)
        # Corroboration for an OVERRIDE (CJK 'wrong' → the pin flips an already-CJK rendering, the
        # AI-agent 智能体->特工 (spy) class): needs >=2 distinct judges on the cue. A Latin 'wrong' (raw
        # untranslated English in a CJK target, e.g. multi-eight-figure) is an unambiguous error →
        # always allowed. NOTE: repetition across cues is NOT corroboration (the same wrong pin
        # applied widely, like agent->特工 ×7, is still one decision) — only distinct judges count.
        # corroborated if: raw-English fix, OR >=2 judges on the cue, OR no judge metadata to
        # assess (don't over-hold when tags are absent — only block a POSITIVELY single-judge flip).
        if (
            _re.search(r"[A-Za-z]", fix["wrong"])
            or len(judges_by_cue.get(i, ())) >= 2
            or i not in judges_by_cue
        ):
            pin["_corrob"] = True
    out, resolved = [], []
    for p in pins.values():
        is_override = not _re.search(r"[A-Za-z]", p["wrong"])  # CJK->CJK flip
        if is_override and not p["_corrob"]:
            # Uncorroborated single-judge CJK->CJK flip. Instead of always deferring to a human,
            # ADJUDICATE it: a focused verification (ideally a different-family model) that checks
            # whether the proposed correction actually fits the SOURCE meaning — what a human
            # reviewer does, and more reliable than merely COUNTING judges (two judges can agree on
            # a wrong flip). A verified-correct flip (promise->前景) is applied; a wrong-sense one
            # (agent->特工/spy) is REJECTED and stays held. Conservative: no adjudicator, or any
            # non-VERIFIED verdict/error, keeps the old hold — so this only ADDS verified fixes and
            # never removes the human safety net (no new gap).
            verified = False
            if adjudicate is not None:
                _rec = by_i.get(p["cues"][0]) if p.get("cues") else None
                _crit = p["_suspects"][0].get("detail", "") if p.get("_suspects") else ""
                try:
                    verified = bool(
                        adjudicate(
                            _rec.src if _rec else "",
                            _rec.txt if _rec else "",
                            p["wrong"],
                            p["correct"],
                            _crit,
                        )
                    )
                except Exception:
                    verified = False
            if not verified:
                log(
                    f"{lang}: term-pin HELD (uncorroborated override) {p['source']!r} "
                    f"{p['wrong']!r}->{p['correct']!r} (cues {p['cues']}) — left for human review"
                )
                continue  # not resolved → stays in human-review tier
            log(
                f"{lang}: term-pin ADJUDICATED single-judge override {p['source']!r} "
                f"{p['wrong']!r}->{p['correct']!r} (cues {p['cues']}) — verified correct, applying"
            )
        resolved.extend(p.pop("_suspects"))
        p.pop("_corrob", None)
        out.append(p)
    for p in out:
        log(
            f"{lang}: term-pin proposed {p['source']!r} {p['wrong']!r}->{p['correct']!r} "
            f"(cues {p['cues']})"
        )
    return out, resolved


def pin_substr_fix(text: str, pins, lang: str) -> str:
    """Substring re-application of harvested pins that the word-boundary glossary enforce
    MISSES: a hyphenated avoid-form ('multi-eight-figure') or a raw untranslated multiword
    source term that reflow re-emitted. Replaces the wrong rendering -> correct, and the raw
    source term -> correct but ONLY when the source is DISTINCTIVE (contains a hyphen or
    space) so a common word that is a substring of another term (e.g. 'Digital' inside
    'Acme Digital') is never clobbered."""
    from .glossary import _replace_form

    for p in pins or []:
        correct = p.get("correct") or ""
        wrong = p.get("wrong") or ""
        if not correct:
            continue
        if wrong and wrong != correct and wrong in text:
            # _replace_form: word-bounded for ASCII, and CJK substring WITH seam-dedup so a short
            # avoid-form that is a prefix of a longer compound can't duplicate the canonical's tail
            # char (网页端 + 网页->Web端 => Web端端 -> Web端).
            text = _replace_form(text, wrong, correct)
        src = p.get("source") or ""
        if src and src != correct and ("-" in src or " " in src) and src in text:
            text = _replace_form(text, src, correct)
    return text


def apply_term_pins(records, pins, glossary, lang, protect=()):
    """Apply term pins DETERMINISTICALLY across ALL records via glossary enforce (source-gated:
    a pin fires only in cues whose English source contains the pin's source term, and converts
    the wrong rendering -> correct wherever the term is present). Fixes every occurrence + any
    inconsistency in one pass, no re-translation. Returns the set of changed record indices.
    A substring pass (pin_substr_fix) follows enforce so hyphenated / raw multiword terms the
    word-boundary enforce misses still get fixed.

    `protect` = keep-Latin identifiers/terms to MASK before applying pins, so a single-word pin
    (e.g. 'video'->视频) can't corrupt a compound identifier that CONTAINS that word (the
    'react-native-video'->'react-native-视频' bug: enforce's word boundary matched 'video' inside
    the hyphenated package name). Longest-first masking protects the whole identifier."""
    import copy

    from .glossary import Term

    if not pins:
        return set()
    g = copy.copy(glossary)  # shallow: share everything but .terms
    g.terms = list(glossary.terms) + [
        Term(
            source=p["source"],
            translations={lang: p["correct"]},
            avoid={lang: [p["wrong"]]},
            enforce="soft",
        )
        for p in pins
    ]
    prot = sorted({t for t in (protect or ()) if t and len(t) >= 3}, key=len, reverse=True)
    changed = set()
    for r in records:
        # Mask protected identifiers (longest first) so no pin fires INSIDE one, then restore.
        masked, mask = r.txt, {}
        for k, t in enumerate(prot):
            if t in masked:
                tok = f"\x00KP{k}\x00"
                masked = masked.replace(t, tok)
                mask[tok] = t
        new_txt, _ = g.enforce(masked, r.src, lang)
        new_txt = pin_substr_fix(new_txt, pins, lang)
        for tok, t in mask.items():
            new_txt = new_txt.replace(tok, t)
        if new_txt != r.txt:
            r.txt = new_txt
            changed.add(r.i)
    return changed


def pins_to_renderings(pins, lang) -> dict:
    """Term pins -> a per-video `renderings` dict {lang: {source: {canonical, avoid}}} for
    persistence into vocab.json (so a re-run reapplies them) and for promotion surfacing."""
    return (
        {lang: {p["source"]: {"canonical": p["correct"], "avoid": [p["wrong"]]} for p in pins}}
        if pins
        else {}
    )


def rescore_after_repair(
    records,
    glossary,
    lang,
    *,
    judges,
    lang_name,
    prior_per_seg,
    prior_mqm_suspects,
    changed,
    batch=8,
    coverage_ran: bool = False,
):
    """Cheap, low-noise post-repair re-score. Re-judge ONLY the `changed` segments with the
    full panel and PATCH their entries into the prior panel — rather than re-drawing the
    whole panel, which (being stochastic ±3-4 DQF pts) swamps the signal from a handful of
    fixes and burns a full second judge (Codex) panel per language. Deterministic suspects
    are recomputed over all current text (cheap, no LLM). Returns (suspects, dqf)."""
    changed = set(changed)
    det = _deterministic_suspects(records, glossary, lang, coverage_ran=coverage_ran)
    new_per, new_mqm = _run_panel(
        [(r.i, r.src, r.txt) for r in records if r.i in changed], judges, lang_name, batch
    )
    per_seg = dict(prior_per_seg or {})
    for i in changed:
        per_seg[i] = new_per.get(i, {})  # empty = no error found now (zero penalty)
    mqm = [s for s in (prior_mqm_suspects or []) if s["i"] not in changed] + new_mqm
    wc = sum(len(r.src.split()) for r in records) or 1
    counts, weak, passes, score = _score_panel(per_seg, wc)
    suspects = det + mqm
    suspects.sort(key=lambda s: (s["i"] is None, s["i"] if s["i"] is not None else 0))
    dqf = {
        "score": round(score, 1),
        "pass": passes,
        "counts": counts,
        "weak": sorted(set(weak)),
        "judges": [n for n, _ in judges],
        "per_seg": per_seg,
    }
    return suspects, dqf


# --- Incremental re-review: refresh the full REVIEW (DQF + MQM tiers) from cache without
# re-judging everything. Persist the QE panel keyed by a hash of each translated segment; a
# later --qe-judge run re-judges ONLY changed segments (reusing rescore_after_repair) so a
# re-review after a cache mutation (QE-repair recovery, term-pins, manual fix) is cheap + correct.


def _txt_hash(s: str) -> str:
    import hashlib

    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()[:16]


def save_qe_state(path, records, dqf, mqm_suspects) -> None:
    """Persist the per-segment QE panel (penalty entry + text hash) + the MQM suspects, so a
    later run can re-judge only what changed. Keyed by SOURCE-text hash (not segment index) so a
    mid-video EN edit that renumbers every later sentence still reuses their unchanged panels
    instead of re-judging the whole tail — mirroring the translation cache's by-source reuse.
    Deterministic suspects are recomputed each time (cheap) and NOT stored."""
    import json as _json
    from pathlib import Path as _P

    per = (dqf or {}).get("per_seg", {})
    rec_by_i = {r.i: r for r in records}
    state = {
        # SOURCE-keyed panel: {src_hash: {h: txt_hash, per_seg: panel_entry}}, in record order.
        "segs": [
            {"src": _txt_hash(r.src), "h": _txt_hash(r.txt), "per_seg": per.get(r.i, {})}
            for r in records
        ],
        # Tag each MQM suspect with its source hash so it can be remapped to the current index.
        "mqm": [
            (
                {**s, "src_hash": _txt_hash(rec_by_i[s["i"]].src)}
                if s.get("i") is not None and s["i"] in rec_by_i
                else s
            )
            for s in (mqm_suspects or [])
            if str(s.get("kind", "")).startswith("mqm-")
        ],
    }
    _P(path).write_text(_json.dumps(state, ensure_ascii=False), encoding="utf-8")


def load_qe_state(path):
    import json as _json
    from pathlib import Path as _P

    p = _P(path)
    if not p.exists():
        return None
    try:
        return _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def incremental_review(
    records, glossary, lang, *, judges, lang_name, prior_state, batch=8, coverage_ran=False
):
    """Re-review from a prior QE state: re-judge ONLY segments whose (source, translation) pair
    is new or changed since the last review (reuse the stored panel for the rest), then recompute
    DQF + suspects. Matching is by SOURCE-text hash, not segment index, so a mid-video EN edit
    that renumbers every later sentence still reuses their unchanged panels instead of re-judging
    the whole renumbered tail — mirroring the translation cache's by-source reuse. changed=[] →
    nothing re-judged, the full prior REVIEW is reproduced instantly. Returns (suspects, dqf) in
    the same shape as translation_suspects/rescore_after_repair, plus the changed indices."""
    segs = prior_state.get("segs")
    if segs is None:
        # LEGACY index-keyed state (pre-source-keying) — fall back to the old index compare.
        by = prior_state.get("by_i", {})
        prior_per_seg = {int(i): e.get("per_seg", {}) for i, e in by.items()}
        prior_mqm = prior_state.get("mqm", [])
        changed = [r.i for r in records if by.get(str(r.i), {}).get("h") != _txt_hash(r.txt)]
    else:
        prior_by_src: dict[str, dict] = {}
        for e in segs:
            prior_by_src.setdefault(e["src"], e)  # first wins (stable for duplicate source lines)
        prior_per_seg = {}
        changed = []
        for r in records:
            p = prior_by_src.get(_txt_hash(r.src))
            if p is None or p.get("h") != _txt_hash(r.txt):
                changed.append(r.i)  # new/edited source, or re-translated text — must re-judge
            else:
                prior_per_seg[r.i] = p.get("per_seg", {})  # reuse prior panel at its CURRENT index
        # Remap prior MQM suspects (stored by src_hash) to their current index; drop any whose
        # source no longer appears (that sentence was deleted).
        cur_i_by_src = {_txt_hash(r.src): r.i for r in records}
        prior_mqm = []
        for m in prior_state.get("mqm", []):
            ci = cur_i_by_src.get(m.get("src_hash"))
            if ci is not None:
                prior_mqm.append({**m, "i": ci})
    return (
        rescore_after_repair(
            records,
            glossary,
            lang,
            judges=judges,
            lang_name=lang_name,
            prior_per_seg=prior_per_seg,
            prior_mqm_suspects=prior_mqm,
            changed=changed,
            batch=batch,
            coverage_ran=coverage_ran,
        ),
        changed,
    )
