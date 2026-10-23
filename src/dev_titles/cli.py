"""dev_titles CLI — thin entry over the deterministic steps.

Implemented now: validate, resegment (line-break/kinsoku/zh-punct on an existing
translated VTT/SRT), convert (vtt<->srt<->ttml). transcribe/translate/qe/publish
subcommands are wired but call stubs pending further research.

  python -m dev_titles.cli validate out.ja.vtt --lang ja
  python -m dev_titles.cli resegment in.ja.vtt --lang ja -o out.ja.vtt
  python -m dev_titles.cli convert in.vtt -o out.srt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .config import spec_for
from .cue_builder import build_en_cues

if TYPE_CHECKING:
    from .model import Transcript

from .formats import read_srt, read_vtt, write_srt, write_ttml, write_vtt
from .formats.sbv import read_sbv
from .segment import break_lines, normalize_zh_punct
from .transcribe import from_transcribe_json
from .validate import validate_subtitle


def _read_any(path: str, lang: str):
    low = path.lower()
    if low.endswith(".srt"):
        return read_srt(path, lang)
    if low.endswith(".sbv"):
        return read_sbv(path, lang)
    return read_vtt(path, lang)


def _write_any(sub, path: str) -> None:
    lower = path.lower()
    if lower.endswith(".srt"):
        text = write_srt(sub)
    elif lower.endswith(".ttml") or lower.endswith(".xml") or lower.endswith(".itt"):
        text = write_ttml(sub)
    else:
        text = write_vtt(sub)
    Path(path).write_text(text, encoding="utf-8")


def cmd_validate(args) -> int:
    sub = _read_any(args.input, args.lang)
    violations = validate_subtitle(sub)
    errors = [v for v in violations if v.severity == "error"]
    for v in violations:
        print(f"[{v.severity.upper()}] cue {v.cue_index} {v.rule}: {v.detail}")
    print(
        f"\n{len(errors)} error(s), {len(violations) - len(errors)} warning(s) "
        f"across {len(sub.cues)} cues."
    )
    return 1 if errors else 0


def cmd_resegment(args) -> int:
    sub = _read_any(args.input, args.lang)
    spec = spec_for(args.lang)
    for cue in sub.cues:
        text = " ".join(cue.lines) if not spec.cjk else "".join(cue.lines)
        if spec.lang == "zh-Hans":
            text = normalize_zh_punct(text)
        cue.lines = break_lines(text, spec)
    _write_any(sub, args.output or args.input)
    print(f"Re-segmented {len(sub.cues)} cues for {args.lang} -> " f"{args.output or args.input}")
    return 0


def cmd_convert(args) -> int:
    sub = _read_any(args.input, args.lang)
    _write_any(sub, args.output)
    print(f"Converted {args.input} -> {args.output}")
    return 0


def cmd_review(args) -> int:
    """Assemble a review bundle from an EN master + translated subtitle files."""
    from .review import LangOutput, build_review_bundle

    en = _read_any(args.master, "en")
    outputs = []
    for pair in args.lang:
        code, _, path = pair.partition(":")
        if not path:
            print(f"bad --lang {pair!r}; use code:path e.g. ja:out.ja.vtt")
            return 2
        outputs.append(LangOutput(language=code, subtitle=_read_any(path, code)))
    written = build_review_bundle(args.out, en, outputs)
    print(f"Wrote {len(written)} files to {args.out}:")
    for p in written:
        print(f"  {p}")
    return 0


def cmd_pilot(args) -> int:
    """Run the pilot harness over clips with the CLI translator engine."""
    from . import llm
    from .glossary import load as load_glossary
    from .pilot import EngineSpec, load_clip, run_pilot

    langs = args.lang or ["zh-Hans", "ja"]
    glossary = load_glossary(args.glossary)
    clips = [load_clip(p) for p in args.inputs]
    translator = llm.translator(model_id=args.model)  # None -> CLI default model
    engine = EngineSpec(name=args.engine_name, translator=translator, reflect=not args.no_reflect)
    qe_scorer = None
    if args.gemba:
        from .eval import GembaMqmScorer

        qe_scorer = GembaMqmScorer(invoke=llm.translator(model_id=args.model)._invoke)
    report = run_pilot(
        clips, [engine], glossary, target_langs=tuple(langs), qe_scorer=qe_scorer, out_dir=args.out
    )
    print(f"Pilot: {len(clips)} clip(s) × 1 engine × {len(langs)} lang(s) " f"-> {args.out}")
    for c in report.cells:
        qe = f", QE {c.qe.system_score:.3f}" if c.qe else ""
        print(
            f"  {c.clip}/{c.engine}/{c.language}: {c.total_cues} cues, "
            f"{c.error_cues} err, {c.missing_terms} missing{qe} ({c.elapsed_s:.1f}s)"
        )
    print(f"Report: {args.out}/PILOT_REPORT.md")
    return 0


def _maybe_clean(sub, do_clean: bool):
    if not do_clean:
        return sub
    from . import llm
    from .cleanup import polish_subtitle

    return polish_subtitle(sub, llm.invoke())


def cmd_buildcues(args) -> int:
    """AWS Transcribe output JSON -> EN WebVTT master."""
    from pathlib import Path

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    transcript = from_transcribe_json(data, language="en")
    sub = _maybe_clean(build_en_cues(transcript), args.clean)
    _write_any(sub, args.output)
    print(
        f"Built {len(sub.cues)} EN cues from {transcript.duration:.1f}s "
        f"transcript{' (cleaned)' if args.clean else ''} -> {args.output}"
    )
    return 0


def cmd_transcribe(args) -> int:
    """Video/audio -> EN WebVTT master via local WhisperX (no AWS)."""
    from .transcribe import WhisperTranscriber

    tr = WhisperTranscriber(model_size=args.model_size, device=args.device)
    vocab = args.vocab.split(",") if args.vocab else None
    print(f"Transcribing {args.input} with WhisperX {args.model_size} ({args.device})…")
    transcript = tr.transcribe(args.input, language="en", custom_vocabulary=vocab)
    sub = _maybe_clean(build_en_cues(transcript), args.clean)
    _write_any(sub, args.output)
    print(
        f"Built {len(sub.cues)} EN cues from {transcript.duration:.1f}s audio"
        f"{' (cleaned)' if args.clean else ''} -> {args.output}"
    )
    return 0


def _persist_term_pins(outputs, out) -> None:
    """Merge auto-harvested term pins (LangOutput.term_pins) into this video's vocab.json so a
    re-run reapplies them, and print promotion suggestions. Video-specific pins persist here
    automatically; promotion to the SHARED glossary stays approval-gated (the reviewer decides
    which generalize) — we surface, we don't auto-edit the shared glossary."""
    from .vocab_prep import load_video_vocab, save_video_vocab

    merged: dict = {}
    for o in outputs:
        for lang, m in (getattr(o, "term_pins", {}) or {}).items():
            merged.setdefault(lang, {}).update(m)
    if not merged:
        return
    vpath = Path(out) / "vocab.json"
    vv = load_video_vocab(vpath)
    for lang, m in merged.items():
        vv.setdefault("renderings", {}).setdefault(lang, {}).update(m)
    save_video_vocab(vpath, vv)
    n = sum(len(m) for m in merged.values())
    print(
        f"Auto-harvested {n} term pin(s) -> {vpath} (reapplied on re-run). Consider promoting "
        f"recurring/generic ones to the shared glossary (reviewer-approved):"
    )
    for lang, m in merged.items():
        for src, spec in m.items():
            print(f"    [{lang}] {src!r} -> {spec['canonical']!r} (avoid {spec.get('avoid', [])})")


def _write_en_keep_orig(out, en_before, en_after) -> None:
    """Write the corrected EN to `out/en.vtt`, first preserving the original as `en.orig.vtt`
    (once: the earliest original wins)."""
    from .formats.vtt import write_vtt

    p, orig = Path(out) / "en.vtt", Path(out) / "en.orig.vtt"
    if not orig.exists():
        orig.write_text(
            p.read_text(encoding="utf-8") if p.exists() else write_vtt(en_before), encoding="utf-8"
        )
    p.write_text(write_vtt(en_after), encoding="utf-8")


def _clean_read_stage(en, out, invoke, title: str, en_style: str | None):
    """CLEAN-READ unscripted speech (deletions / punctuation only), so the review and the sign-off
    see the cleaned script. Scripted videos (tutorials, keynotes) are left as written. Returns the
    (possibly cleaned) EN and the en_review.md section describing it."""
    from .en_cleanread import clean_read, speech_style

    style = speech_style(en, invoke, title, Path(out) / "en_style.json", override=en_style)
    cleaned, rejected = [], 0
    if style["style"] == "unscripted":
        try:
            en_c, cleaned, rejected = clean_read(en, invoke)
        except Exception as e:
            print(f"EN clean-read skipped ({type(e).__name__})")
            en_c = en
        if cleaned:
            _write_en_keep_orig(out, en, en_c)
            en = en_c
        print(
            f"EN clean-read ({style['style']}: {style['why']}): {len(cleaned)} cue(s) cleaned, {rejected} rejected"
        )
    else:
        print(f"EN clean-read skipped: scripted ({style['why']})")
    md = [
        f"## Clean-read ({style['style']}: {style['why']}) — {len(cleaned)} cue(s), deletions only",
        "",
    ]
    if cleaned:
        md += ["| Time | Before | After |", "|---|---|---|"]
        md += [
            f"| {int(c['start']) // 60:02d}:{int(c['start']) % 60:02d} | {c['old']!r} | {c['new']!r} |"
            for c in cleaned
        ]
    else:
        md += ["_None._"]
    return en, md + [""]


def _en_review_md(title: str, cr_lines: list[str], applied: list, held: list) -> str:
    lines = [
        f"# EN editorial review — {title}".rstrip(),
        "",
        "Standard proofread of the English script. Clear-error fixes were auto-applied to the "
        "EN (original saved as en.orig.vtt); the rest are held for you to adjudicate.",
        "",
    ]
    lines += cr_lines
    lines += [f"## Auto-applied clear-error fixes ({len(applied)})", ""]
    if applied:
        lines += ["| Time | Kind | Fix | Corrected cue |", "|---|---|---|---|"]
        lines += [
            f"| {s['ts']} | {s['kind']} | {s['detail']} | {s['new_text']!r} |" for s in applied
        ]
    else:
        lines += ["_None._"]
    lines += ["", f"## Needs your adjudication ({len(held)})", ""]
    if held:
        lines += ["| Time | Kind | Detail |", "|---|---|---|"]
        lines += [f"| {s['ts']} | {s['kind']} | {s['detail']} |" for s in held]
    else:
        lines += ["_None._"]
    return "\n".join(lines) + "\n"


def _en_review(en, out, title: str = "", en_style: str | None = None):
    """General editorial review of the EN script (spelling, grammar, terminology, garbled
    phrasing) — the standard proofread any script gets, run on ANY EN source (a human transcript
    carries errors too, e.g. OpenTelemetry/'open telemetry').

    STANDARD PROCEDURE (mirrors QE auto-repair): auto-APPLY the clear-error fixes to the EN and HOLD
    the judgment calls for the operator (policy: auto-apply clear-error fixes, a human
    adjudicates the rest). When fixes are applied, the corrected EN is written to `out/en.vtt` (original preserved
    as `out/en.orig.vtt`) and RETURNED so translation runs on the corrected source. en_review.md
    lists what was auto-applied + what needs adjudication. Runs once per out dir (skipped if
    en_review.md already exists). Returns the (possibly corrected) EN subtitle."""
    rpath = Path(out) / "en_review.md"
    if rpath.exists():
        # a re-run: translate the CORRECTED EN this step saved (clean-read / review fixes), not the input
        saved = Path(out) / "en.vtt"
        if saved.exists() and (Path(out) / "en.orig.vtt").exists():
            return _read_any(str(saved), "en")
        return en
    from . import en_gate, llm
    from .transcribe_qe import en_autofix
    from .transcribe_qe import en_review as _rev

    invoke = llm.invoke()
    en, cr_lines = _clean_read_stage(en, out, invoke, title, en_style)
    try:
        issues = _rev(en, invoke)
    except Exception as e:
        print(f"EN review FAILED ({type(e).__name__}) — translation is blocked until it runs")
        en_gate.save_review_state(out, "failed", [])
        return en
    if not issues:
        en_gate.save_review_state(out, "ok", [])
        rpath.write_text(
            f"# EN editorial review — {title}\n\n"
            + "\n".join(cr_lines)
            + "\nNo editorial issues flagged.\n",
            encoding="utf-8",
        )
        print("EN review: no editorial issues flagged.")
        return en
    try:
        new_en, applied, held = en_autofix(en, issues, invoke)
    except Exception as e:
        print(f"EN autofix skipped ({type(e).__name__})")
        new_en, applied, held = en, [], issues
    if applied:  # persist corrected EN; keep the original
        _write_en_keep_orig(out, en, new_en)
    rpath.write_text(_en_review_md(title, cr_lines, applied, held), encoding="utf-8")
    en_gate.save_review_state(
        out, "ok", [{k: s.get(k, "") for k in ("ts", "kind", "detail")} for s in held], len(applied)
    )
    print(f"EN review: {len(applied)} auto-applied, {len(held)} held -> {rpath}")
    return new_en


def _looks_like_raw_caption(en) -> bool:
    """A raw caption (never clean-read / never built into our EN cues): spoken filler tokens, or a
    meaningful share of lines far beyond the EN line limit."""
    import re as _re

    spec = spec_for("en")
    lines = [ln for c in en.cues for ln in c.lines if ln.strip()]
    if not lines:
        return False
    text = " ".join(lines)
    fillers = len(_re.findall(r"(?i)\b(um+|uh+|erm)\b", text))
    overlong = sum(1 for ln in lines if len(ln) > spec.max_cpl * 1.5)
    return fillers >= 3 or overlong / len(lines) > 0.2


def _speech_align(en, out: Path):
    """EN vs the SPEECH (validate/speech_sync.py). Speech word timings come from this video's ASR
    (out/raw.json), its YouTube auto-captions (out/_ingest), or — via SOURCE-URL.txt — downloaded
    once into out/_speech. When EN cues are > 2 s off, re-time them to the speech (text unchanged;
    kept only if it validates and has fewer errors), then record what's still off in
    speech_sync.json, which the EN gate turns into held items. Returns the (possibly re-timed) EN.
    """
    from . import en_gate
    from .formats.vtt import write_vtt
    from .validate import validate_subtitle
    from .validate.speech_sync import (
        load_speech_words,
        retime_to_speech,
        speech_sync_report,
        speech_words_for,
    )

    speech: list = []
    raw = out / "raw.json"
    autos = sorted((out / "_ingest").glob("*.en*.vtt")) if (out / "_ingest").is_dir() else []
    try:
        if raw.exists():
            speech = load_speech_words(raw)
        for a in autos:
            if not speech:
                w = load_speech_words(a)
                speech = w if len(w) > 50 else []  # a human caption has no word timings
        if not speech and (out / "SOURCE-URL.txt").exists():
            import re as _re

            m = _re.search(
                r"v=([A-Za-z0-9_-]{11})|youtu\.be/([A-Za-z0-9_-]{11})",
                (out / "SOURCE-URL.txt").read_text(),
            )
            if m:
                speech = speech_words_for(m.group(1) or m.group(2), out / "_speech")
    except Exception as e:
        print(f"EN speech-sync check skipped ({type(e).__name__})")
        speech = []
    if not speech:
        print("EN speech-sync check: no word-timed speech for this video — not checked")
        return en
    rep = speech_sync_report(en, speech)
    errs = sum(o["severity"] == "error" for o in rep["off"])
    if errs:
        new, moved = retime_to_speech(en, speech)
        rep2 = speech_sync_report(new, speech)
        errs2 = sum(o["severity"] == "error" for o in rep2["off"])
        clean = not [
            v
            for v in validate_subtitle(new)
            if v.severity == "error" and v.rule != "duplicate_text"
        ]
        if clean and errs2 < errs:
            p = out / "en.vtt"
            if p.exists() and not (out / "en.orig.vtt").exists():
                (out / "en.orig.vtt").write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
            p.write_text(write_vtt(new), encoding="utf-8")
            print(f"EN re-timed to the speech: {moved} cue(s); sync errors {errs} -> {errs2}")
            en, rep, errs = new, rep2, errs2
    (out / en_gate.SPEECH_SYNC).write_text(
        json.dumps({"hash": en_gate.en_hash(en), **rep}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(
        f"EN speech-sync: median {rep['median']:+.2f} s over {rep['checked']} cues; "
        f"{errs} cue(s) > 2 s late"
    )
    return en


def _translate_and_bundle(
    args, en, out: Path, glossary, video_vocab, *, title: str, en_path: str, cache_path
) -> int:
    """The shared translate half of `localize` and `run`: EN review + sign-off gate, then
    localize_en into every language, persist harvested term pins, write the review bundle."""
    from . import en_gate, llm
    from .localize import localize_en
    from .review import build_review_bundle

    langs = args.lang or ["zh-Hans", "ja"]
    bank_for = _exemplar_bank_for(args, langs)
    en = _en_review(en, out, title, getattr(args, "en_style", None))
    en = _speech_align(en, out)
    if not en_gate.require(en, out, en_path):
        return 3
    print(
        f"Localizing {en_path} -> {', '.join(langs)} (glossary={args.glossary}"
        f"{', vocab=' + args.vocab if args.vocab else ''})…"
    )
    _save_description_file(args)
    outputs = localize_en(
        en,
        llm.translator(),
        glossary,
        langs=langs,
        video_vocab=video_vocab,
        backcheck=not args.no_backcheck,
        exemplar_bank_for=bank_for,
        tm=_translation_memory(args, langs),
        qe_judge=args.qe_judge,
        qe_second_judge=_qe_second_judge(args),
        qe_repair=not getattr(args, "no_qe_repair", False),
        conversational=_conversational(args, video_vocab),
        cache_path=cache_path,
        condense=not getattr(args, "no_condense", False),
        native_editor=not getattr(args, "no_editor", False),
        logger=lambda _m: print(_m, flush=True),
    )
    _persist_term_pins(outputs, out)
    written = build_review_bundle(out, en, outputs)
    print(f"Wrote {len(written)} files to {out}/ (REVIEW.md + per-language VTT+SRT)")
    return 0


def cmd_localize(args) -> int:
    """EN subtitle -> zh-Hans/ja review-ready cues (translate + glossary + register +
    reflect + completeness gates + CJK re-segmentation), then a review bundle."""
    from .cue_builder import strip_sound_tags
    from .glossary import load as load_glossary
    from .vocab_prep import load_video_vocab

    en = strip_sound_tags(_read_any(args.input, "en"))  # [music]/(applause) never reach translation
    glossary = load_glossary(args.glossary)
    vocab = load_video_vocab(args.vocab) if args.vocab else {"names": []}
    names = [n.strip() for n in (args.names or "").split(",") if n.strip()]
    if names:
        vocab["names"] = sorted(set(vocab.get("names", []) + names))
    # None when the vocab has nothing useful — localize treats that as "no per-video vocab".
    video_vocab = (
        vocab
        if any(vocab.get(k) for k in ("identifiers", "terms", "renderings", "names"))
        else None
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # RAW-CAPTION GUARD: `localize` used to take an EN subtitle as-is. Fed a raw YouTube caption,
    # that shipped 'um/uh', stutters and 100+ char lines as the EN deliverable (WindowsDevToTVs:
    # 126/132 cues invalid, 28 fillers). `run` already proofreads captions; give `localize` the
    # same clean-read whenever the EN looks raw.
    if _looks_like_raw_caption(en):
        from .cue_builder import caption_to_transcript, dedup_rolling_cues

        print(
            "EN looks like a raw caption (filler / over-long lines) — running caption clean-read…"
        )
        _vv = video_vocab or {}
        transcript = caption_to_transcript(dedup_rolling_cues(en))
        transcript = _correct_transcript(transcript, glossary, _vv, names=_vv.get("names"))
        en = build_en_cues(transcript)
    return _translate_and_bundle(
        args,
        en,
        out,
        glossary,
        video_vocab,
        title=args.input,
        en_path=args.input,
        cache_path=None if args.no_cache else str(out / "cache.jsonl"),
    )


def _save_raw(transcript, path) -> None:
    from pathlib import Path as _P

    _P(path).write_text(
        json.dumps(
            [
                {
                    "text": w.text,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence,
                    **({"speaker": w.speaker} if getattr(w, "speaker", None) else {}),
                }
                for w in transcript.words
            ]
        ),
        encoding="utf-8",
    )


def _load_raw(path):
    from .model import Transcript, Word

    ws = [Word(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]
    return Transcript(language="en", words=ws)


def _correct_transcript(transcript, glossary, video_vocab, names=None):
    """Glossary-aware clean-read correction of a WhisperX transcript, re-timed onto
    the original word timestamps: fixes mis-heard jargon, re-joins spelled-out
    identifiers (TVFocusGuideView), corrects non-native grammar, drops filler, and
    fixes mangled speaker names to their confirmed spelling."""
    from . import llm
    from .cleanup import correct_transcript
    from .cue_builder import low_confidence_spans
    from .transcribe.retime import retime_corrected

    terms = sorted(
        {f for t in glossary.terms for f in t.all_source_forms if any(c.isalpha() for c in f)}
    ) + list(glossary.identifiers)
    names = list(names or [])
    if video_vocab:
        terms += video_vocab.get("identifiers", []) + video_vocab.get("terms", [])
        names += video_vocab.get("names", [])
    suspects = [s["word"] for s in low_confidence_spans(transcript, threshold=0.5)]
    corrected = correct_transcript(
        llm.invoke(),
        transcript.text,
        terms,
        suspect_words=suspects,
        clean_read=True,
        names=sorted(set(names)) or None,
    )
    return retime_corrected(transcript, corrected)


def cmd_audit(args) -> int:
    """Audit OURS against a REFERENCE subtitle: substantive word-diffs, LLM-classified
    (ERROR/A_OK/EDGE), into a verdict-labeled review sheet. Transcription or translation."""
    from . import llm
    from .audit import classify, word_diffs, write_review_xlsx

    ours = _read_any(args.ours, args.lang or "en")
    ref = _read_any(args.reference, args.lang or "en")
    rows = word_diffs(ours, ref)
    print(f"{len(rows)} substantive diffs; classifying…")
    classify(rows, llm.invoke(), domain=args.domain, lang=args.lang or "")
    out = Path(args.out)
    counts = write_review_xlsx(rows, out, title=args.title or Path(args.ours).stem)
    print(f"verdicts {counts} -> {out}")
    return 0


def _exemplar_bank_for(args, langs):
    """Build the few-shot exemplar bank factory from the human corpus (the highest-ROI
    parity lever). Excludes the video being processed (held-out). Returns a
    `lang -> bank` callable, or None if disabled / no corpus found."""
    if getattr(args, "no_exemplars", False):
        return None
    from .corpus import build_exemplar_banks, video_id_from_input

    try:
        exclude = video_id_from_input(args.input)
        banks = build_exemplar_banks(langs, root=args.exemplars, exclude_video=exclude)
    except Exception as e:
        print(f"Exemplar mining skipped ({type(e).__name__}); prose-style only")
        return None
    if not banks:
        print(f"No exemplar corpus at {args.exemplars} — prose-style only")
        return None
    print(
        "Exemplar bank: "
        + ", ".join(f"{lang}={len(b.exemplars)} pairs" for lang, b in banks.items())
        + (f" (excluding {exclude})" if exclude else "")
    )
    return lambda lang: banks.get(lang)


def _translation_memory(args, langs):
    """Load (or seed-from-corpus) the translation memory: exact-match verbatim reuse +
    series-specific fuzzy exemplars. Held-out on the video being processed. None if --no-tm."""
    if getattr(args, "no_tm", False):
        return None
    from pathlib import Path as _P

    from .corpus import video_id_from_input
    from .tm import TranslationMemory

    path = args.tm
    if _P(path).exists():
        tm = TranslationMemory.load(path)
    else:
        print(f"Seeding TM from human corpus -> {path}")
        tm = TranslationMemory().seed_from_corpus(langs=tuple(langs))
        tm.save(path)
    tm = tm.exclude(video_id_from_input(args.input))
    print(f"Translation memory: {len(tm)} entries {tm.counts()}")
    return tm


def cmd_tm(args) -> int:
    """Manage the translation memory: seed from corpus, add an approved bundle, export TMX."""
    from .tm import TranslationMemory

    tm = TranslationMemory.load(args.tm) if Path(args.tm).exists() else TranslationMemory()
    if args.action == "seed":
        n0 = len(tm)
        tm.seed_from_corpus()
        tm.save(args.tm)
        print(
            f"Seeded TM from corpus: {len(tm)} entries (+{len(tm) - n0}) {tm.counts()} -> {args.tm}"
        )
    elif args.action == "add":
        en = _read_any(args.master, "en")
        added = 0
        for pair in args.lang:
            code, _, p = pair.partition(":")
            added += tm.add_from_subtitles(
                en, _read_any(p, code), code, args.video or Path(args.master).stem
            )
        tm.save(args.tm)
        print(f"Added {added} approved pair(s); TM now {len(tm)} {tm.counts()} -> {args.tm}")
    elif args.action == "import-tmx":
        n0 = len(tm)
        for f in args.tmx:
            added = tm.import_tmx(f, max_pairs=args.max_pairs)
            print(f"  imported {added} pair(s) from {Path(f).name}")
        tm.save(args.tm)
        print(f"TM now {len(tm)} entries (+{len(tm) - n0}) {tm.counts()} -> {args.tm}")
    elif args.action == "export":
        tm.export_tmx(args.out)
        print(f"Exported {len(tm)} entries -> {args.out}")
    elif args.action == "stats":
        print(f"TM {args.tm}: {len(tm)} entries {tm.counts()}")
    return 0


def cmd_rescore(args) -> int:
    """Re-judge a deliverable's CURRENT SRTs (e.g. after manual edits) and refresh REVIEW.md +
    status.json, so the reported DQF/suspects reflect the SHIPPED files — a hand-fix otherwise
    leaves the old, stale-low score. Re-judges cue-for-cue (target cue N vs EN cue N, strict 1:1)
    with the same panel as the pipeline and recomputes the deterministic suspects. Writes ONLY
    REVIEW.md + status.json — never the SRT/VTT, so hand-fixed line breaks are preserved."""
    from . import llm
    from .glossary import load as load_glossary
    from .localize import SentenceRecord, _write_status, llm_serialize_lock
    from .review import LangOutput, render_review_markdown
    from .translation_qe import translation_suspects

    out = Path(args.out)
    en_p = out / "en.vtt"
    if not en_p.exists():
        print(f"rescore: no en.vtt in {out}")
        return 2
    en = _read_any(str(en_p), "en")
    glossary = load_glossary(args.glossary)
    inv = llm.invoke()
    second = _qe_second_judge(args)
    outputs = []
    for lang in ("zh-Hans", "ja"):
        f = out / f"{lang}.srt"
        if not f.exists():
            continue
        tgt = _read_any(str(f), lang)
        recs = [
            SentenceRecord(lang, i, " ".join(e.lines), " ".join(t.lines), "", [])
            for i, (e, t) in enumerate(zip(en.cues, tgt.cues))
        ]
        with llm_serialize_lock(print):
            suspects, dqf = translation_suspects(
                recs, glossary, lang, invoke=inv, judge=args.qe_judge, second_judge=second
            )
        outputs.append(LangOutput(lang, tgt, qe_suspects=suspects))
        if dqf:
            _write_status(str(out / "cache.jsonl"), **{f"dqf_{lang}": dqf["score"]})
            print(
                f"{lang}: DQF {dqf['score']} {'PASS' if dqf['pass'] else 'FAIL'} "
                f"({len(suspects)} suspect(s))"
            )
    if not outputs:
        print(f"rescore: no zh-Hans.srt/ja.srt in {out}")
        return 2
    (out / "REVIEW.md").write_text(render_review_markdown(en, outputs), encoding="utf-8")
    print(f"rescore: refreshed {out}/REVIEW.md + status.json (SRTs untouched)")
    return 0


def _save_description_file(args):
    """--description-file: copy the supplied description into <out>/description.txt for the brief."""
    f = getattr(args, "description_file", None)
    if f and getattr(args, "out", None):
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "description.txt").write_text(
            Path(f).read_text(encoding="utf-8"), encoding="utf-8"
        )


def _qe_second_judge(args):
    """The different-family second judge for the two-judge QE panel (Claude <-> GPT, see llm.py).
    Only with --qe-judge; None when disabled or no other family is installed (single judge).
    De-biases a model grading its own family's output (self-preference)."""
    if not getattr(args, "qe_judge", False):
        return None
    from . import llm

    if getattr(args, "single_judge", False):
        print(
            f"QE judge: single-judge ({llm.label()} only) — 2nd judge disabled via --single-judge"
        )
        return None
    judge, what = llm.second_judge()
    if judge is None:
        print(f"QE judge: {what} — single-judge ({llm.label()}) DQF only")
        return None
    print(f"QE judge: two-judge panel ({llm.label()} + {what})")
    return judge


def _resolve_engine(name: str) -> str:
    """--engine auto: Parakeet on Apple Silicon when parakeet-mlx is installed (fastest, best on
    jargon), else WhisperX when installed; otherwise stop with the install options."""
    if name != "auto":
        return name
    import importlib.util
    import os
    import platform

    apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    if apple_silicon and importlib.util.find_spec("parakeet_mlx"):
        return "parakeet"
    if importlib.util.find_spec("whisperx"):
        print(
            "ASR: WhisperX (Parakeet needs Apple Silicon + parakeet-mlx)"
            if not apple_silicon
            else "ASR: WhisperX (parakeet-mlx not installed)"
        )
        return "whisper"
    if os.environ.get("ASSEMBLYAI_API_KEY"):
        print("ASR: AssemblyAI (no local ASR installed; $ASSEMBLYAI_API_KEY is set)")
        return "assemblyai"
    raise SystemExit(
        "No speech-recognition engine installed. Pick one:\n"
        + (
            "  pip install 'dev-titles[parakeet,whisper]'   # Parakeet + WhisperX drop recovery (Apple Silicon)\n"
            if apple_silicon
            else ""
        )
        + "  pip install 'dev-titles[whisper]'            # WhisperX, local, any platform\n"
        "  pip install 'dev-titles[assemblyai]'         # AssemblyAI cloud: set $ASSEMBLYAI_API_KEY, --engine assemblyai\n"
        "Or pass an existing English subtitle file instead of media."
    )


def _second_transcriber(name: str, device: str):
    """Pick the 2nd-ASR engine for the cross-check. `parakeet` (or a
    `parakeet-*`/`mlx-community/*` model id) routes to the DIFFERENT-architecture
    Parakeet (best for catching Whisper's silent drops); any other value is a
    WhisperX model size (e.g. `medium`)."""
    low = name.lower()
    if low == "parakeet" or low.startswith(("parakeet", "mlx-community")):
        from .transcribe import ParakeetTranscriber
        from .transcribe.parakeet import DEFAULT_MODEL

        return ParakeetTranscriber(model=DEFAULT_MODEL if low == "parakeet" else name)
    from .transcribe import WhisperTranscriber

    return WhisperTranscriber(model_size=name, device=device)


def _auto_conversational(video_vocab) -> bool:
    """Interview/dialogue heuristic: >=2 distinct named speakers -> conversational register
    (casual zh 你, spoken JA). Monologue tutorials have 0-1 speaker. Without this, an
    interview can ship in the formal register simply because nobody flagged it conversational;
    now conversational is the default for multi-speaker content. Overridable via --conversational / --no-conversational.
    """
    names = {
        (n or "").strip().lower() for n in (video_vocab or {}).get("names", []) if n and n.strip()
    }
    return len(names) >= 2


def _conversational(args, video_vocab) -> bool:
    """Resolve the register: explicit --conversational wins, then explicit --no-conversational,
    else auto-detect from speaker count."""
    if getattr(args, "conversational", False):
        return True
    if getattr(args, "no_conversational", False):
        return False
    return _auto_conversational(video_vocab)


def _run_vocab(args, out: Path) -> dict:
    """Prefer explicit --vocab; else auto-load a vocab.json the transcribe phase persisted in
    this out dir (the two-phase backlog flow: transcribe pins renderings, translate reuses).
    Operator-supplied --names are DNT too."""
    from .vocab_prep import load_video_vocab

    auto = out / "vocab.json"
    if args.vocab:
        video_vocab = load_video_vocab(args.vocab)
    elif auto.exists():
        video_vocab = load_video_vocab(auto)
        print(
            f"Loaded per-video vocab from {auto} "
            f"({sum(len(m) for m in video_vocab.get('renderings', {}).values())} renderings)."
        )
    else:
        video_vocab = {"names": []}
    names = [n.strip() for n in (args.names or "").split(",") if n.strip()]
    if names:
        video_vocab.setdefault("names", [])
        video_vocab["names"] = sorted(set(video_vocab["names"] + names))
    return video_vocab


def _en_from_caption(args, ri, glossary, video_vocab, out: Path):
    from .cue_builder import strip_sound_tags

    en = strip_sound_tags(_read_any(ri.path, "en"))  # [music] etc. never reach the EN master
    if not args.no_correct:
        # A human/imported caption gets the SAME proofread as ASR audio: merge rolling-caption
        # duplicates, then clean-read correct (fix mis-transcribed tech terms/homophones like
        # 'right once run anywhere'->'write once…', re-join code identifiers, normalize brand/
        # spelling, drop filler, fix mangled names). Reuses the transcript correction path via a
        # cue->word adapter so captions and audio share one correction — then re-segments cues.
        from .cue_builder import caption_to_transcript, dedup_rolling_cues

        deduped = dedup_rolling_cues(en)
        transcript = caption_to_transcript(deduped)
        transcript = _correct_transcript(
            transcript, glossary, video_vocab, names=video_vocab.get("names")
        )
        en = _maybe_clean(build_en_cues(transcript), args.clean)
        print(
            f"Proofread caption: {len(deduped.cues)} cues after rolling-dedup -> "
            f"{len(en.cues)} clean-read cues."
        )
    _write_any(en, str(out / "en.vtt"))
    return en


def _transcribe_media(args, ri, out: Path, video_vocab):
    """The primary ASR pass. Returns (transcript, recovery_second): the WhisperX drop-recovery
    transcript, reused as the auto-caption consult's 2nd ASR (None when it didn't run)."""
    from .transcribe import WhisperTranscriber

    # Reuse a saved raw transcript so a glossary/vocab change can re-correct
    # WITHOUT re-running WhisperX; else transcribe and persist it.
    raw_p = out / "raw.json"
    recovery_second = None
    if raw_p.exists() and not args.retranscribe:
        transcript = _load_raw(raw_p)
        print(f"Reusing saved raw transcript ({len(transcript.words)} words) — skip ASR")
        return transcript, None
    engine = _resolve_engine(args.engine)
    if engine == "parakeet":
        # Parakeet (parakeet-mlx): ~20-30x faster than WhisperX-CPU and gets the
        # technical proper nouns right (React Native/MCP/Claude Code) where Whisper mis-hears
        # them confidently. Its own errors are cosmetic (segmentation / British
        # spelling) and are cleaned up deterministically + by the clean-read pass.
        # No initial_prompt biasing (jargon is handled downstream).
        from .transcribe import ParakeetTranscriber

        print("Transcribing with Parakeet (parakeet-mlx)…")
        transcript = ParakeetTranscriber().transcribe(ri.path, language="en")
        # DROP RECOVERY: Parakeet (RNNT/TDT) silently deletes whole spans on
        # music/non-speech (NeMo #14430). A cheap WhisperX pass (base) catches
        # them; we splice ONLY the dropped spans back (surgical — no regression
        # on what Parakeet got right). Off with --no-recover.
        if not args.no_recover:
            from .transcribe_qe import recover_dropped_spans

            print(f"  drop-recovery cross-check: WhisperX {args.recover_model}…")
            try:
                second = WhisperTranscriber(
                    model_size=args.recover_model, device=args.device
                ).transcribe(ri.path, language="en")
                transcript, n = recover_dropped_spans(transcript, second)
                recovery_second = second
                print(
                    f"  recovered {n} dropped word(s) Parakeet omitted"
                    if n
                    else "  no dropped spans found"
                )
            except Exception as e:
                if "pip install" in str(e):  # WhisperX not installed: say so loudly
                    print(
                        "  WARNING: drop recovery is OFF — WhisperX is not installed, so speech "
                        "Parakeet drops over music/non-speech will be missing. Install it with "
                        "pip install 'dev-titles[whisper]' (or pass --no-recover to silence this)."
                    )
                else:
                    print(
                        f"  recovery skipped ({type(e).__name__}: {str(e)[:120]}); Parakeet output kept"
                    )
    elif engine == "assemblyai":
        # AssemblyAI cloud (Universal + diarization) — for public interview content.
        # A/B showed it isn't more COMPLETE than Parakeet, but transcribes conversational
        # JARGON right at the source (fewer real errors: 1 vs 4 on a 10-min interview), for
        # ~$0.26/hr. No drop-recovery needed. Key from $ASSEMBLYAI_API_KEY.
        from .transcribe import AssemblyAITranscriber

        print("Transcribing with AssemblyAI (cloud; diarized)…")
        transcript = AssemblyAITranscriber().transcribe(ri.path, language="en")
    else:
        tr = WhisperTranscriber(model_size=args.model_size, device=args.device)
        # Light acoustic prime: WhisperX's initial_prompt is for SHORT priming — a
        # long term list (48) hung the decode (~6x slower, confirming prior research
        # that initial_prompt biasing is weak+slow). So bias with a SMALL, high-value
        # set only (speaker names + a few per-video terms); code identifiers are
        # handled by the glossary-aware correction + per-video vocab instead.
        bias = list(
            dict.fromkeys(
                list(video_vocab.get("names", [])) + list(video_vocab.get("terms", []))[:6]
            )
        )[:8]
        msg = f"; priming {len(bias)} terms" if bias else ""
        print(f"Transcribing with WhisperX {args.model_size} ({args.device}){msg}…")
        transcript = tr.transcribe(ri.path, language="en", custom_vocabulary=bias or None)
    _save_raw(transcript, raw_p)
    return transcript, recovery_second


def _apply_speaker_names(args, ri, transcript, video_vocab) -> None:
    """Names as part of ingestion: metadata (video description) is reliable -> auto-apply;
    transcript self-intros are candidates (ASR may mangle) -> hint. Sets video_vocab['names']."""
    from .vocab_prep import (
        extract_name_candidates,
        harvest_names,
        load_name_registry,
        match_registry_names,
        reconcile_names,
    )

    meta_names = harvest_names("", ri.metadata)
    if ri.metadata and getattr(args, "out", None):
        # keep the written description for the per-video brief (names/roles are authoritative there)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "description.txt").write_text(ri.metadata, encoding="utf-8")
    # Channel NAME REGISTRY (glossary/names.yaml): canonical host/guest spellings so a
    # mangled surname is corrected to the KNOWN spelling, not a per-run guess (the
    # per-video metadata harvest is transient/unreliable). Match against this video's
    # description (reliable) + transcript.
    registry = load_name_registry()
    reg_names = match_registry_names(
        (ri.metadata or "") + "\n" + transcript.text,
        registry,
        first_name_context=ri.metadata or "",
    )
    reliable = sorted(set(video_vocab.get("names", []) + meta_names + reg_names))  # authoritative
    if reg_names:
        print(f"Names from channel registry: {', '.join(reg_names)}")
    if meta_names:
        print(f"Speaker names from video metadata: {', '.join(meta_names)}")
    # add self-intro names only if they don't collide with a reliable name
    # (a colliding one is a mishearing -> stays a correction TARGET, not a name)
    cand = extract_name_candidates(transcript.text)
    video_vocab["names"] = reconcile_names(reliable, cand)
    dropped = [c for c in cand if c not in video_vocab["names"]]
    if video_vocab["names"]:
        print(
            f"Speaker names applied: {', '.join(video_vocab['names'])}"
            + (f" (ignored likely-mishears {dropped})" if dropped else "")
        )


def _prep_vocab(args, ri, transcript, video_vocab):
    """Live vocab-prep (no YouTube metadata to lean on): identify the subject FROM the
    transcript, pull source docs, and build per-video vocab. If it can't identify the
    subject/docs, REQUEST a docs URL — do NOT silently proceed. Returns the (possibly new)
    video vocab, or None to halt (--require-prep)."""
    have_vocab = video_vocab and (video_vocab.get("identifiers") or video_vocab.get("renderings"))
    if args.no_prep or have_vocab:
        return video_vocab
    from . import llm
    from .vocab_prep import build_video_vocab, curated_forms, http_fetch, identify_subject

    inv = llm.invoke()
    subj = identify_subject(transcript.text, inv)
    urls = list(args.docs_url or []) + (subj.get("doc_urls") or [])
    if not urls and not args.no_search:
        # No known doc URLs: web-search for the product's docs so brand/product
        # names get grounded automatically (Google CSE -> DuckDuckGo, best-effort).
        from .search import web_search

        q = (subj.get("subject") or ri.title or "").strip()
        found = web_search(f"{q} product documentation", k=4) if q else []
        if found:
            urls = found[:2]
            print(f"  web-search: found {len(urls)} candidate doc(s) for {q!r}")
    names0 = video_vocab.get("names", []) if video_vocab else []
    if urls:
        print(f"Subject: {subj.get('subject', '?')} — pre-ingesting vocab from {len(urls)} doc(s)…")
        try:
            vv = build_video_vocab(
                transcript.text,
                docs_search=lambda q: "",
                web_fetch=http_fetch,
                third_party={f"d{i}": u for i, u in enumerate(urls)},
                translate=inv,
                known=curated_forms(args.glossary),
            )
            vv["names"] = names0
            print(
                f"  pre-ingested {len(vv['identifiers'])} identifiers, "
                f"{sum(len(m) for m in vv['renderings'].values())} pinned renderings"
            )
            return vv
        except Exception as e:
            print(f"  doc fetch failed ({type(e).__name__}); continuing on glossary only")
            return video_vocab
    if not subj.get("confident") and args.require_prep:
        print(
            "NEEDS INPUT (--require-prep): couldn't identify the video's subject or "
            "find docs to pre-ingest vocab. Halting — re-run with --docs-url "
            "<feature/library doc URL> or --vocab <file>."
        )
        return None
    # No docs (typical for partner interviews). STILL pin per-video renderings for
    # CONSISTENCY: propose_renderings gives one canonical per recurring term, and
    # resolve_entity_renderings looks up brand/product names authoritatively
    # (Wikipedia). Needs no docs — just the transcript + LLM + web fetch. The
    # reviewer approves the choices; this stops per-video terms flip-flopping.
    try:
        vv = build_video_vocab(
            transcript.text,
            docs_search=lambda q: "",
            web_fetch=http_fetch,
            translate=inv,
            known=curated_forms(args.glossary),
        )
        vv["names"] = names0
        print(
            f"No source docs — pinned {sum(len(m) for m in vv['renderings'].values())} "
            f"per-video renderings for consistency (reviewer approves)."
        )
        return vv
    except Exception as e:
        print(f"  rendering-proposal failed ({type(e).__name__}); glossary + names only.")
        return video_vocab


def _en_from_transcript(args, ri, transcript, glossary, video_vocab, out: Path, recovery_second):
    """Raw ASR transcript -> the corrected EN master (`out/en.vtt`)."""
    from . import llm
    from .vocab_prep import save_video_vocab

    # Persist per-video vocab so a later --translate-only run (the two-phase backlog flow)
    # reuses the pinned renderings/identifiers/names — else they're built at transcribe and
    # lost before translation. Auto-loaded by _run_vocab when --vocab isn't passed.
    if video_vocab and (
        video_vocab.get("renderings") or video_vocab.get("identifiers") or video_vocab.get("names")
    ):
        save_video_vocab(out / "vocab.json", video_vocab)
    # PRE-CLEAN-READ repeated-run recovery: collapse a repeated-token run ('the the the …') —
    # a genuine speaker stutter or a decoder loop — from the auto-caption BEFORE clean-read,
    # otherwise clean-read rewrites the run into duplicated real text the cue-level consult
    # can't detect.
    if ri.auto_captions:
        from .transcribe_qe import load_autosub, recover_repeated_runs

        try:
            transcript, _nh = recover_repeated_runs(transcript, load_autosub(ri.auto_captions))
            if _nh:
                print(f"  collapsed {_nh} repeated-token run(s) from auto-caption (pre-clean-read)")
        except Exception as e:
            print(f"  repeated-run recovery skipped ({type(e).__name__})")
    if not args.no_correct:
        transcript = _correct_transcript(
            transcript, glossary, video_vocab, names=video_vocab.get("names")
        )
        print("Applied glossary-aware clean-read correction (jargon + grammar + filler + names).")
    en = _maybe_clean(build_en_cues(transcript), args.clean)
    # PRUNE VOCAB TO THE CORRECTED TEXT: the vocab was harvested from the RAW ASR (above,
    # before clean-read), so an ASR mishear became a bogus keep-Latin term ('Chief Devilist'
    # <- 'Chief Evangelist'). Now that clean-read has fixed the captions, drop any term the
    # corrected EN no longer contains, and re-save — else the garble is forced, untranslated,
    # into the output.
    if video_vocab and video_vocab.get("terms"):
        from .vocab_prep import prune_terms_to_text

        _corrected_text = " ".join(" ".join(c.lines) for c in en.cues)
        _dropped = prune_terms_to_text(video_vocab, _corrected_text)
        if _dropped:
            save_video_vocab(out / "vocab.json", video_vocab)
            print(
                f"  pruned {len(_dropped)} ASR-garble term(s) absent from corrected EN: {_dropped}"
            )
    # AUTO-CAPTION CONSULT: use YouTube auto-captions + the WhisperX 2nd ASR to auto-
    # correct garbled/OOV spans the primary got wrong — only where BOTH agree (safe).
    # Needs the recovery WhisperX pass (default parakeet path) as the 2nd source.
    if not args.no_qe and ri.auto_captions and recovery_second is not None:
        from .transcribe_qe import autosub_consult, load_autosub, plausibility_suspects

        try:
            autosub = load_autosub(ri.auto_captions)
            en_text = " ".join(" ".join(c.lines) for c in en.cues)
            flags = [p["phrase"] for p in plausibility_suspects(llm.invoke(), en_text)]
            fixes = autosub_consult(
                en, recovery_second, autosub, flags, logger=lambda m: print(m, flush=True)
            )
            if fixes:
                print(
                    f"Auto-caption consult: auto-corrected {len(fixes)} span(s) "
                    "(auto-caption + 2nd-ASR agree)"
                )
        except Exception as e:
            print(f"  auto-caption consult skipped ({type(e).__name__})")
    _write_any(en, str(out / "en.vtt"))
    print(f"EN master: {len(en.cues)} cues -> {out}/en.vtt")
    return en


def _write_suspect_list(args, ri, raw_transcript, en, out: Path) -> None:
    """Reference-free QE: a timestamped suspect list for a human ear (no caption needed)."""
    from . import llm
    from .transcribe_qe import transcription_suspects, write_suspects_md

    second_asr: "Transcript | None" = None
    if args.second_pass:
        print(f"Second ASR pass ({args.second_pass}) for cross-check…")
        second_asr = _second_transcriber(args.second_pass, args.device).transcribe(
            ri.path, language="en"
        )
    suspects = transcription_suspects(
        raw_transcript, llm.invoke(), second_transcript=second_asr, corrected=en
    )
    write_suspects_md(suspects, out / "suspects.md", ri.title)
    print(f"Reference-free QE: {len(suspects)} suspects to spot-check -> {out}/suspects.md")


def cmd_run(args) -> int:
    """One entry: resolve an input (local file or YouTube URL) then do transcription,
    translation, or both. Media -> both by default; a subtitle input -> translate only."""
    from .glossary import load as load_glossary
    from .ingest import resolve_input

    ri = resolve_input(args.input, workdir=str(Path(args.out) / "_ingest"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (
        str(args.input).startswith(("http://", "https://"))
        and not (out / "SOURCE-URL.txt").exists()
    ):
        # the source link: listen-list pages (EN_LISTEN.html), held-out exclusion and publishing use it
        (out / "SOURCE-URL.txt").write_text(str(args.input).strip() + "\n", encoding="utf-8")
    glossary = load_glossary(args.glossary)
    video_vocab = _run_vocab(args, out)
    print(f"Resolved {args.input} -> {ri.kind} ({ri.title or ri.path})")

    # 1) EN master: from a subtitle input directly, else transcribe (+ clean-read correct) the media.
    if ri.kind == "subtitle":
        en = _en_from_caption(args, ri, glossary, video_vocab, out)
    else:
        if args.translate_only and not args.transcribe_only:
            print("Media input needs transcription before translation; transcribing.")
        transcript, recovery_second = _transcribe_media(args, ri, out, video_vocab)
        _apply_speaker_names(args, ri, transcript, video_vocab)
        video_vocab = _prep_vocab(args, ri, transcript, video_vocab)
        if video_vocab is None:
            return 2
        en = _en_from_transcript(args, ri, transcript, glossary, video_vocab, out, recovery_second)
        if not args.no_qe:
            # the pre-correction transcript (has confidences) is what QE scores
            _write_suspect_list(args, ri, transcript, en, out)
    if args.transcribe_only:
        return 0

    # 2) Translate the EN master.
    return _translate_and_bundle(
        args,
        en,
        out,
        glossary,
        video_vocab,
        title=ri.title or ri.path,
        en_path=str(out / "en.vtt"),
        cache_path=str(out / "cache.jsonl"),
    )


def cmd_approve_en(args) -> int:
    import getpass

    from . import en_gate

    en_path = args.en or str(Path(args.out) / "en.vtt")
    en = _read_any(en_path, "en")
    status, held = en_gate.review_state(args.out)
    if status != "ok":
        print(f"Cannot sign off: the EN review has not run successfully (status: {status}).")
        return 3
    for h in held:
        print(f"  held {h.get('ts', '?'):>6}  {h.get('kind', '')}: {h.get('detail', '')}")
    rec = en_gate.approve(args.out, en, by=getpass.getuser(), note=args.note)
    print(
        f"EN signed off: {en_path} ({len(held)} held line(s) resolved by {rec['by']}, {rec['at']})"
    )
    return 0


def cmd_promote_terms(args) -> int:
    from .promote import promote_recurring_terms, promotion_renderings

    if args.dry_run:
        r = promotion_renderings(args.ledger, min_videos=args.min_videos)
        n = sum(len(m) for m in r.values())
        if not n:
            print(f"No terms corroborated across >= {args.min_videos} videos in {args.ledger}.")
            return 0
        print(f"Would promote (>= {args.min_videos} videos, consistent rendering):")
        for lang, m in r.items():
            for src, spec in m.items():
                av = f"  (avoid: {', '.join(spec['avoid'])})" if spec.get("avoid") else ""
                print(f"  [{lang}] {src} -> {spec['canonical']}{av}")
        return 0
    promoted = promote_recurring_terms(
        args.ledger, args.glossary, min_videos=args.min_videos, logger=print
    )
    return 0 if promoted is not None else 1


def _add_simple_parsers(sub) -> None:
    v = sub.add_parser("validate", help="check a subtitle file against per-language rules")
    v.add_argument("input")
    v.add_argument("--lang", required=True)
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("resegment", help="re-break lines + kinsoku/zh-punct for a language")
    r.add_argument("input")
    r.add_argument("--lang", required=True)
    r.add_argument("-o", "--output")
    r.set_defaults(func=cmd_resegment)

    c = sub.add_parser("convert", help="convert between vtt/srt/ttml")
    c.add_argument("input")
    c.add_argument("-o", "--output", required=True)
    c.add_argument("--lang", default="en")
    c.set_defaults(func=cmd_convert)

    b = sub.add_parser("buildcues", help="AWS Transcribe JSON -> EN WebVTT master")
    b.add_argument("input", help="AWS Transcribe output .json")
    b.add_argument("-o", "--output", required=True)
    b.add_argument("--clean", action="store_true", help="LLM cleanup pass per cue")
    b.set_defaults(func=cmd_buildcues)

    tc = sub.add_parser("transcribe", help="video/audio -> EN WebVTT via local WhisperX")
    tc.add_argument("input", help="media file (mp4/mp3/wav/…)")
    tc.add_argument("-o", "--output", required=True)
    tc.add_argument("--model-size", default="large-v3", dest="model_size")
    tc.add_argument("--device", default="cpu", help="cpu | cuda")
    tc.add_argument("--vocab", default=None, help="comma-separated jargon terms to bias")
    tc.add_argument("--clean", action="store_true", help="LLM cleanup pass per cue")
    tc.set_defaults(func=cmd_transcribe)

    pl = sub.add_parser("pilot", help="run the pilot harness over clips (CLI engine)")
    pl.add_argument("inputs", nargs="+", help="EN clips: .vtt/.srt or Transcribe .json")
    pl.add_argument(
        "--lang",
        action="append",
        default=None,
        help="target language (repeatable; default zh-Hans + ja)",
    )
    pl.add_argument("--out", default="pilot_out", help="output directory")
    pl.add_argument("--glossary", default="glossary/glossary.yaml")
    pl.add_argument("--model", default=None, help="pin a model id (default: CLI default)")
    pl.add_argument("--engine-name", default="cli", dest="engine_name")
    pl.add_argument("--no-reflect", action="store_true", help="skip the refine pass")
    pl.add_argument("--gemba", action="store_true", help="QE via GEMBA-MQM (CLI judge)")
    pl.set_defaults(func=cmd_pilot)


def _add_translate_options(p) -> None:
    """The translation options `run` and `localize` share."""
    p.add_argument(
        "--en-style",
        choices=["scripted", "unscripted"],
        help="override the scripted/unscripted call that decides whether the EN gets a clean-read "
        "(unscripted: stutters/filler removed, deletions only; scripted e.g. tutorials: left as written)",
    )
    p.add_argument(
        "--lang",
        action="append",
        default=None,
        help="target language (repeatable; default zh-Hans + ja)",
    )
    p.add_argument("--glossary", default="glossary/glossary.yaml")
    p.add_argument("--vocab", default=None, help="per-video vocab JSON (vocab-prep output)")
    p.add_argument(
        "--names",
        default=None,
        help="comma-separated speaker/proper names (DNT; ASR mishears corrected to these)",
    )
    p.add_argument(
        "--no-backcheck",
        action="store_true",
        help="skip the source-anchored coverage edit-fix gate",
    )
    p.add_argument(
        "--exemplars",
        default="input/channel",
        help="human parallel-subtitle corpus dir to mine few-shot exemplars from",
    )
    p.add_argument(
        "--no-exemplars",
        action="store_true",
        help="disable few-shot exemplars (prose-style directives only)",
    )
    p.add_argument(
        "--tm",
        default="tm/tm.jsonl",
        help="translation-memory store (exact-match reuse + series exemplars; "
        "seeded from the human corpus if absent)",
    )
    p.add_argument("--no-tm", action="store_true", help="disable the translation memory")
    p.add_argument(
        "--qe-judge",
        action="store_true",
        dest="qe_judge",
        help="add the LLM MQM two-judge panel (primary model + a different-family judge if installed) "
        "scored via calibrated DQF; deterministic QE runs regardless",
    )
    p.add_argument(
        "--single-judge",
        action="store_true",
        dest="single_judge",
        help="force single-model QE (disable the different-family 2nd judge) — use when "
        "codex/Bedrock is unavailable",
    )
    p.add_argument(
        "--no-qe-repair",
        action="store_true",
        help="skip QE-guided auto-repair of judge-flagged majors (accept-only-if-better)",
    )
    p.add_argument(
        "--conversational",
        action="store_true",
        help="casual/interview content: zh uses 你 (not formal 您); docs default keeps 您",
    )
    p.add_argument(
        "--no-conversational",
        action="store_true",
        help="force formal register (zh 您) even if auto-detected as an interview",
    )
    p.add_argument(
        "--no-condense",
        action="store_true",
        help="ja: skip the condensation pass (on by default: tightens translations like a subtitle "
        "editor before QE, guarded; ~12%% shorter)",
    )
    p.add_argument(
        "--no-editor",
        action="store_true",
        help="skip the native-editor pass (on by default: a native subtitle editor flags lines a "
        "viewer would notice and rewrites them; accepted only if a second judge agrees)",
    )
    p.add_argument(
        "--description-file",
        help="video description text (names, roles, topic) for the per-video brief — for videos "
        "not on YouTube yet; published videos fetch it automatically",
    )


def _add_run_parser(sub) -> None:
    rn = sub.add_parser(
        "run", help="one entry: input (file/YouTube URL) -> transcribe and/or translate"
    )
    rn.add_argument("input", help="local media/subtitle file OR a YouTube URL")
    rn.add_argument("--out", required=True, help="output directory")
    _add_translate_options(rn)
    rn.add_argument("--transcribe-only", action="store_true", help="EN transcription only")
    rn.add_argument(
        "--translate-only", action="store_true", help="translation only (input is EN subs)"
    )
    rn.add_argument("--clean", action="store_true", help="LLM cleanup pass on ASR cues")
    rn.add_argument(
        "--no-correct",
        action="store_true",
        help="skip the glossary-aware clean-read ASR correction",
    )
    rn.add_argument(
        "--retranscribe",
        action="store_true",
        help="force WhisperX even if a saved raw transcript exists",
    )
    rn.add_argument(
        "--docs-url",
        action="append",
        default=None,
        metavar="URL",
        help="source doc URL to pre-ingest vocab from (repeatable; for the no-metadata live case)",
    )
    rn.add_argument(
        "--no-prep",
        action="store_true",
        help="skip the identify-subject + doc-fetch vocab pre-ingestion",
    )
    rn.add_argument(
        "--no-search",
        action="store_true",
        help="don't web-search for product docs when no --docs-url is given "
        "(search via DuckDuckGo; needs `pip install ddgs`)",
    )
    rn.add_argument(
        "--require-prep",
        action="store_true",
        help="halt if per-video vocab can't be built (strict pipelines)",
    )
    rn.add_argument(
        "--no-qe",
        action="store_true",
        help="skip the reference-free suspect list (confidence + LLM plausibility)",
    )
    rn.add_argument(
        "--second-pass",
        default=None,
        metavar="ENGINE",
        help="run a 2nd ASR and flag disagreements as suspects: 'parakeet' "
        "(different architecture; best at catching Whisper's silent "
        "drops) or a WhisperX model size (e.g. medium)",
    )
    rn.add_argument(
        "--engine",
        choices=["auto", "whisper", "parakeet", "assemblyai"],
        default="auto",
        help="primary ASR: auto (default: parakeet on Apple Silicon if installed, else "
        "whisper), parakeet (fast, local; Apple Silicon), whisper (WhisperX large-v3, local, "
        "any platform; slower on CPU), or assemblyai (cloud, diarized; uploads the audio; "
        "needs $ASSEMBLYAI_API_KEY)",
    )
    rn.add_argument(
        "--no-recover",
        action="store_true",
        help="skip the WhisperX drop-recovery cross-check (parakeet engine): "
        "Parakeet can silently delete spans on music/non-speech",
    )
    rn.add_argument(
        "--recover-model",
        default="base",
        dest="recover_model",
        help="WhisperX model for drop recovery (default base; small = better/slower)",
    )
    rn.add_argument("--model-size", default="large-v3", dest="model_size")
    rn.add_argument("--device", default="cpu", help="cpu | cuda")
    rn.set_defaults(func=cmd_run)


def _add_localize_parser(sub) -> None:
    lo = sub.add_parser("localize", help="EN subtitle -> zh-Hans/ja review-ready cues + bundle")
    lo.add_argument("input", help="EN master subtitle (.vtt/.srt/.sbv)")
    lo.add_argument("--out", required=True, help="output bundle directory")
    _add_translate_options(lo)
    lo.add_argument("--no-cache", action="store_true", help="don't write a resumable cache")
    lo.set_defaults(func=cmd_localize)


def _add_review_parsers(sub) -> None:
    au = sub.add_parser(
        "audit", help="audit OURS vs a reference subtitle -> verdict-labeled review sheet"
    )
    au.add_argument("ours", help="our output subtitle (.vtt/.srt/.sbv)")
    au.add_argument("reference", help="reference subtitle (human caption/translation)")
    au.add_argument("--out", required=True, help="output .xlsx path")
    au.add_argument("--domain", choices=["transcription", "translation"], default="transcription")
    au.add_argument("--lang", default=None, help="language (default en; e.g. ja for translation)")
    au.add_argument("--title", default=None)
    au.set_defaults(func=cmd_audit)

    rv = sub.add_parser("review", help="assemble a docs-team review bundle")
    rv.add_argument("master", help="EN master .vtt/.srt")
    rv.add_argument(
        "--lang",
        action="append",
        required=True,
        metavar="code:path",
        help="target language file, e.g. ja:out.ja.vtt (repeatable)",
    )
    rv.add_argument("--out", required=True, help="output bundle directory")
    rv.set_defaults(func=cmd_review)

    rs = sub.add_parser(
        "rescore", help="re-judge a deliverable's current SRTs; refresh REVIEW.md + status.json"
    )
    rs.add_argument(
        "--out", required=True, help="deliverable dir (with en.vtt + zh-Hans.srt/ja.srt)"
    )
    rs.add_argument("--glossary", default="glossary/glossary.yaml")
    rs.add_argument(
        "--qe-judge", action="store_true", default=True, help="run the LLM MQM panel (default on)"
    )
    rs.add_argument(
        "--single-judge",
        action="store_true",
        help="primary-model-only DQF (skip the different-family 2nd judge)",
    )
    rs.set_defaults(func=cmd_rescore)

    ap = sub.add_parser(
        "approve-en",
        help="sign off the English script (after resolving every held EN-review line) so it can be translated",
    )
    ap.add_argument("out", help="the video's output dir (holds en_review.json)")
    ap.add_argument("--en", help="the EN file that will be translated (default <out>/en.vtt)")
    ap.add_argument("--note", default="", help="what was resolved, for the record")
    ap.set_defaults(func=cmd_approve_en)


def _add_glossary_parsers(sub) -> None:
    tm = sub.add_parser("tm", help="manage the translation memory (seed/add/export/stats)")
    tm.add_argument("action", choices=["seed", "add", "export", "stats", "import-tmx"])
    tm.add_argument("--tm", default="tm/tm.jsonl", help="TM store path (JSONL)")
    tm.add_argument(
        "--tmx",
        action="append",
        default=[],
        help="[import-tmx] TMX file(s) to import (repeatable; import Gold AFTER Silver)",
    )
    tm.add_argument("--max-pairs", type=int, default=None, help="[import-tmx] cap pairs (testing)")
    tm.add_argument("--master", help="[add] EN master subtitle to align approved targets to")
    tm.add_argument(
        "--lang",
        action="append",
        default=[],
        metavar="code:path",
        help="[add] approved target file, e.g. ja:out.ja.vtt (repeatable)",
    )
    tm.add_argument("--video", default=None, help="[add] provenance video id")
    tm.add_argument("--out", help="[export] output .tmx path")
    tm.set_defaults(func=cmd_tm)

    pt = sub.add_parser(
        "promote-terms",
        help="promote recurring term-pins (corroborated across videos) to the shared glossary",
    )
    pt.add_argument(
        "--ledger",
        default="output/ticket/term_ledger.jsonl",
        help="cross-video term-pin ledger (written during localize runs)",
    )
    pt.add_argument("--glossary", default="glossary/glossary.yaml")
    pt.add_argument(
        "--min-videos",
        type=int,
        default=2,
        help="promote only terms corroborated across >= this many distinct videos",
    )
    pt.add_argument(
        "--dry-run",
        action="store_true",
        help="show what WOULD promote without editing the glossary",
    )
    pt.set_defaults(func=cmd_promote_terms)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dev_titles")
    p.add_argument(
        "--llm",
        choices=("claude", "codex", "openai"),
        default=None,
        help="primary model backend: claude (local Claude Code CLI, default), codex (local OpenAI "
        "Codex CLI, e.g. signed in with ChatGPT), openai (any OpenAI-compatible API via "
        "$OPENAI_API_KEY / $OPENAI_BASE_URL). Also $DEV_TITLES_LLM.",
    )
    p.add_argument(
        "--llm-model", default=None, help="model id for the --llm backend (required for openai)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    _add_simple_parsers(sub)
    _add_run_parser(sub)
    _add_localize_parser(sub)
    _add_review_parsers(sub)
    _add_glossary_parsers(sub)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from . import llm

    llm.configure(args.llm, args.llm_model)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
