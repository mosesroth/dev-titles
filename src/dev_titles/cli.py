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
    from .glossary import load as load_glossary
    from .pilot import EngineSpec, load_clip, run_pilot
    from . import llm

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
    from .cleanup import polish_subtitle
    from . import llm

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


def _en_review(en, out, title: str = ""):
    """General editorial review of the EN script (spelling, grammar, terminology, garbled
    phrasing) — the standard proofread any script gets, run on ANY EN source (a human transcript
    carries errors too, e.g. OpenTelemetry/'open telemetry').

    STANDARD PROCEDURE (mirrors QE auto-repair): auto-APPLY the clear-error fixes to the EN and HOLD
    the judgment calls for the operator (policy: auto-apply clear-error fixes, a human
    adjudicates the rest). When fixes are applied, the corrected EN is written to `out/en.vtt` (original preserved
    as `out/en.orig.vtt`) and RETURNED so translation runs on the corrected source. en_review.md
    lists what was auto-applied + what needs adjudication. Runs once per out dir (skipped if
    en_review.md already exists). Returns the (possibly corrected) EN subtitle."""
    from pathlib import Path

    from .formats.vtt import write_vtt

    rpath = Path(out) / "en_review.md"
    if rpath.exists():
        return en
    from .transcribe_qe import en_autofix
    from .transcribe_qe import en_review as _rev
    from . import llm

    invoke = llm.invoke()
    try:
        issues = _rev(en, invoke)
    except Exception as e:
        print(f"EN review skipped ({type(e).__name__})")
        return en
    if not issues:
        rpath.write_text(
            f"# EN editorial review — {title}\n\nNo editorial issues flagged.\n", encoding="utf-8"
        )
        print("EN review: no editorial issues flagged.")
        return en
    try:
        new_en, applied, held = en_autofix(en, issues, invoke)
    except Exception as e:
        print(f"EN autofix skipped ({type(e).__name__})")
        new_en, applied, held = en, [], issues
    if applied:  # persist corrected EN; keep the original
        p = Path(out) / "en.vtt"
        if p.exists() and not (Path(out) / "en.orig.vtt").exists():
            (Path(out) / "en.orig.vtt").write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        p.write_text(write_vtt(new_en), encoding="utf-8")
    lines = [
        f"# EN editorial review — {title}".rstrip(),
        "",
        "Standard proofread of the English script. Clear-error fixes were auto-applied to the "
        "EN (original saved as en.orig.vtt); the rest are held for you to adjudicate.",
        "",
    ]
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
    rpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
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


def cmd_localize(args) -> int:
    """EN subtitle -> zh-Hans/ja review-ready cues (translate + glossary + register +
    reflect + completeness gates + CJK re-segmentation), then a review bundle."""
    from .glossary import load as load_glossary
    from .localize import localize_en
    from .review import build_review_bundle
    from . import llm

    from .cue_builder import strip_sound_tags

    en = strip_sound_tags(_read_any(args.input, "en"))  # [music]/(applause) never reach translation
    glossary = load_glossary(args.glossary)
    langs = args.lang or ["zh-Hans", "ja"]
    from .vocab_prep import load_video_vocab

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
    cache = None if args.no_cache else str(out / "cache.jsonl")
    bank_for = _exemplar_bank_for(args, langs)
    en = _en_review(en, out, args.input)
    print(
        f"Localizing {args.input} -> {', '.join(langs)} (glossary={args.glossary}"
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
        cache_path=cache,
        condense=not getattr(args, "no_condense", False),
        native_editor=not getattr(args, "no_editor", False),
        logger=lambda _m: print(_m, flush=True),
    )
    _persist_term_pins(outputs, out)
    written = build_review_bundle(out, en, outputs)
    print(f"Wrote {len(written)} files to {out}/ (REVIEW.md + per-language VTT+SRT)")
    return 0


def _save_raw(transcript, path) -> None:
    from pathlib import Path as _P

    _P(path).write_text(
        json.dumps(
            [
                {"text": w.text, "start": w.start, "end": w.end, "confidence": w.confidence}
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
    from .cleanup import correct_transcript
    from .cue_builder import low_confidence_spans
    from .transcribe.retime import retime_corrected
    from . import llm

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
    from .audit import classify, word_diffs, write_review_xlsx
    from . import llm

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
    from .glossary import load as load_glossary
    from .localize import SentenceRecord, _write_status, llm_serialize_lock
    from .review import LangOutput, render_review_markdown
    from . import llm
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
        (Path(args.out) / "description.txt").write_text(Path(f).read_text(encoding="utf-8"), encoding="utf-8")


def _qe_second_judge(args):
    """The different-family second judge for the two-judge QE panel (Claude <-> GPT, see llm.py).
    Only with --qe-judge; None when disabled or no other family is installed (single judge).
    De-biases a model grading its own family's output (self-preference)."""
    if not getattr(args, "qe_judge", False):
        return None
    from . import llm

    if getattr(args, "single_judge", False):
        print(f"QE judge: single-judge ({llm.label()} only) — 2nd judge disabled via --single-judge")
        return None
    judge, what = llm.second_judge()
    if judge is None:
        print(f"QE judge: {what} — single-judge ({llm.label()}) DQF only")
        return None
    print(f"QE judge: two-judge panel ({llm.label()} + {what})")
    return judge


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
    now conversational is the default for multi-speaker content. Overridable via --conversational / --no-conversational."""
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


def cmd_run(args) -> int:
    """One entry: resolve an input (local file or YouTube URL) then do transcription,
    translation, or both. Media -> both by default; a subtitle input -> translate only."""
    from .glossary import load as load_glossary
    from .ingest import resolve_input
    from .localize import localize_en
    from .review import build_review_bundle
    from .transcribe import WhisperTranscriber
    from . import llm

    ri = resolve_input(args.input, workdir=str(Path(args.out) / "_ingest"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want_tx = not args.translate_only
    want_tr = not args.transcribe_only
    glossary = load_glossary(args.glossary)
    from .vocab_prep import load_video_vocab

    # Prefer explicit --vocab; else auto-load a vocab.json the transcribe phase persisted in
    # this out dir (the two-phase backlog flow: transcribe pins renderings, translate reuses).
    _auto_vocab = out / "vocab.json"
    if args.vocab:
        video_vocab = load_video_vocab(args.vocab)
    elif _auto_vocab.exists():
        video_vocab = load_video_vocab(_auto_vocab)
        print(
            f"Loaded per-video vocab from {_auto_vocab} "
            f"({sum(len(m) for m in video_vocab.get('renderings', {}).values())} renderings)."
        )
    else:
        video_vocab = {"names": []}
    names = [n.strip() for n in (args.names or "").split(",") if n.strip()]
    if names:  # operator-supplied names are DNT too
        video_vocab.setdefault("names", [])
        video_vocab["names"] = sorted(set(video_vocab["names"] + names))
    print(f"Resolved {args.input} -> {ri.kind} ({ri.title or ri.path})")

    # 1) EN master: from a subtitle input directly, else transcribe (+ clean-read correct) the media.
    if ri.kind == "subtitle":
        en = _read_any(ri.path, "en")
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
    else:
        recovery_second = None  # WhisperX drop-recovery pass, reused as the auto-consult 2nd ASR
        if not want_tx and want_tr:
            print("Media input needs transcription before translation; transcribing.")
            want_tx = True
        # Reuse a saved raw transcript so a glossary/vocab change can re-correct
        # WITHOUT re-running WhisperX; else transcribe and persist it.
        raw_p = out / "raw.json"
        if raw_p.exists() and not args.retranscribe:
            transcript = _load_raw(raw_p)
            print(f"Reusing saved raw transcript ({len(transcript.words)} words) — skip ASR")
        elif args.engine == "parakeet":
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
                    recovery_second = second  # reuse as the auto-caption consult's 2nd ASR
                    print(
                        f"  recovered {n} dropped word(s) Parakeet omitted"
                        if n
                        else "  no dropped spans found"
                    )
                except Exception as e:
                    print(f"  recovery skipped ({type(e).__name__}); Parakeet output kept")
            _save_raw(transcript, raw_p)
        elif args.engine == "assemblyai":
            # AssemblyAI cloud (Universal + diarization) — for public interview content.
            # A/B showed it isn't more COMPLETE than Parakeet, but transcribes conversational
            # JARGON right at the source (fewer real errors: 1 vs 4 on a 10-min interview), for
            # ~$0.26/hr. No drop-recovery needed. Key from $ASSEMBLYAI_API_KEY.
            from .transcribe import AssemblyAITranscriber

            print("Transcribing with AssemblyAI (cloud; diarized)…")
            transcript = AssemblyAITranscriber().transcribe(ri.path, language="en")
            _save_raw(transcript, raw_p)
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
        # Names as part of ingestion: metadata (video description) is reliable ->
        # auto-apply; transcript self-intros are candidates (ASR may mangle) -> hint.
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
        reliable = sorted(
            set(video_vocab.get("names", []) + meta_names + reg_names)
        )  # authoritative
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
        # Live vocab-prep (no YouTube metadata to lean on): identify the subject FROM
        # the transcript, pull source docs, and build per-video vocab. If it can't
        # identify the subject/docs, REQUEST a docs URL — do NOT silently proceed.
        _have_vocab = video_vocab and (
            video_vocab.get("identifiers") or video_vocab.get("renderings")
        )
        if not args.no_prep and not _have_vocab:
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
            if urls:
                print(
                    f"Subject: {subj.get('subject', '?')} — pre-ingesting vocab from {len(urls)} doc(s)…"
                )
                names0 = video_vocab.get("names", []) if video_vocab else []
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
                    video_vocab = vv
                    print(
                        f"  pre-ingested {len(vv['identifiers'])} identifiers, "
                        f"{sum(len(m) for m in vv['renderings'].values())} pinned renderings"
                    )
                except Exception as e:
                    print(f"  doc fetch failed ({type(e).__name__}); continuing on glossary only")
            else:
                if not subj.get("confident") and args.require_prep:
                    print(
                        "NEEDS INPUT (--require-prep): couldn't identify the video's subject or "
                        "find docs to pre-ingest vocab. Halting — re-run with --docs-url "
                        "<feature/library doc URL> or --vocab <file>."
                    )
                    return 2
                # No docs (typical for partner interviews). STILL pin per-video renderings for
                # CONSISTENCY: propose_renderings gives one canonical per recurring term, and
                # resolve_entity_renderings looks up brand/product names authoritatively
                # (Wikipedia). Needs no docs — just the transcript + LLM + web fetch. The
                # reviewer approves the choices; this stops per-video terms flip-flopping.
                names0 = video_vocab.get("names", []) if video_vocab else []
                try:
                    vv = build_video_vocab(
                        transcript.text,
                        docs_search=lambda q: "",
                        web_fetch=http_fetch,
                        translate=inv,
                        known=curated_forms(args.glossary),
                    )
                    vv["names"] = names0
                    video_vocab = vv
                    print(
                        f"No source docs — pinned {sum(len(m) for m in vv['renderings'].values())} "
                        f"per-video renderings for consistency (reviewer approves)."
                    )
                except Exception as e:
                    print(
                        f"  rendering-proposal failed ({type(e).__name__}); glossary + names only."
                    )
        # Persist per-video vocab so a later --translate-only run (the two-phase backlog flow)
        # reuses the pinned renderings/identifiers/names — else they're built at transcribe and
        # lost before translation. Auto-loaded below when --vocab isn't passed.
        if video_vocab and (
            video_vocab.get("renderings")
            or video_vocab.get("identifiers")
            or video_vocab.get("names")
        ):
            from .vocab_prep import save_video_vocab

            save_video_vocab(out / "vocab.json", video_vocab)
        raw_transcript = transcript  # keep pre-correction (has confidences) for QE
        # PRE-CLEAN-READ repeated-run recovery: collapse a repeated-token run ('the the the …') —
        # a genuine speaker stutter or a decoder loop — from the auto-caption BEFORE clean-read,
        # otherwise clean-read rewrites the run into duplicated real text the cue-level consult
        # can't detect.
        if ri.auto_captions:
            from .transcribe_qe import load_autosub, recover_repeated_runs

            try:
                transcript, _nh = recover_repeated_runs(transcript, load_autosub(ri.auto_captions))
                if _nh:
                    print(
                        f"  collapsed {_nh} repeated-token run(s) from auto-caption (pre-clean-read)"
                    )
            except Exception as e:
                print(f"  repeated-run recovery skipped ({type(e).__name__})")
        if not args.no_correct:
            transcript = _correct_transcript(
                transcript, glossary, video_vocab, names=video_vocab.get("names")
            )
            print(
                "Applied glossary-aware clean-read correction (jargon + grammar + filler + names)."
            )
        en = _maybe_clean(build_en_cues(transcript), args.clean)
        # PRUNE VOCAB TO THE CORRECTED TEXT: the vocab was harvested from the RAW ASR (above,
        # before clean-read), so an ASR mishear became a bogus keep-Latin term ('Chief Devilist'
        # <- 'Chief Evangelist'). Now that clean-read has fixed the captions, drop any term the
        # corrected EN no longer contains, and re-save — else the garble is forced, untranslated,
        # into the output.
        if video_vocab and video_vocab.get("terms"):
            from .vocab_prep import prune_terms_to_text, save_video_vocab

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
                flags = [
                    p["phrase"] for p in plausibility_suspects(llm.invoke(), en_text)
                ]
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
        # Reference-free QE: a timestamped suspect list for a human ear (no caption needed).
        if not args.no_qe:
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
    if not want_tr:
        return 0

    # 2) Translate the EN master.
    langs = args.lang or ["zh-Hans", "ja"]
    bank_for = _exemplar_bank_for(args, langs)
    en = _en_review(en, out, ri.title or ri.path)
    print(f"Localizing -> {', '.join(langs)}…")
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
        cache_path=str(out / "cache.jsonl"),
        condense=not getattr(args, "no_condense", False),
        native_editor=not getattr(args, "no_editor", False),
        logger=lambda _m: print(_m, flush=True),
    )
    _persist_term_pins(outputs, out)
    written = build_review_bundle(out, en, outputs)
    print(f"Wrote {len(written)} files to {out}/")
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dev_titles")
    p.add_argument(
        "--llm",
        choices=("claude", "codex", "openai"),
        default=None,
        help="primary model backend: claude (local Claude Code CLI, default), codex (local OpenAI "
        "Codex CLI, e.g. signed in with ChatGPT), openai (any OpenAI-compatible API via "
        "$OPENAI_API_KEY / $OPENAI_BASE_URL). Also $DEV_TITLES_LLM.",
    )
    p.add_argument("--llm-model", default=None, help="model id for the --llm backend (required for openai)")
    sub = p.add_subparsers(dest="cmd", required=True)

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

    rn = sub.add_parser(
        "run", help="one entry: input (file/YouTube URL) -> transcribe and/or translate"
    )
    rn.add_argument("input", help="local media/subtitle file OR a YouTube URL")
    rn.add_argument("--out", required=True, help="output directory")
    rn.add_argument(
        "--lang",
        action="append",
        default=None,
        help="target language (repeatable; default zh-Hans + ja)",
    )
    rn.add_argument("--transcribe-only", action="store_true", help="EN transcription only")
    rn.add_argument(
        "--translate-only", action="store_true", help="translation only (input is EN subs)"
    )
    rn.add_argument("--glossary", default="glossary/glossary.yaml")
    rn.add_argument("--vocab", default=None, help="per-video vocab JSON (vocab-prep output)")
    rn.add_argument(
        "--names",
        default=None,
        help="comma-separated speaker/proper names (DNT; ASR mishears corrected to these)",
    )
    rn.add_argument("--no-backcheck", action="store_true", help="skip the coverage edit-fix gate")
    rn.add_argument(
        "--exemplars",
        default="input/channel",
        help="human parallel-subtitle corpus dir to mine few-shot exemplars from",
    )
    rn.add_argument(
        "--no-exemplars",
        action="store_true",
        help="disable few-shot exemplars (prose-style directives only)",
    )
    rn.add_argument(
        "--tm",
        default="tm/tm.jsonl",
        help="translation-memory store (exact-match reuse + series exemplars; "
        "seeded from the human corpus if absent)",
    )
    rn.add_argument("--no-tm", action="store_true", help="disable the translation memory")
    rn.add_argument(
        "--qe-judge",
        action="store_true",
        dest="qe_judge",
        help="add the LLM MQM two-judge panel (primary model + a different-family judge if installed) "
        "scored via calibrated DQF; deterministic QE runs regardless",
    )
    rn.add_argument(
        "--single-judge",
        action="store_true",
        dest="single_judge",
        help="force single-model QE (disable the different-family 2nd judge) — use when "
        "codex/Bedrock is unavailable",
    )
    rn.add_argument(
        "--no-qe-repair",
        action="store_true",
        help="skip QE-guided auto-repair of judge-flagged majors (accept-only-if-better)",
    )
    rn.add_argument(
        "--conversational",
        action="store_true",
        help="casual/interview content: zh uses 你 (not formal 您); docs default keeps 您",
    )
    rn.add_argument(
        "--no-conversational",
        action="store_true",
        help="force formal register (zh 您) even if auto-detected as an interview",
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
        choices=["whisper", "parakeet", "assemblyai"],
        default="parakeet",
        help="primary ASR: parakeet (fast, local; default), whisper (WhisperX "
        "large-v3; slower on CPU), or assemblyai (cloud, diarized; for the "
        "PUBLIC interview backlog — cleaner jargon; needs $ASSEMBLYAI_API_KEY)",
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
    rn.add_argument(
        "--no-condense",
        action="store_true",
        help="ja: skip the condensation pass (on by default: tightens translations like a subtitle "
        "editor before QE, guarded; ~12%% shorter)",
    )
    rn.add_argument(
        "--no-editor",
        action="store_true",
        help="skip the native-editor pass (on by default: a native subtitle editor flags lines a "
        "viewer would notice and rewrites them; accepted only if a second judge agrees)",
    )
    rn.add_argument(
        "--description-file",
        help="video description text (names, roles, topic) for the per-video brief — for videos "
        "not on YouTube yet; published videos fetch it automatically",
    )
    rn.add_argument("--model-size", default="large-v3", dest="model_size")
    rn.add_argument("--device", default="cpu", help="cpu | cuda")
    rn.set_defaults(func=cmd_run)

    lo = sub.add_parser("localize", help="EN subtitle -> zh-Hans/ja review-ready cues + bundle")
    lo.add_argument("input", help="EN master subtitle (.vtt/.srt/.sbv)")
    lo.add_argument("--out", required=True, help="output bundle directory")
    lo.add_argument(
        "--lang",
        action="append",
        default=None,
        help="target language (repeatable; default zh-Hans + ja)",
    )
    lo.add_argument("--glossary", default="glossary/glossary.yaml")
    lo.add_argument("--vocab", default=None, help="per-video vocab JSON (vocab-prep output)")
    lo.add_argument(
        "--names", default=None, help="comma-separated speaker/proper names (kept Latin)"
    )
    lo.add_argument(
        "--no-backcheck",
        action="store_true",
        help="skip the source-anchored coverage edit-fix gate",
    )
    lo.add_argument(
        "--exemplars",
        default="input/channel",
        help="human parallel-subtitle corpus dir to mine few-shot exemplars from",
    )
    lo.add_argument(
        "--no-exemplars",
        action="store_true",
        help="disable few-shot exemplars (prose-style directives only)",
    )
    lo.add_argument(
        "--tm",
        default="tm/tm.jsonl",
        help="translation-memory store (exact-match reuse + series exemplars)",
    )
    lo.add_argument("--no-tm", action="store_true", help="disable the translation memory")
    lo.add_argument(
        "--qe-judge",
        action="store_true",
        dest="qe_judge",
        help="add the LLM MQM two-judge panel (primary model + a different-family judge if installed) "
        "scored via calibrated DQF; deterministic QE runs regardless",
    )
    lo.add_argument(
        "--single-judge",
        action="store_true",
        dest="single_judge",
        help="force single-model QE (disable the different-family 2nd judge) — use when "
        "codex/Bedrock is unavailable",
    )
    lo.add_argument(
        "--no-qe-repair",
        action="store_true",
        help="skip QE-guided auto-repair of judge-flagged majors (accept-only-if-better)",
    )
    lo.add_argument(
        "--conversational",
        action="store_true",
        help="casual/interview content: zh uses 你 (not formal 您); docs default keeps 您",
    )
    lo.add_argument(
        "--no-conversational",
        action="store_true",
        help="force formal register (zh 您) even if auto-detected as an interview",
    )
    lo.add_argument(
        "--no-condense",
        action="store_true",
        help="ja: skip the condensation pass (on by default: tightens translations like a subtitle "
        "editor before QE, guarded; ~12%% shorter)",
    )
    lo.add_argument(
        "--no-editor",
        action="store_true",
        help="skip the native-editor pass (on by default: a native subtitle editor flags lines a "
        "viewer would notice and rewrites them; accepted only if a second judge agrees)",
    )
    lo.add_argument(
        "--description-file",
        help="video description text (names, roles, topic) for the per-video brief — for videos "
        "not on YouTube yet; published videos fetch it automatically",
    )
    lo.add_argument("--no-cache", action="store_true", help="don't write a resumable cache")
    lo.set_defaults(func=cmd_localize)

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
        "--single-judge", action="store_true", help="primary-model-only DQF (skip the different-family 2nd judge)"
    )
    rs.set_defaults(func=cmd_rescore)

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

    args = p.parse_args(argv)
    from . import llm

    llm.configure(args.llm, args.llm_model)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
