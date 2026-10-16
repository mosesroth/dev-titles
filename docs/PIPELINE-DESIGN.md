# Pipeline & Packaging Design

## Form factor: skill + harness
The toolkit has two layers that solve different halves of the problem:

- **Claude Code skill** (`subtitle-localizer`) carries the *judgment*: the procedure, when to
  ask for docs or credentials, register choices, when to condense, and how to read the QE
  scores. Anyone with an agent can use it.
- **Python harness** (`dev_titles/`) carries the *deterministic execution*: ASR adapters,
  glossary retrieval and enforcement, CJK re-segmentation and kinsoku, per-language validation,
  VTT/SRT/TTML generation, QE scoring, and YouTube publishing. The skill calls it, and it can
  also run standalone or in batch over a back catalog.

The split follows a common pattern: the procedure document carries judgment and the CLI
carries execution.

## End-to-end flow
```
source: local media file, EN subtitle file, or YouTube URL
       │
   [1] TRANSCRIBE (local Parakeet / WhisperX; optional AssemblyAI)
       → EN transcript with word timestamps
       → optional 2nd-ASR cross-check (different model family) → suspect list
       → glossary-aware clean-read correction
       │
   [2] EN CUE BUILD → EN WebVTT master (42 CPL / 17 cps, 2 lines)
       │
   [3] TRANSLATE per target lang (Claude):
        • merge cues → whole sentences; full-transcript context
        • retrieve glossary terms present in the sentence + translation memory + exemplars
        • register: JA teineigo / ZH formal written (您)
        • reflect-then-refine (accuracy/fluency/style/terminology)
        • completeness gates (omission retry + coverage edit-fix)
       │
   [4] DETERMINISTIC TERM ENFORCEMENT (hard glossary terms, avoid-forms, DNT masking)
       │
   [5] JA CONDENSATION (reading-speed driven, guarded)
       │
   [6] QE: deterministic checks + optional two-judge MQM panel scored as DQF
        • judge-flagged majors → auto-repair, accepted only if the re-score improves
       │
   [7] CJK RE-SEGMENT (do NOT reuse EN cues):
        • re-derive events to per-lang CPL + cps
        • JA kinsoku shori + bunsetsu boundaries; ZH full-width punct + single-space rule
        • keep multi-word terms / glossary targets together
       │
   [8] VALIDATE (per-language checklist) → auto-fix / flag
       │
   [9] EMIT WebVTT (canonical) + SRT per language
       │
   ├─→ REVIEW BUNDLE (default deliverable): per-language VTT+SRT + REVIEW.md
   │    (validation + glossary enforcement + QE summary) for the reviewers.
   └─→ PUBLISH to YouTube (optional): captions update/insert per language.
```

## Repo layout
```
dev-titles/
  docs/                      # design + setup notes
  glossary/
    glossary.example.yaml    # schema
    glossary.yaml            # shared terminology
  src/dev_titles/
    config.py                #   per-language limits + kinsoku/zh-punct tables
    model.py textwidth.py    #   Cue/Subtitle model + display width
    linguistics.py           #   joiner + break-point maps + repetition detection
    glossary.py              #   load / retrieve / enforce
    segment/ resegment.py    #   CJK re-segmentation (boundary-aware) + kinsoku + cps
    validate/                #   per-language rule checks + validity_report
    formats/                 #   vtt / srt / sbv / ttml
    transcribe/              #   Parakeet, WhisperX, AssemblyAI, AWS Transcribe JSON
    translate/               #   Claude CLI, Anthropic API, Bedrock translators + prompts
    eval/                    #   MQM/DQF judges (Claude, Codex CLI, GPT on Bedrock)
    localize.py              #   sentence-level translate + gates + QE-repair orchestration
    publish/                 #   YouTube publisher
    cli.py                   #   run / transcribe / localize / validate / resegment / convert / …
  scripts/                   # operational helpers (detached runs, YouTube publish, re-wrap)
  .claude/skills/subtitle-localizer/SKILL.md   # the skill
```

## Engine A/B (the eval harness picks winners)
Run the **same** 3–5 pilot clips through each candidate, score them with the QE harness, and
rank at the system level:
- **Translation:** Claude is the primary. Other LLMs or MT engines (glossary-enabled) can be
  compared through the same `Translator` interface.
- **Transcription:** Parakeet vs WhisperX vs AssemblyAI, with a WER spot-check on jargon.
- Freeze the winning config for each language, and re-run when models update.

## Validation checklist (harness enforces per language)
> Target platform is assumed to be **YouTube**, which auto-wraps and imposes no hard CPL/cps
> limit or vertical/ruby layout. The cps caps are the real quality bar because they track human
> reading speed. The CPL caps are conservative readability targets: enforced strictly before
> review, and relaxable later.

- **All:** ≤2 lines; 0.83s ≤ dur ≤ 7s; gap ≥ 2 frames; no bad shot-change straddle; breaks at grammatical boundaries.
- **EU (EN/FR/DE/ES/IT):** ≤42 CPL; cps ≤ 17 (EN 20); char = 1.
- **ZH:** ≤16 CPL; cps ≤ 9; single full-width space instead of ，。; ellipsis U+2026; no ！？／？？／！！; full-width = 1, half = 0.5.
- **JA:** ≤13 CPL horiz (11 vert); cps ≤ 4; min dur 500ms; kinsoku (forbid 、。」）】！？ small-kana/長音 at line start; 「（【 at line end); full-width = 1, half = 0.5.
