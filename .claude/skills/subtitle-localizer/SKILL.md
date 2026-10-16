---
name: subtitle-localizer
description: Localize video subtitles from English into Simplified Chinese and Japanese at review-ready quality. It transcribes, translates with glossary + register + reflect, re-segments for CJK, validates, and packages the result for review. Use when asked to generate or translate subtitles/captions for a video, re-segment CJK subtitles, validate a subtitle file against per-language rules, or run the localization pipeline.
---

# Subtitle Localizer

This skill carries the **judgment** for the subtitle-localization pipeline. The deterministic
steps live in the `dev_titles` Python harness at the repo root. The skill decides *what* to
run, *what to ask the operator*, and *how to read the results*.

## Engines (what actually runs)
- **ASR:** local and nothing leaves the machine.
  - `--engine parakeet` (default; `parakeet-mlx`, Apple Silicon only) is fast and gets technical
    proper nouns right. Parakeet (RNNT/TDT) can silently **drop** whole spans on music or
    non-speech, so `run` adds a cheap WhisperX cross-check (`--recover-model`, default `base`)
    that splices only the dropped spans back. `--no-recover` disables it.
  - `--engine whisper` (WhisperX large-v3) is cross-platform, slower on CPU, and more robust on
    heavy conversational or interview audio.
  - `--engine assemblyai` is cloud-based with speaker diarization and needs `$ASSEMBLYAI_API_KEY`.
  - `--second-pass whisper|parakeet` runs a different-family ASR and flags disagreements, which
    catches the primary engine's confident mishears.
- **Translation / clean-read / condensation:** the `claude` CLI (`CliTranslator`). The SDK
  translators (`AnthropicTranslator`, `BedrockTranslator`) are available for library use.
- **QE:** deterministic checks always run. `--qe-judge` adds a two-judge MQM panel (Claude plus
  the Codex CLI as a different-family judge when `codex` is installed) scored as calibrated
  DQF (0–100, gate 96, zero criticals). Findings are unioned and severity is the max across
  judges. Judge-flagged majors are auto-repaired, and a repair is accepted only if it improves
  the score. `--single-judge` forces Claude only.

## Pipeline
1. **Transcribe** EN → word-timed transcript. Filler (um/uh) is always dropped. **Clean-read
   correction** (on by default) fixes misheard jargon against the glossary, rejoins spelled-out
   identifiers ("TV focus guide view" → `TVFocusGuideView`), fixes small grammar slips, and
   drops discourse filler and false starts. It keeps technical meaning, numbers and identifiers;
   it is not verbatim. Raw non-speech tokens (`[music]`, ♪) are dropped. Following DCMP, sound
   is captioned only when essential, as a lowercase descriptive bracket in its own cue.
2. **EN master:** 42 CPL / 17 cps / ≤2 lines, DCMP number formatting.
3. **Translate** whole sentences with full-transcript context: glossary terms retrieved per
   sentence, translation memory and few-shot exemplars (from `--exemplars`, a folder of
   human-translated subtitle pairs), register (**ja** です・ます; **zh-Hans** formal written 您,
   or `--conversational` for interviews), reflect-then-refine, and completeness gates (omission
   retry and coverage edit-fix; residuals are listed in `REVIEW.md`, never dropped silently).
4. **Enforce** hard glossary terms and avoid-forms deterministically, and mask DNT terms.
5. **ja condensation** (on by default; `--no-condense`): tightens over-long Japanese like a
   subtitle editor would, under guards.
6. **QE** as above.
7. **Re-segment for CJK.** Never reuse EN line breaks or timing. Apply per-language CPL/cps,
   JA kinsoku and bunsetsu boundaries, ZH full-width punctuation, and keep multi-word Latin
   terms and glossary targets together on one line and in one cue.
8. **Validate** every cue. Errors block; warnings flag.
9. **Deliver** a review bundle: EN master, VTT+SRT for each language, and `REVIEW.md`.
   Optionally publish to YouTube with `scripts/publish_en_youtube.py`.

## Running
```bash
python -m dev_titles.cli run talk.mp4 --out bundle/                 # transcribe + translate
python -m dev_titles.cli run talk.en.srt --out bundle/ --qe-judge   # translate only
python -m dev_titles.cli run https://youtu.be/ID --out bundle/ --transcribe-only
python -m dev_titles.cli localize talk.en.vtt --out bundle/ --vocab vocab/talk.json
python -m dev_titles.cli validate  out.ja.vtt --lang ja
python -m dev_titles.cli resegment in.ja.vtt --lang ja -o out.ja.vtt
python -m dev_titles.cli rescore --out bundle/      # after a reviewer edits the SRTs
```
Every `run`/`localize` is **resumable**: re-running the same command skips cached sentences.

**Long jobs must be detached.** Transcription and translation can take 30 minutes to 2 hours.
Claude Code's `run_in_background` tasks live in the session's process group and are killed
when the launching turn ends. Instead, launch with
`python3 scripts/run_detached.py --log run.log -- <cmd…>`. It `setsid()`s into its own
session. There is no completion callback, so poll the log.

## Terminology (two layers)
- **Shared glossary** (`glossary/glossary.yaml`): verified, cross-video terms such as brand and
  product names, DNT rules, and style notes. The same term always renders the same way.
- **Per-video vocab** (`dev_titles.vocab_prep`, `--vocab <file>.json`): discovery for a new
  video. It finds the terms and, for ones not already shared, proposes one rendering per language.
  The loop is **propose → reviewer approves → promote** (`promote-terms`, or
  `promote_renderings_to_glossary`). The shared glossary always wins.
- **Doc grounding:** without metadata, `run` identifies the video's subject from a first-pass
  transcript and fetches product docs, either from `--docs-url` or from a DuckDuckGo search
  (`pip install ddgs`; `--no-search` opts out). If it prints `NEEDS INPUT: re-run with
  --docs-url`, **ask the operator for the docs URL** rather than guessing. Video-specific APIs
  and libraries get mis-transcribed without it.

## Credentials: ask, never guess
Local ASR needs no key. Translation and judging use the `claude` / `codex` CLIs with whatever
auth they are configured with. Optional keys (`$ASSEMBLYAI_API_KEY`, `$ANTHROPIC_API_KEY`,
`$BEDROCK_API_KEY` / `$AWS_BEARER_TOKEN_BEDROCK`) are read from the environment only. If a step
fails for missing auth, **stop and ask the operator** to log in or supply the key. Never
hardcode, invent, or commit a credential.

## Quality bar
Native-speaker reviewers should rate the output "publishable with light edits" or better. QE
flags problems and auto-repair fixes what it can, but reviewers give the final word.
