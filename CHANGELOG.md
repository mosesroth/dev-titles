# Changelog

All notable changes to dev-titles. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/) (0.x: the CLI and module APIs may still change).

## [Unreleased]

### Added
- zh second-person referent pass: a "you" that means the guest's team (你们) or the viewers (大家),
  or a generic possessive "your" (将您的应用 -> 将应用), no longer comes out as the 您 reserved for
  the one person addressed. Guarded (only pronoun characters may change) and idempotent.

- `sound_tag` validation error ([music], (applause), ♪, [音乐] / ［音楽］; a ja （笑） is a warning) and
  `scripts/check_upload.py`, which validates every subtitle file in an upload folder before
  handover. A subtitle input to `run` is now stripped of sound tags before it becomes the EN master.

- Sound-tag detection is vocabulary-based ([MUSIC PLAYING], (light music continues), [applause and
  cheering], [speaking Japanese]…; UI names like [Build] untouched); YouTube `>>` speaker markers are
  stripped and flagged; the review bundle strips both before writing any file; the EN publish script
  refuses a track that has either.
- ja `notes_conversational` (spoken register for interviews), closing-thanks rules (ja/zh), React /
  Linux Foundation names; the zh pronoun pass also checks "you all / you guys" rendered 大家.
- ja validation: lines 20-22 wide are a warning (`hard_cpl`), over 22 an error; 19 stays the
  segmenter's target.

- **Sync with the speech** (`validate/speech_sync.py`): EN cue starts are checked against word-timed
  speech (our ASR words, else the video's YouTube auto-captions). More than 2 s late is an error.
  `run`/`localize` re-time the EN when needed (text unchanged), and the EN gate blocks translation
  while cues are still off. `scripts/retime_en_to_speech.py` handles existing tracks.
- **zh/ja vs the speech, by content** (`validate/content_sync.py`): each translated cue is mapped
  (one cached LLM call) to the EN sentences it carries, and those are timed by the speech. A run of
  2+ cues more than 2 s off is an error. Runs in `check_upload.py` and after `localize`. EN more than
  5 s early is now an error too.
- **Silence inside one EN cue** (more than 3 s) is an error: part of the text shows early. The
  re-timer splits such a cue at the silence.
- **zh/ja auto re-time** (`retime_target_to_speech`, also `scripts/retime_target_to_speech.py`):
  out-of-step runs move into their spoken window. Runs after translation.
- **A sign-off survives a timing-only change** (text-only hash).
- **New line rules:** `punct_line_start`, `line_whitespace`, `space_before_punct`.
- **`zh_line_end_punct` rule** (CN style guide: no ，、； at a line end).
- **`duplicate_text` rule:** the same 9+ word run within 3 cues is an error; 6–8 words is a warning.
- **The EN gate blocks mechanical defects** (sound tags, `>>`, duplicates, overlaps, line count/width,
  speech sync) even when the EN is signed off.
- **The cue builder rebalances an orphan** into two 2-line cues before using a third line.
- **`check_upload.py` also checks** EN vs speech, translation drift vs EN, and parity with the
  library copy.
- **A PostToolUse hook** (`scripts/hooks/check_deliverables_hook.py`) validates subtitle files
  written outside the pipeline.

### Fixed
- Translation prompts: per-line speaker / meaning notes were appended to the end of the
  whole-transcript reference block, nothing marked which line was being translated, and the
  reflect pass saw neither. They now sit beside the line in both prompts, with the two lines
  before and after it (`TranslationRequest.line_context`), so a correct note is actually followed.

### Changed
- `translate_subtitle` split into cache / reuse / fresh-translation / cue-building helpers (no
  behavior change; unit tests for cache reuse and the retry path).

## [0.2.2] — 2026-10-23

### Changed
- Internal refactor, no behavior change: the localization pipeline (a 726-line function) is now a
  driver over per-stage functions sharing run / per-language state; the CLI's `run` (378 lines),
  the EN review and the argument parser are split the same way, and `run` / `localize` share one
  translate step and one definition of their 17 common options. Checked against the previous
  version with a fake model: identical subtitles, side files, cache, QE state, stdout and
  parsed arguments.

### Fixed
- An EN-review fix on a `localize` run whose output dir had no `en.vtt` saved the corrected EN
  without `en.orig.vtt`, so a re-run translated the uncorrected input. The original is now kept.

## [0.2.1] — 2026-10-23

### Fixed
- Python 3.10: the Codex judge's model / provider settings were silently dropped, because `tomllib` is
  3.11+. It now falls back to the `tomli` backport, a dependency on 3.10 only (caught by the new CI's
  3.10 job).

## [0.2.0] — 2026-10-23

### Added
- **English sign-off gate**: nothing is translated until the EN review ran and the exact English is
  signed off (`en_approved.json`, bound to a hash of the EN; `dev-titles approve-en`; no bypass flag).
  A blocked run writes `<out>/EN_LISTEN.html`: one numbered item per held line, a YouTube link that
  opens 2 s before it, the flagged words highlighted, the reviewer's note and what Parakeet heard.
- **Deletion-only clean-read** for unscripted speech (interviews, chats, panels): stutters, false
  starts, self-corrections and filler removed; rewording is rejected by construction. Scripted videos
  are left as written; `--en-style` overrides the call.
- **Meaning notes**: idioms and ambiguous phrases resolved once in the English and attached to every
  language's prompt and to the native editor.
- **Sync check** (`validate/sync.py`) and `scripts/retime_drifted.py` to detect and re-time a track that
  drifted from the speech, without changing its text.
- `scripts/check_vendor_review.py`, `scripts/triage_held_en.py`, `scripts/cleanread_en.py`;
  `scripts/eval/ab_vs_human.py`. Evaluation scripts moved to `scripts/eval/`.
- Native-editor pass with a measured acceptance gate and a video brief; reply check at speaker turns;
  speaker labels; names policy (official localized partner names locked in `glossary/partners.yaml`).
- `--llm claude|codex|openai` (primary model choice); `--engine auto` ASR selection.
- CI (GitHub Actions: black, isort, flake8, mypy, Bandit, tests on Python 3.10–3.12), SECURITY.md,
  issue and pull-request templates. The pre-push hook runs the same checks.

### Changed
- zh: 您 for direct address, the generic "you" omitted, spoken register for interviews; ja: reply
  renderings (なるほど / はい), ご参加 vs ご視聴, hedges kept, no added material.
- Glossary: Hook, 词元 (AI token), Brownfield, 网页, 设备端, 服务器端, 服务器函数, schema, footprint.
- CJK segmentation breaks where a subtitler would (sentence / clause ends, balanced lines, whole
  katakana words, no particle at a line start); condensation never touches one-line sentences.
- `run` records `SOURCE-URL.txt`; re-runs translate the corrected `en.vtt` the EN review saved.
- Lint tools are pinned in the `dev` extra so local checks and CI agree.

### Fixed
- Held-out evaluation excluded nothing for `<dir>/en.vtt` inputs; editor examples honor
  `SUBTITLE_HOLDOUT_VIDEOS`.
- A crashed EN review no longer lets translation proceed.
- The `.xlsx` audit test skips cleanly without the optional `[xlsx]` extra.

## [0.1.0] — 2026-10-16
- First public release: transcription (Parakeet / WhisperX / AssemblyAI), clean-read, glossary + TM +
  exemplars, two-judge MQM QE with auto-repair, CJK re-segmentation, Japanese condensation,
  validation, review bundle.
