# Changelog

All notable changes to dev-titles. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/) (0.x: the CLI and module APIs may still change).

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
