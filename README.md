# dev-titles

A subtitle localization toolkit for technical video. It turns an English video (or an
existing English subtitle file) into review-ready English, Simplified Chinese (`zh-Hans`)
and Japanese (`ja`) subtitles.

The goal is subtitles that are as good as possible *before* a human reviewer touches
them (post-editing), not to replace the reviewer. LLMs handle the language work. A
deterministic, unit-tested harness around them handles segmentation, terminology,
layout rules, validation and the guards that keep the LLM honest.

## What it does

1. **Transcribe English locally.** The default engine is
   [Parakeet](https://github.com/senstella/parakeet-mlx) (`parakeet-mlx`, Apple Silicon).
   It includes a WhisperX cross-check that splices back spans Parakeet silently drops on
   music or non-speech. [WhisperX](https://github.com/m-bain/whisperX) large-v3 works on
   any platform. AssemblyAI is an optional cloud engine with speaker diarization. A second
   ASR pass from a different model family can flag disagreements as suspects.
2. **Clean-read correction.** Fixes misheard jargon against the glossary, rejoins
   spelled-out identifiers, drops filler and false starts, and corrects small grammar
   slips. It keeps technical meaning, numbers and identifiers. Builds an English master
   (42 characters per line, about 17 characters per second, at most 2 lines).
3. **Translate EN to zh-Hans / ja** one whole sentence at a time, with the full transcript
   as context:
   - a **glossary**: hard/soft terms, do-not-translate (DNT) masking, avoid-forms, and
     sense-specific renderings,
   - **register**: Japanese polite form (です・ます); formal written Chinese (您), or
     `--conversational` for interviews,
   - a **translation memory** (exact-match reuse and series exemplars) plus few-shot
     exemplars mined from your own human-translated subtitles,
   - draft, reflect and refine, plus completeness gates (an omission retry and a coverage
     edit-fix pass).
4. **Quality estimation.** Deterministic checks always run: omissions, dropped
   numbers/identifiers, and inconsistent terms. With `--qe-judge`, a **two-judge MQM
   panel** (Claude plus an optional OpenAI/Codex judge from a different model family)
   scores the output on a calibrated DQF scale. It then **auto-repairs** segments the
   judges flagged as major, and keeps a repair only if a re-score shows it is better.
5. **CJK re-segmentation.** Cues are re-timed and re-broken for each language; English
   line breaks are never reused. It enforces per-language cps/cpl limits, Japanese kinsoku
   and phrase (bunsetsu) boundaries, Chinese full-width punctuation rules, and keeps
   multi-word terms and glossary targets on one line and in one cue.
6. **Japanese condensation.** Japanese has a much lower reading-speed budget than English.
   Over-long Japanese lines are tightened the way a subtitle editor would. The pass is
   guarded and can be turned off with `--no-condense`.
7. **Validation.** Every cue is checked against the language's rules (duration, gaps,
   lines, cps, cpl, punctuation). Errors block; warnings are flagged.
8. **Review bundle.** You get the EN master, a VTT and an SRT for each language, and
   `REVIEW.md`, which lists validation results, glossary enforcement, QE findings, coverage
   residuals and suspect cues.
9. **Optional YouTube publishing.** Replaces or adds caption tracks through the YouTube
   Data API (see [docs/YOUTUBE-API-SETUP.md](docs/YOUTUBE-API-SETUP.md)).

Runs are **resumable**: per-sentence results are cached in `cache.jsonl` inside the output
directory.

## Install

Requires Python 3.10+ and `ffmpeg` on `PATH` for media input.

```bash
git clone <this repo> dev-titles && cd dev-titles
pip install -e '.[cjk]'            # core + word-safe CJK line breaking
pip install -e '.[cjk,parakeet]'   # + Parakeet ASR (Apple Silicon only)
pip install -e '.[cjk,whisper]'    # + WhisperX ASR (cross-platform; ~4 GB of models on first run)
pip install -e '.[youtube]'        # + YouTube URL input / caption publishing
pip install -e '.[dev]'            # tests and linters
```

Other extras: `api` (Anthropic SDK), `bedrock` (Claude on AWS Bedrock), `aws` (boto3),
`search` (DuckDuckGo doc-grounding).

## Quick start

The installed console script is `dev-titles`. You can also run
`python -m dev_titles.cli`.

```bash
# One entry point: a local file or a YouTube URL.
# Media gets transcribed then translated; an EN subtitle file is translated only.
dev-titles run talk.mp4 --out bundle/
dev-titles run https://youtu.be/VIDEO_ID --out bundle/ --transcribe-only
dev-titles run talk.en.srt --out bundle/ --lang ja --qe-judge

# Useful run options
#   --engine {parakeet,whisper,assemblyai}   primary ASR (default parakeet)
#   --second-pass whisper|parakeet|<size>    2nd ASR; disagreements flagged as suspects
#   --lang zh-Hans --lang ja                 target languages (default: both)
#   --glossary glossary/glossary.yaml        shared glossary
#   --vocab vocab/talk.json                  per-video vocab (see below)
#   --names "Jane Doe,John Smith"            speaker names (kept as-is, ASR corrected to them)
#   --docs-url URL                           product docs used to ground spellings
#   --qe-judge / --single-judge              LLM QE panel / Claude-only panel
#   --conversational                         casual register for interviews
#   --no-condense, --no-tm, --no-exemplars, --no-qe-repair

# Primitives
dev-titles transcribe talk.mp4 -o talk.en.vtt --clean
dev-titles localize   talk.en.srt --out bundle/ --vocab vocab/talk.json
dev-titles validate   bundle/talk.ja.vtt --lang ja
dev-titles resegment  in.ja.vtt --lang ja -o out.ja.vtt
dev-titles convert    in.vtt -o out.srt               # vtt / srt / sbv / ttml
dev-titles rescore    --out bundle/                   # re-judge edited SRTs, refresh REVIEW.md
dev-titles audit      ours.en.srt human.en.srt --out audit.xlsx   # compare to a reference
dev-titles tm stats                                   # translation memory: seed/add/export/stats/import-tmx
dev-titles promote-terms --dry-run                    # promote recurring term pins to the glossary
```

For multi-hour jobs launched from an agent session, use `scripts/run_detached.py`. It
detaches the job into its own session so the job outlives the agent's turn.

## Glossary

`glossary/glossary.yaml` is the shared, cross-video terminology file. Only terms whose
`source` or an alias appears in a sentence are added to the translation prompt.
`enforce: hard` terms are also checked deterministically after translation.

```yaml
version: 0.1
languages: [zh-Hans, ja]
terms:
  - source: "React Native"          # DNT: same in every language
    zh-Hans: "React Native"
    ja: "React Native"
    enforce: hard
  - source: "in-app purchase"
    aliases: ["IAP", "in-app purchasing"]
    zh-Hans: "应用内购买"
    ja: "アプリ内課金"
    enforce: hard
    avoid: { zh-Hans: ["内购"] }    # known-wrong forms normalized to the canonical one
  - source: "update"
    zh-Hans: "更新"
    ja: "更新"
    enforce: soft                    # a preference, not a hard rule
    notes: "Prefer the native word over katakana アップデート in ja."
```

See `glossary/glossary.example.yaml` for the full schema, including `senses`,
`case_sensitive`, `style` and `identifiers`.

Terminology has two layers:

- **Shared glossary**: verified, cross-video vocabulary. The same term renders the same
  way everywhere, and the file grows over time.
- **Per-video vocab** (`--vocab`, built by `dev_titles.vocab_prep`): a discovery step.
  It finds a video's terms and proposes one rendering per language for terms that are not
  already shared. A reviewer approves them, then they are promoted to the shared glossary.
  The shared glossary always wins.

## LLM backends

| Role | Default | Alternatives |
|---|---|---|
| Translation, clean-read, condensation, judge #1 | the `claude` CLI ([Claude Code](https://docs.anthropic.com/claude-code)), which uses whatever auth it is configured with (Anthropic account or API key, or AWS Bedrock) | `AnthropicTranslator` (Anthropic SDK, `ANTHROPIC_API_KEY`) or `BedrockTranslator` (Claude on AWS Bedrock, standard AWS credentials) in `dev_titles.translate` |
| Judge #2 (different model family, optional) | the OpenAI **Codex CLI** (`codex exec`), auto-enabled if `codex` is on `PATH` (or `$CODEX_BINARY`) | `GptMantleJudge`: OpenAI models on AWS Bedrock via `$BEDROCK_API_KEY` / `$AWS_BEARER_TOKEN_BEDROCK` |
| ASR | local Parakeet / WhisperX (nothing leaves the machine) | AssemblyAI (`$ASSEMBLYAI_API_KEY`), AWS Transcribe JSON via `buildcues` |

Override the `claude` binary with `$CLAUDE_BINARY`. Credentials are always read from the
environment or from each tool's own config, never from the repo. If a required credential
is missing, the tool exits with a clear message.

## Development

```bash
pip install -e '.[dev,cjk]'
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [DEVELOPMENT.md](DEVELOPMENT.md). The design
is described in [docs/PIPELINE-DESIGN.md](docs/PIPELINE-DESIGN.md). A Claude Code skill
that drives the pipeline lives in
[`.claude/skills/subtitle-localizer/SKILL.md`](.claude/skills/subtitle-localizer/SKILL.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
