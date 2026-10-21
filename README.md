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
   any platform. `--engine auto` (the default) picks Parakeet on Apple Silicon when it is
   installed, else WhisperX. [AssemblyAI](https://www.assemblyai.com) is an optional cloud
   engine with speaker diarization; it uploads the audio and uses your own AssemblyAI account
   (it has a free plan). A second
   ASR pass from a different model family can flag disagreements as suspects.
2. **Clean-read correction.** Fixes misheard jargon against the glossary, rejoins
   spelled-out identifiers, drops filler and false starts, and corrects small grammar
   slips. It keeps technical meaning, numbers and identifiers. Builds an English master
   (42 characters per line, about 17 characters per second, at most 2 lines).
3. **Translate EN to zh-Hans / ja** one whole sentence at a time, with the full transcript
   and a **video brief** as context. The brief names the speakers, their roles, the topic and
   any context-dependent jargon. It is built from the video description (saved at ingest,
   fetched with `yt-dlp`, or supplied with `--description-file`) plus the whole transcript:
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
   line breaks are never reused. Lines break at sentence ends, then clause ends, with balanced
   lines; cues end at sentence ends; short neighbouring sentences share a subtitle when they
   still fit two lines, and short cues stay on screen into the following silence (about 3 s a
   line). Two speakers never share a subtitle. It enforces per-language cps/cpl limits,
   Japanese kinsoku (moving a whole word, never splitting one) and phrase (bunsetsu)
   boundaries, Chinese full-width punctuation rules, and keeps names and glossary targets on
   one line and in one cue. `scripts/seg_eval.py` scores segmentation against a reviewer's
   corrected tracks.
6. **Japanese condensation.** Japanese has a much lower reading-speed budget than English.
   Only multi-line sentences faster than about 10 characters a second are tightened, the way a
   subtitle editor would; one-line sentences get reading time from merging and lingering
   instead, because trimming them only removes politeness. Names, numbers, thanks and
   politeness are protected. Turn it off with `--no-condense`; measure it with
   `scripts/condense_benchmark.py`.
7. **Native-editor pass.** A reference-free native subtitle editor flags lines a native viewer
   would notice: literal idioms, odd terms, wrong register, run-ons, context misreads. It
   proposes a minimal rewrite for each. A rewrite lands only if:
   - every Latin token and number survives, and it is not much longer;
   - the second judge (pairwise, randomized order) says it keeps the meaning and reads better;
   - neither judge, rating the current and the suggested line side by side, rates it worse.

   A suggestion the pairwise judge rejected is still applied if *every* judge rates it better.
   Measured against human-reviewed subtitles, that rescue was almost never wrong. QE repair
   later can't overwrite an editor rewrite unless both judges flag it as a major error.
   Turn the pass off with `--no-editor`.

   **Speaker turns.** Who says each sentence comes from ASR diarization when the engine
   provides it, else one LLM pass guided by the video brief. The translator sees the turns,
   and at each speaker change a **reply check** fixes replies translated literally ("Brilliant."
   answering thanks is not 素晴らしい; "Thank you for having me" → こちらこそ…). A reply fix is
   accepted only if every judge, shown the exchange, prefers it.
   **Names policy.** Partner company names use the partner's own official localized name where its
   website brands itself that way, else stay in Latin; a product named with an ordinary word gets its
   official rendering from the partner's site, else one consistent per-video choice. Each partner is
   looked up once and locked in `glossary/partners.yaml` (edit by hand to correct).
8. **Validation.** Every cue is checked against the language's rules (duration, gaps,
   lines, cps, cpl, punctuation). Errors block; warnings are flagged.
9. **Review bundle.** You get the EN master, a VTT and an SRT for each language, and
   `REVIEW.md`, which lists validation results, glossary enforcement, QE findings, coverage
   residuals and suspect cues.
10. **Optional YouTube publishing.** Replaces or adds caption tracks through the YouTube
   Data API (see [docs/YOUTUBE-API-SETUP.md](docs/YOUTUBE-API-SETUP.md)).

Runs are **resumable**: per-sentence results are cached in `cache.jsonl` inside the output
directory.

## Install

Requires Python 3.10+ and `ffmpeg` on `PATH` for media input.

```bash
git clone https://github.com/mosesroth/dev-titles && cd dev-titles
pip install -e '.[cjk]'                   # core + word-safe CJK line breaking
pip install -e '.[cjk,parakeet,whisper]'  # Apple Silicon: Parakeet ASR + WhisperX drop recovery
pip install -e '.[cjk,whisper]'           # any platform: WhisperX ASR (~4 GB of models on first run)
pip install -e '.[cjk,assemblyai]'        # cloud ASR instead: AssemblyAI ($ASSEMBLYAI_API_KEY)
pip install -e '.[youtube]'               # + YouTube URL input (yt-dlp) and caption publishing
pip install -e '.[dev]'                   # tests and linters
```

Other extras: `api` (Anthropic SDK), `bedrock` (Claude, and the optional GPT judge, on AWS
Bedrock), `aws` (boto3), `xlsx` (.xlsx term-base import and the `audit` report), `qe`
(CometKiwi neural QE), `search` (DuckDuckGo doc-grounding).

### What you need to sign in to

You need at least one model backend. Everything else is optional.

| For | Install | Sign in / configure |
|---|---|---|
| **Translation and judging (required, pick one)** | [Claude Code](https://code.claude.com) (default), or the [Codex CLI](https://github.com/openai/codex), or nothing for an OpenAI-compatible API | Claude Code: `claude` once to sign in (Claude subscription, `ANTHROPIC_API_KEY`, or Bedrock/Vertex). Codex: `codex login` (ChatGPT account or API key). API: `$OPENAI_API_KEY` / `$OPENAI_BASE_URL`. See [LLM backends](#llm-backends) |
| Second judge (recommended) | the *other* CLI: Codex if your primary is Claude, Claude Code if it's GPT | its own sign-in, as above |
| Local speech recognition | `parakeet` / `whisper` extras, `ffmpeg` | nothing (models download from Hugging Face on first run, no account) |
| AssemblyAI (optional cloud ASR) | `assemblyai` extra | `$ASSEMBLYAI_API_KEY` from your AssemblyAI account (free plan available) |
| YouTube URL input | `youtube` extra | nothing for public videos |
| YouTube caption publishing | `youtube` extra | a Google Cloud OAuth client and a one-time sign-in by the channel owner: [docs/YOUTUBE-API-SETUP.md](docs/YOUTUBE-API-SETUP.md) |
| CometKiwi QE (optional) | `qe` extra | the model is gated: accept its license on Hugging Face, then `hf auth login` (older: `huggingface-cli login`) |
| Claude or GPT on AWS Bedrock (optional) | `bedrock` extra | standard AWS credentials; the GPT judge uses `$BEDROCK_API_KEY` / `$AWS_BEARER_TOKEN_BEDROCK` |

Windows: the pipeline runs, but Parakeet is macOS-only and the shared job lock is skipped,
so avoid running two translation jobs at once.

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
#   --engine {auto,parakeet,whisper,assemblyai}   primary ASR (default auto)
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

### Learning from human reviews

Once reviewers have corrected some of your subtitles, these scripts use their corrections:

- `scripts/mine_review.py` mines before/after pairs (machine draft vs reviewer's final) and
  keeps the generalizable ones as few-shot examples for the native editor
  (`glossary/editor_examples.<lang>.jsonl`). It also writes a report of recurring
  substitutions, which are candidate glossary rules.
- `scripts/benchmark.py` grades two runs (OLD/NEW) against the human subtitles, pairing each
  sentence across runs, as OK / NOTICEABLE / MEANING.
- `scripts/gate_benchmark.py`, `gate_compare.py` and `gate_rescue.py` measure the editor's
  acceptance gates against human reviews. Each suggestion is graded before and after by two
  graders from different model families, and the candidate gates are replayed. The current
  gate was chosen this way.

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

Every model call is "prompt in, text out", so you can pick the model that does the work:

```bash
dev-titles run talk.mp4 --out bundle/                               # Claude (default)
dev-titles --llm codex run talk.mp4 --out bundle/                   # GPT via the Codex CLI
dev-titles --llm openai --llm-model gpt-5.5 run talk.mp4 --out bundle/
OPENAI_BASE_URL=http://localhost:11434/v1 dev-titles --llm openai --llm-model qwen3 run …   # local model
```

| `--llm` | Uses | Sign-in |
|---|---|---|
| `claude` (default) | the `claude` CLI ([Claude Code](https://docs.anthropic.com/claude-code)) | whatever Claude Code is signed in with: a Claude account, an Anthropic API key, or AWS Bedrock |
| `codex` | the OpenAI [Codex CLI](https://github.com/openai/codex) (`codex exec`) | whatever Codex is signed in with: a **ChatGPT account** (`codex login`), an OpenAI API key, or a provider configured in `~/.codex` |
| `openai` | any OpenAI-compatible chat API: OpenAI, Gemini's compatible endpoint, Ollama, LM Studio, vLLM… | `$OPENAI_API_KEY` (not needed for local servers); `$OPENAI_BASE_URL` (default `https://api.openai.com/v1`); `--llm-model` is required |

You can also set the backend with `$DEV_TITLES_LLM` and `$DEV_TITLES_LLM_MODEL`. Override the
binaries with `$CLAUDE_BINARY` / `$CODEX_BINARY`. From Python, `AnthropicTranslator` (Anthropic SDK)
and `BedrockTranslator` (Claude on AWS Bedrock) are also available in `dev_titles.translate`.

**Second judge.** With `--qe-judge`, the second judge is always from a *different* model family
than the primary: Claude primary → GPT judge via Codex; GPT primary → Claude judge. If the
other family isn't installed, the panel runs with the primary model alone. That still works,
but the editor's acceptance gate was measured with two families checking each other, and one
model judging its own suggestions is likely to let more bad edits through.

**Quality.** The prompts, thresholds and acceptance gates were tuned and measured with Claude
as the primary model. GPT and other models are expected to work but are unmeasured. Small
local models may struggle with the long, style-heavy prompts.

| Role | Default | Alternatives |
|---|---|---|
| ASR | local Parakeet / WhisperX (nothing leaves the machine; no account) | AssemblyAI (`--engine assemblyai`, your own account via `$ASSEMBLYAI_API_KEY`; free plan available; uploads the audio), AWS Transcribe JSON via `buildcues` |

Credentials are always read from the environment or from each tool's own config, never from
the repo. If a required credential is missing, the tool exits with a clear message.

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
