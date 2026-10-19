# Contributing

## Dev setup
```bash
pip install -e '.[dev,cjk]'      # + [whisper]/[parakeet] for local ASR, [api]/[bedrock] for translators
pytest                           # full suite
```
See [DEVELOPMENT.md](DEVELOPMENT.md) for linting, coverage and docs.

## Pre-push checks

Enable the pre-push hook once per clone:

```bash
pip install bandit defusedxml
git config core.hooksPath hooks
```

Every `git push` then runs the test suite and a [Bandit](https://bandit.readthedocs.io) security
scan (medium severity and up) on `src/` and `scripts/`. The suite includes a check that no code
reads or writes fixed temp or home paths (`/tmp/...`, `/Users/...`): use `tempfile` or a path
inside the project. A reviewed false positive can be marked `# nosec <ID>`, with the reason in a
comment on the line above. `PRE_PUSH_SKIP=1 git push` bypasses the hook deliberately.

## Ground rules
- **Determinism first.** The harness is the deterministic scaffold around the LLM steps:
  segmentation, glossary/DNT enforcement, CJK re-segmentation, validation, and lossless guards.
  Prefer a deterministic, unit-tested check to another LLM call.
- **Every change ships with a test.** The suite is `test/test_*.py`, split by area, with shared
  helpers in `test/helpers.py`. Keep it green with `pytest`.
- **Never lose text in re-segmentation.** Re-timed/re-broken cues must losslessly reconstruct
  the translation. The lossless guards and their tests (`test/test_resegment.py`,
  `test/test_validate.py`) gate this.
- **Do not commit** source media or captions, translation memories (`tm/*.jsonl`, `*.tmx`),
  `output/` deliverables, or any credentials, tokens or cookies. See `.gitignore`.
- **Translation quality:** translate whole sentences (for context), keep the `reflect` pass, and
  enforce glossary/DNT terms through masking. Cue layout is a separate step and should not be
  folded into the translation call.

## Architecture
See `README.md` and `docs/PIPELINE-DESIGN.md`. The pipeline is: transcribe → merge into
sentences → translate → QE (two-judge DQF + auto-repair + term pins) → CJK re-segmentation →
validate → review bundle.
