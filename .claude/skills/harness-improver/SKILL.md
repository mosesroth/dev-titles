---
name: harness-improver
description: One supervised improvement cycle of the subtitle harness — find the biggest recurring gap in the latest A/B panel / learn reports / upload check, make ONE prompt, code or data change on a branch, prove it with a spot A/B, and report. Never touches deliverables, never pushes.
---

# Harness improver (one cycle)

You are improving the subtitle-localization harness in THIS git worktree (branch `agent/<date>`).
A person reviews your branch and report; you never merge, push, or write deliverables.

## Hard rules
- Never write under `$DEV_TITLES_OUTPUT/FINAL_SUBTITLES`, `$DEV_TITLES_OUTPUT/UPLOAD_*`, `$DEV_TITLES_OUTPUT/ticket` — the
  runner snapshots them and will restore and report any change.
- Never `git push`, never merge, never edit `.claude/settings*`, never disable a test or a check.
- ONE change per cycle (a prompt, a function, a style note, glossary entries — one idea). Small diffs.
- English subtitle TEXT and reviewer style calls (your native reviewers) are the user's decisions:
  report them, don't encode them as rules unless the evidence spans >= 2 videos.
- Every claim in the report cites rows (video, lang, sentence #) — no hand-waving.

## The cycle
1. **Read the evidence** (paths are in the task prompt): the panel's
   `<run>/vs_human/REPORT.md` + `ab_rows.json` (rows where the human final was better, with the judges'
   reasons), the learn-job reports (`$DEV_TITLES_OUTPUT/learn/<panel>_*/REPORT.md`: lessons accepted /
   rejected / needing a person), and `scripts/check_upload.py` output if given.
2. **Find the biggest recurring gap**: group the human-better rows by mechanism (not by surface
   word). A mechanism must appear in >= 2 videos or >= 4 rows. Read the code that produces it
   (`src/dev_titles/translate/prompts.py`, `localize.py`, `editor.py`, `glossary/glossary.yaml` …).
   Prefer a ROOT-CAUSE fix (how the prompt is built, what context a step sees) over another rule.
3. **Make the change** on the branch. Add or update a unit test when you change code.
4. **Check**: `python3 -m black --check src scripts test`, `python3 -m isort --check-only src scripts test`,
   `python3 -m flake8 src scripts test`, `python3 -m mypy src/dev_titles`,
   `PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider -o addopts="" test`. All must pass.
5. **Spot A/B** against the baseline snapshot given in the prompt:
   `PYTHONPATH=src python3 scripts/eval/spot_ab.py --runs <panel runs> --baseline <baseline root>
   --candidate . --samples 3 --cache-dir $DEV_TITLES_OUTPUT/agent/<date>/spot_cache
   --out $DEV_TITLES_OUTPUT/agent/<date>/spot_ab [--lang <l>]`. Always --samples 3: with one translation per
   line, the baseline judged against ITSELF swings 3-4 lines (2026-10-26). If the change touches
   the native-editor or reply prompts (editor.py), add --editor: without it the spot A/B runs the
   translate step only and cannot see an editor change.
   PROMISING = target wins beat losses in a sign test (p < 0.1) and the controls show no
   significant harm — the script computes both; report the p-values. Read the
   control LOSSES anyway — a real side effect there is your next finding. If NOT PROMISING, you may revise once
   and re-run; then stop.
6. **Commit** on the branch (`git add -A && git commit`), message: the mechanism, the change, the
   spot-A/B numbers.
7. **Report** to `$DEV_TITLES_OUTPUT/agent/<date>/REPORT.md`: the gap (with rows), the root cause, the diff
   summary, test + spot-A/B results, risks, and your recommendation: merge / full panel first / drop.
