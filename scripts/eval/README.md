# Evaluation scripts

Benchmarks and A/B comparisons used to measure the harness against human-reviewed subtitles —
not part of producing subtitles. Run from the repo root, e.g.
`PYTHONPATH=src python3 scripts/eval/seg_eval.py …`. The operational tools (vendor-review check,
held-line triage, re-timing, clean-read, review mining, publishing) stay in `scripts/`.
