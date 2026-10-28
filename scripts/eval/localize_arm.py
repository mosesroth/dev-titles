#!/usr/bin/env python3
"""Run the full pipeline on ONE A/B arm: a code snapshot (a folder with src/ + glossary/), fully isolated.

Runs from inside the snapshot, so EVERYTHING the code reads or writes relative to its working folder —
glossary.yaml, partners.yaml, editor examples, names.yaml — is the snapshot's own copy, for snapshots
made before or after glossary_dir() existed. Before 2026-10-27 the arms ran from the repo: both read
the LIVE glossary + editor examples (so an "old code" arm still had today's lessons) and wrote
partner lookups into the live partners.yaml.

The translation memory and the reviewers' corpus are shared, read-only, and passed as absolute paths
(the corpus: SUBTITLE_REVIEWED_GLOB / SUBTITLE_EN_DIR, defaults resolved from the current folder —
see dev_titles.latin_names).

  python3 scripts/eval/localize_arm.py <snapshot> <en.vtt> <out dir> [--holdout <video>] [-- extra localize args]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dev_titles.latin_names import default_en_dir, default_reviewed_glob  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("en")
    ap.add_argument("out")
    ap.add_argument("--holdout", default="")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    snap = Path(a.snapshot).resolve()
    if not (snap / "src").is_dir() or not (snap / "glossary" / "glossary.yaml").exists():
        sys.exit(f"{snap}: not a snapshot (needs src/ and glossary/glossary.yaml)")
    if snap == ROOT:
        sys.exit("refusing to run an A/B arm on the live repo — snapshot it first (git archive)")
    env = {
        **os.environ,
        "PYTHONPATH": str(snap / "src"),
        "SUBTITLE_GLOSSARY_DIR": str(snap / "glossary"),
        "SUBTITLE_REVIEWED_GLOB": os.path.abspath(default_reviewed_glob()),
        "SUBTITLE_EN_DIR": os.path.abspath(default_en_dir()),
    }
    if a.holdout:
        env["SUBTITLE_HOLDOUT_VIDEOS"] = a.holdout
    rest = [x for x in a.rest if x != "--"]
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "dev_titles.cli",
        "localize",
        str(Path(a.en).resolve()),
        "--out",
        str(Path(a.out).resolve()),
        "--glossary",
        "glossary/glossary.yaml",
        "--tm",
        str(ROOT / "tm" / "tm.jsonl"),
        *rest,
    ]
    return subprocess.run(cmd, cwd=snap, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
