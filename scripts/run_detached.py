#!/usr/bin/env python3
"""Launch a long-running command DETACHED into its own session.

Why: Claude Code's `run_in_background` bash tasks live in the launching session's
process group, and the harness reaps them when the turn that spawned them tears
down (the task shows `[killed]`). A multi-hour job (transcription, translation)
therefore dies at the next turn boundary with only its cache preserved.

This launcher forks a child that calls `os.setsid()` (macOS has no `setsid` CLI,
but the syscall exists), redirects stdout/stderr to a log, and `execvp`s the
command. The child becomes a session leader reparented to init (PID 1), so the
harness's process-group kill no longer reaches it. The parent prints the detached
PID and returns immediately, so the calling shell (and turn) can end freely.

Usage:
    PYTHONPATH=src python3 scripts/run_detached.py --log run.log -- \\
        python3 -m dev_titles.cli localize in.en.vtt --out bundle --qe-judge

Env (e.g. PYTHONPATH) is inherited by the exec'd command. Monitor via the log /
the job's own resumable cache; there is no completion callback (that's the point
— it outlives the session). Re-running a resumable command after a crash/sleep
picks up where it left off.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a command detached in its own session.")
    ap.add_argument("--log", required=True, help="file to append stdout+stderr to")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- then the command to run")
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("no command given (use: --log FILE -- CMD [ARGS...])")

    pid = os.fork()
    if pid > 0:  # parent: report and return so the turn can end
        print(f"detached pid {pid} -> {args.log}")
        return 0

    # child: new session (escape the harness process group), redirect, exec
    os.setsid()
    fd = os.open(args.log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.execvp(cmd[0], cmd)
    return 0  # unreachable if execvp succeeds


if __name__ == "__main__":
    sys.exit(main())
