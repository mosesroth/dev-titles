"""Run hours-long commands from a frozen copy of the code, never from the live tree.

Many modules are imported lazily, at the step that needs them (localize imports adjudicate only at
review-and-fix). A run that started on old code and reached that step after a change was merged
loaded the NEW adjudicate against the OLD localize: review-and-fix raised a TypeError and was
skipped on two videos, while the runs still reported PASS.

So `run` / `localize` copy src/dev_titles to scratch/run_code/<content hash>/ and re-exec from
there (same pid: caffeinate -w and the queue scripts keep tracking it). Identical code shares one
copy; copies untouched for PRUNE_DAYS are removed. The working directory is unchanged, so glossary/,
input/ and output/ resolve as before.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

ENV = "DEV_TITLES_FROZEN"
PRUNE_DAYS = 14
_PKG = Path(__file__).resolve().parent  # src/dev_titles
ROOT = _PKG.parents[1]  # the repo
BASE = ROOT / "scratch" / "run_code"


def _files(pkg: Path) -> list:
    return sorted(
        p
        for p in pkg.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )


def code_hash(pkg: Path = _PKG) -> str:
    h = hashlib.sha256()
    for p in _files(pkg):
        h.update(str(p.relative_to(pkg)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:12]


def snapshot(pkg: Path = _PKG, base: Path = BASE) -> Path:
    """The frozen src dir holding a copy of `pkg` (made once per distinct code)."""
    dst = base / code_hash(pkg) / "src"
    if not (dst / pkg.name / "__init__.py").exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.parent / f"src.tmp{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)  # left by a crashed run that had this pid
        shutil.copytree(pkg, tmp / pkg.name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        try:
            tmp.rename(dst)  # atomic: a concurrent run never sees a half copy
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            if not (dst / pkg.name / "__init__.py").exists():
                raise  # a real failure, not another run winning the race
    # temp copies of crashed runs (any pid) under a reused copy: a copy takes seconds, so an hour
    # old one is orphaned
    for t in dst.parent.glob("src.tmp*"):
        try:
            orphan = t.stat().st_mtime < time.time() - 3600
        except OSError:
            continue
        if orphan:
            shutil.rmtree(t, ignore_errors=True)
    os.utime(dst.parent)  # mark as recently used
    return dst


def prune(base: Path = BASE, keep: Path | None = None, days: int = PRUNE_DAYS) -> list:
    """Remove copies untouched for `days` (a run in flight touched its copy when it started; runs
    last hours, not weeks). Only direct children of `base` are ever removed."""
    gone: list = []
    if not base.is_dir():
        return gone
    cutoff = time.time() - days * 86400
    for d in base.iterdir():
        try:
            stale = d.is_dir() and d.parent == base and d != keep and d.stat().st_mtime < cutoff
        except OSError:  # removed by a concurrent run's prune
            continue
        if stale:
            shutil.rmtree(d, ignore_errors=True)
            gone.append(d.name)
    return gone


def reexec(argv: list, execve=os.execve) -> None:
    """Re-exec this command from a frozen copy; returns only when already frozen."""
    if os.environ.get(ENV):
        return
    try:
        src = snapshot()
        prune(keep=src.parent)
    except OSError as e:  # never lose a run over it: fall back to the live tree, loudly
        print(f"WARNING: could not freeze the code ({e}); running from the live tree", flush=True)
        return
    # unbuffered: the queue scripts tail these logs (a `python -u` launch has no other trace here)
    env = {**os.environ, ENV: src.parent.name, "PYTHONPATH": str(src), "PYTHONUNBUFFERED": "1"}
    # the reviewers' corpus is resolved from the current folder — pin it as an absolute path, so
    # where the frozen copy lives can never change which files the run reads
    from .latin_names import default_reviewed_glob

    env.setdefault("SUBTITLE_REVIEWED_GLOB", os.path.abspath(default_reviewed_glob()))
    print(f"code frozen for this run: {src.parent}", flush=True)
    # fixed interpreter + module, the operator's own argv
    execve(sys.executable, [sys.executable, "-m", "dev_titles.cli", *argv], env)  # nosec B606
