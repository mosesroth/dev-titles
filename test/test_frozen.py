"""dev_titles.frozen: long runs execute from a frozen copy of the code."""

import os
import time

from dev_titles import cli, frozen


def _pkg(tmp_path, body="x = 1\n"):
    pkg = tmp_path / "live" / "dev_titles"
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text(body)
    (pkg / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"junk")
    return pkg


def test_snapshot_copies_once_per_code_and_a_change_gets_a_new_copy(tmp_path):
    pkg, base = _pkg(tmp_path), tmp_path / "run_code"
    a = frozen.snapshot(pkg, base)
    assert (a / "dev_titles" / "mod.py").read_text() == "x = 1\n"
    assert not (a / "dev_titles" / "__pycache__").exists()
    assert frozen.snapshot(pkg, base) == a  # same code: same copy
    (pkg / "mod.py").write_text("x = 2\n")  # an edit mid-run
    b = frozen.snapshot(pkg, base)
    assert b != a and (a / "dev_titles" / "mod.py").read_text() == "x = 1\n"  # old run's copy kept
    # the pyc doesn't change the hash; source does
    (pkg / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"other")
    assert frozen.snapshot(pkg, base) == b


def test_prune_removes_only_stale_copies_and_never_the_current(tmp_path):
    base = tmp_path / "run_code"
    old, cur, fresh = (base / n for n in ("old", "cur", "fresh"))
    for d in (old, cur, fresh):
        d.mkdir(parents=True)
    stale = time.time() - 30 * 86400
    os.utime(old, (stale, stale))
    os.utime(cur, (stale, stale))
    assert frozen.prune(base, keep=cur) == ["old"]
    assert cur.exists() and fresh.exists() and not old.exists()
    assert frozen.prune(tmp_path / "missing") == []


def test_reexec_runs_the_same_command_from_the_copy_once(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv(frozen.ENV, raising=False)
    monkeypatch.delenv("SUBTITLE_REVIEWED_GLOB", raising=False)
    monkeypatch.setattr(frozen, "snapshot", lambda: tmp_path / "abc123" / "src")
    monkeypatch.setattr(frozen, "prune", lambda keep: calls.append(("prune", keep)))
    frozen.reexec(["run", "x.mp4"], execve=lambda exe, argv, env: calls.append((argv, env)))
    argv, env = calls[-1]
    assert argv[-4:] == ["-m", "dev_titles.cli", "run", "x.mp4"]
    assert env["PYTHONPATH"] == str(tmp_path / "abc123" / "src") and env[frozen.ENV] == "abc123"
    assert env["PYTHONUNBUFFERED"] == "1"
    glob = env["SUBTITLE_REVIEWED_GLOB"]  # pinned, absolute: not relative to the frozen copy
    assert glob.startswith("/") and glob.endswith("*Reviewed*.srt")
    assert calls[0] == ("prune", tmp_path / "abc123")
    # already frozen: no second exec
    monkeypatch.setenv(frozen.ENV, "abc123")
    n = len(calls)
    frozen.reexec(["run"], execve=lambda *a: calls.append(a))
    assert len(calls) == n


def test_only_long_commands_from_the_command_line_are_frozen(monkeypatch):
    seen = []
    monkeypatch.setattr(frozen, "reexec", lambda raw: seen.append(raw))
    monkeypatch.setattr(cli, "cmd_run", lambda a: 0)
    monkeypatch.setattr(cli, "_LONG_COMMANDS", (cli.cmd_run,))
    monkeypatch.setattr(cli, "_stay_awake", lambda: None)
    parser = cli.build_parser

    class P:
        def parse_args(self, raw):
            from types import SimpleNamespace

            return SimpleNamespace(func=cli.cmd_run if raw[0] == "run" else (lambda a: 0))

    monkeypatch.setattr(cli, "build_parser", lambda: P())
    monkeypatch.setattr(cli.sys, "argv", ["dev_titles", "run", "v.mp4"])
    assert cli.main() == 0 and seen == [["run", "v.mp4"]]
    assert (
        cli.main(["run", "v.mp4"]) == 0 and len(seen) == 1
    )  # called in-process (tests): not frozen
    monkeypatch.setattr(cli.sys, "argv", ["dev_titles", "validate"])
    assert cli.main() == 0 and len(seen) == 1
    assert parser


def test_a_stale_temp_copy_is_replaced_and_a_real_rename_failure_raises(tmp_path, monkeypatch):
    import pathlib

    import pytest

    pkg, base = _pkg(tmp_path), tmp_path / "run_code"
    stale = base / frozen.code_hash(pkg) / f"src.tmp{os.getpid()}" / "dev_titles"
    stale.mkdir(parents=True)  # a crashed run with this pid
    assert (frozen.snapshot(pkg, base) / "dev_titles" / "mod.py").exists()
    (pkg / "mod.py").write_text("x = 3\n")

    def boom(self, target):
        raise OSError("disk full")

    monkeypatch.setattr(pathlib.Path, "rename", boom)
    with pytest.raises(OSError):
        frozen.snapshot(pkg, base)


def test_a_snapshot_failure_runs_from_the_live_tree_instead(monkeypatch, capsys):
    monkeypatch.delenv(frozen.ENV, raising=False)

    def fail():
        raise OSError("read-only")

    monkeypatch.setattr(frozen, "snapshot", fail)
    calls = []
    frozen.reexec(["run"], execve=lambda *a: calls.append(a))
    assert calls == [] and "running from the live tree" in capsys.readouterr().out


def test_prune_skips_an_entry_a_concurrent_run_removed(tmp_path, monkeypatch):
    import pathlib

    base = tmp_path / "run_code"
    (base / "gone").mkdir(parents=True)
    real = pathlib.Path.stat

    def stat(self, *a, **k):
        if self.name == "gone":
            raise FileNotFoundError(self)
        return real(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "stat", stat)
    assert frozen.prune(base) == []


def test_orphaned_temp_copies_of_other_runs_are_swept(tmp_path):
    pkg, base = _pkg(tmp_path), tmp_path / "run_code"
    dst = frozen.snapshot(pkg, base)
    old, new = dst.parent / "src.tmp1", dst.parent / "src.tmp2"
    old.mkdir()
    new.mkdir()
    t = time.time() - 7200
    os.utime(old, (t, t))
    frozen.snapshot(pkg, base)
    assert not old.exists() and new.exists()  # a fresh one may be a copy in progress
