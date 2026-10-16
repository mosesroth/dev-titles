"""Throttle-retry hardening + LLM serialize lock (a concurrency incident in a batch run)."""

import pytest

import dev_titles.translate.cli_translator as clm
from dev_titles.localize import llm_serialize_lock
from dev_titles.translate.cli_translator import CliTranslator


def test_is_throttle_detection():
    assert CliTranslator._is_throttle(RuntimeError("claude exited 1: 429 TooManyRequests"))
    assert CliTranslator._is_throttle(RuntimeError("ThrottlingException: Rate exceeded"))
    assert not CliTranslator._is_throttle(RuntimeError("binary not found"))


def test_retries_ride_out_transient_throttle(monkeypatch):
    monkeypatch.setattr(clm.time, "sleep", lambda *_: None)  # no real backoff in tests
    calls = {"n": 0}

    def runner(argv, prompt, timeout, env):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 ThrottlingException")
        return "  译文  "

    out = CliTranslator(runner=runner)._invoke("x")
    assert out == "译文" and calls["n"] == 3  # succeeded on the 3rd try, not dropped


def test_gives_up_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(clm.time, "sleep", lambda *_: None)

    def runner(argv, prompt, timeout, env):
        raise RuntimeError("429 throttled")

    with pytest.raises(RuntimeError):
        CliTranslator(runner=runner, retries=2)._invoke("x")


def test_llm_serialize_lock_acquires_and_releases():
    with llm_serialize_lock():
        pass  # no error = acquired + released cleanly
