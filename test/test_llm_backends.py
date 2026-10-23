"""Backend selection (--llm) and the non-Claude adapters."""

import urllib.error

import pytest


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    from dev_titles import llm

    monkeypatch.delenv("DEV_TITLES_LLM", raising=False)
    monkeypatch.delenv("DEV_TITLES_LLM_MODEL", raising=False)
    llm.configure(None, None)
    yield
    llm.configure(None, None)


def test_default_backend_is_claude_cli():
    from dev_titles import llm
    from dev_titles.translate import CliTranslator

    assert llm.backend() == "claude" and isinstance(llm.translator(), CliTranslator)


def test_backend_from_flag_and_env(monkeypatch):
    from dev_titles import llm
    from dev_titles.translate.codex_translator import CodexTranslator

    monkeypatch.setenv("DEV_TITLES_LLM", "codex")
    assert llm.backend() == "codex"
    monkeypatch.setattr("dev_titles.eval.codex_judge.ensure_clean_codex_home", lambda *a, **k: "")
    assert isinstance(llm.translator(), CodexTranslator)
    llm.configure("openai", "gpt-5.5")
    assert llm.backend() == "openai" and llm.model() == "gpt-5.5"
    with pytest.raises(SystemExit):
        llm.configure("gemini")


def test_cli_flag_configures_backend(monkeypatch):
    from dev_titles import cli, llm

    seen = {}
    monkeypatch.setattr(
        cli, "cmd_validate", lambda a: seen.setdefault("b", (llm.backend(), llm.model())) and 0
    )
    cli.main(["--llm", "openai", "--llm-model", "llama3.1", "validate", "x.vtt", "--lang", "ja"])
    assert seen["b"] == ("openai", "llama3.1")


def test_second_judge_is_the_other_family(monkeypatch):
    from dev_titles import llm

    monkeypatch.setattr(llm, "_codex_available", lambda: True)
    monkeypatch.setattr(llm, "_claude_available", lambda: True)
    monkeypatch.setattr("dev_titles.eval.codex_judge.ensure_clean_codex_home", lambda *a, **k: "")
    assert llm.second_judge()[1] == "Codex/GPT"  # claude primary -> GPT judge
    llm.configure("codex")
    assert llm.second_judge()[1] == "Claude"  # GPT primary -> Claude judge
    llm.configure("openai", "gpt-5.5")
    assert llm.second_judge()[1] == "Claude"  # GPT via API -> still Claude, never GPT judging GPT
    llm.configure("openai", "gemini-2.5-pro")
    assert llm.second_judge()[1] == "Codex/GPT"  # a third family -> either works; codex first
    monkeypatch.setattr(llm, "_claude_available", lambda: False)
    llm.configure("codex")
    judge, why = llm.second_judge()
    assert judge is None and "claude" in why


def test_openai_compat_request_and_retry(monkeypatch):
    from dev_titles.translate import openai_compat as oc
    from dev_titles.translate.base import TranslationRequest

    monkeypatch.setattr(oc.time, "sleep", lambda *_: None)
    calls = []

    def post(url, body, headers, timeout):
        calls.append((url, body, headers))
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 429, "slow down", {}, None)
        return {"choices": [{"message": {"content": "  こんにちは  "}}]}

    t = oc.OpenAICompatTranslator(
        model_id="llama3.1", base_url="http://localhost:11434/v1/", api_key="", post=post
    )
    out = t.translate(TranslationRequest("Hello", "en", "ja"))
    assert out == "こんにちは" and len(calls) == 2  # retried the 429
    url, body, headers = calls[-1]
    assert url == "http://localhost:11434/v1/chat/completions"
    assert body["model"] == "llama3.1" and body["messages"][0]["role"] == "user"
    assert "Authorization" not in headers  # local server, no key


def test_openai_compat_needs_model_and_key(monkeypatch):
    from dev_titles.translate.openai_compat import OpenAICompatTranslator

    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="needs a model"):
        OpenAICompatTranslator()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAICompatTranslator(model_id="gpt-5.5")
    t = OpenAICompatTranslator(
        model_id="gpt-5.5",
        api_key="sk-test",
        post=lambda *a: {"choices": [{"message": {"content": "ok"}}]},
    )
    assert t._invoke("x") == "ok"


def test_codex_translator_retries_empty_output(monkeypatch):
    from dev_titles.translate import codex_translator as ct

    monkeypatch.setattr(ct.time, "sleep", lambda *_: None)
    outs = iter(["", "訳文"])

    class FakeCodex:
        def invoke(self, prompt):
            return next(outs)

    assert ct.CodexTranslator(judge=FakeCodex())._invoke("x") == "訳文"
    t = ct.CodexTranslator(judge=type("C", (), {"invoke": lambda self, p: ""})(), retries=1)
    with pytest.raises(RuntimeError, match="codex login"):
        t._invoke("x")


def test_asr_engine_auto(monkeypatch):
    import importlib.util
    import platform
    import sys

    from dev_titles import cli

    have = set()
    monkeypatch.setattr(importlib.util, "find_spec", lambda n: object() if n in have else None)
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    assert cli._resolve_engine("whisper") == "whisper"  # explicit choice untouched
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    have.update({"parakeet_mlx", "whisperx"})
    assert cli._resolve_engine("auto") == "whisper"  # Parakeet needs Apple Silicon
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    assert cli._resolve_engine("auto") == "parakeet"
    have.clear()
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "k")
    assert cli._resolve_engine("auto") == "assemblyai"
    monkeypatch.delenv("ASSEMBLYAI_API_KEY")
    with pytest.raises(SystemExit, match="No speech-recognition engine"):
        cli._resolve_engine("auto")
