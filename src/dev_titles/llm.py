"""Which model does the work: one place that picks the primary model and the second judge.

Every model call in the pipeline is "prompt in, text out", so any backend works behind it:

  claude  (default) the local `claude` CLI (Claude Code), with whatever it is signed in with
  codex   the local OpenAI Codex CLI, with whatever it is signed in with (e.g. a ChatGPT account)
  openai  any OpenAI-compatible chat API: OpenAI, Gemini's compatible endpoint, Ollama, LM Studio…

Choose with `dev-titles --llm <backend> [--llm-model <id>] …` or $DEV_TITLES_LLM /
$DEV_TITLES_LLM_MODEL. The second QE judge is always a DIFFERENT model family from the primary
(Claude <-> GPT); if none is available the panel runs with the primary model alone.
"""

from __future__ import annotations

import os
import shutil

BACKENDS = ("claude", "codex", "openai")
_cfg: dict = {"backend": None, "model": None}


def configure(backend: str | None = None, model: str | None = None) -> None:
    if backend and backend not in BACKENDS:
        raise SystemExit(f"--llm must be one of {', '.join(BACKENDS)} (got {backend!r})")
    _cfg["backend"], _cfg["model"] = backend, model


def backend() -> str:
    b = _cfg["backend"] or os.environ.get("DEV_TITLES_LLM") or "claude"
    if b not in BACKENDS:
        raise SystemExit(f"$DEV_TITLES_LLM must be one of {', '.join(BACKENDS)} (got {b!r})")
    return b


def model() -> str | None:
    return _cfg["model"] or os.environ.get("DEV_TITLES_LLM_MODEL") or None


def family(name: str | None = None, model_id: str | None = None) -> str:
    """Model family, used to keep the second judge independent of the primary."""
    name = name or backend()
    if name == "claude":
        return "claude"
    if name == "codex":
        return "gpt"
    m = (model_id or model() or "").lower()
    if m.startswith(("gpt", "o1", "o3", "o4", "o5", "openai")):
        return "gpt"
    if "claude" in m:
        return "claude"
    return m or "other"


def _claude_available() -> bool:
    from .translate.cli_translator import DEFAULT_BINARY

    return bool(shutil.which(DEFAULT_BINARY) or os.path.exists(DEFAULT_BINARY))


def _codex_available() -> bool:
    from .eval.codex_judge import DEFAULT_BINARY

    return bool(shutil.which(DEFAULT_BINARY) or os.path.exists(DEFAULT_BINARY))


def translator(model_id: str | None = None, name: str | None = None):
    """The primary Translator. `model_id` overrides --llm-model (e.g. the run's --model pin)."""
    name = name or backend()
    mid = model_id or model()
    if name == "claude":
        from .translate import CliTranslator

        return CliTranslator(model_id=mid)
    if name == "codex":
        from .translate.codex_translator import CodexTranslator

        return CodexTranslator(model_id=mid)
    from .translate.openai_compat import OpenAICompatTranslator

    return OpenAICompatTranslator(model_id=mid)


def invoke(model_id: str | None = None):
    """prompt -> text with the primary model (clean-read, condensation, editor, judge #1…)."""
    return translator(model_id)._invoke


def second_judge():
    """(invoke, label) for a judge from a different family than the primary, else (None, reason)."""
    fam = family()
    if fam != "gpt" and _codex_available():
        from .eval.codex_judge import CodexCliJudge

        return CodexCliJudge().invoke, "Codex/GPT"
    if fam != "claude" and _claude_available():
        from .translate import CliTranslator

        return CliTranslator()._invoke, "Claude"
    want = "the codex CLI" if fam != "gpt" else "the claude CLI"
    return None, f"no different-family judge available (install {want} for a two-judge panel)"


def label() -> str:
    b = backend()
    return {"claude": "Claude", "codex": "Codex/GPT"}.get(b, f"{model() or 'openai-compatible'}")
