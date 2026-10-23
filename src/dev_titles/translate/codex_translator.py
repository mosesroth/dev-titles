"""Translator that shells out to the local OpenAI Codex CLI (`--llm codex`).

Uses whatever Codex is signed in with — a ChatGPT account, an OpenAI API key, or a provider
configured in ~/.codex — through the same clean, MCP-free CODEX_HOME as the Codex judge
(see eval/codex_judge.py). Same prompts as the Claude adapter. A failed or empty call is retried
with backoff so one hiccup doesn't drop a segment.
"""

from __future__ import annotations

import random
import subprocess
import time

from ..eval.codex_judge import CodexCliJudge
from .base import TranslationRequest, Translator
from .prompts import build_reflect_prompt, build_translate_prompt


class CodexTranslator(Translator):
    def __init__(
        self, *, model_id: str | None = None, timeout: float = 600.0, retries: int = 4, judge=None
    ):
        self.model_id = model_id
        self.retries = retries
        self._codex = judge or CodexCliJudge(model=model_id, timeout=timeout)

    def _invoke(self, prompt: str) -> str:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                out = self._codex.invoke(prompt)
                if out:
                    return out
                last = RuntimeError(
                    "codex returned no output (check `codex login` / ~/.codex config)"
                )
            except (subprocess.TimeoutExpired, OSError, RuntimeError) as e:
                last = e
            if attempt < self.retries:
                time.sleep(min(2.0 * (2**attempt), 60.0) + random.uniform(0, 2.0))
        raise last or RuntimeError("codex CLI failed with no captured error")

    def translate(self, req: TranslationRequest) -> str:
        return self._invoke(build_translate_prompt(req))

    def reflect_and_refine(self, req: TranslationRequest, draft: str) -> str:
        return self._invoke(build_reflect_prompt(req, draft))
