"""Translator for any OpenAI-compatible chat-completions API (pipeline step [3], `--llm openai`).

Covers OpenAI itself and the many services and local servers that expose the same API
(e.g. Gemini's OpenAI-compatible endpoint, Ollama, LM Studio, vLLM). Standard library only.

  OPENAI_API_KEY    API key (optional for local servers)
  OPENAI_BASE_URL   default https://api.openai.com/v1 (e.g. http://localhost:11434/v1 for Ollama)
  --llm-model / $DEV_TITLES_LLM_MODEL   the model id (required: there is no safe default)

The prompts are the same as for Claude. Quality was measured with Claude as the primary model;
other models are expected to work but are unmeasured, and small local models may struggle
with the long, style-heavy prompts.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request

from .base import TranslationRequest, Translator
from .prompts import build_reflect_prompt, build_translate_prompt


def _default_post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    if not url.lower().startswith(("https://", "http://")):
        raise RuntimeError(f"OPENAI_BASE_URL must be an http(s) URL (got {url[:80]})")
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    # scheme restricted to http(s) above; Bandit can't see that check
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return json.loads(resp.read().decode("utf-8"))


class OpenAICompatTranslator(Translator):
    def __init__(
        self,
        *,
        model_id: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
        retries: int = 5,
        post=_default_post,
    ):
        self.model_id = (
            model_id or os.environ.get("DEV_TITLES_LLM_MODEL") or os.environ.get("OPENAI_MODEL")
        )
        if not self.model_id:
            raise RuntimeError(
                "--llm openai needs a model: pass --llm-model <id> or set $DEV_TITLES_LLM_MODEL "
                "(e.g. a GPT model for OpenAI, or the model name your local server serves)"
            )
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key and "api.openai.com" in self.base_url:
            raise RuntimeError("--llm openai with the OpenAI API needs $OPENAI_API_KEY")
        self.timeout = timeout
        self.retries = retries
        self._post = post  # injectable for tests

    @staticmethod
    def _retryable(err: Exception) -> bool:
        if isinstance(err, urllib.error.HTTPError):
            return err.code == 429 or err.code >= 500
        return isinstance(err, (urllib.error.URLError, TimeoutError, ConnectionError))

    def _invoke(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": self.model_id, "messages": [{"role": "user", "content": prompt}]}
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                out = self._post(f"{self.base_url}/chat/completions", body, headers, self.timeout)
                return (out["choices"][0]["message"].get("content") or "").strip()
            except Exception as e:  # noqa: BLE001 — classify below
                last = e
                if attempt < self.retries and self._retryable(e):
                    base = 8.0 if getattr(e, "code", None) == 429 else 2.0
                    time.sleep(min(base * (2**attempt), 90.0) + random.uniform(0, base))
                    continue
                break
        if isinstance(last, urllib.error.HTTPError):
            detail = last.read().decode("utf-8", "replace")[:300] if hasattr(last, "read") else ""
            raise RuntimeError(f"OpenAI-compatible API error {last.code}: {detail}") from last
        raise RuntimeError(f"OpenAI-compatible API call failed: {last}") from last

    def translate(self, req: TranslationRequest) -> str:
        return self._invoke(build_translate_prompt(req))

    def reflect_and_refine(self, req: TranslationRequest, draft: str) -> str:
        return self._invoke(build_reflect_prompt(req, draft))
