"""Claude-on-Bedrock translator (primary engine, requirement R2).

Uses the Anthropic **Mantle** Bedrock client (the Messages-API Bedrock endpoint),
not raw boto3 Converse — it exposes the same `.messages.create` surface as the
first-party SDK. Model IDs take an `anthropic.` prefix on Bedrock, e.g.
`anthropic.claude-opus-5`.

Model/thinking notes (per the Claude API reference, 2026):
  - Default to **Claude Opus 5** for the quality translate/refine pass; Sonnet 5
    for the cost-balanced bulk pass.
  - Opus 5 has thinking ON by default. We keep **adaptive thinking** and control
    cost with `output_config.effort` (default "low" for the draft pass) rather
    than DISABLING thinking — disabling has documented failure modes (it can leak
    tool-call text or `<thinking>` tags into the visible output).
  - Prefill is removed on Opus 5 — output format is controlled by the prompt
    (see translate/prompts.py), never by an assistant prefill.

Requires `pip install 'anthropic[bedrock]'` and AWS credentials/region. The
import is lazy so this module loads (and the rest of the harness tests run)
without the dependency installed.
"""

from __future__ import annotations

import os

from .base import TranslationRequest, Translator
from .prompts import build_reflect_prompt, build_translate_prompt

# Bedrock model IDs (anthropic. prefix). Use an inference profile / regional ID
# per your account if required; verify Bedrock parity for thinking/effort.
DEFAULT_MODEL = "anthropic.claude-opus-5"


class BedrockTranslator(Translator):
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL,
        region: str | None = None,
        max_tokens: int = 1024,
        effort: str = "low",
        thinking: bool = True,
    ):
        self.model_id = model_id
        self.region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        self.max_tokens = max_tokens
        self.effort = effort  # low | medium | high | xhigh | max
        self.thinking = thinking  # keep adaptive thinking on (recommended)
        self._client: object | None = None

    def _client_lazy(self):
        if self._client is None:
            if not self.region:
                raise RuntimeError("No AWS region set — pass region= or set AWS_REGION.")
            try:
                from anthropic import AnthropicBedrockMantle
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "Install the Bedrock SDK: pip install 'anthropic[bedrock]'"
                ) from e
            self._client = AnthropicBedrockMantle(aws_region=self.region)
        return self._client

    def _invoke(self, prompt: str) -> str:
        kwargs: dict = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "output_config": {"effort": self.effort},
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        resp = self._client_lazy().messages.create(**kwargs)
        # Return the concatenated text blocks (skip thinking blocks).
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def translate(self, req: TranslationRequest) -> str:
        return self._invoke(build_translate_prompt(req))

    def reflect_and_refine(self, req: TranslationRequest, draft: str) -> str:
        return self._invoke(build_reflect_prompt(req, draft))
