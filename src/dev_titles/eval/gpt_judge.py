"""Non-Claude judge via OpenAI GPT-5.x on AWS Bedrock ("Mantle").

Our parity audit runs on Opus (Claude) judging Claude output — self-preference
bias inflates "machine ~= human" verdicts. GPT-5.5 on Bedrock is a DIFFERENT model
family, so it de-biases the judge. Bedrock exposes an OpenAI-compatible endpoint
(Mantle), so the official `openai` SDK works unchanged — only base_url + key change
(per the OpenAI-on-Bedrock launch docs).

Auth: a short-term **Bedrock API key** (Bedrock console) passed as the OpenAI
api_key, OR set env AWS_BEARER_TOKEN_BEDROCK / BEDROCK_API_KEY. Requires
`bedrock-mantle` IAM access in your AWS account. GPT-5.5 is in us-east-2.
"""

from __future__ import annotations

import os

MANTLE_BASE = "https://bedrock-mantle.us-east-2.api.aws/openai/v1"
DEFAULT_MODEL = "openai.gpt-5.5"


class GptMantleJudge:
    """`invoke(prompt) -> str`, matching the callable the audit/coverage code uses,
    so it drops in wherever `CliTranslator._invoke` is used as the judge."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = MANTLE_BASE,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        client=None,
    ):
        self.model = model
        self.effort = effort
        self._client = client  # injectable for tests
        self._api_key = (
            api_key
            or os.environ.get("BEDROCK_API_KEY")
            or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        )
        self._base_url = base_url

    def _get(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("pip install openai") from e
            if not self._api_key:
                raise RuntimeError(
                    "No Bedrock key: set BEDROCK_API_KEY (short-term key from the "
                    "Bedrock console) or pass api_key=."
                )
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    def invoke(self, prompt: str) -> str:
        resp = self._get().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()
