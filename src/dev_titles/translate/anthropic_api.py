"""Claude translator via the first-party Anthropic API (non-Bedrock path).

Same Translator interface and prompts as the Bedrock adapter — this is the path
to use when Bedrock access isn't available. Uses the official `anthropic` SDK;
model IDs have NO `anthropic.` prefix here (e.g. `claude-opus-5`).

Credentials resolve the SDK's normal way (first match wins): ANTHROPIC_API_KEY,
ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile — a zero-arg client works
after `ant auth login` with no env var set. Install with `pip install anthropic`.

Model notes (per the Claude API reference): the model defaults to "auto" — the
newest model in the Opus family is picked from the Models API at first use (so
the pipeline auto-upgrades as new models ship); pass a concrete id to pin. Keep
adaptive thinking ON and tune cost with `output_config.effort` (don't disable
thinking); prefill is removed, so output format is controlled by the prompt.
"""

from __future__ import annotations

from .base import TranslationRequest, Translator
from .model_select import resolve_model
from .prompts import build_reflect_prompt, build_translate_prompt


class AnthropicTranslator(Translator):
    def __init__(
        self,
        *,
        model_id: str = "auto",
        max_tokens: int = 1024,
        effort: str = "low",
        thinking: bool = True,
        client=None,
        model_family: str = "claude-opus",
    ):
        # model_id="auto" (default) picks the newest model in `model_family` from
        # the Models API at first use, so the pipeline upgrades itself as new
        # models ship. Pass a concrete id to pin.
        self.requested_model = model_id
        self.model_family = model_family
        self._resolved_model: str | None = None
        self.max_tokens = max_tokens
        self.effort = effort  # low | medium | high | xhigh | max
        self.thinking = thinking
        self._client = client  # injectable for tests

    def _client_lazy(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("Install the SDK: pip install anthropic") from e
            self._client = anthropic.Anthropic()  # resolves env/profile creds
        return self._client

    @property
    def model_id(self) -> str:
        """The resolved model id (auto-selected once, then cached)."""
        if self._resolved_model is None:
            self._resolved_model = resolve_model(
                self._client_lazy(), self.requested_model, family=self.model_family
            )
        return self._resolved_model

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
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def translate(self, req: TranslationRequest) -> str:
        return self._invoke(build_translate_prompt(req))

    def reflect_and_refine(self, req: TranslationRequest, draft: str) -> str:
        return self._invoke(build_reflect_prompt(req, draft))
