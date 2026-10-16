"""Claude translator that shells out to the local `claude` CLI.

Routes through the local Claude Code binary. If that CLI is configured for
Bedrock (CLAUDE_CODE_USE_BEDROCK=1, AWS_REGION=...), it needs no API key and
inherits your AWS credentials. This is the path to use when direct Anthropic-API / Bedrock-SDK creds
aren't available but the CLI is.

Invocation: `claude -p --output-format text --bare [--model <id>]` with the prompt
on **stdin** (never as an arg). Omitting `--model` uses the CLI's configured
default model, which auto-upgrades as new models ship (matches the "use the
default model" preference); pass `model_id=` to pin.

Env is inherited from the parent (the harness must forward, not strip, it).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time


def _resolve_binary() -> str:
    """Locate the `claude` CLI: explicit $CLAUDE_BINARY, else the first `claude`
    on $PATH, else the bare name (subprocess resolves it / errors clearly)."""
    return os.environ.get("CLAUDE_BINARY") or shutil.which("claude") or "claude"


DEFAULT_BINARY = _resolve_binary()


def _default_runner(argv: list[str], prompt: str, timeout: float, env) -> str:
    proc = subprocess.run(
        argv, input=prompt, capture_output=True, text=True, timeout=timeout, env=env
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI exited {proc.returncode}: {proc.stderr.strip()[:500]}")
    return proc.stdout


from .base import TranslationRequest, Translator  # noqa: E402
from .prompts import build_reflect_prompt, build_translate_prompt  # noqa: E402


class CliTranslator(Translator):
    def __init__(
        self,
        *,
        binary: str = DEFAULT_BINARY,
        model_id: str | None = None,
        timeout: float = 300.0,
        env=None,
        runner=_default_runner,
        retries: int = 5,
    ):
        self.binary = binary
        self.model_id = model_id  # None -> CLI default model (auto-upgrades)
        self.timeout = timeout
        self.env = env  # None -> inherit parent env (Bedrock vars)
        self._runner = runner  # injectable for tests
        # transient CLI hang/timeout or Bedrock THROTTLING -> retry, don't crash the run.
        # 5 retries with exponential backoff rides out a sustained-throttle window (a load
        # spike degrades to slower, not untranslated — learned from a concurrency incident on a long run).
        self.retries = retries

    def _argv(self) -> list[str]:
        argv = [self.binary, "-p", "--output-format", "text", "--bare"]
        if self.model_id:
            argv += ["--model", self.model_id]
        return argv

    @staticmethod
    def _is_throttle(err: Exception) -> bool:
        s = str(err).lower()
        return any(k in s for k in ("429", "throttl", "toomanyrequests", "rate exceed", "quota"))

    def _invoke(self, prompt: str) -> str:
        # A single hung/slow CLI call (Bedrock under load) or transient non-zero exit
        # shouldn't crash a long batch — retry with backoff. THROTTLE errors (429) get a
        # longer exponential backoff + jitter so a sustained-throttle window is ridden out
        # rather than exhausting retries and dropping the segment to an EN placeholder.
        import random

        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._runner(self._argv(), prompt, self.timeout, self.env).strip()
            except (subprocess.TimeoutExpired, RuntimeError) as e:
                last = e
                if attempt < self.retries:
                    base = 8.0 if self._is_throttle(e) else 2.0
                    delay = min(base * (2**attempt), 90.0) + random.uniform(0, base)
                    time.sleep(delay)
        raise last or RuntimeError("claude CLI failed with no captured error")

    def translate(self, req: TranslationRequest) -> str:
        return self._invoke(build_translate_prompt(req))

    def reflect_and_refine(self, req: TranslationRequest, draft: str) -> str:
        return self._invoke(build_reflect_prompt(req, draft))
