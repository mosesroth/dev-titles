"""Non-Claude judge via the Codex CLI (OpenAI GPT-5.x, e.g. served from AWS Bedrock).

De-biases the parity audit: our default judge is Opus (Claude) grading Claude
output, which self-preference bias inflates. The Codex CLI can route to GPT-5.x on
AWS Bedrock using your existing AWS credentials/profile (configured in ~/.codex), so
no separate OpenAI key is needed.

Non-interactive: `codex exec --skip-git-repo-check -o <file> -` with the prompt on
stdin; the model's final message is written to <file>. `invoke(prompt) -> str`
matches the callable the audit uses, so it drops in for `CliTranslator._invoke`.
Install the Codex CLI and put it on $PATH (or set $CODEX_BINARY).

CLEAN CODEX_HOME: the judge runs codex under a minimal, self-generated CODEX_HOME
(`~/.codex_judge`) that mirrors the user's model + Bedrock provider but DROPS
mcp_servers / plugins / marketplaces. With a heavily customized config, codex may load
slow-starting MCP servers (init hang) + hundreds of plugin skills (context-budget blowout),
which hangs / instantly compacts a scoring call; the clean home returns in ~15s. Related
lesson: a stalled judge is usually a config/model problem (a dead model id — a retired
model's stream can open but never emit — or an MCP init hang), NOT the network; verify the
model/config before blaming credentials or connectivity.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

# $CODEX_BINARY, else the first `codex` on $PATH, else the bare name (callers check existence).
DEFAULT_BINARY = os.environ.get("CODEX_BINARY") or shutil.which("codex") or "codex"
JUDGE_HOME = os.path.expanduser("~/.codex_judge")

# Sane Bedrock defaults if the user's ~/.codex config is missing/unreadable.
_JUDGE_DEFAULTS = {
    "model": "openai.gpt-5.6-sol",
    "effort": "low",
    "provider": "amazon-bedrock",
    "region": "us-east-2",
    "profile": "default",
    "base_url": "http://127.0.0.1:9",
}


def ensure_clean_codex_home(home: str = JUDGE_HOME, src_home: str = "~/.codex") -> str:
    """Create/refresh a minimal CODEX_HOME for the judge: mirror the model + Bedrock provider
    from the user's real ~/.codex config, but WITHOUT mcp_servers / plugins / marketplaces
    (those hang or bloat codex startup and are irrelevant to a scoring call). Idempotent;
    falls back to `_JUDGE_DEFAULTS` when the source config can't be read/parsed."""
    d = dict(_JUDGE_DEFAULTS)
    try:
        import tomllib

        with open(os.path.join(os.path.expanduser(src_home), "config.toml"), "rb") as fh:
            cfg = tomllib.load(fh)
        d["model"] = cfg.get("model") or d["model"]
        d["effort"] = cfg.get("model_reasoning_effort") or d["effort"]
        d["provider"] = cfg.get("model_provider") or d["provider"]
        d["base_url"] = cfg.get("chatgpt_base_url") or d["base_url"]
        aws = ((cfg.get("model_providers") or {}).get(d["provider"]) or {}).get("aws") or {}
        d["region"] = aws.get("region") or d["region"]
        d["profile"] = aws.get("profile") or d["profile"]
    except Exception:
        pass  # ~/.codex config missing/unreadable → keep the sane Bedrock defaults above
    os.makedirs(home, exist_ok=True)
    text = (
        f'model_provider = "{d["provider"]}"\n'
        'forced_login_method = "api"\n'
        "check_for_update_on_startup = false\n"
        'commit_attribution = ""\n'
        f'chatgpt_base_url = "{d["base_url"]}"\n'
        f'model_reasoning_effort = "{d["effort"]}"\n'
        f'model = "{d["model"]}"\n\n'
        '[shell_environment_policy]\ninherit = "all"\n\n'
        "[feedback]\nenabled = false\n\n"
        "[analytics]\nenabled = false\n\n"
        f'[model_providers.{d["provider"]}.aws]\n'
        f'region = "{d["region"]}"\nprofile = "{d["profile"]}"\n'
    )
    with open(os.path.join(home, "config.toml"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return home


def _default_runner(argv: list[str], prompt: str, timeout: float, out_path: str) -> str:
    subprocess.run(argv, input=prompt, capture_output=True, text=True, timeout=timeout, check=False)
    with open(out_path, encoding="utf-8") as fh:
        return fh.read()


class CodexCliJudge:
    def __init__(
        self,
        *,
        binary: str = DEFAULT_BINARY,
        model: str | None = None,
        timeout: float = 1500.0,
        runner=_default_runner,
        codex_home: str | None = JUDGE_HOME,
    ):
        self.binary = binary
        self.model = model  # None -> the clean CODEX_HOME's model (mirrors ~/.codex)
        self.timeout = timeout
        self._runner = runner  # injectable for tests
        # Only wire the clean CODEX_HOME for the real subprocess runner — an injected runner
        # (tests) must not trigger filesystem writes or env mutation.
        self.codex_home = codex_home if (codex_home and runner is _default_runner) else None
        if self.codex_home:
            ensure_clean_codex_home(self.codex_home)

    def _argv(self, out_path: str) -> list[str]:
        argv = [self.binary, "exec", "--skip-git-repo-check", "-o", out_path]
        if self.model:
            argv += ["-m", self.model]
        argv.append("-")  # read the prompt from stdin
        return argv

    def invoke(self, prompt: str) -> str:
        fd, out_path = tempfile.mkstemp(suffix=".codex.txt")
        os.close(fd)
        prev = os.environ.get("CODEX_HOME")
        try:
            if self.codex_home:
                os.environ["CODEX_HOME"] = self.codex_home  # lean config: no mcp/plugins
            return self._runner(self._argv(out_path), prompt, self.timeout, out_path).strip()
        finally:
            if self.codex_home:
                if prev is None:
                    os.environ.pop("CODEX_HOME", None)
                else:
                    os.environ["CODEX_HOME"] = prev
            try:
                os.unlink(out_path)
            except OSError:
                pass  # temp prompt file; cleanup is best-effort
