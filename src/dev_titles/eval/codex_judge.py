"""Non-Claude model via the Codex CLI (OpenAI GPT models): the second QE judge, or the primary
model with `--llm codex`. Codex uses whatever it is signed in with — a ChatGPT account, an
OpenAI API key, or a custom provider such as AWS Bedrock configured in ~/.codex.

As a judge it de-biases the panel: Claude grading Claude output is inflated by
self-preference.

Non-interactive: `codex exec --skip-git-repo-check -o <file> -` with the prompt on
stdin; the model's final message is written to <file>. `invoke(prompt) -> str`
matches the callable the audit uses, so it drops in for `CliTranslator._invoke`.
Install the Codex CLI and put it on $PATH (or set $CODEX_BINARY).

CLEAN CODEX_HOME: calls run codex under a minimal, self-generated CODEX_HOME
(`~/.codex_judge`) that mirrors the user's model, provider and login but DROPS
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


def ensure_clean_codex_home(home: str = JUDGE_HOME, src_home: str = "~/.codex") -> str:
    """Create/refresh a minimal CODEX_HOME for non-interactive calls: mirror the model and
    provider from the user's real ~/.codex config, but WITHOUT mcp_servers / plugins /
    marketplaces (those hang or bloat codex startup and are irrelevant to a scoring call).

    Two setups are supported:
      - default OpenAI provider (signed in with a ChatGPT account or an OpenAI API key): the
        user's auth.json is linked into the clean home, so the same login is used;
      - a custom provider such as AWS Bedrock (`model_provider` + `[model_providers.<p>.aws]`):
        the provider block is mirrored and API login is forced.
    Idempotent; with no readable config it falls back to codex's own defaults."""
    src = os.path.expanduser(src_home)
    cfg = {}
    try:
        import tomllib

        with open(os.path.join(src, "config.toml"), "rb") as fh:
            cfg = tomllib.load(fh)
    except Exception:
        pass  # no/unreadable config -> codex defaults (OpenAI provider, the user's login)
    os.makedirs(home, exist_ok=True)
    lines = ["check_for_update_on_startup = false", 'commit_attribution = ""']
    if cfg.get("model"):
        lines.append(f'model = "{cfg["model"]}"')
    if cfg.get("model_reasoning_effort"):
        lines.append(f'model_reasoning_effort = "{cfg["model_reasoning_effort"]}"')
    provider = cfg.get("model_provider")
    tail = ['[shell_environment_policy]\ninherit = "all"', "[feedback]\nenabled = false",
            "[analytics]\nenabled = false"]
    if provider and provider != "openai":
        lines += [f'model_provider = "{provider}"', 'forced_login_method = "api"']
        if cfg.get("chatgpt_base_url"):
            lines.append(f'chatgpt_base_url = "{cfg["chatgpt_base_url"]}"')
        pblock = (cfg.get("model_providers") or {}).get(provider) or {}
        aws = pblock.get("aws") or {}
        if aws:
            tail.append(f"[model_providers.{provider}.aws]\n"
                        + "\n".join(f'{k} = "{v}"' for k, v in aws.items() if isinstance(v, str)))
    with open(os.path.join(home, "config.toml"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n\n" + "\n\n".join(tail) + "\n")
    # reuse the user's login (ChatGPT sign-in or stored API key); a symlink keeps token
    # refreshes shared with the user's normal codex
    auth_src, auth_dst = os.path.join(src, "auth.json"), os.path.join(home, "auth.json")
    if os.path.exists(auth_src) and not os.path.lexists(auth_dst):
        try:
            os.symlink(auth_src, auth_dst)
        except OSError:
            pass
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
