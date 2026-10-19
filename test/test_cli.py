"""Split from test_core: cli tests (5)."""

from dev_titles.config import spec_for
from dev_titles.cue_builder import build_en_cues
from dev_titles.eval import (
    GembaMqmScorer,
    QeResult,
    QeScorer,
    below_threshold_segments,
    parse_gemba_mqm,
    rank_engines,
)
from dev_titles.formats import read_vtt, write_srt, write_vtt
from dev_titles.glossary import load
from dev_titles.linguistics import get_joiner, is_repeated_text
from dev_titles.model import Cue, Subtitle
from dev_titles.segment import apply_kinsoku, break_lines, normalize_zh_punct
from dev_titles.textwidth import display_width
from dev_titles.transcribe import from_transcribe_json
from dev_titles.translate import (
    TranslationRequest,
    Translator,
    build_translate_prompt,
    translate_segment,
)
from dev_titles.validate import validate_subtitle
from dev_titles.validate.rules import validity_report


def test_run_pilot_multi_engine_ranks(tmp_path):
    from dev_titles.eval import QeResult
    from dev_titles.pilot import Clip, EngineSpec, run_pilot

    g = load("glossary/glossary.yaml")
    en = Subtitle(
        "en", [Cue(0.0, 2.0, ["Submit your app."]), Cue(2.5, 4.5, ["Publish your VS Code extension."])]
    )
    clip = Clip("clip1", en)

    class _Eng(Translator):
        def __init__(self, out):
            self.out = out

        def translate(self, req):
            return self.out

        def reflect_and_refine(self, req, draft):
            return self.out

    engines = [EngineSpec("good", _Eng("提交")), EngineSpec("bad", _Eng("x"))]

    class _QE(QeScorer):
        # 'good' engine scores high, 'bad' low — deterministic by hypothesis text.
        def score(self, sources, hyps, s, t):
            val = 0.9 if hyps and hyps[0] == "提交" else 0.3
            return QeResult([val] * len(hyps), val, "fake")

    report = run_pilot(
        clip and [clip],
        engines,
        g,
        target_langs=("zh-Hans",),
        qe_scorer=_QE(),
        qe_threshold=0.5,
        out_dir=tmp_path,
    )
    ranked = report.rankings()["zh-Hans"]
    assert ranked[0][0] == "good" and ranked[-1][0] == "bad"
    # report + bundles written
    assert (tmp_path / "PILOT_REPORT.md").exists()
    assert (tmp_path / "clip1" / "good" / "zh-Hans.vtt").exists()
    md = (tmp_path / "PILOT_REPORT.md").read_text(encoding="utf-8")
    assert "Engine ranking" in md and "good" in md


def test_cli_translator_argv_and_stdin():
    from dev_titles.translate import CliTranslator, TranslationRequest

    seen = {}

    def fake_runner(argv, prompt, timeout, env):
        seen["argv"], seen["prompt"] = argv, prompt
        return "  提交您的应用  \n"

    # No model_id -> --model omitted (CLI default, auto-upgrades).
    tr = CliTranslator(binary="/bin/claude", runner=fake_runner, env={"ANTHROPIC_API_KEY": "k"})
    out = tr.translate(TranslationRequest("Submit your app", "en", "zh-Hans"))
    assert out == "提交您的应用"  # stripped
    assert seen["argv"] == ["/bin/claude", "-p", "--output-format", "text", "--bare"]  # API key -> bare
    assert "--model" not in seen["argv"]
    assert "<TRANSLATE_THIS>" in seen["prompt"]  # prompt goes on stdin

    tr2 = CliTranslator(binary="/bin/claude", model_id="claude-opus-5", runner=fake_runner, env={})
    tr2.translate(TranslationRequest("x", "en", "ja"))
    assert seen["argv"][-2:] == ["--model", "claude-opus-5"]
    # subscription login (no key / cloud provider): bare mode can't read it -> safe mode
    assert "--safe-mode" in seen["argv"] and "--bare" not in seen["argv"]


def test_cli_translator_falls_back_when_safe_mode_unknown():
    from dev_titles.translate import CliTranslator

    calls = []

    def runner(argv, prompt, timeout, env):
        calls.append(list(argv))
        if "--safe-mode" in argv:
            raise RuntimeError("claude CLI exited 1: error: unknown option '--safe-mode'")
        return "ok"

    tr = CliTranslator(binary="/bin/claude", runner=runner, env={})
    assert tr._invoke("x") == "ok" and tr.mode == "plain"
    assert calls[-1] == ["/bin/claude", "-p", "--output-format", "text"]
    assert CliTranslator(env={"DEV_TITLES_CLAUDE_MODE": "bare"}).mode == "bare"
    assert CliTranslator(env={"CLAUDE_CODE_USE_BEDROCK": "1"}).mode == "bare"


def test_detect_cli_commands():
    from dev_titles.qa import detect_cli_commands, missing_numbers_or_idents

    src = (
        "In the terminal we'll use npx create-expo-app, then npm run android, and "
        "npx expo run:android. Export your JAVA_HOME variable and run adb devices -l. "
        "We use the react-native-tvos fork. Keep it up-to-date for high-quality apps."
    )
    cmds = detect_cli_commands(src)
    assert "create-expo-app" in cmds and "run:android" in cmds
    assert "react-native-tvos" in cmds and "JAVA_HOME" in cmds
    assert any(c.startswith("npm run") for c in cmds)  # leader phrase
    assert any(c.startswith("adb devices") for c in cmds)
    # common English hyphenated words are NOT force-kept
    assert "up-to-date" not in cmds and "high-quality" not in cmds
    # dropped command is flagged by the omission detector (single-token only)
    miss = missing_numbers_or_idents(
        src, "ターミナルでアプリを作成します"
    )  # target drops everything
    assert "create-expo-app" in miss and "run:android" in miss


def test_cli_translator_retries_on_timeout():
    import subprocess

    from dev_titles.translate import CliTranslator, TranslationRequest

    calls = {"n": 0}

    def flaky(argv, prompt, timeout, env):
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.TimeoutExpired(argv, timeout)  # first call hangs
        return "翻訳"

    tr = CliTranslator(runner=flaky, retries=2)
    assert tr.translate(TranslationRequest("hi", "en", "ja")) == "翻訳"
    assert calls["n"] == 2  # retried once after the timeout


def test_codex_cli_judge_argv_and_output():
    from dev_titles.eval.codex_judge import CodexCliJudge

    seen = {}

    def fake_runner(argv, prompt, timeout, out_path):
        seen["argv"], seen["prompt"] = argv, prompt
        return "  PARITY: 84\n\nanalysis...  "

    j = CodexCliJudge(binary="/bin/codex", model="openai.gpt-5.5", runner=fake_runner)
    out = j.invoke("judge this")
    assert out == "PARITY: 84\n\nanalysis..."  # stripped, from -o file
    assert seen["argv"][:4] == ["/bin/codex", "exec", "--skip-git-repo-check", "-o"]
    assert seen["argv"][-3:] == ["-m", "openai.gpt-5.5", "-"]  # model + stdin sentinel
    assert seen["prompt"] == "judge this"
