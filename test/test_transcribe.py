"""Split from test_core: transcribe tests (16)."""

from helpers import _transcribe_json

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


def test_review_bundle_and_report(tmp_path):
    from dev_titles.glossary import EnforcementIssue
    from dev_titles.review import LangOutput, build_review_bundle, render_review_markdown

    en = Subtitle("en", [Cue(0.0, 2.0, ["Hello world."])])
    ja = Subtitle("ja", [Cue(0.0, 2.0, ["こんにちは"])])
    out = LangOutput(
        language="ja",
        subtitle=ja,
        enforcement_issues=[EnforcementIssue("submit", "ja", "missing", "申請", "dropped")],
        qe_metric="cometkiwi",
        qe_system_score=0.82,
        qe_weak_segments=[0],
    )
    md = render_review_markdown(en, [out])
    assert "## ja" in md and "needs attention" in md and "申請" in md
    assert "score 0.8" in md and "weak segment" in md
    paths = build_review_bundle(tmp_path, en, [out])
    names = {__import__("os").path.basename(p) for p in paths}
    assert {"en.vtt", "ja.vtt", "ja.srt", "REVIEW.md"} <= names
    assert (tmp_path / "REVIEW.md").read_text(encoding="utf-8").startswith("# Subtitle review")


def test_youtube_publisher_requires_credentials():
    import pytest

    from dev_titles.publish import YouTubePublisher

    pub = YouTubePublisher()  # no credentials
    with pytest.raises(RuntimeError, match="credentials"):
        pub.insert_caption("vid123", "ja", "nonexistent.vtt")


def test_transcription_suspects_reference_free():
    """Reference-free QE: combine ASR confidence, gaps, LLM plausibility, and a 2nd-ASR
    diff into one timestamped suspect list — no human caption needed."""
    from dev_titles.model import Transcript, Word
    from dev_titles.transcribe_qe import transcription_suspects

    t = Transcript(
        "en",
        [
            Word("We", 0.0, 0.3, 0.99),
            Word("use", 0.3, 0.6, 0.98),
            Word("Hermes", 0.6, 1.0, 0.20),  # low confidence
            Word("here.", 1.0, 1.3, 0.97),
            Word("Next.", 20.0, 20.4, 0.97),
        ],
    )  # 18.7s gap

    def fake(prompt):
        return '[{"phrase":"Hermes here","reason":"possible mishear"}]'

    sus = transcription_suspects(t, fake)
    kinds = {s["kind"] for s in sus}
    assert "low-confidence" in kinds and "gap" in kinds and "implausible" in kinds
    # 2nd-ASR disagreement path
    t2 = Transcript(
        "en",
        [
            Word("We", 0.0, 0.3),
            Word("use", 0.3, 0.6),
            Word("capital", 0.6, 1.0),
            Word("here.", 1.0, 1.3),
            Word("Next.", 20.0, 20.4),
        ],
    )
    sus2 = transcription_suspects(t, None, second_transcript=t2)
    assert any(s["kind"] == "asr-disagreement" for s in sus2)


def test_avoid_correction_no_cjk_corruption_ascii_target_driven():
    """Regression: a dangerous short-CJK per-video avoid ('Native'->ネイティブ avoid '生')
    must NOT corrupt unrelated words (発生->発ネイティブ) — dropped in renderings_to_terms
    and source-gated in enforce. ASCII avoid-forms stay target-driven (Strict Mode)."""
    from dev_titles.vocab_prep import renderings_to_terms

    terms = renderings_to_terms(
        {
            "ja": {
                "Native": {"canonical": "ネイティブ", "avoid": ["生"]},
                "side effect": {"canonical": "副作用", "avoid": ["サイドエフェクト"]},
            }
        }
    )
    av = {t.source: t.avoid.get("ja", []) for t in terms}
    assert "生" not in av.get("Native", [])  # short CJK avoid dropped
    assert "サイドエフェクト" in av.get("side effect", [])  # long CJK avoid kept
    g = load("glossary/glossary.yaml")
    assert g.enforce("遅延が発生する", "x", "ja")[0] == "遅延が発生する"  # not corrupted
    assert g.enforce("最後にStrict Modeは", "x", "ja")[0].startswith(
        "最後にStrictMode"
    )  # ASCII target-driven


def test_whisperx_result_parser():
    from dev_titles.transcribe import from_whisperx_result

    result = {
        "segments": [
            {
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.4, "score": 0.9},
                    {"word": "world", "start": 0.5, "end": 0.9, "score": 0.8},
                ]
            },
            {"words": [{"word": "again", "start": None, "end": None}]},  # unaligned -> skipped
        ]
    }
    t = from_whisperx_result(result)
    assert [w.text for w in t.words] == ["Hello", "world"]
    assert abs(t.words[0].start) < 1e-9 and abs(t.words[1].end - 0.9) < 1e-9


def test_assemblyai_to_transcript():
    from dev_titles.transcribe.assemblyai_cloud import _to_transcript

    payload = {
        "words": [
            {"text": "Acme", "start": 1000, "end": 1400, "confidence": 0.99},
            {"text": "Q", "start": 1400, "end": 1600, "confidence": 0.98},
            {"text": "  ", "start": 1600, "end": 1650},  # blank -> skipped
            {"text": "rocks", "start": 1700, "end": 2000, "confidence": None},
        ]
    }
    t = _to_transcript(payload)
    assert [w.text for w in t.words] == ["Acme", "Q", "rocks"]
    assert abs(t.words[0].start - 1.0) < 1e-9 and abs(t.words[1].end - 1.6) < 1e-9  # ms -> s
    assert t.words[0].confidence == 0.99


def test_detect_domains_and_protection():
    from dev_titles.qa import detect_domains, missing_numbers_or_idents, protected_spans

    src = (
        "Visit developer.acme.com and code.acme.com/packages for docs, "
        "also https://developer.android.com/guide works."
    )
    doms = detect_domains(src)
    assert "developer.acme.com" in doms
    assert any(d.startswith("code.acme.com/packages") for d in doms)
    assert any("developer.android.com" in d for d in doms)
    # NOT domains: a filename, Node.js, a version number
    assert detect_domains("open guidelines.md with Node.js on 1.2.3") == []
    # CJK-glued domain still detected, path/CJK not swallowed
    assert "developer.acme.com" in detect_domains("我们还有developer.acme.com网站")
    # protected_spans unions CLI commands + domains
    assert "developer.acme.com" in protected_spans("run npm install at developer.acme.com")
    # a mangled domain in the target is flagged as dropped (developer.艾克米.com != source)
    assert "developer.acme.com" in missing_numbers_or_idents(
        "See developer.acme.com", "请见 developer.艾克米.com"
    )


def test_review_surfaces_residuals():
    from dev_titles.review import LangOutput, render_review_markdown

    en = Subtitle("en", [Cue(0.0, 2.0, ["Hello."])])
    out = LangOutput(
        "ja",
        Subtitle("ja", [Cue(0.0, 2.0, ["こんにちは"])]),
        residuals=[("the network calls download images", "画像をダウンロード")],
    )
    md = render_review_markdown(en, [out])
    assert "Coverage residuals" in md and "still missing" in md
    assert "the network calls download images" in md


def test_cleanup_polish_subtitle():
    from dev_titles.cleanup import build_cleanup_prompt, polish_subtitle

    assert "English" in build_cleanup_prompt("um hello", "en")

    # Fake LLM that strips a leading "um " and capitalizes.
    def fake(prompt):
        line = prompt.rsplit("Line:", 1)[-1].strip()
        return line.replace("um ", "").capitalize()

    sub = Subtitle("en", [Cue(0.0, 2.0, ["um hello world"])])
    out = polish_subtitle(sub, fake)
    assert out.cues[0].lines == ["Hello world"]
    assert out.cues[0].start == 0.0 and out.cues[0].end == 2.0  # timing preserved


def test_transcribe_json_parser():
    data = _transcribe_json([("Hello", 0.0, 0.4), ("world", 0.5, 0.9)])
    data["results"]["items"].append({"type": "punctuation", "alternatives": [{"content": "."}]})
    t = from_transcribe_json(data)
    assert len(t.words) == 2
    assert t.words[1].text == "world."  # punctuation attached to prior word
    assert abs(t.words[0].start - 0.0) < 1e-9 and abs(t.words[1].end - 0.9) < 1e-9


def test_ensure_clean_codex_home_mirrors_model_strips_mcp_plugins():
    """The judge's CODEX_HOME mirrors the user's model + Bedrock provider but DROPS
    mcp_servers / plugins / marketplaces (those hang/bloat codex startup)."""
    import os
    import tempfile

    from dev_titles.eval.codex_judge import ensure_clean_codex_home

    d = tempfile.mkdtemp()
    src = os.path.join(d, "src")
    os.makedirs(src)
    with open(os.path.join(src, "config.toml"), "w") as fh:
        fh.write(
            'model = "openai.gpt-5.6-sol"\nmodel_reasoning_effort = "low"\n'
            'model_provider = "amazon-bedrock"\n'
            '[mcp_servers.some-mcp]\ncommand = "some-mcp"\n'
            '[plugins."X-all@market"]\nenabled = true\n'
            '[model_providers.amazon-bedrock.aws]\nregion = "us-east-2"\nprofile = "p"\n'
        )
    home = os.path.join(d, "judge")
    ensure_clean_codex_home(home, src_home=src)
    out = open(os.path.join(home, "config.toml")).read()
    assert 'model = "openai.gpt-5.6-sol"' in out  # mirrored model
    assert 'profile = "p"' in out and "us-east-2" in out  # mirrored aws provider
    assert "mcp_servers" not in out and "[plugins" not in out and "marketplaces" not in out
    assert 'forced_login_method = "api"' in out  # custom provider -> API login
    # no readable config -> codex's own defaults (OpenAI provider, the user's login), no Bedrock
    home2 = os.path.join(d, "judge2")
    ensure_clean_codex_home(home2, src_home=os.path.join(d, "nonexistent"))
    out2 = open(os.path.join(home2, "config.toml")).read()
    assert "model_provider" not in out2 and "forced_login_method" not in out2 and "mcp_servers" not in out2


def test_clean_codex_home_keeps_chatgpt_login():
    """A ChatGPT-subscriber setup (default OpenAI provider, login in auth.json) keeps working:
    no forced API login, no provider override, and the user's auth.json is linked in."""
    import os
    import tempfile

    from dev_titles.eval.codex_judge import ensure_clean_codex_home

    d = tempfile.mkdtemp()
    src = os.path.join(d, "src")
    os.makedirs(src)
    with open(os.path.join(src, "config.toml"), "w") as fh:
        fh.write('model = "gpt-5.5"\n[mcp_servers.x]\ncommand = "x"\n')
    with open(os.path.join(src, "auth.json"), "w") as fh:
        fh.write('{"auth_mode": "chatgpt"}')
    home = os.path.join(d, "judge")
    ensure_clean_codex_home(home, src_home=src)
    out = open(os.path.join(home, "config.toml")).read()
    assert 'model = "gpt-5.5"' in out and "forced_login_method" not in out and "model_provider" not in out
    assert "mcp_servers" not in out
    assert open(os.path.join(home, "auth.json")).read() == '{"auth_mode": "chatgpt"}'


def test_correct_transcript_completeness_guard():
    """If the correction drops a whole clause (result dramatically shorter), keep the
    ORIGINAL chunk rather than the truncated one."""
    from dev_titles.cleanup import correct_transcript

    src = "I think that will happen less and less over time as the tools get better honestly."
    out = correct_transcript(
        lambda p: "I think.", src, ["Expo"], clean_read=True
    )  # fake drops content
    assert "less and less" in out and "tools get better" in out  # original content preserved


def test_retime_corrected_substitution_and_merge():
    from dev_titles.model import Transcript, Word
    from dev_titles.transcribe.retime import retime_corrected

    orig = Transcript(
        "en",
        [
            Word("Install", 0, 1),
            Word("npn", 1, 2),
            Word("the", 2, 3),
            Word("next", 3, 4),
            Word("focus", 4, 5),
            Word("right", 5, 6),
            Word("prop", 6, 7),
        ],
    )
    out = retime_corrected(orig, "Install npm the nextFocusRight prop")
    by = {w.text: (w.start, w.end) for w in out.words}
    assert by["npm"] == (1.0, 2.0)  # substitution inherits timing
    assert by["nextFocusRight"] == (3.0, 6.0)  # merge spans next..right
    assert by["prop"] == (6.0, 7.0)  # tail unchanged


def test_transcription_gaps_flags_long_silence():
    from dev_titles.cue_builder import transcription_gaps
    from dev_titles.model import Transcript, Word

    t = Transcript("en", [Word("hello", 0, 1), Word("world", 1, 2), Word("later", 30, 31)])
    g = transcription_gaps(t, max_gap=12.0)
    assert len(g) == 1 and g[0]["after"] == "world" and g[0]["gap"] == 28.0


def test_correction_prompt_includes_suspects():
    from dev_titles.cleanup import build_correction_prompt

    p = build_correction_prompt("we use the tracker", ["Shaka"], suspect_words=["tracker"])
    assert "LOW-CONFIDENCE" in p and "tracker" in p
    # no suspects -> no suspect block
    assert "LOW-CONFIDENCE" not in build_correction_prompt("x", ["Shaka"])


def test_incremental_review_reuses_unchanged(tmp_path):
    """Automatic re-review: with a saved QE state and NO text changes, incremental_review
    reproduces the full review (DQF + MQM tiers) WITHOUT invoking the judge — only changed
    segments get re-judged. This is what makes a post-cache-mutation re-review cheap."""
    from dev_titles.glossary import load as load_glossary
    from dev_titles.localize import SentenceRecord
    from dev_titles.translation_qe import incremental_review, load_qe_state, save_qe_state

    g = load_glossary("glossary/glossary.yaml")
    recs = [
        SentenceRecord("zh-Hans", 0, "Hello.", "你好", "", []),
        SentenceRecord("zh-Hans", 1, "The tool is great.", "工具很棒", "", []),
    ]
    dqf = {
        "score": 98.0,
        "pass": True,
        "counts": {},
        "weak": [],
        "judges": ["claude"],
        "per_seg": {0: {}, 1: {"claude": {"critical": 0, "major": 0, "minor": 1}}},
    }
    mqm = [{"kind": "mqm-minor", "i": 1, "detail": "awkward"}]
    p = tmp_path / "qe_state.zh-Hans.json"
    save_qe_state(p, recs, dqf, mqm)
    st = load_qe_state(p)
    assert st and len(st["segs"]) == 2  # source-keyed state (was index-keyed "by_i")
    called = []
    judges = [("claude", lambda *a, **k: called.append(1) or {})]
    (suspects, dqf2), chg = incremental_review(
        recs, g, "zh-Hans", judges=judges, lang_name="Simplified Chinese", prior_state=st
    )
    assert chg == [] and not called  # nothing changed → judge never called
    assert any(s["i"] == 1 and s["kind"] == "mqm-minor" for s in suspects)  # prior tier reproduced
    # now change segment 1's text → it (and only it) is flagged as changed
    recs[1].txt = "这个工具棒极了"
    _, chg2 = incremental_review(
        recs, g, "zh-Hans", judges=judges, lang_name="Simplified Chinese", prior_state=st
    )
    assert chg2 == [1]


def test_dedup_rolling_cues_and_caption_adapter():
    # Human/YouTube captions "roll": the same line is emitted twice with split timings. The
    # harness must merge those before proofreading, and adapt cues->words for the clean-read path.
    from dev_titles.cue_builder import caption_to_transcript, dedup_rolling_cues
    from dev_titles.model import Cue, Subtitle

    sub = Subtitle(
        "en",
        [
            Cue(0.0, 2.0, ["hello there"]),
            Cue(2.0, 3.0, ["hello there"]),  # rolling duplicate of the previous cue
            Cue(3.0, 5.0, ["next line here"]),
        ],
    )
    d = dedup_rolling_cues(sub)
    assert len(d.cues) == 2  # duplicate merged away
    assert d.cues[0].start == 0.0 and d.cues[0].end == 3.0  # first cue's span extended over dup
    assert d.cues[1].lines == ["next line here"]

    tr = caption_to_transcript(d)
    assert [w.text for w in tr.words] == ["hello", "there", "next", "line", "here"]
    assert tr.words[0].start == 0.0 and tr.words[-1].end <= 5.0 + 1e-6
    assert all(w.start < w.end for w in tr.words)  # every word has a positive span
