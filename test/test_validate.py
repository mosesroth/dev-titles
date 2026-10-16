"""Split from test_core: validate tests (8)."""

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


def test_validate_flags_ja_punctuation_and_zh_informal():
    from dev_titles.validate import validate_subtitle

    ja = Subtitle("ja", [Cue(0.0, 3.0, ["これはテストです。"])])
    assert any(v.rule == "ja_punctuation" and v.severity == "error" for v in validate_subtitle(ja))
    zh = Subtitle("zh-Hans", [Cue(0.0, 3.0, ["你可以使用它"])])
    # Singular 你 is flagged in BOTH registers (house default is 您, per the native reviewers).
    assert any(v.rule == "zh_informal_you" for v in validate_subtitle(zh))
    assert any(v.rule == "zh_informal_you" for v in validate_subtitle(zh, conversational=True))
    # ...but the PLURAL 你们 is idiomatic and must not be flagged.
    plural = Subtitle("zh-Hans", [Cue(0.0, 3.0, ["你们都可以试试"])])
    assert not any(
        v.rule == "zh_informal_you" for v in validate_subtitle(plural, conversational=True)
    )


def test_kinsoku_no_cpl_overflow_and_no_forbidden_start():
    from dev_titles.config import KINSOKU_NO_LINE_START
    from dev_titles.segment import wrap_lines

    spec = spec_for("ja")  # cpl 20
    # 20 full-width chars then a closing 。 : must NOT overflow width AND must NOT
    # leave 。 at a line start (the swap fix).
    lines = wrap_lines("あ" * 20 + "。" + "い" * 5, spec)
    assert len(lines) >= 2
    for ln in lines:
        assert display_width(ln, cjk=True) <= spec.max_cpl
    for ln in lines[1:]:
        assert ln[0] not in KINSOKU_NO_LINE_START


def test_missing_numbers_recognizes_cjk_forms():
    from dev_titles.qa import missing_numbers_or_idents as m

    assert (
        m("serving 130,000 business partners", "13万のビジネスパートナー") == []
    )  # 万 myriad (ja)
    assert m("over 130,000 partners", "超过13万个合作伙伴") == []  # 万 (zh)
    assert m("that took 1.5 to two years", "1年半から2年") == []  # 半 half (ja)
    assert m("1.5 to two years", "一年半到两年") == []  # 半 (zh, CJK int)
    assert m("100,000,000 users", "1亿用户") == []  # 亿
    # a genuinely dropped number is STILL flagged
    assert "58" in m("latency was 58 milliseconds", "遅延がありました")


def test_break_lines_zh_respects_cpl():
    spec = spec_for("zh-Hans")  # max_cpl 16
    text = "这是一个用于测试的中文字幕行需要在正确的位置换行处理"
    lines = break_lines(text, spec)
    assert all(display_width(ln, cjk=True) <= spec.max_cpl for ln in lines)
    assert len(lines) <= spec.max_lines


def test_validate_flags_cps_and_cpl():
    # 30 full-width chars in 1 second, single line -> cpl + cps errors for ja.
    cue = Cue(
        start=0.0, end=1.0, lines=["ああああああああああああああああああああああああああああ"]
    )
    sub = Subtitle("ja", [cue])
    violations = validate_subtitle(sub)
    err_rules = {v.rule for v in violations if v.severity == "error"}
    warn_rules = {v.rule for v in violations if v.severity == "warn"}
    assert "max_cpl" in err_rules
    assert "max_cps" in warn_rules  # cps is a warning (completeness-first policy)


def test_validity_report():
    sub = Subtitle(
        "ja",
        [
            Cue(0.0, 1.0, ["あ" * 25]),  # 25 > cpl 20 -> cpl error (cps is warn-only now)
            Cue(2.0, 4.0, ["短い"]),
        ],
    )
    rep = validity_report(sub)
    assert rep["total_cues"] == 2
    assert rep["cues_with_errors"] >= 1
    assert "violations_by_rule" in rep


def test_missing_numbers_or_idents():
    from dev_titles.qa import missing_numbers_or_idents

    src = "UI Manager took 58 milliseconds and performMountingInstructions took 235 microseconds"
    miss = missing_numbers_or_idents(src, "performMountingInstructions は235マイクロ秒かかりました")
    assert "58" in miss  # dropped measurement caught
    assert "performMountingInstructions" not in miss  # identifier present
    assert missing_numbers_or_idents("took 58 ms", "58ミリ秒") == []  # localized unit ok


def test_missing_numbers_scale_words_and_cjk_units():
    """Scale words fold into the magnitude so CJK myriad renderings aren't false-flagged as
    dropped: '50/100/200 billion' -> 500億/1000億/2000億 (ja) and 亿 (zh); '1 trillion' ->
    1兆 / 一万亿. A genuinely dropped scaled number still flags."""
    from dev_titles.qa import missing_numbers_or_idents as m

    assert (
        m(
            "most orgs do 50 billion, 100 billion, maybe 200 billion",
            "ほとんどの組織は500億や1000億、大企業でも2000億でしょう",
        )
        == []
    )  # ja 億
    assert m("we did 50 billion", "我们做了500亿") == []  # zh 亿
    assert m("we processed 1 trillion tokens", "1兆トークンを処理しました") == []  # ja 兆
    assert m("we processed 1 trillion tokens", "我们处理了一万亿个token") == []  # zh 万亿
    assert "50000000000" in m("we did 50 billion", "我们做了很多")  # truly dropped still flags


def test_stray_ascii_period_flagged_in_cjk():
    """Reviewer-edit artifact: a lone ASCII '.' in CJK text is an error; Latin tokens
    (Next.js, v1.2) and ellipses are not."""
    from dev_titles.config import spec_for
    from dev_titles.validate.rules import validate_cue

    spec = spec_for("ja")
    flag = lambda t: any(
        v.rule == "stray_ascii_period" for v in validate_cue(Cue(0.0, 2.0, [t]), spec)
    )
    assert flag(".良い質問ですね") and flag("使えます.")
    assert not flag("Next.jsを使います") and not flag("v1.2です") and not flag("待って...")
