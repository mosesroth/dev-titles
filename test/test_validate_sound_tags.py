"""Non-speech sound tags are a validation error (they never ship)."""

from dev_titles.model import Cue, Subtitle
from dev_titles.validate import validate_subtitle


def _rules(lang, text):
    sub = Subtitle(lang, [Cue(0.0, 3.0, [text])])
    return {(v.rule, v.severity) for v in validate_subtitle(sub)}


def test_sound_tags_error_in_every_language():
    assert ("sound_tag", "error") in _rules("en", "[music] Hi, I'm Sam")
    assert ("sound_tag", "error") in _rules("en", "Thanks! (applause)")
    assert ("sound_tag", "error") in _rules("zh-Hans", "[音乐] 大家好")
    assert ("sound_tag", "error") in _rules("ja", "［音楽］こんにちは")


def test_ui_brackets_and_laughter():
    assert not any(
        r == "sound_tag" for r, _ in _rules("ja", "[設定] を開きます")
    )  # JP SG: UI in [ ]
    assert ("sound_tag", "warn") in _rules("ja", "詰め込んでみました（笑）")


def test_detector_covers_real_variants_and_spares_ui_names():
    from dev_titles.cue_builder import _SOUND_TAG

    for t in [
        "[MUSIC PLAYING]",
        "(light music continues)",
        "[applause and cheering]",
        "[ Music ]",
        "[speaking Japanese]",
        "[audience laughs]",
        "(gentle bright music)",
        "[clapping]",
    ]:
        assert _SOUND_TAG.search(t), t
    for t in [
        "[Build]",
        "[OK]",
        "[Sync Now]",
        "Single Sign-On (SSO)",
        "(sound familiar)",
        "[Music app]",
        "array[0]",
        "[[0]]",
    ]:
        assert not _SOUND_TAG.search(t), t


def test_strip_and_validate_speaker_markers():
    from dev_titles.cue_builder import strip_sound_tags

    sub = Subtitle("en", [Cue(0.0, 3.0, [">> Welcome to the show.", "[MUSIC PLAYING]"])])
    assert ("caption_artifact", "error") in {(v.rule, v.severity) for v in validate_subtitle(sub)}
    assert strip_sound_tags(sub).cues[0].lines == ["Welcome to the show."]


def test_bundle_never_writes_sound_tags(tmp_path):
    from dev_titles.review import LangOutput, build_review_bundle

    en = Subtitle("en", [Cue(0.0, 3.0, ["[music] Hi there."]), Cue(3.0, 6.0, ["[applause]"])])
    zh = Subtitle("zh-Hans", [Cue(0.0, 3.0, ["[音乐] 大家好"])])
    build_review_bundle(tmp_path, en, [LangOutput(language="zh-Hans", subtitle=zh)])
    assert (
        "music" not in (tmp_path / "en.vtt").read_text()
        and "applause" not in (tmp_path / "en.vtt").read_text()
    )
    assert "音乐" not in (tmp_path / "zh-Hans.srt").read_text(encoding="utf-8")


def test_zh_line_end_punct_is_an_error():
    assert ("zh_line_end_punct", "error") in _rules("zh-Hans", "我们来聊聊，")
    assert not any(r == "zh_line_end_punct" for r, _ in _rules("zh-Hans", "我们来聊聊"))


def test_empty_cue_is_an_error():
    sub = Subtitle("en", [Cue(0.0, 2.0, ["Hello."]), Cue(2.0, 4.0, [""])])
    assert ("empty_cue", "error") in {(v.rule, v.severity) for v in validate_subtitle(sub)}
