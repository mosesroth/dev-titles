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
