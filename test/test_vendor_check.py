"""scripts/check_vendor_review.py rule_fixes: errors the blind meaning check votes "both" on."""

import importlib.util
from pathlib import Path

from dev_titles.model import Cue

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "check_vendor_review", ROOT / "scripts" / "check_vendor_review.py"
)
cvr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cvr)  # type: ignore[union-attr]


def _cue(text, start=0.0, end=3.0):
    return Cue(start=start, end=end, lines=text.split("\n"))


def test_name_spelled_as_in_english():
    en = [_cue("called Wit's End, which is an AI RPG. I love Nimbus+.")]
    rev = [_cue("打造一款名为Wits End的游戏"), _cue("这是我喜欢Nimbus Plus的一点")]
    got = cvr.rule_fixes(rev, en, "zh-Hans", set())
    assert got[1][1] == "打造一款名为Wit's End的游戏"
    assert got[2][1] == "这是我喜欢Nimbus+的一点"


def test_name_rule_leaves_correct_lines_alone():
    en = [
        _cue("How does one participate in the Expo Dev"),
        _cue("Awards? If you go to expo.dev/awards, run sam init,", 3.0, 6.0),
        _cue("on Apple TV,\ntvOS and Hi there.", 3.0, 6.0),
    ]
    rev = [
        _cue("要怎么才能参与Expo Dev Awards呢？"),  # the name, not the URL
        _cue("现在运行sam init"),  # a command the English has lowercase too
        _cue("在Apple TV tvOS上"),  # two names, not "Apple TV,tvOS"
    ]
    assert cvr.rule_fixes(rev, en, "zh-Hans", set()) == {}


def test_lost_separator_restored():
    en = [_cue("So if I'm using Expo, Expo's Tools or Expo Go")]
    rev = [_cue("如果我用Expo Expo的工具或Expo Go")]
    assert cvr.rule_fixes(rev, en, "zh-Hans", set())[1][1] == "如果我用Expo、Expo的工具或Expo Go"


def test_track_consistency_and_latin_variant_guard():
    en = [_cue("build the Android version using the command npm run android.")]
    rev = [_cue("使用这个挂钩"), _cue("命令npm run android构建Android版本")]
    got = cvr.rule_fixes(rev, en, "zh-Hans", {("挂钩", "Hook"), ("android", "Android")})
    assert got[1][1] == "使用这个Hook"
    assert 2 not in got  # `npm run android` is in the English verbatim


def test_flat_keeps_space_in_a_name_broken_across_lines():
    assert cvr._flat(_cue("打开React Native\nDevTools")) == "打开React Native DevTools"
    assert cvr._flat(_cue("这是第一行\n第二行")) == "这是第一行第二行"
