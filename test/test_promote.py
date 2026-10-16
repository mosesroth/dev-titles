"""Tests for the cross-video term-pin ledger + auto-promotion gate (promote.py)."""

from dev_titles.promote import (
    promote_recurring_terms,
    promotion_renderings,
    record_pins,
)


def _pin(source, wrong, correct):
    return {"source": source, "wrong": wrong, "correct": correct, "cues": [1]}


def test_record_pins_appends_and_dedups_within_video(tmp_path):
    led = tmp_path / "ledger.jsonl"
    n1 = record_pins(led, [_pin("Apple TV", "Apple电视", "Apple TV")], "zh-Hans", "vidA")
    # re-recording the SAME (video, lang, source) must not add a second line
    n2 = record_pins(led, [_pin("Apple TV", "Apple电视", "Apple TV")], "zh-Hans", "vidA")
    assert n1 == 1 and n2 == 0
    assert len([ln for ln in led.read_text(encoding="utf-8").splitlines() if ln.strip()]) == 1


def test_promotion_requires_two_distinct_videos(tmp_path):
    led = tmp_path / "ledger.jsonl"
    record_pins(led, [_pin("Lumora", "卢莫拉", "Lumora")], "zh-Hans", "vidA")
    # one video only -> not corroborated
    assert promotion_renderings(led, min_videos=2) == {}
    # a second DISTINCT video with the SAME rendering -> promotes
    record_pins(led, [_pin("Lumora", "卢莫拉", "Lumora")], "zh-Hans", "vidB")
    r = promotion_renderings(led, min_videos=2)
    assert r["zh-Hans"]["Lumora"]["canonical"] == "Lumora"
    assert "卢莫拉" in r["zh-Hans"]["Lumora"]["avoid"]


def test_conflicting_rendering_is_not_promoted(tmp_path):
    led = tmp_path / "ledger.jsonl"
    # two videos disagree on the correct rendering -> sense-dependent, leave for reviewer
    record_pins(led, [_pin("agent", "特工", "智能体")], "zh-Hans", "vidA")
    record_pins(led, [_pin("agent", "智能体", "代理")], "zh-Hans", "vidB")
    assert promotion_renderings(led, min_videos=2) == {}


def test_languages_judged_independently(tmp_path):
    led = tmp_path / "ledger.jsonl"
    # zh corroborated across 2 videos; ja only in 1 -> only zh promotes
    record_pins(led, [_pin("Brightly", "布赖特利", "Brightly")], "zh-Hans", "vidA")
    record_pins(led, [_pin("Brightly", "布赖特利", "Brightly")], "zh-Hans", "vidB")
    record_pins(led, [_pin("Brightly", "ブライトリー", "Brightly")], "ja", "vidA")
    r = promotion_renderings(led, min_videos=2)
    assert "Brightly" in r.get("zh-Hans", {})
    assert "ja" not in r or "Brightly" not in r["ja"]


_MIN_GLOSSARY = """version: 0.1
languages: [zh-Hans, ja]
terms:
  - source: "Existing"
    zh-Hans: "现有"
    ja: "既存"
    enforce: hard
style:
  zh-Hans:
    register: "x"
  ja:
    register: "y"
"""


def test_promote_recurring_terms_appends_new_and_skips_existing(tmp_path):
    led = tmp_path / "ledger.jsonl"
    gloss = tmp_path / "glossary.yaml"
    gloss.write_text(_MIN_GLOSSARY, encoding="utf-8")
    # a NEW recurring term (2 videos) + an EXISTING one (should be skipped)
    for v in ("vidA", "vidB"):
        record_pins(led, [_pin("Halvorsen", "", "哈尔沃森")], "zh-Hans", v)
        record_pins(led, [_pin("Existing", "", "新译")], "zh-Hans", v)
    promoted = promote_recurring_terms(led, gloss, min_videos=2)
    assert "Halvorsen" in promoted
    assert "Existing" not in promoted  # curated glossary already owns it
    assert "Halvorsen" in gloss.read_text(encoding="utf-8")
