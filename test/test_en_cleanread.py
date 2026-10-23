"""Clean-read of unscripted EN: deletions / punctuation only; scripted videos are left alone."""
import json

from dev_titles.en_cleanread import clean_read, deletions_only, speech_style, wrap2
from dev_titles.model import Cue, Subtitle


def _en(*texts):
    return Subtitle("en", [Cue(i * 3.0, i * 3.0 + 2.5, [t]) for i, t in enumerate(texts)])


def test_deletions_only_guard():
    assert deletions_only("the standard, in the, in the industry", "the standard in the industry")
    assert deletions_only("and, you stay, hooks.", "And you stay, hooks.")
    assert not deletions_only("we like moving with full speed", "we're moving at full speed")  # reworded
    assert not deletions_only("so it's like, done", "so it's done now")  # added a word
    assert not deletions_only("anything", "")


def test_clean_read_applies_deletions_and_rejects_rewrites():
    en = _en("to, to help that, help us", "we like moving with full speed")
    fake = lambda prompt: json.dumps([{"id": 0, "text": "to help us"}, {"id": 1, "text": "we're moving fast"}])
    out, applied, rejected = clean_read(en, fake)
    assert out.cues[0].lines == ["to help us"] and len(applied) == 1 and rejected == 1
    assert out.cues[1].lines == ["we like moving with full speed"]  # rewrite refused
    assert en.cues[0].lines == ["to, to help that, help us"]  # input untouched
    assert (out.cues[0].start, out.cues[0].end) == (en.cues[0].start, en.cues[0].end)


def test_style_override_and_cache(tmp_path):
    en = _en("Hello.")
    assert speech_style(en, None, override="scripted")["style"] == "scripted"
    p = tmp_path / "en_style.json"
    r = speech_style(en, lambda _: '{"style": "scripted", "why": "tutorial"}', "Getting Started", p)
    assert r["style"] == "scripted" and p.exists()
    assert speech_style(en, lambda _: 1 / 0, "x", p)["style"] == "scripted"  # cached, no call


def test_wrap2_balances_two_lines():
    lines = wrap2("it comes to features, some of them are going to be provided in the core of")
    assert len(lines) == 2 and all(len(l) <= 42 for l in lines)
