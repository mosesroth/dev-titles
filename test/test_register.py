"""Register handling: auto-detect conversational for interviews (#1) + validate 您/你 (#3)."""

from types import SimpleNamespace

from dev_titles.cli import _auto_conversational, _conversational
from dev_titles.config import spec_for
from dev_titles.model import Cue
from dev_titles.validate.rules import validate_cue


def test_auto_conversational_two_speakers():
    assert _auto_conversational({"names": ["Emerson Vale", "James Whitfield"]}) is True


def test_auto_conversational_single_speaker_monologue():
    assert _auto_conversational({"names": ["Dana"]}) is False
    assert _auto_conversational({"names": []}) is False


def test_explicit_flags_override_autodetect():
    two = {"names": ["A", "B"]}
    one = {"names": ["A"]}
    # --conversational forces on even for a monologue
    assert (
        _conversational(SimpleNamespace(conversational=True, no_conversational=False), one) is True
    )
    # --no-conversational forces off even for an interview
    assert (
        _conversational(SimpleNamespace(conversational=False, no_conversational=True), two) is False
    )
    # neither flag -> auto-detect
    assert (
        _conversational(SimpleNamespace(conversational=False, no_conversational=False), two) is True
    )


def _cue(text):
    return Cue(start=0.0, end=3.0, lines=[text], index=1)


def test_polite_you_accepted_in_conversational_zh():
    """Native-reviewer verdict: 您 is the house default in BOTH registers (reviewers
    changed our conversational 你 -> 您 141:4 across four videos), so 您 must NOT be flagged."""
    spec = spec_for("zh-Hans")
    v = validate_cue(_cue("您好吗"), spec, conversational=True)
    assert not any("you" in x.rule for x in v)


def test_informal_you_flagged_in_formal_zh():
    spec = spec_for("zh-Hans")
    v = validate_cue(_cue("你好吗"), spec, conversational=False)
    assert any(x.rule == "zh_informal_you" for x in v)


def test_singular_you_flagged_but_plural_allowed():
    """Singular 你 is flagged (house register is 您); the PLURAL 你们 is fine — 您们 is unidiomatic."""
    spec = spec_for("zh-Hans")
    assert any(
        x.rule == "zh_informal_you" for x in validate_cue(_cue("你好吗"), spec, conversational=True)
    )
    assert not any(
        "you" in x.rule for x in validate_cue(_cue("你们都可以试试"), spec, conversational=True)
    )
