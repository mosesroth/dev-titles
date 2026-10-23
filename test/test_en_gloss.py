"""Meaning notes: idioms / ambiguous phrases resolved once in the English and attached per sentence."""
import json

from dev_titles.en_gloss import meaning_note, meaning_notes, meaning_tag
from dev_titles.resegment import Sentence


def _sents(*texts):
    return [Sentence(t, i * 3.0, i * 3.0 + 2.5) for i, t in enumerate(texts)]


def test_notes_kept_only_when_the_phrase_is_in_that_sentence(tmp_path):
    s = _sents("Bring it home.", "And there are two ways of having local models.")
    fake = lambda _: json.dumps([{"id": 0, "phrase": "Bring it home", "meaning": "make it concrete"},
                                 {"id": 1, "phrase": "boil the ocean", "meaning": "invented"}])
    n = meaning_notes(s, fake, tmp_path / "mn.json")
    assert list(n) == [0] and n[0][0]["meaning"] == "make it concrete"
    assert meaning_notes(s, lambda _: 1 / 0, tmp_path / "mn.json") == n  # cached


def test_prompt_and_editor_forms():
    n = {3: [{"phrase": "Bring it home", "meaning": "make it concrete"}]}
    assert "translate the meaning" in meaning_note(3, n) and meaning_note(4, n) == ""
    assert meaning_tag(3, n) == " [means: Bring it home = make it concrete]"
