"""Speaker turns: labels per sentence, translator notes, no cross-speaker subtitle merges."""

import json

from dev_titles.resegment import Sentence


def _sents():
    return [Sentence("Jane, thanks for coming.", 0, 2), Sentence("Glad to be here.", 2, 3),
            Sentence("Brilliant.", 3, 4), Sentence("Thank you for having me.", 4, 5)]


def test_llm_labels_cached_and_single_speaker_is_noop(tmp_path):
    from dev_titles.speakers import label_speakers

    calls = []

    def inv(p):
        calls.append(p)
        return json.dumps(["Sam (host)", "Jane (guest)", "Sam (host)", "Jane (guest)"])

    cp = tmp_path / "speakers.json"
    got = label_speakers(_sents(), inv, cp, brief="Sam hosts; Jane is the guest.", logger=lambda m: None)
    assert got == ["Sam (host)", "Jane (guest)", "Sam (host)", "Jane (guest)"]
    assert label_speakers(_sents(), inv, cp, brief="Sam hosts; Jane is the guest.") == got
    assert len(calls) == 1  # cached
    mono = label_speakers(_sents(), lambda p: json.dumps(["Danny"] * 4), tmp_path / "m.json")
    assert mono == []


def test_asr_diarization_preferred(tmp_path):
    from dev_titles.speakers import label_speakers

    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps([
        {"text": "Jane,", "start": 0.2, "end": 0.5, "speaker": "A"},
        {"text": "Glad", "start": 2.2, "end": 2.5, "speaker": "B"},
        {"text": "Brilliant.", "start": 3.2, "end": 3.6, "speaker": "A"},
        {"text": "Thank", "start": 4.1, "end": 4.4, "speaker": "B"},
    ]))
    got = label_speakers(_sents(), lambda p: (_ for _ in ()).throw(AssertionError("no LLM")),
                         tmp_path / "s.json", raw_path=raw, logger=lambda m: None)
    assert got == ["A", "B", "A", "B"]


def test_speaker_note_and_labeled_transcript():
    from dev_titles.speakers import labeled_transcript, speaker_note

    labels = ["Sam (host)", "Jane (guest)", "Jane (guest)"]
    assert "different person" in speaker_note(1, labels)
    assert "different person" not in speaker_note(2, labels)
    assert speaker_note(0, []) == ""
    t = labeled_transcript(_sents()[:3], labels)
    assert t.startswith("[Sam (host)] Jane") and t.count("[Jane (guest)]") == 1


def test_short_sentences_never_merge_across_speakers():
    from dev_titles.config import spec_for
    from dev_titles.resegment import merge_short_sentences

    ja = spec_for("ja")
    s = [Sentence("Great.", 0, 1), Sentence("Thanks.", 1, 2), Sentence("Yes.", 2, 3)]
    same = merge_short_sentences(s, ["凄い", "どうも", "はい"], ja, ["A", "A", "A"])
    turns = merge_short_sentences(s, ["凄い", "どうも", "はい"], ja, ["A", "B", "B"])
    assert len(same) == 1 and len(turns) == 2
