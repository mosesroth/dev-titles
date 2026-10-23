"""EN vs speech timing (validate/speech_sync.py), the EN gate's mechanical checks, cue rebalancing."""

import json

from dev_titles.model import Cue, Subtitle, Transcript, Word
from dev_titles.validate import validate_subtitle
from dev_titles.validate.speech_sync import (
    load_speech_words,
    retime_to_speech,
    speech_sync_report,
)

WORDS = "so you can run it on the device and then we ship the app to every user today".split()


def _speech(offset=0.0):
    return [(w, 10.0 + 0.4 * i + offset) for i, w in enumerate(WORDS)]


def _en(late=0.0):
    # three cues; the 3rd starts `late` s after its first word ("we", word 11 -> 14.4 s)
    return Subtitle(
        "en",
        [
            Cue(9.9, 12.0, ["So you can run it on the device"]),
            Cue(12.0, 14.3 + late, ["and then"]),
            Cue(14.3 + late, 17.5 + late, ["we ship the app to every user today."]),
        ],
    )


def test_late_cue_is_an_error_and_retime_fixes_it():
    rep = speech_sync_report(_en(late=4.0), _speech())
    assert [o["severity"] for o in rep["off"]] == ["error"] and rep["off"][0]["offset"] < -2
    new, moved = retime_to_speech(_en(late=4.0), _speech())
    assert moved and not [
        o for o in speech_sync_report(new, _speech())["off"] if o["severity"] == "error"
    ]
    assert [c.text for c in new.cues] == [c.text for c in _en().cues]  # text untouched
    assert not [v for v in validate_subtitle(new) if v.severity == "error"]


def test_in_sync_track_passes():
    assert speech_sync_report(_en(), _speech())["off"] == []


def test_load_json3_and_karaoke_vtt(tmp_path):
    j = tmp_path / "x.en.json3"
    j.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "tStartMs": 1000,
                        "segs": [{"utf8": "Hello"}, {"utf8": " world", "tOffsetMs": 400}],
                    }
                ]
            }
        )
    )
    assert load_speech_words(j) == [("hello", 1.0), ("world", 1.4)]
    v = tmp_path / "x.en.vtt"
    v.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nhello<00:00:01.400><c> big</c><00:00:01.800><c> world</c>\n"
    )
    assert load_speech_words(v) == [("hello", 1.0), ("big", 1.4), ("world", 1.8)]


def test_duplicate_text_graded_by_length():
    dup = Subtitle(
        "en",
        [
            Cue(0, 3, ["instead of just defining how a product's going to work,"]),
            Cue(3, 6, ["so instead of just defining how a product's going to work writing it"]),
        ],
    )
    assert ("duplicate_text", "error") in {(v.rule, v.severity) for v in validate_subtitle(dup)}
    rhet = Subtitle(
        "en",
        [
            Cue(0, 3, ["what is working, what is"]),
            Cue(3, 6, ["tell us what is working, what is not."]),
        ],
    )
    assert ("duplicate_text", "error") not in {
        (v.rule, v.severity) for v in validate_subtitle(rhet)
    }


def test_gate_blocks_mechanical_defects_even_when_signed_off(tmp_path):
    from dev_titles import en_gate

    en = Subtitle("en", [Cue(0, 3, ["[music] Hi there."])])
    en_gate.save_review_state(tmp_path, "ok", [])
    en_gate.approve(tmp_path, en, by="t")
    ok, why, held = en_gate.check(en, tmp_path)
    assert not ok and held[0]["kind"] == "sound_tag"


def test_builder_rebalances_an_orphan_instead_of_a_third_line():
    from dev_titles.cue_builder import build_en_cues

    text = (
        "You have to spin up full Android Studio, then click start and I've been there already. Yes."
    ).split()
    words = [Word(t, 0.3 * i, 0.3 * i + 0.25) for i, t in enumerate(text)]
    sub = build_en_cues(Transcript("en", words), drop_disfluencies=False)
    assert all(len(c.lines) <= 2 for c in sub.cues)
