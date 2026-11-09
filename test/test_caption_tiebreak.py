"""caption_tiebreak: YouTube auto-captions as evidence for held likely-mishears."""

import json

from dev_titles import caption_tiebreak as ct
from dev_titles.model import Cue, Subtitle, Transcript, Word


def _en(*lines):
    return Subtitle(
        "en", [Cue(2.0 * k, 2.0 * k + 2.0, [t]) for k, t in enumerate(lines)]
    ).renumber()


def _judge(line, changed=True):
    return lambda prompt: json.dumps({"line": line, "changed": changed})


def _item(k, ts, kind="en-garbled"):
    return {"ts": ts, "kind": kind, "detail": f"[cue {k}] looks misheard"}


def test_resolves_a_mishear_both_models_agree_on_and_the_captions_heard():
    en = _en("Thanks.", "getting stuff done into NWA for customers.")
    caps = Subtitle("en", [Cue(2.0, 4.0, ["getting stuff done in end-to-end way for customers"])])
    fix = "getting stuff done in an end-to-end way for customers."
    second = Transcript("en", [Word("an", 2.5, 2.6)])
    en, res, still = ct.resolve(en, [_item(1, "00:02")], caps, second, [_judge(fix), _judge(fix)])
    assert [r["now"] for r in res] == [fix] and not still
    assert " ".join(" ".join(c.lines) for c in en.cues[1:]) == fix


def test_keeps_the_line_when_the_models_disagree_and_adds_the_caption_reading():
    en = _en("Thanks.", "Vantoo is always pioneering.")
    caps = Subtitle("en", [Cue(2.0, 4.0, ["The view is always pioneering"])])
    judges = [_judge("The view is always pioneering."), _judge("x", changed=False)]
    en, res, still = ct.resolve(en, [_item(1, "00:02")], caps, None, judges)
    assert not res and en.cues[1].lines == ["Vantoo is always pioneering."]
    assert "YouTube captions" in still[0]["detail"]


def test_rejects_fillers_stutters_and_unheard_words():
    assert ct.acceptable("so I hold the key", "so I like hold the key", "so I like hold the key")
    assert ct.acceptable("take that burden", "take that that burden", "take that that burden")
    assert "not heard" in ct.acceptable("done into NWA", "done by the whole team", "done into NBA")
    assert ct.acceptable("It was a blood", "It was a blow", "it was a blow") == ""


def test_only_review_kinds_about_mishears_are_tried():
    en = _en("Thanks.", "Trust is the great barrier.")
    caps = Subtitle("en", [Cue(2.0, 4.0, ["trust is the great barrier"])])
    item = {"ts": "00:02", "kind": "en-meaning", "detail": 'What does "great barrier" mean?'}
    calls = []
    _, res, still = ct.resolve(en, [item], caps, None, [lambda p: calls.append(p) or "{}"] * 2)
    assert not calls and still == [item]


def test_a_correction_that_overflows_the_cue_splits_it_at_a_word_never_rewords():
    en = _en("So I think getting stuff done into NWA for customers is really what matters.")
    words = [Word("for", 1.2, 1.3)]
    ct._place(
        en,
        0,
        "So I think getting stuff done in an end-to-end way for customers is really what matters.",
        words,
    )
    assert len(en.cues) == 2 and en.cues[1].start == 1.2
    joined = " ".join(" ".join(c.lines) for c in en.cues)
    assert (
        joined
        == "So I think getting stuff done in an end-to-end way for customers is really what matters."
    )
    assert all(len(x) <= 42 for c in en.cues for x in c.lines)


def test_meaning_question_about_a_fixed_line_is_moot():
    held = [{"kind": "en-meaning", "detail": 'What does "getting stuff done into NWA" mean here?'}]
    res = [{"was": "So I think getting stuff done into NWA for customers."}]
    assert ct.drop_moot_meaning(held, res) == []


def test_a_line_all_three_recognizers_heard_as_written_is_confirmed_without_asking():
    en = _en("Thanks.", "It's about doing your own experience, testing it.")
    caps = Subtitle(
        "en",
        [
            Cue(
                2.0,
                4.0,
                ["It's it's about getting started. Doing your own experience, testing it."],
            )
        ],
    )
    second = Transcript(
        "en",
        [
            Word(w, 2.0 + 0.1 * i, 2.1 + 0.1 * i)
            for i, w in enumerate("about doing your own experience testing it".split())
        ],
    )
    calls = []
    en, res, still = ct.resolve(
        en, [_item(1, "00:02")], caps, second, [lambda p: calls.append(p) or "{}"] * 2
    )
    assert not calls and not still and res[0]["confirmed"] and res[0]["now"] == res[0]["was"]


def test_a_near_repeat_of_a_corrected_line_gets_that_correction_as_context():
    en = _en(
        "So I think getting stuff done into NWA for customers.",
        "Later on.",
        "I think getting stuff done into NBA for customers.",
    )
    caps = Subtitle(
        "en",
        [
            Cue(0.0, 2.0, ["So I think getting stuff done in end-to-end way for customers"]),
            Cue(4.0, 6.0, ["I think getting stuff done in in two end way for customers"]),
        ],
    )
    second = Transcript("en", [Word("an", 0.5, 0.6)])
    seen = []

    def judge(p):
        seen.append(p)
        fix = (
            "So I think getting stuff done in an end-to-end way for customers."
            if "OUR LINE: So" in p
            else "I think getting stuff done in an end-to-end way for customers."
        )
        return json.dumps({"line": fix, "changed": True})

    en, res, still = ct.resolve(
        en, [_item(0, "00:00"), _item(2, "00:04")], caps, second, [judge, judge]
    )
    assert len(res) == 2 and not still
    assert "near-identical line was corrected" in seen[-1]


def test_sound_alike_context_fix_when_every_recognizer_misheard():
    assert ct.sounds_like("I know Lukash was", "I know Łukasz was")
    assert ct.sounds_like("What's your logo?", "What's your flow?")
    assert not ct.sounds_like(
        "Vantoo is always", "The view is always"
    )  # a name lost to common words
    assert ct.acceptable("I know Lukash was", "I know Łukasz was", "I know Lukasz was") == ""


def test_terminology_is_not_confirmed_just_because_all_recognizers_wrote_it():
    en = _en("Thanks.", "any LM telling them the answer")
    caps = Subtitle("en", [Cue(2.0, 4.0, ["any LM telling them the answer"])])
    second = Transcript(
        "en",
        [
            Word(w, 2.0 + 0.1 * i, 2.1)
            for i, w in enumerate("any LM telling them the answer".split())
        ],
    )
    item = {"ts": "00:02", "kind": "en-terminology", "detail": '[cue 1] "LM" is likely "LLM"'}
    judge = _judge("any LLM telling them the answer")
    en, res, still = ct.resolve(en, [item], caps, second, [judge, judge], known="LLM")
    assert (
        res and not res[0].get("confirmed") and res[0]["now"] == "any LLM telling them the answer"
    )


def test_held_item_found_when_its_timestamp_drifted_from_the_cue():
    en = Subtitle(
        "en",
        [
            Cue(380.0, 383.0, ["Earlier line."]),
            Cue(383.06, 386.8, ["One thing that brought up as you're", "talking is"]),
        ],
    ).renumber()
    item = {
        "ts": "06:20",
        "kind": "en-grammar",
        "detail": "[cue 97] No subject: 'One thing that brought up' -> 'came up'",
    }
    assert ct._cue_of(en, item) == 1


def test_confirm_tolerates_the_2nd_asr_missing_the_cues_first_word():
    line = "Over here is sometimes a good way to get some new visions."
    caps = "Over here is is sometimes a good way to get some new visions. We build for"
    sec = "is sometimes a good way to get some new visions we build for everything"
    assert ct.heard_as_written(line, caps, sec)
    assert not ct.heard_as_written("Over here is sometimes a great way", caps, sec)


def test_numbers_match_by_digits_and_punctuation_findings_change_only_punctuation():
    from dev_titles import caption_tiebreak as ct

    caps = "The starter app was recently uplevelled to React Native 83"
    assert (
        ct.acceptable("upleveled to React Native AD3.", "upleveled to React Native 0.83.", caps)
        == ""
    )
    p = "en-punctuation"
    assert ct.acceptable("It has been so. I'm glad", "It has been. So I'm glad", "", p) == ""
    assert "only change punctuation" in ct.acceptable(
        "It has been so. I'm glad", "It has been. So I am glad", "", p
    )
    assert "only change punctuation" in ct.acceptable("It has been so.", "It has been so.", "", p)


def test_a_held_name_question_about_replaced_words_is_moot():
    from dev_titles import caption_tiebreak as ct

    held = [
        {
            "ts": "08:00",
            "kind": "en-name",
            "detail": "[cue 111] 'React Native AD3' -> 'React Native 0.83'? (desc) — not applied",
        },
        {
            "ts": "00:05",
            "kind": "en-name",
            "detail": "[cue 5] 'Bob' -> 'Rob'? (desc) — not applied",
        },
    ]
    resolved = [
        {
            "ts": "08:00",
            "was": "up-leveled to React Native AD3.",
            "now": "up-leveled to React Native 0.83.",
        }
    ]
    assert ct.drop_moot_meaning(held, resolved) == held[1:]


def test_punctuation_fix_cannot_change_a_number_and_moot_handles_apostrophes():
    from dev_titles import caption_tiebreak as ct

    assert "only change punctuation" in ct.acceptable(
        "React Native 0.83", "React Native 83", "", "en-punctuation"
    )
    held = [
        {
            "ts": "00:03",
            "kind": "en-name",
            "detail": '[cue 3] "O\'Brian" -> "O\'Brien"? (d) — not applied',
        }
    ]
    assert (
        ct.drop_moot_meaning(
            held, [{"ts": "00:03", "was": "thanks O'Brian", "now": "thanks O'Brien"}]
        )
        == []
    )


def test_a_moot_name_question_needs_the_same_timestamped_line():
    from dev_titles import caption_tiebreak as ct

    q = {"ts": "08:00", "kind": "en-name", "detail": "[cue 111] 'Bob' -> 'Rob'? (d) — not applied"}
    other = {"ts": "03:00", "was": "Bob said it", "now": "Bobby said it"}
    same = {"ts": "08:00", "was": "thanks Bob", "now": "thanks Rob"}
    assert ct.drop_moot_meaning([q], [other]) == [q]  # a different line keeps the question
    assert ct.drop_moot_meaning([q], [same]) == []
    untimed = {"kind": "en-name", "detail": "[cue 1] 'Bob' -> 'Rob'? (d)"}
    assert ct.drop_moot_meaning([untimed], [{"was": "Bob", "now": "Rob"}]) == [untimed]
    assert ct.acceptable("It has been so", "It has been — so", "", "en-punctuation") == ""


def test_a_punctuation_finding_is_resolved_and_a_word_change_under_it_is_not():
    en = _en("Totally.", "It has been so. I'm glad to see it.")
    caps = Subtitle("en", [Cue(2.0, 4.0, ["It has been. So, yeah, I'm glad to see it"])])
    fix = "It has been. So I'm glad to see it."
    en, res, still = ct.resolve(
        en, [_item(1, "00:02", "en-punctuation")], caps, None, [_judge(fix), _judge(fix)]
    )
    assert [r["now"] for r in res] == [fix] and not still
    reword = "It has been. So yeah, I'm glad to see it."  # adds a word: not a punctuation fix
    en2 = _en("Totally.", "It has been so. I'm glad to see it.")
    _, res, still = ct.resolve(
        en2, [_item(1, "00:02", "en-punctuation")], caps, None, [_judge(reword)] * 2
    )
    assert res == [] and len(still) == 1


def test_tiebreak_version_covers_punctuation_and_numbers():
    from dev_titles import cli

    assert cli.TIEBREAK_VERSION >= 7  # existing reviews are re-checked under v7's rules
    assert "en-punctuation" in ct.KINDS
