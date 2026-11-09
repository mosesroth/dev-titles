import json
from types import SimpleNamespace as S

from dev_titles.en_gloss import disputed, meaning_notes, set_operator_meaning


def _sents(*t):
    return [S(text=x, start=float(k)) for k, x in enumerate(t)]


def test_disputed_notes_are_dropped_and_operator_answers_win(tmp_path):
    sents = _sents(
        "The plugin installs.", "Take the jump from 0.72 to 0.83.", "It stays navigable."
    )
    notes = (
        '[{"id": 1, "phrase": "Take the jump", "meaning": "for example, not an instruction"},'
        ' {"id": 2, "phrase": "stays navigable", "meaning": "usable as you go"}]'
    )
    inv = lambda p: notes  # noqa: E731

    def verify(p):
        if p.startswith("Two readings"):  # the comparison call
            return '{"relation": "conflict"}' if "for example" in p else '{"relation": "same"}'
        return (
            '{"meaning": "make the upgrade"}'
            if "PHRASE: Take the jump" in p
            else '{"meaning": "usable as you go"}'
        )

    cp = tmp_path / "meaning_notes.json"
    mn = meaning_notes(sents, inv, cp, verify=verify)
    assert 1 not in mn and 2 in mn  # the disputed note never reaches the translators
    q = disputed(cp)
    assert (
        len(q) == 1
        and q[0]["sentence"] == "Take the jump from 0.72 to 0.83."
        and q[0]["b"] == "make the upgrade"
    )
    set_operator_meaning(cp, q[0]["sentence"], "make the upgrade from 0.72 to 0.83")
    assert disputed(cp) == []
    mn = meaning_notes(sents, inv, cp, verify=verify)  # cached: no model calls needed
    assert (
        mn[1][0]["meaning"] == "make the upgrade from 0.72 to 0.83" and mn[1][0]["by"] == "operator"
    )
    assert json.loads(cp.read_text(encoding="utf-8"))["operator"]


def test_a_meaning_question_about_a_sentence_no_longer_in_the_en_is_moot(tmp_path):
    import json

    from dev_titles.en_gloss import disputed

    p = tmp_path / "meaning_notes.json"
    p.write_text(
        json.dumps(
            {"disputed": [{"sentence": "getting stuff done into NWA."}, {"sentence": "Go to 13."}]}
        ),
        encoding="utf-8",
    )
    assert len(disputed(p)) == 2
    assert [
        x["sentence"]
        for x in disputed(p, ["getting stuff done in an end-to-end way.", "Go to 13."])
    ] == ["Go to 13."]


def test_readings_that_agree_but_one_adds_context_are_not_a_question(tmp_path):
    """Both meaning questions a run raised were one reading plus context ('the biggest obstacle'
    vs '...preventing customers from letting AI agents buy'). Keep the plainer one."""
    sents = _sents(
        "It requires trust.",
        "Trust is the great barrier in e-commerce.",
        "If you don't trust it...",
    )
    inv = (
        lambda p: '[{"id": 1, "phrase": "the great barrier", "meaning": "the biggest obstacle"}]'
    )  # noqa: E731

    def verify(p):
        if p.startswith("Two readings"):
            return '{"relation": "b_adds"}'
        return '{"meaning": "the biggest obstacle to letting AI agents make purchases"}'

    cp = tmp_path / "meaning_notes.json"
    mn = meaning_notes(sents, inv, cp, verify=verify)
    assert disputed(cp) == [] and mn[1][0]["meaning"] == "the biggest obstacle"


def test_recheck_disputes_settles_non_conflicts_and_keeps_real_ones(tmp_path):
    import json

    from dev_titles.en_gloss import disputed, recheck_disputes

    p = tmp_path / "meaning_notes.json"
    p.write_text(
        json.dumps(
            {
                "notes": {},
                "disputed": [
                    {
                        "id": 3,
                        "sentence": "That's my hot take.",
                        "phrase": "hot take",
                        "a": "a provocative opinion",
                        "b": "a deliberately controversial unpopular opinion",
                    },
                    {
                        "id": 5,
                        "sentence": "There is a slope.",
                        "phrase": "a slope",
                        "a": "AI slop: low-quality output",
                        "b": "a learning curve",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    verify = lambda prompt: (
        '{"relation": "conflict"}' if "slope" in prompt else '{"relation": "b_adds"}'
    )  # noqa: E731
    assert recheck_disputes(p, verify) == ["That's my hot take."]
    assert [x["sentence"] for x in disputed(p)] == ["There is a slope."]
    assert (
        json.loads(p.read_text(encoding="utf-8"))["notes"]["3"][0]["meaning"]
        == "a provocative opinion"
    )


def test_a_conflict_needs_both_families_and_a_silent_judge_does_not_clear_it():
    from dev_titles.en_gloss import _relation

    conflict = lambda p: '{"relation": "conflict"}'  # noqa: E731
    agree = lambda p: '{"relation": "b_adds"}'  # noqa: E731
    junk = lambda p: "sorry"  # noqa: E731
    args = ("universal shipping", "By shipping, I mean universal shipping.", "", "", "a", "b")
    assert _relation([conflict, agree], *args) == "b_adds"
    assert _relation([conflict, conflict], *args) == "conflict"
    assert _relation([conflict, junk], *args) == "conflict"


def test_an_en_edit_re_asks_only_the_changed_sentences(tmp_path):
    """Four name fixes used to re-ask every sentence, and a new dispute appeared on a line nobody
    had touched. Unchanged sentences keep their notes and their open disputes."""
    sents = _sents(
        "The SDK plugs in.",
        "Take the jump from 0.72 to 0.83.",
        "It stays navigable.",
        "Thanks, Lukash.",
    )
    notes = (
        '[{"id": 1, "phrase": "Take the jump", "meaning": "for example"},'
        ' {"id": 2, "phrase": "stays navigable", "meaning": "usable as you go"}]'
    )
    asked = []

    def inv(p):
        if p.startswith(
            "You prepare"
        ):  # the finder (the same model also takes part in the compare)
            asked.append(p)
            return notes
        return '{"relation": "conflict"}' if "for example" in p else '{"relation": "same"}'

    def verify(p):
        if p.startswith("Two readings"):
            return '{"relation": "conflict"}' if "for example" in p else '{"relation": "same"}'
        return (
            '{"meaning": "make the upgrade"}'
            if "PHRASE: Take the jump" in p
            else '{"meaning": "usable as you go"}'
        )

    cp = tmp_path / "meaning_notes.json"
    meaning_notes(sents, inv, cp, verify=verify)
    assert len(disputed(cp)) == 1 and len(asked) == 1
    edited = _sents(
        "The SDK plugs in.",
        "Take the jump from 0.72 to 0.83.",
        "It stays navigable.",
        "Thanks, Łukasz.",
    )
    asked.clear()
    mn = meaning_notes(edited, inv, cp, verify=verify, batch=2)
    # batches of 2: only the second batch (sentences 2-3) holds the change, so only it is re-asked
    assert len(asked) == 1 and "Thanks, Łukasz." in asked[0]
    assert mn[2][0]["meaning"] == "usable as you go"  # an unchanged sentence keeps its note
    assert [d["sentence"] for d in disputed(cp)] == [
        "Take the jump from 0.72 to 0.83."
    ]  # same dispute, not a new one


def test_duplicate_lines_keep_their_own_notes_and_all_their_disputes(tmp_path):
    import json as _j

    sents = _sents("Okay.", "Take the jump now.", "Okay.", "Bye.")
    cp = tmp_path / "meaning_notes.json"
    cp.write_text(
        _j.dumps(
            {
                "key": "old",
                "side": None,
                "texts": [s.text for s in sents],
                "notes": {"2": [{"phrase": "Okay", "meaning": "agreed"}]},
                "disputed": [
                    {
                        "id": 1,
                        "sentence": "Take the jump now.",
                        "phrase": "Take the jump",
                        "a": "x",
                        "b": "y",
                    },
                    {
                        "id": 1,
                        "sentence": "Take the jump now.",
                        "phrase": "now",
                        "a": "p",
                        "b": "q",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    import hashlib

    from dev_titles import en_gloss

    side = hashlib.sha1(("" + "\x02v").encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    d = _j.loads(cp.read_text())
    d["side"] = side
    cp.write_text(_j.dumps(d), encoding="utf-8")
    asked = []
    mn = en_gloss.meaning_notes(
        sents + _sents("x")[0:0], lambda p: asked.append(p) or "[]", cp, verify=lambda p: "{}"
    )
    assert (
        asked == [] and 0 not in mn and mn[2][0]["meaning"] == "agreed"
    )  # the 2nd 'Okay.' keeps its note
    assert len(disputed(cp)) == 2  # both disputes on sentence 1 survive
