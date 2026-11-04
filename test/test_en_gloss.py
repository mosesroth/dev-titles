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
            return '{"same": false}' if "for example" in p else '{"same": true}'
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
