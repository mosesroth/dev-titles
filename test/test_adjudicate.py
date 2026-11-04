import json
from types import SimpleNamespace as NS

from dev_titles import adjudicate
from dev_titles.localize import SentenceRecord, _restore_locked


def _r(i, src, txt, locked=False):
    return SentenceRecord("zh-Hans", i, src, txt, "", [], locked=locked)


def test_locked_lines_survive_any_step():
    L = NS(records=[_r(0, "a", "fixed by hand", locked=True), _r(1, "b", "x")])
    locked = {0: "fixed by hand"}
    L.records[0].txt = "editor rewrote it"
    L.records[1].txt = "y"
    _restore_locked(L, locked)
    assert L.records[0].txt == "fixed by hand" and L.records[1].txt == "y"


def test_review_and_fix_applies_only_judge_preferred_fixes(monkeypatch):
    recs = [
        _r(0, "It stays navigable.", "始终可以用遥控器操作"),
        _r(1, "Take the jump.", "以升级为例"),
        _r(2, "Locked one.", "人工修改", locked=True),
    ]
    sus = [
        {"kind": "mqm-minor[claude]", "i": 0, "detail": "adds 用遥控器"},
        {"kind": "mqm-major[codex]", "i": 1, "detail": "misread as example"},
        {"kind": "mqm-minor[claude]", "i": 2, "detail": "anything"},
    ]
    monkeypatch.setattr(adjudicate, "coverage_missing", lambda *a: "")

    def inv(p):
        if "ISSUES" in p:
            return "始终可以操作" if "遥控器" in p else "NO_CHANGE"
        return "[]"

    judge = lambda p: (
        '{"better": "A"}' if "A: 始终可以操作" in p else '{"better": "B"}'
    )  # noqa: E731
    log = adjudicate.review_and_fix(
        "zh-Hans", "Chinese", recs, sus, inv, [judge], logger=lambda m: None
    )
    assert recs[0].txt == "始终可以操作"  # the added detail removed
    assert recs[1].txt == "以升级为例"  # the finding was wrong: NO_CHANGE
    assert recs[2].txt == "人工修改"  # locked: never a candidate
    assert {e["i"]: e["status"] for e in log} == {0: "fixed", 1: "not real"}


def test_fix_line_edits_and_locks(tmp_path):
    from dev_titles import cli

    rows = [
        {
            "lang": "ja",
            "i": 0,
            "src": "First up, the React compiler.",
            "txt": "まずはReact compilerです",
            "residual_miss": "",
            "issues": [],
        }
    ]
    (tmp_path / "cache.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    rc = cli.main(
        [
            "fix-line",
            str(tmp_path),
            "--lang",
            "ja",
            "--find",
            "React compiler",
            "--replace",
            "React Compiler",
        ]
    )
    assert rc == 0
    r = json.loads((tmp_path / "cache.jsonl").read_text(encoding="utf-8"))
    assert r["txt"] == "まずはReact Compilerです" and r["locked"] is True
    assert (tmp_path / "locked_edits.jsonl").exists()


def test_enforce_fixes_casing_of_a_hard_latin_term():
    from dev_titles.glossary import Glossary, Term

    g = Glossary(
        version="t",
        languages=["ja"],
        terms=[
            Term(
                source="React Compiler",
                translations={"ja": "React Compiler"},
                enforce="hard",
                aliases=["React compiler"],
            )
        ],
    )
    txt, _ = g.enforce("まずはReact compilerです", "First up, the React compiler.", "ja")
    assert txt == "まずはReact Compilerです"


def test_a_split_is_resolved_by_a_second_round(monkeypatch):
    recs = [_r(0, "It stays navigable.", "始终可以用遥控器操作")]
    monkeypatch.setattr(adjudicate, "coverage_missing", lambda *a: "")
    inv = lambda p: "始终可以操作" if "ISSUES" in p else "[]"  # noqa: E731
    calls = {"n": 0}

    def j1(p):  # round 1: prefers the old line; round 2: the fix
        calls["n"] += 1
        new_is_a = "A: 始终可以操作" in p
        want_new = calls["n"] > 1
        return '{"better": "A"}' if new_is_a == want_new else '{"better": "B"}'

    j2 = lambda p: '{"better": "A"}' if "A: 始终可以操作" in p else '{"better": "B"}'  # noqa: E731
    sus = [{"kind": "mqm-minor[claude]", "i": 0, "detail": "adds 用遥控器"}]
    log = adjudicate.review_and_fix(
        "zh-Hans", "Chinese", recs, sus, inv, [j1, j2], logger=lambda m: None
    )
    assert (
        log[0]["status"] == "fixed" and recs[0].txt == "始终可以操作"
    )  # votes y,x then x,x -> 3 of 4


def test_term_consistency_only_unifies_toward_the_majority():
    recs = [_r(0, "tiles", "图块"), _r(1, "tiles", "图块"), _r(2, "tiles", "磁贴")]
    to_minority = '[{"term": "tile", "use": "磁贴", "ids": [0, 1]}]'
    assert adjudicate.candidates("zh-Hans", "Chinese", recs, [], invoke=lambda p: to_minority) == {}
    to_majority = '[{"term": "tile", "use": "图块", "ids": [2]}]'
    assert 2 in adjudicate.candidates("zh-Hans", "Chinese", recs, [], invoke=lambda p: to_majority)
