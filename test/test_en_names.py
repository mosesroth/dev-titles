"""en_names: names and slips in the EN, checked against the video's own title/description/brief."""

import json

from dev_titles import degraded, en_names
from dev_titles.model import Cue, Subtitle

DESC = "🎙️ Ghizlan Bardiowil - SVP of Product & Content, Northwind\n🎙️ Chris Hartley - Host"
TITLE = "Smaller Models, Faster Shipping: The Local Model Cliff"


def _en(*lines):
    return Subtitle("en", [Cue(float(k), k + 1.0, [t]) for k, t in enumerate(lines)]).renumber()


def _finder(cands):
    return lambda prompt: json.dumps(cands) if "Lines (id | text)" in prompt else "[]"


YES = lambda p: '{"ok": true}'  # noqa: E731
NO = lambda p: '{"ok": false}'  # noqa: E731


def test_a_name_written_in_the_description_is_corrected_and_a_split_name_is_finished():
    en = _en(
        "and I'm joined today by Gazelle Bardio",
        "Wil, the SVP at Northwind.",
        "Thanks, Cutter, for having me.",
    )
    cands = [
        {
            "id": 0,
            "type": "name",
            "old": "Gazelle Bardio",
            "new": "Ghizlan Bardiowil",
            "evidence": "description",
        },
        {"id": 1, "type": "name", "old": "Wil", "new": "", "evidence": "rest of the name"},
        {
            "id": 2,
            "type": "name",
            "old": "Cutter",
            "new": "Chris",
            "evidence": "the host, per the brief",
        },
    ]
    brief = "The 'Cutter' in the thanks is a mis-transcription of the host Chris."
    en, applied, held = en_names.check(
        en, [_finder(cands), YES], title="t", description=DESC, brief=brief
    )
    assert [" ".join(c.lines) for c in en.cues] == [
        "and I'm joined today by Ghizlan Bardiowil",
        "the SVP at Northwind.",
        "Thanks, Chris, for having me.",
    ]
    assert len(applied) == 3 and held == []
    assert "Names and slips corrected (3" in en_names.section(applied)[0]


def test_a_name_not_in_the_evidence_or_unconfirmed_is_held_not_applied():
    en = _en("Thanks, Carson.", "And Otto, thank you.")
    cands = [
        {
            "id": 0,
            "type": "name",
            "old": "Carson",
            "new": "Karsten",
            "evidence": "?",
        },  # not written anywhere
        {"id": 1, "type": "name", "old": "Otto", "new": "Chris", "evidence": "description"},
    ]
    en, applied, held = en_names.check(en, [_finder(cands), NO], description=DESC)
    assert applied == [] and len(held) == 2
    assert "not written in the title/description/brief: karsten" in held[0]["detail"]
    assert (
        "doesn't sound like what was heard" in held[1]["detail"]
    )  # a different name, not a mishear
    nahim = [{"id": 0, "type": "name", "old": "Nahim", "new": "Naeem", "evidence": "d"}]
    _, applied, held = en_names.check(
        _en("Thanks, Nahim."), [_finder(nahim), NO], description="Idris Naeem"
    )
    assert applied == [] and "didn't confirm" in held[0]["detail"]
    assert [" ".join(c.lines) for c in en.cues] == ["Thanks, Carson.", "And Otto, thank you."]
    # with no second model nothing is auto-applied
    nahim = [{"id": 0, "type": "name", "old": "Nahim", "new": "Naeem", "evidence": "d"}]
    _, applied, held = en_names.check(
        _en("Thanks, Nahim."), [_finder(nahim)], description="Idris Naeem"
    )
    assert applied == [] and "no second model" in held[0]["detail"]


def test_a_slip_needs_the_meaning_notes_and_a_word_choice_is_held():
    en = _en(
        "So the local model cliff.",
        "And that feeds into that local model clip",
        "it's AI-generated and human-refined.",
        "AI-generated, but human-verified.",
    )
    finder = _finder(
        [
            {
                "id": 1,
                "type": "slip",
                "old": "local model clip",
                "new": "local model cliff",
                "evidence": "title",
            },
            {
                "id": 2,
                "type": "slip",
                "old": "human-refined",
                "new": "human-verified",
                "evidence": "line 3",
            },
        ]
    )
    notes = {
        "1": [
            {
                "phrase": "local model clip",
                "meaning": "mis-said 'local model cliff': the tipping point",
            }
        ]
    }
    en, applied, held = en_names.check(en, [finder, YES], title=TITLE, notes=notes)
    assert [a["now"] for a in applied] == ["And that feeds into that local model cliff"]
    assert len(held) == 1 and "meaning notes didn't flag" in held[0]["detail"]
    assert " ".join(en.cues[2].lines) == "it's AI-generated and human-refined."


def test_gate_rules():
    g = en_names.gate
    assert (
        g("slip", "MI5", "MI6", "the Q division of MI5", evidence="", others="") != ""
    )  # factual: not established
    assert (
        g(
            "slip",
            "local model clip",
            "lunch break",
            "that local model clip",
            evidence=TITLE,
            others="",
        )
        != ""
    )
    assert (
        g("name", "Nahim", "Naeem", "Idris Nahim, GM", evidence="Idris Naeem - GM", others="") == ""
    )
    assert "not in the line" in g("name", "Bob", "Rob", "Thanks, Alice.", evidence="Rob", others="")
    assert "too long" in g("name", "a b c d e", "x", "a b c d e", evidence="x", others="")
    assert g("name", "Chris", "Chris", "Chris", evidence="Chris", others="") == "no change"


def test_note_slips_parses_mis_said_notes_only():
    from types import SimpleNamespace as T

    sents = [T(text="a"), T(text="b")]
    notes = {
        "0": [
            {
                "phrase": "local model clip",
                "meaning": "mis-said “local model cliff”: the tipping point",
            }
        ],
        "1": [{"phrase": "scratch the surface", "meaning": "start looking into it"}],
        "9": [{"phrase": "x", "meaning": "misspoke 'y'"}],  # out of range
    }
    assert en_names.note_slips(notes, sents) == [
        {"sentence": "a", "old": "local model clip", "new": "local model cliff"}
    ]


def test_a_failing_finder_is_survivable():
    def boom(p):
        raise RuntimeError("down")

    en, applied, held = en_names.check(_en("Thanks, Cutter."), [boom, YES], description=DESC)
    assert applied == held == []


def test_degraded_steps_are_listed_once_and_reset():
    degraded.reset()
    assert degraded.banner() == ""
    assert degraded.note("review-and-fix", TypeError()) == "review-and-fix skipped (TypeError)"
    degraded.note("review-and-fix", TypeError())
    assert degraded.steps() == ["review-and-fix skipped (TypeError)"]
    assert degraded.banner().startswith("DEGRADED RUN — 1 step(s) skipped")
    degraded.reset()
    assert degraded.steps() == []


def _stub_review(monkeypatch, issues):
    from dev_titles import cli, en_gate, en_terms, transcribe_qe
    from dev_titles.translate import CliTranslator

    monkeypatch.setattr(CliTranslator, "_invoke", lambda self, p: "[]")
    monkeypatch.setattr(cli, "_clean_read_stage", lambda en, out, inv, t, s: (en, []))
    monkeypatch.setattr(en_terms, "apply", lambda en: (en, []))
    monkeypatch.setattr(transcribe_qe, "en_review", lambda en, inv: issues)
    monkeypatch.setattr(transcribe_qe, "en_autofix", lambda en, iss, inv: (en, [], list(iss)))
    monkeypatch.setattr(cli, "_caption_tiebreak", lambda en, held, out, inv: (en, [], held))
    monkeypatch.setattr(cli, "_meaning_holds", lambda en, out, inv: [])
    states = []
    monkeypatch.setattr(
        en_gate, "save_review_state", lambda out, st, held, n=0: states.append(held)
    )

    def fake(en, out, inv, title=""):
        en.cues[0].lines = ["Thanks, Chris."]
        return (
            en,
            ["## Names and slips corrected (1)", ""],
            [{"ts": "00:01", "kind": "en-slip", "detail": "held one"}],
        )

    monkeypatch.setattr(cli, "_names_and_slips", fake)
    return states


def test_en_review_applies_names_and_slips_in_both_branches(monkeypatch, tmp_path):
    from dev_titles import cli

    for issues, sub in (
        ([], "clean"),
        ([{"ts": "00:00", "kind": "en-grammar", "detail": "x"}], "issues"),
    ):
        out = tmp_path / sub
        out.mkdir()
        states = _stub_review(monkeypatch, issues)
        en = _en("Thanks, Cutter.")
        new = cli._en_review(en, out, "T")
        assert new.cues[0].lines == ["Thanks, Chris."]
        assert en.cues[0].lines == ["Thanks, Cutter."]  # the caller's EN is not edited in place
        assert (
            "Thanks, Chris." in (out / "en.vtt").read_text()
            and "Cutter" in (out / "en.orig.vtt").read_text()
        )
        md = (out / "en_review.md").read_text()
        assert "Names and slips corrected" in md and "held one" in md
        assert md.count("# EN editorial review") == 1
        assert any(h.get("detail") == "held one" for h in states[-1])


def test_names_and_slips_feeds_the_videos_evidence_and_survives_errors(monkeypatch, tmp_path):
    from dev_titles import cli, en_names

    (tmp_path / "description.txt").write_text(DESC, encoding="utf-8")
    (tmp_path / "video_brief.txt").write_text("brief text", encoding="utf-8")
    notes = {"1": [{"phrase": "p", "meaning": "m"}]}
    (tmp_path / "meaning_notes.json").write_text(json.dumps({"notes": notes}), encoding="utf-8")
    monkeypatch.setattr(cli, "_qe_second_judge", lambda a: YES)
    seen = {}

    def check(en, judges, **kw):
        seen.update(kw, n=len(judges))
        return en, [{"ts": "00:00", "kind": "en-name", "was": "a", "now": "b", "evidence": "e"}], []

    monkeypatch.setattr(en_names, "check", check)
    en, md, held = cli._names_and_slips(_en("Thanks, Cutter."), tmp_path, lambda p: "[]", "Title")
    assert seen["description"] == DESC and seen["brief"] == "brief text" and seen["notes"] == notes
    assert (
        seen["title"] == "Title"
        and seen["n"] == 2
        and md[0].startswith("## Names and slips corrected (1")
    )

    def boom(*a, **k):
        raise RuntimeError

    monkeypatch.setattr(en_names, "check", boom)
    degraded.reset()
    assert cli._names_and_slips(_en("x"), tmp_path, lambda p: "[]")[1:] == ([], [])
    assert degraded.steps() == ["EN names/slips check skipped (RuntimeError)"]
    degraded.reset()


def test_a_degraded_run_says_so_in_review_status_and_the_closing_line(tmp_path):
    from dev_titles import cli

    (tmp_path / "REVIEW.md").write_text("# Subtitle review bundle\n", encoding="utf-8")
    degraded.reset()
    assert cli._report_degraded(tmp_path, tmp_path / "cache.jsonl") == ""
    degraded.note("ja: review-and-fix", TypeError())
    flag = cli._report_degraded(tmp_path, tmp_path / "cache.jsonl")
    assert flag.startswith(" — DEGRADED RUN")
    assert (tmp_path / "REVIEW.md").read_text().startswith("> **DEGRADED RUN")
    assert json.loads((tmp_path / "status.json").read_text())["degraded"] == [
        "ja: review-and-fix skipped (TypeError)"
    ]
    degraded.reset()


def test_a_rerun_stops_when_the_delivered_en_has_a_fix_the_working_copy_lacks(tmp_path):
    import os
    import time

    from dev_titles import cli
    from dev_titles.formats import write_srt, write_vtt

    out, lib = tmp_path / "ticket" / "MyVideo", tmp_path / "FINAL_SUBTITLES"
    out.mkdir(parents=True)
    lib.mkdir()
    work = _en("joined by Gazelle")
    (out / "en.vtt").write_text(write_vtt(work), encoding="utf-8")
    assert cli._delivered_en_drift(out, work) == ""  # nothing delivered yet
    (lib / "MyVideo.en.srt").write_text(write_srt(_en("joined by Ghizlan")), encoding="utf-8")
    t = time.time()
    os.utime(out / "en.vtt", (t - 60, t - 60))
    msg = cli._delivered_en_drift(out, work)
    assert msg.startswith("STOP") and "+joined by Ghizlan" in msg and "-joined by Gazelle" in msg
    assert f"dev_titles sync-delivered {out}" in msg  # the stop names the way past it
    os.utime(out / "en.vtt", (t + 60, t + 60))  # the working copy is the newer one: it wins
    assert cli._delivered_en_drift(out, work) == ""
    os.utime(out / "en.vtt", (t - 60, t - 60))
    assert cli._delivered_en_drift(out, _en("joined by Ghizlan")) == ""  # same text


def test_two_fixes_in_one_cue_both_land_and_a_backslash_is_literal():
    en = _en("Thanks Carson and Nahim.")
    cands = [
        {"id": 0, "type": "name", "old": "Carson", "new": "Carsten", "evidence": "d"},
        {"id": 0, "type": "name", "old": "Nahim", "new": "Naeem", "evidence": "d"},
    ]
    en, applied, _ = en_names.check(
        en, [_finder(cands), YES], description=DESC + " Idris Naeem, Carsten"
    )
    assert " ".join(en.cues[0].lines) == "Thanks Carsten and Naeem." and len(applied) == 2
    assert en_names._replace("a Cutter b", "Cutter", r"C\1") == r"a C\1 b"


def test_a_clean_rerun_clears_a_stale_degraded_list(tmp_path):
    from dev_titles import cli

    (tmp_path / "status.json").write_text(
        json.dumps({"degraded": ["old skipped (X)"]}), encoding="utf-8"
    )
    degraded.reset()
    assert cli._report_degraded(tmp_path, tmp_path / "cache.jsonl") == ""
    assert json.loads((tmp_path / "status.json").read_text())["degraded"] == []


def test_a_tail_deletion_never_leaves_an_empty_cue():
    en = _en("joined by Gazelle Bardio", "Wil")
    cands = [
        {
            "id": 0,
            "type": "name",
            "old": "Gazelle Bardio",
            "new": "Ghizlan Bardiowil",
            "evidence": "d",
        },
        {"id": 1, "type": "name", "old": "Wil", "new": "", "evidence": "rest of the name"},
    ]
    en, applied, held = en_names.check(en, [_finder(cands), YES], description=DESC)
    assert len(applied) == 1 and "leave the cue empty" in held[0]["detail"]
    assert all(" ".join(c.lines).strip() for c in en.cues)


def test_every_swallowed_error_step_is_recorded_as_degraded():
    """A step that catches its own error and carries on must call degraded.note — the bare
    'X skipped (TypeError)' print is how review-and-fix vanished from two PASS runs."""
    import importlib
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "dev_titles"
    bare = re.compile(r"skipped \(\{type\(e\)\.__name__\}\)")
    users = 0
    for p in sorted(src.rglob("*.py")):
        text = p.read_text(encoding="utf-8")
        assert not bare.search(text), f"{p.name}: a swallowed error not recorded via _degraded.note"
        if "_degraded.note(" in text:
            users += 1
            mod = importlib.import_module(
                "dev_titles." + ".".join(p.relative_to(src).with_suffix("").parts)
            )
            assert (
                getattr(mod, "_degraded", None) is degraded
            ), f"{p.name} uses _degraded without importing it"
    assert users >= 4
    counts = sum(p.read_text(encoding="utf-8").count("_degraded.note(") for p in src.rglob("*.py"))
    assert counts >= 19


def test_each_command_starts_with_an_empty_degraded_list(monkeypatch):
    from types import SimpleNamespace as NS

    from dev_titles import cli

    seen = []

    class P:
        def parse_args(self, raw):
            return NS(func=lambda a: seen.append(degraded.steps()) or 0)

    monkeypatch.setattr(cli, "build_parser", lambda: P())
    degraded.note("a previous command's step", RuntimeError())
    assert cli.main(["anything"]) == 0 and seen == [[]]


def test_a_name_that_runs_into_the_next_cue_drops_its_tail_there():
    en = _en("And I want to talk about the Acme App", "Store, why now?")
    cands = [
        {"id": 0, "type": "name", "old": "Acme App", "new": "Acme Appstore", "evidence": "title"}
    ]
    en, applied, held = en_names.check(en, [_finder(cands), YES], title="Acme Appstore Launch")
    assert [" ".join(c.lines) for c in en.cues] == [
        "And I want to talk about the Acme Appstore",
        "why now?",
    ]


def test_gate_capitalization_only_and_changed_words_only():
    g = en_names.gate
    assert g("name", "Jetflix", "JetFlix", "the Jetflix app", evidence="JetFlix", others="") == ""
    assert "isn't spelled" in g(
        "name", "Jetflix", "JETFLIX", "the Jetflix app", evidence="JetFlix", others=""
    )
    # only the changed word needs evidence ('idea' isn't a name)
    assert (
        g("name", "John's idea", "Jon's idea", "it was John's idea", evidence="Jon Samp", others="")
        == ""
    )
    assert (
        g(
            "name",
            "Otto",
            "Idris",
            "Otto, thanks",
            evidence="Idris",
            others="",
            brief="The 'Otto' is a transcription error for 'Idris'.",
        )
        == ""
    )


def test_a_hedged_brief_is_not_proof_of_a_mistranscription():
    brief = 'The "Bill" heard mid-interview is likely a mistranscription of James.'
    assert "operator decides" in en_names.gate(
        "name", "Bill", "James", "for sure, Bill, I think", evidence="James", others="", brief=brief
    )


def test_a_name_tail_in_a_cue_whose_first_line_is_empty_is_handled():
    en = Subtitle(
        "en", [Cue(0.0, 1.0, ["about the Acme App"]), Cue(1.0, 2.0, ["", "Store, why now?"])]
    ).renumber()
    cands = [
        {"id": 0, "type": "name", "old": "Acme App", "new": "Acme Appstore", "evidence": "title"}
    ]
    en, applied, held = en_names.check(en, [_finder(cands), YES], title="Acme Appstore Launch")
    assert [" ".join(x for x in c.lines if x) for c in en.cues] == [
        "about the Acme Appstore",
        "why now?",
    ]
