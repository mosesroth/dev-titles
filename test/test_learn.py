"""learn.py: triage, in-memory lesson, validation verdicts, comment-preserving glossary writes."""

import random

from dev_titles.glossary import Glossary, Term
from dev_titles.learn import apply_lessons, propose, triage, validate, with_lesson

G = Glossary(
    version="t",
    languages=["ja"],
    terms=[Term(source="hook", translations={"ja": "フック"})],
    style={"ja": {"notes": "BASE RULE.", "fixed_lines": {"Sure.": "はい"}}},
)


def test_triage_drops_known_and_flags_conflicts():
    xs = [
        {
            "kind": "glossary_term",
            "lang": "ja",
            "source": "hook",
            "target": "フック",
            "evidence": [],
        },
        {"kind": "glossary_term", "lang": "ja", "source": "Hook", "target": "Hook", "evidence": []},
        {"kind": "fixed_line", "lang": "ja", "source": "sure.", "target": "はい", "evidence": []},
        {"kind": "style_note", "lang": "ja", "text": "Use 携わる for 'work on'.", "evidence": []},
    ]
    keep, person = triage(xs, G)
    assert [x["kind"] for x in keep] == ["style_note"]
    assert len(person) == 1 and "conflict" in person[0]


def test_with_lesson_is_in_memory_only():
    g = with_lesson(G, {"kind": "style_note", "lang": "ja", "text": "NEW RULE."})
    assert "NEW RULE." in g.style["ja"]["notes"] and "NEW RULE." not in G.style["ja"]["notes"]
    g2 = with_lesson(
        G, {"kind": "glossary_term", "lang": "ja", "source": "OSS", "target": "オープンソース"}
    )
    assert len(g2.terms) == 2 and len(G.terms) == 1


def test_propose_merges_and_filters():
    rows = [
        {"i": 0, "en": "Sure.", "ours": "もちろんです", "human": "はい"},
        {"i": 1, "en": "Sure!", "ours": "もちろん", "human": "はい"},
    ]
    inv = (
        lambda p: '[{"id": 0, "kind": "fixed_line", "source": "Sure.", "target": "はい"}, {"id": 1, "kind": "none"}]'
    )  # noqa: E731
    out = propose(rows, inv, "ja", "")
    assert len(out) == 1 and out[0]["kind"] == "fixed_line" and out[0]["evidence"][0]["i"] == 0


def test_validate_accepts_only_when_evidence_improves_and_controls_hold():
    rows = [{"i": k, "en": f"line {k}", "ours": f"o{k}", "human": f"h{k}"} for k in range(6)]
    x = {
        "kind": "style_note",
        "lang": "ja",
        "text": "r",
        "evidence": [rows[0]],
        "_glossary": object(),
    }
    judge_new = lambda p: '{"better": "A"}' if "A: NEW" in p else '{"better": "B"}'  # noqa: E731
    tr = lambda g, r: "NEW" if g is not None and r["i"] == 0 else f"old{r['i']}"  # noqa: E731
    assert validate(x, tr, [judge_new, judge_new], rows)["accepted"]
    tr_harm = lambda g, r: "NEW" if g is not None else "OLD"  # noqa: E731  (changes controls too)
    judge_old = lambda p: '{"better": "A"}' if "A: OLD" in p else '{"better": "B"}'  # noqa: E731
    assert not validate(x, tr_harm, [judge_old, judge_old], rows, seed=random.randint(0, 9))[
        "accepted"
    ]


def test_apply_lessons_keeps_comments(tmp_path):
    y = tmp_path / "glossary.yaml"
    y.write_text(
        'version: t\nlanguages: [ja]\nterms:\n  # keep this comment\n  - source: "hook"\n    ja: "フック"\n'
        'style:\n  ja:\n    fixed_lines:\n      "Sure.": "はい"\n    notes: >-\n      BASE RULE.\n',
        encoding="utf-8",
    )
    acc = [
        {
            "kind": "glossary_term",
            "lang": "ja",
            "source": "OSS",
            "target": "オープンソース",
            "avoid": ["OSS"],
            "evidence": [],
        },
        {"kind": "style_note", "lang": "ja", "text": "Use 携わる.", "evidence": []},
        {"kind": "fixed_line", "lang": "ja", "source": "Yeah.", "target": "はい", "evidence": []},
        {
            "kind": "editor_example",
            "lang": "ja",
            "label": "x",
            "evidence": [{"en": "e", "ours": "o", "human": "h"}],
        },
    ]
    done = apply_lessons(acc, y, tmp_path, "T")
    s = y.read_text(encoding="utf-8")
    assert (
        "# keep this comment" in s
        and '"OSS"' in s
        and "LEARNED (T): Use 携わる." in s
        and '"Yeah."' in s
    )
    from dev_titles.glossary import load

    g = load(str(y))
    assert (
        any(t.source == "OSS" for t in g.terms) and g.style["ja"]["fixed_lines"]["Yeah."] == "はい"
    )
    assert len(done) == 4 and (tmp_path / "editor_examples.ja.jsonl").exists()


def test_triage_adds_a_missing_language_rendering():
    g = Glossary(
        version="t",
        languages=["ja"],
        terms=[Term(source="on-device", translations={"zh-Hans": "设备端"})],
    )
    keep, person = triage(
        [
            {
                "kind": "glossary_term",
                "lang": "ja",
                "source": "on-device",
                "target": "オンデバイス",
                "evidence": [],
            }
        ],
        g,
    )
    assert not person and keep[0]["adds_rendering_to"] == "on-device"


def test_reviewer_counterevidence(tmp_path):
    from dev_titles.learn import reviewer_counterevidence

    (tmp_path / "Other.ja.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nサイドプロジェクトです\n", encoding="utf-8"
    )
    x = {
        "kind": "glossary_term",
        "lang": "ja",
        "source": "side project",
        "target": "個人プロジェクト",
        "avoid": ["サイドプロジェクト"],
    }
    assert reviewer_counterevidence(x, tmp_path, skip="SkipMe") == [
        "Other: 'サイドプロジェクト' x1"
    ]
