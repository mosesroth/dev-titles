from pathlib import Path

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
    tr = lambda g, r, s=0: "NEW" if g is not None and r["i"] == 0 else f"old{r['i']}"  # noqa: E731
    # one evidence row: 3 samples winning 3-0 is p = 0.125 -> not enough on its own
    assert not validate(x, tr, [judge_new, judge_new], rows)["accepted"]
    x2 = {**x, "evidence": [rows[0], rows[1]]}
    tr2 = lambda g, r, s=0: (
        "NEW" if g is not None and r["i"] in (0, 1) else f"old{r['i']}"
    )  # noqa: E731
    assert validate(x2, tr2, [judge_new, judge_new], rows)["accepted"]  # 6-0: p = 0.016
    tr_harm = lambda g, r, s=0: (
        "NEW" if g is not None else "OLD"
    )  # noqa: E731  (changes controls too)
    judge_old = lambda p: '{"better": "A"}' if "A: OLD" in p else '{"better": "B"}'  # noqa: E731
    assert not validate(x2, tr_harm, [judge_old, judge_old], rows, seed=random.randint(0, 9))[
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

    # the reviewer corpus layout: <reviewed>/<batch>/<video>.<lang>_Reviewed.srt — found at any
    # depth, and the video the lesson came from is skipped (it is not counter-evidence)
    (tmp_path / "batch1").mkdir()
    (tmp_path / "batch1" / "Other.ja_Reviewed.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nサイドプロジェクトです\n", encoding="utf-8"
    )
    (tmp_path / "batch1" / "SkipMe.ja_Reviewed.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nサイドプロジェクト\n", encoding="utf-8"
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


def test_fixed_line_needs_an_established_corpus_rendering():
    from dev_titles.learn import fixed_line_established

    x = {
        "kind": "fixed_line",
        "lang": "ja",
        "source": "Thank you.",
        "target": "ありがとうございました",
    }
    varied = [
        ("Thank you.", "ありがとうございました", "a"),
        ("Thank you.", "ご視聴ありがとうございました", "b"),
    ]
    assert not fixed_line_established(x, varied)[0]
    steady = [("Thank you.", "ありがとうございました", v) for v in "aabb"]
    assert fixed_line_established(x, steady)[0]
    assert not fixed_line_established({**x, "source": "Sure."}, steady)[0]  # never in the corpus


def _srt(text):
    return f"1\n00:00:01,000 --> 00:00:02,000\n{text}\n"


def test_conventions_find_a_house_term_judges_cannot_know(tmp_path):
    """Web -> 网页 in 3 videos (20261104_CN): a Latin draft word replaced consistently."""
    from dev_titles.learn import convention_lessons, conventions

    rows = [
        {"video": v, "en": en, "ours": o, "human": h}
        for v, en, o, h in [
            ("A", "web developers", "Web开发者都能用上", "网页开发者都能用上"),
            ("A", "port it from the web", "把应用从Web移植过来", "把应用从网页移植过来"),
            ("B", "I did web development", "我做Web开发的头四年", "我做网页开发的头四年"),
            ("C", "catch up with web", "努力追上Web", "努力追上网页"),
            ("C", "a question?", "开发者为什么要关注它呢", "开发者为什么要关注它呢？"),
            ("A", "right?", "关键是赋能，对吧", "关键是赋能，对吧？"),
            ("B", "is it there?", "是在GitHub上吗", "是在GitHub上吗？"),
        ]
    ]
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "D.zh-Hans_Reviewed.srt").write_text(_srt("没有相关内容"), encoding="utf-8")
    words, punct = conventions(rows, "zh-Hans", tmp_path)
    assert [(w["from"], w["to"], len(w["videos"])) for w in words] == [("Web", "网页", 3)]
    assert any(p["to"] == "？" and len(p["videos"]) == 3 for p in punct)

    def invoke(prompt):
        return '[{"id": 0, "term": "web", "kind": "terminology", "why": "house term"}]'

    lessons, rest = convention_lessons(words, invoke, "zh-Hans")
    assert [(x["source"], x["target"], x["avoid"]) for x in lessons] == [("web", "网页", ["Web"])]
    assert not rest


def test_convention_rejected_when_reviewers_keep_the_draft_word(tmp_path):
    from dev_titles.learn import convention_lessons, conventions

    rows = [
        {"video": v, "en": "web app", "ours": "Web应用很好", "human": "网页应用很好"} for v in "ABC"
    ]
    (tmp_path / "b").mkdir()
    for v in ("D", "E"):  # other reviewers keep "Web" — a contextual choice, not a rule
        (tmp_path / "b" / f"{v}.zh-Hans_Reviewed.srt").write_text(_srt("Web版"), encoding="utf-8")
    words, _ = conventions(rows, "zh-Hans", tmp_path)
    lessons, rest = convention_lessons(
        words, lambda p: '[{"id": 0, "term": "web", "kind": "terminology"}]', "zh-Hans"
    )
    assert not lessons and "keep it elsewhere=2" in rest[0]["not_a_lesson"]


def test_learn_batch_ledger_skips_done_and_finds_missing(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "lrb", Path(__file__).resolve().parents[1] / "scripts" / "learn_review_batch.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    monkeypatch.setattr(m, "REVIEWED", tmp_path / "reviewed")
    monkeypatch.setattr(m, "LEARN", tmp_path / "learn")
    monkeypatch.setattr(m, "LEDGER", tmp_path / "learn" / "ledger.json")
    for b, v in (("batch1", "FirstVideo"), ("batch2", "SecondVideo")):
        (tmp_path / "reviewed" / b).mkdir(parents=True)
        (tmp_path / "reviewed" / b / f"{v}.zh-Hans_Reviewed.srt").write_text(_srt("x"), "utf-8")
    m._save_ledger({"batch1/FirstVideo/zh-Hans": {"status": "done"}})
    seen = []
    monkeypatch.setattr(
        m, "learn_video", lambda b, v, lang, p, apply: seen.append(f"{b}/{v}") or {"status": "done"}
    )
    monkeypatch.setattr(m, "corpus_pass", lambda lang, apply, invoke: [])
    assert m.main(["--missing", "--no-apply"]) == 0
    assert seen == ["batch2/SecondVideo"]
    assert m.ledger()["batch2/SecondVideo/zh-Hans"]["status"] == "done"
