"""Split from test_core: eval tests (13)."""

from helpers import _mqm_by_target_marker

from dev_titles.eval import (
    GembaMqmScorer,
    QeResult,
    below_threshold_segments,
    parse_gemba_mqm,
    rank_engines,
)


def test_gemba_mqm_parse_penalty():
    assert parse_gemba_mqm("no-error")[0] == 0.0
    penalty, counts = parse_gemba_mqm("Major: wrong term\nMinor: awkward phrasing")
    assert counts == {"critical": 0, "major": 1, "minor": 1}
    assert penalty == 6.0  # 5 (major) + 1 (minor)


def test_gemba_scorer_with_fake_judge():
    # Fake judge: flawless for the first, one major error for the second.
    replies = iter(["no-error", "Major: mistranslation of 'submit'"])
    scorer = GembaMqmScorer(invoke=lambda prompt: next(replies))
    res = scorer.score(["a", "b"], ["x", "y"], "en", "ja")
    assert res.segment_scores == [0.0, -5.0]  # higher = better; second is worse
    assert res.system_score == -2.5
    assert res.details[1]["major"] == 1


def test_rank_engines_and_threshold():
    results = {
        "claude": QeResult([0.9, 0.8], 0.85, "cometkiwi"),
        "deepl": QeResult([0.7, 0.6], 0.65, "cometkiwi"),
    }
    ranked = rank_engines(results)
    assert ranked[0][0] == "claude" and ranked[-1][0] == "deepl"
    weak = below_threshold_segments(results["deepl"], threshold=0.65)
    assert weak == [1]  # only the 0.6 segment is below 0.65


def test_translation_qe_latin_typo():
    from types import SimpleNamespace as T

    from dev_titles.translation_qe import _latin_typo_suspects

    glossary = T(terms=[], identifiers=["IAP"])
    recs = [
        T(i=0, src="Make sure USB debugging is on.", txt="确保 USP 调试已开启"),  # USB->USP typo
        T(i=1, src="This enables in-app purchase.", txt="这会启用 IAP"),  # IAP known -> ok
        T(i=2, src="Call the API from here.", txt="从这里调用 API"),
    ]  # API in source -> ok
    sus = _latin_typo_suspects(recs, glossary)
    flagged = {s["i"] for s in sus}
    assert flagged == {0}  # only the USP typo
    assert "USP" in sus[0]["detail"]


def test_translation_qe_deterministic_and_judge():
    from types import SimpleNamespace as T

    from dev_titles.translation_qe import translation_suspects

    # records: one complete, one that dropped content + a number
    recs = [
        T(
            i=0,
            src="Welcome to the show, everyone, thanks for coming today.",
            txt="皆さん番組へようこそ 本日はお越しいただきありがとうございます",
        ),
        T(
            i=1,
            src="The latency dropped from 58 milliseconds to well under 5 for every frame.",
            txt="遅延が下がりました",
        ),
    ]  # short + drops 58 and 5
    glossary = T(terms=[])  # no term-consistency in this test
    suspects, dqf = translation_suspects(recs, glossary, "ja")  # deterministic only
    kinds = {s["kind"] for s in suspects}
    assert "omission" in kinds and "dropped-token" in kinds
    assert dqf is None  # judge off

    # with the LLM judge (fake): seg 1 has a major error -> DQF populated + weak
    def fake(prompt):
        return "[0] no-error\n[1] major: dropped the latency numbers and half the sentence"

    suspects2, dqf2 = translation_suspects(recs, glossary, "ja", invoke=fake, judge=True)
    assert dqf2 is not None and dqf2["counts"]["major"] == 1 and dqf2["weak"] == [1]
    assert any(s["kind"] == "mqm-major" for s in suspects2)
    assert 0.0 <= dqf2["score"] <= 100.0


def test_translation_qe_two_judge_corroboration():
    """Panel aggregation: major/critical = UNION (max across judges); minor = CORROBORATION
    (min — a lone-judge minor is dropped as subjective/noise, a minor both judges see counts)."""
    from types import SimpleNamespace as T

    from dev_titles.translation_qe import translation_suspects

    recs = [
        T(i=0, src="one two three four five", txt="一二"),  # both: major
        T(i=1, src="alpha beta gamma", txt="A B G"),  # lone minor (codex only)
        T(i=2, src="ay bee see dee", txt="X Y"),  # both: minor -> corroborated
        T(i=3, src="red green blue", txt="R G B"),
    ]  # lone major (claude only)
    glossary = T(terms=[], identifiers=[])
    claude = (
        lambda p: "[0] major: dropped\n[1] no-error\n[2] minor: awkward\n[3] major: mistranslation"
    )
    codex = lambda p: "[0] major: dropped\n[1] minor: unnatural\n[2] minor: unnatural\n[3] no-error"
    _, dqf = translation_suspects(
        recs, glossary, "ja", invoke=claude, judge=True, second_judge=codex
    )
    assert dqf["judges"] == ["claude", "codex"]
    assert dqf["counts"]["major"] == 2  # UNION: seg0 (both) + seg3 (lone claude)
    assert dqf["counts"]["minor"] == 1  # CORROBORATION: seg1 lone -> 0, seg2 both -> 1
    assert dqf["weak"] == [0, 3]  # any major/critical


def test_qe_repair_accepts_when_strictly_better():
    from types import SimpleNamespace as T

    from dev_titles.translation_qe import build_judges, qe_repair

    recs = [T(i=0, src="It's a fork of VS Code.", txt="BAD rendering")]
    judges = build_judges(_mqm_by_target_marker)  # single-judge panel is fine
    per_seg = {0: {"critical": 0, "major": 1, "minor": 0}}
    fixed = qe_repair(
        recs,
        "ja",
        judges=judges,
        lang_name="Japanese",
        retranslate=lambda rec, crit: "GOOD rendering",
        per_seg=per_seg,
        judge_suspects=[{"i": 0, "kind": "mqm-major", "detail": "loses fork sense"}],
    )
    assert fixed == [0]
    assert recs[0].txt == "GOOD rendering"  # accepted: old major -> new clean


def test_qe_repair_include_minor_targets_corroborated():
    from types import SimpleNamespace as T

    from dev_titles.translation_qe import build_judges, qe_repair

    per_seg = {
        0: {
            "claude": {"critical": 0, "major": 0, "minor": 1},
            "codex": {"critical": 0, "major": 0, "minor": 1},
        }
    }  # corroborated minor
    recs = [T(i=0, src="today", txt="BAD")]
    # without include_minor a lone corroborated-minor segment is NOT targeted
    assert (
        qe_repair(
            recs,
            "ja",
            judges=build_judges(_mqm_by_target_marker),
            lang_name="Japanese",
            retranslate=lambda r, c: "GOOD",
            per_seg=per_seg,
            judge_suspects=[],
            include_minor=False,
        )
        == []
    )
    assert recs[0].txt == "BAD"
    # with include_minor it IS targeted and the improving fix lands
    assert qe_repair(
        recs,
        "ja",
        judges=build_judges(_mqm_by_target_marker),
        lang_name="Japanese",
        retranslate=lambda r, c: "GOOD",
        per_seg=per_seg,
        judge_suspects=[],
        include_minor=True,
    ) == [0]
    assert recs[0].txt == "GOOD"


def test_qe_repair_extra_critiques_targets_term_inconsistency():
    """term-inconsistency cues are routed into repair with a canonical-rendering critique
    (closes the gap where inconsistencies were detected but fixed by nothing)."""
    from types import SimpleNamespace as T

    from dev_titles.translation_qe import build_judges, qe_repair

    recs = [T(i=4, src="the steering documents", txt="BAD 指导文件")]
    seen = {}

    def retrans(rec, crit):
        seen["crit"] = crit
        return "GOOD 引导文档"

    fixed = qe_repair(
        recs,
        "zh-Hans",
        judges=build_judges(_mqm_by_target_marker),
        lang_name="Simplified Chinese",
        retranslate=retrans,
        per_seg={},
        judge_suspects=[],
        extra_critiques={4: "render 'steering documents' as '引导文档' (consistency)"},
    )
    assert fixed == [4] and recs[0].txt == "GOOD 引导文档"
    assert "引导文档" in seen["crit"]  # the canonical critique reached the retranslate


def test_qe_repair_rejects_when_no_gain():
    from types import SimpleNamespace as T

    from dev_titles.translation_qe import build_judges, qe_repair

    recs = [T(i=0, src="It's a fork of VS Code.", txt="BAD rendering")]
    judges = build_judges(_mqm_by_target_marker)
    per_seg = {0: {"critical": 0, "major": 1, "minor": 0}}
    # the "repair" is still bad (judge still flags major) -> penalty doesn't drop -> revert
    fixed = qe_repair(
        recs,
        "ja",
        judges=judges,
        lang_name="Japanese",
        retranslate=lambda rec, crit: "STILL BAD",
        per_seg=per_seg,
        judge_suspects=[{"i": 0, "kind": "mqm-major", "detail": "x"}],
    )
    assert fixed == []
    assert recs[0].txt == "BAD rendering"  # original preserved


def test_qe_repair_extra_flags_deterministic():
    """A segment with NO judge major but flagged deterministically (dropped-token/typo)
    is still offered for repair via extra_flags."""
    from types import SimpleNamespace as T

    from dev_titles.translation_qe import build_judges, qe_repair

    recs = [T(i=3, src="Call the API.", txt="BAD garble")]
    fixed = qe_repair(
        recs,
        "ja",
        judges=build_judges(_mqm_by_target_marker),
        lang_name="Japanese",
        retranslate=lambda rec, crit: "GOOD API call",
        per_seg={},
        judge_suspects=[],
        extra_flags=[3],
    )
    assert fixed == [3] and recs[0].txt == "GOOD API call"


def test_qe_review_tiers_confirmed_vs_subjective():
    from dev_titles.review import _render_qe_tiers

    suspects = [
        {"i": 5, "kind": "mqm-major[codex]", "detail": "mistranslation", "source": ""},
        {
            "i": 8,
            "kind": "mqm-minor[claude]",
            "detail": "awkward",
            "source": "",
        },  # lone -> subjective
        {
            "i": 8,
            "kind": "mqm-minor[codex]",
            "detail": "awkward",
            "source": "",
        },  # both @8 -> confirmed
        {
            "i": 12,
            "kind": "mqm-minor[codex]",
            "detail": "taste",
            "source": "",
        },  # lone -> subjective
        {"i": None, "kind": "term-inconsistency", "detail": "waitlist", "source": "waitlist"},
    ]
    out = "\n".join(_render_qe_tiers(suspects))
    assert (
        "4 for human judgment, 1 single-judge" in out
    )  # major@5 + both-minor@8(×2) + term = 4; lone@12 = 1
    # cue 8 (both judges) and major@5 and term-inconsistency are CONFIRMED; lone minors subjective
    assert "HUMAN-JUDGMENT" in out and "SUBJECTIVE" in out
    conf, subj = out.split("SUBJECTIVE")
    assert "cue 6" in conf and "cue 9" in conf  # major@5(->6) + corroborated minor@8(->9)
    assert "waitlist" in conf  # deterministic term issue is confirmed
    assert "cue 13" in subj  # lone minor@12(->13) is subjective


def test_qe_repair_accepts_omission_fix_on_penalty_tie():
    """A completed retranslation of a dropped-content segment is accepted even when the holistic
    re-judge TIES (short-of-expected -> complete) — so critical omissions aren't rejected 'no gain'
    (a regression seen on a long interview)."""
    from types import SimpleNamespace as T

    from dev_titles.qa import is_short_of_expected
    from dev_titles.translation_qe import build_judges, qe_repair

    src = "because the team was mad that I did not include the view model here at all"  # 15 words
    old = "ダメ"  # short-of-expected (omission)
    new = "チームは私がビューモデルをここにまったく含めなかったことに腹を立てていました"  # complete, not short
    assert is_short_of_expected(src, old, "ja") and not is_short_of_expected(src, new, "ja")
    # neither target contains 'BAD' -> the marker judge ties at 0/0, so ONLY the omission-accept
    # path can accept the fix
    recs = [T(i=0, src=src, txt=old)]
    per_seg = {0: {"critical": 1, "major": 0, "minor": 0}}
    fixed = qe_repair(
        recs,
        "ja",
        judges=build_judges(_mqm_by_target_marker),
        lang_name="Japanese",
        retranslate=lambda rec, crit: new,
        per_seg=per_seg,
        judge_suspects=[{"i": 0, "kind": "mqm-critical", "detail": "massive omission"}],
    )
    assert fixed == [0] and recs[0].txt == new
