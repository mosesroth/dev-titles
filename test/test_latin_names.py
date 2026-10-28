from types import SimpleNamespace as R

from dev_titles.latin_names import decide, load_corpus, names_in, never_latin


def test_names_in_strips_sentence_initial_filler():
    assert names_in("So Universal React and React Strict DOM.") == [
        "Universal React",
        "React Strict DOM",
    ]
    assert names_in("Can Figma do it?") == []


def test_never_latin_skips_pinned_and_names_kept_somewhere():
    recs = [
        R(i=0, src="Universal React is here.", txt="通用React来了"),
        R(i=1, src="Universal React again.", txt="又是通用React"),
        R(i=2, src="Tailwind CSS is fast.", txt="Tailwind CSS很快"),
        R(i=3, src="Acme Cloud works.", txt="Acme云可以"),
    ]
    assert never_latin(recs, exclude=["acme"]) == ["Universal React"]


def test_decide_corpus_majority_wins_over_the_model():
    corpus = [
        ("We use Universal React", "我们用Universal React"),
        ("Ask the Vice President", "问副总裁"),
    ]
    calls = []

    def inv(prompt):
        calls.append(prompt)
        return '{"Weather Tool": "latin"}'

    got = decide(
        ["Universal React", "Vice President", "Weather Tool"], "zh-Hans", "Chinese", corpus, inv
    )
    assert got == {
        "Universal React": "latin",
        "Vice President": "translate",
        "Weather Tool": "latin",
    }
    assert (
        len(calls) == 1 and '"Universal React":' not in calls[0]
    )  # only unseen names go to the model


def test_load_corpus_holds_out_the_video(tmp_path):
    rev, en = tmp_path / "in", tmp_path / "final"
    rev.mkdir(), en.mkdir()
    (rev / "A.zh-Hans_Reviewed.srt").write_text("x Universal React")
    (rev / "B.zh-Hans_Reviewed.srt").write_text("y")
    (en / "A.en.srt").write_text("Universal React")
    (en / "B.en.srt").write_text("B")
    got = load_corpus(
        "zh-Hans",
        reviewed=f"{rev}/**/*Reviewed*.srt",
        en_dir=en,
        tm_path=tmp_path / "none",
        holdout=["A"],
    )
    assert got == [("B", "y")]


def test_restore_keeps_a_retranslation_only_if_no_judge_prefers_the_old(monkeypatch):
    from types import SimpleNamespace as NS

    from dev_titles import latin_names, localize
    from dev_titles.glossary import Glossary

    recs = [
        NS(i=0, src="Universal React is here.", txt="通用React来了"),
        NS(i=1, src="I love Universal React.", txt="我爱通用React"),
    ]
    monkeypatch.setattr(latin_names, "load_corpus", lambda lang: [])
    monkeypatch.setattr(latin_names, "classify", lambda names, *a, **k: {n: "latin" for n in names})
    new = {0: "Universal React来了", 1: "我爱Universal React"}
    monkeypatch.setattr(localize, "make_retranslate", lambda *a, **k: (lambda r, c: new[r.i]))

    def judge(prompt):  # prefers the OLD line for sentence 1 only
        old_is_a = "A: 我爱通用React" in prompt
        if "我爱" in prompt:
            return '{"better": "A"}' if old_is_a else '{"better": "B"}'
        return '{"better": "equal"}'

    run = NS(
        glossary=Glossary(version="t", languages=["zh-Hans"], terms=[]),
        video_vocab={},
        conversational=False,
        qe_invoke=judge,
        qe_second_judge=None,
        en=None,
        translator=None,
        backcheck=False,
        logger=lambda m: None,
    )
    L = NS(lang="zh-Hans", records=recs, bank=None, suspects=[])
    localize._latin_name_consistency(run, L)
    assert recs[0].txt == "Universal React来了"  # restored: judge found it no worse
    assert recs[1].txt == "我爱通用React"  # kept: the judge preferred the original
    assert L.suspects and L.suspects[0]["kind"] == "latin-name-restored"
