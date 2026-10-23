"""Split from test_core: resegment tests (20)."""

import pytest
from helpers import _transcribe_json

from dev_titles.config import spec_for
from dev_titles.cue_builder import build_en_cues
from dev_titles.model import Cue, Subtitle
from dev_titles.textwidth import display_width
from dev_titles.transcribe import from_transcribe_json


def test_merge_en_to_sentences():
    from dev_titles.resegment import merge_en_to_sentences

    en = Subtitle(
        "en",
        [
            Cue(0.0, 4.0, ["_"]),  # placeholder dropped
            Cue(4.0, 7.0, ["So welcome to my talk, which will be about"]),
            Cue(7.0, 9.0, ["performance tracing."]),  # sentence ends here
            Cue(9.0, 12.0, ["Why care?"]),
        ],
    )
    sents = merge_en_to_sentences(en)
    assert len(sents) == 2
    assert sents[0].text.endswith("performance tracing.")
    assert sents[0].start == 4.0 and sents[0].end == 9.0  # spans both cues
    assert sents[1].text == "Why care?" and sents[1].start == 9.0


def test_resegment_distributes_time_and_fits_spec():
    from dev_titles.resegment import resegment_translation

    spec = spec_for("ja")  # max_cpl 20, max_lines 2 -> 40 full-width units/cue
    text = "あ" * 100
    cues = resegment_translation(text, 0.0, 10.0, spec)
    assert len(cues) >= 2
    assert abs(cues[0].start) < 1e-9 and abs(cues[-1].end - 10.0) < 1e-6
    for a, b in zip(cues, cues[1:]):  # contiguous within the window
        assert abs(a.end - b.start) < 1e-9
    for cue in cues:  # obey lines + CPL
        assert len(cue.lines) <= spec.max_lines
        for ln in cue.lines:
            assert display_width(ln, cjk=True) <= spec.max_cpl


def test_audit_word_diffs_classify_and_sheet(tmp_path):
    pytest.importorskip("openpyxl")  # the .xlsx sheet needs the optional [xlsx] extra
    """audit module: substantive word-diffs (skip trivial/cue-only), conservative
    classify, and a verdict-labeled sheet."""
    from dev_titles.audit import classify, word_diffs, write_review_xlsx
    from dev_titles.model import Cue, Subtitle

    ours = Subtitle(
        "en", [Cue(0.0, 3.0, ["[music] we use native driver and the capital runtime"])]
    ).renumber()
    ref = Subtitle("en", [Cue(0.0, 3.0, ["we use nativeDriver and the Hermes runtime"])]).renumber()
    diffs = word_diffs(ours, ref)
    got = {(d["ours"], d["ref"]) for d in diffs}
    assert ("native driver", "nativeDriver") in got and ("capital", "Hermes") in got
    assert not any(d["ours"] == "[music]" for d in diffs)  # cue-only skipped
    classify(
        diffs,
        lambda p: '[{"n":1,"verdict":"ERROR","why":"x"},{"n":2,"verdict":"ERROR","why":"y"}]',
        domain="transcription",
    )
    counts = write_review_xlsx(diffs, tmp_path / "a.xlsx", title="t")
    assert (tmp_path / "a.xlsx").exists() and counts.get("ERROR", 0) >= 1


def test_strip_nonspeech_cues_dcmp():
    """DCMP: bare [music]/(applause)/♪ artifacts from WhisperX are not captioned in
    technical tutorials — build_en_cues drops them; real speech is kept."""
    from dev_titles.cue_builder import build_en_cues, strip_nonspeech_cues
    from dev_titles.model import Transcript, Word

    t = Transcript(
        "en",
        [
            Word("[music]", 0.0, 0.4),
            Word("Hi,", 0.5, 0.8),
            Word("I'm", 0.8, 1.0),
            Word("Priya.", 1.0, 1.6),
            Word("♪", 1.6, 2.0),
            Word("(applause)", 2.0, 2.5),
        ],
    )
    kept = [w.text for w in strip_nonspeech_cues(t).words]
    assert kept == ["Hi,", "I'm", "Priya."]
    text = " ".join(l for c in build_en_cues(t).cues for l in c.lines)
    assert "[music]" not in text and "♪" not in text and "(applause)" not in text
    assert "Priya." in text


def test_snap_gaps_closes_tiny_gaps_only():
    from dev_titles.resegment import snap_gaps

    spec = spec_for("ja")  # min_gap 2 frames @24fps = 0.083s
    cues = [Cue(0.0, 1.0, ["a"]), Cue(1.03, 2.0, ["b"]), Cue(3.0, 4.0, ["c"])]
    out = snap_gaps(cues, spec)
    assert abs(out[0].end - 1.03) < 1e-9  # tiny 0.03s gap closed
    assert abs(out[1].end - 2.0) < 1e-9  # 1.0s gap to next left intact
    assert out[2].start == 3.0


def test_resegment_caps_max_dur():
    from dev_titles.resegment import resegment_translation

    spec = spec_for("zh-Hans")  # max_dur 7
    # Little text, huge 30s window -> a single cue must not display > max_dur.
    cues = resegment_translation("你好", 0.0, 30.0, spec)
    assert all(c.duration <= spec.max_dur + 1e-6 for c in cues)


def test_resegment_merges_sub_min_dur_cue():
    from dev_titles.resegment import resegment_translation

    spec = spec_for("ja")
    # A long chunk + a 1-char tail; the tail alone would be < min_dur -> merged.
    cues = resegment_translation("あ" * 26 + "い", 0.0, 6.0, spec)
    assert all(c.duration >= spec.min_dur - 1e-6 for c in cues)


def test_audit_retiming_filter():
    """word_diffs must not flag an omission/addition when the content is present
    ELSEWHERE (re-segmentation move), but must keep a genuinely-absent clause."""
    from dev_titles.audit import word_diffs
    from dev_titles.model import Cue, Subtitle

    # ours has the same content as ref but the 'if no options click about' clause is
    # RE-TIMED to a different cue order; plus ours is genuinely MISSING 'serial number seven'
    ref = Subtitle(
        "en",
        [
            Cue(0.0, 2.0, ["Open the developer options menu"]),
            Cue(2.0, 4.0, ["if no options click about settings"]),
            Cue(4.0, 6.0, ["then tap serial number seven times"]),
        ],
    )
    ours = Subtitle(
        "en",
        [
            Cue(0.0, 2.0, ["if no options click about settings"]),  # moved earlier
            Cue(2.0, 4.0, ["open the developer options menu"]),
        ],
    )  # missing the serial clause
    rows = word_diffs(ours, ref)
    blob = " ".join(f"{r['ours']}|{r['ref']}" for r in rows)
    assert "click about settings" not in blob  # re-timed clause NOT flagged
    assert "serial number seven times" in blob  # genuinely-absent clause IS flagged


def test_build_en_cues_respects_spec_and_splits_on_pause():
    # Two sentences separated by a >0.7s pause -> at least two cues.
    words = [
        ("The", 0.0, 0.2),
        ("quick", 0.2, 0.5),
        ("brown", 0.5, 0.8),
        ("fox.", 0.8, 1.1),
        ("Then", 3.0, 3.3),
        ("it", 3.3, 3.5),
        ("ran.", 3.5, 3.9),
    ]
    t = from_transcribe_json(_transcribe_json(words))
    sub = build_en_cues(t)
    assert len(sub.cues) >= 2
    spec = spec_for("en")
    for cue in sub.cues:
        assert cue.duration >= spec.min_dur - 1e-9
        for ln in cue.lines:
            assert len(ln) <= spec.max_cpl
    # cue indices are 1-based and contiguous
    assert [c.index for c in sub.cues] == list(range(1, len(sub.cues) + 1))


def test_exemplar_align_and_select():
    from dev_titles.exemplars import ExemplarBank, align_by_time
    from dev_titles.model import Cue, Subtitle

    en = Subtitle(
        "en",
        [
            Cue(0.0, 2.0, ["You can control focus."]),
            Cue(2.0, 4.0, ["It changes how apps work."]),
        ],
    )
    human = Subtitle(
        "ja",
        [
            Cue(0.0, 2.0, ["フォーカスを制御できます"]),
            Cue(2.0, 4.0, ["アプリの動作が変わります"]),
        ],
    )
    pairs = align_by_time(en, human, "ja", "T")
    assert len(pairs) == 2
    assert pairs[0].source == "You can control focus."
    assert pairs[0].target == "フォーカスを制御できます"

    bank = ExemplarBank("ja").add(pairs)
    # lexical overlap ranks the focus sentence first
    sel = bank.select("How do I control focus on an element?", k=1)
    assert sel and "フォーカス" in sel[0][1]
    # a query identical to an exemplar source is skipped (no answer leakage)
    sel2 = bank.select("You can control focus.", k=2)
    assert all(s != "You can control focus." for s, _ in sel2)


def test_exemplar_min_overlap_filters_grazing_cue():
    from dev_titles.exemplars import align_by_time
    from dev_titles.model import Cue, Subtitle

    en = Subtitle("en", [Cue(0.0, 10.0, ["One long sentence here."])])
    # first human cue overlaps fully; second only grazes the last 0.1s (2% of its 5s)
    human = Subtitle(
        "ja",
        [
            Cue(0.0, 9.9, ["本文"]),
            Cue(9.9, 14.9, ["次の文"]),
        ],
    )
    pairs = align_by_time(en, human, "ja", "T", min_overlap=0.2)
    assert pairs[0].target == "本文"


def test_merge_splits_multi_sentence_block():
    from dev_titles.resegment import merge_en_to_sentences

    en = Subtitle(
        "en",
        [
            Cue(
                0.0,
                9.0,
                [
                    "The latest release has an improved algorithm. Apps should behave the same. Navigation is intuitive."
                ],
            )
        ],
    )
    s = merge_en_to_sentences(en)
    assert len(s) == 3  # blob split into single sentences
    assert s[0].text.startswith("The latest") and s[0].end <= s[1].start
    assert s[-1].end == 9.0  # last keeps the block end


def test_write_status_merges_json():
    """Observability #2: _write_status merges fields into <dir>/status.json + stamps updated_at."""
    import json
    import os
    import tempfile

    from dev_titles.localize import _write_status

    d = tempfile.mkdtemp()
    cache = os.path.join(d, "cache.jsonl")
    _write_status(cache, phase="translate", lang="ja")
    _write_status(cache, phase="done", dqf_ja=96.1)
    st = json.load(open(os.path.join(d, "status.json")))
    assert st["phase"] == "done" and st["lang"] == "ja" and st["dqf_ja"] == 96.1
    assert "updated_at" in st
    _write_status(None)  # no-op, must not raise


def test_split_cjk_proportional_token_safe_and_lossless():
    """Proportional cue-split must be lossless AND never cut inside a Latin/number token
    (Expo->'Ex'|'po'). Snaps mid-token boundaries to the nearest token edge."""
    from dev_titles.resegment import _in_latin_token, _split_cjk_proportional

    # a translation with an embedded Latin token where the ~50% split lands mid-'Expo'
    t = "用React NativeでExpoのアプリを開発する"
    parts = _split_cjk_proportional(t, [1, 1])
    assert "".join(parts) == t  # lossless
    assert all(parts)  # no empty piece
    # no boundary sits inside a Latin/number token
    cut = len(parts[0])
    assert not _in_latin_token(t, cut)
    # explicit: 'Expo' and 'React' and 'Native' never split across the two pieces
    for tok in ("Expo", "React", "Native"):
        assert (tok in parts[0]) or (tok in parts[1]) or (tok not in t)
    # 3-way split of a number-bearing line stays lossless + token-safe
    t2 = "13か国に100人を超えるメンバーがいて3拠点で急成長中"
    p3 = _split_cjk_proportional(t2, [2, 1, 1])
    assert "".join(p3) == t2 and all(p3)
    for c in (len(p3[0]), len(p3[0]) + len(p3[1])):
        assert not _in_latin_token(t2, c)


def test_sentence_split_lossless_with_dotted_token():
    # a mid-token dot (skills.sh, v1.2) must NOT drop the preceding clause
    from dev_titles.resegment import merge_en_to_sentences

    en = Subtitle(
        "en", [Cue(0.0, 6.0, ["Grab skills from skills.sh. Once configured, you can use it."])]
    )
    s = merge_en_to_sentences(en)
    joined = " ".join(x.text for x in s)
    assert "skills.sh" in joined and "Grab skills from" in joined  # nothing dropped
    assert len(s) == 2


def test_build_en_cues_line_cap_and_no_orphan():
    from dev_titles.model import Transcript, Word

    # contiguous words; a full clause then a short sentence-final tail ("coding agent.")
    toks = (
        "a new way to accelerate Orbit TV app development using your preferred AI " "coding agent."
    ).split()
    words = [Word(t, i * 0.4, i * 0.4 + 0.4) for i, t in enumerate(toks)]
    sub = build_en_cues(Transcript("en", words))
    # no cue exceeds 2 lines, and no 1-word orphan cue survives
    assert all(len(c.lines) <= 2 for c in sub.cues), [c.lines for c in sub.cues]
    assert all(len(" ".join(c.lines).split()) > 1 for c in sub.cues), [c.lines for c in sub.cues]


def test_condense_dense_cues_only_touches_fast_ones():
    from dev_titles.cleanup import condense_dense_cues
    from dev_titles.model import Cue, Subtitle

    # EN cap 20 cps. Fast cue (60 chars / 1s = 60 cps) rewritten; slow cue left alone.
    fast = Cue(0.0, 1.0, ["this is a very wordy and redundant sentence indeed yes"])
    slow = Cue(2.0, 6.0, ["short line"])
    calls = {"n": 0}

    def fake(p):
        calls["n"] += 1
        return "tight sentence"

    condense_dense_cues(Subtitle("en", [fast, slow]), fake)
    assert calls["n"] == 1  # only the fast cue
    assert fast.lines == ["tight sentence"] and slow.lines == ["short line"]


def test_align_translation_to_en_1to1():
    """Strict 1:1 EN-cue alignment: one target cue per EN cue, same timestamps; a multi-cue
    sentence's translation is distributed across its cues (lossless concat)."""
    from dev_titles.config import spec_for
    from dev_titles.model import Cue, Subtitle
    from dev_titles.resegment import align_translation_to_en, merge_en_to_sentences

    en = Subtitle(
        "en",
        [
            Cue(0.0, 2.0, ["Hello there."]),
            Cue(2.0, 4.0, ["This is a longer"]),
            Cue(4.0, 6.0, ["sentence that spans two cues."]),
        ],
    )
    sents = merge_en_to_sentences(en)  # -> 2 sentences: [0,2] and [2,6]
    texts = ["你好。", "这是一个跨越两个提示的更长句子。"]
    cues = align_translation_to_en(en, sents, texts, spec_for("zh-Hans"))
    assert len(cues) == 3  # 1 cue per EN content cue
    assert [(c.start, c.end) for c in cues] == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]  # EN timestamps
    assert "".join("".join(c.lines) for c in cues[1:]).replace(" ", "") == texts[1].replace(" ", "")
    assert "".join(cues[0].lines) == "你好。"


def test_build_target_cues_free_segment():
    """Free segmentation (1:1 lifted): each sentence's translation is re-timed within its OWN
    window, sized to the language's cps/max_lines — no EN-cue-count constraint. A dense sentence
    splits into MORE cues than EN had; invariants: sorted, no overlap, every cue <= max_dur."""
    from dev_titles.config import spec_for
    from dev_titles.model import Cue, Subtitle
    from dev_titles.resegment import build_target_cues, merge_en_to_sentences

    spec = spec_for("ja")  # max_lines 2, max_dur 7
    en = Subtitle(
        "en", [Cue(0.0, 3.0, ["A short cue."]), Cue(3.0, 20.0, ["A very long sentence."])]
    )
    sents = merge_en_to_sentences(en)
    texts = ["短い。", "非常に長い文章が" * 6]  # dense -> must split across several cues
    one = build_target_cues(en, sents, texts, spec, free_segment=False)
    free = build_target_cues(en, sents, texts, spec, free_segment=True)
    assert len(one) == 2  # 1:1 keeps the EN cue count
    assert len(free) > 2  # free splits the dense sentence into more cues
    for a, b in zip(free, free[1:]):
        assert b.start >= a.end - 1e-6  # no overlap, sorted
    for c in free:
        assert c.end > c.start
        assert c.duration <= spec.max_dur + 1e-6  # capped at max_dur


def test_enforce_min_dur_extends_and_borrows():
    """A sub-min_dur cue extends into the following gap; if butted against the next cue it borrows
    from the next's slack (never starving the neighbor below min_dur)."""
    from dev_titles.config import spec_for
    from dev_titles.model import Cue
    from dev_titles.resegment import enforce_min_dur

    spec = spec_for("zh-Hans")  # min_dur 0.833
    # cue 0 is 0.3s then a gap; cue 1 is 0.3s butted against a long cue 2 with slack
    cues = [
        Cue(0.0, 0.3, ["对"]),  # gap after -> extend into it
        Cue(2.0, 2.3, ["是的"]),  # butted to cue2 -> borrow from cue2's slack
        Cue(2.3, 9.0, ["一段很长的话"]),
    ]
    out = enforce_min_dur(cues, spec)
    assert out[0].duration >= spec.min_dur - 1e-6  # extended into the gap
    assert out[1].duration >= spec.min_dur - 1e-6  # borrowed from the next
    assert out[-1].duration >= spec.min_dur - 1e-6  # neighbor not starved
    for a, b in zip(out, out[1:]):
        assert b.start >= a.end - 1e-6  # still no overlap


def test_reflow_lossless_guard():
    """LLM re-flow re-segments (never re-translates); a good split is used, a lossy one trips the
    guard and falls back to the proportional split — output is always lossless."""
    from dev_titles.config import spec_for
    from dev_titles.model import Cue, Subtitle
    from dev_titles.resegment import align_translation_to_en, merge_en_to_sentences

    en = Subtitle("en", [Cue(0.0, 2.0, ["A B"]), Cue(2.0, 4.0, ["C D E"])])
    sents = merge_en_to_sentences(en)  # one sentence [0,4], two cues
    texts = ["甲乙丙丁戊"]
    good = lambda items: [["甲乙", "丙丁戊"]]  # batched: one item → its chunks (lossless)
    cues = align_translation_to_en(en, sents, texts, spec_for("zh-Hans"), reflow=good)
    assert [("".join(c.lines)) for c in cues] == ["甲乙", "丙丁戊"]  # used the re-flow
    bad = lambda items: [["甲乙", "丙"]]  # DROPS 丁戊 → guard must reject
    cues2 = align_translation_to_en(en, sents, texts, spec_for("zh-Hans"), reflow=bad)
    assert (
        "".join("".join(c.lines) for c in cues2).replace(" ", "") == "甲乙丙丁戊"
    )  # fell back, lossless


def test_reflow_cache_reuses_unchanged_no_llm():
    """reflow_llm_batch reuses cached (cues,translation) splits with NO LLM call — so re-aligning
    after a small edit re-flows only changed sentences (the ~40min -> seconds win)."""
    from dev_titles.resegment import reflow_llm_batch

    calls = {"n": 0}

    def invoke(prompt):
        calls["n"] += 1
        # split sentence 0's translation "甲乙丙丁" into its 2 cues
        return '{"0": ["甲乙", "丙丁"]}'

    items = [(["hello there", "world"], "甲乙丙丁")]
    cache = {}
    out1 = reflow_llm_batch(items, invoke, "Japanese", cache=cache)
    assert out1 == [["甲乙", "丙丁"]] and calls["n"] == 1 and cache  # computed + stored
    # second run with the SAME item: served from cache, no new LLM call
    out2 = reflow_llm_batch(items, invoke, "Japanese", cache=cache)
    assert out2 == [["甲乙", "丙丁"]] and calls["n"] == 1  # unchanged
    # a CHANGED translation is not cached -> triggers an LLM call
    reflow_llm_batch([(["hello there", "world"], "戊己庚辛")], invoke, "Japanese", cache=cache)
    assert calls["n"] == 2


def test_normalize_en_layout_rewraps_corrected_cues():
    """EN corrections write a cue back as ONE long line; the EN sink must re-wrap it (and split a
    cue that needs >2 lines) without changing the text."""
    from dev_titles.model import Cue, Subtitle
    from dev_titles.resegment import normalize_en_layout
    from dev_titles.validate.rules import validity_report

    long = "you need to fill out another Jira ticket, for example, it should know what that ticket is about"
    sub = Subtitle("en", [Cue(0.0, 6.0, [long]), Cue(6.0, 9.0, ["A fine cue."])])
    out = normalize_en_layout(sub)
    assert validity_report(out, True)["cues_with_errors"] == 0
    assert " ".join(" ".join(c.lines) for c in out.cues) == long + " A fine cue."
