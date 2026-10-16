"""Tests for the YouTube auto-caption 3-way consult (transcribe_qe.autosub_consult)."""

from dev_titles.model import Cue, Subtitle, Transcript, Word
from dev_titles.transcribe_qe import autosub_consult


def _words(pairs):
    # pairs: (text, start) -> Word with a 0.4s span
    return [Word(text=t, start=s, end=s + 0.4) for t, s in pairs]


def test_fixes_oov_name_when_autosub_and_second_agree():
    corrected = Subtitle("en", [Cue(6.0, 8.0, ["greatest voice agents and Freistan"])]).renumber()
    autosub = Subtitle("en", [Cue(6.0, 8.0, ["greatest voice agents and Lumora"])])
    second = Transcript("en", _words([("greatest", 6.0), ("agents", 6.8), ("Lumora", 7.5)]))
    fixes = autosub_consult(corrected, second, autosub, ["and Freistan"])
    assert corrected.cues[0].lines == ["greatest voice agents and Lumora"]
    assert fixes and fixes[0]["now"].endswith("Lumora")


def test_fixes_garbled_multiword_span():
    corrected = Subtitle("en", [Cue(9.0, 11.0, ["let's lower new tasks"])]).renumber()
    autosub = Subtitle("en", [Cue(9.0, 11.0, ["lets them learn new tasks"])])
    second = Transcript(
        "en",
        _words([("lets", 9.0), ("them", 9.4), ("learn", 9.8), ("new", 10.2), ("tasks", 10.6)]),
    )
    fixes = autosub_consult(corrected, second, autosub, ["let's lower new tasks"])
    # the garble 'lower' -> 'them learn' is fixed; tokens that normalize-equal keep the
    # primary's form (so 'let's' is preserved — the consult doesn't touch matching tokens).
    result = corrected.cues[0].lines[0]
    assert "lower" not in result and "them learn new tasks" in result
    assert len(fixes) == 1


def test_no_change_when_second_disagrees_with_autosub():
    # auto-caption says Lumora but the 2nd ASR does NOT -> single-source, don't overwrite
    corrected = Subtitle("en", [Cue(6.0, 8.0, ["agents and Freistan"])]).renumber()
    autosub = Subtitle("en", [Cue(6.0, 8.0, ["agents and Lumora"])])
    second = Transcript("en", _words([("agents", 6.0), ("and", 6.5), ("something", 7.2)]))
    fixes = autosub_consult(corrected, second, autosub, ["Freistan"])
    assert corrected.cues[0].lines == ["agents and Freistan"]
    assert fixes == []


def test_only_touches_flagged_cues():
    # a clean cue not in the flag list is never altered, even if autosub differs
    corrected = Subtitle("en", [Cue(1.0, 3.0, ["this part is fine"])]).renumber()
    autosub = Subtitle("en", [Cue(1.0, 3.0, ["this part is totally different"])])
    second = Transcript("en", _words([("totally", 1.5), ("different", 2.0)]))
    fixes = autosub_consult(corrected, second, autosub, ["some other garble"])
    assert corrected.cues[0].lines == ["this part is fine"]
    assert fixes == []


def test_noop_without_autosub_or_second():
    corrected = Subtitle("en", [Cue(6.0, 8.0, ["agents and Freistan"])]).renumber()
    assert autosub_consult(corrected, None, None, ["Freistan"]) == []


def test_repeated_run_replaced_from_autosub_no_2nd_asr():
    # a repeated-token cue (speaker stutter or decoder loop) is replaced from the auto-caption
    # UNCONDITIONALLY (no 2nd ASR), since both ASRs render it identically (e.g. a
    # 'the, the, the …' 6x stutter).
    corrected = Subtitle("en", [Cue(10.0, 13.0, ["the, the, the, the, the, the"])]).renumber()
    autosub = Subtitle("en", [Cue(10.0, 13.0, ["shaping the product in real time"])])
    fixes = autosub_consult(corrected, None, autosub, [])  # second_transcript=None on purpose
    assert corrected.cues[0].lines == ["shaping the product in real time"]
    assert fixes and "real time" in fixes[0]["now"]


def test_repeated_RUN_distributed_no_duplication():
    # a CONTIGUOUS run of repeated-token cues gets the auto-caption block distributed across
    # them by duration — no cross-cue duplication, each word used once.
    corrected = Subtitle(
        "en",
        [
            Cue(10.0, 12.0, ["the, the, the, the"]),
            Cue(12.0, 14.0, ["the, the, the, the"]),
            Cue(14.0, 16.0, ["the, the, the, the"]),
        ],
    ).renumber()
    autosub = Subtitle("en", [Cue(10.0, 16.0, ["alpha beta gamma delta epsilon zeta"])])
    autosub_consult(corrected, None, autosub, [])
    joined = " ".join(" ".join(c.lines) for c in corrected.cues).split()
    assert joined == ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]  # each once, in order
    assert all("the" not in " ".join(c.lines) for c in corrected.cues)


def test_recover_repeated_runs_pre_clean_read():
    from dev_titles.model import Transcript, Word
    from dev_titles.transcribe_qe import recover_repeated_runs

    # raw transcript with a 6x 'the' run between real words — a genuine speaker stutter
    # faithfully transcribed, NOT a decoder hallucination.
    words = [Word("real", 10.0, 10.3), Word("start", 10.3, 10.6)]
    words += [Word("the", 11.0 + 0.1 * i, 11.1 + 0.1 * i) for i in range(6)]
    words += [Word("end", 12.0, 12.3)]
    tr = Transcript("en", words)
    autosub = Subtitle("en", [Cue(11.0, 11.9, ["shaping the product in real time"])])
    fixed, n = recover_repeated_runs(tr, autosub, min_repeat=5)
    txt = " ".join(w.text for w in fixed.words)
    assert n == 1
    assert "the the the" not in txt
    assert "shaping the product" in txt


def test_recover_repeated_runs_noop_on_clean_transcript():
    from dev_titles.model import Transcript, Word
    from dev_titles.transcribe_qe import recover_repeated_runs

    tr = Transcript(
        "en", [Word(w, i, i + 0.4) for i, w in enumerate("this is a normal sentence".split())]
    )
    _, n = recover_repeated_runs(tr, Subtitle("en", [Cue(0, 5, ["x"])]))
    assert n == 0
