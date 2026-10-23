"""Split from test_core: misc tests (24)."""

from dev_titles.glossary import load
from dev_titles.model import Cue, Subtitle


def test_anthropic_translator_builds_request_and_extracts_text():
    from dev_titles.translate import AnthropicTranslator, TranslationRequest

    class _Block:
        def __init__(self, text):
            self.type, self.text = "text", text

    class _Resp:
        content = [_Block("翻訳結果")]

    class _FakeClient:
        def __init__(self):
            self.calls = []

        class messages:  # noqa: N801
            pass

    captured = {}

    class _Msgs:
        def create(self, **kw):
            captured.update(kw)
            return _Resp()

    class _Mid:
        def __init__(self, id):
            self.id = id

    class _Models:
        def list(self):
            return [_Mid("claude-opus-4-8"), _Mid("claude-opus-5")]

    client = _FakeClient()
    client.messages = _Msgs()
    client.models = _Models()
    tr = AnthropicTranslator(client=client, effort="low")  # model_id defaults to "auto"
    out = tr.translate(TranslationRequest("Submit your app", "en", "ja"))
    assert out == "翻訳結果"
    assert captured["model"] == "claude-opus-5"  # auto-picked newest opus
    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["output_config"] == {"effort": "low"}
    assert captured["messages"][0]["role"] == "user"


def test_model_select_version_and_pick():
    from dev_titles.translate.model_select import parse_version, pick_latest, resolve_model

    assert parse_version("claude-opus-5", "claude-opus") == (5,)
    assert parse_version("claude-opus-4-8", "claude-opus") == (4, 8)
    assert parse_version("claude-opus-4-5-20251101", "claude-opus") == (4, 5)
    assert parse_version("claude-sonnet-5", "claude-opus") is None
    ids = ["claude-opus-4-7", "claude-opus-5", "claude-opus-4-8", "claude-sonnet-5"]
    assert pick_latest(ids, "claude-opus") == "claude-opus-5"
    # a future opus-6 auto-wins:
    assert pick_latest(ids + ["claude-opus-6"], "claude-opus") == "claude-opus-6"

    class _C:
        class models:
            @staticmethod
            def list():
                return [type("M", (), {"id": i})() for i in ids + ["claude-opus-6"]]

    assert resolve_model(_C(), "auto") == "claude-opus-6"  # newest wins
    assert resolve_model(_C(), "claude-opus-5") == "claude-opus-5"  # explicit pins


def test_speaker_name_handling():
    """Speaker names: proposed from self-intros, injected into the correction prompt
    as spell-exactly targets, and kept Latin (DNT) in translation."""
    from dev_titles.cleanup import build_correction_prompt
    from dev_titles.localize import effective_glossary
    from dev_titles.vocab_prep import extract_name_candidates, renderings_to_terms  # noqa

    cands = extract_name_candidates(
        "Hello, my name is Chris Hartley. Today I'm Priya and we build."
    )
    assert "Chris Hartley" in cands and "Priya" in cands

    from dev_titles.vocab_prep import harvest_names

    # presenter LISTS in metadata (comma / Oxford-comma-and / ampersand)
    assert harvest_names(
        "", "Speakers: Michael Brennan, Chris Hartley, and Priya Kumar from Acme"
    ) == ["Michael Brennan", "Chris Hartley", "Priya Kumar"]
    assert harvest_names("", "A talk by Jane Doe & John Smith") == ["Jane Doe", "John Smith"]
    # role-title + name (real YouTube-description forms); the title itself is not a name
    assert harvest_names(
        "", "Join Senior Solutions Architect Chris Hartley as he walks through"
    ) == ["Chris Hartley"]
    assert harvest_names(
        "", "In this video Sr. Developer Evangelist Giovanni Rinaldi discusses"
    ) == ["Giovanni Rinaldi"]

    p = build_correction_prompt("we use it", ["Expo"], names=["Chris Hartley"])
    assert "SPEAKER" in p and "Chris Hartley" in p

    g = load("glossary/glossary.yaml")
    _, dnt = effective_glossary(
        g, {"identifiers": [], "terms": [], "renderings": {}, "names": ["Chris Hartley", "Priya"]}
    )
    assert "Chris Hartley" in dnt and "Priya" in dnt  # names kept Latin in translation


def test_identify_subject_for_live_no_metadata():
    """Live (no YouTube metadata): identify the subject FROM the transcript and name
    doc URLs to pre-ingest; flag confident=false so the caller asks for a URL."""
    from dev_titles.vocab_prep import identify_subject

    ok = identify_subject(
        "today we add drag and drop with react native",
        lambda p: '{"subject":"RN drag & drop","products":["React Native"],'
        '"doc_urls":["https://reactnative.dev/docs/x","not-a-url"],"confident":true}',
    )
    assert ok["confident"] and ok["products"] == ["React Native"]
    assert ok["doc_urls"] == ["https://reactnative.dev/docs/x"]  # non-URL dropped
    unsure = identify_subject(
        "mumble mumble", lambda p: '{"subject":"","doc_urls":[],"confident":false}'
    )
    assert unsure["confident"] is False and unsure["doc_urls"] == []


def test_recover_dropped_spans():
    """Splice back a whole span the primary dropped but the 2nd engine heard (with the
    2nd engine's timestamps), touching only the gap — not short word-choice diffs."""
    from dev_titles.model import Transcript, Word
    from dev_titles.transcribe_qe import recover_dropped_spans

    # primary dropped a 5-word span between 'system.' and 'Welcome'
    primary = Transcript(
        "en",
        [
            Word("new", 0.0, 0.3),
            Word("operating", 0.3, 0.7),
            Word("system.", 0.7, 1.1),
            Word("Welcome", 9.0, 9.4),
            Word("everyone.", 9.4, 9.9),
        ],
    )
    second = Transcript(
        "en",
        [
            Word("new", 0.0, 0.3),
            Word("operating", 0.3, 0.7),
            Word("system.", 0.7, 1.1),
            Word("This", 2.0, 2.2),
            Word("is", 2.2, 2.4),
            Word("the", 2.4, 2.6),
            Word("next", 2.6, 2.9),
            Word("Orbit.", 2.9, 3.4),
            Word("Welcome", 9.0, 9.4),
            Word("everyone.", 9.4, 9.9),
        ],
    )
    merged, n = recover_dropped_spans(primary, second, min_run=4)
    assert n == 5
    texts = [w.text for w in merged.words]
    assert texts == [
        "new",
        "operating",
        "system.",
        "This",
        "is",
        "the",
        "next",
        "Orbit.",
        "Welcome",
        "everyone.",
    ]  # spliced in by timestamp
    # a short 1-2 word difference is NOT recovered (word-choice, not a drop)
    p2 = Transcript("en", [Word("we", 0.0, 0.3), Word("ran.", 0.3, 0.6)])
    s2 = Transcript("en", [Word("we", 0.0, 0.3), Word("all", 0.3, 0.5), Word("ran.", 0.5, 0.8)])
    _, n2 = recover_dropped_spans(p2, s2, min_run=4)
    assert n2 == 0


def test_second_pass_drops_segmentation_noise():
    """Pure segmentation differences (same letters, different word boundaries) are
    cosmetic — NOT flagged; a genuine word difference IS."""
    from dev_titles.model import Transcript, Word
    from dev_titles.transcribe_qe import _second_pass_disagreements

    # 'Orbit OS' vs 'OrbitOS' (segmentation) and 'Big' vs 'Nimbus' (real mishear),
    # separated by matching words as they are in real transcripts.
    a = Transcript(
        "en",
        [
            Word("Orbit", 0.0, 0.3),
            Word("OS", 0.3, 0.6),
            Word("is", 0.6, 0.8),
            Word("great.", 0.8, 1.1),
            Word("Big", 1.1, 1.5),
            Word("win.", 1.5, 1.8),
        ],
    )
    b = Transcript(
        "en",
        [
            Word("OrbitOS", 0.0, 0.6),
            Word("is", 0.6, 0.8),
            Word("great.", 0.8, 1.1),
            Word("Nimbus", 1.1, 1.5),
            Word("win.", 1.5, 1.8),
        ],
    )
    dis = _second_pass_disagreements(a, b)
    details = " ".join(d["detail"] for d in dis)
    assert "Big" in details  # real disagreement kept
    assert "Orbit" not in details  # segmentation-only dropped
    assert len(dis) == 1


def test_ingest_resolve_input(tmp_path):
    """Input resolver: local subtitle/media pass through by extension, frame.io is a
    clear not-yet-wired error, missing paths raise, YouTube URLs route to the fetcher."""
    import pytest

    from dev_titles.ingest import _YT, resolve_input

    sub = tmp_path / "clip.srt"
    sub.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
    r = resolve_input(str(sub))
    assert r.kind == "subtitle" and r.title == "clip"
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"\x00")
    assert resolve_input(str(media)).kind == "media"
    with pytest.raises(NotImplementedError):
        resolve_input("https://frame.io/abc")
    with pytest.raises(FileNotFoundError):
        resolve_input(str(tmp_path / "nope.mp4"))
    assert _YT.search("https://youtu.be/xyz") and _YT.search("https://www.youtube.com/watch?v=x")


def test_localize_en_end_to_end_with_fake_translator(tmp_path):
    """localize_en runs the full flow (translate + glossary + coverage gate +
    CJK re-segmentation) for each language, keeps multi-word terms whole, applies
    per-video renderings, and is resumable via cache."""
    from dev_titles.localize import localize_en
    from dev_titles.model import Cue, Subtitle
    from dev_titles.translate.base import Translator

    class Fake(Translator):
        def translate(self, req):
            return "本日はReact Nativeの並行レンダリングについてご紹介します"

        def reflect_and_refine(self, req, draft):
            return draft

        def _invoke(self, prompt):
            return "COMPLETE"  # coverage gate: nothing missing

    en = Subtitle(
        "en", [Cue(0.0, 4.0, ["Today, about concurrent rendering in React Native."])]
    ).renumber()
    g = load("glossary/glossary.yaml")
    vocab = {"identifiers": [], "terms": [], "renderings": {}}
    cache = tmp_path / "cache.jsonl"
    outs = localize_en(
        en,
        Fake(),
        g,
        langs=("ja", "zh-Hans"),
        video_vocab=vocab,
        backcheck=True,
        cache_path=str(cache),
    )
    assert {o.language for o in outs} == {"ja", "zh-Hans"}
    ja = next(o for o in outs if o.language == "ja")
    joined = " ".join(l for c in ja.subtitle.cues for l in c.lines)
    assert "React Native" in joined and "React /" not in joined  # multi-word term whole
    # avoid-form 並行レンダリング is normalized to the reviewer-decided 並列レンダリング
    assert "並列レンダリング" in joined and "並行レンダリング" not in joined
    assert cache.exists()  # resumable cache written


def test_qa_omission_and_dqf_gate():
    from dev_titles.config import spec_for
    from dev_titles.qa import dqf_gate, dqf_score, omission_suspects

    spec = spec_for("ja")  # cap 7
    # 4 chars in 6s = 0.67 cps << 7 for an 8-word source -> suspect
    pairs = [("this is a fairly long source sentence here now", "短い訳", 6.0)]
    assert omission_suspects(pairs, spec)
    # a full cue at cap is not flagged
    assert not omission_suspects(
        [("short one two three four five six seven", "あ" * 30, 6.0)], spec
    )
    # DQF: 1 major in 1000 words -> 97.5; gate(96) passes; any critical fails
    assert abs(dqf_score({"major": 1}, 1000) - 97.5) < 1e-6
    assert dqf_gate({"major": 1}, 1000)[0] is True
    assert dqf_gate({"critical": 1}, 1000)[0] is False


def test_parakeet_words_from_tokens():
    from types import SimpleNamespace as T

    from dev_titles.transcribe import words_from_tokens

    # Leading-space = word boundary; no-space tokens (subwords, punctuation) glue on.
    # Mirrors real Parakeet output for "Hi, I'm Priya".
    toks = [
        T(text=" H", start=9.20, end=9.36, confidence=0.98),
        T(text="i", start=9.36, end=9.60, confidence=1.0),
        T(text=",", start=9.60, end=9.68, confidence=0.99),
        T(text=" I", start=9.68, end=9.76, confidence=1.0),
        T(text="'", start=9.76, end=9.84, confidence=1.0),
        T(text="m", start=9.84, end=9.92, confidence=1.0),
        T(text=" Pr", start=9.92, end=10.16, confidence=0.70),
        T(text="iy", start=10.16, end=10.32, confidence=0.90),
        T(text="a", start=10.56, end=10.64, confidence=0.95),
    ]
    ws = words_from_tokens(toks)
    assert [w.text for w in ws] == ["Hi,", "I'm", "Priya"]
    assert abs(ws[0].start - 9.20) < 1e-9 and abs(ws[2].end - 10.64) < 1e-9
    # word confidence = MIN over its subwords (the shaky "Pr" 0.70 drives "Priya")
    assert abs(ws[2].confidence - 0.70) < 1e-9
    # space-only tokens are pure boundaries (no empty word emitted)
    ws2 = words_from_tokens(
        [
            T(text=" ", start=0.0, end=0.1, confidence=1.0),
            T(text="ok", start=0.1, end=0.3, confidence=1.0),
        ]
    )
    assert [w.text for w in ws2] == ["ok"]


def test_tm_import_tmx(tmp_path):
    import pytest

    pytest.importorskip("defusedxml")  # optional `tmx` extra
    from dev_titles.tm import TranslationMemory

    tmx = tmp_path / "t.tmx"
    tmx.write_text(
        '<?xml version="1.0"?><tmx version="1.4"><body>'
        '<tu><tuv xml:lang="en-us"><seg>Get started</seg></tuv>'
        '<tuv xml:lang="zh-cn"><seg>开始使用</seg></tuv>'
        '<tuv xml:lang="ja-jp"><seg>開始する</seg></tuv></tu>'
        '<tu><tuv xml:lang="en-us"><seg>Learn more</seg></tuv>'
        '<tuv xml:lang="zh-cn"><seg>了解更多</seg></tuv></tu></body></tmx>',
        encoding="utf-8",
    )
    tm = TranslationMemory()
    assert tm.import_tmx(tmx) == 3  # 2 zh + 1 ja pairs
    assert tm.exact("Get started", "zh-Hans") == "开始使用"
    assert tm.exact("Get started", "ja") == "開始する"
    assert tm.exact("Learn more", "zh-Hans") == "了解更多"


def test_dedupe_adjacent_words():
    from dev_titles.cleanup import dedupe_adjacent_words

    assert dedupe_adjacent_words("improve the the software") == "improve the software"
    assert dedupe_adjacent_words("we we should go") == "we should go"
    # validly-doubling / content-word repeats are NOT touched
    assert dedupe_adjacent_words("the fact that that happened") == "the fact that that happened"
    assert dedupe_adjacent_words("I had had enough") == "I had had enough"
    assert dedupe_adjacent_words("a big big win") == "a big big win"
    # a repeat carrying punctuation is left alone (no comma loss)
    assert dedupe_adjacent_words("So so, it works") == "So so, it works"


def test_name_registry_match(tmp_path):
    from dev_titles.vocab_prep import load_name_registry, match_registry_names

    reg = [
        "Eric Vandermolen",
        "Giovanni",
        "Giovanni Rinaldi",
        "Gio",
        "Chris Delgado",
        "Chris Hartley",
        "Priya Malhotra",
        "Will Jones",
    ]
    text = "Acme Product Manager Eric Vandermolen and Developer Advocate Giovanni sit down."
    hits = match_registry_names(text, reg, first_name_context=text)
    assert "Eric Vandermolen" in hits  # full-name whole-word
    assert "Giovanni Rinaldi" in hits  # bare "Giovanni" promoted to unique fuller
    assert "Gio Rinaldi" not in hits and "Gio" not in hits  # 'gio' must NOT match inside 'giovanni'
    # common-English-word first name must NOT match on the verb "will"
    assert "Will Jones" not in match_registry_names(
        "we will build apps", reg, first_name_context="we will build apps"
    )
    # ambiguous common first name alone must NOT match (two Chrises in the registry)
    assert (
        match_registry_names("So then Chris said", reg, first_name_context="So then Chris said")
        == []
    )
    # but a full mention of an ambiguous-first-name person does
    assert "Chris Hartley" in match_registry_names("here with Chris Hartley", reg)
    # MANGLED SURNAME in the transcript (no description) still resolves to the canonical via the
    # unique uncommon first name — 'Emerson Vail' (ASR mishear) -> registry 'Emerson Vale'.
    # A redundant bare 'Emerson' entry must NOT block this (uniqueness is among FULL names).
    reg2 = ["Emerson", "Emerson Vale", "Will Jones"]
    hits2 = match_registry_names("I'm Emerson Vail, Chief Evangelist", reg2, first_name_context="")
    assert "Emerson Vale" in hits2  # surfaced despite wrong surname + no metadata
    assert "Emerson" not in hits2  # bare entry not matched (description-only) / contained
    # a registry file loads (people[].name); a missing file is an empty registry
    reg_file = tmp_path / "names.yaml"
    reg_file.write_text(
        "people:\n  - name: Eric Vandermolen\n  - name: ' Priya Malhotra '\n  - {role: host}\n",
        encoding="utf-8",
    )
    assert load_name_registry(str(reg_file)) == ["Eric Vandermolen", "Priya Malhotra"]
    assert load_name_registry(str(tmp_path / "missing.yaml")) == []


def test_rescore_after_repair_patches_changed_only():
    from types import SimpleNamespace as T

    from dev_titles.translation_qe import build_judges, rescore_after_repair

    recs = [T(i=0, src="one two three", txt="一二三"), T(i=1, src="four five six", txt="四五六")]
    glossary = T(terms=[], identifiers=[])
    # prior panel: seg0 had a major (now fixed); judge is called ONLY for changed seg0
    prior = {0: {"claude": {"critical": 0, "major": 1, "minor": 0}}}
    prior_mqm = [{"i": 0, "kind": "mqm-major", "detail": "old"}]
    calls = []

    def judge(prompt):
        calls.append(prompt)
        return "[0] no-error"  # the repaired seg0 is now clean

    suspects, dqf = rescore_after_repair(
        recs,
        glossary,
        "ja",
        judges=build_judges(judge),
        lang_name="Japanese",
        prior_per_seg=prior,
        prior_mqm_suspects=prior_mqm,
        changed={0},
    )
    assert dqf["counts"]["major"] == 0  # seg0 patched to clean
    assert len(calls) == 1 and "four five six" not in calls[0]  # only the CHANGED seg re-judged
    assert not any(s["kind"] == "mqm-major" for s in suspects)  # stale major dropped


def test_incremental_review_survives_segmentation_shift(tmp_path):
    # A mid-video EN edit inserts a sentence, renumbering every later one. The QE state is keyed
    # by SOURCE text, so the renumbered-but-identical sentences reuse their stored panels — only
    # the genuinely-new sentence is re-judged (not the whole tail, the old index-keyed bug).
    from types import SimpleNamespace as T

    from dev_titles.translation_qe import (
        build_judges,
        incremental_review,
        load_qe_state,
        save_qe_state,
    )

    glossary = T(terms=[], identifiers=[])
    recs0 = [T(i=0, src="alpha beta", txt="AB"), T(i=1, src="gamma delta", txt="GD")]
    dqf = {
        "per_seg": {
            0: {"claude": {"critical": 0, "major": 0, "minor": 0}},
            1: {"claude": {"critical": 0, "major": 0, "minor": 1}},
        }
    }
    state_path = tmp_path / "qe_state.ja.json"
    save_qe_state(state_path, recs0, dqf, [{"i": 1, "kind": "mqm-minor", "detail": "x"}])

    # insert a NEW sentence at the front -> the two originals renumber (0->1, 1->2), unchanged text
    recs1 = [
        T(i=0, src="new sentence here", txt="NS"),
        T(i=1, src="alpha beta", txt="AB"),
        T(i=2, src="gamma delta", txt="GD"),
    ]
    calls = []

    def judge(prompt):
        calls.append(prompt)
        return "[0] no-error"

    (suspects, _dqf1), changed = incremental_review(
        recs1,
        glossary,
        "ja",
        judges=build_judges(judge),
        lang_name="Japanese",
        prior_state=load_qe_state(state_path),
    )
    assert changed == [0]  # ONLY the new sentence re-judged, NOT the renumbered tail
    assert len(calls) == 1 and "alpha beta" not in calls[0] and "gamma delta" not in calls[0]
    # the reused minor (originally seg1) is remapped to its CURRENT index 2
    assert any(s["kind"] == "mqm-minor" and s["i"] == 2 for s in suspects)


def test_harvest_term_pins_rejects_translating_keep_latin_name():
    # A judge sometimes flags a keep-Latin/DNT name's correct Latin form as "wrong" and proposes
    # a CJK rendering (Penny->便士). Harvesting that inverts the DNT contract and ping-pongs
    # run-to-run — the guard must drop it, while still harvesting normal term fixes.
    from types import SimpleNamespace as T

    from dev_titles.translation_qe import harvest_term_pins

    recs = [T(i=0, src="Penny helps and China too", txt="便士帮助")]
    two_judges = [
        {"i": 0, "kind": "mqm-major[claude]", "detail": "x"},
        {"i": 0, "kind": "mqm-major[codex]", "detail": "y"},
    ]
    # keep-Latin name proposed to be TRANSLATED -> rejected
    p1, _ = harvest_term_pins(
        recs,
        "zh-Hans",
        two_judges,
        classify=lambda c, s, t: {"source": "Penny", "wrong": "Penny", "correct": "便士"},
        keep_latin=["Penny"],
    )
    assert p1 == []
    # a normal (non-keep-Latin) term fix is still harvested
    p2, _ = harvest_term_pins(
        recs,
        "zh-Hans",
        two_judges,
        classify=lambda c, s, t: {"source": "China", "wrong": "China", "correct": "中国"},
        keep_latin=["Penny"],
    )
    assert p2 and p2[0]["correct"] == "中国"


def test_targeted_fix_ok():
    from dev_titles.translation_qe import _targeted_fix_ok

    crit = "「conflicts」を「競合」と誤訳し、予定の重複の意味を外している"  # cue-93 pattern
    old = "競合を管理したり 価格の急騰を管理したり"
    new_good = "予定の重複や価格の急騰などを管理します"
    assert _targeted_fix_ok(crit, old, new_good)  # 競合 removed AND 予定の重複 added
    assert not _targeted_fix_ok(crit, old, old)  # nothing changed
    assert not _targeted_fix_ok(crit, old, "何かを管理します")  # wrong gone but right not added
    assert not _targeted_fix_ok("awkward phrasing", old, new_good)  # critique names no term


def test_omission_length_ratio_gate():
    from dev_titles.qa import is_short_of_expected

    # 12-word EN sentence; a 6-char ja target is far too short -> flagged
    en = "First we have hasTVPreferredFocus and setting this to true does the thing"
    assert is_short_of_expected(en, "設定します", "ja")
    # a full-length ja translation is fine
    assert not is_short_of_expected(
        en,
        "まず hasTVPreferredFocus があり これを true に設定すると初回マウント時に動作します",
        "ja",
    )
    # short EN sentence is never gated
    assert not is_short_of_expected("Thanks for watching.", "どうも", "ja")


def test_ja_config_widened():
    from dev_titles.config import spec_for

    assert spec_for("ja").max_cpl >= 18  # was 13 (too tight -> mid-word cuts)
    assert spec_for("ja").max_cps >= 8.0


def test_coverage_gate_and_edit_fix():
    from dev_titles.translate.coverage import coverage_missing, edit_fix, lang_name, parse_missing

    assert lang_name("ja") == "Japanese" and lang_name("zh-Hans") == "Simplified Chinese"
    # parse verdict
    assert parse_missing("PRESENT: a\nPRESENT: b\nCOMPLETE") == ""
    assert (
        parse_missing("PRESENT: a\nMISSING: the leading condition\nINCOMPLETE")
        == "the leading condition"
    )
    # coverage_missing + edit_fix drive an injected invoke
    calls = {}

    def fake_invoke(prompt):
        calls["last"] = prompt
        if "Decompose the SOURCE" in prompt:
            return "PRESENT: main clause\nMISSING: unlike touch interfaces\nINCOMPLETE"
        return "タッチUIとは異なり TVはフォーカスに依存します"  # edit-fix result

    miss = coverage_missing(
        fake_invoke,
        "Unlike touch interfaces, a TV relies on focus.",
        "TVはフォーカスに依存します",
        "Japanese",
    )
    assert miss == "unlike touch interfaces"
    fixed = edit_fix(
        fake_invoke,
        "Unlike touch interfaces, a TV relies on focus.",
        "TVはフォーカスに依存します",
        miss,
        "Japanese",
    )
    assert "タッチUI" in fixed and "MISSING CONTENT" in calls["last"]


def test_gpt_mantle_judge_invoke():
    from dev_titles.eval.gpt_judge import GptMantleJudge

    class _Msg:  # fake OpenAI client
        def __init__(self, t):
            self.content = t

    class _Choice:
        def __init__(self, t):
            self.message = _Msg(t)

    class _Resp:
        def __init__(self, t):
            self.choices = [_Choice(t)]

    captured = {}

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    captured.update(kw)
                    return _Resp("  PARITY: 90  ")

    j = GptMantleJudge(client=_FakeClient(), model="openai.gpt-5.5")
    assert j.invoke("rate this") == "PARITY: 90"
    assert captured["model"] == "openai.gpt-5.5"
    assert captured["messages"][0]["content"] == "rate this"


def test_low_confidence_spans_flags_uncertain_words():
    from dev_titles.cue_builder import low_confidence_spans
    from dev_titles.model import Transcript, Word

    t = Transcript(
        "en",
        [
            Word("MCP", 0, 0.5, confidence=0.31),
            Word("server", 0.5, 1.0, confidence=0.95),
            Word("noconf", 1.0, 1.5, confidence=None),
        ],
    )
    flags = low_confidence_spans(t, threshold=0.5)
    assert (
        len(flags) == 1 and flags[0]["word"] == "MCP"
    )  # low-conf flagged, high-conf & no-conf skipped


def test_spell_small_numbers_guards():
    from dev_titles.cleanup import spell_small_numbers as s

    assert s("focus moves in 4 directions") == "focus moves in four directions"
    assert s("pick 2 or 3 options") == "pick two or three options"
    assert s("took 6 milliseconds") == "took 6 milliseconds"  # unit -> numeral
    assert s("about 6% faster") == "about 6% faster"  # percent
    assert s("upgrade to v6 today") == "upgrade to v6 today"  # version
    assert s("it was 16.67ms") == "it was 16.67ms"  # code/decimal token
    assert s("costs $5 total") == "costs $5 total"  # money
    assert s("use 11 or 20 items") == "use 11 or 20 items"  # >10 -> numeral


def test_en_consistency_suspects():
    """EN self-consistency: flag single-token inconsistent casing (iX/IX, camelCase variants);
    do NOT flag ordinary sentence-initial capitals."""
    from dev_titles.transcribe_qe import en_consistency_suspects

    s = en_consistency_suspects(
        "Fuel iX is great. Also Fuel IX ships. OpenTelemetry vs "
        "openTelemetry. The the THE cat runs."
    )
    dets = " | ".join(x["detail"] for x in s).lower()
    assert "'ix'" in dets and "opentelemetry" in dets  # inconsistent casing flagged
    assert "'the'" not in dets  # sentence-initial 'The' NOT flagged
    assert en_consistency_suspects("React Native is great. React Native rules.") == []  # consistent


def test_resolve_entity_renderings_authoritative():
    """Brand/product canonical names come from the AUTHORITATIVE source (Wikipedia
    langlinks), not an LLM guess. English name becomes an avoid-form so cues that left
    it in English get normalized. Entities with no localized article are omitted."""
    import json

    from dev_titles.vocab_prep import resolve_entity_renderings

    LANGLINKS = {
        ("Northwind Airlines", "ja"): "ノースウィンド航空",
        ("Northwind Airlines", "zh"): "北風航空",  # Wikipedia stores Traditional variant
    }

    def fake_fetch(url):
        import urllib.parse as up

        q = up.parse_qs(up.urlparse(url).query)
        ent = up.unquote(q["titles"][0])
        lang = q["lllang"][0]
        title = LANGLINKS.get((ent, lang))
        if not title:
            return json.dumps({"query": {"pages": {"1": {"title": ent}}}})  # no langlink
        return json.dumps(
            {"query": {"pages": {"1": {"title": ent, "langlinks": [{"lang": lang, "*": title}]}}}}
        )

    def fake_invoke(prompt):  # T->S conversion of the verified name
        return "北风航空" if "北風航空" in prompt else ""

    r = resolve_entity_renderings(
        ["Northwind Airlines", "Concierge"], web_fetch=fake_fetch, invoke=fake_invoke
    )
    assert r["ja"]["Northwind Airlines"]["canonical"] == "ノースウィンド航空"
    assert r["zh-Hans"]["Northwind Airlines"]["canonical"] == "北风航空"  # converted to Simplified
    assert "Northwind Airlines" in r["ja"]["Northwind Airlines"]["avoid"]  # English -> normalize
    assert "Concierge" not in r.get("ja", {})  # no article -> omitted (not guessed)


def test_en_autofix_applies_clear_holds_judgment():
    """en_autofix auto-applies mechanical fixes, holds garble/factual guesses + global findings."""
    from dev_titles.transcribe_qe import en_autofix

    en = Subtitle(
        "en",
        [
            Cue(0.0, 2.0, ["So it takes care about loading the sources."]),  # cue 0: clear grammar
            Cue(
                2.0, 4.0, ["We had a Samsung application, a Chile application."]
            ),  # cue 1: garble guess
        ],
    )
    findings = [
        {
            "ts": "00:00",
            "kind": "en-grammar",
            "detail": "[cue 0] 'takes care about' -> 'takes care of'",
        },
        {
            "ts": "00:02",
            "kind": "en-garbled",
            "detail": "[cue 1] 'a Chile application' -> 'an LG application'",
        },
        {"ts": "—", "kind": "en-inconsistency", "detail": "'iX' inconsistent casing: ['IX', 'iX']"},
    ]

    def fake_invoke(prompt):
        # clear for the grammar fix (returns corrected cue), judgment for the garble guess
        return (
            '[{"i": 0, "verdict": "clear", "new_text": "So it takes care of loading the sources."},'
            ' {"i": 1, "verdict": "judgment"}]'
        )

    new_en, applied, held = en_autofix(en, findings, fake_invoke)
    assert (
        len(applied) == 1 and applied[0]["new_text"] == "So it takes care of loading the sources."
    )
    assert new_en.cues[0].lines == ["So it takes care of loading the sources."]
    assert new_en.cues[1].lines == [
        "We had a Samsung application, a Chile application."
    ]  # untouched
    # both the garble guess and the global consistency finding are held
    assert len(held) == 2
    details = " ".join(h["detail"] for h in held)
    assert "Chile" in details and "iX" in details


def test_dnt_survival_marks_moved_vs_garbled():
    # A keep-Latin term missing from its aligned cue but present ELSEWHERE in the target merely
    # moved on re-segmentation (survived=True); one absent from the whole target is garbled (False).
    from types import SimpleNamespace as T

    from dev_titles.localize import dnt_survival_violations
    from dev_titles.model import Cue, Subtitle

    g = T(identifiers=["React Native"], terms=[])
    en = Subtitle(
        "en", [Cue(0, 2, ["We use React Native here"]), Cue(2, 4, ["React Native again"])]
    )
    moved = Subtitle("zh-Hans", [Cue(0, 2, ["我们这里使用"]), Cue(2, 4, ["React Native 再次"])])
    v = [x for x in dnt_survival_violations(en, moved, g, None, "zh-Hans") if x["cue"] == 1]
    assert v and v[0]["survived"] is True
    gone = Subtitle("zh-Hans", [Cue(0, 2, ["我们这里使用原生"]), Cue(2, 4, ["再次"])])
    assert any(
        x["survived"] is False for x in dnt_survival_violations(en, gone, g, None, "zh-Hans")
    )


def test_dropped_token_suppressed_when_moved():
    from types import SimpleNamespace as T

    from dev_titles.translation_qe import _deterministic_suspects

    g = T(terms=[], identifiers=[])
    moved = [
        T(i=0, src="set the boundaries to 100 percent", txt="设定边界"),
        T(i=1, src="okay", txt="100% 好的"),
    ]
    assert not any(
        x["kind"] == "dropped-token"
        for x in _deterministic_suspects(moved, g, "zh-Hans", coverage_ran=True)
    )
    gone = [T(i=0, src="set to 100 percent", txt="设定边界"), T(i=1, src="okay", txt="好的")]
    assert any(
        x["kind"] == "dropped-token"
        for x in _deterministic_suspects(gone, g, "zh-Hans", coverage_ran=True)
    )


def test_override_adjudicator_verdict():
    from dev_titles.translation_qe import make_override_adjudicator

    ok = make_override_adjudicator(lambda p: "VERIFIED")
    bad = make_override_adjudicator(lambda p: "REJECT — 特工 means spy")
    assert ok("It's a great promise", "伟大的承诺", "承诺", "前景") is True
    assert bad("our AI agent", "我们的特工", "智能体", "特工") is False
    assert make_override_adjudicator(None)("s", "t", "a", "b") is False  # no invoke -> held


def test_harvest_adjudicates_single_judge_override():
    from types import SimpleNamespace as T

    from dev_titles.translation_qe import harvest_term_pins

    recs = [T(i=0, src="It's a great promise", txt="这是一个伟大的承诺")]
    suspects = [{"i": 0, "kind": "mqm-major[claude]", "detail": "promise 误译为 承诺, 应为 前景"}]
    classify = lambda c, s, t: {
        "source": "promise",
        "wrong": "承诺",
        "correct": "前景",
    }  # noqa: E731
    # VERIFIED -> the single-judge override is applied
    pins, _ = harvest_term_pins(
        recs, "zh-Hans", suspects, classify=classify, keep_latin=[], adjudicate=lambda *a: True
    )
    assert pins and pins[0]["correct"] == "前景"
    # REJECT -> held (no pin)
    pins2, _ = harvest_term_pins(
        recs, "zh-Hans", suspects, classify=classify, keep_latin=[], adjudicate=lambda *a: False
    )
    assert pins2 == []
    # no adjudicator -> unchanged (held)
    pins3, _ = harvest_term_pins(recs, "zh-Hans", suspects, classify=classify, keep_latin=[])
    assert pins3 == []


def test_adjacent_duplicate_clauses():
    from dev_titles.localize import adjacent_duplicate_clauses
    from dev_titles.model import Cue, Subtitle

    dup = Subtitle(
        "ja",
        [
            Cue(0, 2, ["プロジェクトの今後がとても楽しみです"]),
            Cue(2, 4, ["プロジェクトの今後がとても楽しみです", "ありがとう"]),
        ],
    )
    d = adjacent_duplicate_clauses(dup)
    assert d and d[0]["cue"] == 2 and "楽しみ" in d[0]["dup"]
    # distinct adjacent cues -> nothing
    assert (
        adjacent_duplicate_clauses(
            Subtitle("ja", [Cue(0, 2, ["こんにちは"]), Cue(2, 4, ["さようなら"])])
        )
        == []
    )
    # short shared line (function word) below min_len -> not flagged
    short = Subtitle("zh-Hans", [Cue(0, 2, ["好的"]), Cue(2, 4, ["好的"])])
    assert adjacent_duplicate_clauses(short, min_len=6) == []


def test_invalidate_cache_surgical(tmp_path):
    """Surgical remediation: drop ONLY the matching lang records (by src/tgt substring, index, or
    predicate); keep every other cue AND every other language verbatim; never touch qe_state."""
    import json

    from dev_titles.localize import invalidate_cache

    cache = tmp_path / "cache.jsonl"
    rows = [
        {"lang": "zh-Hans", "i": 0, "src": "open App.tsx", "txt": "打开App.tsx"},
        {"lang": "zh-Hans", "i": 1, "src": "and you can do it", "txt": "你可以做到"},
        {"lang": "zh-Hans", "i": 2, "src": "the web view", "txt": "网页view"},
        {"lang": "ja", "i": 0, "src": "open App.tsx", "txt": "App.tsxを開く"},  # other lang: keep
    ]
    cache.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    # drop the zh cue whose target has the bad 'view' rendering -> only i=2
    n = invalidate_cache(str(cache), "zh-Hans", tgt_contains="view")
    left = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]
    assert n == 1
    assert {(r["lang"], r["i"]) for r in left} == {("zh-Hans", 0), ("zh-Hans", 1), ("ja", 0)}

    # by source substring + explicit index, both langs untouched except the matches
    cache.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    n = invalidate_cache(str(cache), "zh-Hans", src_contains="App.tsx", indices={1})
    left = {
        (r["lang"], r["i"])
        for r in (json.loads(l) for l in cache.read_text().splitlines() if l.strip())
    }
    assert n == 2 and left == {("zh-Hans", 2), ("ja", 0)}  # ja kept, zh 0+1 dropped


def test_looks_like_raw_caption():
    """localize must recognise a raw YouTube caption (filler / over-long lines) and clean-read it;
    a properly built EN subtitle is left alone."""
    from dev_titles.cli import _looks_like_raw_caption
    from dev_titles.model import Cue, Subtitle

    raw = Subtitle(
        "en",
        [
            Cue(
                0,
                3,
                [
                    "Um, I thought we could start, uh, today and, um, talk about React Native for desktop and TV devices"
                ],
            )
        ],
    )
    clean = Subtitle(
        "en", [Cue(0, 3, ["I thought we could start today", "and talk about React Native."])]
    )
    assert _looks_like_raw_caption(raw)
    assert not _looks_like_raw_caption(clean)


def test_condense_guard():
    """Condensation guardrails: never longer, and no Latin token or number may be lost."""
    from dev_titles.condense import guard_ok

    orig = "現在コードの90%以上がAIで生成されており本当に加速しています"
    assert guard_ok(orig, "コードの90%以上がAIで生成され加速しています")
    assert not guard_ok(orig, "コードの大半がAIで生成されています")  # lost "90"
    assert not guard_ok(orig, "現在コードの90%以上が生成されています")  # lost "AI"
    assert not guard_ok("短い", "もっと長くなった文")  # longer


def test_youtube_pick_existing_track():
    """Replace the uploaded EN track, never YouTube's auto-generated (ASR) one; no match -> insert."""
    from dev_titles.publish.youtube import YouTubePublisher as P

    tracks = [
        {"id": "asr1", "language": "en", "trackKind": "asr"},
        {
            "id": "up1",
            "language": "en",
            "trackKind": "standard",
            "isDraft": False,
            "name": "English",
        },
        {"id": "ja1", "language": "ja", "trackKind": "standard"},
    ]
    assert P.pick_existing(tracks, "en")["id"] == "up1"
    assert P.pick_existing([tracks[0]], "en") is None  # only ASR -> insert, don't touch ASR
    assert P.pick_existing(tracks, "zh-Hans") is None
    assert (
        P.pick_existing([{"id": "x", "language": "en-US", "trackKind": "standard"}], "en")["id"]
        == "x"
    )


def test_condense_lang_idempotent(tmp_path):
    """Production condensation must not re-condense on a re-run (cache reuses condensed text)."""
    import json

    from dev_titles.condense import condense_lang
    from dev_titles.localize import SentenceRecord

    calls = {"n": 0}

    def invoke(prompt):
        calls["n"] += 1
        return json.dumps(["一見良さそうに見えました"])

    recs = [
        SentenceRecord(
            "ja",
            0,
            "It looked really good at first.",
            "一見するととても良さそうに見えました",
            "",
            [],
        )
    ]
    sp = tmp_path / "condense_state.ja.json"
    kw = dict(coverage_missing=lambda *a: "", normalize=lambda t, s=None: t, logger=lambda m: None)
    assert condense_lang(recs, invoke, "ja", "Japanese", sp, **kw) == {0}
    assert recs[0].txt == "一見良さそうに見えました"
    n = calls["n"]
    assert condense_lang(recs, invoke, "ja", "Japanese", sp, **kw) == set()  # second run: skipped
    assert calls["n"] == n and recs[0].txt == "一見良さそうに見えました"


def test_condense_reading_speed_gate(tmp_path):
    """Only lines over their reading-speed budget are condensed; within-budget lines (e.g. slow
    tutorial narration) are never even sent to the model. Over-budget lines carry their target."""
    import json

    from dev_titles.condense import condense_lang
    from dev_titles.localize import SentenceRecord

    prompts = []

    def invoke(p):
        prompts.append(p)
        return json.dumps(["これは長すぎる行を短くした版です"])

    slow = SentenceRecord("ja", 0, "Click Save.", "保存をクリックします", "", [])
    dense = SentenceRecord(
        "ja",
        1,
        "A long line.",
        "これは画面に表示される時間に対して明らかに長すぎる非常に長い行です",
        "",
        [],
    )
    kw = dict(coverage_missing=lambda *a: "", normalize=lambda t, s=None: t, logger=lambda m: None)
    changed = condense_lang(
        [slow, dense], invoke, "ja", "Japanese", tmp_path / "s.json", budgets={0: 40, 1: 20}, **kw
    )
    assert changed == {1} and slow.txt == "保存をクリックします"
    assert (
        len(prompts) == 1
        and "保存をクリックします" not in prompts[0]
        and "BUDGET: <= 20" in prompts[0]
    )


def test_strip_sound_tags_and_entities(tmp_path):
    """Imported captions: drop non-speech sound tags (keep real parentheticals) and decode HTML
    entities, so neither reaches translation ('[music]' -> ja '[音楽]', '&nbsp;' in the EN)."""
    from dev_titles.cue_builder import strip_sound_tags
    from dev_titles.formats import read_vtt
    from dev_titles.model import Cue, Subtitle

    s = Subtitle(
        "en",
        [
            Cue(0, 2, ["[music] Hi, I'm Priya"]),
            Cue(2, 4, ["Sign in with Acme (SIWA)"]),
            Cue(4, 6, ["(applause)"]),
        ],
    )
    assert [c.lines for c in strip_sound_tags(s).cues] == [
        ["Hi, I'm Priya"],
        ["Sign in with Acme (SIWA)"],
    ]
    f = tmp_path / "x.vtt"
    f.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\ngoing to be&nbsp; talking Q&amp;A\n",
        encoding="utf-8",
    )
    assert read_vtt(str(f)).cues[0].lines == ["going to be talking Q&A"]


def test_condense_trigger_and_original_log(tmp_path):
    """Lines only slightly over budget (below the trigger) are left alone; condensed lines log
    their original so a QE-flagged edit can be reverted; a reverted line is not re-condensed."""
    import json

    from dev_titles.condense import condense_lang, load_originals, mark_processed
    from dev_titles.localize import SentenceRecord

    def invoke(p):
        return json.dumps(["短くした版です"])

    near = SentenceRecord("ja", 0, "A.", "少しだけ長い行です", "", [])  # 9 chars
    far = SentenceRecord("ja", 1, "B.", "これは明らかに長すぎる非常に長い行です", "", [])
    orig_far = far.txt
    sp = tmp_path / "condense_state.ja.json"
    kw = dict(coverage_missing=lambda *a: "", normalize=lambda t, s=None: t, logger=lambda m: None)
    changed = condense_lang(
        [near, far],
        invoke,
        "ja",
        "Japanese",
        sp,
        budgets={0: 8, 1: 8},
        triggers={0: 10, 1: 10},
        **kw,
    )
    assert changed == {1} and near.txt == "少しだけ長い行です"
    assert load_originals(sp, [near, far]) == {1: orig_far}
    far.txt = orig_far  # QE revert
    mark_processed(sp, [far])
    assert load_originals(sp, [far]) == {}  # current text is no longer the condensed edit
    calls = []
    assert (
        condense_lang(
            [far],
            lambda p: calls.append(p) or "[]",
            "ja",
            "Japanese",
            sp,
            budgets={1: 8},
            triggers={1: 10},
            **kw,
        )
        == set()
    )
    assert not calls  # reverted line stays reverted


def test_vocab_fragments_and_consistent_longer_terms():
    """Per-video renderings: a word INSIDE a curated multi-word name is never its own term
    (Orbit->轨道 with avoid 'Orbit' turned 'My Orbit TV' into 'My 轨道 TV'); a longer term whose rendering
    is CONSISTENT with a contained curated word is kept (Developer Options->开发者选项), else the
    soft one-word term half-translates it (开发者Options)."""
    from dev_titles.glossary import Glossary, Term
    from dev_titles.localize import effective_glossary

    g = Glossary(
        "1",
        ["zh-Hans"],
        terms=[
            Term(source="Orbit TV", translations={"zh-Hans": "Orbit TV"}, enforce="hard"),
            Term(source="Developer", translations={"zh-Hans": "开发者"}, enforce="soft"),
        ],
    )
    vocab = {
        "renderings": {
            "zh-Hans": {
                "Orbit": {"canonical": "轨道", "avoid": ["Orbit"]},
                "Developer Options": {"canonical": "开发者选项", "avoid": []},
            }
        }
    }
    eg, _ = effective_glossary(g, vocab, lang="zh-Hans")
    srcs = {t.source for t in eg.terms}
    assert "Orbit" not in srcs and "Developer Options" in srcs
    out, _ = eg.enforce(
        "回到My Orbit TV，会出现Developer Options",
        "Return to My Orbit TV and Developer Options appears.",
        "zh-Hans",
    )
    assert out == "回到My Orbit TV，会出现开发者选项"


def test_enforce_resolves_caller_placeholders():
    """The translate pipeline enforces while its own DNT masks are still in the text; a masked
    DNT inside a canonical ([[0]]虚拟设备 for Nimbus虚拟设备) must not be flagged missing."""
    from dev_titles.glossary import Glossary, Term

    g = Glossary(
        "1",
        ["zh-Hans"],
        terms=[
            Term(
                source="Nimbus Virtual Device",
                translations={"zh-Hans": "Nimbus虚拟设备"},
                enforce="hard",
            )
        ],
    )
    _, iss = g.enforce(
        "启动[[0]]虚拟设备",
        "Start Nimbus Virtual Device.",
        "zh-Hans",
        placeholders={"[[0]]": "Nimbus"},
    )
    assert not [i for i in iss if i.kind == "missing"]


def test_leftover_english_only_whole_latin_run():
    """'TV' inside 'Multi-TV' / 'Orbit TV' is part of a kept-Latin name, not leftover English."""
    from dev_titles.glossary import _replace_latin_run
    from dev_titles.localize import stale_untranslated_reuse

    assert (
        _replace_latin_run("在Hello World Multi-TV中", "TV", "电视") == "在Hello World Multi-TV中"
    )
    assert _replace_latin_run("打开Orbit TV", "TV", "电视") == "打开Orbit TV"
    assert _replace_latin_run("看TV吧", "TV", "电视") == "看电视吧"
    from dev_titles.glossary import Glossary, Term

    g = Glossary("1", ["zh-Hans"], terms=[Term(source="TV", translations={"zh-Hans": "电视"})])
    assert not stale_untranslated_reuse(
        "在Orbit TV Stick上验证", "validate on an Orbit TV Stick", g, "zh-Hans"
    )
    assert stale_untranslated_reuse("在TV上观看", "watch on TV", g, "zh-Hans")


def test_term_pin_guards_curated_and_common_phrase():
    """A pin must not override a curated hard rendering (ja Nimbus仮想デバイス -> Latin flipped every
    cue), nor pin a recurring plain English phrase Latin video-wide (home screen -> 在home screen上);
    a one-off lowercase CLI phrase still pins."""
    from dev_titles.localize import SentenceRecord
    from dev_titles.translation_qe import harvest_term_pins

    recs = [
        SentenceRecord("ja", 0, "Start Nimbus Virtual Device.", "Nimbus仮想デバイスを起動", "", []),
        SentenceRecord("ja", 1, "Open home screen dot tsx.", "ホーム画面を開く", "", []),
        SentenceRecord(
            "ja", 2, "From the home screen, go to Settings.", "ホーム画面から設定へ", "", []
        ),
        SentenceRecord("ja", 3, "Run nimbus device list.", "Nimbus デバイス listを実行", "", []),
    ]
    fixes = {
        0: {
            "source": "Nimbus Virtual Device",
            "wrong": "Nimbus仮想デバイス",
            "correct": "Nimbus Virtual Device",
        },
        1: {"source": "home screen", "wrong": "ホーム画面", "correct": "home screen"},
        3: {
            "source": "nimbus device list",
            "wrong": "Nimbus デバイス list",
            "correct": "nimbus device list",
        },
    }
    sus = [{"kind": "mqm-major", "i": i, "detail": "x"} for i in fixes]
    pins, _ = harvest_term_pins(
        recs,
        "ja",
        sus,
        classify=lambda c, s, t: next(f for i, f in fixes.items() if recs[i].src == s),
        curated={"Nimbus Virtual Device": "Nimbus仮想デバイス"},
    )
    assert [p["source"] for p in pins] == ["nimbus device list"]


def test_zh_interior_period_becomes_comma_and_no_line_end_comma():
    """Interior 。 must not weld two sentences (开发者模式为此请…): it becomes the vendor's sentence-break space; CN SG: no line-final ，."""
    from dev_titles.config import spec_for
    from dev_titles.segment.cjk import normalize_zh_punct, wrap_lines

    assert (
        normalize_zh_punct("首先启用开发者模式。为此请打开终端。")
        == "首先启用开发者模式 为此请打开终端"
    )
    assert normalize_zh_punct("“Orbit TV才刚刚起步。”") == "“Orbit TV才刚刚起步”"
    lines = wrap_lines(
        "这是一个比较长的句子，需要在逗号处换行显示，然后继续说明后面的内容", spec_for("zh-Hans")
    )
    assert len(lines) > 1 and not any(ln.endswith("，") for ln in lines)


def test_vocab_latin_name_not_respelled():
    """A Latin name is never re-spelled as other Latin (VS Code -> Visual Studio Code with avoid
    'VS Code' rewrote the speaker's words); a self-avoid (Ubuntu avoid Ubuntu) is dropped."""
    from dev_titles.vocab_prep import renderings_to_terms

    terms = renderings_to_terms(
        {
            "zh-Hans": {
                "VS Code": {"canonical": "Visual Studio Code", "avoid": ["VS Code"]},
                "Ubuntu": {"canonical": "Ubuntu", "avoid": ["Ubuntu"]},
            }
        }
    )
    assert not terms


def test_vocab_terms_need_standalone_use_and_skip_curated():
    """Words that only occur inside a longer name (World in Hello World) or sentence-initially
    (Press, Choose) are not terms; curated names, their words and sub-phrases get no per-video
    rendering, and single words never go to the Wikipedia entity lookup (Nimbus -> the cloud)."""
    import json

    from dev_titles.vocab_prep import build_video_vocab, extract_lookup_terms

    txt = (
        "Open Nimbus Studio and build Hello World. Press Enter. Choose a folder. "
        "Then run Hello World in Nimbus Studio on the Nimbus Virtual device with Perfetto traces."
    )
    terms = extract_lookup_terms(txt)
    assert "World" not in terms and "Press" not in terms and "Choose" not in terms
    assert "Perfetto" in terms
    seen = []

    def translate(prompt):
        seen.append(prompt)
        return json.dumps({})

    fetched = []
    build_video_vocab(
        txt,
        lambda q: "",
        web_fetch=lambda u: fetched.append(u) or "",
        translate=translate,
        known=["Nimbus Studio", "Hello World", "Nimbus Virtual Device", "Nimbus"],
    )
    body = "\n".join(seen)
    assert (
        "- Nimbus Studio" not in body
        and "- Nimbus Virtual" not in body
        and "- Hello World" not in body
    )
    assert not any("Perfetto" in u for u in fetched)  # single word: no Wikipedia sense lookup


def test_zh_han_space_han_is_a_clause_break():
    """A space between two Han chars is a sentence break (vendor convention); deleting it welded them."""
    from dev_titles.segment.cjk import normalize_zh_punct

    assert (
        normalize_zh_punct("我将在Mac上演示 不过如果您使用Ubuntu")
        == "我将在Mac上演示 不过如果您使用Ubuntu"
    )
    assert normalize_zh_punct("使用 Nimbus Studio 扩展") == "使用Nimbus Studio扩展"


def test_legacy_single_word_entity_pin_ignored():
    from dev_titles.vocab_prep import renderings_to_terms

    t = renderings_to_terms(
        {
            "ja": {
                "Rally": {"canonical": "ラリー", "avoid": ["Rally"]},
                "Northwind Airlines": {
                    "canonical": "ノースウィンド航空",
                    "avoid": ["Northwind Airlines"],
                },
            }
        }
    )
    assert [x.source for x in t] == ["Northwind Airlines"]


def test_zh_latin_list_separator_kept_and_no_line_start_comma():
    from dev_titles.config import spec_for
    from dev_titles.segment.cjk import normalize_zh_punct, wrap_lines

    assert (
        normalize_zh_punct("包括Nimbus、Orbit OS、Expo、Android")
        == "包括Nimbus、Orbit OS、Expo、Android"
    )
    assert normalize_zh_punct("运行Nimbus OS。Expo也支持") == "运行Nimbus OS Expo也支持"
    for ln in wrap_lines(
        "现在单击Nimbus图标，然后单击“创建Nimbus项目”，接着单击Hello World，再为项目选择一个目录",
        spec_for("zh-Hans"),
    ):
        assert not ln.startswith("，") and not ln.endswith("，")


def test_native_editor_accepts_only_guarded_and_judged_rewrites(tmp_path):
    """Editor rewrites land only if Latin tokens/numbers survive AND the pairwise judge prefers them
    with the same meaning; the pass is idempotent and logs before/after."""
    import json

    from dev_titles.editor import native_edit
    from dev_titles.localize import SentenceRecord

    recs = [
        SentenceRecord(
            "ja",
            0,
            "If nothing happens, nothing happens.",
            "何も起きなければ何も起きないだけです",
            "",
            [],
        ),
        SentenceRecord("ja", 1, "We ship 3 Acme builds.", "Acmeビルドを3つ出荷します", "", []),
    ]
    calls = {"edit": 0}

    def invoke(p):
        calls["edit"] += 1
        return json.dumps(
            [
                {
                    "id": 1,
                    "issue": "literal tautology",
                    "rewrite": "入力がなければ何も起こりません",
                },
                {
                    "id": 2,
                    "issue": "x",
                    "rewrite": "ビルドを出荷します",
                },  # drops Acme + 3 -> guard rejects
            ]
        )

    judge = lambda p: json.dumps(
        {"same_meaning": True, "better": "B" if "B: 入力" in p else "A", "why": ""}
    )
    sp = tmp_path / "editor_state.ja.json"
    changed = native_edit(
        recs,
        invoke,
        "ja",
        "Japanese",
        sp,
        normalize=lambda t, s=None: t,
        pair_judge=judge,
        logger=lambda m: None,
        seed=1,
    )
    assert changed == {0} and recs[0].txt == "入力がなければ何も起こりません"
    assert recs[1].txt == "Acmeビルドを3つ出荷します"
    n = calls["edit"]
    assert (
        native_edit(
            recs,
            invoke,
            "ja",
            "Japanese",
            sp,
            normalize=lambda t, s=None: t,
            pair_judge=judge,
            logger=lambda m: None,
        )
        == set()
    )
    assert calls["edit"] == n  # idempotent: nothing re-reviewed
    log = json.loads((tmp_path / "editor_log.ja.json").read_text())
    assert log[0]["before"] == "何も起きなければ何も起きないだけです"


def test_native_editor_rating_veto(tmp_path):
    """A rewrite the pairwise judge prefers is still rejected if a judge rates it worse than the
    current line (reference-free); the rejection is logged for the reviewer."""
    import json
    import re

    from dev_titles.editor import native_edit
    from dev_titles.localize import SentenceRecord

    recs = [SentenceRecord("ja", 0, "It's a great promise.", "素晴らしい将来性です", "", [])]

    def rate(p):  # current line OK, suggestion MEANING
        items = re.findall(r"(\d+)\. EN: .*\n   SUBTITLE: (.*)", p)
        return json.dumps(
            [{"id": int(n), "level": "MEANING" if "理念" in t else "OK"} for n, t in items]
        )

    def invoke(p):
        if "SUBTITLE:" in p:
            return rate(p)
        return json.dumps([{"id": 1, "issue": "x", "rewrite": "素晴らしい理念です"}])

    judge = lambda p: (
        rate(p)
        if "SUBTITLE:" in p
        else json.dumps({"same_meaning": True, "better": "B" if "B: 素晴らしい理念" in p else "A"})
    )
    changed = native_edit(
        recs,
        invoke,
        "ja",
        "Japanese",
        tmp_path / "editor_state.ja.json",
        normalize=lambda t, s=None: t,
        pair_judge=judge,
        logger=lambda m: None,
    )
    assert changed == set() and recs[0].txt == "素晴らしい将来性です"
    log = json.loads((tmp_path / "editor_log.ja.json").read_text())
    assert log[0]["status"] == "rejected" and "rated it worse" in log[0]["why"]


def test_native_editor_rescues_rewrite_every_judge_rates_better(tmp_path):
    """A rewrite the pairwise judge rejects (e.g. 'meaning differs') is still applied when every
    judge rates it better than the current line."""
    import json
    import re

    from dev_titles.editor import native_edit
    from dev_titles.localize import SentenceRecord

    recs = [SentenceRecord("ja", 0, "We had an aha moment.", "アハモーメントがありました", "", [])]

    def rate(p):
        items = re.findall(r"(\d+)\. EN: .*\n   SUBTITLE: (.*)", p)
        return json.dumps(
            [{"id": int(n), "level": "OK" if "気づき" in t else "NOTICEABLE"} for n, t in items]
        )

    def invoke(p):
        if "SUBTITLE:" in p:
            return rate(p)
        if '"same_meaning"' in p:
            return json.dumps({"same_meaning": False, "better": "A"})
        return json.dumps([{"id": 1, "issue": "calque", "rewrite": "大きな気づきがありました"}])

    changed = native_edit(
        recs,
        invoke,
        "ja",
        "Japanese",
        tmp_path / "editor_state.ja.json",
        normalize=lambda t, s=None: t,
        pair_judge=invoke,
        logger=lambda m: None,
    )
    assert changed == {0} and recs[0].txt == "大きな気づきがありました"
    log = json.loads((tmp_path / "editor_log.ja.json").read_text())
    assert log[0]["status"] == "accepted" and "rescued" in log[0]["why"]


def test_editor_quote_fix_and_zh_list_separators():
    from dev_titles.editor import fix_unbalanced_quotes
    from dev_titles.localize import zh_list_separators

    assert (
        fix_unbalanced_quotes("保留异议，全力执行”，我们就经常这么做")
        == "保留异议，全力执行，我们就经常这么做"
    )
    assert fix_unbalanced_quotes("“保留异议”，全力执行") == "“保留异议”，全力执行"
    src = "You can learn from the JavaScript world, TypeScript world, ReScript, Java, Kotlin, iOS, C++."
    assert zh_list_separators(
        "从JavaScript TypeScript ReScript Java Kotlin iOS和C++的世界学习", src
    ) == ("从JavaScript、TypeScript、ReScript、Java、Kotlin、iOS和C++的世界学习")
    assert (
        zh_list_separators("使用React Native开发", "We use React Native, Expo, and Hermes.")
        == "使用React Native开发"
    )


def test_video_brief_uses_description_and_whole_transcript(tmp_path):
    from dev_titles.editor import video_brief, video_description

    (tmp_path / "description.txt").write_text(
        "Jane Roe, Senior Developer Advocate at Acme, interviews Bo."
    )
    assert video_description(tmp_path).startswith("Jane Roe")
    prompts = []

    def inv(p):
        prompts.append(p)
        return "brief"

    text = "word " * 6000  # 30k chars -> chunked notes, then the brief
    b = video_brief(
        text, inv, tmp_path / "video_brief.txt", description=video_description(tmp_path)
    )
    assert b == "brief" and len(prompts) == 4  # 3 chunk notes + 1 brief
    assert "Senior Developer Advocate" in prompts[-1]
    n = len(prompts)
    assert (
        video_brief(
            text, inv, tmp_path / "video_brief.txt", description=video_description(tmp_path)
        )
        == "brief"
    )
    assert len(prompts) == n  # cached
    assert video_description(tmp_path / "missing") == ""


def test_qe_repair_protects_editor_rewrites_and_runs_accept_check():
    """A native-editor rewrite is only repaired when BOTH judges flag a major; and any repair must
    pass accept_check (blocks literal-sense 'fixes' like 可能性 -> 約束)."""
    from dev_titles.localize import SentenceRecord
    from dev_titles.translation_qe import qe_repair

    recs = [
        SentenceRecord(
            "ja",
            0,
            "Pick the lowest common denominator.",
            "最も性能の低いデバイスに合わせる",
            "",
            [],
        ),
        SentenceRecord("ja", 1, "It's a great promise.", "素晴らしい可能性です", "", []),
    ]
    per_seg = {
        0: {
            "claude": {"critical": 0, "major": 1, "minor": 0},
            "codex": {"critical": 0, "major": 0, "minor": 0},
        },
        1: {
            "claude": {"critical": 0, "major": 1, "minor": 0},
            "codex": {"critical": 0, "major": 1, "minor": 0},
        },
    }
    sus = [
        {"kind": "mqm-major", "i": 0, "detail": "narrowed"},
        {"kind": "mqm-major", "i": 1, "detail": "promise"},
    ]
    tried = []

    def retrans(rec, crit):
        tried.append(rec.i)
        return {0: "共通の最低基準に合わせる", 1: "素晴らしい約束です"}[rec.i]

    fixed = qe_repair(
        recs,
        "ja",
        judges=[("claude", lambda p: "[]")],
        lang_name="Japanese",
        retranslate=retrans,
        per_seg=per_seg,
        judge_suspects=sus,
        protected={0},
        accept_check=lambda rec, old, new: False,
    )
    assert 0 not in tried  # protected: only one judge flagged it
    assert (
        fixed == [] and recs[1].txt == "素晴らしい可能性です"
    )  # accept_check rejected the literal sense
