"""Split from test_core: translate tests (9)."""

from dev_titles.model import Cue, Subtitle


def test_extract_final_translation():
    from dev_titles.translate.prompts import extract_final_translation

    assert (
        extract_final_translation("<final>将您的应用提交至艾克米应用商店</final>")
        == "将您的应用提交至艾克米应用商店"
    )
    # the exact leak seen in the live run (critique + **Final:** + answer)
    leaked = (
        "**Critique:**\n\n1. **ACCURACY** — fine.\n\n**Final:**\n\n"
        "将您的应用提交至艾克米应用商店"
    )
    assert extract_final_translation(leaked) == "将您的应用提交至艾克米应用商店"
    assert extract_final_translation("アプリを申請する") == "アプリを申請する"


def test_translate_subtitle_resilient_to_segment_failure(monkeypatch):
    """A single segment's LLM failure must NOT abort the run — keep EN + flag the residual
    (after the retry-with-backoff is exhausted). Backoff zeroed so the test doesn't sleep."""
    import dev_titles.localize as _loc

    monkeypatch.setattr(_loc, "_SEGMENT_RETRY_BACKOFF", (0, 0))
    from dev_titles.glossary import load as load_glossary
    from dev_titles.localize import translate_subtitle
    from dev_titles.translate.base import Translator

    class Flaky(Translator):
        def translate(self, req):
            if "boom" in req.source_text:
                raise TimeoutError("hung CLI")
            return "译文内容"

        def reflect_and_refine(self, req, d):
            return d

        def _invoke(self, p):
            return "COMPLETE"

    en = Subtitle(
        "en",
        [
            Cue(0.0, 2.0, ["Hello there friend."]),
            Cue(2.0, 4.0, ["This is the boom segment."]),
            Cue(4.0, 6.0, ["Another line is fine."]),
        ],
    )
    sub, issues, recs = translate_subtitle(
        en, Flaky(), load_glossary("glossary/glossary.yaml"), "zh-Hans", backcheck=False
    )
    assert len(recs) == 3  # run completed, all segments processed
    boom = [r for r in recs if "boom" in r.src.lower()][0]
    assert "boom" in boom.txt  # failed segment kept EN passthrough
    assert "translation failed" in boom.residual_miss  # flagged for the operator
    assert any("译文" in r.txt for r in recs)  # the good segments translated


def test_translation_memory_core(tmp_path):
    from dev_titles.tm import TranslationMemory

    tm = TranslationMemory()
    tm.add("Welcome to the show.", "ようこそ番組へ", "ja", "vidA")
    tm.add("Building apps for TV", "TVアプリの構築", "ja", "vidB")
    # exact match is whitespace/case-normalized
    assert tm.exact("  welcome to THE show. ", "ja") == "ようこそ番組へ"
    assert tm.exact("unknown line", "ja") is None
    # held-out exclusion drops a video's own entries
    assert (
        len(tm.exclude("vidA")) == 1
        and tm.exclude("vidA").exact("Welcome to the show.", "ja") is None
    )
    # fuzzy retrieval via the ExemplarBank (query differs from the entry so it isn't
    # skipped by the no-answer-leak guard)
    assert any("TV" in s for s, _ in tm.bank("ja").select("Building an app for TV screens", k=2))
    # save/load round-trip + TMX export
    p = tmp_path / "tm.jsonl"
    tm.save(p)
    tm2 = TranslationMemory.load(p)
    assert len(tm2) == 2 and tm2.exact("Welcome to the show.", "ja") == "ようこそ番組へ"
    px = tmp_path / "tm.tmx"
    tm.export_tmx(px)
    x = px.read_text(encoding="utf-8")
    assert "<tmx" in x and "ようこそ番組へ" in x and 'xml:lang="ja"' in x


def test_translate_subtitle_tm_exact_reuse():
    from dev_titles.glossary import load as load_glossary
    from dev_titles.localize import translate_subtitle
    from dev_titles.tm import TranslationMemory
    from dev_titles.translate.base import Translator

    class Boom(Translator):
        def translate(self, req):
            raise AssertionError("TM exact-match must skip the translator")

        def reflect_and_refine(self, req, d):
            return d

        def _invoke(self, p):
            return "COMPLETE"

    tm = TranslationMemory()
    tm.add("Hello and welcome.", "大家好，欢迎观看", "zh-Hans", "v")
    en = Subtitle("en", [Cue(0.0, 2.0, ["Hello and welcome."])])
    sub, issues, recs = translate_subtitle(
        en, Boom(), load_glossary("glossary/glossary.yaml"), "zh-Hans", tm=tm, backcheck=False
    )
    joined = "".join(l for c in sub.cues for l in c.lines)
    assert "大家好" in joined  # reused verbatim, translator never called


def test_conversational_register_override():
    from dev_titles.glossary import load as load_glossary
    from dev_titles.localize import effective_glossary

    g = load_glossary("glossary/glossary.yaml")
    formal, _ = effective_glossary(g, None, conversational=False)
    conv, _ = effective_glossary(g, None, conversational=True)
    assert "您 consistently" in formal.style["zh-Hans"]["notes"]  # formal keeps the 您 rule
    conv_notes = conv.style["zh-Hans"]["notes"]
    # Native-reviewer verdict: conversational is ALSO 您 (plural 你们 excepted), so the
    # prepended conversational directive now states the 您 default rather than overriding to 你.
    assert "您 as the default singular second person" in conv_notes
    assert "你们" in conv_notes  # plural exception documented
    assert "polite 您" in conv.style["zh-Hans"]["register"]
    # must NOT mutate the shared glossary
    assert g.style["zh-Hans"]["notes"].startswith("REGISTER: formal")
    # ja register is unchanged (teineigo suits peers on camera)
    assert conv.style["ja"]["register"] == g.style["ja"]["register"]


def test_corpus_exemplar_banks(tmp_path):
    from dev_titles.corpus import (
        build_exemplar_banks,
        discover_channel_corpus,
        video_id_from_input,
    )

    # two fake channel videos with EN + ja/zh human captions
    for vid, ja in [("aaaaaaaaaaa", "こんにちは"), ("bbbbbbbbbbb", "ようこそ")]:
        d = tmp_path / vid
        d.mkdir()
        (d / f"{vid}.en.srt").write_text(
            "1\n00:00:00,000 --> 00:00:02,000\nHello there.\n", encoding="utf-8"
        )
        (d / f"{vid}.ja.srt").write_text(
            f"1\n00:00:00,000 --> 00:00:02,000\n{ja}\n", encoding="utf-8"
        )
        (d / f"{vid}.zh-Hans.srt").write_text(
            "1\n00:00:00,000 --> 00:00:02,000\n你好\n", encoding="utf-8"
        )
    assert len(discover_channel_corpus(tmp_path)) == 2
    banks = build_exemplar_banks(["ja", "zh-Hans"], root=tmp_path)
    assert banks["ja"].exemplars and banks["zh-Hans"].exemplars
    # held-out: excluding a video drops its pairs
    only_b = build_exemplar_banks(["ja"], root=tmp_path, exclude_video="aaaaaaaaaaa")
    assert all(e.video == "bbbbbbbbbbb" for e in only_b["ja"].exemplars)
    # id extraction: URL id, then file-stem lang-suffix stripping
    assert video_id_from_input("https://www.youtube.com/watch?v=R4dZcyfunEA") == "R4dZcyfunEA"
    assert video_id_from_input("/x/foo.ja.srt") == "foo"


def test_translate_prompt_includes_char_budget():
    from dev_titles.translate import TranslationRequest, build_translate_prompt

    req = TranslationRequest("Set up in-app purchasing", "en", "ja", char_budget=18)
    prompt = build_translate_prompt(req)
    assert "18 characters" in prompt
    # condense filler, keep facts: budget allows trimming filler but protects content
    assert "substantive content" in prompt and "never facts" in prompt
    # no budget -> no budget clause
    assert "characters)" not in build_translate_prompt(TranslationRequest("x", "en", "ja"))


def test_subtitle_from_records_preserves_conversational_register():
    """The post-repair rebuild must apply the same normalizer as the main path. Since the
    native-reviewer verdict the house register is 您 in BOTH modes, so a 你 in the record is
    folded to 您 on the delivered file (plural 你们 excepted)."""
    from dev_titles.glossary import load as load_glossary
    from dev_titles.localize import SentenceRecord, _subtitle_from_records

    en = Subtitle("en", [Cue(0.0, 3.0, ["You can use your account."])])
    recs = [SentenceRecord("zh-Hans", 0, "You can use your account.", "你可以使用你的账户", "", [])]
    g = load_glossary("glossary/glossary.yaml")
    body = "".join(
        l
        for c in _subtitle_from_records(en, recs, g, "zh-Hans", None, conversational=True).cues
        for l in c.lines
    )
    assert "您" in body and "你" not in body  # folded to the polite house register


def test_cache_reuse_folds_conversational_register(tmp_path):
    """Regression: the cache-reuse branch must run the normalizer (it used to skip it, so a
    reused segment kept whatever register it was cached with). The house register
    is 您 in both modes, so a cached 你 must come back as 您 on reuse."""
    import json

    from dev_titles.glossary import load as load_glossary
    from dev_titles.localize import translate_subtitle
    from dev_titles.translate.base import Translator

    class Boom(Translator):
        def translate(self, req):
            raise AssertionError("cache reuse must skip the translator")

        def reflect_and_refine(self, req, d):
            return d

        def _invoke(self, p):
            return "COMPLETE"

    cache = tmp_path / "cache.jsonl"
    cache.write_text(
        json.dumps(
            {
                "lang": "zh-Hans",
                "i": 0,
                "src": "You can do this.",
                "txt": "您可以做到",
                "residual_miss": "",
                "issues": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    en = Subtitle("en", [Cue(0.0, 2.0, ["You can do this."])])
    sub, _iss, _recs = translate_subtitle(
        en,
        Boom(),
        load_glossary("glossary/glossary.yaml"),
        "zh-Hans",
        cache_path=str(cache),
        conversational=True,
        backcheck=False,
    )
    body = "".join(line for c in sub.cues for line in c.lines)
    assert (
        "您" in body and "你" not in body
    )  # reused translation folded to the polite house register


def test_pin_substr_fix_hyphenated_and_source_guard():
    """A hyphenated avoid-form / raw multiword source term the word-boundary glossary enforce
    MISSES (reflow-bypasses-term-pins) is substituted; a common single word that is a substring
    of another term is NOT clobbered (distinctive-source guard)."""
    from dev_titles.glossary import load as load_glossary
    from dev_titles.localize import SentenceRecord
    from dev_titles.translation_qe import apply_term_pins, pin_substr_fix

    g = load_glossary("glossary/glossary.yaml")
    pins = [
        {
            "source": "multi-eight-figure",
            "wrong": "multi-eight-figure",
            "correct": "数千万ドル規模",
            "cues": [0],
        }
    ]
    # raw hyphenated source term replaced on final text
    assert (
        pin_substr_fix("ビジネスにとって multi-eight-figure の影響", pins, "ja")
        == "ビジネスにとって 数千万ドル規模 の影響"
    )
    # distinctive-source guard: a single common word (no hyphen/space) is NOT replaced as raw
    pins2 = [{"source": "Digital", "wrong": "数码", "correct": "数字化"}]
    assert pin_substr_fix("Acme Digital 团队", pins2, "zh-Hans") == "Acme Digital 团队"
    assert (
        pin_substr_fix("这是数码产品", pins2, "zh-Hans") == "这是数字化产品"
    )  # wrong-form still fixed
    # apply_term_pins now fixes a hyphenated term the word-boundary enforce missed
    recs = [
        SentenceRecord("ja", 0, "an multi-eight-figure impact", "multi-eight-figure の影響", "", [])
    ]
    changed = apply_term_pins(recs, pins, g, "ja")
    assert (
        changed == {0}
        and "multi-eight-figure" not in recs[0].txt
        and "数千万ドル規模" in recs[0].txt
    )


def _brand_glossary():
    """Fictional brand glossary: the glossary TRANSLATES 'Appstore' (an alias) in zh."""
    from dev_titles.glossary import Glossary, Term

    return Glossary(
        "0",
        ["zh-Hans", "ja"],
        [
            Term("Acme", {"zh-Hans": "艾克米", "ja": "Acme"}, enforce="hard"),
            Term(
                "Acme Appstore",
                {"zh-Hans": "艾克米应用商店", "ja": "Acmeアプリストア"},
                aliases=["Appstore"],
                enforce="hard",
            ),
            Term(
                "Orbit Appstore",
                {"zh-Hans": "Orbit应用商店", "ja": "Orbitアプリストア"},
                enforce="hard",
            ),
            Term("app", {"zh-Hans": "应用", "ja": "アプリ"}, enforce="hard"),
            Term("submit", {"zh-Hans": "提交", "ja": "申請"}, enforce="hard"),
        ],
    )


def test_keep_latin_defers_to_glossary_translation():
    # A per-video vocab term/pin must NOT keep a term Latin when the glossary translates it:
    # vocab had 'Orbit Appstore' + an 'Appstore'->Latin pin, which masked it away from its
    # approved 应用商店 rendering. Glossary translation wins.
    from dev_titles.localize import keep_latin_terms

    g = _brand_glossary()
    vocab = {
        "terms": ["Orbit Appstore", "Perfetto"],
        "identifiers": [],
        "names": [],
        "renderings": {
            "zh-Hans": {"Appstore": {"canonical": "Appstore", "avoid": ["艾克米应用商店"]}}
        },
    }
    kl = keep_latin_terms(g, vocab, "zh-Hans")
    assert "Orbit Appstore" not in kl  # the glossary translates it -> not keep-Latin
    assert "Appstore" not in kl  # glossary alias is translated -> the Latin pin can't survive
    assert "Perfetto" in kl  # a genuine proper noun with no glossary translation stays Latin


def test_stale_untranslated_reuse_self_heals_glossary_fix():
    # The cache must self-heal against a glossary fix: a cached target that left a now-TRANSLATED
    # term in Latin ('Appstore' where the glossary says 应用商店) is stale and must re-translate,
    # since enforce can't reverse a free Latin rendering. A properly-translated target reuses fine.
    from dev_titles.localize import stale_untranslated_reuse

    g = _brand_glossary()
    src = "Submit your app to the Acme Appstore today."
    assert (
        stale_untranslated_reuse("今天就把您的应用提交到 Acme Appstore。", src, g, "zh-Hans")
        is True
    )
    assert (
        stale_untranslated_reuse("今天就把您的应用提交到艾克米应用商店。", src, g, "zh-Hans")
        is False
    )


def test_apply_term_pins_protects_identifier_from_substring_pin():
    # A single-word pin (video->视频) must NOT fire INSIDE a keep-Latin identifier that contains
    # that word (the react-native-video -> react-native-视频 bug). `protect` masks the identifier.
    from types import SimpleNamespace as T

    from dev_titles.glossary import load as load_glossary
    from dev_titles.translation_qe import apply_term_pins

    g = load_glossary("glossary/glossary.yaml")
    recs = [T(i=0, src="try react-native-video and video", txt="试用 react-native-video 和 video")]
    pins = [{"source": "video", "wrong": "video", "correct": "视频"}]
    apply_term_pins(recs, pins, g, "zh-Hans", protect=["react-native-video"])
    assert "react-native-video" in recs[0].txt  # identifier intact
    assert "react-native-视频" not in recs[0].txt  # not corrupted


def test_restore_question_mark():
    """zh: restore ？ on an EN question ending in a question particle. ja: NOT forced — native
    reviewers split (5 videos stripped every ？, 4 added them; the JP subtitle SG is silent)."""
    from dev_titles.localize import restore_question_mark as rq

    assert rq("Does Q support MCP?", "Q支持MCP吗", "zh-Hans").endswith("？")
    assert rq("It's ready.", "准备好了吗", "zh-Hans") == "准备好了吗"  # not a question
    assert (
        rq("What have you been using?", "最近何を使っていますか", "ja") == "最近何を使っていますか"
    )


def _sent(text, start=0.0, end=2.0):
    from types import SimpleNamespace

    return SimpleNamespace(text=text, start=start, end=end)


def test_reuse_record_index_then_source_then_retranslate():
    """Cache reuse: same index + same source -> reuse; edited source at that index -> fall back
    to the by-source index (segmentation shift) or None (re-translate)."""
    from dev_titles.glossary import Glossary
    from dev_titles.localize import SentenceRecord, _reuse_record

    g = Glossary(version="t", languages=["ja", "zh-Hans"], terms=[])
    norm = lambda t, s: t  # noqa: E731
    old = SentenceRecord("ja", 0, "Hello there.", "こんにちは。", "", [])
    moved = SentenceRecord("ja", 5, "Moved line.", "移動した行。", "", [])
    cache, by_src = {0: old, 5: moved}, {old.src: old, moved.src: moved}
    rec, _ = _reuse_record(0, _sent("Hello there."), cache, by_src, g, "ja", norm, print)
    assert rec.txt == "こんにちは。" and rec.i == 0
    rec, _ = _reuse_record(0, _sent("Moved line."), cache, by_src, g, "ja", norm, print)
    assert rec.txt == "移動した行。" and rec.i == 0  # found by content at its new index
    assert _reuse_record(0, _sent("Hello, LG."), cache, by_src, g, "ja", norm, print) is None


def test_reuse_record_applies_current_normalizer():
    """A reused translation goes through the CURRENT normalizer (register/punct), like fresh ones."""
    from dev_titles.glossary import Glossary
    from dev_titles.localize import SentenceRecord, _reuse_record

    rec0 = SentenceRecord("zh-Hans", 0, "Thanks.", "谢谢你", "", [])
    rec, _ = _reuse_record(
        0,
        _sent("Thanks."),
        {0: rec0},
        {rec0.src: rec0},
        Glossary(version="t", languages=["ja", "zh-Hans"], terms=[]),
        "zh-Hans",
        lambda t, s: t.replace("你", "您"),
        print,
    )
    assert rec.txt == "谢谢您"


def test_translate_fresh_retries_then_succeeds(monkeypatch):
    """A transient failure is retried; the second attempt's translation is kept."""
    import dev_titles.localize as _loc
    from dev_titles.config import spec_for
    from dev_titles.glossary import Glossary

    monkeypatch.setattr(_loc, "_SEGMENT_RETRY_BACKOFF", (0, 0))
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("hung")
        return "訳文", []

    monkeypatch.setattr(_loc, "translate_segment", flaky)
    monkeypatch.setattr(
        _loc, "_omission_gate", lambda tr, g, s, lang, txt, iss, *a, **k: (txt, iss)
    )
    txt, iss, residual = _loc._translate_fresh(
        0,
        _sent("Hi."),
        None,
        Glossary(version="t", languages=["ja", "zh-Hans"], terms=[]),
        "ja",
        "Japanese",
        spec_for("ja"),
        ctx="",
        exemplar_bank=None,
        extra_dnt=[],
        invoke=None,
        backcheck=False,
        edit_fix_rounds=0,
        logger=lambda m: None,
    )
    assert (txt, residual, len(calls)) == ("訳文", "", 2)


def test_line_context_sits_beside_the_line_in_both_prompts():
    """Meaning/speaker notes + neighbors go next to the line (translate AND reflect prompts), not
    into the whole-transcript reference block; an empty line_context leaves the prompts unchanged.
    """
    from dev_titles.translate.base import TranslationRequest
    from dev_titles.translate.prompts import build_reflect_prompt, build_translate_prompt

    lc = "THE LINE IN CONTEXT:\n>> Bring it home.\nMEANING NOTE for this line: make it concrete"
    req = TranslationRequest("Bring it home.", "en", "ja", full_context="T", line_context=lc)
    tp, rp = build_translate_prompt(req), build_reflect_prompt(req, "x")
    assert tp.index("</CONTEXT>") < tp.index("MEANING NOTE") < tp.index("<TRANSLATE_THIS>")
    assert "MEANING NOTE" in rp and rp.index("</SOURCE>") < rp.index("MEANING NOTE")
    bare = TranslationRequest("Bring it home.", "en", "ja", full_context="T")
    assert "</CONTEXT>\n\nTranslate ONLY" in build_translate_prompt(bare)


def test_line_context_window_marks_the_line():
    from dev_titles.localize import _line_context

    sents = [_sent(t) for t in ["A one.", "B two.", "C three.", "D four.", "E five.", "F six."]]
    out = _line_context(2, sents, ["Host", "Guest", "Host", "Guest", "Host", "Guest"], None)
    assert ">> Host: C three." in out and "   Guest: B two." in out and "E five." in out
    assert "F six." not in out and "SPEAKER OF THE LINE TO TRANSLATE: Host" in out


def test_fixed_line_matches_whole_sentence_only():
    from dev_titles.glossary import Glossary
    from dev_titles.localize import _fixed_line

    g = Glossary(version="t", languages=["ja"], terms=[])
    g.style = {"ja": {"fixed_lines": {"Sure.": "はい"}}}
    assert _fixed_line(g, "ja", "  sure. ") == "はい"
    assert _fixed_line(g, "ja", "Make sure.") is None
    assert _fixed_line(g, "zh-Hans", "Sure.") is None
