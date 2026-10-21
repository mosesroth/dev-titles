import re
"""Split from test_core: segment tests (25)."""

from dev_titles.config import spec_for
from dev_titles.cue_builder import build_en_cues
from dev_titles.eval import (
    GembaMqmScorer,
    QeResult,
    QeScorer,
    below_threshold_segments,
    parse_gemba_mqm,
    rank_engines,
)
from dev_titles.formats import read_vtt, write_srt, write_vtt
from dev_titles.glossary import load
from dev_titles.linguistics import get_joiner, is_repeated_text
from dev_titles.model import Cue, Subtitle
from dev_titles.segment import apply_kinsoku, break_lines, normalize_zh_punct
from dev_titles.textwidth import display_width
from dev_titles.transcribe import from_transcribe_json
from dev_titles.translate import (
    TranslationRequest,
    Translator,
    build_translate_prompt,
    translate_segment,
)
from dev_titles.validate import validate_subtitle
from dev_titles.validate.rules import validity_report


def test_cjk_break_keeps_latin_words_whole():
    from dev_titles.segment import wrap_lines

    spec = spec_for("ja")  # cpl 13
    # "Nova Developer Tools" must not split mid-word across lines.
    lines = wrap_lines("あ" * 8 + "Nova Developer Toolsを使う", spec)
    joined = "\n".join(lines)
    for word in ["Nova", "Developer", "Tools"]:
        # each word appears intact on some line (never split across a newline)
        assert any(word in ln for ln in lines), (word, lines)
    # no line ends with a partial-word boundary like "Ve" immediately continued
    assert "No\nva" not in joined and not any(ln.endswith("No") for ln in lines)


def test_cjk_prefers_script_boundary_no_midword_hiragana():
    from dev_titles.segment import wrap_lines

    spec = spec_for("ja")
    lines = wrap_lines("Novaアプリ作成に必要なすべてが揃うNova Developer Toolsを使います。", spec)
    # the hiragana word すべて must not be split across lines
    assert any("すべて" in ln for ln in lines), lines
    assert not any(ln.endswith("すべ") for ln in lines)


def test_cjk_keeps_single_kanji_noun_compound_and_dependent_verb():
    """親|画面 (single-kanji-noun compound) and 必要|があります (dependent verb) must
    stay whole (native-reviewer findings)."""
    from dev_titles.segment import wrap_lines

    spec = spec_for("ja")
    lines = wrap_lines(
        "これをtrueに設定すると親画面またはコンポーネントが初めてマウントされます", spec
    )
    assert any("親画面" in ln for ln in lines) and not any(ln.endswith("親") for ln in lines), lines


def test_normalize_ja_punct_keeps_latin_identifiers_apart():
    """A 、 between two Latin identifiers becomes a space, not nothing — else
    nextFocusUp、nextFocusDown fuses into one token (a native-reviewer finding)."""
    from dev_titles.segment import normalize_ja_punct

    out = normalize_ja_punct("nextFocusUp、nextFocusDown、nextFocusLeft、nextFocusRight")
    assert out == "nextFocusUp nextFocusDown nextFocusLeft nextFocusRight", out
    # a 、 between CJK becomes a HALF-width space (JP subtitle style guide, confirmed by a native reviewer)
    assert normalize_ja_punct("こんにちは、皆さん") == "こんにちは 皆さん"


def test_normalize_ja_possessive_calque():
    """Stiff 我々の/私共の stays stripped, but 当社の/弊社の are KEPT: native
    reviewers deliberately WRITE 当社の (7 occurrences: 当社のYouTubeチャンネル, これが当社のアプリです),
    so stripping it deleted their content."""
    from dev_titles.segment import normalize_ja_punct

    assert normalize_ja_punct("これが当社のアプリです") == "これが当社のアプリです"
    assert normalize_ja_punct("弊社のGitHubをご覧ください") == "弊社のGitHubをご覧ください"
    assert (
        normalize_ja_punct("我々の開発者ポータル") == "開発者ポータル"
    )  # stiff form still stripped
    # a normal 〜の possessive is untouched
    assert normalize_ja_punct("アプリの設定") == "アプリの設定"


def test_normalize_ja_arabic_tsu_counter():
    """JP subtitle style guide arabic counters, confirmed by the reviewer set (1つ 27 / 一つ 0, 2つ 4 / 二つ 0).
    NARROW: only the 〜つ counter — kanji numerals inside words must survive (一部 4 / 1部 0)."""
    from dev_titles.segment import normalize_ja_punct

    assert normalize_ja_punct("理由の一つはUXです") == "理由の1つはUXです"
    assert normalize_ja_punct("二つの方法があります") == "2つの方法があります"
    assert normalize_ja_punct("一部のアプリ") == "一部のアプリ"  # compound word, unchanged


def test_normalize_ja_punct_drops_marks():
    """JP subtitle style guide + native reviewer: 、 -> half-width space, 。 -> full-width space;
    ？ omitted after an explicit question ending, ！ omitted — both become a sentence break."""
    from dev_titles.segment import normalize_ja_punct

    out = normalize_ja_punct("こんにちは、Priyaです。今日は開発の話をします。")
    assert out == "こんにちは Priyaです\u3000今日は開発の話をします"
    assert normalize_ja_punct("何を使っていますか？Novaです") == "何を使っていますか\u3000Novaです"
    assert (
        normalize_ja_punct("本当？すごいですね！") == "本当？すごいですね"
    )  # ？ kept: not a か-ending
    assert normalize_ja_punct("「いいですか？」と聞く") == "「いいですか」と聞く"


def test_normalize_zh_punct_drops_period_and_han_latin_space():
    from dev_titles.segment import normalize_zh_punct

    assert normalize_zh_punct("您将使用 Nova Developer Tools。") == "您将使用Nova Developer Tools"
    assert normalize_zh_punct("Nova OS 是艾克米的系统") == "Nova OS是艾克米的系统"
    # spaces between two Latin words are preserved
    assert "Nova Developer Tools" in normalize_zh_punct("使用 Nova Developer Tools")


def test_cjk_word_safe_break_no_midword():
    from dev_titles.segment import wrap_lines

    # tokenizer keeps words whole: 立即 not 立/即, 使い慣れた not 使い慣/れた
    zh = wrap_lines("立即使用Nova开发者工具开始构建你的第一个应用", spec_for("zh-Hans"))
    assert (
        "".join(zh).replace(" ", "").replace("　", "")
        == "立即使用Nova开发者工具开始构建你的第一个应用"
    )
    assert not any("立" == ln[-1] for ln in zh)  # 立即 not split at 立|即


def test_break_lines_never_drops_content():
    from dev_titles.segment import break_lines

    spec = spec_for("zh-Hans")  # cpl 16
    # A long line that needs >2 lines must NOT lose its tail (was lines[:max_lines]).
    txt = "立即使用 Nova Developer Tools 开始构建你的第一个应用程序并发布到应用商店"
    joined = "".join(break_lines(txt, spec)).replace(" ", "")
    assert joined == txt.replace(" ", ""), joined


def test_wrap_lines_preserves_characters():
    # Critical invariant: wrapping must never add/drop/REORDER characters
    # (a kinsoku swap once corrupted します。 -> し。/ます). Compare ignoring spaces.
    from dev_titles.segment import wrap_lines

    cases = [
        "OS向けの開発をご紹介します。",
        "接下来，你可以使用 React Native for Nova，即艾克米的分支版本",
        "Novaアプリ作成に必要なすべてが揃うNova Developer Toolsを使います。",
        "短い文。",
    ]
    for lang in ["ja", "zh-Hans"]:
        spec = spec_for(lang)
        for txt in cases:
            lines = wrap_lines(txt, spec)
            joined, want = "".join(lines).replace(" ", ""), txt.replace(" ", "")
            if lang.startswith("zh"):  # CN SG: a line-edge ，、； is dropped (the break is the pause)
                assert not any(ln.endswith(("，", "、", "；")) for ln in lines), lines
                joined, want = re.sub("[，、；]", "", joined), re.sub("[，、；]", "", want)
            assert joined == want, (lang, txt, joined)


def test_kinsoku_swap_does_not_split_latin_word():
    from dev_titles.segment import wrap_lines

    # Real case: a full-width comma after a Latin word ("Nova") forced a kinsoku
    # swap that split it into "Nov"/"a". Must keep "Nova" whole.
    txt = "接下来，开发应用时，你可以使用 React Native for Nova，即艾克米的分支版本"
    lines = wrap_lines(txt, spec_for("zh-Hans"))
    assert any("Nova" in ln for ln in lines)
    assert not any(ln.endswith("Nov") for ln in lines)


def test_normalize_cjk_numbers():
    from dev_titles.segment.cjk import normalize_cjk_numbers as n

    assert n("超过13万个合作伙伴") == "超过130,000个合作伙伴"  # 万 -> arabic (zh)
    assert n("13万を超えるパートナー") == "130,000を超えるパートナー"  # 万 (ja)
    assert n("1亿用户") == "100,000,000用户"  # 亿
    assert n("数百万の顧客") == "数百万の顧客"  # kanji-only stem: untouched
    assert n("一年半から2年") == "一年半から2年"  # 半 untouched


def test_normalize_en_spelling():
    from dev_titles.cleanup import normalize_en_spelling

    assert normalize_en_spelling("my favourite colour") == "my favorite color"
    assert (
        normalize_en_spelling("Optimise and initialise the behaviour")
        == "Optimize and initialize the behavior"
    )
    assert normalize_en_spelling("Favourite") == "Favorite"  # case of first letter preserved
    # must NOT false-fire on look-alikes that are correct in US English too
    for w in ["exercise", "promise", "precise", "franchise", "surprise", "wise", "rise"]:
        assert normalize_en_spelling(w) == w
    # already-American text is unchanged (safe on the WhisperX path)
    assert normalize_en_spelling("color and behavior") == "color and behavior"


def test_display_width_cjk_vs_latin():
    assert display_width("abcd", cjk=False) == 4
    # 2 full-width (1.0 each) + 2 ascii (0.5 each) = 3.0
    assert display_width("你好ab", cjk=True) == 3.0


def test_kinsoku_no_leading_punct():
    # Force a break that would leave "。" at line start; kinsoku pulls it up.
    lines = ["これはテスト", "。次の行"]
    fixed = apply_kinsoku(lines)
    assert not fixed[1].startswith("。")
    assert fixed[0].endswith("。")


def test_zh_punct_normalization():
    assert normalize_zh_punct("你好，") == "你好"
    assert normalize_zh_punct("等等...") == "等等…"


def test_joiner_and_repetition():
    assert get_joiner("ja") == ""
    assert get_joiner("zh-Hans") == ""
    assert get_joiner("en") == " "
    rep, unit = is_repeated_text("啊啊啊啊啊啊啊啊")
    assert rep and unit == "啊"
    rep2, _ = is_repeated_text("the cat sat on the mat")
    assert not rep2


def test_cjk_break_prefers_boundary():
    spec = spec_for("ja")  # max_cpl 13 (full-width units)
    # Two short clauses separated by 、 — break should fall right after 、.
    lines = break_lines("これはテストです、次の文です", spec)
    assert lines[0].endswith("、") or all(
        display_width(ln, cjk=True) <= spec.max_cpl for ln in lines
    )
    assert all(display_width(ln, cjk=True) <= spec.max_cpl for ln in lines)


def test_punct_profile_and_comma_padding():
    from dev_titles.exemplars import Exemplar, ExemplarBank, punct_profile

    exs = [
        Exemplar("a", "大家好，欢迎观看", "zh-Hans", "V1"),
        Exemplar("b", "这是测试", "zh-Hans", "V1"),
    ]
    prof = punct_profile(exs)
    assert prof["，"] > 0
    # select must include >=1 comma-bearing exemplar even if the top hit lacks one
    bank = ExemplarBank("zh-Hans").add(exs + [Exemplar("focus test", "焦点测试", "zh-Hans", "V2")])
    sel = bank.select("focus test here", k=2)
    assert any("，" in t for _, t in sel)


def test_punct_note_language_specific():
    from dev_titles.translate.pipeline import _punct_note

    assert "、" in _punct_note("ja") and "zero" in _punct_note("ja").lower()
    zh = _punct_note("zh-Hans")
    assert "，" in zh and "house rate" in zh and "NOT zero" in zh  # target house ~2/100, balanced


def test_zh_normalizer_folds_you_to_polite():
    from dev_titles.localize import _normalizer

    conv = _normalizer("zh-Hans", conversational=True)
    formal = _normalizer("zh-Hans", conversational=False)
    # REVERSED (native-reviewer verdict): the fold now goes 你 -> 您 in both registers,
    # but the plural 你们 is preserved (您们 is unidiomatic).
    assert conv("你可以使用你的账户") == "您可以使用您的账户"
    assert conv("你们都可以试试") == "你们都可以试试"
    assert "您" in formal("您可以")
    assert _normalizer("ja", conversational=True)("です") == "です"  # ja untouched


def test_ja_bunsetsu_no_particle_line_start():
    import importlib.util

    if importlib.util.find_spec("fugashi") is None:
        return  # tokenizer not installed; heuristic path covered elsewhere
    from dev_titles.segment import wrap_lines

    lines = wrap_lines("下を押すと真下の項目にフォーカスが移動します", spec_for("ja"))
    # no wrapped line may begin with a particle or the ます auxiliary
    assert all(ln[0] not in "ますはがをにでとへやの" for ln in lines[1:])


def test_latin_identifier_not_split_across_lines():
    from dev_titles.segment import wrap_lines

    for t, ident in [
        (
            "メインのUIスレッドでperformMountingInstructionsを実行します",
            "performMountingInstructions",
        ),
        ("「Systrace.capture」という名前を指定します", "Systrace.capture"),
    ]:
        lines = wrap_lines(t, spec_for("ja"))
        # the identifier must survive intact within a single line (no mid-token break)
        assert any(ident in ln for ln in lines), (ident, lines)
        assert not any(ln.endswith(("pe", "Systrace.")) for ln in lines)
    # a lone over-long identifier overflows whole rather than shattering
    lines = wrap_lines("これはperformMountingInstructionsです", spec_for("ja"))
    assert any("performMountingInstructions" in ln for ln in lines)


def test_strip_disfluencies_and_no_overlap():
    from dev_titles.cue_builder import build_en_cues, strip_disfluencies
    from dev_titles.model import Transcript, Word

    words = [
        Word("Um", 0, 0.3),
        Word("so", 0.3, 0.6),
        Word("uh", 0.6, 0.9),
        Word("submit", 0.9, 1.4),
        Word("it.", 1.4, 1.8),
    ]
    kept = [w.text for w in strip_disfluencies(Transcript("en", words)).words]
    assert kept == ["so", "submit", "it."]  # um/uh dropped, content kept
    # dense words with tiny spans -> min_dur extension must not overlap next cue
    dense = [Word(f"w{i}", i * 0.15, i * 0.15 + 0.12) for i in range(20)]
    sub = build_en_cues(Transcript("en", dense))
    for a, b in zip(sub.cues, sub.cues[1:]):
        assert a.end <= b.start + 1e-6, (a.end, b.start)


def test_latin_line_break_no_function_word_at_end():
    from dev_titles.config import LangSpec
    from dev_titles.segment import break_lines

    spec = LangSpec("en", max_cpl=20, max_cps=17)  # small cpl to force a break
    lines = break_lines("configure the media player component now", spec)
    assert len(lines) >= 2
    # no line ends on a stranded function word (the/to/of/and/a/an/with...)
    from dev_titles.segment.cjk import _NO_BREAK_AFTER

    assert all(
        ln.split()[-1].lower().strip(".,") not in _NO_BREAK_AFTER for ln in lines[:-1]
    ), lines


def test_apply_kinsoku_folds_orphan_punctuation():
    """A re-wrap can orphan a lone punctuation onto its own line; apply_kinsoku must never leave
    one. A leading cue-boundary comma is dropped; a trailing orphan never survives as its own
    line (and no line starts with forbidden punctuation)."""
    from dev_titles.config import spec_for
    from dev_titles.segment.cjk import apply_kinsoku

    spec = spec_for("zh-Hans")
    assert apply_kinsoku(["，", "或者加上一大堆破折号"], spec) == ["或者加上一大堆破折号"]
    out = apply_kinsoku(["了不该失败的却失败了我很喜欢它在流水线", "、"], spec)
    assert not any(ln.strip() and all(ch in "、，。！？；：" for ch in ln.strip()) for ln in out)
    assert not any(ln and ln[0] in "、，。！？；：" for ln in out)


def test_wrap_keeps_number_with_counter():
    """A line break must never separate a number from its CJK counter/unit ('30' | '年')."""
    from dev_titles.segment.cjk import wrap_lines

    lines = wrap_lines(
        "これが私達がやったことです 業界での30年の経験からコンテキストを取り込み", spec_for("ja")
    )
    assert not any(
        ln.endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")) for ln in lines[:-1]
    )
    assert any("30年" in ln for ln in lines)


def test_normalize_ja_prefers_kanji():
    """JP subtitle style guide: kanji over hiragana in subtitles (様々, 予め, 曖昧; 私達)."""
    from dev_titles.segment import normalize_ja_punct

    assert normalize_ja_punct("さまざまな機能") == "様々な機能"
    assert normalize_ja_punct("私たちのアプリ") == "私達のアプリ"


def test_ja_myriad_numbers():
    """JA large counts use 万/億 (JP subtitle SG '1 million USD -> 100万米ドル'; reviewers 5万人, 1,000億)."""
    from dev_titles.segment.cjk import normalize_ja_punct as n

    assert n("Discordには50,000人以上") == "Discordには5万人以上"
    assert n("年間100,000,000,000トークン") == "年間1,000億トークン"
    assert n("13万を超えるパートナー") == "13万を超えるパートナー"
    assert n("12,345人") == "12,345人" and n("10,000msかかる") == "10,000msかかる"


def test_ja_reviewer_kanji_folds_oct():
    from dev_titles.segment.cjk import normalize_ja_punct as n

    assert n("たとえば3ヶ月ごとに") == "例えば3か月ごとに"


def test_zh_myriad_numbers_interior_question_and_quotes():
    """zh: 万/亿 for round large counts (CN SG example 1350万; reviewers 13万), ？ after interior
    吗/对吧, curly quotes."""
    from dev_titles.segment.cjk import normalize_zh_punct as n

    assert n("服务130,000多家商业合作伙伴") == "服务13万多家商业合作伙伴"
    assert n("12,345名用户") == "12,345名用户"
    assert n("真的吗哦好吧") == "真的吗？哦好吧"
    assert n("这是AI部门，对吧我们也一样") == "这是AI部门，对吧？我们也一样"
    assert n('点击"Run and Debug"面板') == "点击“Run and Debug”面板"
    assert n("我们走吧") == "我们走吧"


def test_en_sentences_keep_dotted_names_and_lowercase_continuations():
    """'React.' | 'js developer' (a caption line break inside React.js) is one sentence, not two."""
    from dev_titles.model import Cue, Subtitle
    from dev_titles.resegment import merge_en_to_sentences

    en = Subtitle("en", [
        Cue(0, 2, ["And then also as a React."]),
        Cue(2, 4, ["js developer, you're used to it."]),
        Cue(4, 6, ["Yes, my company is DNA."]),
        Cue(6, 8, ["inc, a short intro. Next one."]),
    ])
    texts = [s.text for s in merge_en_to_sentences(en)]
    assert texts == ["And then also as a React.js developer, you're used to it.",
                     "Yes, my company is DNA. inc, a short intro.", "Next one."]


def test_cjk_breaks_prefer_sentence_ends_and_merge_short_sentences():
    """Lines/cues break at sentence ends (not greedy fill), and short adjacent sentences share a
    subtitle when they fit (a native reviewer merged 'DMを送ります / 凄いですね / …' by hand)."""
    from dev_titles.config import spec_for
    from dev_titles.model import Cue, Subtitle
    from dev_titles.resegment import build_target_cues, merge_en_to_sentences
    from dev_titles.segment import wrap_lines

    ja = spec_for("ja")
    from dev_titles.resegment import resegment_translation

    txt = "両者は密接に結合しているべきではありません　しかしこれまでの動画ライブラリは"
    assert wrap_lines(txt, ja)[:2] == ["両者は密接に", "結合しているべきではありません"]
    first = resegment_translation(txt, 0.0, 6.0, ja)[0]
    assert first.lines[-1].endswith("ありません"), first.lines  # the cue ends with the sentence
    en = Subtitle("en", [Cue(0.0, 1.2, ["I'll DM you."]), Cue(1.2, 2.0, ["Amazing."]),
                         Cue(2.0, 3.4, ["Great app."]), Cue(3.4, 4.2, ["Nice."])])
    sents = merge_en_to_sentences(en)
    cues = build_target_cues(en, sents, ["DMを送ります", "凄いですね", "素晴らしいアプリです", "いいですね"],
                             ja, free_segment=True)
    assert len(cues) <= 2 and all(len(c.lines) <= 2 for c in cues), [c.lines for c in cues]
    assert "".join("".join(c.lines) for c in cues).replace("　", "") == "DMを送ります凄いですね素晴らしいアプリですいいですね"


def test_en_layout_rejoins_dotted_name_split_across_cues():
    from dev_titles.model import Cue, Subtitle
    from dev_titles.resegment import normalize_en_layout

    sub = Subtitle("en", [Cue(0, 3, ["the other one", "is that as a React."]),
                          Cue(3, 6, ["js developer, you're used", "to everything in React."])])
    out = normalize_en_layout(sub).cues
    assert out[0].lines[-1].endswith("React.js") and out[1].lines[0].startswith("developer")
    assert out[1].lines[-1].endswith("in React.")  # a real sentence end is left alone


def test_cjk_short_year_and_stray_english_hint():
    from dev_titles.editor import deterministic_hints
    from dev_titles.segment.cjk import normalize_ja_punct, normalize_zh_punct

    assert normalize_ja_punct("'24年に受賞しました").startswith("24年")
    assert "’" not in normalize_zh_punct("在’24年获奖")
    assert any("LLM" in h for h in deterministic_hints("届けられます LLM", "ja", "to more people."))
    assert not deterministic_hints("Apple TVで動きます", "ja", "It runs on Apple TV.")
    assert not deterministic_hints("LLMを使います", "ja", "We use LLMs.")


def test_kinsoku_never_splits_a_katakana_word():
    """A line can't start with ・; when ・ can't be pulled up, move the whole katakana word down
    (was: one char, 'サムネイ' | 'ル・')."""
    from dev_titles.config import spec_for
    from dev_titles.segment import wrap_lines

    lines = wrap_lines("「サイズの都合上 アセット・サムネイル・プレビューは表示されません」", spec_for("ja"))
    assert not any(a.endswith("サムネイ") for a in lines), lines
    assert all(not ln.startswith("・") for ln in lines)


def test_en_layout_split_has_no_orphans_or_leading_punctuation():
    """Re-fitting an over-full EN cue: two balanced cues, no 1-2 word orphan, no cue starting with
    punctuation, no line ending on 'the'/'into'."""
    from dev_titles.model import Cue, Subtitle
    from dev_titles.resegment import normalize_en_layout

    sub = Subtitle("en", [Cue(0, 6, ["the task might be, and then it just fades into the background",
                                     "when you don't need it. And another long tail here"]),
                          Cue(6, 12, ["change how the team thinks about it. That's been an amazing thing with Orbit+ ,",
                                      "that we're in the hands of millions and millions of users in the US, Canada,"])])
    out = normalize_en_layout(sub).cues
    assert all(len(c.lines) <= 2 and all(len(ln) <= 42 for ln in c.lines) for c in out)
    assert all(len(" ".join(c.lines).split()) >= 3 for c in out)
    assert not any(c.lines[0][:1] in ",.;:" for c in out)
    assert not any(ln.split()[-1].lower() in ("the", "into", "and") for c in out for ln in c.lines)


def test_en_sentence_start_capitalized_but_not_products():
    from dev_titles.cleanup import capitalize_sentence_starts
    from dev_titles.model import Cue, Subtitle
    from dev_titles.resegment import normalize_en_layout

    assert capitalize_sentence_starts("seconds delay. it's like,") == "seconds delay. It's like,"
    assert capitalize_sentence_starts("the project. tmux is good") == "the project. tmux is good"
    assert capitalize_sentence_starts("e.g. the app") == "e.g. the app"
    out = normalize_en_layout(Subtitle("en", [Cue(0, 2, ["a couple of seconds delay."]),
                                              Cue(2, 4, ["it's like, a lot"]), Cue(4, 6, ["tmux is a good way"])]))
    assert out.cues[1].lines[0].startswith("It's") and out.cues[2].lines[0].startswith("tmux")


def test_zh_tech_words_and_particles_stay_attached():
    from dev_titles.config import spec_for
    from dev_titles.segment import wrap_lines

    zh = spec_for("zh-Hans")
    lines = wrap_lines("我们正在更深入地集成来自后端系统的数据和信息", zh)
    assert not any(a.endswith("后") and b.startswith("端") for a, b in zip(lines, lines[1:])), lines
    lines = wrap_lines("这个发现很有意思也促使我们开始思考明年的路线图该怎么规划", zh)
    assert not any(ln.startswith(("的", "了", "们")) for ln in lines), lines
