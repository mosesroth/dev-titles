"""Split from test_core: glossary tests (30)."""

from helpers import _FakeTranslator

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


def test_cjk_keeps_multiword_latin_terms_whole():
    """Multi-word Latin terms were 'splatted' across line/cue breaks (React|Native,
    Orbit|TV). A space-joined Latin run must stay together."""
    from dev_titles.resegment import resegment_translation
    from dev_titles.segment import wrap_lines

    spec = spec_for("ja")
    for term, filler in [("React Native", "あ" * 10), ("Orbit TV", "あ" * 12)]:
        lines = wrap_lines(filler + term + "を使う", spec)
        assert any(term in ln for ln in lines), (term, lines)
    # and not split when it lands at a CUE boundary either
    txt = "アプリは他のReact Nativeプラットフォームと同じ動作をする必要があります"
    cues = resegment_translation(txt, 0.0, 8.0, spec)
    joined = " / ".join(l for c in cues for l in c.lines)
    assert "React Native" in joined and "React /" not in joined, joined


def test_cjk_keeps_glossary_target_whole_across_cue():
    """A localized multi-token glossary term (開発者ポータル) must not split at the
    CJK compound boundary across a cue (a native-reviewer finding)."""
    from dev_titles.resegment import resegment_translation

    spec = spec_for("ja")
    txt = "これはコーディングエージェントを接続し当社の開発者ポータルとの統合などに使用します"
    cues = resegment_translation(txt, 0.0, 10.0, spec, keep_together=["開発者ポータル"])
    joined = " / ".join(l for c in cues for l in c.lines)
    assert "開発者ポータル" in joined, joined


def test_enforce_mcp_acronym_and_mixed_script_avoid():
    """Systemic (cross-video) enforcement: (1) the MCP house rule keeps the acronym
    (never expands to the full name mid-video); (2) a MIXED-script avoid form
    (Strictモード) is corrected — the \\b-regex path silently skipped it, so
    _replace_form must fall back to substring replace for non-ASCII forms. Uses a
    per-video ephemeral term for (2), since descriptive terms live in that layer now."""
    from dev_titles.vocab_prep import renderings_to_terms

    g = load("glossary/glossary.yaml")
    # (1) MCP acronym rule is cross-video house vocabulary -> stays in static glossary
    assert (
        "Model Context Protocol"
        not in g.enforce("Model Context Protocol サーバー", "MCP server", "ja")[0]
    )
    # (2) mixed-script avoid form via a per-video pinned rendering
    terms = renderings_to_terms(
        {
            "ja": {
                "strict mode": {"canonical": "StrictMode", "avoid": ["Strictモード", "厳格モード"]}
            }
        }
    )
    g.terms.extend(terms)
    assert g.enforce("Strictモードを有効に", "strict mode", "ja")[0].startswith("StrictMode")
    assert g.enforce("厳格モードを有効に", "strict mode", "ja")[0].startswith("StrictMode")


def test_promote_renderings_to_glossary(tmp_path):
    """propose→approve→PROMOTE: an approved per-video rendering is appended to the
    SHARED glossary (so future videos reuse it), while DNT / already-present /
    curated-word-overlapping terms are skipped. The glossary still parses + enforces."""
    import shutil

    from dev_titles.vocab_prep import promote_renderings_to_glossary

    gpath = tmp_path / "g.yaml"
    shutil.copy("glossary/glossary.yaml", gpath)
    r = {
        "ja": {
            "useTransition": {"canonical": "useTransition", "avoid": []},  # DNT
            "concurrent rendering": {
                "canonical": "並行レンダリング",
                "avoid": [],
            },  # already present
            "render function": {"canonical": "レンダー関数", "avoid": []},  # ⊃ 'render'
            "time slicing": {"canonical": "タイムスライシング", "avoid": ["時間分割"]},  # NEW
        }
    }
    promoted = promote_renderings_to_glossary(r, gpath, source_video="Demo")
    assert promoted == ["time slicing"], promoted
    g2 = load(gpath)  # must still parse
    assert g2.enforce("時間分割の話", "time slicing", "ja")[0] == "タイムスライシングの話"


def test_replace_form_cjk_seam_dedup():
    """A short CJK avoid-form that is a PREFIX of a longer compound must not duplicate the
    canonical's tail char (网页端 + 网页->Web端 => Web端端). Seam-dedup fixes that WITHOUT
    touching a legitimate standalone fix (网页上 -> Web端上, no duplication)."""
    from dev_titles.glossary import _replace_form

    assert (
        _replace_form("网页端支持", "网页", "Web端") == "Web端支持"
    )  # compound overlap -> deduped
    assert (
        _replace_form("在网页上运行", "网页", "Web端") == "在Web端上运行"
    )  # standalone -> preserved
    assert _replace_form("网页版本", "网页", "Web端") == "Web端版本"  # no false dedup (版≠端)
    # ASCII avoid-forms keep word-boundary behavior (no seam logic)
    assert _replace_form("open the app now", "app", "application") == "open the application now"


def test_enforce_avoid_form():
    from dev_titles.glossary import Glossary, Term

    g = Glossary(
        "0",
        ["zh-Hans"],
        [
            Term(
                "app",
                {"zh-Hans": "应用"},
                aliases=["application"],
                enforce="hard",
                avoid={"zh-Hans": ["应用程序"]},
            )
        ],
    )
    fixed, issues = g.enforce("打开这个应用程序", "open this app", "zh-Hans")
    assert "应用程序" not in fixed and "应用" in fixed
    assert any(i.kind == "fixed" for i in issues)


def test_glossary_loads_and_injects_heuristics():
    from dev_titles.glossary import load
    from dev_titles.translate import TranslationRequest, build_translate_prompt

    g = load("glossary/glossary.yaml")
    assert g.heuristics and "开发者工具" in g.heuristics  # loaded (was dead config)
    req = TranslationRequest(
        "Use the Acme Developer Tools", "en", "ja", localization_policy=g.heuristics
    )
    prompt = build_translate_prompt(req)
    assert "LOCALIZATION POLICY" in prompt and "开发者工具" in prompt


def test_mask_unmask_commands():
    from dev_titles.translate.pipeline import _mask_spans, _unmask

    src = "Run npx create-expo-app then run:android to build."
    masked, mask = _mask_spans(src, ["npx create-expo-app", "create-expo-app", "run:android"])
    assert "create-expo-app" not in masked and "run:android" not in masked  # masked out
    assert "[[0]]" in masked  # placeholder present
    # round-trip restores verbatim
    assert _unmask(masked, mask) == src
    # tolerates full-width / lenticular brackets a CJK model may emit
    cjk = masked.replace("[[0]]", "［［0］］").replace("[[1]]", "【1】")
    restored = _unmask(cjk, mask)
    assert "npx create-expo-app" in restored and "run:android" in restored


def test_termbase_merged_into_glossary(tmp_path):
    """A sibling termbase.yaml is merged in as soft long-tail guidance; the curated glossary
    wins on a shared source form, and `termbase_exclude` rows are skipped."""
    import shutil

    from dev_titles.glossary import load

    shutil.copy("glossary/glossary.yaml", tmp_path / "glossary.yaml")
    (tmp_path / "termbase.yaml").write_text(
        "languages: [zh-Hans, ja]\n"
        "terms:\n"
        "  - {source: application binary interface, zh-Hans: 应用二进制接口, ja: アプリケーションバイナリインターフェイス}\n"
        "  - {source: app, zh-Hans: 应用程序, ja: アプリケーション}\n"
        "  - {source: Hello, zh-Hans: 尊敬的开发者，您好！, ja: こんにちは}\n",
        encoding="utf-8",
    )
    base = load("glossary/glossary.yaml")
    g = load(tmp_path / "glossary.yaml")
    assert len(g.terms) == len(base.terms) + 1  # only the new TB term was added
    srcs = {t.source.lower() for t in g.terms}
    assert "application binary interface" in srcs  # a TB term is present
    assert "hello" not in srcs  # termbase_exclude row skipped
    # hand-curated glossary.yaml WINS on a shared source form
    app = [t for t in g.terms if t.source == "app"]
    assert len(app) == 1 and app[0].translations.get("zh-Hans") == "应用"
    # a TB term resolves to its approved rendering, as a SOFT term
    abi = [t for t in g.terms if t.source == "application binary interface"][0]
    assert abi.translations.get("zh-Hans") == "应用二进制接口" and abi.enforce == "soft"


def test_soft_term_leftover_english_translated():
    """A SOFT term with a CJK canonical left verbatim in English in the target (untranslated)
    is converted to the canonical — source-gated. Fixes 'Chief Evangelist' kept in English."""
    from dev_titles.glossary import load as load_glossary

    g = load_glossary("glossary/glossary.yaml")
    out, iss = g.enforce(
        "我是Acme的Chief Evangelist", "I'm the Chief Evangelist of Acme.", "zh-Hans"
    )
    assert "Chief Evangelist" not in out and "首席布道师" in out
    outj, _ = g.enforce("AcmeのChief Evangelistです", "the Chief Evangelist of Acme", "ja")
    assert "Chief Evangelist" not in outj and "チーフエバンジェリスト" in outj


def test_case_sensitive_dnt_no_false_match():
    """A case-sensitive DNT product name ('Bolt') must not false-match the common lowercased
    word ('bolt-on') — no spurious 'missing canonical' flag."""
    from dev_titles.glossary import Glossary, Term

    g = Glossary(
        "0",
        ["zh-Hans", "ja"],
        [
            Term("Bolt", {"zh-Hans": "Bolt", "ja": "Bolt"}, enforce="hard", case_sensitive=True),
            Term("app", {"zh-Hans": "应用", "ja": "アプリ"}, enforce="hard"),
        ],
    )
    _, iss = g.enforce("一个附加功能", "a bolt-on feature", "zh-Hans")
    assert not any(i.term == "Bolt" for i in iss)  # 'bolt' the word did NOT trigger Bolt
    _, iss2 = g.enforce("这个应用很好", "The Bolt app is great.", "zh-Hans")
    assert any(
        i.term == "Bolt" and i.kind == "missing" for i in iss2
    )  # the product 'Bolt' still enforced


def test_identifier_prefix_of_word_not_masked_inside_it():
    """A DNT identifier that is a PREFIX of a longer word ('Orbi' the company vs.
    'Orbie Hicks' the founder / 'Orbie's List' the old name) must mask ONLY the
    standalone identifier, not the substring inside the longer word — a plain
    substring replace turned 'Orbie Hicks' into '<tok>e Hicks' and mangled the name.
    Also: the sentinel token must not collide with digits in the text."""
    from dev_titles.glossary import Glossary, _mask_identifier_spans

    g = Glossary("0", ["zh-Hans", "ja"], [], identifiers=["Orbi", "Nimbus+"])
    s = "Orbie Hicks, founder of Orbi, formerly Orbie's List, serves 130,000 users."
    masked = g._mask_identifiers(s)
    assert "Orbie Hicks" in masked  # person untouched
    assert "Orbie's List" in masked  # old name untouched
    assert "130,000" in masked  # number not eaten by a bare-digit token
    assert "founder of Orbi," not in masked  # standalone company name WAS masked
    # punctuation-edge identifier still masks
    t, _ = _mask_identifier_spans("Try Nimbus+ now", g.identifiers)
    assert "Nimbus+" not in t


def test_domain_masked_during_translation():
    """A glossary/translator that would translate 'acme' must NOT corrupt a URL —
    the domain is masked to a placeholder before translate, restored verbatim after."""
    from dev_titles.glossary import load as load_glossary
    from dev_titles.translate import translate_segment
    from dev_titles.translate.base import Translator

    class Corrupter(Translator):  # simulates 'acme'->艾克米 leakage
        def translate(self, req):
            return req.source_text.replace("acme", "艾克米")

        def reflect_and_refine(self, req, d):
            return d

    g = load_glossary("glossary/glossary.yaml")
    out, _ = translate_segment(
        Corrupter(), g, "Go to developer.acme.com now.", "zh-Hans", reflect=False
    )
    assert "developer.acme.com" in out  # URL survived verbatim
    assert "艾克米.com" not in out  # not corrupted mid-domain


def test_dnt_term_masked_during_translation():
    """A product-name DNT (Acme Q) must not have its 'Acme' substring localized to
    艾克米Q — masking the DNT span in the source before translate guarantees verbatim."""
    from dev_titles.glossary import load as load_glossary
    from dev_titles.translate import translate_segment
    from dev_titles.translate.base import Translator

    class Corrupter(Translator):  # simulates the 'Acme'->艾克米 term leaking
        def translate(self, req):
            return req.source_text.replace("Acme", "艾克米")

        def reflect_and_refine(self, req, d):
            return d

    g = load_glossary("glossary/glossary.yaml")
    out, _ = translate_segment(
        Corrupter(),
        g,
        "I started using Acme Q today.",
        "zh-Hans",
        reflect=False,
        extra_dnt=["Acme Q"],
    )
    assert "Acme Q" in out and "艾克米Q" not in out


def test_correction_hint_reaches_prompt_and_masks_commands():
    from dev_titles.translate.base import TranslationRequest
    from dev_titles.translate.prompts import build_translate_prompt

    req = TranslationRequest(
        source_text="x",
        source_lang="en",
        target_lang="ja",
        correction_hint="do NOT translate the [[0]] command",
    )
    p = build_translate_prompt(req)
    assert "a prior review flagged" in p and "[[0]]" in p


def test_acronym_hint_iap():
    from dev_titles.cleanup import _acronym_hint, build_correction_prompt

    # fires on the mishear 'IEP' and names IAP + its expansion
    h = _acronym_hint("the demo for IEP is great")
    assert "IAP" in h and "in-app purchase" in h and "IEP" in h
    # fires on the correct form too, and is embedded in the full prompt
    assert "IAP" in _acronym_hint("we support IAP now")
    p = build_correction_prompt("the demo for IEP", ["in-app purchase"], clean_read=True)
    assert "DOMAIN ACRONYMS" in p and "in-app purchase" in p
    # no hint when neither the acronym nor a mishear is present (keeps prompt short)
    assert _acronym_hint("just some ordinary words here") == ""
    # must not false-fire inside a larger word (e.g. 'iep' inside 'recipient')
    assert _acronym_hint("the recipient list") == ""


def test_zh_formal_pronoun_enforced():
    # informal 你 -> formal 您 (matches human zh3 register)
    assert normalize_zh_punct("你的应用二进制文件更小") == "您的应用二进制文件更小"
    assert normalize_zh_punct("这意味着你可以") == "这意味着您可以"
    # protected compounds: 迷你 (mini), 你好 (hello) unchanged
    assert "迷你" in normalize_zh_punct("迷你版本")
    assert normalize_zh_punct("你好") == "你好"


def test_glossary_load_and_retrieve():
    g = load("glossary/glossary.example.yaml")
    assert "zh-Hans" in g.languages
    hits = g.retrieve("Publish your extension to the Acme Appstore today")
    sources = {t.source for t in hits}
    assert "Acme Appstore" in sources
    assert "Acme" not in sources  # longest-match suppresses it here
    appstore = next(t for t in hits if t.source == "Acme Appstore")
    assert appstore.enforce == "hard"
    assert appstore.translations["zh-Hans"] == "艾克米应用商店"
    # the alias alone ("Appstore") also retrieves it
    assert "Acme Appstore" in {t.source for t in g.retrieve("Open the Appstore")}
    # Standalone "Acme" (no "Appstore") is retrieved on its own.
    solo = {t.source for t in g.retrieve("Log in to your Acme account")}
    assert "Acme" in solo
    acme = next(t for t in g.terms if t.source == "Acme")
    assert acme.translations["zh-Hans"] == "艾克米"  # zh translates
    assert acme.translations["ja"] == "Acme"  # ja keeps Latin


def test_enforce_fixes_untranslated_and_leaked_terms():
    g = load("glossary/glossary.yaml")
    # ja target left "submit" in English -> should become 申請 (ja), NOT 提交 (zh).
    src = "Submit your app to the developer console"
    fixed, issues = g.enforce("アプリを developer console に submit します", src, "ja")
    assert "申請" in fixed and "submit" not in fixed
    assert "提交" not in fixed
    assert any(i.term == "submit" and i.kind == "fixed" for i in issues)

    # zh target leaked the ja rendering 申請 -> should be corrected to 提交.
    fixed2, issues2 = g.enforce("将您的应用申請到开发者控制台", src, "zh-Hans")
    assert "提交" in fixed2 and "申請" not in fixed2


def test_acronym_alias_is_case_sensitive():
    from dev_titles.glossary import Glossary, Term

    g = Glossary(
        "0",
        ["zh-Hans", "ja"],
        [
            Term(
                "Acme Skills Kit",
                {"zh-Hans": "艾克米技能工具包", "ja": "Acme Skills Kit"},
                aliases=["ASK"],
                enforce="hard",
            )
        ],
    )
    # The verb "ask" must NOT match the "ASK" alias of Acme Skills Kit.
    assert "Acme Skills Kit" not in {t.source for t in g.retrieve("I would ask the question")}
    # The uppercase acronym "ASK" does match.
    assert "Acme Skills Kit" in {t.source for t in g.retrieve("Use the ASK CLI")}
    # And enforce no longer false-flags on the verb.
    _, issues = g.enforce("質問をしたいと思います", "I would ask the question", "ja")
    assert not any(i.term == "Acme Skills Kit" for i in issues)


def test_enforce_normalizes_boundary_spacing():
    g = load("glossary/glossary.yaml")
    # Model wrote "VS Code 扩展" (stray space) — should normalize to "VS Code扩展",
    # NOT flag it missing.
    fixed, issues = g.enforce(
        "在GitHub上发布您的VS Code 扩展。", "Publish your VS Code extension", "zh-Hans"
    )
    assert "VS Code扩展" in fixed and "VS Code 扩展" not in fixed
    assert not any(i.kind == "missing" and i.term == "VS Code extension" for i in issues)
    assert any(i.kind == "fixed" and i.term == "VS Code extension" for i in issues)


def test_enforce_flags_missing_and_skips_dnt():
    g = load("glossary/glossary.yaml")
    # Canonical rendering absent and no wrong form present -> flagged missing.
    _, issues = g.enforce("完全に無関係な文", "Submit the app", "ja")
    assert any(i.term == "submit" and i.kind == "missing" for i in issues)
    # DNT term (GitHub, canonical == source) present correctly -> no issue, no change.
    txt = "GitHubのプロジェクトを申請します"
    out, iss = g.enforce(txt, "Submit your GitHub project", "ja")
    assert "GitHub" in out
    assert not any(i.term == "GitHub" for i in iss)


def test_translate_prompt_includes_glossary_and_register():
    g = load("glossary/glossary.yaml")
    req = TranslationRequest(
        source_text="Submit your app",
        source_lang="en",
        target_lang="ja",
        glossary_terms=g.retrieve("Submit your app"),
        register="teineigo",
    )
    prompt = build_translate_prompt(req)
    assert "Japanese" in prompt and "teineigo" in prompt
    assert "<TRANSLATE_THIS>" in prompt and "Submit your app" in prompt
    assert "申請" in prompt  # the hard rendering for "submit" is injected


def test_translate_segment_reflects_then_enforces():
    g = load("glossary/glossary.yaml")
    # Model returns a draft that left "submit" untranslated; refine keeps it wrong;
    # deterministic enforce must still fix submit -> 申請.
    fake = _FakeTranslator(draft="アプリを submit", refined="アプリを submit する")
    out, issues = translate_segment(fake, g, "Submit your app", "ja")
    assert "申請" in out and "submit" not in out
    assert any(i.term == "submit" and i.kind == "fixed" for i in issues)
    # register/style were pulled from the glossary and passed to the model.
    assert fake.reqs[0].register == "teineigo"


def test_enforce_longest_match_no_double_rewrite():
    g = load("glossary/glossary.example.yaml")
    # "Acme Appstore" in source must not trigger standalone "Acme" rewrite
    # inside the correct ja rendering.
    fixed, issues = g.enforce("Acmeアプリストアに公開", "Publish to the Acme Appstore", "ja")
    assert "Acmeアプリストア" in fixed
    assert not any(i.kind == "missing" for i in issues)


def test_glossary_retrieve_gerund_alias():
    g = load("glossary/glossary.yaml")
    hits = [t.source for t in g.retrieve("It also supports trapping and focus redirection")]
    assert "trap the focus" in hits  # 'trapping' alias matches


def test_soft_term_avoid_form_corrected():
    # soft terms' avoid-lists must still auto-correct a wrong rendering (they were
    # previously dead — enforce filtered to hard only).
    g = load("glossary/glossary.yaml")
    fixed, iss = g.enforce(
        "它还支持焦点捕获和焦点重定向", "It also supports trapping and focus redirection", "zh-Hans"
    )
    assert "将焦点限制" in fixed and "焦点捕获" not in fixed
    assert any(i.kind == "fixed" for i in iss)


def test_correct_transcript_glossary_aware():
    from dev_titles.cleanup import build_correction_prompt, correct_transcript

    seen = {}

    def fake(prompt):
        seen["p"] = prompt
        return "Install it with npm and connect the MCP server to Claude Code."

    terms = ["npm", "MCP", "Claude Code", "Expo"]
    out = correct_transcript(
        fake, "Install it with npn and connect the mcb server to cloud code.", terms
    )
    assert out == "Install it with npm and connect the MCP server to Claude Code."
    assert "npm" in seen["p"] and "MCP" in seen["p"]  # terms injected
    assert "CLEAN-READ" in seen["p"]  # clean-read editing on by default
    # no terms AND clean_read off -> deterministic no-op (just number spelling)
    assert correct_transcript(fake, "x", [], clean_read=False) == "x"
    # clean_read on still runs (grammar/filler) even with no glossary terms
    assert (
        correct_transcript(fake, "we was testing", [])
        == "Install it with npm and connect the MCP server to Claude Code."
    )


def test_glossary_identifiers_and_in_correction_prompt():
    from dev_titles.cleanup import build_correction_prompt

    g = load("glossary/glossary.yaml")
    assert "performMountingInstructions" in g.identifiers and "TVFocusGuideView" in g.identifiers
    p = build_correction_prompt("x", ["Expo"] + g.identifiers)
    assert "performMountingInstructions" in p and "UIManagerBinding::dispatchEvent" in p


def test_term_pin_harvest_apply_and_tier():
    """Term-pin auto-repair: harvest a mechanical term-substitution from a judge critique,
    apply it DETERMINISTICALLY across ALL cues whose source has the term (fixing every
    occurrence + inconsistency, no re-translation), and leave judgment critiques alone."""
    from dev_titles.glossary import load as load_glossary
    from dev_titles.localize import SentenceRecord
    from dev_titles.translation_qe import apply_term_pins, harvest_term_pins, pins_to_renderings

    g = load_glossary("glossary/glossary.yaml")
    recs = [
        SentenceRecord("zh-Hans", 0, "We built a growth loop.", "我们打造了增长回路", "", []),
        SentenceRecord("zh-Hans", 1, "The growth loop compounds.", "增长回路会复利累积", "", []),
        SentenceRecord("zh-Hans", 2, "The performance is smooth.", "性能很流畅", "", []),
    ]
    suspects = [
        {
            "kind": "mqm-minor",
            "i": 0,
            "detail": "'growth loop' rendered 增长回路; should be 增长闭环",
        },
        {"kind": "mqm-minor", "i": 2, "detail": "awkward fluency in 性能很流畅"},
    ]

    def classify(crit, src, tgt):  # term-fix for the growth-loop critique; judgment for fluency
        if "增长闭环" in crit:
            return {"source": "growth loop", "wrong": "增长回路", "correct": "增长闭环"}
        return None

    pins, resolved = harvest_term_pins(recs, "zh-Hans", suspects, classify=classify)
    assert len(pins) == 1 and pins[0]["source"] == "growth loop" and pins[0]["cues"] == [0]
    assert len(resolved) == 1 and resolved[0]["i"] == 0  # judgment critique NOT resolved
    changed = apply_term_pins(recs, pins, g, "zh-Hans")
    assert changed == {0, 1}  # BOTH cues with "growth loop" converged
    assert recs[0].txt == "我们打造了增长闭环" and "增长回路" not in recs[1].txt
    assert recs[2].txt == "性能很流畅"  # judgment cue untouched
    r = pins_to_renderings(pins, "zh-Hans")
    assert r["zh-Hans"]["growth loop"] == {"canonical": "增长闭环", "avoid": ["增长回路"]}


def test_harvest_term_pins_corroboration_gate():
    """Sense-safety #3: a single-judge CJK->CJK override (智能体->特工 class) is HELD for human;
    a raw-English fix (multi-eight-figure) auto-applies; a >=2-judge override is allowed."""
    from dev_titles.localize import SentenceRecord
    from dev_titles.translation_qe import harvest_term_pins

    recs = [
        SentenceRecord("zh-Hans", 0, "the best agent", "最出色的智能体", "", []),
        SentenceRecord(
            "zh-Hans", 1, "multi-eight-figure impact", "multi-eight-figure的影响", "", []
        ),
        SentenceRecord("zh-Hans", 2, "the growth loop", "增长回路", "", []),
    ]
    # cue0: ONE judge flags agent; classify (hypothetically) proposes the wrong CJK override 智能体->特工
    # cue1: raw-English multi-eight-figure -> CJK (unambiguous)
    # cue2: TWO judges flag growth loop -> corroborated CJK override
    suspects = [
        {"kind": "mqm-major[codex]", "i": 0, "detail": "agent mis-rendered"},
        {"kind": "mqm-major[claude]", "i": 1, "detail": "multi-eight-figure left in English"},
        {"kind": "mqm-major[claude]", "i": 2, "detail": "growth loop wrong term"},
        {"kind": "mqm-minor[codex]", "i": 2, "detail": "growth loop inconsistent"},
    ]

    def classify(crit, src, tgt):
        if "agent" in crit:
            return {"source": "agent", "wrong": "智能体", "correct": "特工"}
        if "multi" in crit:
            return {
                "source": "multi-eight-figure",
                "wrong": "multi-eight-figure",
                "correct": "数千万ドル",
            }
        if "growth" in crit:
            return {"source": "growth loop", "wrong": "增长回路", "correct": "增长闭环"}
        return None

    pins, resolved = harvest_term_pins(recs, "zh-Hans", suspects, classify=classify)
    srcs = {p["source"] for p in pins}
    assert "agent" not in srcs  # single-judge CJK override → HELD
    assert "multi-eight-figure" in srcs  # raw-English → auto-applied
    assert "growth loop" in srcs  # 2-judge corroborated override → applied
    # the held agent critique must NOT be marked resolved (stays in human-review tier)
    assert any(s["i"] == 0 for s in suspects) and all(s.get("i") != 0 for s in resolved)
