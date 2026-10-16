"""Split from test_core: vocab tests (4)."""

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


def test_vocab_prep_renderings_pin_and_static_wins():
    """Per-video renderings pin ONE target per term; static glossary always wins,
    DNT terms are skipped, and a per-video term OVERLAPPING a curated word is dropped
    (render function ⊃ render) so they can't fight."""
    from dev_titles.vocab_prep import propose_renderings, renderings_to_terms

    def fake(prompt):
        return (
            '{"concurrent rendering": {"canonical": "並行レンダリング", "avoid": ["同時レンダリング"]},'
            ' "useTransition": {"canonical": "useTransition", "avoid": []},'
            ' "render function": {"canonical": "レンダー関数", "avoid": ["レンダリング関数"]},'
            ' "side effect": {"canonical": "副作用", "avoid": ["サイドエフェクト"]}}'
        )

    r = propose_renderings(
        ["concurrent rendering", "useTransition", "render function", "side effect"],
        fake,
        langs=("ja",),
    )
    assert r["ja"]["concurrent rendering"]["canonical"] == "並行レンダリング"
    # static owns "concurrent rendering" and the word "render"; DNT useTransition skipped
    terms = renderings_to_terms(r, skip_sources={"concurrent rendering", "render"})
    srcs = {t.source for t in terms}
    assert srcs == {"side effect"}, srcs  # only the true gap survives
    assert terms[0].avoid["ja"] == ["サイドエフェクト"]


def test_prep_video_vocab_live_flow(tmp_path):
    """Live one-call prep: identify subject -> fetch docs -> build+save vocab; and the
    needs_input gate when nothing can be identified."""
    from dev_titles.vocab_prep import prep_video_vocab

    def invoke(p):
        if "identify" in p.lower() or "SUBJECT" in p:
            return '{"subject":"RN drag&drop","products":["React Native"],"doc_urls":["https://x/docs"],"confident":true}'
        return "{}"  # propose_renderings -> empty is fine

    res = prep_video_vocab(
        "we add drag and drop with react native",
        invoke,
        web_fetch=lambda u: "Call onDragStart and setNativeProps in DragStartHelper.",
        out_path=tmp_path / "v.json",
    )
    assert res["needs_input"] is False and (tmp_path / "v.json").exists()
    assert "onDragStart" in res["vocab"]["identifiers"]

    def unsure(p):
        return '{"subject":"","doc_urls":[],"confident":false}'

    r2 = prep_video_vocab("mumble", unsure)  # no docs, not confident
    assert r2["needs_input"] is True


def test_extract_lookup_terms_drops_capitalized_common_nouns():
    # A common noun capitalized only at a sentence start ('Customers') must NOT be harvested as
    # a proper-noun term (else keep_latin_terms force-keeps it Latin). A word whose lowercase
    # uses outnumber its capitalized ones is dropped; an always-capitalized proper noun is kept.
    from dev_titles.vocab_prep import extract_lookup_terms

    text = (
        "Perfetto shows the trace. Customers want speed. "
        "We serve customers daily and customers love it. Perfetto is great."
    )
    terms = extract_lookup_terms(text)
    assert "Perfetto" in terms  # always capitalized -> kept
    assert "Customers" not in terms and "customers" not in terms  # 1 cap / 2 lower -> dropped


def test_prune_terms_to_text_drops_asr_garbles():
    # Vocab is harvested from the RAW ASR, so an ASR mishear becomes a bogus keep-Latin term
    # ('Chief Devilist' <- 'Chief Evangelist'). Once clean-read fixes the captions, a term the
    # corrected EN no longer contains must be dropped (with its rendering); identifiers/names
    # are kept (they legitimately differ from the spoken form).
    from dev_titles.vocab_prep import prune_terms_to_text

    vocab = {
        "terms": ["Chief Devilist", "Penny", "Lumora"],
        "identifiers": ["shaka.Player.configure"],
        "renderings": {
            "zh-Hans": {"Chief Devilist": {"canonical": "X"}, "Penny": {"canonical": "Penny"}}
        },
        "names": ["Emerson Vail"],
    }
    corrected = "I'm the Chief Evangelist of Acme Cloud. Penny books on Lumora."
    dropped = prune_terms_to_text(vocab, corrected)
    assert dropped == ["Chief Devilist"]
    assert vocab["terms"] == ["Penny", "Lumora"]
    assert "Chief Devilist" not in vocab["renderings"]["zh-Hans"]
    assert vocab["identifiers"] == ["shaka.Player.configure"]  # identifiers untouched


def test_vocab_prep_extract_and_harvest():
    from dev_titles.vocab_prep import build_video_vocab, extract_lookup_terms, harvest_identifiers

    seed = "Today I'll use Shaka Player and HLS with Perfetto to trace the app."
    terms = extract_lookup_terms(seed)
    assert "Shaka Player" in terms and "HLS" in terms and "Perfetto" in terms
    ids = harvest_identifiers(
        "Call Systrace.beginEvent and performMountingInstructions in UIManager."
    )
    assert "Systrace.beginEvent" in ids and "performMountingInstructions" in ids

    # build_video_vocab drives injected doc lookups
    def fake_search(q):
        return f"docs for {q}: shaka.Player.configure() and Hls.js loadSource()"

    def fake_fetch(u):
        return "MediaSource appendBuffer sourceBuffer"

    v = build_video_vocab(
        "Hi, my name is Chris Hartley. " + seed,
        fake_search,
        web_fetch=fake_fetch,
        third_party={"Shaka Player": "https://shaka/docs"},
        metadata="Orbit TV performance. Presented by Chris Hartley. Speaker: Priya Kumar.",
    )
    assert "shaka.Player.configure" in v["identifiers"] and "loadSource" in v["identifiers"]
    assert "appendBuffer" in v["identifiers"]  # from the third-party fetch
    # speaker names harvested during vocab ingestion (metadata + self-intro)
    assert "Chris Hartley" in v["names"] and "Priya Kumar" in v["names"]


def test_load_video_vocab_missing_ok(tmp_path):
    from dev_titles.vocab_prep import load_video_vocab, save_video_vocab

    assert load_video_vocab(tmp_path / "none.json") == {
        "terms": [],
        "identifiers": [],
        "renderings": {},
        "names": [],
    }
    save_video_vocab(tmp_path / "v.json", {"terms": ["X"], "identifiers": ["yZ"]})
    v = load_video_vocab(tmp_path / "v.json")
    assert v["identifiers"] == ["yZ"] and v["renderings"] == {}  # back-filled default
