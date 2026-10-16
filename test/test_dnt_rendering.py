"""Keep-Latin per-video renderings (canonical == source) must be masked as DNT.

Regression: 'Trivia Blast' was proposed as a rendering with canonical=='Trivia Blast' (keep
Latin) but NOT added to identifiers; renderings_to_terms skips canonical==source, so it
was left unprotected and the translator rendered it 'トリビア ブラスト' inconsistently across
cues, tanking the ja DQF. effective_glossary must now route such renderings into extra_dnt.
"""

from dev_titles.glossary import Glossary
from dev_titles.localize import effective_glossary


def test_keep_latin_rendering_becomes_dnt():
    g = Glossary(version="0.1", languages=["ja", "zh-Hans"], terms=[])
    vocab = {
        "identifiers": [],
        "terms": [],
        "names": [],
        "renderings": {"ja": {"Trivia Blast": {"canonical": "Trivia Blast", "avoid": []}}},
    }
    _, dnt = effective_glossary(g, vocab)
    assert "Trivia Blast" in dnt  # keep-Latin rendering is now masked as DNT


def test_translated_rendering_is_not_dnt():
    # a rendering with a real translation (canonical != source) is a normal pinned term,
    # NOT a keep-Latin DNT — it must NOT be forced into the DNT/mask list.
    g = Glossary(version="0.1", languages=["ja", "zh-Hans"], terms=[])
    vocab = {
        "identifiers": [],
        "terms": [],
        "names": [],
        "renderings": {"ja": {"waitlist": {"canonical": "ウェイトリスト", "avoid": []}}},
    }
    _, dnt = effective_glossary(g, vocab)
    assert "waitlist" not in dnt


def test_dnt_list_is_deduped():
    g = Glossary(version="0.1", languages=["ja", "zh-Hans"], terms=[], identifiers=["Nimbus"])
    vocab = {"identifiers": ["Nimbus"], "terms": [], "names": [], "renderings": {}}
    _, dnt = effective_glossary(g, vocab)
    assert dnt.count("Nimbus") == 1


def test_glossary_keep_latin_term_masked_per_lang():
    # Orbit TV keeps Latin in both langs -> DNT both; Acme is Latin in ja but 艾克米 in zh
    # -> DNT for ja only (per-lang). This is the Orbit TV / 轨道电视 fix.
    from dev_titles.glossary import Term

    g = Glossary(
        version="0.1",
        languages=["zh-Hans", "ja"],
        terms=[
            Term(source="Orbit TV", translations={"zh-Hans": "Orbit TV", "ja": "Orbit TV"}),
            Term(source="Acme", translations={"zh-Hans": "艾克米", "ja": "Acme"}),
        ],
    )
    _, dnt_zh = effective_glossary(g, None, lang="zh-Hans")
    _, dnt_ja = effective_glossary(g, None, lang="ja")
    assert "Orbit TV" in dnt_zh and "Orbit TV" in dnt_ja
    assert "Acme" in dnt_ja and "Acme" not in dnt_zh


def test_keep_latin_terms_unifies_all_sources():
    from dev_titles.glossary import Term
    from dev_titles.localize import keep_latin_terms

    g = Glossary(
        version="0.1",
        languages=["zh-Hans"],
        terms=[Term(source="Orbit TV", translations={"zh-Hans": "Orbit TV"})],
        identifiers=["Nimbus"],
    )
    vocab = {
        "identifiers": ["SDK"],
        "terms": [],
        "names": ["Dana"],
        "renderings": {"zh-Hans": {"Trivia Blast": {"canonical": "Trivia Blast"}}},
    }
    dnt = keep_latin_terms(g, vocab, "zh-Hans")
    for t in ["Nimbus", "Orbit TV", "SDK", "Dana", "Trivia Blast"]:
        assert t in dnt, f"{t} missing from unified DNT"


def test_dnt_survival_audit_flags_translated_keep_latin():
    from dev_titles.glossary import Term
    from dev_titles.localize import dnt_survival_violations
    from dev_titles.model import Cue, Subtitle

    g = Glossary(
        version="0.1",
        languages=["zh-Hans"],
        terms=[Term(source="Orbit TV", translations={"zh-Hans": "Orbit TV"})],
    )
    en = Subtitle("en", [Cue(0, 3, ["I love Orbit TV"]), Cue(3, 6, ["and Nimbus"])]).renumber()
    bad = Subtitle("zh-Hans", [Cue(0, 3, ["我爱轨道电视"]), Cue(3, 6, ["还有Nimbus"])]).renumber()
    ok = Subtitle("zh-Hans", [Cue(0, 3, ["我爱Orbit TV"]), Cue(3, 6, ["还有Nimbus"])]).renumber()
    v = dnt_survival_violations(en, bad, g, None, "zh-Hans")
    assert any(x["term"] == "Orbit TV" and x["cue"] == 1 for x in v)
    assert dnt_survival_violations(en, ok, g, None, "zh-Hans") == []
