"""Brand-spelling normalization in the clean-read (Appstore one word — brand-qualified only)."""

from dev_titles.cleanup import normalize_en_spelling


def test_appstore_normalized_only_when_brand_qualified():
    out = normalize_en_spelling(
        "Orbit App Store, Acme App Store, the App Store, "
        "an app in the App Store — but Apple App Store"
    )
    assert "Orbit Appstore" in out  # Orbit-qualified -> Appstore
    assert "Acme Appstore" in out  # Acme-qualified -> Appstore
    # BARE / generic 'App Store' is LEFT ALONE — in RN/mobile content it's Apple's or generic,
    # not the brand's; a blanket fold mislabels the brand.
    assert "the App Store" in out
    assert "an app in the App Store" in out
    assert "Apple App Store" in out  # Apple's is two words — untouched (not brand-qualified)


def test_noop_on_already_correct():
    assert normalize_en_spelling("The Acme Appstore is great") == "The Acme Appstore is great"


def test_collapse_immediate_repeats():
    from dev_titles.cleanup import collapse_immediate_repeats

    # degenerate re-emit collapses to one copy
    assert (
        collapse_immediate_repeats("マーケティングマーケティング・ウェブサイト")
        == "マーケティング・ウェブサイト"
    )
    assert collapse_immediate_repeats("abcabc def") == "abc def"
    # short (<=2 char) emphatic doubling is left alone
    assert collapse_immediate_repeats("很好很好") == "很好很好"
    assert collapse_immediate_repeats("そうそう") == "そうそう"
    # clean text unchanged + idempotent
    assert collapse_immediate_repeats("ありがとうございます") == "ありがとうございます"
    once = collapse_immediate_repeats("マーケティングマーケティング")
    assert once == "マーケティング" and collapse_immediate_repeats(once) == "マーケティング"
