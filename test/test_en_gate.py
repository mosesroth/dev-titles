"""The English sign-off gate: nothing is translated from an EN script with held review items."""

from dev_titles import en_gate
from dev_titles.model import Cue, Subtitle


def _en(text="We have this place where we can use it."):
    return Subtitle("en", [Cue(0.0, 2.0, ["Hello."]), Cue(2.0, 5.0, [text])])


def test_clean_review_signs_off_automatically(tmp_path):
    en_gate.save_review_state(tmp_path, "ok", [])
    ok, why, held = en_gate.check(_en(), tmp_path)
    assert ok and held == [] and "automatically" in why
    assert (tmp_path / en_gate.APPROVAL).exists()


def test_held_items_block_until_signed_off(tmp_path):
    held = [{"ts": "13:48", "kind": "en-garbled", "detail": "'this plush' is nonsensical"}]
    en_gate.save_review_state(tmp_path, "ok", held)
    ok, why, open_items = en_gate.check(_en(), tmp_path)
    assert not ok and open_items == held and "held" in why
    assert not en_gate.require(_en(), tmp_path)
    en_gate.approve(tmp_path, _en(), by="tester", note="plush -> place")
    assert en_gate.check(_en(), tmp_path)[0]


def test_editing_the_en_after_sign_off_needs_a_new_sign_off(tmp_path):
    en_gate.save_review_state(
        tmp_path, "ok", [{"ts": "03:29", "kind": "en-garbled", "detail": "x"}]
    )
    en_gate.approve(tmp_path, _en(), by="tester")
    ok, why, _ = en_gate.check(_en("We have this plush where we can use it."), tmp_path)
    assert not ok and "changed" in why


def test_failed_or_missing_review_blocks(tmp_path):
    assert not en_gate.check(_en(), tmp_path)[0]  # no review at all
    en_gate.save_review_state(tmp_path, "failed", [])
    ok, why, _ = en_gate.check(_en(), tmp_path)
    assert not ok and "did not run" in why


def test_legacy_markdown_review_is_read(tmp_path):
    (tmp_path / "en_review.md").write_text(
        "# EN editorial review\n\n## Auto-applied clear-error fixes (0)\n\n_None._\n\n"
        "## Needs your adjudication (1)\n\n| Time | Kind | Detail |\n|---|---|---|\n"
        '| 13:48 | en-garbled | [cue 178] "this plush" is nonsensical |\n',
        encoding="utf-8",
    )
    status, held = en_gate.review_state(tmp_path)
    assert status == "ok" and len(held) == 1 and held[0]["ts"] == "13:48"
    assert not en_gate.check(_en(), tmp_path)[0]


def test_blocked_gate_writes_listen_page(tmp_path):
    (tmp_path / "SOURCE-URL.txt").write_text("https://www.youtube.com/watch?v=abcDEF12345\n")
    en = Subtitle(
        "en",
        [
            Cue(826.0, 828.0, ["Slash AI. Okay."]),
            Cue(828.7, 833.6, ["We have this plush where", "we can just use it."]),
        ],
    )
    en_gate.save_review_state(
        tmp_path,
        "ok",
        [
            {
                "ts": "13:48",
                "kind": "en-garbled",
                "detail": '[cue 2] "this plush where we can just use it" is nonsensical',
            }
        ],
    )
    assert not en_gate.require(en, tmp_path)
    page = (tmp_path / "EN_LISTEN.html").read_text(encoding="utf-8")
    assert "watch?v=abcDEF12345&t=826s" in page  # 2 s before 13:48.7
    assert "<span class='q'>this plush where we can just use it</span>" in page.replace("\n", " ")
