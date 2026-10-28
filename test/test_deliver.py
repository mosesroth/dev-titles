"""deliver.write_deliverable: the guarded write path for a deliverable."""

import pytest

from dev_titles.deliver import DeliverRefused, write_deliverable
from dev_titles.model import Cue, Subtitle

GOOD = Subtitle("en", [Cue(0.0, 2.0, ["Hello there."]), Cue(2.0, 4.0, ["Welcome back."])])


def test_refuses_a_validation_error_of_any_kind(tmp_path):
    empty = Subtitle("en", [Cue(0.0, 2.0, ["Hello there."]), Cue(2.0, 4.0, [""])])
    with pytest.raises(DeliverRefused, match="empty_cue"):
        write_deliverable(empty, tmp_path / "V.en.srt", before=GOOD, text_change="x")
    assert not (tmp_path / "V.en.srt").exists()


def test_refuses_a_silent_text_change_and_allows_an_explained_one(tmp_path):
    edited = Subtitle(
        "en", [Cue(0.0, 2.0, ["Hello there."]), Cue(2.0, 4.0, ["Welcome back, all."])]
    )
    with pytest.raises(DeliverRefused, match="cue text changed"):
        write_deliverable(edited, tmp_path / "V.en.srt", before=GOOD)
    assert write_deliverable(
        edited, tmp_path / "V.en.srt", before=GOOD, text_change="approved EN edit"
    )


def test_writes_both_twins_and_the_library_copy_and_backs_up(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "V.en.srt").write_text("old", encoding="utf-8")
    out = write_deliverable(
        GOOD, tmp_path / "up" / "V.en.srt", also=[lib / "V.en.srt"], backup_dir=tmp_path / "bk"
    )
    assert len(out) == 4 and (tmp_path / "up" / "V.en.vtt").exists() and (lib / "V.en.vtt").exists()
    (b,) = (tmp_path / "bk").glob("lib_*_V.en.srt")
    assert b.read_text(encoding="utf-8") == "old"


def test_ignore_rules_is_explicit(tmp_path):
    line = "and we ship the app to every user today"  # 9 words, within the EN line limit
    dup = Subtitle("en", [Cue(0.0, 3.0, [line]), Cue(3.0, 6.0, [line])])
    with pytest.raises(DeliverRefused, match="duplicate_text"):
        write_deliverable(dup, tmp_path / "V.en.srt")
    assert write_deliverable(dup, tmp_path / "V.en.srt", ignore_rules=("duplicate_text",))
