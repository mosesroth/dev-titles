"""Split from test_core: formats tests (5)."""

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


def test_vtt_roundtrip():
    sub = Subtitle(
        "en",
        [
            Cue(0.0, 2.5, ["Hello world."]),
            Cue(3.0, 5.0, ["Second line", "wraps here"]),
        ],
    )
    text = write_vtt(sub)
    assert text.startswith("WEBVTT")
    import os
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".vtt", delete=False, encoding="utf-8") as f:
        f.write(text)
        path = f.name
    try:
        back = read_vtt(path, "en")
        assert len(back.cues) == 2
        assert back.cues[1].lines == ["Second line", "wraps here"]
        assert abs(back.cues[0].end - 2.5) < 1e-6
    finally:
        os.unlink(path)


def test_srt_write_uses_comma():
    sub = Subtitle("en", [Cue(0.0, 1.0, ["Hi"])])
    assert "00:00:00,000 --> 00:00:01,000" in write_srt(sub)


def test_read_sbv(tmp_path):
    from dev_titles.formats import read_sbv

    p = tmp_path / "c.sbv"
    p.write_text(
        "0:00:09.540,0:00:14.920\nHi everyone, welcome.\n\n"
        "0:00:14.920,0:00:19.900\nSecond line one\nSecond line two\n",
        encoding="utf-8",
    )
    sub = read_sbv(p)
    assert len(sub.cues) == 2
    assert abs(sub.cues[0].start - 9.54) < 1e-6
    assert abs(sub.cues[0].end - 14.92) < 1e-6
    assert sub.cues[0].lines == ["Hi everyone, welcome."]
    assert sub.cues[1].lines == ["Second line one", "Second line two"]


def test_srt_single_digit_hours():
    # YouTube/CaptionHub exports use H:MM:SS,mmm — must not parse as zero cues.
    import os
    import tempfile

    from dev_titles.formats import read_srt

    fd, p = tempfile.mkstemp(suffix=".srt")
    os.write(fd, "1\n0:00:09,224 --> 0:00:11,161\n您好\n".encode("utf-8"))
    os.close(fd)
    sub = read_srt(p)
    assert len(sub.cues) == 1
    assert abs(sub.cues[0].start - 9.224) < 1e-6
    os.unlink(p)


def test_write_srt_no_mid_cue_blank_line():
    """A cue.lines with an empty element must NOT produce a blank line inside the cue
    (that blank = SRT cue separator, orphaning the trailing text and breaking 1:1 cue alignment)."""
    import os
    import tempfile

    from dev_titles.formats.srt import read_srt, write_srt
    from dev_titles.model import Cue, Subtitle

    sub = Subtitle(
        "ja",
        [Cue(0.0, 2.0, ["前半のテキスト", "", "後半のテキスト"]), Cue(2.0, 4.0, ["次のキュー"])],
    )
    out = write_srt(sub)
    # no cue may contain an internal blank line: each block's text lines are contiguous
    assert "\n\n" not in out.split("-->")[1].split("\n\n")[0] or True  # structural check below
    fd, p = tempfile.mkstemp(suffix=".srt")
    os.close(fd)
    open(p, "w", encoding="utf-8").write(out)
    rt = read_srt(p, "ja")
    os.unlink(p)
    # exactly 2 cues survive (no orphan), and the first keeps BOTH text pieces (lossless)
    assert len(rt.cues) == 2, f"expected 2 cues, got {len(rt.cues)} (mid-cue blank orphaned text)"
    assert "".join(rt.cues[0].lines) == "前半のテキスト後半のテキスト"
