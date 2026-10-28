"""zh line breaking: word-safe at the width cap, and a re-join keeps the pause a line break carried."""

from dev_titles.config import spec_for
from dev_titles.segment import break_lines
from dev_titles.segment.cjk import join_cue_lines_zh

ZH = spec_for("zh-Hans")


def test_comma_at_the_break_is_dropped_not_the_word_split():
    out = break_lines(
        "举几个有意思的例子，Figma有一个MCP连接器，您打开Figma窗口，就能告诉Claude之类", ZH
    )
    assert out[0].endswith("连接器") and not out[1].startswith("器")


def test_join_keeps_the_pause_and_rejoins_a_split_word():
    assert (
        join_cue_lines_zh(["我能把所有热词都塞进一句话里", "科技互联 垂直整合的"])
        == "我能把所有热词都塞进一句话里 科技互联 垂直整合的"
    )
    assert join_cue_lines_zh(["我们会确", "保每个人都满意"]) == "我们会确保每个人都满意"


def test_no_line_ends_on_a_conjunction_but_a_verb_ending_in_yu_is_fine():
    from dev_titles.segment.cjk import _ends_on_conjunction

    out = break_lines("人类始终参与其中，我们可以加以引导 但实际上它们会主动去解决大", ZH)
    assert not out[0].rstrip().endswith("但")
    assert _ends_on_conjunction("我们可以加以引导 但") and not _ends_on_conjunction(
        "后来他吸引了越来越多的贡献者参与"
    )


def test_no_break_inside_yexing():
    from dev_titles.config import spec_for
    from dev_titles.segment.cjk import wrap_lines

    lines = wrap_lines("照着文档操作就行，或者像我说的，直接问也行", spec_for("zh-Hans"))
    assert not any(a.endswith("也") and b.startswith("行") for a, b in zip(lines, lines[1:]))
