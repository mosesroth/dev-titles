#!/usr/bin/env python3
"""Restore REAL pauses (clause boundaries) that a re-wrap / re-time welded shut in delivered zh tracks.
Candidates: places where the vendor's file separated two characters (a line break or a mark) and our
delivered track runs them together. Most are plain wraps mid-clause, so an LLM judges each one; only a
confirmed clause boundary gets a space. Writes via deliver.write_deliverable (text otherwise identical).
"""

import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dev_titles.cli import _read_any
from dev_titles.config import spec_for
from dev_titles.deliver import write_deliverable
from dev_titles.model import Cue, Subtitle
from dev_titles.segment import break_lines
from dev_titles.segment.cjk import _word_break_positions
from dev_titles.textwidth import display_width
from dev_titles.translate import CliTranslator

spec = spec_for("zh-Hans")
inv = CliTranslator()._invoke


def mid(a, b):
    if not (a and b and "一" <= a[-1] <= "鿿" and "一" <= b[0] <= "鿿"):
        return False
    ok = _word_break_positions(a[-6:] + b[:6], "zh-Hans")
    return bool(ok) and len(a[-6:]) not in ok


def chars(sub, check):
    c, sep = [], []
    lines = [ln for x in sub.cues for ln in x.lines]
    for li, ln in enumerate(lines):
        for ch in ln:
            if ch in " 　，、；":
                if sep:
                    sep[-1] = True
            else:
                c.append(ch)
                sep.append(ch in "。！？：")
        if sep and li + 1 < len(lines) and not (check and mid(ln, lines[li + 1])):
            sep[-1] = True
    return "".join(c), sep


P = """For each item: a Simplified Chinese subtitle phrase with a ‖ mark between two characters. Is there a
CLAUSE BOUNDARY or natural PAUSE at ‖ (where a comma or a space belongs: a new clause, a new sentence, a
list item)? A line simply wrapping in the middle of a phrase is NOT a pause (大孩子的‖沙盒, 共享‖代码).
Items (id | phrase):
{rows}
Return ONLY JSON: {{"<id>": true|false, ...}}"""

summary = []
for vendor_path, final_path in [a.split("=") for a in sys.argv[1:]]:
    vend = _read_any(vendor_path, "zh-Hans")
    fin = _read_any(final_path, "zh-Hans")
    A, sa = chars(vend, True)
    B, sb = chars(fin, False)
    m = difflib.SequenceMatcher(None, A, B, autojunk=False)
    idx = {}
    for blk in m.get_matching_blocks():
        for d in range(blk.size):
            idx[blk.a + d] = blk.b + d
    cand = [
        idx[k]
        for k in range(len(A))
        if sa[k] and k in idx and k + 1 in idx and idx[k + 1] == idx[k] + 1 and not sb[idx[k]]
    ]
    keep = set()
    for a in range(0, len(cand), 60):
        chunk = cand[a : a + 60]
        rows = "\n".join(f"{j} | {B[max(0, j - 10):j + 1]}‖{B[j + 1:j + 11]}" for j in chunk)
        mm = re.search(r"\{.*\}", inv(P.format(rows=rows)) or "", re.S)
        got = json.loads(mm.group(0)) if mm else {}
        keep |= {j for j in chunk if got.get(str(j)) is True}
    # insert a space after each confirmed character (position counted over the char sequence of fin)
    cues = [Cue(c.start, c.end, list(c.lines)) for c in fin.cues]
    k = -1
    done = 0
    for c in cues:
        new_lines = []
        for ln in c.lines:
            out = ""
            for i, ch in enumerate(ln):
                out += ch
                if ch not in " 　，、；":
                    k += 1
                    if k in keep and i + 1 < len(ln) and ln[i + 1] not in " 　，、；":
                        out += " "
                        done += 1
            new_lines.append(out)
        orig = c.lines
        c.lines = new_lines
        if any(display_width(x, cjk=True) > spec.max_cpl for x in c.lines):
            t = " ".join(c.lines)
            nl = break_lines(t, spec)
            if len(nl) <= spec.max_lines and all(
                display_width(x, cjk=True) <= spec.max_cpl for x in nl
            ):
                c.lines = nl
            else:  # the space doesn't fit this cue: keep it as it was
                done -= sum(x.count(" ") for x in new_lines) - sum(x.count(" ") for x in orig)
                c.lines = orig
    out = Subtitle("zh-Hans", cues).renumber()
    w = write_deliverable(
        out,
        final_path,
        before=fin,
        text_change="restored clause-boundary spaces welded by a re-wrap",
    )
    summary.append((Path(final_path).name, len(cand), len(keep), done))
    print(
        f"{Path(final_path).name}: {len(cand)} candidate(s), {len(keep)} confirmed pause(s), {done} space(s) restored",
        flush=True,
    )
