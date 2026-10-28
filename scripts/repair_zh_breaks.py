#!/usr/bin/env python3
"""Re-break zh lines that split a word, and move a word fragment split across two cues to the side
holding the rest of the word. Writes only through deliver.write_deliverable."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dev_titles.cli import _read_any
from dev_titles.config import spec_for
from dev_titles.deliver import write_deliverable
from dev_titles.model import Cue, Subtitle
from dev_titles.segment import break_lines
from dev_titles.segment.cjk import _word_break_positions, join_cue_lines_zh
from dev_titles.textwidth import display_width

spec = spec_for("zh-Hans")
CJK = lambda ch: "一" <= ch <= "鿿"


def fits(t):
    ls = break_lines(t, spec)
    return len(ls) <= spec.max_lines and all(display_width(x, cjk=True) <= spec.max_cpl for x in ls)


def mid(a, b):
    if not (a and b and CJK(a[-1]) and CJK(b[0])):
        return False
    ok = _word_break_positions(a[-6:] + b[:6], "zh-Hans")
    return bool(ok) and len(a[-6:]) not in ok


for path in sys.argv[1:]:
    p = Path(path)
    s = _read_any(str(p), "zh-Hans")
    before = _read_any(str(p), "zh-Hans")
    cues = [Cue(c.start, c.end, list(c.lines)) for c in s.cues]
    nl = nc = 0
    for k in range(len(cues) - 1):  # cross-cue fragments: move the shorter piece over
        a, b = join_cue_lines_zh(cues[k].lines), join_cue_lines_zh(cues[k + 1].lines)
        if not mid(a, b):
            continue
        ok = _word_break_positions(a[-6:] + b[:6], "zh-Hans")
        cut = len(a[-6:])
        lo = max(x for x in ok if x < cut)
        hi = min(x for x in ok if x > cut)
        left, right = cut - lo, hi - cut  # chars of the split word on each side
        if (
            left <= right and fits(a[:-left]) and fits(a[-left:] + b)
        ):  # move the left fragment forward
            a, b = a[:-left], a[-left:] + b
        elif fits(a + b[:right]) and fits(b[right:]):  # pull the right fragment back
            a, b = a + b[:right], b[right:]
        else:
            continue
        cues[k].lines, cues[k + 1].lines = break_lines(a, spec), break_lines(b, spec)
        nc += 1
    for c in cues:  # CN SG: no ，、； at a line end (the break is the pause)
        c.lines = [re.sub(r"[，、；]+$", "", x) for x in c.lines]
    out_cues = []
    ns = 0
    for c in cues:  # within-cue line breaks; a cue too dense for 2 word-safe lines is split in two
        if not any(mid(x, y) for x, y in zip(c.lines, c.lines[1:])):
            out_cues.append(c)
            continue
        t = join_cue_lines_zh(c.lines)
        new = break_lines(t, spec)
        if len(new) <= spec.max_lines and not any(mid(x, y) for x, y in zip(new, new[1:])):
            c.lines = new
            nl += 1
            out_cues.append(c)
            continue
        best = None
        from dev_titles.segment.cjk import _break_cjk

        raw = _break_cjk(t, spec, ())  # the break points BEFORE display stripping: ，、 still there
        offs, pos = [], 0  # where each raw line ends IN t (the breaker drops a space it breaks at)
        for ln in raw:
            pos = t.index(ln, pos) + len(ln)
            offs.append(pos)
        for j in range(1, len(raw)):  # split only at the segmenter's own (word-safe) line breaks
            a, b = t[: offs[j - 1]].strip(), t[offs[j - 1] :].strip()
            if fits(a) and fits(b):
                d = c.end - c.start
                ta = d * len(a) / len(a + b)
                if ta >= spec.min_dur and d - ta >= spec.min_dur:
                    score = abs(len(a) - len(b))
                    if best is None or score < best[0]:
                        best = (score, a, b, ta)
        if best is None:
            out_cues.append(c)
            continue
        _, a, b, ta = best
        out_cues += [
            Cue(c.start, c.start + ta, break_lines(a, spec)),
            Cue(c.start + ta, c.end, break_lines(b, spec)),
        ]
        ns += 1
    cues = out_cues
    out = Subtitle("zh-Hans", cues).renumber()
    PUN = set(" 　，、；。！？：")

    def chars_and_seps(sub, layout_breaks_need_word_check):
        """(characters without whitespace/，、；, sep_after flags). A line or cue break counts as a
        separation unless it falls inside a word (only checked on the INPUT side, where the
        mid-word breaks are the defect being repaired)."""
        cs, sep = [], []
        lines = [(ln, ci) for ci, c in enumerate(sub.cues) for ln in c.lines]
        for li, (ln, ci) in enumerate(lines):
            for ch in ln:
                if ch in " 　，、；":
                    if sep:
                        sep[-1] = True
                else:
                    cs.append(ch)
                    sep.append(ch in "。！？：")
            if sep and li + 1 < len(lines):
                nxt = lines[li + 1][0]
                if layout_breaks_need_word_check and nxt and mid(ln, nxt):
                    pass
                else:
                    sep[-1] = True
        return "".join(cs), sep

    A, sa = chars_and_seps(before, True)
    B, sb = chars_and_seps(out, False)
    assert A == B, f"{p}: text changed"
    welds = [A[max(0, k - 6) : k + 7] for k in range(len(A)) if sa[k] and not sb[k]]
    if welds:
        for x in welds[:5]:
            print("   WELD:", x)
        raise SystemExit(f"{p}: {len(welds)} pause(s) welded shut")
    w = write_deliverable(
        out,
        p,
        before=before,
        text_change="line-break repair: a ，/、 at a line break dropped (CN SG)",
    )
    print(
        f"{p.name}: re-broke {nl} cue(s), split {ns} over-dense cue(s), moved {nc} cross-cue fragment(s) -> {len(w)} files"
    )
