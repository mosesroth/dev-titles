"""CJK-aware line breaking, kinsoku shori, and ZH punctuation normalization.

Pipeline step [4]: after translation we re-derive display lines to the target
language's CPL, never reusing English line breaks. This module handles the
line-break/punctuation layer. Cue *timing* re-segmentation (splitting a long
translated string across multiple timed cues to satisfy cps) belongs to the
pipeline, which uses `display_width` + the LangSpec; the helpers here operate on
the text of a single cue.
"""

from __future__ import annotations

import re

from ..config import (
    KINSOKU_NO_LINE_END,
    KINSOKU_NO_LINE_START,
    ZH_CUE_BOUNDARY_STRIP,
    LangSpec,
)
from ..linguistics import comma_punct, connectors, end_punct
from ..textwidth import display_width


def break_lines(text: str, spec: LangSpec, keep_together=()) -> list[str]:
    """Break `text` into <= spec.max_lines lines within spec.max_cpl.

    For CJK, breaks may fall between any two characters (then adjusted by
    kinsoku). For Latin, breaks only at spaces. Returns the display lines; the
    caller is responsible for handling text that cannot fit (it should have been
    split into multiple cues by cps re-segmentation first).
    """
    # NEVER truncate: return all wrapped lines. Capping to max_lines here would
    # silently DROP content (a data-loss bug). If a single cue exceeds max_lines,
    # the validator flags it (max_lines) — the caller should have split it into
    # more cues (resegment_translation does), not lose text.
    return wrap_lines(text, spec, keep_together)


def wrap_lines(text: str, spec: LangSpec, keep_together=()) -> list[str]:
    """Break `text` into as many CPL-bounded display lines as needed (no max-lines
    cap), boundary-aware + kinsoku for CJK. `break_lines` is this capped to
    spec.max_lines; the re-segmenter uses the uncapped form to fill multiple cues.

    `keep_together` = terms (glossary targets / DNT) never broken mid-term."""
    text = text.strip()
    if not text:
        return []
    if spec.cjk:
        # KINSOKU RUNS LAST: _cap_widths can introduce NEW kinsoku violations when it width-splits
        # a line (a lone '，' line, a line starting with a small katakana, a line ending with '「'),
        # so re-apply kinsoku after it. apply_kinsoku is width-guarded (it only pulls a char up if
        # it still fits max_cpl) and idempotent, so this can't re-overflow.
        lines = apply_kinsoku(
            _cap_widths(apply_kinsoku(_break_cjk(text, spec, keep_together), spec), spec), spec
        )
        # trim stray ASCII/full-width space at line edges (JA uses U+3000 as an
        # internal phrase separator, but never at a line start/end)
        lines = [s for ln in lines if (s := ln.strip(" 　"))]
        if spec.lang.startswith("zh"):
            # CN subtitle SG: no punctuation at the end of a line (the break is the pause) —
            # human-reviewed zh ends 0.8% of lines with ，, ours ended ~4.5%.
            lines = [s for ln in lines if (s := ln.strip("，、；"))]  # nor at line START (kinsoku)
        return lines
    return _break_latin(text, spec)


def _cap_widths(lines: list[str], spec: LangSpec) -> list[str]:
    """Safety net: guarantee no line exceeds max_cpl (split any that does)."""
    out: list[str] = []
    for ln in lines:
        while display_width(ln, cjk=spec.cjk) > spec.max_cpl:
            k = 0
            for j in range(1, len(ln) + 1):
                if display_width(ln[:j], cjk=spec.cjk) <= spec.max_cpl:
                    k = j
                else:
                    break
            # Don't split inside a Latin word: back up to its start. If the word
            # starts the line and alone exceeds the cap (a long code identifier
            # like performMountingInstructions), let the line OVERFLOW with the
            # identifier whole rather than shatter it (YouTube soft-wraps; a broken
            # identifier is a hard error, an over-long line is cosmetic).
            if 0 < k < len(ln) and _is_wordchar(ln[k - 1]) and _is_wordchar(ln[k]):
                a = k
                while a > 0 and _is_wordchar(ln[a - 1]):
                    a -= 1
                if a > 0:
                    k = a
                else:
                    e = k
                    while e < len(ln) and _is_wordchar(ln[e]):
                        e += 1
                    k = e  # emit the whole over-long identifier, then continue
            # Same guard for a KATAKANA run (one loanword): never split inside it.
            elif 0 < k < len(ln) and _is_katakana(ln[k - 1]) and _is_katakana(ln[k]):
                a = k
                while a > 0 and _is_katakana(ln[a - 1]):
                    a -= 1
                if a > 0:
                    k = a
                else:
                    e = k
                    while e < len(ln) and _is_katakana(ln[e]):
                        e += 1
                    k = e  # emit the whole over-long loanword rather than shatter it
            # Never separate a NUMBER from its CJK counter/unit (30|年, 5|つ, 10|%): back up to
            # before the digit run so '30年' stays whole.
            elif 0 < k < len(ln) and ln[k - 1].isdigit() and not ln[k].isascii():
                a = k
                while a > 0 and (ln[a - 1].isdigit() or ln[a - 1] in ",."):
                    a -= 1
                if a > 0:
                    k = a
            out.append(ln[:k].rstrip())
            ln = ln[k:].lstrip()
        if ln:
            out.append(ln)
    return out


# Words that must not END a line (they bind to what follows): break BEFORE them,
# not after. Keeps article+noun, prepositional phrases, and conjunctions intact
# per DCMP/BBC line-break rules ("do not split ... don't break after a conjunction").
_NO_BREAK_AFTER = frozenset(
    (
        "a an the this that these those my your his her its our their "
        "and or but nor so yet for as if of to in on at by with from into onto "
        "over under about than then is are was were be been to"
    ).split()
)


def _break_latin(text: str, spec: LangSpec) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        candidate = " ".join(cur + [w])
        if display_width(candidate, cjk=False) <= spec.max_cpl or not cur:
            cur.append(w)
        else:
            # Don't strand a function word at the line end: pull trailing
            # no-break-after words down to start the next line with `w`.
            k = len(cur)
            while k > 1 and cur[k - 1].lower().strip(".,;:") in _NO_BREAK_AFTER:
                k -= 1
            lines.append(" ".join(cur[:k]))
            cur = cur[k:] + [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


_TAGGER = None


def _word_break_positions(text: str, lang: str) -> set[int] | None:
    """Char indices where a line break is allowed (word/morpheme boundaries),
    from a morphological tokenizer — fugashi (ja) / jieba (zh). Returns None if
    the tokenizer is unavailable or its output can't be aligned to `text`, so the
    caller falls back to the script-boundary heuristic. Never breaks mid-word."""
    global _TAGGER
    try:
        if lang.startswith("ja"):
            import fugashi

            if _TAGGER is None:
                _TAGGER = fugashi.Tagger()
            words = list(_TAGGER(text))
            return _ja_bunsetsu_positions(text, words)
        elif lang.startswith("zh"):
            import jieba

            _zh_vocab(jieba)
            surfaces = list(jieba.cut(text))
        else:
            return None
    except Exception:
        return None
    pos = {0, len(text)}
    idx = 0
    for s in surfaces:
        if not s:
            continue
        j = text.find(s, idx)
        if j < 0:
            return None  # misaligned -> fall back
        pos.add(j)
        pos.add(j + len(s))
        idx = j + len(s)
    return pos


# 文節-clitic POS: a line break must NOT fall right before these — they bind to
# the preceding content word (else 「ます」/particles get stranded on their own
# line or cue, which a native reviewer flags immediately).
_JA_CLITIC_POS = ("助詞", "助動詞", "接尾辞", "補助記号", "記号")


# Tech words jieba's default dictionary splits (来自后|端系统): never break inside them.
_ZH_WORDS = (
    "后端 前端 全栈 智能体 提示词 可观测性 可扩展性 工作流 数据库 服务器 客户端 服务端 云端 端到端 "
    "大模型 大语言模型 多模态 微服务 开源 代码库 代码审查 开发者 开发板 测试用例 部署 回滚 延迟 "
    "并发 吞吐量 用户体验 用户界面 命令行 终端 插件 框架 组件 模块 接口 应用商店 设备端 电视端"
).split()
_ZH_READY = False


def _zh_vocab(jieba) -> None:
    global _ZH_READY
    if not _ZH_READY:
        for w in _ZH_WORDS:
            jieba.add_word(w, freq=100000)
        _ZH_READY = True


def _ja_bunsetsu_positions(text: str, words) -> set[int] | None:
    """Allowed ja break indices at 文節 boundaries: the start of a token is a
    legal break ONLY if that token is not a particle/auxiliary/suffix (which must
    stay attached to what precedes). Keeps 「〜します」 whole and never breaks
    before は/が/を/に…; falls back (None) if surfaces can't be aligned."""
    pos = {0, len(text)}
    idx = 0
    prev_binds_fwd = False  # previous token binds to what follows (no break after it)
    for w in words:
        s = w.surface
        if not s:
            continue
        j = text.find(s, idx)
        if j < 0:
            return None
        feat = getattr(w, "feature", None)
        pos1 = getattr(feat, "pos1", None) or ""
        pos2 = getattr(feat, "pos2", None) or ""
        is_noun = pos1.startswith("名詞")
        # A break before this token is forbidden when: it's a clitic (particle/
        # auxiliary/suffix — would strand it); it's a dependent verb like あり in
        # 必要がある / する in 設定する (非自立可能 — would strand the 文節 before it);
        # or the previous token binds forward (a 接頭辞 prefix, or a single-kanji
        # noun forming a compound like 親|画面 → keep 親画面 whole).
        clitic = any(pos1.startswith(p) for p in _JA_CLITIC_POS)
        dep_verb = pos1.startswith("動詞") and pos2.startswith("非自立")
        if j > 0 and not (clitic or dep_verb or prev_binds_fwd):
            pos.add(j)  # break allowed before an independent content word only
        prev_binds_fwd = pos1.startswith("接頭") or (is_noun and len(s) == 1)
        idx = j + len(s)
    return pos


def _break_cjk(text: str, spec: LangSpec, keep_together=()) -> list[str]:
    """Fill to max_cpl (display-width units), breaking only at word/morpheme
    boundaries (via tokenizer) so CJK words are never split mid-word; falls back
    to the script-boundary heuristic if no tokenizer is available."""
    bounds = _word_break_positions(text, spec.lang)
    prot = _protected_spans(text, keep_together)
    if bounds:
        lines: list[str] = []
        start, n = 0, len(text)
        while start < n:
            fit = start
            for e in range(start + 1, n + 1):
                if display_width(text[start:e], cjk=True) <= spec.max_cpl:
                    fit = e
                else:
                    break
            if fit >= n:
                lines.append(text[start:])
                break
            cand = [b for b in bounds if start < b <= fit]
            # Prefer breaks that don't fall inside a multi-word Latin term; only
            # break a run internally if there is no other candidate in the window
            # (then it splits BETWEEN whole tokens, e.g. a long identifier list).
            safe = [b for b in cand if not _interior_of_run(b, prot)]
            cand = safe or cand
            # Choose the break a subtitler would: at a sentence end, else a clause boundary,
            # else the fullest line — and, when the rest fits on one more line, balanced lines.
            # (Greedy max-fill broke mid-phrase and stranded sentence ends mid-line; that was
            # most of a native reviewer's hand edits in a real review batch.)
            cut = max(cand, key=lambda b: _cut_score(text, start, b, spec)) if cand else fit
            # Never cut inside a Latin token (a tokenizer may split code
            # identifiers like performMountingInstructions / Systrace.capture into
            # sub-morphemes and offer a boundary mid-identifier). Snap back to the
            # token start; if that would empty the line, force overflow to the
            # token END instead of shattering the identifier.
            if 0 < cut < n and _is_wordchar(text[cut - 1]) and _is_wordchar(text[cut]):
                a = cut
                while a > start and _is_wordchar(text[a - 1]):
                    a -= 1
                if a > start:
                    cut = a
                else:
                    b = cut
                    while b < n and _is_wordchar(text[b]):
                        b += 1
                    cut = b
            # Never cut inside a KATAKANA run either (one loanword): プラットフォーム must not
            # become 'プラットフ' + 'ォーム' (which also orphans the small ォ at a line/cue start).
            elif 0 < cut < n and _is_katakana(text[cut - 1]) and _is_katakana(text[cut]):
                a = cut
                while a > start and _is_katakana(text[a - 1]):
                    a -= 1
                if a > start:
                    cut = a
                else:
                    b = cut
                    while b < n and _is_katakana(text[b]):
                        b += 1
                    cut = b  # overflow with the whole loanword rather than shatter it
            # Never separate a NUMBER from its CJK counter/unit (30|年, 5|つ): keep '30年' whole.
            elif 0 < cut < n and text[cut - 1].isdigit() and not text[cut].isascii():
                a = cut
                while a > start and (text[a - 1].isdigit() or text[a - 1] in ",."):
                    a -= 1
                if a > start:
                    cut = a
            lines.append(text[start:cut])
            start = cut
            while start < n and text[start] in " 　":
                start += 1
        return [ln for ln in lines if ln]
    return _break_cjk_heuristic(text, spec, keep_together)


_SENT_MARKS = set("。！？!?")
_CLAUSE_MARKS = set("、，,；;：:")


def _latin(ch: str) -> bool:
    return bool(ch) and ch.isascii() and ch.isalnum()


def boundary_kind(text: str, b: int, lang: str) -> int:
    """2 = sentence boundary, 1 = clause boundary, 0 = neither, for a break at index `b`.
    House punctuation: ja writes 。 as a full-width space and 、 as a half-width space; zh writes
    a mid-line sentence break as a half-width space. A space between two Latin words
    ("React Native") is never a boundary."""
    prev = text[b - 1] if b > 0 else ""
    nxt = text[b] if b < len(text) else ""
    if b >= len(text) or prev in _SENT_MARKS or "　" in (prev, nxt):
        return 2
    if " " in (prev, nxt):
        j = b - 1 if prev == " " else b
        left = text[j - 1] if j > 0 else ""
        right = text[j + 1] if j + 1 < len(text) else ""
        if _latin(left) and _latin(right):
            return 0
        return 2 if lang.startswith("zh") else 1
    if prev in _CLAUSE_MARKS:
        return 1
    return 0


_ZH_NO_LINE_START = set("的得地了着过们吗呢吧啊么")


def _cut_score(text: str, start: int, b: int, spec: LangSpec) -> float:
    w1 = display_width(text[start:b], cjk=True)
    fill = w1 / spec.max_cpl
    score = fill + {2: 0.6, 1: 0.35, 0: 0.0}[boundary_kind(text, b, spec.lang)]
    if spec.lang.startswith("zh") and b < len(text) and text[b] in _ZH_NO_LINE_START:
        score -= 1.5  # a zh line never starts with a particle (明年|的路线图)
    if fill < 0.45:
        score -= 1.0  # no stubby lines
    w2 = display_width(text[b:].strip(" 　"), cjk=True)
    if w2 <= spec.max_cpl:  # the rest fits on one more line: prefer balanced lines
        score -= 0.5 * abs(w1 - w2) / spec.max_cpl
    return score


def _break_cjk_heuristic(text: str, spec: LangSpec, keep_together=()) -> list[str]:
    """Fallback (no tokenizer): break at the best grammatical/script boundary.

    Within the run that fits on a line, prefer to break right after the last
    sentence-final mark, else the last comma/pause mark, else just before the
    last connector word, else hard-wrap at the cap. Boundary maps come from
    `linguistics` (ported from MX Player's CJK segmenter)."""
    ends = end_punct(spec.lang)
    commas = comma_punct(spec.lang)
    conns = connectors(spec.lang)

    lines: list[str] = []
    rest = text
    while rest:
        # How many leading chars fit within the cap?
        fit = 0
        for i, _ in enumerate(rest):
            if display_width(rest[: i + 1], cjk=True) <= spec.max_cpl:
                fit = i + 1
            else:
                break
        if fit >= len(rest):
            lines.append(rest)
            break
        window = rest[:fit]
        cut = _best_cjk_cut(window, ends, commas, conns)
        # Never split inside a Latin token (e.g. "Nimbus") — snap the cut back to
        # the start of the word that straddles it, unless that word starts the
        # line (then it's longer than the cap; _cap_widths will hard-split it).
        if 0 < cut < len(rest) and _is_wordchar(rest[cut - 1]) and _is_wordchar(rest[cut]):
            a = cut
            while a > 0 and _is_wordchar(rest[a - 1]):
                a -= 1
            if a > 0:
                cut = a
        # Don't split a multi-word Latin term ("React Native") — snap back to the
        # run's start if the cut lands inside it (keeps the whole term on the next
        # line); if the run starts the line, leave it for _cap_widths to overflow.
        prot = _protected_spans(rest, keep_together)
        if 0 < cut < len(rest) and _interior_of_run(cut, prot):
            a = next(s for s, e in prot if s < cut < e)
            if a > 0:
                cut = a
        lines.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    return lines


def _is_wordchar(ch: str) -> bool:
    """Part of an atomic Latin token (don't break inside these).

    Includes the APOSTROPHE (' and the curly ’): without it a line break could fall inside a
    contraction or possessive ("Won'" / "t", "Bob'" / "s List"), splitting one word across two
    display lines."""
    return ch.isascii() and (ch.isalnum() or ch in "+.#-_/@'") or ch == "’"


def _is_katakana(ch: str) -> bool:
    """Katakana (incl. small vowels and the ー prolonged mark): a katakana run is ONE loanword, so a
    width split inside it is a hard error (プラットフォーム -> 'プラットフ' + 'ォーム' orphans the
    small ォ at a line start). Treat such a run as atomic, like a Latin token.

    EXCLUDES the separators ・ (U+30FB middle dot) and ゠ (U+30A0): those DELIMIT items in a list
    (マーケティングチーム・コンテンツチーム・法務チーム) and are valid break points — treating them
    as word-internal made a long list unsplittable and overflowed max_cpl."""
    return (("゠" <= ch <= "ヿ") and ch not in "・゠") or ch == "ｰ"


# A run of Latin tokens joined by single ASCII spaces (e.g. "React Native",
# "Apple TV", "Concurrent Rendering", "React 18"). CJK subtitle text has no
# internal spaces, so a space-joined Latin run is almost always ONE multi-word
# term / product name that a native reviewer expects to stay together — breaking
# it ("React"|"Native", "Apple"|"TV") reads as machine translation (native-reviewer feedback).
_LATIN_TOKEN = r"[A-Za-z0-9][A-Za-z0-9.+#/_@-]*"
_LATIN_RUN = re.compile(rf"{_LATIN_TOKEN}(?: {_LATIN_TOKEN})*")


def _protected_spans(text: str, extra=()) -> list[tuple[int, int]]:
    """(start, end) spans whose interior we avoid breaking: multi-word Latin runs,
    plus every occurrence of an `extra` keep-together term (glossary targets such
    as 開発者ポータル, so a localized multi-token term isn't split across a cue)."""
    spans = [(m.start(), m.end()) for m in _LATIN_RUN.finditer(text) if " " in m.group(0)]
    for term in extra:
        if not term or len(term) < 2:
            continue
        i = text.find(term)
        while i != -1:
            spans.append((i, i + len(term)))
            i = text.find(term, i + 1)
    return spans


def _interior_of_run(i: int, spans: list[tuple[int, int]]) -> bool:
    """True if index `i` is strictly inside a protected Latin run (breaking here
    would split a multi-word term). A break AT a run edge is fine."""
    return any(s < i < e for s, e in spans)


def _best_cjk_cut(window: str, ends: set[str], commas: set[str], conns: set[str]) -> int:
    """Index (within `window`) at which to break: after the last end mark, else
    after the last comma, else before the last connector, else end of window."""
    # After the last sentence-final mark.
    for i in range(len(window) - 1, -1, -1):
        if window[i] in ends:
            return i + 1
    # After the last comma/pause mark.
    for i in range(len(window) - 1, -1, -1):
        if window[i] in commas:
            return i + 1
    # Before the last connector token that starts within the window.
    for conn in conns:
        pos = window.rfind(conn)
        if pos > 0:
            return pos
    # Else break at the last script boundary (kanji/hiragana/katakana/latin
    # transition) — avoids splitting inside a word like すべて without needing a
    # Japanese morphological tokenizer.
    for i in range(len(window) - 1, 0, -1):
        if _script_class(window[i - 1]) != _script_class(window[i]):
            return i
    return len(window)


def _script_class(ch: str) -> str:
    """Coarse writing-system class for JP/CJK line-break boundary preference."""
    o = ord(ch)
    if 0x3040 <= o <= 0x309F:
        return "hira"
    if 0x30A0 <= o <= 0x30FF or o == 0x30FC:
        return "kata"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
        return "kanji"
    if ch.isascii() and ch.isalnum():
        return "latin"
    return "other"


def apply_kinsoku(lines: list[str], spec: LangSpec | None = None) -> list[str]:
    """Adjust breaks so no line starts with a 行頭禁則 char and no line ends
    with a 行末禁則 char, by pushing/pulling one character across the break.

    When `spec` is given, a hanging punctuation is pulled onto the previous line
    only if it still fits max_cpl — so kinsoku never overflows the line width
    (the earlier off-by-one). If it can't fit, the char is left in place (a rare,
    lower-severity kinsoku miss is preferable to a width violation).

    Iterates until stable or a small bound is hit (prevents pathological loops).
    """
    if len(lines) < 2:
        return lines
    cpl = spec.max_cpl if spec else None
    lines = list(lines)
    for _ in range(len(lines) * 4):
        changed = False
        for i in range(len(lines) - 1):
            cur, nxt = lines[i], lines[i + 1]
            if not cur or not nxt:
                continue
            # Forbidden char at start of the next line -> pull it up if it fits,
            # else swap boundary chars (push cur's last char down and pull the
            # punctuation up) so it neither starts a line nor overflows width.
            if nxt[0] in KINSOKU_NO_LINE_START:
                if cpl is None or display_width(cur + nxt[0], cjk=True) <= cpl:
                    lines[i], lines[i + 1] = cur + nxt[0], nxt[1:]
                    changed = True
                else:
                    # Can't pull the punct up without overflow. Prefer to move a
                    # whole trailing Latin word down (never split it); else do a
                    # single-char swap only when the last char isn't a wordchar.
                    k = len(cur)
                    word = _is_katakana if cur and _is_katakana(cur[-1]) else _is_wordchar
                    while k > 0 and word(cur[k - 1]):  # whole trailing Latin OR katakana word
                        k -= 1
                    if 0 < k < len(cur):
                        lines[i], lines[i + 1] = cur[:k].rstrip(), cur[k:] + nxt
                        changed = True
                    elif len(cur) > 1 and not _is_wordchar(cur[-1]) and not _is_katakana(cur[-1]):
                        # push cur's last char down (oidashi) — ORDER PRESERVED;
                        # the forbidden punct is no longer at the line start.
                        # (The old cur[:-1]+nxt[0], cur[-1]+nxt[1:] form reordered
                        # characters and corrupted text, e.g. します。-> し。/ます.)
                        lines[i], lines[i + 1] = cur[:-1], cur[-1] + nxt
                        changed = True
                    # else: leave the punct (rare; better than splitting a word)
                continue
            # Forbidden char at end of the current line -> push it down.
            if cur[-1] in KINSOKU_NO_LINE_END:
                lines[i], lines[i + 1] = cur[:-1], cur[-1] + nxt
                changed = True
                continue
        lines = [ln for ln in lines if ln]
        if not changed:
            break
    # FINAL: never leave a line that is PURELY punctuation (a re-wrap orphan, e.g. a lone '、'
    # the width-guarded pull above couldn't fold). A trailing orphan appends to the previous line
    # (punctuation is fine at a line END); a leading orphan is DROPPED — it's a cue-boundary comma
    # orphaned from the prior clause (no content), and folding it forward would just re-start the
    # line with forbidden punctuation. Guarantees no kinsoku_line_start on a lone-punct line.
    if len(lines) >= 2:

        def _all_punct(s: str) -> bool:
            s = s.strip()
            return bool(s) and all(ch in KINSOKU_NO_LINE_START for ch in s)

        folded: list[str] = []
        for ln in lines:
            if _all_punct(ln):
                if folded:
                    folded[-1] = folded[-1] + ln.strip()  # trailing/middle orphan -> prev line end
                # else: leading orphan -> drop
            else:
                folded.append(ln)
        lines = folded or lines
    return lines


def cjk_myriad_numbers(text: str, ten_thousand: str = "万", hundred_million: str = "億") -> str:
    """Large counts in the myriad form, for BOTH ja and zh: the JP subtitle style guide's '1 million USD -> 100万米ドル',
    the CN subtitle style guide's '13.5 million feet -> 1350万英尺'; human reviewers in both languages write 5万人 /
    13万 / 500億 / 1,000亿 and rewrite our 50,000 / 130,000. Converts an arabic number that is an exact
    multiple of 10^4 / 10^8 when followed by a CJK counter/noun (人/件/トークン/个…) — never before a
    Latin unit (10,000ms stays) and never a number that is not a round multiple (12,345 stays)."""

    def repl(m):
        v = int(m.group(1).replace(",", ""))
        if v >= 10**8 and v % 10**8 == 0:
            return f"{v // 10**8:,}{hundred_million}"
        if v >= 10**4 and v % 10**4 == 0:
            return f"{v // 10**4:,}{ten_thousand}"
        return m.group(1)

    return re.sub(
        r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d{5,})(?![\d.,])(?=\s*[ぁ-んァ-ヶ一-龯])", repl, text
    )


def ja_myriad_numbers(text: str) -> str:
    return cjk_myriad_numbers(text, "万", "億")


def normalize_cjk_numbers(text: str) -> str:
    """ZH number policy (CN subtitle style guide: 'comma as thousand separator'): large counts use arabic + comma
    separator, NOT the myriad 万/亿 form. (JA is the opposite — see ja_myriad_numbers.)
    Convert '13万'->'130,000', '1亿'->'100,000,000'. Only fires on an ARABIC-digit stem —
    kanji 百万/数万/several-万 are left alone. Idempotent."""

    def repl(stem: str, mult: int) -> str:
        v = float(stem) * mult
        return f"{int(v):,}" if v == int(v) else f"{v:,}"

    text = re.sub(r"(\d+(?:\.\d+)?)\s*亿", lambda m: repl(m.group(1), 10**8), text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*万", lambda m: repl(m.group(1), 10**4), text)
    return text


_KANJI_DIGIT = {k: i + 1 for i, k in enumerate("一二三四五六七八九")}


def _short_year(text: str) -> str:
    """English 2-digit-year apostrophe ('24 / ’24) has no place in CJK: '24年 -> 24年 (a native
    native reviewer's correction)."""
    return re.sub(r"[‘’'′](\d{2})(?=\s*年)", r"\1", text)


def normalize_ja_punct(text: str) -> str:
    """Japanese YouTube-caption punctuation, per the JP subtitle style guide §(3) and confirmed by a
    native localization reviewer ("all of the past YouTube captions follow the rule"):
      * 読点 、 -> half-width space;  句点 。 -> full-width space.
        (REVERSES the earlier "drop them" rule, which was based on a noise-space audit.)
      * ？ is omitted where the line is clearly a question — i.e. after an explicit question ending
        (か / かな / っけ) — and becomes a sentence break; kept elsewhere ("本当？") where it carries
        the question.
      * ！ is omitted ("unnatural in Japanese") and becomes a sentence break.
    Keep （）「」・. Space runs collapse; edges are trimmed by the wrapper."""
    # A 、/。 BETWEEN two Latin word-chars is an enumeration separator (the model
    # writes "nextFocusUp、nextFocusDown"); replace it with a space so the
    # identifiers don't fuse into "nextFocusUpnextFocusDown".
    text = re.sub(r"(?<=[0-9A-Za-z])[、。](?=[0-9A-Za-z])", " ", text)
    text = _short_year(text)
    # Clear questions lose ？; exclamations lose ！ (ASCII '!' only when not part of code like !==).
    text = re.sub(r"(か|かな|っけ)[？?]", r"\1。", text)
    text = text.replace("！", "。")
    text = re.sub(r"(?<![A-Za-z0-9=!])!(?![=A-Za-z0-9])", "。", text)
    # 読点 -> half-width space, 句点 -> full-width space
    text = text.replace("、", " ").replace("。", "　")
    text = re.sub(r"[ 　]*　[ 　]*", "　", text)  # a run containing a full-width space -> one
    text = re.sub(r"[ 　]+(?=[」』）\)])", "", text)  # no space before a closing bracket
    text = ja_myriad_numbers(
        text
    )  # 50,000人 -> 5万人 (JP subtitle style guide: 1 million USD -> 100万米ドル)
    # ARABIC COUNTER (JP style guide §1.3-1.4, confirmed by a set of reviewed files: 1つ 27 / 一つ 0,
    # 2つ 4 / 二つ 0). NARROW on purpose — only the 〜つ counter. Kanji numerals stay inside words
    # (一部 4 / 1部 0, 一緒, 一方), so a blanket 一 -> 1 would corrupt them.
    text = re.sub(r"([一二三四五六七八九])つ", lambda m: f"{_KANJI_DIGIT[m.group(1)]}つ", text)
    # KANJI over hiragana (JP subtitle style guide: "字幕翻訳では文字数の節約と視認性を重視して漢字を
    # 使用する", e.g. さまざま->様々, あらかじめ->予め, あいまい->曖昧). 私たち->私達 follows the same
    # principle and the reviewer majority (6 reviewed videos use 私達).
    for _h, _k in (
        ("さまざま", "様々"),
        ("あらかじめ", "予め"),
        ("あいまい", "曖昧"),
        ("私たち", "私達"),
        # cross-reviewer counts (files preferring each form, across 26 reviewed files):
        ("たとえば", "例えば"),  # 21 : 2
        ("ヶ月", "か月"),  # 14 : 0 (also ヵ月)
        ("ヵ月", "か月"),
    ):
        text = text.replace(_h, _k)
    # 'our' as 我々の/私共の is a stiff corporate-possessive calque a native subtitler omits.
    # 当社の / 弊社の are NOT stripped any more: the native reviewers USE 当社の (7 occurrences:
    # 当社のYouTubeチャンネル, これが当社のアプリです, 当社の製品…) where the speaker means "our
    # company's X". Stripping it deleted content the reviewers had deliberately written.
    text = re.sub(r"(我々|私共|わたし共)の", "", text)
    text = re.sub(r" {2,}", " ", text).strip(" 　")  # single spaces; trim edges
    return text


def normalize_zh_punct(text: str, at_cue_boundary: bool = True) -> str:
    """ZH cue-text normalization: strip a single trailing ，/。 at a cue boundary
    (replaced by nothing — the cue break itself carries the pause) and collapse
    ASCII ellipsis to U+2026. Doubled terminal punctuation is left for the
    validator to flag (it may indicate a real translation issue)."""
    text = text.replace("...", "…")
    text = _short_year(text)
    # House register (formal AND conversational, per the native-reviewer verdict):
    # informal 你 -> 您. The model slips into 你 for possessives (你的应用) despite the prompt
    # directive — prose instruction under-delivers, so enforce deterministically.
    # EXCLUSIONS: 迷你 (mini) and similar compounds where 你 isn't the pronoun; 你好 (greeting);
    # and the PLURAL 你们 — the reviewers kept 你们 because 您们 is unidiomatic.
    text = re.sub(r"(?<![迷])你(?!们)(?!好(?=[，,。！!？?」』\s]|$))", "您", text)
    # (Latin、Latin is a LIST separator in zh — 'iOS、Android、Expo' — and stays. It used to be
    # turned into a space, which read as one run-on name: 'iOS Android Expo Web Apple TV'.
    # An interior 。 between Latin tokens becomes ， via the rule below, so idents stay apart.)
    # 句号: the cue/line break ends a sentence, so a FINAL 。 goes. An INTERIOR 。 (two sentences in
    # one record) becomes a SENTENCE-BREAK SPACE — zh reviewers' own convention (40–110 mid-line
    # spaces per video in a vendor-delivered set, e.g. 「…把控好这份自由 我在VS Code里…」, never a mid-line 。).
    # Deleting it welded sentences into run-ons (开发者模式为此请打开…); a comma would be a splice.
    # \x00 marks the break so the Han<->Latin space-stripping below can't eat it.
    text = re.sub(r"。+(?=[”」』）)]|\s*$)", "", text)
    text = re.sub(r"\s*。+\s*", "\x00", text)
    # Remove spaces at a Han<->Latin boundary (keep spaces between two Latin words).
    text = re.sub(r"(?<=[一-鿿])[ 　]+(?=[0-9A-Za-z])", "", text)
    text = re.sub(r"(?<=[0-9A-Za-z])[ 　]+(?=[一-鿿])", "", text)
    # A space BETWEEN two Han chars is a sentence separator (zh reviewer convention, and how the model
    # writes subtitle-style 'A 不过B'); deleting it welded clauses (…进行本次演示不过如果…). Keep it.
    text = re.sub(r"(?<=[一-鿿])[ 　]+(?=[一-鿿])", "\x00", text)
    text = re.sub(r"[ 　]+(?=[一-鿿])", "", text)  # space before a Han char
    text = re.sub(r"(?<=[一-鿿])[ 　]+", "", text)  # space after a Han char
    # Large counts in the myriad form: the CN subtitle style guide's own example renders 13.5 million as 1350万, and the zh
    # reviewers write 13万 / 500亿 (18 vs 1 comma form across 15 files; e.g. 130,000 -> 13万).
    # The style guide's "comma as thousand separator" applies to numbers that stay in digits (2,000).
    text = cjk_myriad_numbers(text, ten_thousand="万", hundred_million="亿")
    # Interior questions keep ？ (zh reviewers added 41 in 6 videos: 真的吗？哦… / …对吧？我们…);
    # restore_question_mark only handles the END of a segment.
    text = re.sub(
        r"(吗|对吧|是吧)(?:\x00|\s)+(?=[^\s])", r"\1？", text
    )  # ？ replaces the break space
    text = re.sub(r"(吗|对吧|是吧)(?=[^？?！!，,。、；;：:”」』\s)\x00]|$)", r"\1？", text)
    text = re.sub(r"\x00+", " ", text)
    # ASCII double quotes -> Chinese curly quotes, paired (reviewers: 0 ASCII, 24 curly).
    _q = iter(range(10**6))
    text = re.sub(r'"', lambda m: "“" if next(_q) % 2 == 0 else "”", text)
    text = re.sub(r" {2,}", " ", text).strip()
    if at_cue_boundary and text and text[-1] in ZH_CUE_BOUNDARY_STRIP:
        text = text[:-1]
    return text
