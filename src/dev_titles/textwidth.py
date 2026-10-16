"""Display-width helpers.

CJK CPL limits are measured in display columns: full-width glyphs count as 1
(the Netflix CJK CPL numbers are in full-width units), half-width ASCII as 0.5.
For Latin languages we count Unicode code points (1 each).
"""

from __future__ import annotations

import unicodedata


def is_fullwidth(ch: str) -> bool:
    """True for East-Asian Wide/Fullwidth characters."""
    return unicodedata.east_asian_width(ch) in ("W", "F")


def display_width(text: str, cjk: bool) -> float:
    """Length of `text` in the units the per-language CPL limit uses.

    For CJK: full-width = 1.0, half-width (ASCII/latin) = 0.5.
    For Latin: 1.0 per code point (newlines excluded by callers).
    """
    if not cjk:
        return float(len(text))
    total = 0.0
    for ch in text:
        total += 1.0 if is_fullwidth(ch) else 0.5
    return total
