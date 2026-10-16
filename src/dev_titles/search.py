"""Web search for automatic doc-grounding, via DuckDuckGo (no key, no signup).

Used to FIND product/feature documentation URLs from a video's subject so the vocab
pre-ingest can harvest correct product/brand spellings (fixing OOV ASR mishears like
'Kubernetes'->'Cuber Netties') without a human passing --docs-url.

DuckDuckGo via the `ddgs` package — zero-setup, good enough at our volume (~1-2
searches/video). Best-effort by design: returns [] when `ddgs` isn't installed or on
any error, so the caller degrades to the glossary-only path instead of failing.
Install with `pip install ddgs` (or `pip install -e '.[search]'`).
"""

from __future__ import annotations


def _duckduckgo(query: str, k: int, timeout: float):
    """DuckDuckGo via the `ddgs` (or legacy `duckduckgo_search`) package. Returns None
    when the package isn't installed (caller degrades), else a list of result URLs."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore[no-redef]
        except ImportError:
            return None
    with DDGS() as ddgs:
        return [r["href"] for r in ddgs.text(query, max_results=k) if r.get("href")][:k]


def web_search(query: str, *, k: int = 5, timeout: float = 15.0) -> list[str]:
    """Top result URLs for `query` via DuckDuckGo. Returns [] if `ddgs` isn't installed or
    on any error (best-effort — the caller treats an empty result as 'no docs found')."""
    if not query or not query.strip():
        return []
    try:
        return _duckduckgo(query, k, timeout) or []
    except Exception:
        return []
