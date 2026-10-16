"""Tests for the DuckDuckGo-backed web search used for auto doc-grounding."""

from dev_titles import search


def test_returns_ddg_hits(monkeypatch):
    monkeypatch.setattr(search, "_duckduckgo", lambda q, k, t: ["https://ddg/1", "https://ddg/2"])
    assert search.web_search("penny docs", k=2) == ["https://ddg/1", "https://ddg/2"]


def test_empty_when_ddgs_not_installed(monkeypatch):
    monkeypatch.setattr(search, "_duckduckgo", lambda q, k, t: None)  # package missing
    assert search.web_search("penny docs") == []


def test_empty_on_error(monkeypatch):
    def boom(q, k, t):
        raise RuntimeError("rate limited / blocked")

    monkeypatch.setattr(search, "_duckduckgo", boom)
    assert search.web_search("penny docs") == []


def test_blank_query_short_circuits(monkeypatch):
    monkeypatch.setattr(search, "_duckduckgo", lambda q, k, t: ["should-not-run"])
    assert search.web_search("   ") == []
