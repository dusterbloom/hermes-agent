"""Tests for the SearXNG search backend in tools/web_tools.py.

Coverage:
  _searxng_search() — success, empty results, connection error, custom URL, num_results cap
  _is_searxng_available() — healthy instance, unavailable instance
  _format_searxng_results() — markdown formatting, empty input
"""

import asyncio
import importlib
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies before any tools.* import.
# tools/web_tools.py does `from firecrawl import Firecrawl` at module level,
# and tools/__init__.py chains to fal_client and many others. We must inject
# stubs into sys.modules before those imports fire.
# ---------------------------------------------------------------------------

def _stub(name: str) -> MagicMock:
    """Insert a MagicMock stub under *name* in sys.modules (if absent)."""
    if name not in sys.modules:
        sys.modules[name] = MagicMock()
    return sys.modules[name]


_firecrawl_stub = _stub("firecrawl")
_firecrawl_stub.Firecrawl = MagicMock()

# Other deps imported by the tools package that may not be installed
for _dep in (
    "fal_client",
    "playwright",
    "playwright.async_api",
    "pynput",
    "pynput.keyboard",
):
    _stub(_dep)

# Now it is safe to import tools.web_tools directly (bypassing tools/__init__.py
# which would pull in even more optional deps).
import importlib.util as _ilu
import tools.web_tools as _web_tools_module  # noqa: E402 — intentional late import

# Grab the symbols under test from the already-loaded module object.
_searxng_search = _web_tools_module._searxng_search
_is_searxng_available = _web_tools_module._is_searxng_available
_format_searxng_results = _web_tools_module._format_searxng_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_httpx_response(status_code: int, json_body: dict = None):
    """Build a minimal mock that satisfies httpx.Response interface."""
    mock = MagicMock()
    mock.status_code = status_code
    if json_body is not None:
        mock.json.return_value = json_body
    if status_code >= 400:
        mock.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return mock


def _make_async_client_mock(fake_get_fn):
    """Return an AsyncMock context-manager whose .get() is *fake_get_fn*."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = fake_get_fn
    return mock_client


# ---------------------------------------------------------------------------
# _searxng_search
# ---------------------------------------------------------------------------

class TestSearxngSearch:
    """Tests for the _searxng_search async function."""

    @pytest.mark.asyncio
    async def test_searxng_search_success(self):
        """Returns parsed results on a valid JSON response."""
        payload = {
            "results": [
                {"title": "Alpha", "url": "https://alpha.example", "content": "Alpha snippet"},
                {"title": "Beta", "url": "https://beta.example", "content": "Beta snippet"},
            ]
        }

        async def fake_get(url, params=None, **kw):
            return _make_httpx_response(200, payload)

        client_mock = _make_async_client_mock(fake_get)
        with patch.object(_web_tools_module.httpx, "AsyncClient", return_value=client_mock):
            results = await _searxng_search("python", num_results=10)

        assert len(results) == 2
        assert results[0] == {
            "title": "Alpha",
            "url": "https://alpha.example",
            "content": "Alpha snippet",
        }
        assert results[1]["title"] == "Beta"

    @pytest.mark.asyncio
    async def test_searxng_search_empty_results(self):
        """Returns empty list when SearXNG returns no results."""
        async def fake_get(url, params=None, **kw):
            return _make_httpx_response(200, {"results": []})

        client_mock = _make_async_client_mock(fake_get)
        with patch.object(_web_tools_module.httpx, "AsyncClient", return_value=client_mock):
            results = await _searxng_search("noresults")

        assert results == []

    @pytest.mark.asyncio
    async def test_searxng_search_connection_error(self):
        """Returns empty list and logs warning when the request fails."""
        async def fake_get(url, params=None, **kw):
            raise ConnectionError("connection refused")

        client_mock = _make_async_client_mock(fake_get)
        with patch.object(_web_tools_module.httpx, "AsyncClient", return_value=client_mock):
            results = await _searxng_search("anything")

        assert results == []

    @pytest.mark.asyncio
    async def test_searxng_search_custom_url(self):
        """Uses the provided searxng_url argument, not the env-var default."""
        captured_urls = []

        async def fake_get(url, params=None, **kw):
            captured_urls.append(url)
            return _make_httpx_response(200, {"results": []})

        client_mock = _make_async_client_mock(fake_get)
        with patch.object(_web_tools_module.httpx, "AsyncClient", return_value=client_mock):
            await _searxng_search("query", searxng_url="http://custom-host:9999")

        assert len(captured_urls) == 1
        assert captured_urls[0] == "http://custom-host:9999/search"

    @pytest.mark.asyncio
    async def test_searxng_search_num_results_limit(self):
        """Result list is capped at num_results even when SearXNG sends more."""
        payload = {
            "results": [
                {"title": f"Result {i}", "url": f"https://ex.com/{i}", "content": ""}
                for i in range(20)
            ]
        }

        async def fake_get(url, params=None, **kw):
            return _make_httpx_response(200, payload)

        client_mock = _make_async_client_mock(fake_get)
        with patch.object(_web_tools_module.httpx, "AsyncClient", return_value=client_mock):
            results = await _searxng_search("query", num_results=3)

        assert len(results) == 3


# ---------------------------------------------------------------------------
# _is_searxng_available
# ---------------------------------------------------------------------------

class TestIsSearxngAvailable:
    """Tests for the _is_searxng_available synchronous function."""

    def test_is_searxng_available_healthy(self):
        """Returns True when /healthz responds with 200."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch.object(_web_tools_module.httpx, "get", return_value=mock_resp):
            assert _is_searxng_available("http://localhost:8888") is True

    def test_is_searxng_available_down(self):
        """Returns False when both /healthz and /search raise exceptions."""
        with patch.object(_web_tools_module.httpx, "get", side_effect=ConnectionError("refused")):
            assert _is_searxng_available("http://localhost:8888") is False

    def test_is_searxng_available_healthz_fails_search_succeeds(self):
        """Falls back to /search endpoint when /healthz raises."""
        call_count = [0]

        def selective_get(url, **kw):
            call_count[0] += 1
            if "healthz" in url:
                raise ConnectionError("no healthz")
            resp = MagicMock()
            resp.status_code = 200
            return resp

        with patch.object(_web_tools_module.httpx, "get", side_effect=selective_get):
            result = _is_searxng_available("http://localhost:8888")

        assert result is True
        assert call_count[0] == 2  # healthz tried first, then /search

    def test_is_searxng_available_uses_env_default(self, monkeypatch):
        """Reads SEARXNG_URL from environment when no url argument is passed."""
        captured = []

        def fake_get(url, **kw):
            captured.append(url)
            raise ConnectionError("down")

        monkeypatch.setenv("SEARXNG_URL", "http://env-host:7777")
        with patch.object(_web_tools_module.httpx, "get", side_effect=fake_get):
            _is_searxng_available()

        assert any("env-host:7777" in u for u in captured)


# ---------------------------------------------------------------------------
# _format_searxng_results
# ---------------------------------------------------------------------------

class TestFormatSearxngResults:
    """Tests for the _format_searxng_results formatting helper."""

    def test_format_searxng_results(self):
        """Produces numbered markdown sections with title, URL, and content."""
        results = [
            {"title": "Page One", "url": "https://one.example", "content": "Some snippet here."},
            {"title": "Page Two", "url": "https://two.example", "content": ""},
        ]
        output = _format_searxng_results(results)

        assert "### 1. Page One" in output
        assert "**URL:** https://one.example" in output
        assert "Some snippet here." in output
        assert "### 2. Page Two" in output
        assert "**URL:** https://two.example" in output

    def test_format_searxng_results_empty(self):
        """Returns 'No results found.' for an empty list."""
        assert _format_searxng_results([]) == "No results found."

    def test_format_searxng_results_missing_fields(self):
        """Handles results with missing optional fields gracefully."""
        results = [{"title": "", "url": "", "content": ""}]
        output = _format_searxng_results(results)

        assert "### 1." in output
        assert "**URL:**" in output
