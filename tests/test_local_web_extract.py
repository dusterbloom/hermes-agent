"""Tests for local web extraction functions in tools/web_tools.py.

Covers _basic_web_extract, _local_web_extract, and _format_searxng_results
using synthetic data — no network, no Firecrawl API key required.

Strategy: load tools/web_tools.py directly via importlib so that the eager
imports in tools/__init__.py (terminal_tool, browser_tool, ...) are never
executed.  We stub out firecrawl and agent.auxiliary_client in sys.modules
first so web_tools.py itself can be loaded without those packages installed.
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Inject lightweight stubs for optional/heavy dependencies BEFORE loading
# web_tools.py.  Stubs are added only if the real package is absent.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent


def _stub(name, **attrs):
    """Create a minimal module stub and register it in sys.modules."""
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules.setdefault(name, m)
    return m


# firecrawl
_stub("firecrawl", Firecrawl=MagicMock())

# agent (parent package)
_stub("agent")

# agent.auxiliary_client
_aux_mod = _stub(
    "agent.auxiliary_client",
    get_async_text_auxiliary_client=MagicMock(return_value=(None, "stub/model")),
    get_auxiliary_extra_body=MagicMock(return_value=None),
    auxiliary_max_tokens_param=MagicMock(return_value={}),
)
# Make it accessible via agent.auxiliary_client
sys.modules["agent"].auxiliary_client = _aux_mod  # type: ignore[attr-defined]

# tools (parent package — stub so that relative imports work)
if "tools" not in sys.modules:
    _tools_pkg = types.ModuleType("tools")
    _tools_pkg.__path__ = [str(_PROJECT_ROOT / "tools")]
    _tools_pkg.__package__ = "tools"
    sys.modules["tools"] = _tools_pkg

# tools.debug_helpers
_dh_session = MagicMock()
_dh_session.active = False
_stub("tools.debug_helpers", DebugSession=MagicMock(return_value=_dh_session))

# tools.interrupt (imported lazily inside web_tools functions)
_stub("tools.interrupt", is_interrupted=MagicMock(return_value=False))

# ---------------------------------------------------------------------------
# Load tools/web_tools.py directly, bypassing tools/__init__.py
# ---------------------------------------------------------------------------

_web_tools_path = str(_PROJECT_ROOT / "tools" / "web_tools.py")
_spec = importlib.util.spec_from_file_location("tools.web_tools", _web_tools_path)
_web_tools_mod = importlib.util.module_from_spec(_spec)
_web_tools_mod.__package__ = "tools"
sys.modules["tools.web_tools"] = _web_tools_mod
_spec.loader.exec_module(_web_tools_mod)

# Pull the functions under test into module scope
_basic_web_extract = _web_tools_mod._basic_web_extract
_local_web_extract = _web_tools_mod._local_web_extract
_format_searxng_results = _web_tools_mod._format_searxng_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_httpx_client(html: str, status_code: int = 200) -> AsyncMock:
    """Return a mock AsyncClient context manager that yields *html*."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = html
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )

    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=resp)
    return client


# ---------------------------------------------------------------------------
# _basic_web_extract
# ---------------------------------------------------------------------------


class TestBasicWebExtract:
    """Unit tests for _basic_web_extract — mocked httpx, no network."""

    @pytest.mark.asyncio
    async def test_strips_html_tags(self):
        """HTML tags are removed from the returned text."""
        html = "<html><body><p>Hello world</p></body></html>"

        with patch.object(_web_tools_mod.httpx, "AsyncClient", return_value=_mock_httpx_client(html)):
            result = await _basic_web_extract("https://example.com/")

        assert result is not None
        assert "Hello world" in result
        assert "<p>" not in result
        assert "<body>" not in result

    @pytest.mark.asyncio
    async def test_removes_script_tags(self):
        """Script blocks and their JavaScript content are stripped."""
        html = (
            "<html><head><script>alert('xss')</script></head>"
            "<body><p>Real content</p></body></html>"
        )

        with patch.object(_web_tools_mod.httpx, "AsyncClient", return_value=_mock_httpx_client(html)):
            result = await _basic_web_extract("https://example.com/")

        assert result is not None
        assert "alert" not in result
        assert "Real content" in result

    @pytest.mark.asyncio
    async def test_truncates_long_content(self):
        """Content exceeding 10 000 characters is cut and annotated."""
        long_text = "B" * 20_000
        html = f"<html><body><p>{long_text}</p></body></html>"

        with patch.object(_web_tools_mod.httpx, "AsyncClient", return_value=_mock_httpx_client(html)):
            result = await _basic_web_extract("https://example.com/")

        assert result is not None
        assert "[Content truncated]" in result
        assert len(result) <= 10_100  # truncated text + label < 10 100 chars

    @pytest.mark.asyncio
    async def test_returns_none_on_connection_error(self):
        """Network failure returns None instead of raising."""
        bad_client = AsyncMock()
        bad_client.__aenter__ = AsyncMock(return_value=bad_client)
        bad_client.__aexit__ = AsyncMock(return_value=False)
        bad_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

        with patch.object(_web_tools_mod.httpx, "AsyncClient", return_value=bad_client):
            result = await _basic_web_extract("https://unreachable.example/")

        assert result is None


# ---------------------------------------------------------------------------
# _local_web_extract
# ---------------------------------------------------------------------------


class TestLocalWebExtract:
    """Unit tests for _local_web_extract — mocked trafilatura."""

    @pytest.mark.asyncio
    async def test_falls_back_to_basic_when_trafilatura_missing(self):
        """When trafilatura raises ImportError the basic extractor is called."""
        expected_fallback = "basic fallback result"

        import builtins
        real_import = builtins.__import__

        def bad_import(name, *args, **kwargs):
            if name == "trafilatura":
                raise ImportError("No module named 'trafilatura'")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=bad_import),
            patch.object(
                _web_tools_mod, "_basic_web_extract", AsyncMock(return_value=expected_fallback)
            ),
        ):
            result = await _local_web_extract("https://example.com/")

        assert result == expected_fallback

    @pytest.mark.asyncio
    async def test_returns_trafilatura_result_on_success(self):
        """When trafilatura is present and succeeds, its output is returned."""
        fake_traf = MagicMock()
        fake_traf.fetch_url.return_value = "<html>raw</html>"
        fake_traf.extract.return_value = "Extracted main content"

        with patch.dict(sys.modules, {"trafilatura": fake_traf}):
            result = await _local_web_extract("https://example.com/article")

        assert result == "Extracted main content"
        fake_traf.fetch_url.assert_called_once_with("https://example.com/article")

    @pytest.mark.asyncio
    async def test_falls_back_to_basic_when_trafilatura_raises(self):
        """A runtime error inside trafilatura triggers the basic fallback."""
        fake_traf = MagicMock()
        fake_traf.fetch_url.side_effect = RuntimeError("unexpected crash")

        fallback = "basic fallback text"

        with (
            patch.dict(sys.modules, {"trafilatura": fake_traf}),
            patch.object(
                _web_tools_mod, "_basic_web_extract", AsyncMock(return_value=fallback)
            ),
        ):
            result = await _local_web_extract("https://example.com/")

        assert result == fallback

    @pytest.mark.asyncio
    async def test_returns_none_when_fetch_url_returns_none(self):
        """trafilatura.fetch_url returning None yields None without calling extract."""
        fake_traf = MagicMock()
        fake_traf.fetch_url.return_value = None

        with patch.dict(sys.modules, {"trafilatura": fake_traf}):
            result = await _local_web_extract("https://example.com/")

        assert result is None
        fake_traf.extract.assert_not_called()


# ---------------------------------------------------------------------------
# _format_searxng_results — missing-fields handling
# ---------------------------------------------------------------------------


class TestFormatSearxngResultsHandlesMissingFields:
    """Tests that _format_searxng_results handles absent fields without crashing."""

    def test_empty_list_returns_no_results_string(self):
        assert _format_searxng_results([]) == "No results found."

    def test_formats_complete_result(self):
        results = [{"title": "Ex", "url": "https://ex.com", "content": "snippet"}]
        out = _format_searxng_results(results)
        assert "Ex" in out
        assert "https://ex.com" in out
        assert "snippet" in out

    def test_handles_missing_title(self):
        """Records without 'title' fall back to 'Untitled'."""
        results = [{"url": "https://example.com", "content": "text"}]
        out = _format_searxng_results(results)
        assert "Untitled" in out

    def test_handles_missing_url(self):
        """Records without 'url' still render without raising."""
        results = [{"title": "No URL here", "content": "text"}]
        out = _format_searxng_results(results)
        assert "No URL here" in out

    def test_handles_missing_content(self):
        """Records without 'content' still show title and URL."""
        results = [{"title": "T", "url": "https://u.com"}]
        out = _format_searxng_results(results)
        assert "T" in out
        assert "https://u.com" in out

    def test_numbers_results_sequentially(self):
        results = [
            {"title": "A", "url": "https://a.com", "content": ""},
            {"title": "B", "url": "https://b.com", "content": ""},
        ]
        out = _format_searxng_results(results)
        assert "### 1. A" in out
        assert "### 2. B" in out

    def test_handles_completely_empty_record(self):
        """An empty dict record renders 'Untitled' without raising."""
        out = _format_searxng_results([{}])
        assert "Untitled" in out
