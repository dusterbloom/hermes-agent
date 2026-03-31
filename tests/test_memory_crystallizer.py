"""Tests for compaction -> HRR crystallization hook.

RED phase: all tests should FAIL before implementation.
GREEN phase: add hrr_store attribute, _crystallize_to_hrr(), and call it from compact().
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from agent.lcm.config import LcmConfig
from agent.lcm.engine import LcmEngine
from agent.lcm.dag import SummaryNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(protect_last_n: int = 0) -> LcmEngine:
    config = LcmConfig(protect_last_n=protect_last_n)
    return LcmEngine(config)


def _ingest_n(engine: LcmEngine, n: int) -> None:
    for i in range(n):
        engine.ingest({"role": "user", "content": f"message {i} " + "word " * 10})


# ---------------------------------------------------------------------------
# Attribute existence
# ---------------------------------------------------------------------------

class TestEngineHasHrrStore:
    """LcmEngine gains an hrr_store attribute."""

    def test_hrr_store_attribute_exists(self):
        """hasattr(engine, 'hrr_store') must be True after __init__."""
        engine = _make_engine()
        assert hasattr(engine, "hrr_store"), (
            "LcmEngine.__init__ must set self.hrr_store"
        )

    def test_hrr_store_defaults_to_none(self):
        """hrr_store must be None when no store is injected."""
        engine = _make_engine()
        assert engine.hrr_store is None, (
            "hrr_store must default to None so the hook is disabled by default"
        )


# ---------------------------------------------------------------------------
# crystallizer called during compact()
# ---------------------------------------------------------------------------

class TestCrystallizerOnCompact:
    """compact() calls hrr_store.add_fact when hrr_store is set."""

    def test_compact_calls_add_fact(self):
        """With a mock hrr_store, compact() must call add_fact exactly once."""
        engine = _make_engine()
        _ingest_n(engine, 4)

        mock_store = MagicMock()
        engine.hrr_store = mock_store

        engine.compact("a brief summary", level=1, block_start=0, block_end=4)

        mock_store.add_fact.assert_called_once()

    def test_compact_passes_summary_text_as_content(self):
        """add_fact must receive the SummaryNode's text as the content argument."""
        engine = _make_engine()
        _ingest_n(engine, 4)

        mock_store = MagicMock()
        engine.hrr_store = mock_store

        summary_text = "important summary content"
        engine.compact(summary_text, level=1, block_start=0, block_end=4)

        call_kwargs = mock_store.add_fact.call_args
        # Accept both positional and keyword argument for content
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("content") == summary_text, (
                f"add_fact content kwarg should be {summary_text!r}, "
                f"got {call_kwargs.kwargs.get('content')!r}"
            )
        else:
            assert call_kwargs.args[0] == summary_text, (
                f"add_fact first positional arg should be {summary_text!r}, "
                f"got {call_kwargs.args[0]!r}"
            )

    def test_compact_uses_compaction_category(self):
        """add_fact must receive category='compaction'."""
        engine = _make_engine()
        _ingest_n(engine, 4)

        mock_store = MagicMock()
        engine.hrr_store = mock_store

        engine.compact("summary text", level=1, block_start=0, block_end=4)

        call_kwargs = mock_store.add_fact.call_args
        assert call_kwargs.kwargs.get("category") == "compaction", (
            f"add_fact must be called with category='compaction', "
            f"got {call_kwargs.kwargs.get('category')!r}"
        )

    def test_compact_includes_level_in_tags(self):
        """add_fact tags must contain the compaction level."""
        engine = _make_engine()
        _ingest_n(engine, 4)

        mock_store = MagicMock()
        engine.hrr_store = mock_store

        engine.compact("summary text", level=2, block_start=0, block_end=4)

        call_kwargs = mock_store.add_fact.call_args
        tags = call_kwargs.kwargs.get("tags")
        assert tags is not None, "add_fact must be called with a 'tags' keyword argument"

        # Tags can be a list, set, or string — the level must appear somewhere
        tags_str = str(tags)
        assert "2" in tags_str or "level" in tags_str.lower(), (
            f"tags {tags!r} should reference the compaction level (2)"
        )


# ---------------------------------------------------------------------------
# No-op when hrr_store is None
# ---------------------------------------------------------------------------

class TestCrystallizerNoop:
    """Crystallizer is a no-op when hrr_store is None."""

    def test_compact_returns_summary_node_when_hrr_store_none(self):
        """compact() must return a valid SummaryNode even without an hrr_store."""
        engine = _make_engine()
        _ingest_n(engine, 4)

        assert engine.hrr_store is None

        node = engine.compact("normal summary", level=1, block_start=0, block_end=4)

        assert node is not None
        assert isinstance(node, SummaryNode)

    def test_compact_active_list_shrinks_when_hrr_store_none(self):
        """Active list must still shrink after compact() with no hrr_store."""
        engine = _make_engine()
        _ingest_n(engine, 4)
        before = len(engine.active)

        engine.compact("normal summary", level=1, block_start=0, block_end=4)

        assert len(engine.active) < before


# ---------------------------------------------------------------------------
# Safety: failures in add_fact never block compaction
# ---------------------------------------------------------------------------

class TestCrystallizerSafety:
    """Crystallizer failures never block compaction."""

    def test_add_fact_exception_does_not_break_compact(self):
        """If add_fact raises, compact() must still complete normally."""
        engine = _make_engine()
        _ingest_n(engine, 4)

        mock_store = MagicMock()
        mock_store.add_fact.side_effect = Exception("db connection lost")
        engine.hrr_store = mock_store

        # Should not raise
        node = engine.compact("summary despite db error", level=1, block_start=0, block_end=4)

        assert node is not None, "compact() must return SummaryNode even when add_fact raises"

    def test_compact_return_value_unchanged_on_add_fact_error(self):
        """The SummaryNode returned is identical regardless of crystallizer outcome."""
        engine_ok = _make_engine()
        _ingest_n(engine_ok, 4)

        engine_fail = _make_engine()
        _ingest_n(engine_fail, 4)

        mock_store = MagicMock()
        mock_store.add_fact.side_effect = RuntimeError("network timeout")
        engine_fail.hrr_store = mock_store

        summary = "the very same summary"
        node_ok = engine_ok.compact(summary, level=1, block_start=0, block_end=4)
        node_fail = engine_fail.compact(summary, level=1, block_start=0, block_end=4)

        assert node_ok.text == node_fail.text
        assert node_ok.level == node_fail.level

    def test_add_fact_failure_is_swallowed_not_reraised(self):
        """Exceptions from add_fact are caught; they must not propagate to caller."""
        engine = _make_engine()
        _ingest_n(engine, 4)

        mock_store = MagicMock()
        mock_store.add_fact.side_effect = ValueError("invalid schema")
        engine.hrr_store = mock_store

        try:
            engine.compact("safe summary", level=1, block_start=0, block_end=4)
        except (ValueError, Exception) as exc:
            pytest.fail(
                f"compact() must swallow add_fact exceptions, but raised {type(exc).__name__}: {exc}"
            )

    def test_add_fact_failure_is_logged(self):
        """Swallowed add_fact exceptions must still be logged (not silently ignored)."""
        import logging

        engine = _make_engine()
        _ingest_n(engine, 4)

        mock_store = MagicMock()
        mock_store.add_fact.side_effect = Exception("boom")
        engine.hrr_store = mock_store

        with patch.object(
            logging.getLogger("agent.lcm.engine"), "warning"
        ) as mock_warn:
            engine.compact("summary with logged error", level=1, block_start=0, block_end=4)
            mock_warn.assert_called_once()


# ---------------------------------------------------------------------------
# async_compact also triggers crystallizer
# ---------------------------------------------------------------------------

class TestAsyncCompactCrystallizes:
    """async_compact also triggers crystallizer through compact()."""

    def test_async_compact_triggers_crystallizer(self):
        """With hrr_store set, async_compact must call add_fact after completion."""
        config = LcmConfig(protect_last_n=2)
        engine = LcmEngine(config)
        _ingest_n(engine, 8)

        # Stub summarizer so no real LLM call is made
        engine.summarizer.summarize = lambda turns, previous_summary=None: "async summary text"

        mock_store = MagicMock()
        engine.hrr_store = mock_store

        # Reset pending flag so async_compact is willing to run
        engine._async_compaction_pending = False

        thread = engine.async_compact()
        assert thread is not None, "async_compact() must return a thread when not already pending"
        thread.join(timeout=5.0)

        mock_store.add_fact.assert_called_once()

    def test_async_compact_crystallizer_failure_does_not_kill_thread(self):
        """If the crystallizer raises during async compaction, the thread finishes cleanly."""
        config = LcmConfig(protect_last_n=2)
        engine = LcmEngine(config)
        _ingest_n(engine, 8)

        engine.summarizer.summarize = lambda turns, previous_summary=None: "async summary text"

        mock_store = MagicMock()
        mock_store.add_fact.side_effect = Exception("store unavailable")
        engine.hrr_store = mock_store

        engine._async_compaction_pending = False

        results: list = []
        event = threading.Event()

        def callback(node):
            results.append(node)
            event.set()

        thread = engine.async_compact(callback=callback)
        assert thread is not None
        thread.join(timeout=5.0)
        event.wait(timeout=5.0)

        # Callback must receive a SummaryNode, not None
        assert len(results) == 1
        assert results[0] is not None, (
            "async_compact callback must receive SummaryNode even when crystallizer fails"
        )
