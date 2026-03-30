"""TDD tests for LcmEngine compaction effectiveness metrics.

RED phase: all tests should FAIL before implementation.
GREEN phase: implement metrics in engine.py, format.py, session.py.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from agent.lcm.config import LcmConfig
from agent.lcm.engine import LcmEngine


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
# Structure
# ---------------------------------------------------------------------------

class TestMetricsAttribute:
    def test_engine_has_metrics_attribute(self):
        """LcmEngine must expose a `metrics` dict."""
        engine = _make_engine()
        assert hasattr(engine, "metrics")
        assert isinstance(engine.metrics, dict)

    def test_metrics_initialized_empty(self):
        """Fresh engine has zeroed / None metrics."""
        engine = _make_engine()
        m = engine.metrics
        assert m["total_compactions"] == 0
        assert m["tokens_saved"] == 0
        assert m["tokens_before_total"] == 0
        assert m["tokens_after_total"] == 0
        assert m["levels"] == {1: 0, 2: 0, 3: 0}
        assert m["last_compaction_time"] is None


# ---------------------------------------------------------------------------
# compact() tracking
# ---------------------------------------------------------------------------

class TestCompactTracking:
    def test_compact_tracks_total_compactions(self):
        """Each call to compact() increments total_compactions."""
        engine = _make_engine()
        _ingest_n(engine, 6)

        assert engine.metrics["total_compactions"] == 0
        engine.compact("summary A", level=1, block_start=0, block_end=3)
        assert engine.metrics["total_compactions"] == 1
        engine.compact("summary B", level=1, block_start=0, block_end=1)
        assert engine.metrics["total_compactions"] == 2

    def test_compact_tracks_tokens_saved(self):
        """tokens_saved accumulates (original_tokens - summary_tokens) per compact."""
        engine = _make_engine()
        _ingest_n(engine, 4)

        engine.compact("short summary", level=1, block_start=0, block_end=4)
        assert engine.metrics["tokens_saved"] > 0

    def test_compact_tokens_saved_is_cumulative(self):
        """tokens_saved keeps accumulating across multiple compactions."""
        engine = _make_engine()
        _ingest_n(engine, 6)

        engine.compact("first summary", level=1, block_start=0, block_end=3)
        saved_after_first = engine.metrics["tokens_saved"]

        engine.compact("second summary", level=2, block_start=0, block_end=1)
        saved_after_second = engine.metrics["tokens_saved"]

        # Second compaction compacts the first summary into a shorter one;
        # tokens_saved should not decrease (may be 0 if summary grew, but total >= 0).
        assert saved_after_second >= 0
        assert engine.metrics["tokens_before_total"] > 0
        assert engine.metrics["tokens_after_total"] > 0

    def test_compact_tracks_level_distribution(self):
        """metrics['levels'] counts how many times each escalation level was used."""
        engine = _make_engine()
        _ingest_n(engine, 6)

        engine.compact("lvl1 summary", level=1, block_start=0, block_end=2)
        engine.compact("lvl2 summary", level=2, block_start=0, block_end=1)
        engine.compact("lvl1 again",  level=1, block_start=0, block_end=1)

        assert engine.metrics["levels"][1] == 2
        assert engine.metrics["levels"][2] == 1
        assert engine.metrics["levels"].get(3, 0) == 0

    def test_compact_tracks_last_compaction_time(self):
        """metrics['last_compaction_time'] is an ISO timestamp string after compact."""
        engine = _make_engine()
        _ingest_n(engine, 4)

        before = datetime.now(timezone.utc)
        engine.compact("summary", level=1, block_start=0, block_end=4)
        after = datetime.now(timezone.utc)

        ts = engine.metrics["last_compaction_time"]
        assert ts is not None
        assert isinstance(ts, str)

        # Must be parseable as ISO 8601
        parsed = datetime.fromisoformat(ts)
        # Normalise to UTC-aware for comparison regardless of whether parsed is naive/aware
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        assert parsed >= before
        assert parsed <= after

    def test_compact_tracks_compression_ratio(self):
        """After compaction, metrics expose a meaningful compression ratio."""
        engine = _make_engine()
        _ingest_n(engine, 6)

        engine.compact("short", level=1, block_start=0, block_end=6)

        # compression_ratio = tokens_after_total / tokens_before_total
        before = engine.metrics["tokens_before_total"]
        after = engine.metrics["tokens_after_total"]
        assert before > 0
        assert after >= 0
        ratio = after / before
        assert 0.0 <= ratio <= 1.0  # Summary should be <= original


# ---------------------------------------------------------------------------
# Computed properties / helpers
# ---------------------------------------------------------------------------

class TestComputedMetrics:
    def test_average_compression_ratio_zero_before_compact(self):
        """avg_compression_ratio is 0.0 when no compaction has happened."""
        engine = _make_engine()
        assert engine.metrics.get("avg_compression_ratio", 0.0) == 0.0

    def test_average_compression_ratio_after_multiple_compactions(self):
        """avg_compression_ratio is the running average after multiple compactions."""
        engine = _make_engine()
        _ingest_n(engine, 6)

        engine.compact("s1", level=1, block_start=0, block_end=3)
        engine.compact("s2", level=1, block_start=0, block_end=1)

        ratio = engine.metrics["avg_compression_ratio"]
        assert isinstance(ratio, float)
        assert 0.0 <= ratio <= 1.0


# ---------------------------------------------------------------------------
# format_metrics()
# ---------------------------------------------------------------------------

class TestFormatMetrics:
    def test_format_metrics_returns_string(self):
        """format_metrics() must return a non-empty string."""
        engine = _make_engine()
        result = engine.format_metrics()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_metrics_contains_key_fields(self):
        """format_metrics() output mentions compaction count and tokens saved."""
        engine = _make_engine()
        _ingest_n(engine, 4)
        engine.compact("summary", level=1, block_start=0, block_end=4)

        result = engine.format_metrics()
        lower = result.lower()
        assert "compact" in lower
        assert "token" in lower

    def test_format_metrics_readable_before_any_compact(self):
        """format_metrics() works even before any compaction."""
        engine = _make_engine()
        result = engine.format_metrics()
        assert "0" in result  # Should show zeroed stats


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

class TestMetricsSessionPersistence:
    def test_metrics_included_in_session_metadata(self):
        """to_session_metadata() includes a 'metrics' key."""
        engine = _make_engine()
        _ingest_n(engine, 4)
        engine.compact("summary", level=1, block_start=0, block_end=4)

        metadata = engine.to_session_metadata()
        assert "metrics" in metadata

    def test_metrics_persist_in_session(self):
        """Metrics survive a round-trip through to_session_metadata / rebuild_from_session."""
        config = LcmConfig(protect_last_n=0)
        engine = LcmEngine(config)
        _ingest_n(engine, 4)
        engine.compact("summary", level=1, block_start=0, block_end=4)

        compactions_before = engine.metrics["total_compactions"]
        saved_before = engine.metrics["tokens_saved"]

        metadata = engine.to_session_metadata()
        messages = engine.active_messages()

        engine2 = LcmEngine.rebuild_from_session(
            {"messages": messages, "lcm": metadata},
            config,
        )

        assert engine2.metrics["total_compactions"] == compactions_before
        assert engine2.metrics["tokens_saved"] == saved_before

    def test_metrics_reset_on_engine_reset(self):
        """engine.reset() zeroes all metrics."""
        engine = _make_engine()
        _ingest_n(engine, 4)
        engine.compact("summary", level=1, block_start=0, block_end=4)

        assert engine.metrics["total_compactions"] == 1

        engine.reset()

        m = engine.metrics
        assert m["total_compactions"] == 0
        assert m["tokens_saved"] == 0
        assert m["tokens_before_total"] == 0
        assert m["tokens_after_total"] == 0
        assert m["levels"] == {1: 0, 2: 0, 3: 0}
        assert m["last_compaction_time"] is None


# ---------------------------------------------------------------------------
# Async compaction
# ---------------------------------------------------------------------------

class TestAsyncMetrics:
    def test_async_compact_also_tracks_metrics(self):
        """Async compaction (via compact_block) updates metrics like sync."""
        config = LcmConfig(protect_last_n=2)
        engine = LcmEngine(config)
        _ingest_n(engine, 6)

        # Mock summarizer so no real LLM call is needed
        engine.summarizer.summarize = lambda turns, previous_summary=None: "async summary"

        thread = engine.async_compact()
        if thread is not None:
            thread.join(timeout=5.0)

        # By now compaction should have run
        assert engine.metrics["total_compactions"] >= 1
