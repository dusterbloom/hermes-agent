"""TDD tests for DAM auto-init and auto-sync wired into LcmEngine.

RED phase: all tests should FAIL before implementation.
GREEN phase: implement to make them pass.
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from agent.lcm.engine import LcmEngine
from agent.lcm.config import LcmConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(**kwargs) -> LcmEngine:
    config = LcmConfig(**kwargs)
    return LcmEngine(config)


def _ingest_n(engine: LcmEngine, n: int) -> None:
    for i in range(n):
        engine.ingest({"role": "user", "content": f"message number {i}"})


# ---------------------------------------------------------------------------
# 1. LcmEngine has a `retriever` attribute
# ---------------------------------------------------------------------------

class TestEngineRetrieverAttribute:
    def test_engine_has_retriever_attribute(self):
        """LcmEngine must expose a `retriever` attribute (DAMRetriever or None)."""
        engine = _make_engine()
        assert hasattr(engine, "retriever"), "LcmEngine must have a 'retriever' attribute"


# ---------------------------------------------------------------------------
# 2. LcmEngine creates DAMRetriever when numpy is available
# ---------------------------------------------------------------------------

class TestEngineCreatesRetrieverWhenNumpyAvailable:
    def test_engine_creates_retriever_when_numpy_available(self):
        """When numpy is importable, __init__ must create a DAMRetriever."""
        # numpy is available in the test environment (it's a dependency of DAM)
        import numpy  # noqa: F401 — just to confirm availability

        engine = _make_engine()
        assert engine.retriever is not None, (
            "Expected LcmEngine.retriever to be a DAMRetriever when numpy is available"
        )

    def test_retriever_is_dam_retriever_instance(self):
        """The created retriever must be a DAMRetriever."""
        import numpy  # noqa: F401

        from agent.lcm.dam import DAMRetriever

        engine = _make_engine()
        assert isinstance(engine.retriever, DAMRetriever)


# ---------------------------------------------------------------------------
# 3. Graceful degradation when numpy unavailable
# ---------------------------------------------------------------------------

class TestEngineRetrieverNoneWhenNumpyUnavailable:
    def test_engine_retriever_none_when_numpy_unavailable(self):
        """When numpy import fails, engine.retriever must be None."""
        # Block the DAM package and all its submodules to simulate numpy absence.
        # Setting a sys.modules entry to None makes Python raise ImportError on
        # any subsequent 'from <key> import ...' within that Python process.
        blocked = {
            "agent.lcm.dam": None,
            "agent.lcm.dam.network": None,
            "agent.lcm.dam.encoder": None,
            "agent.lcm.dam.retrieval": None,
        }
        with patch.dict("sys.modules", blocked):
            engine = _make_engine()
            assert engine.retriever is None, (
                "retriever must be None when DAM (numpy) is unavailable"
            )


# ---------------------------------------------------------------------------
# 4. ingest() auto-syncs after N messages
# ---------------------------------------------------------------------------

class TestIngestAutoSyncsRetriever:
    def test_ingest_auto_syncs_retriever_after_threshold(self):
        """After ingesting >= sync_interval messages the retriever's pattern cache
        must contain those messages."""
        engine = _make_engine()
        if engine.retriever is None:
            pytest.skip("numpy not available")

        sync_interval = engine._dam_sync_interval
        _ingest_n(engine, sync_interval)

        # After exactly sync_interval messages, a sync should have fired
        assert len(engine.retriever._pattern_cache) > 0, (
            f"Pattern cache should be populated after {sync_interval} ingested messages"
        )

    def test_pattern_cache_grows_with_more_ingests(self):
        """Multiple sync cycles should accumulate patterns."""
        engine = _make_engine()
        if engine.retriever is None:
            pytest.skip("numpy not available")

        sync_interval = engine._dam_sync_interval
        _ingest_n(engine, sync_interval * 2)

        assert len(engine.retriever._pattern_cache) >= sync_interval * 2


# ---------------------------------------------------------------------------
# 5. Sync is periodic, not per-message
# ---------------------------------------------------------------------------

class TestIngestSyncsPeriodicNotEveryMessage:
    def test_sync_not_called_on_every_ingest(self):
        """The DAM sync should NOT fire for every single ingested message."""
        engine = _make_engine()
        if engine.retriever is None:
            pytest.skip("numpy not available")

        sync_interval = engine._dam_sync_interval
        assert sync_interval > 1, "sync_interval must be > 1 to test periodic behaviour"

        # Ingest fewer messages than the sync threshold
        _ingest_n(engine, sync_interval - 1)

        # Cache should still be empty — sync hasn't fired yet
        assert len(engine.retriever._pattern_cache) == 0, (
            "Pattern cache should be empty before the sync threshold is reached"
        )

    def test_counter_resets_after_sync(self):
        """After a sync cycle the message-since-sync counter must reset."""
        engine = _make_engine()
        if engine.retriever is None:
            pytest.skip("numpy not available")

        sync_interval = engine._dam_sync_interval
        _ingest_n(engine, sync_interval)
        # Counter should have been reset to 0 after sync
        assert engine._dam_messages_since_sync == 0


# ---------------------------------------------------------------------------
# 6. search() uses DAM when available
# ---------------------------------------------------------------------------

class TestSearchUsesDAMWhenAvailable:
    def test_search_uses_dam_when_available(self):
        """When retriever is initialized and populated, search() should use DAM."""
        engine = _make_engine(semantic_search=False)
        if engine.retriever is None:
            pytest.skip("numpy not available")

        # Populate the engine past the sync threshold so retriever is trained
        sync_interval = engine._dam_sync_interval
        for i in range(sync_interval):
            engine.ingest({"role": "user", "content": f"topic about dogs {i}"})

        # DAM retriever must have been synced
        assert len(engine.retriever._pattern_cache) > 0

        # Replace the retriever's search with a spy
        original_search = engine.retriever.search
        call_log: list = []

        def spy_search(query, limit=10):
            call_log.append(query)
            return original_search(query, limit=limit)

        engine.retriever.search = spy_search

        results = engine.search("dogs", limit=5)
        assert len(call_log) > 0, "engine.search() must delegate to retriever.search()"

    def test_search_returns_results_with_dam(self):
        """search() should still return (msg_id, message) tuples when DAM is used."""
        engine = _make_engine(semantic_search=False)
        if engine.retriever is None:
            pytest.skip("numpy not available")

        sync_interval = engine._dam_sync_interval
        for i in range(sync_interval):
            engine.ingest({"role": "user", "content": f"cats are great {i}"})

        results = engine.search("cats", limit=5)
        # Each result must be (int, dict)
        for msg_id, msg in results:
            assert isinstance(msg_id, int)
            assert isinstance(msg, dict)


# ---------------------------------------------------------------------------
# 7. search() falls back to keyword when retriever is None
# ---------------------------------------------------------------------------

class TestSearchFallsBackToKeywordWithoutDAM:
    def test_search_keyword_fallback_when_retriever_none(self):
        """When retriever is None, search() must still work via keyword fallback."""
        engine = _make_engine(semantic_search=False)
        engine.retriever = None  # Force no DAM

        engine.ingest({"role": "user", "content": "hello world"})
        engine.ingest({"role": "user", "content": "foo bar"})

        results = engine.search("hello", limit=5)
        assert len(results) == 1
        assert results[0][1]["content"] == "hello world"

    def test_search_keyword_fallback_returns_correct_type(self):
        """Keyword fallback must return (msg_id, message) tuples."""
        engine = _make_engine(semantic_search=False)
        engine.retriever = None

        engine.ingest({"role": "user", "content": "test message"})
        results = engine.search("test", limit=5)

        assert len(results) >= 1
        for msg_id, msg in results:
            assert isinstance(msg_id, int)
            assert isinstance(msg, dict)


# ---------------------------------------------------------------------------
# 8. to_session_metadata() includes DAM state
# ---------------------------------------------------------------------------

class TestRetrieverPersistedInSessionMetadata:
    def test_session_metadata_includes_dam_state(self):
        """to_session_metadata() should include a 'dam' key with DAM state."""
        engine = _make_engine()
        if engine.retriever is None:
            pytest.skip("numpy not available")

        _ingest_n(engine, 3)
        metadata = engine.to_session_metadata()

        assert "dam" in metadata, "session metadata must include a 'dam' key"

    def test_session_metadata_dam_has_last_indexed_id(self):
        """DAM state must include 'last_indexed_id'."""
        engine = _make_engine()
        if engine.retriever is None:
            pytest.skip("numpy not available")

        _ingest_n(engine, engine._dam_sync_interval)
        metadata = engine.to_session_metadata()

        assert "last_indexed_id" in metadata["dam"]

    def test_session_metadata_dam_has_n_patterns_trained(self):
        """DAM state must include 'n_patterns_trained'."""
        engine = _make_engine()
        if engine.retriever is None:
            pytest.skip("numpy not available")

        _ingest_n(engine, engine._dam_sync_interval)
        metadata = engine.to_session_metadata()

        assert "n_patterns_trained" in metadata["dam"]


# ---------------------------------------------------------------------------
# 9. rebuild_from_session() restores DAM state
# ---------------------------------------------------------------------------

class TestRetrieverRestoredFromSession:
    def test_rebuild_restores_last_indexed_id(self):
        """rebuild_from_session() should restore last_indexed_id so the DAM
        doesn't re-index everything from scratch."""
        engine = _make_engine()
        if engine.retriever is None:
            pytest.skip("numpy not available")

        # Ingest enough to trigger a sync
        _ingest_n(engine, engine._dam_sync_interval)
        original_indexed_id = engine.retriever._last_indexed_id

        # Serialize
        metadata = engine.to_session_metadata()
        messages = engine.active_messages()

        # Rebuild
        config = LcmConfig()
        engine2 = LcmEngine.rebuild_from_session(
            {"messages": messages, "lcm": metadata},
            config,
        )

        assert engine2.retriever is not None, "Rebuilt engine must have a retriever"
        assert engine2.retriever._last_indexed_id == original_indexed_id, (
            "Rebuilt engine must restore last_indexed_id to avoid re-indexing"
        )

    def test_rebuild_without_dam_state_still_works(self):
        """rebuild_from_session() must work even when session has no 'dam' key
        (backward compatibility with old sessions)."""
        config = LcmConfig()
        session_data = {
            "messages": [{"role": "user", "content": "hello"}],
            "lcm": {
                "original_messages": [{"role": "user", "content": "hello"}],
                # no 'dam' key — old session format
            },
        }
        engine = LcmEngine.rebuild_from_session(session_data, config)
        # Should not raise; retriever may or may not be None depending on numpy
        assert hasattr(engine, "retriever")


# ---------------------------------------------------------------------------
# 10. compact() does NOT trigger a DAM sync
# ---------------------------------------------------------------------------

class TestCompactTriggersNoSync:
    def test_compact_does_not_sync_dam(self):
        """compact() must NOT trigger a DAM sync — it replaces existing messages."""
        config = LcmConfig(protect_last_n=0)
        engine = LcmEngine(config)
        if engine.retriever is None:
            pytest.skip("numpy not available")

        # Ingest some messages (below sync threshold so no auto-sync yet)
        sync_interval = engine._dam_sync_interval
        # Ingest just below threshold
        for i in range(max(sync_interval - 1, 1)):
            engine.ingest({"role": "user", "content": f"msg {i}"})

        # Record counter and cache state before compact
        counter_before = engine._dam_messages_since_sync
        cache_size_before = len(engine.retriever._pattern_cache)

        # Compact whatever is in active (if there are at least 2 messages)
        if len(engine.active) >= 2:
            engine.compact("test summary", level=1, block_start=0, block_end=len(engine.active))

        # Compact must not have changed the DAM counter or triggered a sync
        assert engine._dam_messages_since_sync == counter_before, (
            "compact() must not increment the DAM sync counter"
        )
        assert len(engine.retriever._pattern_cache) == cache_size_before, (
            "compact() must not trigger a DAM sync"
        )
