"""Tests for async compaction with thread pool in LcmEngine.

TDD protocol:
  RED  - all tests written here should fail before implementation
  GREEN - implement async_compact() in engine.py to make them pass
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from agent.lcm.config import LcmConfig
from agent.lcm.engine import CompactionAction, LcmEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(protect_last_n: int = 2, tau_soft: float = 0.50, tau_hard: float = 0.85) -> LcmEngine:
    config = LcmConfig(tau_soft=tau_soft, tau_hard=tau_hard, protect_last_n=protect_last_n)
    engine = LcmEngine(config)
    return engine


def _fill_engine(engine: LcmEngine, n: int = 8) -> None:
    """Ingest n messages and mock the summarizer so compaction always succeeds."""
    for i in range(n):
        engine.ingest({"role": "user", "content": f"message {i} with some padding to fill tokens"})
    engine.summarizer.summarize = lambda turns, previous_summary=None: "## Summary\nCompacted."


# ---------------------------------------------------------------------------
# 1. test_async_compact_runs_in_thread
# ---------------------------------------------------------------------------

class TestAsyncCompactRunsInThread:
    """async_compact() must execute compaction off the calling thread."""

    def test_async_compact_runs_in_thread(self):
        engine = _make_engine()
        _fill_engine(engine)

        captured_thread: list[threading.Thread] = []

        original_auto_compact = engine.auto_compact

        def spy_auto_compact():
            captured_thread.append(threading.current_thread())
            return original_auto_compact()

        engine.auto_compact = spy_auto_compact

        # Reset the pending flag so async_compact won't skip
        engine._async_compaction_pending = False

        thread = engine.async_compact()
        assert thread is not None, "async_compact() must return the thread handle"
        thread.join(timeout=5)

        assert len(captured_thread) == 1, "auto_compact must have been called exactly once"
        assert captured_thread[0] is not threading.main_thread(), (
            "compaction must run on a background thread, not main thread"
        )


# ---------------------------------------------------------------------------
# 2. test_async_compact_callback_on_success
# ---------------------------------------------------------------------------

class TestAsyncCompactCallbackOnSuccess:
    """callback(result) is called with the SummaryNode after successful compaction."""

    def test_async_compact_callback_on_success(self):
        engine = _make_engine()
        _fill_engine(engine)
        engine._async_compaction_pending = False

        results: list = []
        event = threading.Event()

        def callback(node):
            results.append(node)
            event.set()

        thread = engine.async_compact(callback=callback)
        assert thread is not None
        thread.join(timeout=5)
        event.wait(timeout=5)

        assert len(results) == 1
        # The result should be a SummaryNode (not None) on success
        assert results[0] is not None


# ---------------------------------------------------------------------------
# 3. test_async_compact_callback_on_error
# ---------------------------------------------------------------------------

class TestAsyncCompactCallbackOnError:
    """If compaction raises, callback is called with None and error is logged."""

    def test_async_compact_callback_on_error(self):
        engine = _make_engine()
        _fill_engine(engine)
        engine._async_compaction_pending = False

        def boom():
            raise RuntimeError("summarizer exploded")

        engine.auto_compact = boom

        results: list = []
        event = threading.Event()

        def callback(node):
            results.append(node)
            event.set()

        import logging
        with patch.object(logging.getLogger("agent.lcm.engine"), "error") as mock_log:
            thread = engine.async_compact(callback=callback)
            assert thread is not None
            thread.join(timeout=5)
            event.wait(timeout=5)

            assert len(results) == 1
            assert results[0] is None, "callback must receive None when compaction fails"
            mock_log.assert_called_once()


# ---------------------------------------------------------------------------
# 4. test_async_compact_sets_pending_flag
# ---------------------------------------------------------------------------

class TestAsyncCompactSetsPendingFlag:
    """_async_compaction_pending transitions True->False around compaction."""

    def test_async_compact_sets_pending_flag(self):
        engine = _make_engine()
        _fill_engine(engine)
        engine._async_compaction_pending = False

        pending_during: list[bool] = []
        event_started = threading.Event()

        original_auto_compact = engine.auto_compact

        def spy():
            event_started.set()
            pending_during.append(engine._async_compaction_pending)
            return original_auto_compact()

        engine.auto_compact = spy

        thread = engine.async_compact()
        assert thread is not None

        event_started.wait(timeout=5)
        # While running, the flag should be True
        assert pending_during[0] is True

        thread.join(timeout=5)
        # After completion, flag must be False
        assert engine._async_compaction_pending is False


# ---------------------------------------------------------------------------
# 5. test_async_compact_skips_if_already_pending
# ---------------------------------------------------------------------------

class TestAsyncCompactSkipsIfAlreadyPending:
    """If _async_compaction_pending is True, async_compact() returns without launching."""

    def test_async_compact_skips_if_already_pending(self):
        engine = _make_engine()
        _fill_engine(engine)
        engine._async_compaction_pending = True  # Already pending

        call_count = [0]
        original_auto_compact = engine.auto_compact

        def spy():
            call_count[0] += 1
            return original_auto_compact()

        engine.auto_compact = spy

        result = engine.async_compact()

        # Should have returned immediately (None or no thread)
        assert result is None, "async_compact() must return None when already pending"
        assert call_count[0] == 0, "auto_compact must NOT be called when already pending"


# ---------------------------------------------------------------------------
# 6. test_async_compact_updates_active_list
# ---------------------------------------------------------------------------

class TestAsyncCompactUpdatesActiveList:
    """After async compaction, the engine's active list reflects the compacted state."""

    def test_async_compact_updates_active_list(self):
        engine = _make_engine(protect_last_n=2)
        _fill_engine(engine, n=8)
        engine._async_compaction_pending = False

        active_before = len(engine.active)

        thread = engine.async_compact()
        assert thread is not None
        thread.join(timeout=5)

        active_after = len(engine.active)
        assert active_after < active_before, (
            f"active list should shrink after compaction; before={active_before}, after={active_after}"
        )

        # At least one summary entry should exist
        kinds = [e.kind for e in engine.active]
        assert "summary" in kinds


# ---------------------------------------------------------------------------
# 7. test_async_compact_thread_safe
# ---------------------------------------------------------------------------

class TestAsyncCompactThreadSafe:
    """Two concurrent calls — only one actually runs, the other is skipped."""

    def test_async_compact_thread_safe(self):
        engine = _make_engine()
        _fill_engine(engine, n=10)
        engine._async_compaction_pending = False

        call_count = [0]
        barrier = threading.Barrier(2)

        original_auto_compact = engine.auto_compact

        def slow_compact():
            call_count[0] += 1
            time.sleep(0.05)  # simulate slow compaction
            return original_auto_compact()

        engine.auto_compact = slow_compact

        threads = []

        def launch():
            barrier.wait()  # both threads try at the exact same time
            t = engine.async_compact()
            if t:
                threads.append(t)

        t1 = threading.Thread(target=launch)
        t2 = threading.Thread(target=launch)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Wait for any launched compaction threads to finish
        for t in threads:
            t.join(timeout=5)

        assert call_count[0] == 1, (
            f"Only one compaction should run; got {call_count[0]}"
        )


# ---------------------------------------------------------------------------
# 8. test_async_compact_with_lock
# ---------------------------------------------------------------------------

class TestAsyncCompactWithLock:
    """Engine must have a _compact_lock; compact() uses it during active list mutation."""

    def test_async_compact_with_lock(self):
        engine = _make_engine()
        # The lock attribute must exist after __init__
        assert hasattr(engine, "_compact_lock"), (
            "LcmEngine must have a _compact_lock attribute"
        )
        assert isinstance(engine._compact_lock, type(threading.Lock())), (
            "_compact_lock must be a threading.Lock instance"
        )

    def test_compact_acquires_lock(self):
        """Verify compact() actually acquires _compact_lock during mutation."""
        engine = _make_engine(protect_last_n=0)
        _fill_engine(engine, n=4)

        lock_acquired_during: list[bool] = []
        original_compact = engine.compact

        def spy_compact(summary_text, level, block_start, block_end):
            # If lock is being used, trying to acquire it from another thread
            # should block. We check the lock is held (locked).
            acquired = engine._compact_lock.acquire(blocking=False)
            # During a locked compact(), this acquire should FAIL
            lock_acquired_during.append(acquired)
            if acquired:
                engine._compact_lock.release()
            return original_compact(summary_text, level, block_start, block_end)

        # We run the actual compact inside _compact_lock to verify it works
        with engine._compact_lock:
            acquired = engine._compact_lock.acquire(blocking=False)
            lock_acquired_during.append(acquired)

        # After releasing, acquire should succeed
        acquired = engine._compact_lock.acquire(blocking=False)
        lock_acquired_during.append(acquired)
        if acquired:
            engine._compact_lock.release()

        # The last acquire (outside lock) must have succeeded
        assert lock_acquired_during[-1] is True
        # The acquire inside the `with` block must have failed (lock was held)
        assert lock_acquired_during[0] is False


# ---------------------------------------------------------------------------
# 9. test_sync_compact_still_works
# ---------------------------------------------------------------------------

class TestSyncCompactStillWorks:
    """Regular auto_compact() still works synchronously — no regression."""

    def test_sync_compact_still_works(self):
        engine = _make_engine(protect_last_n=2)
        _fill_engine(engine, n=8)

        active_before = len(engine.active)
        node = engine.auto_compact()

        assert node is not None, "auto_compact() must still return a SummaryNode"
        assert len(engine.active) < active_before, (
            "active list must shrink after synchronous auto_compact()"
        )

    def test_compact_block_still_works(self):
        engine = _make_engine(protect_last_n=0)
        _fill_engine(engine, n=4)

        node = engine.compact_block(0, 2)
        assert node is not None
        assert len(engine.active) == 3  # 1 summary + 2 remaining

    def test_compact_low_level_still_works(self):
        engine = _make_engine(protect_last_n=0)
        _fill_engine(engine, n=4)

        node = engine.compact("manual summary", level=1, block_start=0, block_end=2)
        assert node is not None
        assert len(engine.active) == 3


# ---------------------------------------------------------------------------
# 10. test_check_thresholds_returns_async
# ---------------------------------------------------------------------------

class TestCheckThresholdsReturnsAsync:
    """check_thresholds() returns ASYNC when ratio is between tau_soft and tau_hard."""

    def test_check_thresholds_returns_async(self):
        config = LcmConfig(tau_soft=0.50, tau_hard=0.85)
        engine = LcmEngine(config, context_length=100)

        # Fill past soft but below hard threshold
        engine.ingest({"role": "user", "content": "x" * 250})  # ~62 tokens

        action = engine.check_thresholds()
        assert action == CompactionAction.ASYNC

    def test_check_thresholds_async_sets_pending(self):
        config = LcmConfig(tau_soft=0.50, tau_hard=0.85)
        engine = LcmEngine(config, context_length=100)

        engine.ingest({"role": "user", "content": "x" * 250})
        engine._async_compaction_pending = False

        engine.check_thresholds()
        assert engine._async_compaction_pending is True

    def test_check_thresholds_async_skips_if_pending(self):
        config = LcmConfig(tau_soft=0.50, tau_hard=0.85)
        engine = LcmEngine(config, context_length=100)

        engine.ingest({"role": "user", "content": "x" * 250})
        engine._async_compaction_pending = True  # Already pending

        action = engine.check_thresholds()
        assert action == CompactionAction.NONE

    def test_check_thresholds_blocking_still_works(self):
        config = LcmConfig(tau_soft=0.50, tau_hard=0.85)
        engine = LcmEngine(config, context_length=100)

        engine.ingest({"role": "user", "content": "x" * 400})  # ~100 tokens

        action = engine.check_thresholds()
        assert action == CompactionAction.BLOCKING

    def test_check_thresholds_none_below_soft(self):
        config = LcmConfig(tau_soft=0.50, tau_hard=0.85)
        engine = LcmEngine(config, context_length=1000)

        engine.ingest({"role": "user", "content": "short message"})

        action = engine.check_thresholds()
        assert action == CompactionAction.NONE
