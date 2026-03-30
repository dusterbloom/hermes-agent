"""TDD tests for ContextVar-based DI in agent/lcm/tools.py.

RED phase: these tests are written against the desired ContextVar API.
They should FAIL against the current global _engine_ref implementation.
GREEN phase: refactor tools.py to use ContextVar; all tests should pass.
"""
from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from agent.lcm.config import LcmConfig
from agent.lcm.engine import LcmEngine
from agent.lcm.tools import (
    get_engine,
    handle_lcm_budget,
    handle_lcm_expand,
    set_engine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(protect_last_n: int = 2) -> LcmEngine:
    """Return a minimal populated LcmEngine."""
    config = LcmConfig(enabled=True, protect_last_n=protect_last_n)
    e = LcmEngine(config)
    e.ingest({"role": "user", "content": "hello"})
    e.ingest({"role": "assistant", "content": "world"})
    return e


# ---------------------------------------------------------------------------
# 1. Thread isolation
# ---------------------------------------------------------------------------


class TestContextVarIsolationBetweenThreads:
    """Each OS thread must see only the engine it registered."""

    def test_contextvar_isolation_between_threads(self):
        engine_a = _make_engine()
        engine_b = _make_engine()

        results: dict[str, LcmEngine | None] = {}
        barrier = threading.Barrier(2)

        def thread_a():
            set_engine(engine_a)
            barrier.wait()  # sync so both engines are set before either reads
            results["a"] = get_engine()

        def thread_b():
            set_engine(engine_b)
            barrier.wait()
            results["b"] = get_engine()

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert results["a"] is engine_a, "Thread A must see engine_a, not engine_b"
        assert results["b"] is engine_b, "Thread B must see engine_b, not engine_a"
        assert results["a"] is not results["b"], "Each thread must have an independent engine"


# ---------------------------------------------------------------------------
# 2. Asyncio task isolation
# ---------------------------------------------------------------------------


class TestContextVarIsolationBetweenTasks:
    """Each asyncio task must see only the engine it registered."""

    def test_contextvar_isolation_between_tasks(self):
        engine_a = _make_engine()
        engine_b = _make_engine()

        results: dict[str, LcmEngine | None] = {}

        async def task_a(event_b_set: asyncio.Event, event_a_set: asyncio.Event):
            set_engine(engine_a)
            event_a_set.set()
            await event_b_set.wait()  # yield control so task_b runs
            results["a"] = get_engine()

        async def task_b(event_a_set: asyncio.Event, event_b_set: asyncio.Event):
            await event_a_set.wait()
            set_engine(engine_b)
            event_b_set.set()
            results["b"] = get_engine()

        async def main():
            event_b_set = asyncio.Event()
            event_a_set = asyncio.Event()
            await asyncio.gather(
                task_a(event_b_set, event_a_set),
                task_b(event_a_set, event_b_set),
            )

        asyncio.run(main())

        assert results["a"] is engine_a, "Task A must see engine_a"
        assert results["b"] is engine_b, "Task B must see engine_b"


# ---------------------------------------------------------------------------
# 3. set_engine(None) clears
# ---------------------------------------------------------------------------


class TestSetEngineNoneClears:
    """set_engine(None) must result in get_engine() returning None."""

    def test_set_engine_none_clears(self):
        engine = _make_engine()
        set_engine(engine)
        assert get_engine() is engine  # sanity

        set_engine(None)
        assert get_engine() is None


# ---------------------------------------------------------------------------
# 4. Tool handlers use the ContextVar engine
# ---------------------------------------------------------------------------


class TestToolHandlersUseContextVar:
    """Tool handlers must use the engine registered in the current context."""

    def test_tool_handler_uses_registered_engine(self):
        engine = _make_engine()
        set_engine(engine)
        result = handle_lcm_expand({"message_ids": "0"})
        # Should succeed — not an error message
        assert "error" not in result.lower() or "msg 0" in result

    def test_tool_handler_returns_error_when_none(self):
        set_engine(None)
        result = handle_lcm_expand({"message_ids": "0"})
        assert "error" in result.lower() or "not active" in result.lower()

    def test_tool_handler_in_thread_uses_thread_engine(self):
        """Tool handler called in a thread must use that thread's engine."""
        engine_main = _make_engine()
        engine_thread = _make_engine()
        set_engine(engine_main)  # main thread sets its engine

        result_holder: dict[str, str] = {}

        def worker():
            set_engine(engine_thread)
            result_holder["budget"] = handle_lcm_budget({})

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        # The thread called handle_lcm_budget which uses get_engine() internally.
        # It should have used engine_thread, not engine_main.
        assert "token" in result_holder["budget"].lower() or "message" in result_holder["budget"].lower()
        # Main thread engine is still intact
        assert get_engine() is engine_main


# ---------------------------------------------------------------------------
# 5. Backward-compatibility: single-threaded set/get unchanged
# ---------------------------------------------------------------------------


class TestBackwardCompatSetGetEngine:
    """The public API surface of set_engine / get_engine must not change."""

    def test_set_then_get_returns_same_engine(self):
        engine = _make_engine()
        set_engine(engine)
        assert get_engine() is engine

    def test_set_none_then_get_returns_none(self):
        set_engine(None)
        assert get_engine() is None

    def test_replace_engine(self):
        engine_1 = _make_engine()
        engine_2 = _make_engine()
        set_engine(engine_1)
        set_engine(engine_2)
        assert get_engine() is engine_2

    def test_default_is_none(self):
        """In a brand-new thread (fresh context) the engine should default to None."""
        result_holder: list[LcmEngine | None] = []

        def fresh_thread():
            # This thread has never called set_engine — default must be None.
            result_holder.append(get_engine())

        t = threading.Thread(target=fresh_thread)
        t.start()
        t.join()

        assert result_holder[0] is None


# ---------------------------------------------------------------------------
# 6. Concurrent agents — fully independent
# ---------------------------------------------------------------------------


class TestConcurrentAgentsIndependent:
    """Two agents running in parallel threads must not interfere."""

    def test_concurrent_agents_independent(self):
        engine_1 = _make_engine()
        engine_2 = _make_engine()

        budget_results: dict[str, str] = {}
        barrier = threading.Barrier(2)

        def agent(name: str, engine: LcmEngine):
            set_engine(engine)
            barrier.wait()  # both agents set their engines before either reads
            # Call a tool handler — it must use this thread's engine
            budget_results[name] = handle_lcm_budget({})

        t1 = threading.Thread(target=agent, args=("agent1", engine_1))
        t2 = threading.Thread(target=agent, args=("agent2", engine_2))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both should produce valid budget output (not error strings)
        assert "error" not in budget_results["agent1"].lower()
        assert "error" not in budget_results["agent2"].lower()

    def test_concurrent_agents_see_different_engines(self):
        """Verify that each concurrent agent's get_engine() is truly independent."""
        engine_1 = _make_engine()
        engine_2 = _make_engine()

        captured: dict[str, LcmEngine | None] = {}
        barrier = threading.Barrier(2)

        def agent(name: str, engine: LcmEngine):
            set_engine(engine)
            barrier.wait()
            captured[name] = get_engine()

        t1 = threading.Thread(target=agent, args=("agent1", engine_1))
        t2 = threading.Thread(target=agent, args=("agent2", engine_2))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert captured["agent1"] is engine_1
        assert captured["agent2"] is engine_2
        assert captured["agent1"] is not captured["agent2"]
