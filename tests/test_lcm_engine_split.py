"""TDD tests for engine.py split into focused modules.

RED phase: these tests fail until the split is implemented.
GREEN phase: after creating query.py, format.py, session.py as mixin modules.
"""
import pytest

from agent.lcm.config import LcmConfig


def _make_engine():
    from agent.lcm.engine import LcmEngine
    return LcmEngine(LcmConfig())


# ---------------------------------------------------------------------------
# 1. Import tests (module existence)
# ---------------------------------------------------------------------------

class TestModuleImports:
    def test_import_query_mixin(self):
        """agent.lcm.query must exist and export LcmQueryMixin."""
        from agent.lcm.query import LcmQueryMixin
        assert LcmQueryMixin is not None

    def test_import_format_mixin(self):
        """agent.lcm.format must exist and export LcmFormatMixin."""
        from agent.lcm.format import LcmFormatMixin
        assert LcmFormatMixin is not None

    def test_import_session_mixin(self):
        """agent.lcm.session must exist and export LcmSessionMixin."""
        from agent.lcm.session import LcmSessionMixin
        assert LcmSessionMixin is not None


# ---------------------------------------------------------------------------
# 2. Inheritance tests
# ---------------------------------------------------------------------------

class TestEngineInheritance:
    def test_engine_inherits_query_mixin(self):
        from agent.lcm.query import LcmQueryMixin
        engine = _make_engine()
        assert isinstance(engine, LcmQueryMixin)

    def test_engine_inherits_format_mixin(self):
        from agent.lcm.format import LcmFormatMixin
        engine = _make_engine()
        assert isinstance(engine, LcmFormatMixin)

    def test_engine_inherits_session_mixin(self):
        from agent.lcm.session import LcmSessionMixin
        engine = _make_engine()
        assert isinstance(engine, LcmSessionMixin)


# ---------------------------------------------------------------------------
# 3. Method presence tests
# ---------------------------------------------------------------------------

class TestQueryMethodsOnEngine:
    """Query methods must still be callable on LcmEngine instances."""

    def test_active_messages_present(self):
        engine = _make_engine()
        assert callable(engine.active_messages)

    def test_active_tokens_present(self):
        engine = _make_engine()
        assert callable(engine.active_tokens)

    def test_active_token_breakdown_present(self):
        engine = _make_engine()
        assert callable(engine.active_token_breakdown)

    def test_search_present(self):
        engine = _make_engine()
        assert callable(engine.search)

    def test_keyword_search_present(self):
        engine = _make_engine()
        assert callable(engine._keyword_search)

    def test_build_semantic_index_present(self):
        engine = _make_engine()
        assert callable(engine.build_semantic_index)


class TestFormatMethodsOnEngine:
    """Formatting methods must still be callable on LcmEngine instances."""

    def test_format_expanded_present(self):
        engine = _make_engine()
        assert callable(engine.format_expanded)

    def test_format_toc_present(self):
        engine = _make_engine()
        assert callable(engine.format_toc)

    def test_format_budget_present(self):
        engine = _make_engine()
        assert callable(engine.format_budget)


class TestSessionMethodsOnEngine:
    """Session persistence methods must still be callable on LcmEngine instances."""

    def test_rebuild_from_session_present(self):
        engine = _make_engine()
        assert callable(engine.rebuild_from_session)

    def test_to_session_metadata_present(self):
        engine = _make_engine()
        assert callable(engine.to_session_metadata)

    def test_reset_present(self):
        engine = _make_engine()
        assert callable(engine.reset)


# ---------------------------------------------------------------------------
# 4. Backward compatibility: existing imports still work
# ---------------------------------------------------------------------------

class TestBackwardCompatImports:
    def test_backward_compat_engine_module(self):
        """from agent.lcm.engine import LcmEngine, CompactionAction, ContextEntry."""
        from agent.lcm.engine import LcmEngine, CompactionAction, ContextEntry
        assert LcmEngine is not None
        assert CompactionAction is not None
        assert ContextEntry is not None

    def test_backward_compat_init_exports(self):
        """from agent.lcm import LcmEngine must still work."""
        from agent.lcm import LcmEngine
        assert LcmEngine is not None

    def test_backward_compat_all_init_exports(self):
        """All previously exported names must still be importable from agent.lcm."""
        from agent.lcm import (
            LcmConfig,
            SummaryDag, SummaryNode, MessageId,
            ImmutableStore,
            LcmEngine, CompactionAction, ContextEntry,
            Summarizer, SummarizerConfig,
            TokenEstimator, TokenEstimatorConfig, estimate_messages_tokens_rough,
            SemanticIndex, SemanticIndexConfig, NoOpSemanticIndex, create_semantic_index,
        )
        # All must be non-None
        names = [
            LcmConfig, SummaryDag, SummaryNode, MessageId,
            ImmutableStore, LcmEngine, CompactionAction, ContextEntry,
            Summarizer, SummarizerConfig, TokenEstimator, TokenEstimatorConfig,
            estimate_messages_tokens_rough, SemanticIndex, SemanticIndexConfig,
            NoOpSemanticIndex, create_semantic_index,
        ]
        for name in names:
            assert name is not None


# ---------------------------------------------------------------------------
# 5. Core methods remain in engine.py (not only inherited from mixins)
# ---------------------------------------------------------------------------

class TestEngineCoreStaysInEngine:
    """Core compaction methods must be defined directly on LcmEngine (not inherited from mixins)."""

    def _engine_class_defines(self, method_name: str) -> bool:
        """Return True if LcmEngine itself (not a mixin) defines the method."""
        from agent.lcm.engine import LcmEngine
        from agent.lcm.query import LcmQueryMixin
        from agent.lcm.format import LcmFormatMixin
        from agent.lcm.session import LcmSessionMixin

        # Check it's not defined in any mixin
        for mixin in (LcmQueryMixin, LcmFormatMixin, LcmSessionMixin):
            if method_name in mixin.__dict__:
                return False

        # Check it IS in LcmEngine's own dict
        return method_name in LcmEngine.__dict__

    def test_ingest_in_engine(self):
        assert self._engine_class_defines("ingest")

    def test_compact_in_engine(self):
        assert self._engine_class_defines("compact")

    def test_auto_compact_in_engine(self):
        assert self._engine_class_defines("auto_compact")

    def test_check_thresholds_in_engine(self):
        assert self._engine_class_defines("check_thresholds")

    def test_find_compactable_block_in_engine(self):
        assert self._engine_class_defines("find_compactable_block")


# ---------------------------------------------------------------------------
# 6. Functional smoke tests — methods actually work after the split
# ---------------------------------------------------------------------------

class TestFunctionalAfterSplit:
    """Verify the extracted methods still produce correct results."""

    def test_query_active_messages_returns_list(self):
        engine = _make_engine()
        engine.ingest({"role": "user", "content": "hello"})
        msgs = engine.active_messages()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hello"

    def test_query_active_tokens_returns_int(self):
        engine = _make_engine()
        engine.ingest({"role": "user", "content": "hello world"})
        tokens = engine.active_tokens()
        assert isinstance(tokens, (int, float))
        assert tokens > 0

    def test_query_search_returns_list(self):
        engine = _make_engine()
        engine.ingest({"role": "user", "content": "find this needle"})
        results = engine.search("needle")
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_format_toc_returns_string(self):
        engine = _make_engine()
        engine.ingest({"role": "user", "content": "hello"})
        toc = engine.format_toc()
        assert isinstance(toc, str)
        assert len(toc) > 0

    def test_format_budget_returns_string(self):
        engine = _make_engine()
        budget = engine.format_budget()
        assert isinstance(budget, str)
        assert "LCM Context Budget" in budget

    def test_format_expanded_returns_string(self):
        engine = _make_engine()
        msg_id = engine.ingest({"role": "user", "content": "hello"})
        result = engine.format_expanded([msg_id])
        assert isinstance(result, str)
        assert "hello" in result

    def test_session_round_trip(self):
        """to_session_metadata + rebuild_from_session must preserve state."""
        from agent.lcm.engine import LcmEngine
        config = LcmConfig()
        engine = LcmEngine(config)
        engine.ingest({"role": "user", "content": "hello"})
        engine.ingest({"role": "assistant", "content": "hi"})

        meta = engine.to_session_metadata()
        session_data = {"messages": engine.active_messages(), "lcm": meta}

        engine2 = LcmEngine.rebuild_from_session(session_data, config)
        assert len(engine2.active) == 2

    def test_session_reset_clears_state(self):
        engine = _make_engine()
        engine.ingest({"role": "user", "content": "hello"})
        assert len(engine.active) == 1

        engine.reset()
        assert len(engine.active) == 0
        assert len(engine.store) == 0
