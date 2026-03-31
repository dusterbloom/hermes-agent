"""Tests for the unified memory_* tool surface (agent/lcm/hrr/schemas.py + tools.py).

Coverage:
- Schema validation: 6 schemas, each with name/description/parameters
- memory_search: DAM -> HRR -> keyword fallback routing
- memory_pin: delegates to lcm_pin + crystallizes to HRR
- memory_expand: session source -> lcm_expand, memory source -> HRR query
- memory_forget: delegates to lcm_forget, lower_trust -> record_feedback
- memory_reason: probe/related/reason/contradict actions
- memory_budget: delegates to lcm_budget
- No-engine guard: RuntimeError when engine not registered
- Backward compat: old lcm_* and dam_* tools still importable and functional
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.lcm.config import LcmConfig
from agent.lcm.engine import LcmEngine
from agent.lcm.tools import get_engine, set_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(n_messages: int = 3) -> LcmEngine:
    """Return a minimal populated LcmEngine with no HRR store."""
    config = LcmConfig(enabled=True, protect_last_n=2)
    engine = LcmEngine(config)
    engine.hrr_store = None  # disable HRR by default; tests opt-in
    for i in range(n_messages):
        engine.ingest({"role": "user", "content": f"test message {i}"})
    return engine


def _make_mock_hrr_store():
    """Return a mock MemoryStore with common methods pre-configured."""
    store = MagicMock()
    store.add_fact.return_value = 1
    store.search_facts.return_value = [
        {"fact_id": 1, "content": "a persistent fact", "trust_score": 0.8, "category": "general"}
    ]
    store.record_feedback.return_value = {
        "fact_id": 1, "old_trust": 0.8, "new_trust": 0.7, "helpful_count": 0
    }
    return store


def _make_mock_retriever():
    """Return a mock DAMRetriever."""
    r = MagicMock()
    r.search.return_value = [(0, 0.95)]  # list of (msg_id, score)
    return r


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------


class TestMemoryToolSchemas:
    """All 6 schemas must exist and be well-formed."""

    def test_import_memory_tool_schemas(self):
        from agent.lcm.hrr.schemas import MEMORY_TOOL_SCHEMAS  # noqa: F401

    def test_exactly_six_schemas(self):
        from agent.lcm.hrr.schemas import MEMORY_TOOL_SCHEMAS

        assert len(MEMORY_TOOL_SCHEMAS) == 6, (
            f"Expected 6 schemas, got {len(MEMORY_TOOL_SCHEMAS)}: "
            f"{list(MEMORY_TOOL_SCHEMAS.keys())}"
        )

    @pytest.mark.parametrize("tool_name", [
        "memory_search",
        "memory_pin",
        "memory_expand",
        "memory_forget",
        "memory_reason",
        "memory_budget",
    ])
    def test_schema_has_required_keys(self, tool_name: str):
        from agent.lcm.hrr.schemas import MEMORY_TOOL_SCHEMAS

        schema = MEMORY_TOOL_SCHEMAS[tool_name]
        assert "name" in schema, f"{tool_name}: missing 'name'"
        assert "description" in schema, f"{tool_name}: missing 'description'"
        assert "parameters" in schema, f"{tool_name}: missing 'parameters'"

    @pytest.mark.parametrize("tool_name", [
        "memory_search",
        "memory_pin",
        "memory_expand",
        "memory_forget",
        "memory_reason",
        "memory_budget",
    ])
    def test_schema_name_matches_key(self, tool_name: str):
        from agent.lcm.hrr.schemas import MEMORY_TOOL_SCHEMAS

        assert MEMORY_TOOL_SCHEMAS[tool_name]["name"] == tool_name

    def test_memory_search_has_query_required(self):
        from agent.lcm.hrr.schemas import MEMORY_TOOL_SCHEMAS

        schema = MEMORY_TOOL_SCHEMAS["memory_search"]
        assert "query" in schema["parameters"]["required"]

    def test_memory_search_source_is_enum(self):
        from agent.lcm.hrr.schemas import MEMORY_TOOL_SCHEMAS

        props = MEMORY_TOOL_SCHEMAS["memory_search"]["parameters"]["properties"]
        assert "source" in props
        assert "enum" in props["source"]
        assert set(props["source"]["enum"]) == {"auto", "session", "memory"}

    def test_memory_pin_message_ids_is_array(self):
        from agent.lcm.hrr.schemas import MEMORY_TOOL_SCHEMAS

        props = MEMORY_TOOL_SCHEMAS["memory_pin"]["parameters"]["properties"]
        assert props["message_ids"]["type"] == "array"
        assert props["message_ids"]["items"]["type"] == "integer"

    def test_memory_forget_lower_trust_is_boolean(self):
        from agent.lcm.hrr.schemas import MEMORY_TOOL_SCHEMAS

        props = MEMORY_TOOL_SCHEMAS["memory_forget"]["parameters"]["properties"]
        assert props["lower_trust"]["type"] == "boolean"

    def test_memory_reason_action_enum(self):
        from agent.lcm.hrr.schemas import MEMORY_TOOL_SCHEMAS

        props = MEMORY_TOOL_SCHEMAS["memory_reason"]["parameters"]["properties"]
        assert set(props["action"]["enum"]) == {"probe", "related", "reason", "contradict"}

    def test_memory_budget_has_no_required_params(self):
        from agent.lcm.hrr.schemas import MEMORY_TOOL_SCHEMAS

        schema = MEMORY_TOOL_SCHEMAS["memory_budget"]
        assert schema["parameters"]["required"] == []


# ---------------------------------------------------------------------------
# 2. No-engine guard
# ---------------------------------------------------------------------------


class TestRequireEngineGuard:
    """All handlers must raise RuntimeError when no engine is registered."""

    def setup_method(self):
        set_engine(None)

    def teardown_method(self):
        set_engine(None)

    @pytest.mark.parametrize("handler_name,args", [
        ("handle_memory_search", {"query": "test"}),
        ("handle_memory_pin", {"message_ids": [0]}),
        ("handle_memory_expand", {"message_ids": [0]}),
        ("handle_memory_forget", {"message_ids": [0]}),
        ("handle_memory_reason", {"entities": ["test"]}),
        ("handle_memory_budget", {}),
    ])
    def test_raises_without_engine(self, handler_name: str, args: dict):
        import agent.lcm.hrr.tools as t

        handler = getattr(t, handler_name)
        with pytest.raises(RuntimeError, match="not initialized"):
            handler(args)


# ---------------------------------------------------------------------------
# 3. memory_search
# ---------------------------------------------------------------------------


class TestMemorySearch:

    def setup_method(self):
        self.engine = _make_engine()
        set_engine(self.engine)

    def teardown_method(self):
        set_engine(None)

    def test_search_keyword_fallback_when_dam_and_hrr_absent(self):
        """When no DAM and no HRR, falls back to LCM keyword search."""
        self.engine.retriever = None
        self.engine.hrr_store = None
        from agent.lcm.hrr.tools import handle_memory_search

        result = handle_memory_search({"query": "message"})
        # LCM keyword search finds "test message 0/1/2"
        assert "msg" in result or "No results" in result

    def test_search_source_session_uses_dam(self):
        """source='session' calls engine.retriever.search()."""
        mock_retriever = _make_mock_retriever()
        self.engine.retriever = mock_retriever
        from agent.lcm.hrr.tools import handle_memory_search

        result = handle_memory_search({"query": "hello", "source": "session"})
        mock_retriever.search.assert_called_once_with("hello", limit=10)
        assert "Session msg#0" in result

    def test_search_source_memory_skips_dam(self):
        """source='memory' must NOT call engine.retriever.search()."""
        mock_retriever = _make_mock_retriever()
        self.engine.retriever = mock_retriever
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr

        from agent.lcm.hrr.retrieval import FactRetriever
        from agent.lcm.hrr.tools import handle_memory_search

        with patch.object(FactRetriever, "search", return_value=[
            {"fact_id": 42, "content": "cross-session fact", "trust_score": 0.9}
        ]):
            result = handle_memory_search({"query": "cross", "source": "memory"})

        mock_retriever.search.assert_not_called()
        assert "Memory fact#42" in result

    def test_search_auto_queries_both_layers(self):
        """source='auto' (default) calls both DAM and HRR."""
        mock_retriever = _make_mock_retriever()
        self.engine.retriever = mock_retriever
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr

        from agent.lcm.hrr.retrieval import FactRetriever
        from agent.lcm.hrr.tools import handle_memory_search

        with patch.object(FactRetriever, "search", return_value=[
            {"fact_id": 7, "content": "hrr result", "trust_score": 0.7}
        ]):
            result = handle_memory_search({"query": "anything"})

        mock_retriever.search.assert_called_once()
        assert "Session" in result or "Memory" in result

    def test_search_empty_query_returns_error(self):
        from agent.lcm.hrr.tools import handle_memory_search

        result = handle_memory_search({"query": ""})
        assert "Error" in result or "required" in result

    def test_search_no_results_returns_not_found(self):
        """When all layers return nothing, report no results."""
        self.engine.retriever = None
        self.engine.hrr_store = None
        from agent.lcm.hrr.tools import handle_memory_search

        result = handle_memory_search({"query": "xyzzy_not_found_anywhere"})
        # Either LCM keyword search says no matches, or our fallback message
        assert "No results" in result or "No matches" in result


# ---------------------------------------------------------------------------
# 4. memory_pin
# ---------------------------------------------------------------------------


class TestMemoryPin:

    def setup_method(self):
        self.engine = _make_engine()
        set_engine(self.engine)

    def teardown_method(self):
        set_engine(None)

    def test_pin_delegates_to_lcm_pin(self):
        from agent.lcm.hrr.tools import handle_memory_pin

        result = handle_memory_pin({"message_ids": [0]})
        assert "Pinned" in result
        assert 0 in self.engine._pinned_ids

    def test_pin_crystallizes_to_hrr_store(self):
        """Pinned messages must be added to hrr_store if present."""
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr
        from agent.lcm.hrr.tools import handle_memory_pin

        handle_memory_pin({"message_ids": [0], "reason": "important"})
        mock_hrr.add_fact.assert_called_once()
        call_kwargs = mock_hrr.add_fact.call_args
        assert call_kwargs.kwargs.get("category") == "pinned" or (
            len(call_kwargs.args) > 1 and call_kwargs.args[1] == "pinned"
        )

    def test_pin_without_hrr_still_pins(self):
        """Pinning must succeed even when hrr_store is None."""
        self.engine.hrr_store = None
        from agent.lcm.hrr.tools import handle_memory_pin

        result = handle_memory_pin({"message_ids": [0]})
        assert "Pinned" in result

    def test_pin_multiple_ids(self):
        from agent.lcm.hrr.tools import handle_memory_pin

        result = handle_memory_pin({"message_ids": [0, 1]})
        assert 0 in self.engine._pinned_ids
        assert 1 in self.engine._pinned_ids

    def test_pin_reports_crystallization_count(self):
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr
        from agent.lcm.hrr.tools import handle_memory_pin

        result = handle_memory_pin({"message_ids": [0, 1]})
        assert "Crystallized" in result

    def test_pin_passes_reason_as_tags(self):
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr
        from agent.lcm.hrr.tools import handle_memory_pin

        handle_memory_pin({"message_ids": [0], "reason": "critical reference"})
        call = mock_hrr.add_fact.call_args
        # Check tags kwarg contains the reason
        tags = call.kwargs.get("tags") or (call.args[2] if len(call.args) > 2 else "")
        assert "critical reference" in tags


# ---------------------------------------------------------------------------
# 5. memory_expand
# ---------------------------------------------------------------------------


class TestMemoryExpand:

    def setup_method(self):
        self.engine = _make_engine()
        set_engine(self.engine)

    def teardown_method(self):
        set_engine(None)

    def test_expand_session_delegates_to_lcm_expand(self):
        """source='session' (default) calls handle_lcm_expand."""
        from agent.lcm.hrr.tools import handle_memory_expand

        # msg 0 exists in the store
        result = handle_memory_expand({"message_ids": [0], "source": "session"})
        # lcm_expand returns content of msg 0
        assert "test message 0" in result

    def test_expand_default_source_is_session(self):
        """When source is omitted, defaults to 'session' (lcm_expand)."""
        from agent.lcm.hrr.tools import handle_memory_expand

        result = handle_memory_expand({"message_ids": [0]})
        assert "test message 0" in result

    def test_expand_memory_with_no_hrr_returns_helpful_message(self):
        """source='memory' without hrr_store returns a clear error."""
        self.engine.hrr_store = None
        from agent.lcm.hrr.tools import handle_memory_expand

        result = handle_memory_expand({"query": "anything", "source": "memory"})
        assert "not available" in result.lower()

    def test_expand_memory_with_no_query_returns_helpful_message(self):
        """source='memory' without query returns a guidance message."""
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr
        from agent.lcm.hrr.tools import handle_memory_expand

        result = handle_memory_expand({"source": "memory"})
        assert "query" in result.lower()

    def test_expand_memory_queries_hrr(self):
        """source='memory' + query calls FactRetriever.search."""
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr

        from agent.lcm.hrr.retrieval import FactRetriever
        from agent.lcm.hrr.tools import handle_memory_expand

        with patch.object(FactRetriever, "search", return_value=[
            {"fact_id": 3, "content": "recalled fact", "trust_score": 0.6, "category": "general"}
        ]) as mock_search:
            result = handle_memory_expand({"query": "recall this", "source": "memory"})

        mock_search.assert_called_once()
        assert "recalled fact" in result

    def test_expand_memory_no_results(self):
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr

        from agent.lcm.hrr.retrieval import FactRetriever
        from agent.lcm.hrr.tools import handle_memory_expand

        with patch.object(FactRetriever, "search", return_value=[]):
            result = handle_memory_expand({"query": "nothing here", "source": "memory"})

        assert "No persistent memories" in result


# ---------------------------------------------------------------------------
# 6. memory_forget
# ---------------------------------------------------------------------------


class TestMemoryForget:

    def setup_method(self):
        self.engine = _make_engine(n_messages=5)
        set_engine(self.engine)

    def teardown_method(self):
        set_engine(None)

    def test_forget_delegates_to_lcm_forget(self):
        from agent.lcm.hrr.tools import handle_memory_forget

        result = handle_memory_forget({"message_ids": [0]})
        # lcm_forget returns "Compacted N message(s)"
        assert "Compacted" in result or "No matching" in result

    def test_forget_lower_trust_false_no_hrr_call(self):
        """lower_trust=False must not touch hrr_store."""
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr
        from agent.lcm.hrr.tools import handle_memory_forget

        handle_memory_forget({"message_ids": [0], "lower_trust": False})
        mock_hrr.record_feedback.assert_not_called()

    def test_forget_lower_trust_true_calls_record_feedback(self):
        """lower_trust=True must call hrr_store.record_feedback with helpful=False."""
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr
        from agent.lcm.hrr.tools import handle_memory_forget

        handle_memory_forget({"message_ids": [0], "lower_trust": True})
        # record_feedback should have been called for matching facts
        mock_hrr.search_facts.assert_called()
        if mock_hrr.search_facts.return_value:
            mock_hrr.record_feedback.assert_called_with(
                mock_hrr.search_facts.return_value[0]["fact_id"], helpful=False
            )

    def test_forget_without_hrr_still_forgets(self):
        """forget must succeed even when hrr_store is None."""
        self.engine.hrr_store = None
        from agent.lcm.hrr.tools import handle_memory_forget

        result = handle_memory_forget({"message_ids": [0]})
        assert "Compacted" in result or "No matching" in result

    def test_forget_with_reason(self):
        from agent.lcm.hrr.tools import handle_memory_forget

        result = handle_memory_forget({"message_ids": [0], "reason": "outdated"})
        assert "outdated" in result or "Compacted" in result

    def test_forget_list_ids_normalised(self):
        """List-form message_ids must be normalised to comma-string for LCM."""
        from agent.lcm.hrr.tools import handle_memory_forget

        # Should not raise; LCM expects comma-separated string internally
        result = handle_memory_forget({"message_ids": [0, 1]})
        assert "Compacted" in result or "No matching" in result


# ---------------------------------------------------------------------------
# 7. memory_reason
# ---------------------------------------------------------------------------


class TestMemoryReason:

    def setup_method(self):
        self.engine = _make_engine()
        set_engine(self.engine)

    def teardown_method(self):
        set_engine(None)

    def test_reason_without_hrr_returns_clear_message(self):
        self.engine.hrr_store = None
        from agent.lcm.hrr.tools import handle_memory_reason

        result = handle_memory_reason({"entities": ["peppi"]})
        assert "not available" in result.lower()

    def test_reason_empty_entities_returns_error(self):
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr
        from agent.lcm.hrr.tools import handle_memory_reason

        result = handle_memory_reason({"entities": []})
        assert "Error" in result

    def test_reason_default_action_is_reason(self):
        """Default action='reason' calls retriever.reason()."""
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr

        from agent.lcm.hrr.retrieval import FactRetriever
        from agent.lcm.hrr.tools import handle_memory_reason

        with patch.object(FactRetriever, "reason", return_value=[
            {"content": "fact about peppi", "score": 0.8, "trust_score": 0.9}
        ]) as mock_reason:
            result = handle_memory_reason({"entities": ["peppi"]})

        mock_reason.assert_called_once_with(["peppi"], limit=10)
        assert "fact about peppi" in result

    def test_reason_probe_action(self):
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr

        from agent.lcm.hrr.retrieval import FactRetriever
        from agent.lcm.hrr.tools import handle_memory_reason

        with patch.object(FactRetriever, "probe", return_value=[
            {"content": "probed fact", "score": 0.7, "trust_score": 0.8}
        ]) as mock_probe:
            result = handle_memory_reason({"entities": ["alice"], "action": "probe"})

        mock_probe.assert_called_once_with("alice", limit=10)
        assert "probed fact" in result

    def test_reason_probe_requires_single_entity(self):
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr
        from agent.lcm.hrr.tools import handle_memory_reason

        result = handle_memory_reason({"entities": ["a", "b"], "action": "probe"})
        assert "Error" in result and "one entity" in result

    def test_reason_related_action(self):
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr

        from agent.lcm.hrr.retrieval import FactRetriever
        from agent.lcm.hrr.tools import handle_memory_reason

        with patch.object(FactRetriever, "related", return_value=[
            {"content": "related fact", "score": 0.6, "trust_score": 0.7}
        ]) as mock_related:
            result = handle_memory_reason({"entities": ["bob"], "action": "related"})

        mock_related.assert_called_once_with("bob", limit=10)
        assert "related fact" in result

    def test_reason_related_requires_single_entity(self):
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr
        from agent.lcm.hrr.tools import handle_memory_reason

        result = handle_memory_reason({"entities": ["a", "b"], "action": "related"})
        assert "Error" in result and "one entity" in result

    def test_reason_contradict_action(self):
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr

        from agent.lcm.hrr.retrieval import FactRetriever
        from agent.lcm.hrr.tools import handle_memory_reason

        with patch.object(FactRetriever, "contradict", return_value=[
            {
                "fact_a": {"content": "A is true"},
                "fact_b": {"content": "A is false"},
                "contradiction_score": 0.75,
                "shared_entities": ["A"],
            }
        ]):
            result = handle_memory_reason({"entities": ["any"], "action": "contradict"})

        assert "Contradiction" in result
        assert "A is true" in result
        assert "A is false" in result

    def test_reason_contradict_no_results(self):
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr

        from agent.lcm.hrr.retrieval import FactRetriever
        from agent.lcm.hrr.tools import handle_memory_reason

        with patch.object(FactRetriever, "contradict", return_value=[]):
            result = handle_memory_reason({"entities": ["any"], "action": "contradict"})

        assert "No contradictions" in result

    def test_reason_no_results_returns_not_found(self):
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr

        from agent.lcm.hrr.retrieval import FactRetriever
        from agent.lcm.hrr.tools import handle_memory_reason

        with patch.object(FactRetriever, "reason", return_value=[]):
            result = handle_memory_reason({"entities": ["unknown_entity"]})

        assert "No facts found" in result

    def test_reason_score_format_in_output(self):
        """Result lines must include score and trust."""
        mock_hrr = _make_mock_hrr_store()
        self.engine.hrr_store = mock_hrr

        from agent.lcm.hrr.retrieval import FactRetriever
        from agent.lcm.hrr.tools import handle_memory_reason

        with patch.object(FactRetriever, "reason", return_value=[
            {"content": "scored fact", "score": 0.55, "trust_score": 0.65}
        ]):
            result = handle_memory_reason({"entities": ["x"]})

        assert "0.55" in result
        assert "0.65" in result


# ---------------------------------------------------------------------------
# 8. memory_budget
# ---------------------------------------------------------------------------


class TestMemoryBudget:

    def setup_method(self):
        self.engine = _make_engine()
        set_engine(self.engine)

    def teardown_method(self):
        set_engine(None)

    def test_budget_returns_lcm_budget_output(self):
        """memory_budget must return the same output as lcm_budget."""
        from agent.lcm.hrr.tools import handle_memory_budget
        from agent.lcm.tools import handle_lcm_budget

        budget_result = handle_memory_budget({})
        lcm_result = handle_lcm_budget({})
        assert budget_result == lcm_result

    def test_budget_contains_expected_fields(self):
        from agent.lcm.hrr.tools import handle_memory_budget

        result = handle_memory_budget({})
        assert "Active" in result
        assert "tokens" in result.lower()

    def test_budget_accepts_empty_args(self):
        from agent.lcm.hrr.tools import handle_memory_budget

        result = handle_memory_budget({})
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# 9. Handler dispatch table
# ---------------------------------------------------------------------------


class TestHandlerDispatchTable:

    def test_dispatch_table_exists(self):
        from agent.lcm.hrr.tools import MEMORY_TOOL_HANDLERS  # noqa: F401

    def test_dispatch_table_has_six_entries(self):
        from agent.lcm.hrr.tools import MEMORY_TOOL_HANDLERS

        assert len(MEMORY_TOOL_HANDLERS) == 6

    @pytest.mark.parametrize("tool_name", [
        "memory_search",
        "memory_pin",
        "memory_expand",
        "memory_forget",
        "memory_reason",
        "memory_budget",
    ])
    def test_each_handler_is_callable(self, tool_name: str):
        from agent.lcm.hrr.tools import MEMORY_TOOL_HANDLERS

        handler = MEMORY_TOOL_HANDLERS[tool_name]
        assert callable(handler), f"{tool_name}: handler must be callable"


# ---------------------------------------------------------------------------
# 10. Backward compatibility — old tools still importable and functional
# ---------------------------------------------------------------------------


class TestBackwardCompatLcmTools:
    """The old lcm_* tools must still be importable and return results."""

    def setup_method(self):
        self.engine = _make_engine()
        set_engine(self.engine)

    def teardown_method(self):
        set_engine(None)

    def test_import_lcm_tool_schemas(self):
        from agent.lcm.tools import LCM_TOOL_SCHEMAS  # noqa: F401

    def test_lcm_tool_schemas_has_seven_entries(self):
        from agent.lcm.tools import LCM_TOOL_SCHEMAS

        assert len(LCM_TOOL_SCHEMAS) == 7, (
            f"Expected 7 lcm schemas, got {len(LCM_TOOL_SCHEMAS)}"
        )

    def test_handle_lcm_budget_works(self):
        from agent.lcm.tools import handle_lcm_budget

        result = handle_lcm_budget({})
        assert "LCM Context Budget" in result

    def test_handle_lcm_search_works(self):
        from agent.lcm.tools import handle_lcm_search

        result = handle_lcm_search({"query": "message"})
        assert "msg" in result or "No matches" in result

    def test_handle_lcm_pin_works(self):
        from agent.lcm.tools import handle_lcm_pin

        result = handle_lcm_pin({"message_ids": "0"})
        assert "Pinned" in result

    def test_handle_lcm_expand_works(self):
        from agent.lcm.tools import handle_lcm_expand

        result = handle_lcm_expand({"message_ids": "0"})
        assert "test message 0" in result

    def test_handle_lcm_forget_works(self):
        from agent.lcm.tools import handle_lcm_forget

        result = handle_lcm_forget({"message_ids": "0"})
        assert "Compacted" in result or "No matching" in result

    def test_handle_lcm_toc_works(self):
        from agent.lcm.tools import handle_lcm_toc

        result = handle_lcm_toc({})
        assert "Timeline" in result or "empty" in result.lower()


class TestBackwardCompatDamTools:
    """The old dam_* tools must still be importable and functional."""

    def test_import_dam_schemas(self):
        from agent.lcm.dam.schemas import DAM_SEARCH, DAM_RECALL, DAM_COMPOSE, DAM_STATUS  # noqa: F401

    def test_import_dam_tool_handlers(self):
        from agent.lcm.dam.tools import (  # noqa: F401
            handle_dam_search,
            handle_dam_recall,
            handle_dam_compose,
            handle_dam_status,
        )

    def test_dam_search_fallback_without_retriever(self):
        """dam_search must fall back gracefully when retriever is None."""
        engine = _make_engine()
        set_engine(engine)
        engine.retriever = None
        try:
            from agent.lcm.dam.tools import handle_dam_search

            # When DAM not initialized, falls back to LCM keyword search or returns message
            result = handle_dam_search({"query": "message"})
            assert isinstance(result, str)
        finally:
            set_engine(None)

    def test_dam_status_without_retriever(self):
        from agent.lcm.dam.tools import handle_dam_status

        result = handle_dam_status({})
        assert isinstance(result, str)
        assert len(result) > 0
