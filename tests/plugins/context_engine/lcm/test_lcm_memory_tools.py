"""Regression tests for Fix C: memory_* tools exposed via get_tool_schemas / handle_tool_call.

Verifies:
- get_tool_schemas() returns both lcm_* AND memory_* schemas.
- handle_tool_call("memory_search", ...) dispatches correctly without raising
  "Unknown tool" / "not active".
"""
from __future__ import annotations

import json
import pytest

from plugins.context_engine.lcm.__init__ import LcmContextEngine
from plugins.context_engine.lcm.config import LcmConfig


@pytest.fixture
def engine():
    config = LcmConfig(enabled=True, protect_last_n=2)
    e = LcmContextEngine(lcm_config=config)
    # Pre-load a few messages so searches have something to find
    e.compress([
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris is the capital of France."},
        {"role": "user", "content": "Tell me about Python decorators."},
        {"role": "assistant", "content": "Decorators are a Python feature for wrapping functions."},
    ])
    return e


class TestGetToolSchemas:
    def test_lcm_tools_present(self, engine):
        """All 7 lcm_* tool names must appear in the schema list."""
        schemas = engine.get_tool_schemas()
        names = {s["name"] for s in schemas}
        lcm_expected = {"lcm_expand", "lcm_pin", "lcm_forget", "lcm_search",
                        "lcm_budget", "lcm_toc", "lcm_focus"}
        assert lcm_expected.issubset(names), (
            f"Missing lcm_* tools: {lcm_expected - names}"
        )

    def test_memory_tools_present(self, engine):
        """All 5 memory_* tool names must appear in the schema list."""
        schemas = engine.get_tool_schemas()
        names = {s["name"] for s in schemas}
        memory_expected = {"memory_search", "memory_pin", "memory_expand",
                           "memory_forget", "memory_reason"}
        assert memory_expected.issubset(names), (
            f"Missing memory_* tools: {memory_expected - names}"
        )

    def test_schemas_have_required_fields(self, engine):
        """Every schema must have name, description, and parameters."""
        for schema in engine.get_tool_schemas():
            assert "name" in schema, f"Schema missing 'name': {schema}"
            assert "description" in schema, f"Schema {schema['name']} missing 'description'"
            assert "parameters" in schema, f"Schema {schema['name']} missing 'parameters'"

    def test_no_duplicate_names(self, engine):
        """Tool names must be unique — no double-exposure from merging."""
        schemas = engine.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert len(names) == len(set(names)), (
            f"Duplicate tool names in schema list: {[n for n in names if names.count(n) > 1]}"
        )


class TestHandleToolCallMemory:
    def test_memory_search_dispatches_without_error(self, engine):
        """handle_tool_call('memory_search', ...) must not return 'Unknown LCM tool'."""
        result = engine.handle_tool_call("memory_search", {"query": "France"})
        # Should be valid JSON or a plain string result — not an "unknown tool" error
        assert "Unknown LCM tool" not in result, (
            f"memory_search was not dispatched: {result}"
        )

    def test_memory_search_returns_string(self, engine):
        """handle_tool_call result must be a non-empty string."""
        result = engine.handle_tool_call("memory_search", {"query": "Python"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_memory_search_no_results_is_not_error(self, engine):
        """A query with no results must return a graceful message, not an exception."""
        result = engine.handle_tool_call("memory_search", {"query": "nonexistent topic xyz"})
        assert isinstance(result, str)
        # Should not contain exception traceback markers
        assert "Traceback" not in result
        assert "Error" not in result or "No results" in result

    def test_unknown_tool_returns_error_json(self, engine):
        """A truly unknown tool must still return the 'Unknown LCM tool' error."""
        result = engine.handle_tool_call("nonexistent_tool", {})
        data = json.loads(result)
        assert "error" in data

    def test_lcm_tool_still_works_after_memory_integration(self, engine):
        """Existing lcm_search must still work after the memory_* merge."""
        result = engine.handle_tool_call("lcm_search", {"query": "Python"})
        assert isinstance(result, str)
        assert "Unknown LCM tool" not in result
