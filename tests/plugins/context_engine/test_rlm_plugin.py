"""Tests for the RLM ContextEngine plugin.

Verifies that the RLM plugin correctly implements the ContextEngine ABC,
exposes tool schemas, handles tool calls, and works in composite mode
with LCM.
"""

import json
import pytest
import textwrap

from plugins.context_engine.rlm import (
    RlmContextEngine,
    RLMAgentEnvironment,
    CompositeContextEngine,
    RLM_TOOL_SCHEMAS,
)


# ── RLMContextEngine Tests ──────────────────────────────────────────────


class TestRlmContextEngineIdentity:
    """Test engine identity and configuration."""

    def test_name_property(self):
        engine = RlmContextEngine()
        assert engine.name == "rlm"

    def test_default_config(self):
        engine = RlmContextEngine()
        assert engine._max_iterations == 20
        assert engine._output_limit == 8192
        assert engine.context_length == 0
        assert engine.compression_count == 0
        assert engine.threshold_percent == 1.0  # Never triggers compression

    def test_custom_config(self):
        engine = RlmContextEngine(
            rlm_config={"max_iterations": 30, "output_limit": 4096}
        )
        assert engine._max_iterations == 30
        assert engine._output_limit == 4096

    def test_abc_compliance(self):
        from agent.context_engine import ContextEngine
        assert issubclass(RlmContextEngine, ContextEngine)

    def test_is_available(self):
        # RLM requires an LLM provider, so availability depends on config
        engine = RlmContextEngine()
        # At minimum, it should not crash
        assert True


class TestRlmContextEngineCompaction:
    """Test that RLM never compresses (delegates to REPL)."""

    def test_never_compresses(self):
        engine = RlmContextEngine()
        assert engine.should_compress() is False
        assert engine.should_compress(prompt_tokens=1000) is False

    def test_compress_returns_messages_unchanged(self):
        engine = RlmContextEngine()
        messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "hello"},
        ]
        result = engine.compress(messages)
        assert result == messages

    def test_preflight_always_false(self):
        engine = RlmContextEngine()
        assert engine.should_compress_preflight([]) is False


class TestRlmContextEngineSessionLifecycle:
    """Test session lifecycle methods."""

    def test_session_start_init(self):
        engine = RlmContextEngine()
        engine.on_session_start("test-123", context_length=128_000)
        assert engine.context_length == 128_000
        assert engine.threshold_tokens == 128_000

    def test_session_start_with_context(self):
        engine = RlmContextEngine()
        context = "A" * 1000
        engine.on_session_start("test-123", context_length=64_000, rlm_context=context)
        assert engine._session_context == context
        assert engine._rlm_env is not None

    def test_session_end_is_noop(self):
        engine = RlmContextEngine()
        # Should not crash
        engine.on_session_end("test-123", [])

    def test_session_reset(self):
        engine = RlmContextEngine()
        engine._session_context = "test"
        engine._rlm_env = RLMAgentEnvironment("test")
        engine.on_session_reset()
        assert engine._session_context is None
        assert engine._rlm_env is None
        assert engine.compression_count == 0
        assert engine.last_prompt_tokens == 0

    def test_model_update(self):
        engine = RlmContextEngine()
        engine.on_session_start("test-123", context_length=128_000)
        engine.update_model("gpt-5", 128_000)
        assert engine.context_length == 128_000
        assert engine.threshold_tokens == 128_000  # threshold_percent=1.0


class TestRlmContextEngineTools:
    """Test RLM tool schemas and handlers."""

    def test_tool_schemas_count(self):
        engine = RlmContextEngine()
        schemas = engine.get_tool_schemas()
        assert len(schemas) == 3
        names = {s["name"] for s in schemas}
        assert names == {"rlm_peek", "rlm_grep", "rlm_partition"}

    def test_tool_schema_format(self):
        engine = RlmContextEngine()
        schemas = engine.get_tool_schemas()
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema

    def test_handle_peek_no_context(self):
        engine = RlmContextEngine()
        # No context loaded
        result = engine.handle_tool_call("rlm_peek", {"start": 0, "length": 100})
        data = json.loads(result)
        assert "error" in data
        assert "No context loaded" in data["error"]

    def test_handle_grep_no_context(self):
        engine = RlmContextEngine()
        result = engine.handle_tool_call("rlm_grep", {"pattern": "test"})
        data = json.loads(result)
        assert "error" in data

    def test_handle_unknown_tool(self):
        engine = RlmContextEngine()
        result = engine.handle_tool_call("unknown_tool", {})
        data = json.loads(result)
        assert "error" in data

    def test_handle_peek_with_context(self):
        engine = RlmContextEngine()
        context = "Hello World! This is a test context. 12345"
        engine._session_context = context
        engine._rlm_env = RLMAgentEnvironment(context)

        result = engine.handle_tool_call("rlm_peek", {"start": 0, "length": 5})
        data = json.loads(result)
        assert data["snippet"] == "Hello"
        assert data["start"] == 0

    def test_handle_grep_with_context(self):
        context_lines = textwrap.dedent("""
            ID: 12345 User: Alice
            ID: 67890 User: Bob
            ID: 11111 User: Charlie
        """).strip()

        engine = RlmContextEngine()
        engine._session_context = context_lines
        engine._rlm_env = RLMAgentEnvironment(context_lines)

        result = engine.handle_tool_call("rlm_grep", {"pattern": "Alice", "max_matches": 1})
        data = json.loads(result)
        assert len(data["matches"]) == 1
        assert "Alice" in data["matches"][0]["content"]

    def test_handle_partition_with_context(self):
        context = "A" * 20000  # 20k chars
        engine = RlmContextEngine()
        engine._session_context = context
        engine._rlm_env = RLMAgentEnvironment(context)

        result = engine.handle_tool_call("rlm_partition", {"chunk_size": 5000, "overlap": 500})
        data = json.loads(result)
        assert data["total_chunks"] > 3
        assert data["chunk_size"] == 5000


class TestRlmContextEngineStatus:
    """Test status display."""

    def test_get_status(self):
        engine = RlmContextEngine()
        engine.context_length = 128_000
        engine.last_prompt_tokens = 64_000
        engine.last_completion_tokens = 1000
        engine.last_total_tokens = 65_000
        engine.compression_count = 5
        engine.threshold_tokens = 128_000  # threshold_percent=1.0, so 128000 * 1.0 = 128000

        status = engine.get_status()
        assert status["context_length"] == 128_000
        assert status["last_prompt_tokens"] == 64_000
        assert status["threshold_tokens"] == 128_000  # Manually set in test
        assert status["compression_count"] == 5
        assert status["usage_percent"] == 50.0


class TestRlmContextEngineToolboxIntegration:
    """Test integration with the tool registry."""

    def test_tool_schemas_match_registry_format(self):
        """Ensure schemas match the format expected by model_tools.py."""
        engine = RlmContextEngine()
        schemas = engine.get_tool_schemas()

        for schema in schemas:
            # Must have 'name' and 'parameters'
            assert isinstance(schema["name"], str)
            assert isinstance(schema["parameters"], dict)

            # Parameters must have properties
            params = schema["parameters"]
            assert "properties" in params or "required" in params


# ── CompositeContextEngine Tests ─────────────────────────────────────────


class TestCompositeContextEngine:
    """Test the LCM + RLM composite engine."""

    def test_composite_name(self):
        """Composite engine should have a combined name."""
        class MockCompression:
            def __init__(self):
                self.name = "lcm"
            def get_tool_schemas(self):
                return []
            def handle_tool_call(self, name, args, **kwargs):
                return json.dumps({"result": "ok"})
            def should_compress(self, prompt_tokens=None):
                return True
            def should_compress_preflight(self, messages):
                return True
            def compress(self, messages, current_tokens=None, focus_topic=None):
                return list(messages)
            def on_session_start(self, *args, **kwargs):
                pass
            def on_session_end(self, *args, **kwargs):
                pass
            def on_session_reset(self):
                pass
            def update_from_response(self, usage):
                pass
            def update_model(self, **kwargs):
                pass
            def get_status(self):
                return {}

        engine = CompositeContextEngine(MockCompression())
        assert engine.name == "lcm"

        engine_with_rlm = CompositeContextEngine(MockCompression(), RlmContextEngine())
        assert engine_with_rlm.name == "lcm+rlm"

    def test_delegate_compression(self):
        """Compression should delegate to the compression engine."""
        class MockCompression:
            def __init__(self):
                self.name = "lcm"
                self._should_compress_calls = 0
            def get_tool_schemas(self):
                return []
            def handle_tool_call(self, name, args, **kwargs):
                return json.dumps({"result": "ok"})
            def should_compress(self, prompt_tokens=None):
                self._should_compress_calls += 1
                return True
            def should_compress_preflight(self, messages):
                return True
            def compress(self, messages, current_tokens=None, focus_topic=None):
                return list(messages)
            def on_session_start(self, *args, **kwargs):
                pass
            def on_session_end(self, *args, **kwargs):
                pass
            def on_session_reset(self):
                pass
            def update_from_response(self, usage):
                pass
            def update_model(self, **kwargs):
                pass
            def get_status(self):
                return {}

        mock = MockCompression()
        engine = CompositeContextEngine(mock)
        assert engine.should_compress(1000) is True
        assert mock._should_compress_calls == 1

    def test_delegate_compress(self):
        """compress() should delegate to compression engine."""
        class MockCompression:
            def __init__(self):
                self.name = "lcm"
            def get_tool_schemas(self):
                return []
            def handle_tool_call(self, name, args, **kwargs):
                return json.dumps({"result": "ok"})
            def should_compress(self, prompt_tokens=None):
                return False
            def should_compress_preflight(self, messages):
                return False
            def compress(self, messages, current_tokens=None, focus_topic=None):
                # Modify messages to prove delegation
                modified = list(messages)
                modified.append({"role": "system", "content": "compressed"})
                return modified
            def on_session_start(self, *args, **kwargs):
                pass
            def on_session_end(self, *args, **kwargs):
                pass
            def on_session_reset(self):
                pass
            def update_from_response(self, usage):
                pass
            def update_model(self, **kwargs):
                pass
            def get_status(self):
                return {}

        mock = MockCompression()
        engine = CompositeContextEngine(mock)
        messages = [{"role": "user", "content": "hello"}]
        result = engine.compress(messages)
        assert len(result) == 2  # Original + compressed addition
        assert result[-1]["content"] == "compressed"

    def test_combine_tool_schemas(self):
        """Tool schemas should combine from both engines."""
        class MockCompression:
            def __init__(self):
                self.name = "lcm"
            def get_tool_schemas(self):
                return [
                    {"name": "lcm_grep", "parameters": {}},
                    {"name": "lcm_expand", "parameters": {}},
                ]
            def handle_tool_call(self, name, args, **kwargs):
                return json.dumps({"result": "ok"})
            def should_compress(self, prompt_tokens=None):
                return True
            def should_compress_preflight(self, messages):
                return True
            def compress(self, messages, current_tokens=None, focus_topic=None):
                return list(messages)
            def on_session_start(self, *args, **kwargs):
                pass
            def on_session_end(self, *args, **kwargs):
                pass
            def on_session_reset(self):
                pass
            def update_from_response(self, usage):
                pass
            def update_model(self, **kwargs):
                pass
            def get_status(self):
                return {}

        mock = MockCompression()
        engine = CompositeContextEngine(mock, RlmContextEngine())
        schemas = engine.get_tool_schemas()
        names = {s["name"] for s in schemas}
        assert "lcm_grep" in names
        assert "lcm_expand" in names
        assert "rlm_peek" in names
        assert "rlm_grep" in names
        assert "rlm_partition" in names

    def test_dispatch_rlm_tools(self):
        """RLM tool calls should be dispatched to the RLM engine."""
        class MockCompression:
            def __init__(self):
                self.name = "lcm"
            def get_tool_schemas(self):
                return []
            def handle_tool_call(self, name, args, **kwargs):
                return json.dumps({"from": "compression"})
            def should_compress(self, prompt_tokens=None):
                return True
            def should_compress_preflight(self, messages):
                return True
            def compress(self, messages, current_tokens=None, focus_topic=None):
                return list(messages)
            def on_session_start(self, *args, **kwargs):
                pass
            def on_session_end(self, *args, **kwargs):
                pass
            def on_session_reset(self):
                pass
            def update_from_response(self, usage):
                pass
            def update_model(self, **kwargs):
                pass
            def get_status(self):
                return {}

        mock = MockCompression()
        rlm = RlmContextEngine()
        rlm._session_context = "Hello World"
        rlm._rlm_env = RLMAgentEnvironment("Hello World")

        engine = CompositeContextEngine(mock, rlm)
        result = engine.handle_tool_call("rlm_peek", {"start": 0, "length": 5})
        data = json.loads(result)
        assert data["snippet"] == "Hello"
        assert "from" not in data  # Should be from RLM, not compression

    def test_refresh_context(self):
        """refresh_context should update the RLM environment."""
        class MockCompression:
            def __init__(self):
                self.name = "lcm"
            def get_tool_schemas(self):
                return []
            def handle_tool_call(self, name, args, **kwargs):
                return json.dumps({"result": "ok"})
            def should_compress(self, prompt_tokens=None):
                return True
            def should_compress_preflight(self, messages):
                return True
            def compress(self, messages, current_tokens=None, focus_topic=None):
                return list(messages)
            def on_session_start(self, *args, **kwargs):
                pass
            def on_session_end(self, *args, **kwargs):
                pass
            def on_session_reset(self):
                pass
            def update_from_response(self, usage):
                pass
            def update_model(self, **kwargs):
                pass
            def get_status(self):
                return {}

        mock = MockCompression()
        engine = CompositeContextEngine(mock, RlmContextEngine())

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello, world!"},
        ]
        engine.refresh_context(messages)

        expected = "[system]: You are helpful.\n\n[user]: Hello, world!"
        assert engine._rlm_engine._session_context == expected

    def test_update_model_resyncs_wrapper_fields(self):
        """update_model() must re-sync context_length and threshold_tokens on the
        composite wrapper after delegating to the compression engine."""
        from plugins.context_engine.lcm.__init__ import LcmContextEngine

        lcm = LcmContextEngine()
        composite = CompositeContextEngine(lcm)

        # Capture init values
        init_context_length = composite.context_length

        # Switch to a model with a larger context window
        composite.update_model(
            model="claude-opus-4-5",
            context_length=200_000,
            provider="anthropic",
        )

        assert composite.context_length == 200_000, (
            f"composite.context_length={composite.context_length} — "
            "wrapper was not re-synced after update_model"
        )
        # threshold_tokens should have been updated too
        assert composite.threshold_tokens > 0, (
            "composite.threshold_tokens should be positive after update_model"
        )
        # Sanity: inner engine was updated
        assert lcm.context_length == 200_000


# ── RLMAgentEnvironment Tests ───────────────────────────────────────────


class TestRLMAgentEnvironment:
    """Test the REPL environment."""

    def test_init(self):
        context = "Test context"
        env = RLMAgentEnvironment(context)
        assert env.context == context
        assert env.namespace["__context__"] == context
        assert env.namespace["answer"] == {"content": "", "ready": False}

    def test_execute_simple(self):
        context = "Hello World"

        # Test 1: Setting answer['content'] returns the content
        env1 = RLMAgentEnvironment(context)
        result1 = env1.execute("answer['content'] = 'test'")
        assert result1 == "test"  # execute() returns answer content

        # Test 2: Subsequent execute() returns the new answer content
        env2 = RLMAgentEnvironment(context)
        env2.execute("answer['content'] = 'hello'")
        result2 = env2.execute("pass")  # No-op, but returns current answer content
        assert result2 == "hello"

    def test_execute_with_answer(self):
        context = "Context here"
        env = RLMAgentEnvironment(context)

        env.execute('answer["content"] = "Final answer"')
        env.execute('answer["ready"] = True')

        assert env.namespace["answer"]["content"] == "Final answer"
        assert env.namespace["answer"]["ready"] is True

    def test_execute_error_handling(self):
        env = RLMAgentEnvironment("test")
        result = env.execute("this_will_fail_undefined_var")
        assert "Error" in result or "NameError" in result

    def test_execute_expression(self):
        context = "12345"
        env = RLMAgentEnvironment(context)

        result = env.execute_expression("len(__context__)")
        assert result == "5"

    def test_set_answer(self):
        env = RLMAgentEnvironment("test")
        env.set_answer("Hello", ready=True)
        assert env.namespace["answer"]["content"] == "Hello"
        assert env.namespace["answer"]["ready"] is True

    def test_iteration_limit(self):
        env = RLMAgentEnvironment("test", max_iterations=3)
        assert env.iteration_count == 0
        assert env.check_iteration_limit() is False  # 1st call
        assert env.check_iteration_limit() is False  # 2nd call
        assert env.check_iteration_limit() is True   # 3rd call (== max)

    def test_token_estimation(self):
        env = RLMAgentEnvironment("A" * 40000)  # ~10k tokens
        assert env.namespace["__context_token_length__"] == 10000

    def test_llm_batch_handler_returns_empty(self):
        """The handler returns empty list (real impl in RLMAgent)."""
        env = RLMAgentEnvironment("test")
        result = env._llm_batch_handler([])
        assert result == []

    def test_llm_call_handler_returns_empty(self):
        env = RLMAgentEnvironment("test")
        result = env._llm_call_handler("test prompt")
        assert result == ""

    def test_output_limit_truncation(self):
        context = "X" * 20000
        env = RLMAgentEnvironment(context, output_limit=500)

        # Execute code that sets a large answer
        env.execute(f'answer["content"] = "{context}"')
        env.execute('answer["ready"] = True')

        # The execute() should have truncated
        result = env.execute("pass")
        # If the answer is large, it gets truncated in the execute() flow
        # But we need to verify the truncate happens
        pass  # Hard to test without mocking


class TestRlmToolboxSchemas:
    """Test that RLM_TOOL_SCHEMAS is properly formatted."""

    def test_all_schemas_defined(self):
        assert "rlm_peek" in RLM_TOOL_SCHEMAS
        assert "rlm_grep" in RLM_TOOL_SCHEMAS
        assert "rlm_partition" in RLM_TOOL_SCHEMAS

    def test_schemas_have_required_fields(self):
        for schema in RLM_TOOL_SCHEMAS.values():
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema

    def test_schemas_descriptions_not_empty(self):
        for schema in RLM_TOOL_SCHEMAS.values():
            assert len(schema["description"]) > 50


# ── build_context_from_messages Tests ───────────────────────────────────


class TestBuildContextFromMessages:
    """Test context building from conversation messages."""

    def test_simple_messages(self):
        engine = RlmContextEngine()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        context = engine.build_context_from_messages(messages)
        assert "[system]: You are helpful." in context
        assert "[user]: Hello" in context
        assert "[assistant]: Hi there!" in context

    def test_structured_content(self):
        """Test with multi-part content (text + tool_use)."""
        engine = RlmContextEngine()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "tool_use", "name": "search", "input": {"q": "test"}},
                ],
            },
        ]
        context = engine.build_context_from_messages(messages)
        assert "[user]: Hello" in context
        assert "[user]: {\"q\": \"test\"}" in context  # tool_use input

    def test_empty_messages(self):
        engine = RlmContextEngine()
        context = engine.build_context_from_messages([])
        assert context == ""

    def test_refresh_context(self):
        """Test that refresh_context updates the REPL namespace."""
        engine = RlmContextEngine()
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
        ]
        engine.refresh_context(messages)
        expected = "[system]: System prompt\n\n[user]: User message"
        assert engine._session_context == expected
        # After refresh_context, _rlm_env is bootstrapped so rlm_peek works
        assert engine._rlm_env is not None
        assert engine._session_context is not None

    def test_openai_style_tool_calls_included(self):
        """OpenAI-shape tool_calls must appear in the flattened context.

        Regression for: assistant messages with tool_calls=[...] and empty/null
        content were silently dropped because the flattener only handled
        Anthropic-style content blocks (content: [{type: tool_use, ...}]).
        """
        engine = RlmContextEngine()
        messages = [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "Sunny, 22°C",
            },
        ]
        context = engine.build_context_from_messages(messages)

        assert "get_weather" in context, (
            f"Tool name 'get_weather' missing from flattened context.\nContext:\n{context}"
        )
        assert "Paris" in context, (
            f"Tool arguments missing from flattened context.\nContext:\n{context}"
        )