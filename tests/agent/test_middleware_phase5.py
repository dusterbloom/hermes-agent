"""Tests for ApiMessageFinalizer middleware.

RED phase: define the contract for middleware that finalizes the api_messages
list after construction: system prompt assembly, prefill injection, cache
control, sanitize, drop thinking-only, normalize whitespace, strip surrogates.

This extracts the inline logic at lines 12043-12131 of run_conversation().
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from agent.loop import LoopContext, MiddlewareBase


class TestApiMessageFinalizer:
    def test_import(self):
        from agent.middleware import ApiMessageFinalizer
        assert issubclass(ApiMessageFinalizer, MiddlewareBase)

    def test_prepends_system_prompt(self):
        """Should prepend system message when effective_system is set."""
        from agent.middleware import ApiMessageFinalizer

        agent = MagicMock()
        agent.ephemeral_system_prompt = "ephemeral hint"
        agent._use_prompt_caching = False
        agent._sanitize_api_messages = MagicMock(side_effect=lambda x: x)
        agent._drop_thinking_only_and_merge_users = MagicMock(side_effect=lambda x: x)

        mw = ApiMessageFinalizer(agent, active_system_prompt="You are helpful.")
        ctx = LoopContext(max_iterations=10)
        api_msgs = [{"role": "user", "content": "hi"}]

        mw.before_iteration(ctx, 1, messages=api_msgs)
        result = mw.last_finalized_messages

        assert result[0]["role"] == "system"
        assert "You are helpful." in result[0]["content"]
        assert "ephemeral hint" in result[0]["content"]

    def test_no_system_when_empty(self):
        """Should not prepend system message when no system prompt."""
        from agent.middleware import ApiMessageFinalizer

        agent = MagicMock()
        agent.ephemeral_system_prompt = ""
        agent._use_prompt_caching = False
        agent._sanitize_api_messages = MagicMock(side_effect=lambda x: x)
        agent._drop_thinking_only_and_merge_users = MagicMock(side_effect=lambda x: x)

        mw = ApiMessageFinalizer(agent, active_system_prompt="")
        ctx = LoopContext(max_iterations=10)
        api_msgs = [{"role": "user", "content": "hi"}]

        mw.before_iteration(ctx, 1, messages=api_msgs)
        result = mw.last_finalized_messages

        assert all(m["role"] != "system" for m in result)

    def test_injects_prefill_messages(self):
        """Should inject prefill messages after system prompt."""
        from agent.middleware import ApiMessageFinalizer

        agent = MagicMock()
        agent.ephemeral_system_prompt = ""
        agent.prefill_messages = [{"role": "user", "content": "prefill"}]
        agent._use_prompt_caching = False
        agent._sanitize_api_messages = MagicMock(side_effect=lambda x: x)
        agent._drop_thinking_only_and_merge_users = MagicMock(side_effect=lambda x: x)

        mw = ApiMessageFinalizer(agent, active_system_prompt="system")
        ctx = LoopContext(max_iterations=10)
        api_msgs = [{"role": "user", "content": "hi"}]

        mw.before_iteration(ctx, 1, messages=api_msgs)
        result = mw.last_finalized_messages

        # System at [0], prefill at [1], original user at [2]
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "prefill"
        assert result[2]["content"] == "hi"

    def test_calls_sanitize_api_messages(self):
        """Should call _sanitize_api_messages on the result."""
        from agent.middleware import ApiMessageFinalizer

        agent = MagicMock()
        agent.ephemeral_system_prompt = ""
        agent._use_prompt_caching = False
        agent._sanitize_api_messages = MagicMock(side_effect=lambda x: x)
        agent._drop_thinking_only_and_merge_users = MagicMock(side_effect=lambda x: x)

        mw = ApiMessageFinalizer(agent, active_system_prompt="")
        ctx = LoopContext(max_iterations=10)
        api_msgs = [{"role": "user", "content": "hi"}]

        mw.before_iteration(ctx, 1, messages=api_msgs)

        agent._sanitize_api_messages.assert_called_once()

    def test_calls_drop_thinking_only(self):
        """Should call _drop_thinking_only_and_merge_users on the result."""
        from agent.middleware import ApiMessageFinalizer

        agent = MagicMock()
        agent.ephemeral_system_prompt = ""
        agent._use_prompt_caching = False
        agent._sanitize_api_messages = MagicMock(side_effect=lambda x: x)
        agent._drop_thinking_only_and_merge_users = MagicMock(side_effect=lambda x: x)

        mw = ApiMessageFinalizer(agent, active_system_prompt="")
        ctx = LoopContext(max_iterations=10)
        api_msgs = [{"role": "user", "content": "hi"}]

        mw.before_iteration(ctx, 1, messages=api_msgs)

        agent._drop_thinking_only_and_merge_users.assert_called_once()

    def test_strips_content_whitespace(self):
        """Should strip whitespace from string content fields."""
        from agent.middleware import ApiMessageFinalizer

        agent = MagicMock()
        agent.ephemeral_system_prompt = ""
        agent._use_prompt_caching = False
        agent._sanitize_api_messages = MagicMock(side_effect=lambda x: x)
        agent._drop_thinking_only_and_merge_users = MagicMock(side_effect=lambda x: x)

        mw = ApiMessageFinalizer(agent, active_system_prompt="")
        ctx = LoopContext(max_iterations=10)
        api_msgs = [{"role": "user", "content": "  hi  "}]

        mw.before_iteration(ctx, 1, messages=api_msgs)
        result = mw.last_finalized_messages

        assert result[0]["content"] == "hi"

    def test_normalizes_tool_call_json(self):
        """Should normalize tool_call arguments to sorted compact JSON."""
        from agent.middleware import ApiMessageFinalizer

        agent = MagicMock()
        agent.ephemeral_system_prompt = ""
        agent._use_prompt_caching = False
        agent._sanitize_api_messages = MagicMock(side_effect=lambda x: x)
        agent._drop_thinking_only_and_merge_users = MagicMock(side_effect=lambda x: x)
        agent._repair_tool_call_arguments = MagicMock(side_effect=lambda a, n: a)

        mw = ApiMessageFinalizer(agent, active_system_prompt="")
        ctx = LoopContext(max_iterations=10)
        api_msgs = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "tc1",
                "type": "function",
                "function": {
                    "name": "test",
                    "arguments": '{"z":1,"a":2}',
                },
            }],
        }]

        mw.before_iteration(ctx, 1, messages=api_msgs)
        result = mw.last_finalized_messages

        # Arguments should be sorted: {"a":2,"z":1}
        assert result[0]["tool_calls"][0]["function"]["arguments"] == '{"a":2,"z":1}'

    def test_does_not_mutate_input(self):
        """Input messages should not be mutated."""
        from agent.middleware import ApiMessageFinalizer

        agent = MagicMock()
        agent.ephemeral_system_prompt = ""
        agent._use_prompt_caching = False
        agent._sanitize_api_messages = MagicMock(side_effect=lambda x: x)
        agent._drop_thinking_only_and_merge_users = MagicMock(side_effect=lambda x: x)

        mw = ApiMessageFinalizer(agent, active_system_prompt="")
        ctx = LoopContext(max_iterations=10)
        api_msgs = [{"role": "user", "content": "  hi  "}]

        mw.before_iteration(ctx, 1, messages=api_msgs)

        # Input should be unchanged
        assert api_msgs[0]["content"] == "  hi  "

    def test_empty_messages_produces_empty_result(self):
        """Empty input should produce empty output."""
        from agent.middleware import ApiMessageFinalizer

        agent = MagicMock()
        agent.ephemeral_system_prompt = ""
        agent._use_prompt_caching = False
        agent._sanitize_api_messages = MagicMock(side_effect=lambda x: x)
        agent._drop_thinking_only_and_merge_users = MagicMock(side_effect=lambda x: x)

        mw = ApiMessageFinalizer(agent, active_system_prompt="")
        ctx = LoopContext(max_iterations=10)

        mw.before_iteration(ctx, 1, messages=[])

        assert mw.last_finalized_messages == []
