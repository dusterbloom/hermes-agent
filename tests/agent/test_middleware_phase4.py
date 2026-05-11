"""Tests for MessageSanitizationMiddleware.

RED phase: define the contract for middleware that sanitizes tool call
arguments and repairs message role-alternation before each API call.

This extracts the inline logic at lines 12027-12053 of run_conversation().
"""

import pytest
from unittest.mock import MagicMock, call
from agent.loop import LoopContext, MiddlewareBase


class TestMessageSanitizationMiddleware:
    def test_import(self):
        from agent.middleware import MessageSanitizationMiddleware
        assert issubclass(MessageSanitizationMiddleware, MiddlewareBase)

    def test_before_iteration_calls_sanitize_tool_call_arguments(self):
        from agent.middleware import MessageSanitizationMiddleware

        agent = MagicMock()
        agent._sanitize_tool_call_arguments.return_value = 2
        agent._repair_message_sequence.return_value = 0
        agent.session_id = "test-session"
        agent.logger = MagicMock()

        mw = MessageSanitizationMiddleware(agent)
        ctx = LoopContext(max_iterations=10)
        messages = [{"role": "assistant", "tool_calls": [
            {"id": "1", "function": {"name": "test", "arguments": "broken"}}
        ]}]

        mw.before_iteration(ctx, 1, messages=messages)

        agent._sanitize_tool_call_arguments.assert_called_once_with(
            messages,
            logger=agent.logger,
            session_id="test-session",
        )

    def test_before_iteration_calls_repair_message_sequence(self):
        from agent.middleware import MessageSanitizationMiddleware

        agent = MagicMock()
        agent._sanitize_tool_call_arguments.return_value = 0
        agent._repair_message_sequence.return_value = 1
        agent.session_id = "test-session"

        mw = MessageSanitizationMiddleware(agent)
        ctx = LoopContext(max_iterations=10)
        messages = [{"role": "user", "content": "hi"}]

        mw.before_iteration(ctx, 1, messages=messages)

        agent._repair_message_sequence.assert_called_once_with(messages)

    def test_logs_when_tool_calls_sanitized(self):
        from agent.middleware import MessageSanitizationMiddleware

        agent = MagicMock()
        agent._sanitize_tool_call_arguments.return_value = 3
        agent._repair_message_sequence.return_value = 0
        agent.session_id = "sess-123"

        logger = MagicMock()
        mw = MessageSanitizationMiddleware(agent, logger=logger)
        ctx = LoopContext(max_iterations=10)

        mw.before_iteration(ctx, 1, messages=[])

        # Should log the sanitization count
        assert logger.info.called
        log_args = logger.info.call_args[0]
        assert "3" in str(log_args)  # 3 repaired

    def test_logs_when_sequence_repaired(self):
        from agent.middleware import MessageSanitizationMiddleware

        agent = MagicMock()
        agent._sanitize_tool_call_arguments.return_value = 0
        agent._repair_message_sequence.return_value = 2
        agent.session_id = "sess-456"

        logger = MagicMock()
        mw = MessageSanitizationMiddleware(agent, logger=logger)
        ctx = LoopContext(max_iterations=10)

        mw.before_iteration(ctx, 1, messages=[])

        assert logger.info.called
        log_args = logger.info.call_args[0]
        assert "2" in str(log_args)

    def test_no_log_when_nothing_repaired(self):
        from agent.middleware import MessageSanitizationMiddleware

        agent = MagicMock()
        agent._sanitize_tool_call_arguments.return_value = 0
        agent._repair_message_sequence.return_value = 0
        agent.session_id = "sess-clean"

        logger = MagicMock()
        mw = MessageSanitizationMiddleware(agent, logger=logger)
        ctx = LoopContext(max_iterations=10)

        mw.before_iteration(ctx, 1, messages=[])

        logger.info.assert_not_called()

    def test_uses_agent_logger_when_no_explicit_logger(self):
        from agent.middleware import MessageSanitizationMiddleware

        agent = MagicMock()
        agent._sanitize_tool_call_arguments.return_value = 1
        agent._repair_message_sequence.return_value = 0
        agent.session_id = "sess-789"
        agent.logger = MagicMock()

        mw = MessageSanitizationMiddleware(agent)
        ctx = LoopContext(max_iterations=10)

        mw.before_iteration(ctx, 1, messages=[])

        # Should use agent.logger
        agent.logger.info.assert_called()


# ---------------------------------------------------------------------------
# ApiMessageBuilder
# ---------------------------------------------------------------------------

class TestApiMessageBuilder:
    def test_import(self):
        from agent.middleware import ApiMessageBuilder
        assert issubclass(ApiMessageBuilder, MiddlewareBase)

    def test_strips_internal_fields(self):
        """API messages should not contain internal fields."""
        from agent.middleware import ApiMessageBuilder

        agent = MagicMock()
        agent._copy_reasoning_content_for_api = MagicMock()
        agent._should_sanitize_tool_calls.return_value = False

        builder = ApiMessageBuilder(agent, current_turn_user_idx=0)
        ctx = LoopContext(max_iterations=10)

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "reasoning": "thoughts",
             "finish_reason": "stop", "_thinking_prefill": True},
        ]

        builder.before_iteration(ctx, 1, messages=messages)
        api_msgs = builder.last_api_messages

        assert len(api_msgs) == 2
        assert "reasoning" not in api_msgs[1]
        assert "finish_reason" not in api_msgs[1]
        assert "_thinking_prefill" not in api_msgs[1]

    def test_copies_reasoning_content(self):
        """Should call _copy_reasoning_content_for_api for each assistant message."""
        from agent.middleware import ApiMessageBuilder

        agent = MagicMock()
        agent._copy_reasoning_content_for_api = MagicMock()
        agent._should_sanitize_tool_calls.return_value = False

        builder = ApiMessageBuilder(agent, current_turn_user_idx=0)
        ctx = LoopContext(max_iterations=10)

        messages = [
            {"role": "assistant", "content": "hi"},
        ]

        builder.before_iteration(ctx, 1, messages=messages)

        agent._copy_reasoning_content_for_api.assert_called_once()

    def test_injects_context_into_user_message(self):
        """Memory prefetch and plugin context should be injected into the user message."""
        from agent.middleware import ApiMessageBuilder

        agent = MagicMock()
        agent._copy_reasoning_content_for_api = MagicMock()
        agent._should_sanitize_tool_calls.return_value = False

        builder = ApiMessageBuilder(
            agent,
            current_turn_user_idx=0,
            memory_prefetch="cached memory text",
            plugin_context="plugin context",
        )
        ctx = LoopContext(max_iterations=10)

        messages = [
            {"role": "user", "content": "hello"},
        ]

        builder.before_iteration(ctx, 1, messages=messages)
        api_msgs = builder.last_api_messages

        assert "cached memory text" in api_msgs[0]["content"]
        assert "plugin context" in api_msgs[0]["content"]

    def test_does_not_mutate_original_messages(self):
        """Original messages list must remain unchanged."""
        from agent.middleware import ApiMessageBuilder

        agent = MagicMock()
        agent._copy_reasoning_content_for_api = MagicMock()
        agent._should_sanitize_tool_calls.return_value = False

        builder = ApiMessageBuilder(agent, current_turn_user_idx=0)
        ctx = LoopContext(max_iterations=10)

        messages = [
            {"role": "assistant", "content": "hi", "reasoning": "secret"},
        ]
        import copy
        original = copy.deepcopy(messages)

        builder.before_iteration(ctx, 1, messages=messages)

        assert messages == original  # unchanged
