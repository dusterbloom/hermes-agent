"""Tests for agent/local_protocol.py."""

import pytest

from agent.local_protocol import (
    _effective_role,
    _enforce_alternation,
    _fold_system_messages,
    _get_content_str,
    _inject_continue_sentinels,
    _is_private_ip,
    detect_local_server_type,
    format_messages_for_local,
    is_local_endpoint,
)


# ---------------------------------------------------------------------------
# _fold_system_messages
# ---------------------------------------------------------------------------


def test_fold_system_messages_first_preserved():
    """The first system message (index 0) is kept as-is."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    result = _fold_system_messages(messages)
    assert result[0] == {"role": "system", "content": "You are helpful."}
    assert result[1] == {"role": "user", "content": "Hello"}


def test_fold_system_messages_mid_thread_becomes_user():
    """System messages after index 0 become user messages with [System notice] prefix."""
    messages = [
        {"role": "system", "content": "Initial prompt."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "system", "content": "New instructions."},
        {"role": "user", "content": "Continue"},
    ]
    result = _fold_system_messages(messages)
    assert result[0]["role"] == "system"
    assert result[3]["role"] == "user"
    assert result[3]["content"] == "[System notice] New instructions."
    assert result[4]["role"] == "user"


def test_fold_system_messages_no_system():
    """Messages without any system message pass through unchanged."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    result = _fold_system_messages(messages)
    assert result == messages


# ---------------------------------------------------------------------------
# _inject_continue_sentinels
# ---------------------------------------------------------------------------


def test_inject_continue_after_tool_last_message():
    """Sentinel injected when tool result is the last message."""
    messages = [
        {"role": "user", "content": "Run tool"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "result", "tool_call_id": "1"},
    ]
    result = _inject_continue_sentinels(messages)
    assert result[-1] == {"role": "user", "content": "Continue."}


def test_inject_continue_after_tool_before_non_user():
    """Sentinel injected when tool result is followed by an assistant message."""
    messages = [
        {"role": "tool", "content": "result1", "tool_call_id": "1"},
        {"role": "assistant", "content": "Done"},
    ]
    result = _inject_continue_sentinels(messages)
    # Sentinel should be between tool and assistant
    tool_idx = next(i for i, m in enumerate(result) if m.get("role") == "tool")
    assert result[tool_idx + 1] == {"role": "user", "content": "Continue."}


def test_inject_continue_not_between_consecutive_tool_results():
    """No sentinel is injected between consecutive tool results (batch)."""
    messages = [
        {"role": "tool", "content": "result1", "tool_call_id": "1"},
        {"role": "tool", "content": "result2", "tool_call_id": "2"},
    ]
    result = _inject_continue_sentinels(messages)
    # Sentinel only after the last tool result
    tool_roles = [m["role"] for m in result]
    assert tool_roles.count("tool") == 2
    assert result[-1] == {"role": "user", "content": "Continue."}
    # No sentinel between the two tool messages
    assert result[1]["role"] == "tool"


def test_inject_continue_no_injection_before_user():
    """No sentinel when tool result is already followed by a user message."""
    messages = [
        {"role": "tool", "content": "result", "tool_call_id": "1"},
        {"role": "user", "content": "Thanks"},
    ]
    result = _inject_continue_sentinels(messages)
    assert len(result) == 2
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "Thanks"


# ---------------------------------------------------------------------------
# _enforce_alternation
# ---------------------------------------------------------------------------


def test_enforce_alternation_merges_consecutive_user():
    """Consecutive user messages are merged with double newline separator."""
    messages = [
        {"role": "user", "content": "First"},
        {"role": "user", "content": "Second"},
    ]
    result = _enforce_alternation(messages)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "First\n\nSecond"


def test_enforce_alternation_merges_consecutive_assistant():
    """Consecutive assistant messages (without tool_calls) are merged."""
    messages = [
        {"role": "assistant", "content": "Part one."},
        {"role": "assistant", "content": "Part two."},
    ]
    result = _enforce_alternation(messages)
    assert len(result) == 1
    assert result[0]["content"] == "Part one.\n\nPart two."


def test_alternation_preserves_tool_calls():
    """Assistant messages with tool_calls are NOT merged with adjacent messages."""
    messages = [
        {"role": "assistant", "content": "Thinking..."},
        {"role": "assistant", "content": "Using tool", "tool_calls": [{"id": "1"}]},
        {"role": "assistant", "content": "Done"},
    ]
    result = _enforce_alternation(messages)
    # The tool_calls message must remain separate
    tool_call_msgs = [m for m in result if m.get("tool_calls")]
    assert len(tool_call_msgs) == 1
    assert tool_call_msgs[0]["tool_calls"] == [{"id": "1"}]


def test_alternation_system_preserved_at_index_0():
    """System message at index 0 is never merged with the next message."""
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Hello"},
    ]
    result = _enforce_alternation(messages)
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"


def test_tool_messages_treated_as_user_for_alternation():
    """Tool role counts as user for alternation — merged with adjacent user message."""
    messages = [
        {"role": "tool", "content": "result", "tool_call_id": "1"},
        {"role": "user", "content": "Next question"},
    ]
    result = _enforce_alternation(messages)
    # tool + user have effective_role "user", so they merge
    assert len(result) == 1
    assert "result" in result[0]["content"]
    assert "Next question" in result[0]["content"]


def test_enforce_alternation_empty():
    """Empty list returns empty list."""
    assert _enforce_alternation([]) == []


# ---------------------------------------------------------------------------
# format_messages_for_local — full pipeline
# ---------------------------------------------------------------------------


def test_format_messages_full_pipeline():
    """End-to-end: system fold + sentinel inject + alternation all applied.

    Pipeline order: fold_system -> inject_continue -> enforce_alternation.

    The mid-thread system message is first folded into a user message with a
    [System notice] prefix, then alternation merges tool + user (same effective
    role) so the notice text ends up inside the tool message's content.
    The final result must not end on a bare tool message.
    """
    messages = [
        {"role": "system", "content": "Initial."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "content": "tool output", "tool_call_id": "t1"},
        {"role": "system", "content": "Mid-thread notice."},
    ]
    result = format_messages_for_local(messages)

    # The [System notice] text must appear somewhere in the result
    all_content = " ".join(m.get("content", "") for m in result)
    assert "[System notice] Mid-thread notice." in all_content

    # The first system prompt is preserved
    assert result[0] == {"role": "system", "content": "Initial."}

    # The result must not end on a role that would confuse the model —
    # after alternation/merging the last entry should not be a raw tool message
    # (it gets merged with the following user content).
    assert result[-1].get("role") != "tool" or "[System notice]" in result[-1].get("content", "")


def test_format_messages_no_transforms():
    """All flags False: result is a deep copy, no transforms applied."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    result = format_messages_for_local(
        messages,
        strict_alternation=False,
        fold_system_messages=False,
        inject_continue_after_tool=False,
    )
    assert result == messages
    # Must be a copy, not the same objects
    assert result is not messages
    assert result[0] is not messages[0]


def test_format_messages_does_not_mutate_input():
    """Original message list and dicts are not modified."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "World"},
    ]
    original_content = messages[0]["content"]
    format_messages_for_local(messages)
    assert messages[0]["content"] == original_content
    assert len(messages) == 2


# ---------------------------------------------------------------------------
# test_empty_messages
# ---------------------------------------------------------------------------


def test_empty_messages():
    """Empty input list produces empty output for all code paths."""
    assert format_messages_for_local([]) == []
    assert _fold_system_messages([]) == []
    assert _inject_continue_sentinels([]) == []
    assert _enforce_alternation([]) == []


# ---------------------------------------------------------------------------
# _get_content_str
# ---------------------------------------------------------------------------


def test_get_content_str_string():
    """Plain string content is returned as-is."""
    msg = {"role": "user", "content": "Hello"}
    assert _get_content_str(msg) == "Hello"


def test_get_content_str_list_content():
    """Multi-part list content: only text parts are extracted and joined."""
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Part one"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
            {"type": "text", "text": "Part two"},
        ],
    }
    result = _get_content_str(msg)
    assert result == "Part one\nPart two"


def test_get_content_str_none_content():
    """Missing content key returns empty string."""
    msg = {"role": "user"}
    assert _get_content_str(msg) == ""


def test_get_content_str_empty_string():
    """Empty string content returns empty string."""
    msg = {"role": "user", "content": ""}
    assert _get_content_str(msg) == ""


# ---------------------------------------------------------------------------
# test_list_content_handling (via format pipeline)
# ---------------------------------------------------------------------------


def test_list_content_handling_merge():
    """Consecutive same-role messages with list content are merged correctly."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "First part"}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Second part"}],
        },
    ]
    result = format_messages_for_local(
        messages,
        strict_alternation=True,
        fold_system_messages=False,
        inject_continue_after_tool=False,
    )
    assert len(result) == 1
    assert "First part" in result[0]["content"]
    assert "Second part" in result[0]["content"]


# ---------------------------------------------------------------------------
# is_local_endpoint
# ---------------------------------------------------------------------------


def test_is_local_endpoint_localhost():
    assert is_local_endpoint("http://localhost:11434/v1") is True


def test_is_local_endpoint_127():
    assert is_local_endpoint("http://127.0.0.1:8080/v1") is True


def test_is_local_endpoint_0000():
    assert is_local_endpoint("http://0.0.0.0:8000/v1") is True


def test_is_local_endpoint_ipv6():
    assert is_local_endpoint("http://[::1]:1234/v1") is True


def test_is_local_endpoint_remote():
    assert is_local_endpoint("https://api.openai.com/v1") is False


def test_is_local_endpoint_empty():
    assert is_local_endpoint("") is False


def test_is_local_endpoint_none_equivalent():
    """Falsy but non-empty strings that aren't local."""
    assert is_local_endpoint("https://example.com") is False


# ---------------------------------------------------------------------------
# detect_local_server_type
# ---------------------------------------------------------------------------


def test_detect_local_server_type_ollama():
    assert detect_local_server_type("http://localhost:11434/v1") == "ollama"


def test_detect_local_server_type_lm_studio():
    assert detect_local_server_type("http://localhost:1234/v1") == "lm-studio"


def test_detect_local_server_type_vllm():
    assert detect_local_server_type("http://localhost:8000/v1") == "vllm"


def test_detect_local_server_type_llama_cpp():
    assert detect_local_server_type("http://localhost:8080/v1") == "llama-cpp"


def test_detect_local_server_type_unknown_port():
    assert detect_local_server_type("http://localhost:9999/v1") is None


def test_detect_local_server_type_empty():
    assert detect_local_server_type("") is None


def test_detect_local_server_type_none_url():
    assert detect_local_server_type(None) is None


# ---------------------------------------------------------------------------
# _effective_role
# ---------------------------------------------------------------------------


def test_effective_role_tool_is_user():
    assert _effective_role({"role": "tool"}) == "user"


def test_effective_role_user():
    assert _effective_role({"role": "user"}) == "user"


def test_effective_role_assistant():
    assert _effective_role({"role": "assistant"}) == "assistant"


def test_effective_role_system():
    assert _effective_role({"role": "system"}) == "system"


def test_effective_role_missing_defaults_to_user():
    assert _effective_role({}) == "user"


# ---------------------------------------------------------------------------
# is_local_endpoint — LAN / private IP tests
# ---------------------------------------------------------------------------


def test_is_local_endpoint_lan_192_168():
    assert is_local_endpoint("http://192.168.1.22:1234/v1") is True


def test_is_local_endpoint_lan_10():
    assert is_local_endpoint("http://10.0.0.5:8080/v1") is True


def test_is_local_endpoint_lan_172():
    assert is_local_endpoint("http://172.16.0.1:1234/v1") is True


def test_is_local_endpoint_public_ip():
    assert is_local_endpoint("http://8.8.8.8:1234/v1") is False


# ---------------------------------------------------------------------------
# _is_private_ip
# ---------------------------------------------------------------------------


def test_is_private_ip_192_168():
    assert _is_private_ip("192.168.0.1") is True


def test_is_private_ip_10():
    assert _is_private_ip("10.10.20.30") is True


def test_is_private_ip_172_16():
    assert _is_private_ip("172.16.5.4") is True


def test_is_private_ip_172_15():
    """172.15.x.x is NOT in the 172.16-31 private range."""
    assert _is_private_ip("172.15.0.1") is False


def test_is_private_ip_public():
    assert _is_private_ip("8.8.8.8") is False


def test_is_private_ip_not_ip():
    assert _is_private_ip("example.com") is False
