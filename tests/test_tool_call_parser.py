"""Tests for agent/tool_call_parser.py — textual tool calling fallback.

All tests operate on pure functions with synthetic inputs. No filesystem,
no network, no external dependencies beyond stdlib.
"""

import json

import pytest

from agent.tool_call_parser import (
    ParsedToolCall,
    _attempt_json_repair,
    format_tool_result_as_text,
    format_tools_for_prompt,
    has_tool_calls,
    parse_tool_calls,
    parsed_calls_to_openai_format,
    strip_tool_call_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(name: str, description: str = "", params: dict | None = None) -> dict:
    """Build a minimal OpenAI-style tool schema dict."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params or {"type": "object", "properties": {}},
        },
    }


def _make_tool_with_params(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Does {name} things",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results"},
                },
                "required": ["query"],
            },
        },
    }


# ---------------------------------------------------------------------------
# format_tools_for_prompt
# ---------------------------------------------------------------------------

class TestFormatToolsForPrompt:
    def test_format_tools_empty(self):
        """Empty tools list returns empty string."""
        assert format_tools_for_prompt([]) == ""

    def test_format_tools_for_prompt_basic(self):
        """Generated text includes tool name, description, and call format."""
        tools = [_make_tool("web_search", "Search the web")]
        result = format_tools_for_prompt(tools)
        assert "web_search" in result
        assert "Search the web" in result
        assert "TOOL_CALL" in result

    def test_format_tools_includes_parameters(self):
        """Parameters, types, and required marker appear in output."""
        tools = [_make_tool_with_params("web_search")]
        result = format_tools_for_prompt(tools)
        assert "query" in result
        assert "limit" in result
        assert "(required)" in result
        assert "(optional)" in result

    def test_format_tools_multiple(self):
        """All provided tools appear in the output."""
        tools = [
            _make_tool("tool_a", "First tool"),
            _make_tool("tool_b", "Second tool"),
        ]
        result = format_tools_for_prompt(tools)
        assert "tool_a" in result
        assert "tool_b" in result
        assert "First tool" in result
        assert "Second tool" in result

    def test_format_tools_unwrapped_schema(self):
        """Handles schemas not wrapped in {'type': 'function', 'function': ...}."""
        unwrapped = {
            "name": "raw_tool",
            "description": "A raw tool",
            "parameters": {"type": "object", "properties": {}},
        }
        result = format_tools_for_prompt([unwrapped])
        assert "raw_tool" in result
        assert "A raw tool" in result


# ---------------------------------------------------------------------------
# parse_tool_calls — primary pattern
# ---------------------------------------------------------------------------

class TestParsePrimaryPattern:
    def test_parse_primary_pattern(self):
        """[TOOL_CALL: name({...})] is parsed correctly."""
        text = '[TOOL_CALL: web_search({"query": "test"})]'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "web_search"
        assert calls[0].arguments == {"query": "test"}

    def test_parse_multiple_calls(self):
        """Multiple tool calls in one response are all parsed."""
        text = (
            '[TOOL_CALL: web_search({"query": "foo"})]\n'
            '[TOOL_CALL: read_file({"path": "/tmp/bar.txt"})]'
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].name == "web_search"
        assert calls[1].name == "read_file"

    def test_parse_multiline_args(self):
        """JSON args spanning multiple lines inside the pattern are parsed."""
        text = '[TOOL_CALL: write_file({\n  "path": "/tmp/out.txt",\n  "content": "hello"\n})]'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].arguments == {"path": "/tmp/out.txt", "content": "hello"}

    def test_parse_no_args(self):
        """[TOOL_CALL: get_status()] is parsed with empty arguments dict."""
        text = "[TOOL_CALL: get_status()]"
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "get_status"
        assert calls[0].arguments == {}

    def test_call_id_is_unique(self):
        """Each parsed call gets a unique call_id."""
        text = (
            '[TOOL_CALL: tool_a({})]\n'
            '[TOOL_CALL: tool_b({})]'
        )
        calls = parse_tool_calls(text)
        assert calls[0].call_id != calls[1].call_id

    def test_call_id_has_textual_prefix(self):
        """call_id starts with 'textual_'."""
        calls = parse_tool_calls('[TOOL_CALL: foo({})]')
        assert calls[0].call_id.startswith("textual_")


# ---------------------------------------------------------------------------
# parse_tool_calls — alternative patterns
# ---------------------------------------------------------------------------

class TestParseAltPatterns:
    def test_parse_alt_pattern_code_block(self):
        """```tool_call block format is parsed."""
        text = '```tool_call\n{"name": "web_search", "arguments": {"query": "hello"}}\n```'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "web_search"
        assert calls[0].arguments == {"query": "hello"}

    def test_parse_alt_pattern_xml(self):
        """<tool_call> XML format is parsed."""
        text = '<tool_call>{"name": "list_files", "arguments": {"path": "/tmp"}}</tool_call>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "list_files"
        assert calls[0].arguments == {"path": "/tmp"}

    def test_parse_alt_pattern_hermes(self):
        """<|tool_call|> Hermes format is parsed."""
        text = '<|tool_call|>{"name": "execute_code", "arguments": {"code": "print(1)"}}'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "execute_code"
        assert calls[0].arguments == {"code": "print(1)"}

    def test_alt_patterns_not_tried_when_primary_matches(self):
        """When primary pattern matches, alt patterns are not used."""
        # Both primary and an alt appear; only primary call should be returned
        # because primary pattern results are returned early.
        text = (
            '[TOOL_CALL: primary_tool({"a": 1})]\n'
            '<tool_call>{"name": "alt_tool", "arguments": {"b": 2}}</tool_call>'
        )
        calls = parse_tool_calls(text)
        # Primary matched, so we get 1 call (from primary)
        assert len(calls) == 1
        assert calls[0].name == "primary_tool"


# ---------------------------------------------------------------------------
# parse_tool_calls — JSON repair
# ---------------------------------------------------------------------------

class TestParseJsonRepair:
    def test_parse_invalid_json_repaired_trailing_comma(self):
        """Trailing comma in JSON args is repaired."""
        text = '[TOOL_CALL: foo({"a": 1,})]'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].arguments == {"a": 1}

    def test_parse_invalid_json_unrecoverable(self):
        """Completely broken JSON causes the call to be skipped."""
        text = '[TOOL_CALL: broken(not json at all !!!)]'
        calls = parse_tool_calls(text)
        assert len(calls) == 0

    def test_parse_valid_tool_names_filter(self):
        """Tool calls with names not in valid_tool_names are filtered out."""
        text = (
            '[TOOL_CALL: allowed_tool({"x": 1})]\n'
            '[TOOL_CALL: forbidden_tool({"y": 2})]'
        )
        calls = parse_tool_calls(text, valid_tool_names={"allowed_tool"})
        assert len(calls) == 1
        assert calls[0].name == "allowed_tool"

    def test_parse_valid_tool_names_none_means_all_allowed(self):
        """None valid_tool_names allows all tool names through."""
        text = '[TOOL_CALL: any_tool({"z": 3})]'
        calls = parse_tool_calls(text, valid_tool_names=None)
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# _attempt_json_repair (unit tests of the repair helper directly)
# ---------------------------------------------------------------------------

class TestAttemptJsonRepair:
    def test_json_repair_trailing_comma(self):
        """Trailing comma is removed before parsing."""
        result = _attempt_json_repair('{"a": 1,}')
        assert result == {"a": 1}

    def test_json_repair_single_quotes(self):
        """Single quotes are replaced with double quotes."""
        result = _attempt_json_repair("{'a': 'b'}")
        assert result == {"a": "b"}

    def test_json_repair_missing_braces(self):
        """"key": "value" without braces gets braces added."""
        result = _attempt_json_repair('"a": "b"')
        assert result == {"a": "b"}

    def test_json_repair_empty_string(self):
        """Empty string returns empty dict."""
        result = _attempt_json_repair("")
        assert result == {}

    def test_json_repair_valid_json_unchanged(self):
        """Already valid JSON is returned without modification."""
        result = _attempt_json_repair('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_repair_unrecoverable_returns_none(self):
        """Truly broken JSON that can't be repaired returns None."""
        result = _attempt_json_repair("not json at all !!!")
        assert result is None


# ---------------------------------------------------------------------------
# parsed_calls_to_openai_format
# ---------------------------------------------------------------------------

class TestParsedCallsToOpenaiFormat:
    def test_parsed_calls_to_openai_format(self):
        """Correct OpenAI ChatCompletionMessageToolCall shape is produced."""
        calls = [
            ParsedToolCall(name="web_search", arguments={"query": "test"}, call_id="textual_abc123"),
        ]
        result = parsed_calls_to_openai_format(calls)
        assert len(result) == 1
        item = result[0]
        assert item["id"] == "textual_abc123"
        assert item["type"] == "function"
        assert item["function"]["name"] == "web_search"
        # arguments must be a JSON string, not a dict
        parsed_args = json.loads(item["function"]["arguments"])
        assert parsed_args == {"query": "test"}

    def test_parsed_calls_to_openai_format_empty(self):
        """Empty list of calls returns empty list."""
        assert parsed_calls_to_openai_format([]) == []

    def test_parsed_calls_to_openai_format_multiple(self):
        """Multiple calls are all converted."""
        calls = [
            ParsedToolCall(name="tool_a", arguments={"x": 1}, call_id="id_1"),
            ParsedToolCall(name="tool_b", arguments={"y": 2}, call_id="id_2"),
        ]
        result = parsed_calls_to_openai_format(calls)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool_a"
        assert result[1]["function"]["name"] == "tool_b"


# ---------------------------------------------------------------------------
# format_tool_result_as_text
# ---------------------------------------------------------------------------

class TestFormatToolResultAsText:
    def test_format_tool_result_as_text(self):
        """Correct text format for a tool result message."""
        result = format_tool_result_as_text("web_search", "textual_abc", "42 results found")
        assert "web_search" in result
        assert "textual_abc" in result
        assert "42 results found" in result

    def test_format_tool_result_contains_separator(self):
        """Result block is separated from header by a newline."""
        result = format_tool_result_as_text("my_tool", "id_1", "output here")
        assert "\n" in result
        # The result text follows the header
        assert result.endswith("output here")


# ---------------------------------------------------------------------------
# strip_tool_call_text
# ---------------------------------------------------------------------------

class TestStripToolCallText:
    def test_strip_tool_call_text(self):
        """Tool call patterns are removed, surrounding text is preserved."""
        text = "Here is my answer. [TOOL_CALL: foo({\"x\": 1})] And some follow-up."
        stripped = strip_tool_call_text(text)
        assert "TOOL_CALL" not in stripped
        assert "Here is my answer." in stripped
        assert "And some follow-up." in stripped

    def test_strip_tool_call_text_only_call(self):
        """Text that is only a tool call strips to empty string."""
        text = '[TOOL_CALL: foo({})]'
        stripped = strip_tool_call_text(text)
        assert stripped == ""

    def test_strip_tool_call_text_no_calls(self):
        """Text with no tool calls is returned unchanged."""
        text = "Just a regular response with no tool calls."
        assert strip_tool_call_text(text) == text

    def test_strip_removes_alt_patterns(self):
        """Alt patterns are also stripped."""
        text = 'Some text. <tool_call>{"name": "x", "arguments": {"a": 1}}</tool_call> More text.'
        stripped = strip_tool_call_text(text)
        assert "tool_call" not in stripped
        assert "Some text." in stripped
        assert "More text." in stripped


# ---------------------------------------------------------------------------
# has_tool_calls
# ---------------------------------------------------------------------------

class TestHasToolCalls:
    def test_has_tool_calls_true_primary(self):
        """Primary pattern is detected."""
        assert has_tool_calls('[TOOL_CALL: foo({})]') is True

    def test_has_tool_calls_true_alt_xml(self):
        """Alt XML pattern is detected."""
        assert has_tool_calls('<tool_call>{"name": "x", "arguments": {}}</tool_call>') is True

    def test_has_tool_calls_true_alt_hermes(self):
        """Alt Hermes pattern is detected."""
        assert has_tool_calls('<|tool_call|>{"name": "x", "arguments": {}}') is True

    def test_has_tool_calls_false(self):
        """Normal text produces no false positives."""
        assert has_tool_calls("This is just a regular response.") is False

    def test_has_tool_calls_false_partial_match(self):
        """Partial bracket text is not a false positive."""
        assert has_tool_calls("[TOOL_CALL incomplete") is False
