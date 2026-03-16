"""Textual tool calling fallback for models without native function calling.

When a model doesn't support OpenAI-style tool_calls, this module:
1. Formats tool schemas as text in the system prompt
2. Parses LLM text output for tool call patterns
3. Converts parsed calls into standard tool_call objects
"""

import json
import re
import uuid
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedToolCall:
    """A tool call parsed from model text output."""
    name: str
    arguments: dict
    call_id: str  # generated UUID


# The format we instruct models to use
TOOL_CALL_PATTERN = re.compile(
    r'\[TOOL_CALL:\s*(\w+)\((.*?)\)\]',
    re.DOTALL
)

# Alternative patterns some models might use
ALT_PATTERNS = [
    # ```tool_call\n{"name": "...", "arguments": {...}}\n```
    re.compile(r'```tool_call\s*\n\s*\{.*?"name"\s*:\s*"(\w+)".*?"arguments"\s*:\s*(\{.*?\}).*?\}?\s*\n\s*```', re.DOTALL),
    # <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    re.compile(r'<tool_call>\s*\{.*?"name"\s*:\s*"(\w+)".*?"arguments"\s*:\s*(\{.*?\})\s*\}?\s*</tool_call>', re.DOTALL),
    # <|tool_call|> ... (Hermes format)
    re.compile(r'<\|tool_call\|>\s*\{.*?"name"\s*:\s*"(\w+)".*?"arguments"\s*:\s*(\{.*?\})\s*\}?', re.DOTALL),
]


def format_tools_for_prompt(tools: list[dict]) -> str:
    """Format tool schemas as text to inject into the system prompt.

    Args:
        tools: List of OpenAI tool schema dicts (with type, function, etc.)

    Returns:
        Text block describing available tools and the calling format.
    """
    if not tools:
        return ""

    lines = [
        "\n## Available Tools\n",
        "You have access to the following tools. To call a tool, use this exact format:",
        "",
        '[TOOL_CALL: tool_name({"param1": "value1", "param2": "value2"})]',
        "",
        "You may call multiple tools in a single response. Each tool call must be on its own line.",
        "After calling tools, wait for the results before continuing.",
        "",
        "### Tools:\n",
    ]

    for tool in tools:
        func = tool.get("function", tool)  # handle both wrapped and unwrapped
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})

        lines.append(f"**{name}**: {desc}")

        # Format parameters
        properties = params.get("properties", {})
        required = params.get("required", [])
        if properties:
            lines.append("  Parameters:")
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                req_marker = " (required)" if param_name in required else " (optional)"
                lines.append(f"  - `{param_name}` ({param_type}{req_marker}): {param_desc}")
        lines.append("")

    return "\n".join(lines)


def parse_tool_calls(text: str, valid_tool_names: set[str] | None = None) -> list[ParsedToolCall]:
    """Parse tool calls from model text output.

    Tries the primary [TOOL_CALL: ...] pattern first, then falls back to
    alternative patterns (```tool_call```, <tool_call>, Hermes format).

    Args:
        text: The model's text response.
        valid_tool_names: Optional set of valid tool names to filter against.

    Returns:
        List of parsed tool calls. Empty if no tool calls found.
    """
    calls = []

    # Try primary pattern first
    for match in TOOL_CALL_PATTERN.finditer(text):
        name = match.group(1)
        args_str = match.group(2).strip()

        if valid_tool_names and name not in valid_tool_names:
            logger.warning(f"Parsed tool call '{name}' not in valid tools, skipping")
            continue

        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse arguments for tool '{name}': {args_str}")
            # Try to fix common JSON issues
            args = _attempt_json_repair(args_str)
            if args is None:
                continue

        calls.append(ParsedToolCall(
            name=name,
            arguments=args,
            call_id=f"textual_{uuid.uuid4().hex[:12]}"
        ))

    if calls:
        return calls

    # Try alternative patterns
    for pattern in ALT_PATTERNS:
        for match in pattern.finditer(text):
            name = match.group(1)
            args_str = match.group(2).strip()

            if valid_tool_names and name not in valid_tool_names:
                continue

            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                args = _attempt_json_repair(args_str)
                if args is None:
                    continue

            calls.append(ParsedToolCall(
                name=name,
                arguments=args,
                call_id=f"textual_{uuid.uuid4().hex[:12]}"
            ))

    return calls


def _attempt_json_repair(s: str) -> Optional[dict]:
    """Try to repair common JSON issues in model output."""
    if not s:
        return {}

    # Try adding/fixing braces
    s = s.strip()
    if not s.startswith("{"):
        s = "{" + s
    if not s.endswith("}"):
        s = s + "}"

    # Try with trailing comma removed
    s_no_trailing = re.sub(r',\s*}', '}', s)
    s_no_trailing = re.sub(r',\s*]', ']', s_no_trailing)

    try:
        return json.loads(s_no_trailing)
    except json.JSONDecodeError:
        pass

    # Try with single quotes replaced
    try:
        return json.loads(s_no_trailing.replace("'", '"'))
    except json.JSONDecodeError:
        pass

    logger.debug(f"JSON repair failed for: {s}")
    return None


def parsed_calls_to_openai_format(calls: list[ParsedToolCall]) -> list[dict]:
    """Convert ParsedToolCall objects to OpenAI tool_call format.

    Returns a list of dicts matching the OpenAI ChatCompletionMessageToolCall shape:
    {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
    """
    return [
        {
            "id": call.call_id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(call.arguments),
            }
        }
        for call in calls
    ]


def format_tool_result_as_text(tool_name: str, call_id: str, result: str) -> str:
    """Format a tool result as a text message for textual mode.

    In textual mode, tool results are sent as user messages instead of
    role:tool messages (since models in textual mode don't understand role:tool).
    """
    return f"[Tool result — {tool_name}({call_id})]:\n{result}"


def strip_tool_call_text(text: str) -> str:
    """Remove tool call patterns from text to get the non-tool-call content.

    Useful for extracting the model's actual response text from a message
    that also contains tool calls.
    """
    result = TOOL_CALL_PATTERN.sub("", text)
    for pattern in ALT_PATTERNS:
        result = pattern.sub("", result)
    return result.strip()


def has_tool_calls(text: str) -> bool:
    """Quick check if text contains any tool call patterns."""
    if TOOL_CALL_PATTERN.search(text):
        return True
    return any(p.search(text) for p in ALT_PATTERNS)
