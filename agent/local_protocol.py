"""Local protocol handler for formatting messages for local LLM servers.

Local models often have stricter requirements than cloud APIs:
- Strict user/assistant alternation
- No mid-thread system messages
- Can't end conversation on tool results
"""

import copy
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def format_messages_for_local(
    messages: list[dict],
    strict_alternation: bool = True,
    fold_system_messages: bool = True,
    inject_continue_after_tool: bool = True,
) -> list[dict]:
    """Transform messages to comply with local model constraints.

    Args:
        messages: Standard OpenAI-format message list.
        strict_alternation: Merge consecutive same-role messages.
        fold_system_messages: Convert mid-thread system messages to user messages.
        inject_continue_after_tool: Add "Continue." user message after tool results.

    Returns:
        New list of transformed messages (original list is not modified).
    """
    result = [copy.deepcopy(msg) for msg in messages]

    if fold_system_messages:
        result = _fold_system_messages(result)

    if inject_continue_after_tool:
        result = _inject_continue_sentinels(result)

    if strict_alternation:
        result = _enforce_alternation(result)

    return result


def _fold_system_messages(messages: list[dict]) -> list[dict]:
    """Convert mid-thread system messages into user messages.

    The first system message (index 0) is kept as-is.
    Any subsequent system messages become user messages with a [System notice] prefix.
    """
    result = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "system" and i > 0:
            content = msg.get("content", "")
            result.append({
                "role": "user",
                "content": f"[System notice] {content}",
            })
        else:
            result.append(msg)
    return result


def _inject_continue_sentinels(messages: list[dict]) -> list[dict]:
    """Inject 'Continue.' user message after sequences of tool result messages.

    Many local models can't handle ending on a tool result or having
    tool results without a user message following them.
    """
    result = []
    for i, msg in enumerate(messages):
        result.append(msg)
        if msg.get("role") == "tool":
            next_idx = i + 1
            # Skip if the next message is another tool result (batch)
            if next_idx < len(messages) and messages[next_idx].get("role") == "tool":
                continue
            # Inject if this is the last message or the next isn't a user message
            if next_idx >= len(messages) or messages[next_idx].get("role") != "user":
                result.append({"role": "user", "content": "Continue."})
    return result


def _enforce_alternation(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages with the same role.

    Local models typically require strict user/assistant/user/assistant alternation.
    When consecutive messages have the same role, merge their content.

    Special handling:
    - System message at index 0 is preserved as-is
    - Tool messages are treated as user messages for alternation purposes
    - Assistant messages with tool_calls metadata are preserved (not merged)
    """
    if not messages:
        return []

    result = []

    for msg in messages:
        effective_role = _effective_role(msg)

        if not result:
            result.append(msg)
            continue

        prev_effective = _effective_role(result[-1])

        # Don't merge if previous message is system at position 0
        if len(result) == 1 and result[0].get("role") == "system":
            result.append(msg)
            continue

        # Don't merge assistant messages that have tool_calls
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            result.append(msg)
            continue
        if result[-1].get("role") == "assistant" and result[-1].get("tool_calls"):
            result.append(msg)
            continue

        if effective_role == prev_effective:
            prev_content = _get_content_str(result[-1])
            curr_content = _get_content_str(msg)
            if prev_content and curr_content:
                result[-1] = {**result[-1], "content": f"{prev_content}\n\n{curr_content}"}
            elif curr_content:
                result[-1] = {**result[-1], "content": curr_content}
        else:
            result.append(msg)

    return result


def _effective_role(msg: dict) -> str:
    """Get effective role for alternation purposes.

    Tool messages count as user messages.
    """
    role = msg.get("role", "user")
    if role == "tool":
        return "user"
    return role


def _get_content_str(msg: dict) -> str:
    """Extract content as string from a message, handling list content."""
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(parts)
    return str(content) if content else ""


def is_local_endpoint(base_url: str) -> bool:
    """Check if a base URL points to a local or LAN server."""
    if not base_url:
        return False
    lower = base_url.lower()
    # Loopback
    local_indicators = ("localhost", "127.0.0.1", "0.0.0.0", "[::1]")
    if any(indicator in lower for indicator in local_indicators):
        return True
    # Private network IPs (RFC 1918)
    import re
    # Extract host from URL
    match = re.search(r'://([^:/]+)', lower)
    if match:
        host = match.group(1)
        if _is_private_ip(host):
            return True
    return False


def _is_private_ip(host: str) -> bool:
    """Check if a hostname is a private/LAN IP address."""
    parts = host.split('.')
    if len(parts) != 4:
        return False
    try:
        octets = [int(p) for p in parts]
    except ValueError:
        return False
    # 10.0.0.0/8
    if octets[0] == 10:
        return True
    # 172.16.0.0/12
    if octets[0] == 172 and 16 <= octets[1] <= 31:
        return True
    # 192.168.0.0/16
    if octets[0] == 192 and octets[1] == 168:
        return True
    return False


def detect_local_server_type(base_url: str) -> Optional[str]:
    """Try to identify which local server is running based on port or URL pattern.

    Returns: "ollama" | "lm-studio" | "vllm" | "llama-cpp" | None
    """
    if not base_url:
        return None
    lower = base_url.lower()

    if ":11434" in lower:
        return "ollama"
    if ":1234" in lower:
        return "lm-studio"
    if ":8000" in lower:
        return "vllm"
    if ":8080" in lower:
        return "llama-cpp"

    return None
