"""Three-level summary escalation for LCM.

Level 1: LLM summarize with mode="preserve_details" (added in Step 4)
Level 2: LLM summarize with mode="bullet_points" (added in Step 4)
Level 3: Deterministic truncation — no LLM, guaranteed convergence
"""
from __future__ import annotations
import logging
import re
from typing import Any

from agent.lcm.refusal import contains_refusal

logger = logging.getLogger(__name__)


def _call_summary_llm(messages_text: str, mode: str, target_tokens: int, summary_model: str = "") -> str:
    """Call an LLM to summarize. Placeholder for Step 8 integration.

    In the final integration, this will use hermes's auxiliary_client.
    For now, it's a stub that can be mocked in tests.
    """
    raise NotImplementedError("LLM integration added in Step 8")


def _estimate_input_tokens(messages: list[dict]) -> int:
    """Rough token count for a list of messages."""
    total = sum(len(str(m.get("content", "") or "")) for m in messages)
    return total // 4


def _format_messages_for_summary(messages: list[dict]) -> str:
    """Format messages into a text block for the summarizer."""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = str(msg.get("content", "") or "")
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def escalated_summary(
    messages: list[dict],
    target_tokens: int,
    deterministic_target: int,
    summary_model: str = "",
) -> tuple[str, int]:
    """Three-level escalation. Returns (summary_text, level).

    Level 1: LLM with mode="preserve_details"
    Level 2: LLM with mode="bullet_points" (half the token budget)
    Level 3: Deterministic truncation (no LLM, guaranteed convergence)
    """
    input_tokens = _estimate_input_tokens(messages)
    messages_text = _format_messages_for_summary(messages)
    # 5% floor: reject Level 1 summaries that are suspiciously short (vacuous)
    min_acceptable_tokens = max(1, int(input_tokens * 0.05))
    # Upper-bound: a summary must fit within the requested token budget.
    # Using target_tokens * 4 (chars) as the ceiling avoids rounding issues
    # with short inputs where integer token division loses precision.
    l1_max_chars = target_tokens * 4
    l2_max_chars = (target_tokens // 2) * 4

    # Level 1: preserve_details
    try:
        summary = _call_summary_llm(messages_text, mode="preserve_details",
                                     target_tokens=target_tokens, summary_model=summary_model)
        summary_tokens = len(summary) // 4
        if (not contains_refusal(summary)
                and len(summary) <= l1_max_chars  # within budget
                and summary_tokens >= min_acceptable_tokens):  # not vacuous
            return summary, 1
        logger.debug("Level 1 summary rejected (refusal=%s, len=%d max=%d, tokens=%d min=%d)",
                     contains_refusal(summary), len(summary), l1_max_chars,
                     summary_tokens, min_acceptable_tokens)
    except Exception as e:
        logger.debug("Level 1 summary failed: %s", e)

    # Level 2: bullet_points (half budget)
    # Vacuous check uses a lower floor (3%) since bullet-point summaries are shorter.
    min_acceptable_l2 = max(1, int(input_tokens * 0.03))
    try:
        summary = _call_summary_llm(messages_text, mode="bullet_points",
                                     target_tokens=target_tokens // 2, summary_model=summary_model)
        summary_tokens_l2 = len(summary) // 4
        if (not contains_refusal(summary)
                and len(summary) <= l2_max_chars
                and summary_tokens_l2 >= min_acceptable_l2):
            return summary, 2
        logger.debug("Level 2 summary rejected (refusal=%s, len=%d max=%d, tokens=%d min=%d)",
                     contains_refusal(summary), len(summary), l2_max_chars,
                     summary_tokens_l2, min_acceptable_l2)
    except Exception as e:
        logger.debug("Level 2 summary failed: %s", e)

    # Level 3: deterministic (always succeeds)
    return deterministic_truncate(messages, deterministic_target), 3

# Sentence boundary pattern
_SENTENCE_END = re.compile(r'[.!?](?:\s|$)')

def first_sentence(text: str) -> str:
    """Extract the first sentence from text."""
    if not text:
        return ""
    match = _SENTENCE_END.search(text)
    if match:
        return text[:match.end()].rstrip()
    return text

def deterministic_truncate(messages: list[dict[str, Any]], target_tokens: int) -> str:
    """Level 3: Extract key facts without any LLM call.

    Keeps first sentence of each message with role attribution.
    Guaranteed output <= target_tokens (estimated as ~4 chars/token).
    """
    if not messages:
        return ""

    char_budget = target_tokens * 4  # conservative chars-per-token estimate
    parts: list[str] = []
    used = 0

    for msg in messages:
        role = msg.get("role", "unknown")
        content = str(msg.get("content", "") or "")

        # For tool messages, take a brief extract
        if role == "tool":
            extract = content[:80].replace("\n", " ")
            if len(content) > 80:
                extract += "..."
            line = f"[tool]: {extract}"
        else:
            sentence = first_sentence(content)
            line = f"{role}: {sentence}"

        if used + len(line) > char_budget:
            # Try to fit a truncated version
            remaining = char_budget - used
            if remaining > 20:  # only add if meaningful
                parts.append(line[:remaining])
            break

        parts.append(line)
        used += len(line) + 1  # +1 for newline

    return "\n".join(parts)
