"""Tests for LCM escalation: _call_summary_llm integration with auxiliary_client."""
import pytest
from unittest.mock import MagicMock, patch, call

from agent.lcm.config import LcmConfig
from agent.lcm.escalation import (
    _call_summary_llm,
    escalated_summary,
    deterministic_truncate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_messages(n: int = 5) -> list[dict]:
    """Return a small list of realistic messages."""
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"Message {i}: some content about task {i}"})
    return msgs


def _make_response(text: str):
    """Build a mock LLM response object (mirrors openai.ChatCompletion shape)."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    return resp


# ---------------------------------------------------------------------------
# 1. test_call_summary_llm_level1_preserve_details
# ---------------------------------------------------------------------------

class TestCallSummaryLlmPreserveDetails:
    def test_calls_call_llm_with_preserve_details_prompt(self):
        messages = _make_messages()
        mock_response = _make_response("## Goal\nSome goal\n## Progress\nDone\n## Decisions\nNone\n## Files\nnone\n## Next Steps\nn/a")

        with patch("agent.lcm.escalation.call_llm", return_value=mock_response) as mock_call:
            result = _call_summary_llm(messages, mode="preserve_details", budget=500)

        assert result is not None
        # Verify call_llm was invoked
        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args.kwargs
        # The system prompt should mention key details, decisions, file paths
        system_content = call_kwargs["messages"][0]["content"]
        assert "preserve" in system_content.lower() or "key details" in system_content.lower()
        assert "decisions" in system_content.lower()
        assert "file" in system_content.lower()

    def test_returns_llm_response_text(self):
        messages = _make_messages()
        expected = "## Goal\nFix the bug\n## Progress\nDone\n## Decisions\nUsed patch\n## Files\nfoo.py\n## Next Steps\nTest"
        mock_response = _make_response(expected)

        with patch("agent.lcm.escalation.call_llm", return_value=mock_response):
            result = _call_summary_llm(messages, mode="preserve_details", budget=500)

        assert result == expected


# ---------------------------------------------------------------------------
# 2. test_call_summary_llm_level2_bullet_points
# ---------------------------------------------------------------------------

class TestCallSummaryLlmBulletPoints:
    def test_calls_call_llm_with_bullet_points_prompt(self):
        messages = _make_messages()
        mock_response = _make_response("## Goal\n- Fix bug\n## Progress\n- Done\n## Decisions\n- Patch\n## Files\n- foo.py\n## Next Steps\n- Test")

        with patch("agent.lcm.escalation.call_llm", return_value=mock_response) as mock_call:
            result = _call_summary_llm(messages, mode="bullet_points", budget=250)

        assert result is not None
        call_kwargs = mock_call.call_args.kwargs
        system_content = call_kwargs["messages"][0]["content"]
        assert "bullet" in system_content.lower() or "condense" in system_content.lower()

    def test_returns_llm_response_text_for_bullets(self):
        messages = _make_messages()
        expected = "## Goal\n- Fix\n## Progress\n- Done\n## Decisions\n- x\n## Files\n- y\n## Next Steps\n- z"
        mock_response = _make_response(expected)

        with patch("agent.lcm.escalation.call_llm", return_value=mock_response):
            result = _call_summary_llm(messages, mode="bullet_points", budget=250)

        assert result == expected


# ---------------------------------------------------------------------------
# 3. test_call_summary_llm_uses_structured_template
# ---------------------------------------------------------------------------

class TestCallSummaryLlmStructuredTemplate:
    def test_prompt_includes_all_template_sections(self):
        messages = _make_messages()
        mock_response = _make_response("summary text")

        with patch("agent.lcm.escalation.call_llm", return_value=mock_response) as mock_call:
            _call_summary_llm(messages, mode="preserve_details", budget=500)

        call_kwargs = mock_call.call_args.kwargs
        system_content = call_kwargs["messages"][0]["content"]
        for section in ["Goal", "Progress", "Decisions", "Files", "Next Steps"]:
            assert section in system_content, f"Section '{section}' missing from prompt"

    def test_bullet_points_prompt_also_includes_all_sections(self):
        messages = _make_messages()
        mock_response = _make_response("summary text")

        with patch("agent.lcm.escalation.call_llm", return_value=mock_response) as mock_call:
            _call_summary_llm(messages, mode="bullet_points", budget=250)

        call_kwargs = mock_call.call_args.kwargs
        system_content = call_kwargs["messages"][0]["content"]
        for section in ["Goal", "Progress", "Decisions", "Files", "Next Steps"]:
            assert section in system_content, f"Section '{section}' missing from bullet_points prompt"


# ---------------------------------------------------------------------------
# 4. test_call_summary_llm_respects_summary_model_config
# ---------------------------------------------------------------------------

class TestCallSummaryLlmSummaryModel:
    def test_uses_summary_model_from_config(self):
        messages = _make_messages()
        config = LcmConfig(summary_model="test-model-override")
        mock_response = _make_response("summary")

        with patch("agent.lcm.escalation.call_llm", return_value=mock_response) as mock_call:
            _call_summary_llm(messages, mode="preserve_details", budget=500, config=config)

        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs.get("model") == "test-model-override"

    def test_no_model_kwarg_when_config_model_empty(self):
        messages = _make_messages()
        config = LcmConfig(summary_model="")
        mock_response = _make_response("summary")

        with patch("agent.lcm.escalation.call_llm", return_value=mock_response) as mock_call:
            _call_summary_llm(messages, mode="preserve_details", budget=500, config=config)

        call_kwargs = mock_call.call_args.kwargs
        # Should not pass model kwarg when empty
        assert "model" not in call_kwargs or call_kwargs.get("model") is None

    def test_no_model_kwarg_when_config_is_none(self):
        messages = _make_messages()
        mock_response = _make_response("summary")

        with patch("agent.lcm.escalation.call_llm", return_value=mock_response) as mock_call:
            _call_summary_llm(messages, mode="preserve_details", budget=500, config=None)

        call_kwargs = mock_call.call_args.kwargs
        assert "model" not in call_kwargs or call_kwargs.get("model") is None


# ---------------------------------------------------------------------------
# 5. test_call_summary_llm_fallback_on_error
# ---------------------------------------------------------------------------

class TestCallSummaryLlmFallbackOnError:
    def test_returns_none_when_call_llm_raises(self):
        messages = _make_messages()

        with patch("agent.lcm.escalation.call_llm", side_effect=RuntimeError("No provider")):
            result = _call_summary_llm(messages, mode="preserve_details", budget=500)

        assert result is None

    def test_returns_none_when_call_llm_raises_generic_exception(self):
        messages = _make_messages()

        with patch("agent.lcm.escalation.call_llm", side_effect=Exception("API error")):
            result = _call_summary_llm(messages, mode="preserve_details", budget=500)

        assert result is None

    def test_returns_none_when_response_content_is_empty(self):
        messages = _make_messages()
        mock_response = _make_response("")

        with patch("agent.lcm.escalation.call_llm", return_value=mock_response):
            result = _call_summary_llm(messages, mode="preserve_details", budget=500)

        # Empty string is falsy, should return None
        assert result is None


# ---------------------------------------------------------------------------
# 6. test_escalated_summary_l1_to_l2_fallback
# ---------------------------------------------------------------------------

class TestEscalatedSummaryL1ToL2Fallback:
    def test_vacuous_l1_summary_escalates_to_l2(self):
        """When L1 returns a very short (vacuous) summary, escalate to L2."""
        # Create a large-ish input so min_acceptable_tokens > 0
        messages = [{"role": "user", "content": "x" * 2000} for _ in range(5)]
        # input: 10000 chars = 2500 tokens, min_acceptable_l2 = 75 tokens
        # L1 returns a suspiciously short summary (vacuous — under 5% of input)
        l1_response = _make_response("ok")
        # L2 response must be >= 75 tokens (300 chars) to pass vacuousness check
        l2_body = "x" * 320
        l2_response = _make_response(l2_body)

        responses = [l1_response, l2_response]
        call_count = [0]

        def side_effect(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx]

        with patch("agent.lcm.escalation.call_llm", side_effect=side_effect):
            summary, level = escalated_summary(messages, target_tokens=1000, deterministic_target=512)

        assert level == 2
        assert summary == l2_body

    def test_l1_refusal_escalates_to_l2(self):
        """When L1 returns a refusal, escalate to L2."""
        messages = _make_messages(10)
        # A refusal string that triggers contains_refusal
        l1_response = _make_response("I cannot and will not summarize this content.")
        l2_text = "## Goal\n- Task\n## Progress\n- Done\n## Decisions\n- None\n## Files\n- x\n## Next Steps\n- y"
        l2_response = _make_response(l2_text)

        responses = [l1_response, l2_response]
        call_count = [0]

        def side_effect(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx]

        with patch("agent.lcm.escalation.call_llm", side_effect=side_effect):
            summary, level = escalated_summary(messages, target_tokens=500, deterministic_target=512)

        # Should be level 2 (or 3 if refusal check also triggers on l2)
        assert level in (2, 3)


# ---------------------------------------------------------------------------
# 7. test_escalated_summary_l2_to_l3_fallback
# ---------------------------------------------------------------------------

class TestEscalatedSummaryL2ToL3Fallback:
    def test_both_llm_levels_fail_uses_l3(self):
        """When both L1 and L2 raise exceptions, fall back to L3 deterministic."""
        messages = _make_messages(5)

        with patch("agent.lcm.escalation.call_llm", side_effect=RuntimeError("No provider")):
            summary, level = escalated_summary(messages, target_tokens=500, deterministic_target=256)

        assert level == 3
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_l3_summary_is_within_deterministic_target(self):
        """L3 output must respect deterministic_target token budget."""
        messages = [{"role": "user", "content": "x" * 500} for _ in range(20)]

        with patch("agent.lcm.escalation.call_llm", side_effect=RuntimeError("No provider")):
            summary, level = escalated_summary(messages, target_tokens=500, deterministic_target=100)

        assert level == 3
        # 100 tokens * 4 chars = 400 char budget
        assert len(summary) <= 400 + 50  # slight tolerance for line structure


# ---------------------------------------------------------------------------
# 8. test_escalated_summary_full_chain
# ---------------------------------------------------------------------------

class TestEscalatedSummaryFullChain:
    def test_l1_fails_l2_succeeds(self):
        """L1 call raises exception, L2 call succeeds — result comes from L2."""
        messages = _make_messages(8)
        l2_text = "## Goal\n- Complete task\n## Progress\n- All done\n## Decisions\n- Used patch\n## Files\n- main.py\n## Next Steps\n- Deploy"
        l2_response = _make_response(l2_text)

        call_count = [0]

        def side_effect(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                raise RuntimeError("L1 failed")
            return l2_response

        with patch("agent.lcm.escalation.call_llm", side_effect=side_effect):
            summary, level = escalated_summary(messages, target_tokens=500, deterministic_target=256)

        assert level == 2
        assert summary == l2_text

    def test_l1_succeeds_returns_level_1(self):
        """When L1 returns a good summary, the chain stops at level 1."""
        # input: 5 * 400 = 2000 chars = 500 tokens, min_acceptable = 25 tokens (100 chars)
        # L1 text must be >= 100 chars to pass the 5% vacuousness floor
        messages = [{"role": "user", "content": "x" * 400} for _ in range(5)]
        l1_text = "## Goal\n- Fix the reported bug\n## Progress\n- Patch applied and verified\n## Decisions\n- Use monkey patch\n## Files\n- app.py, tests/test_app.py\n## Next Steps\n- Review and merge"
        l1_response = _make_response(l1_text)

        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            return l1_response

        with patch("agent.lcm.escalation.call_llm", side_effect=side_effect):
            summary, level = escalated_summary(messages, target_tokens=2000, deterministic_target=512)

        assert level == 1
        assert summary == l1_text
        # call_llm should have been called exactly once (L1 only)
        assert call_count[0] == 1

    def test_both_llm_succeed_but_l1_too_long_uses_l2(self):
        """If L1 summary exceeds the token budget, fall through to L2."""
        messages = _make_messages(5)
        # L1 returns a summary much longer than the budget allows
        l1_text = "A" * 10000  # way too long
        l2_text = "## Goal\n- x\n## Progress\n- y\n## Decisions\n- z\n## Files\n- w\n## Next Steps\n- v"
        l1_response = _make_response(l1_text)
        l2_response = _make_response(l2_text)

        responses = [l1_response, l2_response]
        call_count = [0]

        def side_effect(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx]

        with patch("agent.lcm.escalation.call_llm", side_effect=side_effect):
            # small target so L1's 10000 chars vastly exceeds budget
            summary, level = escalated_summary(messages, target_tokens=100, deterministic_target=64)

        # L1 too long -> L2 or L3
        assert level in (2, 3)
