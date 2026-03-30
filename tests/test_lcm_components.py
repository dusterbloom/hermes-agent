"""Unit tests for LCM components: Summarizer, TokenEstimator, SemanticIndex."""
import pytest
from unittest.mock import MagicMock, patch
import os

from agent.lcm.summarizer import (
    Summarizer,
    SummarizerConfig,
    SUMMARY_PREFIX,
    MIN_SUMMARY_TOKENS,
    MAX_SUMMARY_TOKENS,
)
from agent.lcm.tokens import (
    TokenEstimator,
    TokenEstimatorConfig,
    estimate_messages_tokens_rough,
    CHAR_RATIOS,
)
from agent.lcm.semantic import (
    SemanticIndex,
    SemanticIndexConfig,
    NoOpSemanticIndex,
    create_semantic_index,
)
from agent.lcm.store import ImmutableStore


# ------------------------------------------------------------------
# Summarizer Tests
# ------------------------------------------------------------------

class TestSummarizerConfig:
    def test_defaults(self):
        config = SummarizerConfig()
        assert config.model == ""
        assert config.min_tokens == MIN_SUMMARY_TOKENS
        assert config.max_tokens == MAX_SUMMARY_TOKENS
        assert config.ratio == 0.20

    def test_custom_values(self):
        config = SummarizerConfig(
            model="gpt-4o-mini",
            min_tokens=1000,
            max_tokens=8000,
            ratio=0.25,
        )
        assert config.model == "gpt-4o-mini"
        assert config.min_tokens == 1000
        assert config.max_tokens == 8000
        assert config.ratio == 0.25


class TestSummarizerSerialize:
    def test_serialize_simple_messages(self):
        config = SummarizerConfig()
        summarizer = Summarizer(config)

        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = summarizer.serialize_for_summary(turns)

        assert "[USER]: Hello" in result
        assert "[ASSISTANT]: Hi there!" in result

    def test_serialize_tool_result(self):
        config = SummarizerConfig()
        summarizer = Summarizer(config)

        turns = [
            {"role": "tool", "content": "File content here", "tool_call_id": "call_123"},
        ]
        result = summarizer.serialize_for_summary(turns)

        assert "[TOOL RESULT call_123]" in result
        assert "File content here" in result

    def test_serialize_tool_calls(self):
        config = SummarizerConfig()
        summarizer = Summarizer(config)

        turns = [
            {
                "role": "assistant",
                "content": "Let me check that.",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/src/main.py"}',
                        },
                    }
                ],
            }
        ]
        result = summarizer.serialize_for_summary(turns)

        assert "[ASSISTANT]" in result
        assert "read_file" in result
        assert "/src/main.py" in result

    def test_serialize_truncates_long_content(self):
        config = SummarizerConfig()
        summarizer = Summarizer(config)

        long_content = "x" * 5000
        turns = [{"role": "user", "content": long_content}]
        result = summarizer.serialize_for_summary(turns, max_chars=1000)

        # Should be truncated
        assert len(result) < len(long_content) + 100
        assert "...[truncated]..." in result


class TestSummarizerBudget:
    def test_compute_budget_small_content(self):
        config = SummarizerConfig()
        summarizer = Summarizer(config)

        # Small content should hit minimum
        turns = [{"role": "user", "content": "Hello"}]
        budget = summarizer.compute_summary_budget(turns)
        assert budget == MIN_SUMMARY_TOKENS

    def test_compute_budget_large_content(self):
        config = SummarizerConfig()
        summarizer = Summarizer(config)

        # Large content should scale but cap at max
        turns = [{"role": "user", "content": "x" * 100_000}]
        budget = summarizer.compute_summary_budget(turns)
        assert budget == MAX_SUMMARY_TOKENS

    def test_compute_budget_medium_content(self):
        config = SummarizerConfig()
        summarizer = Summarizer(config)

        # Medium content should be proportional
        turns = [{"role": "user", "content": "x" * 10_000}]  # ~2500 tokens
        budget = summarizer.compute_summary_budget(turns)
        # 2500 * 0.20 = 500, clamped to MIN_SUMMARY_TOKENS (512)
        assert MIN_SUMMARY_TOKENS <= budget <= MAX_SUMMARY_TOKENS


class TestSummarizerSummaryPrefix:
    def test_adds_prefix(self):
        config = SummarizerConfig()
        summarizer = Summarizer(config)

        result = summarizer._with_summary_prefix("Test summary")
        assert result.startswith(SUMMARY_PREFIX)
        assert "Test summary" in result

    def test_removes_existing_prefix(self):
        config = SummarizerConfig()
        summarizer = Summarizer(config)

        # Already has prefix
        text = SUMMARY_PREFIX + "\nOld summary"
        result = summarizer._with_summary_prefix(text)
        # Should only have one prefix
        assert result.count(SUMMARY_PREFIX) == 1


class TestSummarizerSummarize:
    def test_summarize_empty_returns_none(self):
        config = SummarizerConfig()
        summarizer = Summarizer(config)

        result = summarizer.summarize([])
        assert result is None

    def test_summarize_with_mock_llm(self):
        config = SummarizerConfig(model="test-model")
        summarizer = Summarizer(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "## Goal\nTest goal\n## Progress\n### Done\nTest done"

        with patch("agent.lcm.summarizer.call_llm", return_value=mock_response):
            turns = [
                {"role": "user", "content": "Write a function"},
                {"role": "assistant", "content": "Done"},
            ]
            result = summarizer.summarize(turns)

        assert result is not None
        assert SUMMARY_PREFIX in result
        assert "Test goal" in result

    def test_summarize_iterative_with_previous(self):
        config = SummarizerConfig(model="test-model")
        summarizer = Summarizer(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "## Goal\nUpdated goal"

        with patch("agent.lcm.summarizer.call_llm", return_value=mock_response) as mock_call:
            turns = [{"role": "user", "content": "New request"}]
            result = summarizer.summarize(
                turns,
                previous_summary="## Goal\nOld goal"
            )

            # Check that previous summary was included in prompt
            call_args = mock_call.call_args
            prompt = call_args.kwargs["messages"][0]["content"]
            assert "PREVIOUS SUMMARY" in prompt
            assert "Old goal" in prompt

    def test_summarize_handles_non_string_content(self):
        config = SummarizerConfig(model="test-model")
        summarizer = Summarizer(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        # Simulate llama.cpp returning dict
        mock_response.choices[0].message.content = {"text": "Summary"}

        with patch("agent.lcm.summarizer.call_llm", return_value=mock_response):
            turns = [{"role": "user", "content": "Test"}]
            result = summarizer.summarize(turns)

        # Should convert to string
        assert result is not None

    def test_summarize_runtime_error_returns_none(self):
        config = SummarizerConfig()
        summarizer = Summarizer(config)

        with patch("agent.lcm.summarizer.call_llm", side_effect=RuntimeError("No provider")):
            turns = [{"role": "user", "content": "Test"}]
            result = summarizer.summarize(turns)

        assert result is None

    def test_summarize_exception_returns_none(self):
        config = SummarizerConfig()
        summarizer = Summarizer(config)

        with patch("agent.lcm.summarizer.call_llm", side_effect=Exception("API error")):
            turns = [{"role": "user", "content": "Test"}]
            result = summarizer.summarize(turns)

        assert result is None


# ------------------------------------------------------------------
# TokenEstimator Tests
# ------------------------------------------------------------------

class TestTokenEstimatorConfig:
    def test_defaults(self):
        config = TokenEstimatorConfig()
        assert config.model == ""
        assert config.provider == ""
        assert config.context_length == 128_000
        assert config.use_tiktoken is True

    def test_custom_values(self):
        config = TokenEstimatorConfig(
            model="claude-sonnet-4",
            provider="anthropic",
            context_length=200_000,
            use_tiktoken=False,
        )
        assert config.model == "claude-sonnet-4"
        assert config.provider == "anthropic"
        assert config.context_length == 200_000
        assert config.use_tiktoken is False


class TestTokenEstimatorCharRatio:
    def test_char_ratio_claude(self):
        config = TokenEstimatorConfig(model="claude-opus-4", provider="anthropic")
        estimator = TokenEstimator(config)
        assert estimator._get_char_ratio() == CHAR_RATIOS["claude"]

    def test_char_ratio_gpt(self):
        config = TokenEstimatorConfig(model="gpt-4o", provider="openai")
        estimator = TokenEstimator(config)
        # Won't use char ratio in practice (uses tiktoken), but method exists
        assert estimator._get_char_ratio() == CHAR_RATIOS["gpt"]

    def test_char_ratio_unknown(self):
        config = TokenEstimatorConfig(model="unknown-model", provider="unknown")
        estimator = TokenEstimator(config)
        assert estimator._get_char_ratio() == CHAR_RATIOS["default"]


class TestTokenEstimatorEstimate:
    def test_estimate_simple_messages(self):
        config = TokenEstimatorConfig(use_tiktoken=False)
        estimator = TokenEstimator(config)

        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        tokens = estimator.estimate(messages)

        # Should include content + overhead
        assert tokens > 0
        # Rough check: "Hello world" + "Hi there!" = ~21 chars / 4 + 2*4 overhead ≈ 13
        assert tokens >= 10

    def test_estimate_empty_messages(self):
        config = TokenEstimatorConfig(use_tiktoken=False)
        estimator = TokenEstimator(config)

        tokens = estimator.estimate([])
        assert tokens == 0

    def test_estimate_with_tool_calls(self):
        config = TokenEstimatorConfig(use_tiktoken=False)
        estimator = TokenEstimator(config)

        messages = [
            {
                "role": "assistant",
                "content": "Let me check",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/very/long/path/to/file.py"}',
                        }
                    }
                ],
            }
        ]
        tokens = estimator.estimate(messages)
        assert tokens > 0

    def test_estimate_single_string(self):
        config = TokenEstimatorConfig(use_tiktoken=False)
        estimator = TokenEstimator(config)

        tokens = estimator.estimate_single("Hello world")
        # 11 chars / 4 ≈ 2-3 tokens
        assert tokens >= 2

    def test_estimate_single_empty(self):
        config = TokenEstimatorConfig(use_tiktoken=False)
        estimator = TokenEstimator(config)

        tokens = estimator.estimate_single("")
        assert tokens == 0

    def test_context_length_property(self):
        config = TokenEstimatorConfig(context_length=100_000)
        estimator = TokenEstimator(config)

        assert estimator.context_length == 100_000

        estimator.context_length = 200_000
        assert estimator.context_length == 200_000


class TestEstimateMessagesTokensRough:
    def test_rough_estimate(self):
        messages = [
            {"role": "user", "content": "Hello world"},
        ]
        tokens = estimate_messages_tokens_rough(messages)
        assert tokens > 0

    def test_rough_estimate_with_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"arguments": '{"key": "value"}'}}
                ],
            }
        ]
        tokens = estimate_messages_tokens_rough(messages)
        assert tokens > 0


# ------------------------------------------------------------------
# SemanticIndex Tests
# ------------------------------------------------------------------

class TestSemanticIndexConfig:
    def test_defaults(self):
        config = SemanticIndexConfig()
        assert config.enabled is False
        assert config.model == "text-embedding-3-small"
        assert config.min_messages == 50

    def test_custom_values(self):
        config = SemanticIndexConfig(
            enabled=True,
            model="text-embedding-3-large",
            min_messages=100,
        )
        assert config.enabled is True
        assert config.model == "text-embedding-3-large"
        assert config.min_messages == 100


class TestSemanticIndexAvailability:
    def test_disabled_config_unavailable(self):
        config = SemanticIndexConfig(enabled=False)
        index = SemanticIndex(config)
        assert index.is_available() is False

    def test_no_api_key_unavailable(self):
        config = SemanticIndexConfig(enabled=True)
        with patch.dict(os.environ, {}, clear=True):
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            index = SemanticIndex(config)
            assert index.is_available() is False

    def test_enabled_with_api_key_available(self):
        config = SemanticIndexConfig(enabled=True)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            index = SemanticIndex(config)
            assert index.is_available() is True


class TestSemanticIndexShouldIndex:
    def test_small_store_skipped(self):
        config = SemanticIndexConfig(enabled=True, min_messages=10)
        index = SemanticIndex(config)

        store = ImmutableStore()
        for i in range(5):
            store.append({"role": "user", "content": f"Message {i}"})

        assert index.should_index(store) is False

    def test_large_store_indexed(self):
        config = SemanticIndexConfig(enabled=True, min_messages=10)
        index = SemanticIndex(config)

        store = ImmutableStore()
        for i in range(20):
            store.append({"role": "user", "content": f"Message {i}"})

        assert index.should_index(store) is True


class TestSemanticIndexSearch:
    def test_search_empty_index(self):
        config = SemanticIndexConfig(enabled=True)
        index = SemanticIndex(config)

        results = index.search("test query")
        assert results == []

    def test_search_after_clear(self):
        config = SemanticIndexConfig(enabled=True)
        index = SemanticIndex(config)

        # Manually add some data
        index.embeddings = []
        index.msg_ids = [1, 2, 3]

        index.clear()

        assert index.embeddings == []
        assert index.msg_ids == []


class TestNoOpSemanticIndex:
    def test_always_unavailable(self):
        index = NoOpSemanticIndex()
        assert index.is_available() is False

    def test_search_returns_empty(self):
        index = NoOpSemanticIndex()
        assert index.search("any query") == []

    def test_index_returns_false(self):
        index = NoOpSemanticIndex()
        store = ImmutableStore()
        assert index.index(store) is False


class TestCreateSemanticIndex:
    def test_disabled_returns_noop(self):
        config = SemanticIndexConfig(enabled=False)
        index = create_semantic_index(config)
        assert isinstance(index, NoOpSemanticIndex)

    def test_no_api_key_returns_noop(self):
        config = SemanticIndexConfig(enabled=True)
        with patch.dict(os.environ, {}, clear=True):
            index = create_semantic_index(config)
            assert isinstance(index, NoOpSemanticIndex)

    def test_enabled_with_key_returns_real_index(self):
        config = SemanticIndexConfig(enabled=True)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            index = create_semantic_index(config)
            assert isinstance(index, SemanticIndex)
