"""TDD tests for tiktoken integration in LCM token estimation.

RED phase: These tests define the expected behavior.
GREEN phase: Implementation makes them pass.

The tiktoken integration should:
- Use tiktoken for accurate token counts when available
- Fall back to chars//4 + overhead when tiktoken is unavailable (ImportError)
- Cache the encoding per model to avoid repeated lookups
- Fall back to cl100k_base for unknown model names (KeyError)
- Work via the engine's active_tokens() method
- Handle empty message lists
- Handle multimodal messages (image content parts) by counting text only
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, call
import pytest

from agent.lcm.tokens import TokenEstimator, TokenEstimatorConfig
from agent.lcm.engine import LcmEngine
from agent.lcm.config import LcmConfig
import agent.lcm.tokens as tokens_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_estimator_cache(estimator: TokenEstimator) -> None:
    """Reset lazy-init state so each test starts fresh."""
    estimator._initialized = False
    estimator._encoding = None


# ---------------------------------------------------------------------------
# 1. test_estimate_tokens_with_tiktoken
# ---------------------------------------------------------------------------

class TestEstimateTokensWithTiktoken:
    """When tiktoken is available, estimate() should return accurate counts."""

    def test_estimate_tokens_with_tiktoken(self):
        """Tiktoken-based estimate should be close to actual encoding length."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        # Use a GPT model so TokenEstimator activates tiktoken path
        config = TokenEstimatorConfig(model="gpt-4o", provider="openai")
        estimator = TokenEstimator(config)
        _clear_estimator_cache(estimator)

        content = "The quick brown fox jumps over the lazy dog"
        messages = [{"role": "user", "content": content}]

        result = estimator.estimate(messages)

        # Expected: actual tokens in content + MESSAGE_OVERHEAD_TOKENS (4)
        expected_content_tokens = len(enc.encode(content))
        expected = expected_content_tokens + 4  # per-message overhead

        assert result == expected, (
            f"Expected {expected} tokens (tiktoken), got {result}"
        )

    def test_estimate_tokens_not_using_char_division(self):
        """Tiktoken result must NOT equal chars//4 + overhead (it's more accurate)."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        config = TokenEstimatorConfig(model="gpt-4o", provider="openai")
        estimator = TokenEstimator(config)
        _clear_estimator_cache(estimator)

        # Choose content where chars//4 != actual tokens
        content = "Hello!"  # 6 chars -> chars//4 = 1, actual tiktoken = 2
        messages = [{"role": "user", "content": content}]

        tiktoken_result = estimator.estimate(messages)
        char_based = len(content) // 4 + 4  # fallback formula

        actual_content_tokens = len(enc.encode(content))
        actual_total = actual_content_tokens + 4

        assert tiktoken_result == actual_total
        # Confirm the two methods differ for this input (proving tiktoken is used)
        assert tiktoken_result != char_based


# ---------------------------------------------------------------------------
# 2. test_estimate_tokens_fallback_without_tiktoken
# ---------------------------------------------------------------------------

class TestEstimateTokensFallbackWithoutTiktoken:
    """When tiktoken is not importable, fall back to chars//4 + overhead."""

    def test_fallback_when_tiktoken_import_fails(self):
        """ImportError on tiktoken import should trigger char-based fallback."""
        config = TokenEstimatorConfig(
            model="gpt-4o",
            provider="openai",
            use_tiktoken=True,
        )
        estimator = TokenEstimator(config)
        _clear_estimator_cache(estimator)

        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        with patch.dict(sys.modules, {"tiktoken": None}):
            # Force re-init to pick up the mocked import
            _clear_estimator_cache(estimator)
            result = estimator.estimate(messages)

        # Fallback formula: sum(len(content)//ratio) + n*4
        # "Hello world" = 11 chars, "Hi there!" = 9 chars
        # ratio for "openai" provider maps to CHAR_RATIOS["gpt"] = 4
        # (11//4) + (9//4) + 2*4 = 2 + 2 + 8 = 12
        total_chars_estimate = (11 // 4) + (9 // 4) + 2 * 4
        assert result == total_chars_estimate

    def test_fallback_use_tiktoken_false(self):
        """use_tiktoken=False should always use char-based fallback."""
        config = TokenEstimatorConfig(
            model="gpt-4o",
            provider="openai",
            use_tiktoken=False,
        )
        estimator = TokenEstimator(config)

        messages = [{"role": "user", "content": "Hello world"}]
        result = estimator.estimate(messages)

        # Must use char-based path
        assert estimator._encoding is None
        # 11 chars // 4 = 2, + 4 overhead = 6
        assert result == (11 // 4) + 4


# ---------------------------------------------------------------------------
# 3. test_estimate_tokens_caches_encoding
# ---------------------------------------------------------------------------

class TestEstimateTokensCachesEncoding:
    """TokenEstimator should initialize encoding only once (lazy + cached)."""

    def test_lazy_init_called_once(self):
        """_lazy_init / _get_tiktoken_encoding should run only on first estimate."""
        config = TokenEstimatorConfig(model="gpt-4o", provider="openai")
        estimator = TokenEstimator(config)
        _clear_estimator_cache(estimator)

        messages = [{"role": "user", "content": "test"}]

        # Spy on _get_tiktoken_encoding
        original_get = estimator._get_tiktoken_encoding
        call_count = {"n": 0}

        def counting_get():
            call_count["n"] += 1
            return original_get()

        estimator._get_tiktoken_encoding = counting_get

        # Call estimate twice
        estimator.estimate(messages)
        estimator.estimate(messages)

        # _get_tiktoken_encoding should have been called only once
        # (second call uses cached _encoding via _initialized flag)
        assert call_count["n"] == 1, (
            f"Expected _get_tiktoken_encoding called 1 time, got {call_count['n']}"
        )

    def test_encoding_persists_across_calls(self):
        """After first estimate, _encoding should be set and reused."""
        config = TokenEstimatorConfig(model="gpt-4o", provider="openai")
        estimator = TokenEstimator(config)
        _clear_estimator_cache(estimator)

        messages = [{"role": "user", "content": "hello"}]
        estimator.estimate(messages)

        encoding_after_first = estimator._encoding
        assert encoding_after_first is not None

        estimator.estimate(messages)

        # Same object — not re-created
        assert estimator._encoding is encoding_after_first


# ---------------------------------------------------------------------------
# 4. test_estimate_tokens_unknown_model_fallback
# ---------------------------------------------------------------------------

class TestEstimateTokensUnknownModelFallback:
    """Unknown model names should fall back to cl100k_base, not crash."""

    def test_unknown_model_uses_cl100k_base(self):
        """A model not known to tiktoken should still return a valid estimate."""
        import tiktoken

        # Use a completely unknown model string with gpt prefix to trigger tiktoken path
        # The encoding resolution in _get_tiktoken_encoding will fall through to cl100k_base
        config = TokenEstimatorConfig(
            model="gpt-unknown-model-xyz-9999",
            provider="openai",
        )
        estimator = TokenEstimator(config)
        _clear_estimator_cache(estimator)

        messages = [{"role": "user", "content": "Hello world"}]
        result = estimator.estimate(messages)

        # Should not raise; should return a positive integer
        assert isinstance(result, int)
        assert result > 0

    def test_unknown_model_encoding_is_cl100k_base(self):
        """After resolving unknown model, encoding should be cl100k_base."""
        import tiktoken

        config = TokenEstimatorConfig(
            model="gpt-unknown-xyz",
            provider="openai",
        )
        estimator = TokenEstimator(config)
        _clear_estimator_cache(estimator)

        # Trigger init
        estimator.estimate([{"role": "user", "content": "test"}])

        # The encoding name should be cl100k_base
        assert estimator._encoding is not None
        expected_enc = tiktoken.get_encoding("cl100k_base")
        assert estimator._encoding.name == expected_enc.name


# ---------------------------------------------------------------------------
# 5. test_active_tokens_uses_tiktoken
# ---------------------------------------------------------------------------

class TestActiveTokensUsesTiktoken:
    """LcmEngine.active_tokens() should delegate to tiktoken when available."""

    def test_active_tokens_uses_tiktoken_for_gpt_model(self):
        """active_tokens() with a GPT model should use tiktoken encoding."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        config = LcmConfig()
        engine = LcmEngine(config, model="gpt-4o", provider="openai")
        # Reset estimator state for clean test
        _clear_estimator_cache(engine.token_estimator)

        content = "The quick brown fox"
        engine.ingest({"role": "user", "content": content})

        result = engine.active_tokens()

        expected = len(enc.encode(content)) + 4  # content tokens + overhead
        assert result == expected

    def test_active_tokens_positive(self):
        """active_tokens() must always return a positive integer after ingestion."""
        config = LcmConfig()
        engine = LcmEngine(config, model="gpt-4o", provider="openai")
        _clear_estimator_cache(engine.token_estimator)

        engine.ingest({"role": "user", "content": "Hello"})
        result = engine.active_tokens()

        assert isinstance(result, int)
        assert result > 0


# ---------------------------------------------------------------------------
# 6. test_estimate_tokens_empty_messages
# ---------------------------------------------------------------------------

class TestEstimateTokensEmptyMessages:
    """Empty message list should return 0."""

    def test_empty_list_returns_zero_char_based(self):
        config = TokenEstimatorConfig(use_tiktoken=False)
        estimator = TokenEstimator(config)

        result = estimator.estimate([])
        assert result == 0

    def test_empty_list_returns_zero_tiktoken(self):
        config = TokenEstimatorConfig(model="gpt-4o", provider="openai")
        estimator = TokenEstimator(config)
        _clear_estimator_cache(estimator)

        result = estimator.estimate([])
        assert result == 0

    def test_message_with_empty_content(self):
        """A message with empty string content should only contribute overhead."""
        config = TokenEstimatorConfig(model="gpt-4o", provider="openai")
        estimator = TokenEstimator(config)
        _clear_estimator_cache(estimator)

        result = estimator.estimate([{"role": "user", "content": ""}])
        # Only overhead: 4 tokens
        assert result == 4

    def test_message_with_none_content(self):
        """A message with None content should only contribute overhead."""
        config = TokenEstimatorConfig(model="gpt-4o", provider="openai")
        estimator = TokenEstimator(config)
        _clear_estimator_cache(estimator)

        result = estimator.estimate([{"role": "user", "content": None}])
        # None -> "" -> 0 content tokens + 4 overhead
        assert result == 4


# ---------------------------------------------------------------------------
# 7. test_estimate_tokens_multimodal_messages
# ---------------------------------------------------------------------------

class TestEstimateTokensMultimodalMessages:
    """Messages with image content parts should count text portions only."""

    def test_multimodal_message_counts_text_parts_only(self):
        """Image parts in multimodal content should not contribute to token count."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        config = TokenEstimatorConfig(model="gpt-4o", provider="openai")
        estimator = TokenEstimator(config)
        _clear_estimator_cache(estimator)

        text_part = "Describe this image"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_part},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                ],
            }
        ]

        result = estimator.estimate(messages)

        # Only the text part should be counted
        expected = len(enc.encode(text_part)) + 4
        assert result == expected

    def test_multimodal_mixed_text_types(self):
        """Both 'text' and 'input_text' part types should be counted."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        config = TokenEstimatorConfig(model="gpt-4o", provider="openai")
        estimator = TokenEstimator(config)
        _clear_estimator_cache(estimator)

        text1 = "Hello"
        text2 = "World"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text1},
                    {"type": "input_text", "text": text2},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]

        result = estimator.estimate(messages)

        expected = len(enc.encode(text1)) + len(enc.encode(text2)) + 4
        assert result == expected

    def test_multimodal_str_content_unchanged(self):
        """String content in messages should still work (not multimodal path)."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        config = TokenEstimatorConfig(model="gpt-4o", provider="openai")
        estimator = TokenEstimator(config)
        _clear_estimator_cache(estimator)

        content = "Plain string content"
        messages = [{"role": "user", "content": content}]

        result = estimator.estimate(messages)
        expected = len(enc.encode(content)) + 4
        assert result == expected
