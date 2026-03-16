"""Tests for agent/model_capabilities.py — pure-function model registry."""

import sys
import os

# Allow importing from the agent package without a full install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))

from model_capabilities import (
    ModelCapabilities,
    ModelSizeClass,
    ReaderTier,
    _has_size_marker,
    lookup,
    supports_native_tool_calling,
    supports_vision,
    needs_strict_alternation,
)


# ---------------------------------------------------------------------------
# _has_size_marker — pure-function unit tests
# ---------------------------------------------------------------------------


class TestHasSizeMarker:
    def test_exact_match(self):
        assert _has_size_marker("llama-3b-instruct", "3b") is True

    def test_no_match(self):
        assert _has_size_marker("llama-8b-instruct", "3b") is False

    def test_digit_prefix_prevents_match(self):
        # "35b" must NOT match "3b"
        assert _has_size_marker("qwen2.5-35b-instruct", "3b") is False

    def test_letter_prefix_prevents_match(self):
        # MoE active-param suffix "a3b" must NOT match "3b"
        assert _has_size_marker("mixtral-a3b", "3b") is False

    def test_at_start_of_string(self):
        assert _has_size_marker("3b-model", "3b") is True

    def test_separator_before_marker(self):
        assert _has_size_marker("model-7b-q4", "7b") is True

    def test_dot_before_marker(self):
        # e.g. "1.5b" — the '5' is a digit, so "5b" would be blocked,
        # but the full marker "1.5b" should match at position 0 of the substring
        assert _has_size_marker("phi-1.5b", "1.5b") is True

    def test_72b_does_not_match_2b(self):
        assert _has_size_marker("model-72b", "2b") is False

    def test_70b_match(self):
        assert _has_size_marker("llama-3-70b-instruct", "70b") is True

    def test_marker_not_present(self):
        assert _has_size_marker("claude-sonnet", "70b") is False


# ---------------------------------------------------------------------------
# Cloud models
# ---------------------------------------------------------------------------


class TestCloudModels:
    def test_claude_tool_calling(self):
        caps = lookup("claude-3-5-sonnet-20241022")
        assert caps.tool_calling is True

    def test_claude_vision(self):
        assert lookup("claude-3-haiku").vision is True

    def test_claude_no_strict_alternation(self):
        assert lookup("claude-opus-4").strict_alternation is False

    def test_claude_reader_tier(self):
        assert lookup("claude-opus-4").reader_tier == ReaderTier.ADVANCED

    def test_claude_max_output(self):
        assert lookup("claude-sonnet").max_output_tokens == 16384

    def test_gpt4_tool_calling(self):
        assert lookup("gpt-4o").tool_calling is True

    def test_gpt4_vision(self):
        assert lookup("gpt-4o").vision is True

    def test_gpt4_no_strict_alternation(self):
        assert lookup("gpt-4-turbo").strict_alternation is False

    def test_gpt35_no_vision(self):
        assert lookup("gpt-3.5-turbo").vision is False

    def test_gpt35_tool_calling(self):
        assert lookup("gpt-3.5-turbo").tool_calling is True

    def test_gemini_tool_calling(self):
        assert lookup("gemini-1.5-pro").tool_calling is True

    def test_gemini_vision(self):
        assert lookup("gemini-2.0-flash").vision is True

    def test_gemini_no_strict_alternation(self):
        assert lookup("gemini-pro").strict_alternation is False


# ---------------------------------------------------------------------------
# Local — Qwen family
# ---------------------------------------------------------------------------


class TestQwenModels:
    def test_qwen25_coder_tool_calling(self):
        assert lookup("qwen2.5-coder:32b-instruct-q4_K_M").tool_calling is True

    def test_qwen25_coder_size_adjusted_to_large(self):
        caps = lookup("qwen2.5-coder:32b-instruct-q4_K_M")
        assert caps.size_class == ModelSizeClass.LARGE

    def test_qwen25_coder_7b_size_medium(self):
        caps = lookup("qwen2.5-coder:7b-instruct")
        assert caps.size_class == ModelSizeClass.MEDIUM

    def test_qwen3_thinking(self):
        assert lookup("qwen3-30b-a3b").thinking is True

    def test_qwen3_a3b_size_not_small(self):
        # "a3b" suffix must NOT trigger SMALL via "3b" marker
        caps = lookup("qwen3-30b-a3b")
        assert caps.size_class == ModelSizeClass.LARGE  # matched by "32b"-class? No — 30b not in table
        # 30b is not in the size-marker table so size stays at whatever pattern gives
        # qwen3 pattern_caps has LARGE, and "30b" marker is not in the table → stays LARGE
        assert caps.size_class == ModelSizeClass.LARGE

    def test_qwen_vl_vision(self):
        assert lookup("qwen-vl-chat").vision is True

    def test_qwen25_strict_alternation(self):
        assert lookup("qwen2.5-72b-instruct").strict_alternation is True


# ---------------------------------------------------------------------------
# Local — Hermes (NousResearch)
# ---------------------------------------------------------------------------


class TestHermesModels:
    def test_hermes3_tool_calling(self):
        assert lookup("hermes-3-llama-3.1-8b").tool_calling is True

    def test_hermes3_reader_tier(self):
        assert lookup("hermes-3-llama-3.1-70b").reader_tier == ReaderTier.ADVANCED

    def test_hermes2_medium(self):
        caps = lookup("hermes-2-pro-mistral-7b")
        assert caps.size_class == ModelSizeClass.MEDIUM  # 7b marker -> MEDIUM

    def test_hermes2_tool_calling(self):
        assert lookup("nous-hermes-2-mixtral-8x7b").tool_calling is True

    def test_hermes3_strict_alternation(self):
        assert lookup("hermes-3-llama-3.1-8b").strict_alternation is True


# ---------------------------------------------------------------------------
# Local — Llama family
# ---------------------------------------------------------------------------


class TestLlamaModels:
    def test_llama31_tool_calling(self):
        assert lookup("llama-3.1-8b-instruct").tool_calling is True

    def test_llama31_8b_size_medium(self):
        caps = lookup("llama-3.1-8b-instruct")
        assert caps.size_class == ModelSizeClass.MEDIUM

    def test_llama31_70b_size_large(self):
        caps = lookup("llama-3.1-70b-instruct")
        assert caps.size_class == ModelSizeClass.LARGE

    def test_llama33_advanced(self):
        assert lookup("llama-3.3-70b-instruct").reader_tier == ReaderTier.ADVANCED

    def test_llama32_standard(self):
        assert lookup("llama-3.2-3b-instruct").reader_tier == ReaderTier.STANDARD

    def test_llama32_3b_size_small(self):
        caps = lookup("llama-3.2-3b-instruct")
        assert caps.size_class == ModelSizeClass.SMALL

    def test_llama3_base_no_tool_calling(self):
        # plain "llama-3" without .1/.2/.3 suffix
        assert lookup("llama-3-8b").tool_calling is False

    def test_llama4_vision(self):
        assert lookup("llama-4-scout").vision is True

    def test_llama4_thinking(self):
        assert lookup("llama-4-maverick").thinking is True


# ---------------------------------------------------------------------------
# Local — Mistral / Mixtral
# ---------------------------------------------------------------------------


class TestMistralModels:
    def test_mistral_large_tool_calling(self):
        assert lookup("mistral-large-2411").tool_calling is True

    def test_mistral_large_reader_tier(self):
        assert lookup("mistral-large-2411").reader_tier == ReaderTier.ADVANCED

    def test_mixtral_tool_calling(self):
        assert lookup("mixtral-8x7b-instruct").tool_calling is True

    def test_mistral_base_medium(self):
        caps = lookup("mistral-7b-instruct")
        assert caps.size_class == ModelSizeClass.MEDIUM  # 7b marker -> MEDIUM

    def test_mistral_base_tool_calling(self):
        assert lookup("mistral-7b-instruct").tool_calling is True


# ---------------------------------------------------------------------------
# Local — DeepSeek
# ---------------------------------------------------------------------------


class TestDeepSeekModels:
    def test_deepseek_r1_thinking(self):
        assert lookup("deepseek-r1-distill-llama-70b").thinking is True

    def test_deepseek_r1_no_tool_calling(self):
        assert lookup("deepseek-r1").tool_calling is False

    def test_deepseek_v3_tool_calling(self):
        assert lookup("deepseek-v3-0324").tool_calling is True

    def test_deepseek_coder_tool_calling(self):
        assert lookup("deepseek-coder-v2-lite").tool_calling is True

    def test_deepseek_base_no_tool_calling(self):
        assert lookup("deepseek-llm-67b-chat").tool_calling is False

    def test_deepseek_base_medium(self):
        caps = lookup("deepseek-llm-7b-chat")
        assert caps.size_class == ModelSizeClass.MEDIUM


# ---------------------------------------------------------------------------
# Local — Phi
# ---------------------------------------------------------------------------


class TestPhiModels:
    def test_phi4_tool_calling(self):
        assert lookup("phi-4").tool_calling is True

    def test_phi35_no_tool_calling(self):
        assert lookup("phi-3.5-mini-instruct").tool_calling is False

    def test_phi35_small(self):
        assert lookup("phi-3.5-mini-instruct").size_class == ModelSizeClass.SMALL

    def test_phi3_minimal_reader(self):
        assert lookup("phi-3-mini-4k-instruct").reader_tier == ReaderTier.MINIMAL

    def test_phi4_not_matched_by_phi3(self):
        # "phi-4" must match the "phi-4" pattern, not "phi-3"
        assert lookup("phi-4").tool_calling is True


# ---------------------------------------------------------------------------
# Local — Gemma
# ---------------------------------------------------------------------------


class TestGemmaModels:
    def test_gemma2_no_tool_calling(self):
        assert lookup("gemma-2-9b-it").tool_calling is False

    def test_gemma2_9b_size_medium(self):
        caps = lookup("gemma-2-9b-it")
        assert caps.size_class == ModelSizeClass.MEDIUM

    def test_gemma_base_small(self):
        assert lookup("gemma-2b").size_class == ModelSizeClass.SMALL

    def test_gemma2_not_matched_by_gemma(self):
        # "gemma-2" pattern should be hit before "gemma"
        caps = lookup("gemma-2-27b-it")
        # 27b is not in the size-marker table; gemma-2 pattern defaults to MEDIUM
        assert caps.size_class == ModelSizeClass.MEDIUM


# ---------------------------------------------------------------------------
# Vision models
# ---------------------------------------------------------------------------


class TestVisionModels:
    def test_llava_vision(self):
        assert lookup("llava-v1.6-mistral-7b").vision is True

    def test_bakllava_vision(self):
        assert lookup("bakllava-1").vision is True

    def test_llava_no_tool_calling(self):
        assert lookup("llava-13b").tool_calling is False


# ---------------------------------------------------------------------------
# Size marker adjustments
# ---------------------------------------------------------------------------


class TestSizeMarkerAdjustments:
    def test_3b_gives_small(self):
        caps = lookup("some-model-3b")
        assert caps.size_class == ModelSizeClass.SMALL

    def test_35b_does_not_give_small(self):
        # "35b" must NOT trigger the "3b" SMALL marker
        caps = lookup("qwen2.5-35b-instruct")
        assert caps.size_class != ModelSizeClass.SMALL

    def test_a3b_does_not_give_small(self):
        # MoE active-param "a3b" must not trigger "3b"
        caps = lookup("qwen3-30b-a3b")
        assert caps.size_class == ModelSizeClass.LARGE

    def test_7b_gives_medium(self):
        caps = lookup("llama-3.1-7b")
        assert caps.size_class == ModelSizeClass.MEDIUM

    def test_8b_gives_medium(self):
        caps = lookup("llama-3.1-8b-instruct")
        assert caps.size_class == ModelSizeClass.MEDIUM

    def test_70b_gives_large(self):
        caps = lookup("llama-3.1-70b-instruct")
        assert caps.size_class == ModelSizeClass.LARGE

    def test_72b_gives_large(self):
        caps = lookup("qwen2.5-72b-instruct")
        assert caps.size_class == ModelSizeClass.LARGE

    def test_1b_gives_small(self):
        caps = lookup("llama-3.2-1b-instruct")
        assert caps.size_class == ModelSizeClass.SMALL

    def test_14b_gives_medium(self):
        caps = lookup("qwen2.5-14b-instruct")
        assert caps.size_class == ModelSizeClass.MEDIUM

    def test_13b_gives_medium(self):
        caps = lookup("llama-2-13b-chat")
        assert caps.size_class == ModelSizeClass.MEDIUM

    def test_32b_gives_large(self):
        caps = lookup("qwen2.5-coder:32b-instruct")
        assert caps.size_class == ModelSizeClass.LARGE


# ---------------------------------------------------------------------------
# Provider prefix stripping
# ---------------------------------------------------------------------------


class TestProviderPrefixStripping:
    def test_ollama_prefix_stripped(self):
        caps_plain = lookup("qwen2.5-coder:7b-instruct")
        caps_prefixed = lookup("ollama/qwen2.5-coder:7b-instruct")
        assert caps_plain.tool_calling == caps_prefixed.tool_calling
        assert caps_plain.size_class == caps_prefixed.size_class

    def test_openai_prefix_stripped(self):
        caps = lookup("openai/gpt-4o")
        assert caps.tool_calling is True
        assert caps.vision is True

    def test_anthropic_prefix_stripped(self):
        caps = lookup("anthropic/claude-3-5-sonnet")
        assert caps.tool_calling is True
        assert caps.strict_alternation is False

    def test_generic_prefix_stripped(self):
        caps = lookup("provider/hermes-3-llama-3.1-70b")
        assert caps.tool_calling is True


# ---------------------------------------------------------------------------
# User overrides
# ---------------------------------------------------------------------------


class TestUserOverrides:
    def test_override_tool_calling(self):
        overrides = {"my-custom-model": {"tool_calling": True}}
        caps = lookup("my-custom-model-7b", user_overrides=overrides)
        assert caps.tool_calling is True

    def test_override_vision(self):
        overrides = {"qwen2.5": {"vision": True}}
        caps = lookup("qwen2.5-7b-instruct", user_overrides=overrides)
        assert caps.vision is True

    def test_override_max_output_tokens(self):
        overrides = {"llama-3.1": {"max_output_tokens": 32768}}
        caps = lookup("llama-3.1-8b-instruct", user_overrides=overrides)
        assert caps.max_output_tokens == 32768

    def test_override_context_length(self):
        overrides = {"hermes-3": {"context_length": 131072}}
        caps = lookup("hermes-3-llama-3.1-70b", user_overrides=overrides)
        assert caps.context_length == 131072

    def test_override_unknown_key_is_ignored(self):
        overrides = {"llama-3.1": {"nonexistent_field": "some_value"}}
        caps = lookup("llama-3.1-8b-instruct", user_overrides=overrides)
        assert not hasattr(caps, "nonexistent_field")

    def test_override_first_match_wins(self):
        # Both keys would match; the first one in dict iteration order should win.
        overrides = {"llama": {"max_output_tokens": 1111}, "llama-3.1": {"max_output_tokens": 2222}}
        caps = lookup("llama-3.1-8b-instruct", user_overrides=overrides)
        # Whichever key appears first in dict iteration wins
        expected = next(
            v["max_output_tokens"]
            for k, v in overrides.items()
            if k.lower() in "llama-3.1-8b-instruct"
        )
        assert caps.max_output_tokens == expected

    def test_no_overrides(self):
        caps = lookup("llama-3.1-8b-instruct", user_overrides=None)
        assert caps.tool_calling is True  # from built-in pattern

    def test_empty_overrides(self):
        caps = lookup("llama-3.1-8b-instruct", user_overrides={})
        assert caps.tool_calling is True


# ---------------------------------------------------------------------------
# Default fallback for unknown models
# ---------------------------------------------------------------------------


class TestDefaultFallback:
    def test_unknown_model_returns_defaults(self):
        caps = lookup("completely-unknown-model-xyz")
        assert isinstance(caps, ModelCapabilities)
        assert caps.tool_calling is False
        assert caps.vision is False
        assert caps.thinking is False
        assert caps.strict_alternation is True
        assert caps.size_class == ModelSizeClass.MEDIUM
        assert caps.reader_tier == ReaderTier.STANDARD
        assert caps.max_output_tokens == 4096
        assert caps.context_length is None

    def test_empty_string_returns_defaults(self):
        caps = lookup("")
        assert isinstance(caps, ModelCapabilities)
        assert caps.tool_calling is False


# ---------------------------------------------------------------------------
# Convenience helper functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_supports_native_tool_calling_true(self):
        assert supports_native_tool_calling("claude-3-5-sonnet") is True

    def test_supports_native_tool_calling_false(self):
        assert supports_native_tool_calling("deepseek-r1") is False

    def test_supports_vision_true(self):
        assert supports_vision("gpt-4o") is True

    def test_supports_vision_false(self):
        assert supports_vision("llama-3.1-8b-instruct") is False

    def test_needs_strict_alternation_local(self):
        assert needs_strict_alternation("qwen2.5-7b-instruct") is True

    def test_needs_strict_alternation_cloud(self):
        assert needs_strict_alternation("claude-opus-4") is False

    def test_convenience_passes_overrides(self):
        overrides = {"deepseek-r1": {"tool_calling": True}}
        assert supports_native_tool_calling("deepseek-r1", user_overrides=overrides) is True

    def test_nemotron_thinking(self):
        caps = lookup("nvidia/nemotron-ultra-253b-v1")
        assert caps.thinking is True
        assert caps.tool_calling is True
