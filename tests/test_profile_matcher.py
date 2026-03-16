"""Tests for agent/profile_matcher.py — hardware profile matching.

All tests use synthetic data and do NOT depend on the actual local_profiles/
directory being present on the machine running the suite (except
test_load_profiles which verifies the shipped profiles load cleanly).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.profile_matcher import (
    PROFILES_DIR,
    format_profile_recommendation,
    load_profiles,
    match_profile,
)

# ---------------------------------------------------------------------------
# Minimal synthetic profiles used by most tests
# ---------------------------------------------------------------------------

_RTX_PROFILE = {
    "name": "RTX 3090 / RTX 4090 (24GB VRAM)",
    "gpu_vram_range": [20000, 25000],
    "description": "High-end NVIDIA GPU with 24GB VRAM",
    "recommended_models": {
        "main": [
            {
                "name": "qwen2.5-coder:32b-instruct-q4_K_M",
                "context": 32768,
                "description": "Best coding model that fits in 24GB",
            }
        ],
        "auxiliary": [
            {"name": "qwen2.5:3b-instruct-q8_0", "description": "Fast aux"}
        ],
        "vision": [
            {"name": "llava:13b-v1.6-q4_K_M", "description": "Vision"}
        ],
    },
    "search": {
        "backend": "searxng",
        "setup": "docker run -d -p 8888:8080 searxng/searxng",
    },
    "server": {"recommended": "ollama", "alternatives": ["vllm"]},
    "_file": "rtx_3090.yaml",
}

_APPLE_16_PROFILE = {
    "name": "Apple Silicon — 16GB Unified Memory (M1/M2/M3)",
    "unified_memory_range": [14000, 18000],
    "description": "Entry-level Apple Silicon Mac",
    "recommended_models": {
        "main": [
            {
                "name": "qwen2.5-coder:7b-instruct-q4_K_M",
                "context": 16384,
                "description": "Best coding model for 16GB",
            }
        ],
        "auxiliary": [{"name": "qwen2.5:0.5b-instruct", "description": "Minimal"}],
    },
    "search": {
        "backend": "searxng",
        "setup": "docker run -d -p 8888:8080 searxng/searxng",
    },
    "server": {"recommended": "ollama"},
    "_file": "apple_silicon_16gb.yaml",
}

_APPLE_32_PROFILE = {
    "name": "Apple Silicon — 32GB Unified Memory (M1 Pro/Max, M2 Pro)",
    "unified_memory_range": [28000, 36000],
    "description": "Mid-range Apple Silicon Mac",
    "recommended_models": {
        "main": [
            {
                "name": "qwen2.5-coder:14b-instruct-q6_K",
                "context": 32768,
                "description": "Strong coding model",
            }
        ],
        "auxiliary": [{"name": "qwen2.5:3b-instruct-q4_K_M", "description": "Capable"}],
    },
    "search": {
        "backend": "searxng",
        "setup": "docker run -d -p 8888:8080 searxng/searxng",
    },
    "server": {"recommended": "lm-studio"},
    "_file": "apple_silicon_32gb.yaml",
}

_APPLE_64_PROFILE = {
    "name": "Apple Silicon — 64GB+ Unified Memory (M1 Max/Ultra, M2 Max/Ultra, M3 Max, M4 Max)",
    "unified_memory_range": [60000, 200000],
    "description": "High-end Apple Silicon Mac",
    "recommended_models": {
        "main": [
            {
                "name": "qwen2.5-coder:32b-instruct-q6_K",
                "context": 65536,
                "description": "Top coding model",
            }
        ],
        "auxiliary": [{"name": "qwen2.5:7b-instruct-q4_K_M", "description": "Strong"}],
        "vision": [{"name": "llava:13b-v1.6-q4_K_M", "description": "High quality"}],
    },
    "search": {
        "backend": "searxng",
        "setup": "docker run -d -p 8888:8080 searxng/searxng",
    },
    "server": {"recommended": "lm-studio"},
    "_file": "apple_silicon_64gb.yaml",
}

_CPU_PROFILE = {
    "name": "CPU Only (No GPU)",
    "description": "Systems without dedicated GPU or insufficient VRAM",
    "recommended_models": {
        "main": [
            {
                "name": "qwen2.5:3b-instruct-q4_K_M",
                "context": 8192,
                "description": "Lightweight model for CPU inference",
            }
        ],
        "auxiliary": [],
    },
    "search": {
        "backend": "searxng",
        "setup": "docker run -d -p 8888:8080 searxng/searxng",
    },
    "server": {"recommended": "ollama", "alternatives": ["llama-cpp"]},
    "notes": "CPU inference is slow but functional.",
    "_file": "cpu_only.yaml",
}

_ALL_PROFILES = [
    _RTX_PROFILE,
    _APPLE_16_PROFILE,
    _APPLE_32_PROFILE,
    _APPLE_64_PROFILE,
    _CPU_PROFILE,
]


# ---------------------------------------------------------------------------
# test_load_profiles — verifies the shipped YAML files exist and parse
# ---------------------------------------------------------------------------


class TestLoadProfiles:
    def test_profiles_directory_exists(self):
        assert PROFILES_DIR.exists(), f"Expected {PROFILES_DIR} to exist"

    def test_loads_all_shipped_profiles(self):
        profiles = load_profiles()
        assert len(profiles) >= 5, f"Expected at least 5 profiles, got {len(profiles)}"

    def test_each_profile_has_name(self):
        for p in load_profiles():
            assert "name" in p, f"Profile {p.get('_file')} is missing 'name'"

    def test_each_profile_has_recommended_models(self):
        for p in load_profiles():
            assert "recommended_models" in p, (
                f"Profile {p.get('_file')} is missing 'recommended_models'"
            )

    def test_returns_empty_list_when_dir_missing(self):
        with patch("agent.profile_matcher.PROFILES_DIR", Path("/nonexistent/path")):
            profiles = load_profiles()
        assert profiles == []


# ---------------------------------------------------------------------------
# test_match_nvidia_rtx3090
# ---------------------------------------------------------------------------


class TestMatchNvidiaRtx3090:
    def test_exact_24gb_matches(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(gpu_vram_mb=24576, gpu_type="nvidia")
        assert result is not None
        assert result["_file"] == "rtx_3090.yaml"

    def test_lower_bound_20gb_matches(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(gpu_vram_mb=20000, gpu_type="nvidia")
        assert result is not None
        assert result["_file"] == "rtx_3090.yaml"

    def test_upper_bound_25gb_matches(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(gpu_vram_mb=25000, gpu_type="nvidia")
        assert result is not None
        assert result["_file"] == "rtx_3090.yaml"

    def test_rtx3090_profile_has_correct_name(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(gpu_vram_mb=24576, gpu_type="nvidia")
        assert result is not None
        assert "RTX 3090" in result["name"]


# ---------------------------------------------------------------------------
# test_match_apple_16gb
# ---------------------------------------------------------------------------


class TestMatchApple16Gb:
    def test_16gb_matches(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(unified_memory_mb=16384, gpu_type="apple")
        assert result is not None
        assert result["_file"] == "apple_silicon_16gb.yaml"

    def test_lower_bound_14gb_matches(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(unified_memory_mb=14000, gpu_type="apple")
        assert result is not None
        assert result["_file"] == "apple_silicon_16gb.yaml"

    def test_upper_bound_18gb_matches(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(unified_memory_mb=18000, gpu_type="apple")
        assert result is not None
        assert result["_file"] == "apple_silicon_16gb.yaml"


# ---------------------------------------------------------------------------
# test_match_apple_32gb
# ---------------------------------------------------------------------------


class TestMatchApple32Gb:
    def test_32gb_matches(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(unified_memory_mb=32768, gpu_type="apple")
        assert result is not None
        assert result["_file"] == "apple_silicon_32gb.yaml"

    def test_28gb_lower_bound_matches(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(unified_memory_mb=28000, gpu_type="apple")
        assert result is not None
        assert result["_file"] == "apple_silicon_32gb.yaml"

    def test_36gb_upper_bound_matches(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(unified_memory_mb=36000, gpu_type="apple")
        assert result is not None
        assert result["_file"] == "apple_silicon_32gb.yaml"


# ---------------------------------------------------------------------------
# test_match_apple_64gb
# ---------------------------------------------------------------------------


class TestMatchApple64Gb:
    def test_64gb_matches(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(unified_memory_mb=65536, gpu_type="apple")
        assert result is not None
        assert result["_file"] == "apple_silicon_64gb.yaml"

    def test_128gb_matches(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(unified_memory_mb=131072, gpu_type="apple")
        assert result is not None
        assert result["_file"] == "apple_silicon_64gb.yaml"

    def test_lower_bound_60gb_matches(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(unified_memory_mb=60000, gpu_type="apple")
        assert result is not None
        assert result["_file"] == "apple_silicon_64gb.yaml"


# ---------------------------------------------------------------------------
# test_match_cpu_only
# ---------------------------------------------------------------------------


class TestMatchCpuOnly:
    def test_no_gpu_falls_back_to_cpu(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(has_gpu=False, gpu_type="none")
        assert result is not None
        assert result["_file"] == "cpu_only.yaml"

    def test_zero_vram_nvidia_falls_back_to_cpu(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(gpu_vram_mb=0, gpu_type="nvidia")
        assert result is not None
        assert result["_file"] == "cpu_only.yaml"

    def test_zero_memory_apple_falls_back_to_cpu(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(unified_memory_mb=0, gpu_type="apple")
        assert result is not None
        assert result["_file"] == "cpu_only.yaml"

    def test_cpu_profile_name(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(has_gpu=False, gpu_type="none")
        assert result is not None
        assert "CPU" in result["name"]


# ---------------------------------------------------------------------------
# test_match_unknown_vram — unusual VRAM (e.g. 8GB RTX 3070) has no range match
# ---------------------------------------------------------------------------


class TestMatchUnknownVram:
    def test_8gb_nvidia_falls_back_to_cpu(self):
        """8GB VRAM doesn't match the [20000, 25000] RTX range; falls to CPU."""
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(gpu_vram_mb=8192, gpu_type="nvidia")
        assert result is not None
        assert result["_file"] == "cpu_only.yaml"

    def test_returns_none_when_no_profiles_at_all(self):
        with patch("agent.profile_matcher.load_profiles", return_value=[]):
            result = match_profile(gpu_vram_mb=8192, gpu_type="nvidia")
        assert result is None

    def test_19gb_just_below_rtx_range_falls_back(self):
        with patch("agent.profile_matcher.load_profiles", return_value=_ALL_PROFILES):
            result = match_profile(gpu_vram_mb=19999, gpu_type="nvidia")
        assert result is not None
        assert result["_file"] == "cpu_only.yaml"


# ---------------------------------------------------------------------------
# test_format_profile_recommendation
# ---------------------------------------------------------------------------


class TestFormatProfileRecommendation:
    def test_contains_profile_name(self):
        output = format_profile_recommendation(_RTX_PROFILE)
        assert "RTX 3090" in output

    def test_contains_main_model_name(self):
        output = format_profile_recommendation(_RTX_PROFILE)
        assert "qwen2.5-coder:32b-instruct-q4_K_M" in output

    def test_contains_auxiliary_model_name(self):
        output = format_profile_recommendation(_RTX_PROFILE)
        assert "qwen2.5:3b-instruct-q8_0" in output

    def test_contains_vision_model_name(self):
        output = format_profile_recommendation(_RTX_PROFILE)
        assert "llava:13b-v1.6-q4_K_M" in output

    def test_contains_server_recommendation(self):
        output = format_profile_recommendation(_RTX_PROFILE)
        assert "ollama" in output

    def test_contains_search_setup(self):
        output = format_profile_recommendation(_RTX_PROFILE)
        assert "searxng" in output

    def test_contains_notes_for_cpu_profile(self):
        output = format_profile_recommendation(_CPU_PROFILE)
        assert "slow" in output.lower()

    def test_no_vision_section_when_absent(self):
        output = format_profile_recommendation(_APPLE_16_PROFILE)
        assert "Vision models" not in output

    def test_no_notes_section_when_absent(self):
        output = format_profile_recommendation(_RTX_PROFILE)
        assert "Note:" not in output

    def test_context_length_shown_in_main_models(self):
        output = format_profile_recommendation(_RTX_PROFILE)
        assert "32768" in output
