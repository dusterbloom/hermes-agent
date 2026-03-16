"""Tests for VRAM-aware context sizing functions in model_metadata.py"""

import pytest
from unittest.mock import patch, MagicMock

from agent.model_metadata import (
    compute_kv_cache_bytes_per_token,
    practical_context_cap,
    compute_optimal_context_size,
)


class TestComputeKvCacheBytesPerToken:
    def test_llama_70b_architecture(self):
        # 70B: 80 layers, 8 kv_heads (GQA), 64 heads, 8192 embed_dim
        metadata = {"n_layers": 80, "n_kv_heads": 8, "n_heads": 64, "embedding_dim": 8192}
        result = compute_kv_cache_bytes_per_token(metadata)
        head_dim = 8192 // 64  # = 128
        expected = 2 * 80 * 8 * 128 * 2  # = 327680
        assert result == expected

    def test_small_7b_architecture(self):
        # 7B: 32 layers, 8 kv_heads, 32 heads, 4096 embed_dim
        metadata = {"n_layers": 32, "n_kv_heads": 8, "n_heads": 32, "embedding_dim": 4096}
        result = compute_kv_cache_bytes_per_token(metadata)
        head_dim = 4096 // 32  # = 128
        expected = 2 * 32 * 8 * 128 * 2  # = 131072
        assert result == expected

    def test_missing_fields_returns_zero(self):
        assert compute_kv_cache_bytes_per_token({}) == 0
        assert compute_kv_cache_bytes_per_token({"n_layers": 32}) == 0

    def test_zero_heads_returns_zero(self):
        metadata = {"n_layers": 32, "n_kv_heads": 0, "n_heads": 32, "embedding_dim": 4096}
        assert compute_kv_cache_bytes_per_token(metadata) == 0


class TestPracticalContextCap:
    def test_tiny_model(self):
        assert practical_context_cap(1 * 1024**3) == 8192  # 1GB

    def test_small_model(self):
        assert practical_context_cap(3 * 1024**3) == 16384  # 3GB

    def test_medium_model(self):
        assert practical_context_cap(6 * 1024**3) == 32768  # 6GB

    def test_large_model(self):
        assert practical_context_cap(12 * 1024**3) == 65536  # 12GB

    def test_very_large_model(self):
        assert practical_context_cap(40 * 1024**3) == 131072  # 40GB


class TestComputeOptimalContextSize:
    @patch("agent.model_metadata.get_available_vram_mb", return_value=0)
    def test_no_gpu_returns_model_context(self, mock_vram):
        result = compute_optimal_context_size(model_context_length=8192)
        assert result == 8192

    @patch("agent.model_metadata.get_available_vram_mb", return_value=0)
    def test_no_gpu_no_context_returns_default(self, mock_vram):
        result = compute_optimal_context_size()
        assert result == 4096

    def test_with_vram_and_kv_cost(self):
        # 24GB VRAM, 7B model (131072 bytes/token KV cache)
        with patch("agent.model_metadata.parse_gguf_metadata") as mock_parse, \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=4 * 1024**3):
            mock_parse.return_value = {
                "n_layers": 32, "n_kv_heads": 8, "n_heads": 32,
                "embedding_dim": 4096, "context_length": 32768
            }
            result = compute_optimal_context_size(
                gguf_path="/fake/model.gguf",
                available_vram_mb=24576,
            )
            assert result > 2048
            assert result <= 32768
            assert result % 1024 == 0  # rounded to 1024

    def test_minimum_context_enforced(self):
        # Very little VRAM
        with patch("agent.model_metadata.parse_gguf_metadata") as mock_parse, \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=40 * 1024**3):
            mock_parse.return_value = {
                "n_layers": 80, "n_kv_heads": 8, "n_heads": 64,
                "embedding_dim": 8192, "context_length": 131072
            }
            result = compute_optimal_context_size(
                gguf_path="/fake/model.gguf",
                available_vram_mb=600,  # barely anything
            )
            assert result >= 2048
