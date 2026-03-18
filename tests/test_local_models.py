"""Tests for agent/local_models.py.

Tests covering WSL detection, model path resolution, server probing,
server detection priority, and model listing.
"""

import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import httpx
import pytest

from agent.local_models import (
    LocalModel,
    ServerStatus,
    _detect_wsl_host_ip,
    _probe_server,
    detect_best_server,
    detect_running_servers,
    get_openai_base_url,
    list_models,
    resolve_model_path,
)


# ---------------------------------------------------------------------------
# _detect_wsl_host_ip
# ---------------------------------------------------------------------------


class TestDetectWslHostIp:
    def test_returns_host_ip_on_wsl(self, tmp_path):
        proc_version = "Linux version 5.15.167.4-microsoft-standard-WSL2"
        resolv_conf = "nameserver 172.28.208.1\n"

        with (
            patch("builtins.open", side_effect=_make_open_map({
                "/proc/version": proc_version,
                "/etc/resolv.conf": resolv_conf,
            })),
        ):
            result = _detect_wsl_host_ip()
        assert result == "172.28.208.1"

    def test_returns_none_on_non_wsl(self):
        proc_version = "Linux version 5.15.0-generic Ubuntu"

        with patch("builtins.open", side_effect=_make_open_map({
            "/proc/version": proc_version,
        })):
            result = _detect_wsl_host_ip()
        assert result is None

    def test_skips_loopback_nameserver(self):
        proc_version = "Linux version 5.15.167.4-microsoft-standard-WSL2"
        # 127.0.0.53 is systemd-resolved loopback
        resolv_conf = "nameserver 127.0.0.53\n"

        with patch("builtins.open", side_effect=_make_open_map({
            "/proc/version": proc_version,
            "/etc/resolv.conf": resolv_conf,
        })):
            result = _detect_wsl_host_ip()
        assert result is None

    def test_returns_none_when_proc_version_missing(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = _detect_wsl_host_ip()
        assert result is None

    def test_returns_none_when_resolv_conf_missing(self):
        proc_version = "Linux version 5.15.167.4-microsoft-standard-WSL2"

        def open_side_effect(path, *args, **kwargs):
            if "/proc/version" in str(path):
                return mock_open(read_data=proc_version)()
            raise FileNotFoundError(path)

        with patch("builtins.open", side_effect=open_side_effect):
            result = _detect_wsl_host_ip()
        assert result is None

    def test_picks_first_non_loopback_nameserver(self):
        proc_version = "Linux version 5.15.167.4-microsoft-WSL2"
        # First nameserver is loopback, second is the host
        resolv_conf = "nameserver 127.0.0.53\nnameserver 192.168.1.1\n"

        with patch("builtins.open", side_effect=_make_open_map({
            "/proc/version": proc_version,
            "/etc/resolv.conf": resolv_conf,
        })):
            result = _detect_wsl_host_ip()
        assert result == "192.168.1.1"


# ---------------------------------------------------------------------------
# resolve_model_path
# ---------------------------------------------------------------------------


class TestResolveModelPath:
    def test_absolute_path_returned_as_is(self, tmp_path):
        model_file = tmp_path / "my_model.gguf"
        model_file.touch()
        result = resolve_model_path(str(model_file))
        assert result == str(model_file)

    def test_relative_to_models_dir(self, tmp_path):
        model_file = tmp_path / "llama-7b.gguf"
        model_file.touch()
        result = resolve_model_path("llama-7b.gguf", models_dir=str(tmp_path))
        assert result == str(model_file)

    def test_gguf_extension_auto_appended(self, tmp_path):
        model_file = tmp_path / "llama-7b.gguf"
        model_file.touch()
        # Pass name without extension
        result = resolve_model_path("llama-7b", models_dir=str(tmp_path))
        assert result == str(model_file)

    def test_fuzzy_search_by_name(self, tmp_path):
        model_file = tmp_path / "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        model_file.touch()
        # Search with a partial name
        result = resolve_model_path("qwen2.5-coder-7b", models_dir=str(tmp_path))
        assert result == str(model_file)

    def test_returns_none_when_not_found(self, tmp_path):
        result = resolve_model_path("nonexistent-model", models_dir=str(tmp_path))
        assert result is None

    def test_absolute_path_not_existing_returns_none(self, tmp_path):
        fake_path = str(tmp_path / "does_not_exist.gguf")
        result = resolve_model_path(fake_path)
        assert result is None

    def test_models_dir_does_not_exist(self, tmp_path):
        # Directory that doesn't exist — should return None without raising
        result = resolve_model_path("model", models_dir=str(tmp_path / "nonexistent"))
        assert result is None


# ---------------------------------------------------------------------------
# get_openai_base_url
# ---------------------------------------------------------------------------


class TestGetOpenaiBaseUrl:
    def test_ollama_adds_v1(self):
        url = get_openai_base_url("http://localhost:11434", "ollama")
        assert url == "http://localhost:11434/v1"

    def test_lm_studio_adds_v1_if_missing(self):
        url = get_openai_base_url("http://localhost:1234", "lm-studio")
        assert url == "http://localhost:1234/v1"

    def test_does_not_double_add_v1(self):
        url = get_openai_base_url("http://localhost:1234/v1", "lm-studio")
        assert url == "http://localhost:1234/v1"

    def test_vllm_adds_v1(self):
        url = get_openai_base_url("http://localhost:8000", "vllm")
        assert url == "http://localhost:8000/v1"

    def test_llama_cpp_adds_v1(self):
        url = get_openai_base_url("http://localhost:8080", "llama-cpp")
        assert url == "http://localhost:8080/v1"

    def test_trailing_slash_stripped(self):
        url = get_openai_base_url("http://localhost:1234/", "lm-studio")
        assert url == "http://localhost:1234/v1"


# ---------------------------------------------------------------------------
# _probe_server
# ---------------------------------------------------------------------------


class TestProbeServer:
    def test_probe_ollama_running(self):
        root_resp = MagicMock()
        root_resp.status_code = 200

        tags_resp = MagicMock()
        tags_resp.status_code = 200
        tags_resp.json.return_value = {
            "models": [
                {"name": "llama3:latest"},
                {"name": "qwen2.5-coder:7b"},
            ]
        }

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get.side_effect = [root_resp, tags_resp]

        with patch("agent.local_models.httpx.Client", return_value=mock_client):
            status = _probe_server("http://127.0.0.1:11434", "ollama", 2.0)

        assert status is not None
        assert status.running is True
        assert status.server_type == "ollama"
        assert "llama3:latest" in status.models_loaded
        assert "qwen2.5-coder:7b" in status.models_loaded

    def test_probe_ollama_not_running(self):
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("connection refused")

        with patch("agent.local_models.httpx.Client", return_value=mock_client):
            status = _probe_server("http://127.0.0.1:11434", "ollama", 2.0)

        assert status is None

    def test_probe_lm_studio_running(self):
        models_resp = MagicMock()
        models_resp.status_code = 200
        models_resp.json.return_value = {
            "data": [{"id": "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"}]
        }

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get.return_value = models_resp

        with patch("agent.local_models.httpx.Client", return_value=mock_client):
            status = _probe_server("http://127.0.0.1:1234", "lm-studio", 2.0)

        assert status is not None
        assert status.running is True
        assert status.server_type == "lm-studio"
        assert "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF" in status.models_loaded

    def test_probe_timeout_returns_none(self):
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.TimeoutException("timed out")

        with patch("agent.local_models.httpx.Client", return_value=mock_client):
            status = _probe_server("http://127.0.0.1:11434", "ollama", 0.001)

        assert status is None

    def test_probe_non_200_returns_none(self):
        resp = MagicMock()
        resp.status_code = 404

        mock_client = MagicMock()
        mock_client.get.return_value = resp

        with patch("agent.local_models.httpx.Client", return_value=mock_client):
            status = _probe_server("http://127.0.0.1:11434", "ollama", 2.0)

        assert status is None


# ---------------------------------------------------------------------------
# detect_running_servers / detect_best_server
# ---------------------------------------------------------------------------


class TestDetectRunningServers:
    def test_returns_running_servers(self):
        ollama_status = ServerStatus(
            server_type="ollama",
            url="http://127.0.0.1:11434",
            running=True,
            models_loaded=["llama3:latest"],
        )

        def fake_probe(url, server_type, timeout):
            if "11434" in url and server_type == "ollama":
                return ollama_status
            return None

        with (
            patch("agent.local_models._probe_server", side_effect=fake_probe),
            patch("agent.local_models._detect_wsl_host_ip", return_value=None),
        ):
            servers = detect_running_servers(timeout=2.0)

        assert len(servers) == 1
        assert servers[0].server_type == "ollama"

    def test_includes_wsl_host(self):
        wsl_status = ServerStatus(
            server_type="lm-studio",
            url="http://172.28.208.1:1234",
            running=True,
            models_loaded=[],
        )

        def fake_probe(url, server_type, timeout):
            if "172.28.208.1" in url and server_type == "lm-studio":
                return wsl_status
            return None

        with (
            patch("agent.local_models._probe_server", side_effect=fake_probe),
            patch("agent.local_models._detect_wsl_host_ip", return_value="172.28.208.1"),
        ):
            servers = detect_running_servers(timeout=2.0)

        assert len(servers) == 1
        assert servers[0].url == "http://172.28.208.1:1234"

    def test_returns_empty_when_none_running(self):
        with (
            patch("agent.local_models._probe_server", return_value=None),
            patch("agent.local_models._detect_wsl_host_ip", return_value=None),
        ):
            servers = detect_running_servers(timeout=2.0)

        assert servers == []


class TestDetectBestServer:
    def test_ollama_preferred_over_lm_studio(self):
        ollama = ServerStatus("ollama", "http://127.0.0.1:11434", True)
        lm_studio = ServerStatus("lm-studio", "http://127.0.0.1:1234", True)

        with patch("agent.local_models.detect_running_servers", return_value=[lm_studio, ollama]):
            best = detect_best_server()

        assert best is not None
        assert best.server_type == "ollama"

    def test_lm_studio_preferred_over_vllm(self):
        lm_studio = ServerStatus("lm-studio", "http://127.0.0.1:1234", True)
        vllm = ServerStatus("vllm", "http://127.0.0.1:8000", True)

        with patch("agent.local_models.detect_running_servers", return_value=[vllm, lm_studio]):
            best = detect_best_server()

        assert best is not None
        assert best.server_type == "lm-studio"

    def test_returns_none_when_no_servers(self):
        with patch("agent.local_models.detect_running_servers", return_value=[]):
            best = detect_best_server()

        assert best is None

    def test_returns_only_server(self):
        vllm = ServerStatus("vllm", "http://127.0.0.1:8000", True)

        with patch("agent.local_models.detect_running_servers", return_value=[vllm]):
            best = detect_best_server()

        assert best is not None
        assert best.server_type == "vllm"


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


class TestListModels:
    def test_list_models_ollama(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "models": [
                {"name": "llama3:latest", "size": 4_000_000_000},
                {"name": "qwen2.5-coder:7b", "size": 3_500_000_000},
            ]
        }

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get.return_value = resp

        with patch("agent.local_models.httpx.Client", return_value=mock_client):
            models = list_models("http://localhost:11434", "ollama")

        assert len(models) == 2
        assert models[0].name == "llama3:latest"
        assert models[0].size_bytes == 4_000_000_000
        assert models[0].loaded is True
        assert models[0].server_type == "ollama"
        assert models[1].name == "qwen2.5-coder:7b"

    def test_list_models_lm_studio(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": [
                {"id": "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"},
            ]
        }

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get.return_value = resp

        with patch("agent.local_models.httpx.Client", return_value=mock_client):
            models = list_models("http://localhost:1234", "lm-studio")

        assert len(models) == 1
        assert models[0].name == "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
        assert models[0].server_type == "lm-studio"

    def test_list_models_vllm(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": [{"id": "Qwen/Qwen2.5-Coder-7B-Instruct"}]
        }

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get.return_value = resp

        with patch("agent.local_models.httpx.Client", return_value=mock_client):
            models = list_models("http://localhost:8000", "vllm")

        assert len(models) == 1
        assert models[0].name == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert models[0].server_type == "vllm"

    def test_list_models_error_returns_empty(self):
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("refused")

        with patch("agent.local_models.httpx.Client", return_value=mock_client):
            models = list_models("http://localhost:11434", "ollama")

        assert models == []

    def test_list_models_non_200_returns_empty(self):
        resp = MagicMock()
        resp.status_code = 503

        mock_client = MagicMock()
        mock_client.get.return_value = resp

        with patch("agent.local_models.httpx.Client", return_value=mock_client):
            models = list_models("http://localhost:11434", "ollama")

        assert models == []

    def test_list_models_llama_cpp(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": [{"id": "my-local-model"}]
        }

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get.return_value = resp

        with patch("agent.local_models.httpx.Client", return_value=mock_client):
            models = list_models("http://localhost:8080", "llama-cpp")

        assert len(models) == 1
        assert models[0].server_type == "llama-cpp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_open_map(path_to_content: dict):
    """Create an open() side-effect that serves different content per path."""
    def open_side_effect(path, *args, **kwargs):
        path_str = str(path)
        for key, content in path_to_content.items():
            if key in path_str:
                m = mock_open(read_data=content)()
                # mock_open supports iteration via __iter__
                m.__iter__ = lambda self: iter(content.splitlines(keepends=True))
                return m
        raise FileNotFoundError(path_str)
    return open_side_effect
