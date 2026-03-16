"""Local model manager for Ollama, LM Studio, vLLM, and llama.cpp servers.

Provides a unified interface for discovering, listing, loading, and managing
models across different local inference servers.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Default ports for auto-detection
SERVER_PORTS = {
    "ollama": 11434,
    "lm-studio": 1234,
    "vllm": 8000,
    "llama-cpp": 8080,
}


@dataclass
class LocalModel:
    """Represents a model available on a local server."""
    name: str
    size_bytes: Optional[int] = None
    quantization: Optional[str] = None
    loaded: bool = False
    server_type: str = ""


@dataclass
class ServerStatus:
    """Status of a local inference server."""
    server_type: str          # ollama | lm-studio | vllm | llama-cpp
    url: str                  # base URL
    running: bool
    models_loaded: list[str] = field(default_factory=list)


def detect_running_servers(timeout: float = 2.0) -> list[ServerStatus]:
    """Probe known ports to detect which local servers are running.

    Tries each server type's default port and checks for a valid response.
    Also checks for WSL2 host IP if running under WSL.
    """
    servers = []
    hosts = ["127.0.0.1"]

    # Check for WSL2 - add Windows host IP
    wsl_host = _detect_wsl_host_ip()
    if wsl_host:
        hosts.append(wsl_host)

    for host in hosts:
        for server_type, port in SERVER_PORTS.items():
            url = f"http://{host}:{port}"
            status = _probe_server(url, server_type, timeout)
            if status and status.running:
                servers.append(status)

    return servers


def detect_best_server(timeout: float = 2.0) -> Optional[ServerStatus]:
    """Detect the best available local server.

    Priority: Ollama > LM Studio > vLLM > llama.cpp
    (Ollama first because it has the best model management)
    """
    servers = detect_running_servers(timeout)
    if not servers:
        return None

    priority = {"ollama": 0, "lm-studio": 1, "vllm": 2, "llama-cpp": 3}
    servers.sort(key=lambda s: priority.get(s.server_type, 99))
    return servers[0]


def _probe_server(url: str, server_type: str, timeout: float) -> Optional[ServerStatus]:
    """Probe a single server URL to check if it's running."""
    try:
        client = httpx.Client(timeout=timeout)

        if server_type == "ollama":
            # Ollama has a simple root endpoint
            resp = client.get(f"{url}")
            if resp.status_code == 200:
                # Get loaded models
                models_resp = client.get(f"{url}/api/tags")
                loaded = []
                if models_resp.status_code == 200:
                    data = models_resp.json()
                    loaded = [m["name"] for m in data.get("models", [])]
                client.close()
                return ServerStatus(
                    server_type="ollama",
                    url=url,
                    running=True,
                    models_loaded=loaded,
                )

        elif server_type in ("lm-studio", "vllm", "llama-cpp"):
            # All use OpenAI-compatible /v1/models
            resp = client.get(f"{url}/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                loaded = [m["id"] for m in data.get("data", [])]
                client.close()
                return ServerStatus(
                    server_type=server_type,
                    url=url,
                    running=True,
                    models_loaded=loaded,
                )

        client.close()
    except (httpx.ConnectError, httpx.TimeoutException, Exception) as e:
        logger.debug(f"Server probe failed for {url} ({server_type}): {e}")

    return None


def list_models(server_url: str, server_type: str) -> list[LocalModel]:
    """List models available on a local server.

    Args:
        server_url: Base URL of the server (e.g. "http://localhost:11434")
        server_type: Type of server ("ollama", "lm-studio", "vllm", "llama-cpp")
    """
    try:
        client = httpx.Client(timeout=10.0)

        if server_type == "ollama":
            resp = client.get(f"{server_url}/api/tags")
            client.close()
            if resp.status_code == 200:
                data = resp.json()
                return [
                    LocalModel(
                        name=m["name"],
                        size_bytes=m.get("size"),
                        loaded=True,  # Ollama only lists loaded models here
                        server_type="ollama",
                    )
                    for m in data.get("models", [])
                ]

        elif server_type == "lm-studio":
            resp = client.get(f"{server_url}/v1/models")
            client.close()
            if resp.status_code == 200:
                data = resp.json()
                return [
                    LocalModel(
                        name=m["id"],
                        loaded=True,
                        server_type="lm-studio",
                    )
                    for m in data.get("data", [])
                ]

        elif server_type in ("vllm", "llama-cpp"):
            resp = client.get(f"{server_url}/v1/models")
            client.close()
            if resp.status_code == 200:
                data = resp.json()
                return [
                    LocalModel(
                        name=m["id"],
                        loaded=True,
                        server_type=server_type,
                    )
                    for m in data.get("data", [])
                ]

        client.close()
    except Exception as e:
        logger.error(f"Failed to list models from {server_url}: {e}")

    return []


def load_model(server_url: str, server_type: str, model_name: str,
               context_length: Optional[int] = None) -> bool:
    """Load a model on the local server.

    Args:
        server_url: Base URL of the server
        server_type: Type of server
        model_name: Name/path of model to load
        context_length: Optional context length override

    Returns:
        True if model was loaded successfully
    """
    try:
        client = httpx.Client(timeout=120.0)  # loading can be slow

        if server_type == "ollama":
            # Ollama loads models on demand; we can pre-warm with a generate call
            payload = {"model": model_name, "prompt": "", "keep_alive": "10m"}
            resp = client.post(f"{server_url}/api/generate", json=payload)
            client.close()
            return resp.status_code == 200

        elif server_type == "lm-studio":
            # LM Studio REST API for loading
            payload: dict = {"model": model_name}
            if context_length:
                payload["context_length"] = context_length
            resp = client.post(f"{server_url}/api/v1/models/load", json=payload)
            client.close()
            return resp.status_code == 200

        elif server_type == "vllm":
            # vLLM doesn't support dynamic model loading — model is set at server start
            logger.warning(
                "vLLM doesn't support dynamic model loading. "
                "Restart server with the desired model."
            )
            client.close()
            return False

        elif server_type == "llama-cpp":
            # llama.cpp server doesn't support dynamic model loading
            logger.warning(
                "llama.cpp server doesn't support dynamic model loading. "
                "Restart server with the desired model."
            )
            client.close()
            return False

        client.close()
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")

    return False


def unload_model(server_url: str, server_type: str, model_name: str) -> bool:
    """Unload a model from the local server."""
    try:
        client = httpx.Client(timeout=30.0)

        if server_type == "ollama":
            payload = {"model": model_name, "keep_alive": 0}
            resp = client.post(f"{server_url}/api/generate", json=payload)
            client.close()
            return resp.status_code == 200

        elif server_type == "lm-studio":
            payload = {"model": model_name}
            resp = client.post(f"{server_url}/api/v1/models/unload", json=payload)
            client.close()
            return resp.status_code == 200

        client.close()
    except Exception as e:
        logger.error(f"Failed to unload model {model_name}: {e}")

    return False


def pull_model(server_url: str, server_type: str, model_name: str,
               progress_callback=None) -> bool:
    """Pull/download a model. Currently only supported by Ollama.

    Args:
        server_url: Base URL
        server_type: Server type (only "ollama" supported)
        model_name: Model name to pull (e.g. "qwen2.5-coder:7b")
        progress_callback: Optional callable(status: str, completed: int, total: int)
    """
    if server_type != "ollama":
        logger.warning(f"Model pulling not supported for {server_type}")
        return False

    try:
        client = httpx.Client(timeout=600.0)  # downloads can be very slow

        with client.stream(
            "POST",
            f"{server_url}/api/pull",
            json={"name": model_name, "stream": True},
        ) as resp:
            if resp.status_code != 200:
                return False

            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    status = data.get("status", "")
                    completed = data.get("completed", 0)
                    total = data.get("total", 0)

                    if progress_callback:
                        progress_callback(status, completed, total)

                    if data.get("error"):
                        logger.error(f"Pull error: {data['error']}")
                        return False
                except json.JSONDecodeError:
                    continue

        client.close()
        return True

    except Exception as e:
        logger.error(f"Failed to pull model {model_name}: {e}")
        return False


def show_model_info(server_url: str, server_type: str, model_name: str) -> Optional[dict]:
    """Get detailed information about a model. Currently only Ollama."""
    if server_type != "ollama":
        return None

    try:
        client = httpx.Client(timeout=10.0)
        resp = client.post(f"{server_url}/api/show", json={"name": model_name})
        client.close()
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.debug(f"Failed to get model info: {e}")

    return None


def resolve_model_path(name: str, models_dir: str = "~/models") -> Optional[str]:
    """Resolve a model name to a file path.

    Checks:
    1. Absolute path
    2. Relative to models_dir
    3. Bare name (search in models_dir for matching .gguf files)
    """
    # Absolute path
    if os.path.isabs(name) and os.path.exists(name):
        return name

    # Relative to models_dir
    expanded = os.path.expanduser(models_dir)
    full_path = os.path.join(expanded, name)
    if os.path.exists(full_path):
        return full_path

    # With .gguf extension
    if not name.endswith(".gguf"):
        full_path_gguf = full_path + ".gguf"
        if os.path.exists(full_path_gguf):
            return full_path_gguf

    # Search for matching files
    models_path = Path(expanded)
    if models_path.exists():
        name_lower = name.lower()
        for f in models_path.glob("*.gguf"):
            if name_lower in f.stem.lower():
                return str(f)

    return None


def get_openai_base_url(server_url: str, server_type: str) -> str:
    """Get the OpenAI-compatible base URL for a server.

    Some servers need /v1 appended, others already include it.
    """
    if server_type == "ollama":
        return f"{server_url}/v1"
    else:
        # LM Studio, vLLM, llama.cpp all serve at /v1
        base = server_url.rstrip("/")
        if not base.endswith("/v1"):
            return f"{base}/v1"
        return base


def _detect_wsl_host_ip() -> Optional[str]:
    """Detect Windows host IP when running under WSL2.

    Reads /etc/resolv.conf nameserver — in WSL2, this points to the Windows host.
    """
    try:
        with open("/proc/version", "r") as f:
            version = f.read().lower()
        if "microsoft" not in version and "wsl" not in version:
            return None

        with open("/etc/resolv.conf", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("nameserver"):
                    parts = line.split()
                    if len(parts) >= 2:
                        ip = parts[1]
                        # Validate it's not a loopback
                        if not ip.startswith("127."):
                            return ip
    except (FileNotFoundError, PermissionError):
        pass

    return None
