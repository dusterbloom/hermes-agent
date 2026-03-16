"""Tests for local model fallback in auxiliary_client.py.

Covers:
- Local vision fallback when OPENAI_BASE_URL is set and model supports vision
- Local vision fallback skipped when model doesn't support vision
- Local vision fallback skipped when OPENAI_BASE_URL is not set
- AUXILIARY_VISION_PROVIDER=local explicit override
- Local text fallback in auto mode
- Local text fallback with OPENAI_MODEL env var
- Forced "local" provider in _resolve_forced_provider
"""
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Stub out modules that may not be installed in test environment
for mod_name in ("dotenv",):
    if mod_name not in sys.modules:
        fake = types.ModuleType(mod_name)
        fake.load_dotenv = lambda *a, **k: None
        sys.modules[mod_name] = fake

# ---------------------------------------------------------------------------
# Ensure the real agent.auxiliary_client is loaded before each test.
#
# Several test files (test_local_web_extract.py, test_searxng.py) inject a
# lightweight stub for "agent.auxiliary_client" into sys.modules so they can
# load tools/web_tools.py without side-effects.  The stub only has a small
# set of attributes (get_async_text_auxiliary_client, etc.) and lacks OpenAI,
# get_vision_auxiliary_client, and get_text_auxiliary_client.
#
# The tricky part: test_local_web_extract.py also does:
#   sys.modules["agent"].auxiliary_client = _aux_mod
# which overwrites the attribute on the real "agent" package even though
# sys.modules["agent.auxiliary_client"] still holds the real module.  Python's
# "from agent import auxiliary_client" checks the package attribute FIRST, so
# it returns the stub.
#
# Fix: an autouse fixture that runs before every test in this file restores
# both sys.modules["agent.auxiliary_client"] and the agent package attribute
# to the real module.
# ---------------------------------------------------------------------------

def _load_real_auxiliary_client():
    """Load the real agent.auxiliary_client and return it.

    Evicts any stub that may be installed in sys.modules, then imports the
    real module.  Also repairs agent.<attribute> so that
    ``from agent import auxiliary_client`` returns the real module.
    """
    # Evict stub if present
    current = sys.modules.get("agent.auxiliary_client")
    if current is not None and not hasattr(current, "get_vision_auxiliary_client"):
        del sys.modules["agent.auxiliary_client"]

    # Evict stub agent package if it was injected (has no __file__)
    agent_mod = sys.modules.get("agent")
    if agent_mod is not None and not hasattr(agent_mod, "__file__"):
        del sys.modules["agent"]

    # Load the real module
    import agent.auxiliary_client as _real  # noqa: F401

    real = sys.modules["agent.auxiliary_client"]

    # Repair the package attribute so "from agent import auxiliary_client"
    # doesn't resolve to a cached stub attribute on the agent package.
    agent_pkg = sys.modules.get("agent")
    if agent_pkg is not None:
        agent_pkg.auxiliary_client = real

    # Guarantee OpenAI is present (defensive; it's in the real module's imports)
    if not hasattr(real, "OpenAI"):
        from openai import OpenAI
        real.OpenAI = OpenAI

    return real


@pytest.fixture(autouse=True)
def _real_auxiliary_client():
    """Ensure the real agent.auxiliary_client is in place for every test."""
    _load_real_auxiliary_client()
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_local_envs(monkeypatch):
    """Clear env vars relevant to local model routing."""
    for key in (
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "AUXILIARY_VISION_PROVIDER",
        "AUXILIARY_VISION_MODEL",
        "AUXILIARY_TEXT_MODEL",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_fake_openai_client():
    """Return a minimal mock that looks like an OpenAI client."""
    client = MagicMock()
    client.api_key = "test-key"
    client.base_url = "http://localhost:11434/v1"
    return client


# ---------------------------------------------------------------------------
# Vision local fallback — auto mode
# ---------------------------------------------------------------------------


class TestVisionLocalFallbackAuto:
    """_try_local_vision() is reached when no cloud provider is configured."""

    def test_local_vision_used_when_model_supports_vision(self, monkeypatch):
        """If OPENAI_BASE_URL is set and model supports vision, return client."""
        _clear_local_envs(monkeypatch)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENAI_MODEL", "llava")

        from agent import auxiliary_client
        # Patch OpenAI constructor to avoid real HTTP
        fake_client = _make_fake_openai_client()
        with patch("agent.auxiliary_client.OpenAI", return_value=fake_client):
            client, model = auxiliary_client.get_vision_auxiliary_client()

        assert client is not None
        assert model == "llava"

    def test_local_vision_skipped_when_model_lacks_vision(self, monkeypatch):
        """If the local model does not support vision, return (None, None)."""
        _clear_local_envs(monkeypatch)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENAI_MODEL", "deepseek-r1")  # no vision

        from agent import auxiliary_client
        fake_client = _make_fake_openai_client()
        with patch("agent.auxiliary_client.OpenAI", return_value=fake_client):
            client, model = auxiliary_client.get_vision_auxiliary_client()

        assert client is None
        assert model is None

    def test_local_vision_skipped_when_no_base_url(self, monkeypatch):
        """Without OPENAI_BASE_URL no local client is created."""
        _clear_local_envs(monkeypatch)
        monkeypatch.setenv("OPENAI_MODEL", "llava")
        # No OPENAI_BASE_URL

        from agent import auxiliary_client
        with patch("agent.auxiliary_client.OpenAI") as mock_oi:
            client, model = auxiliary_client.get_vision_auxiliary_client()

        # OpenAI should not have been called for local vision (cloud providers
        # also won't fire because no keys are set)
        assert client is None

    def test_local_vision_skipped_when_no_model(self, monkeypatch):
        """Without OPENAI_MODEL no local vision client is created."""
        _clear_local_envs(monkeypatch)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        # No OPENAI_MODEL

        from agent import auxiliary_client
        fake_client = _make_fake_openai_client()
        with patch("agent.auxiliary_client.OpenAI", return_value=fake_client):
            client, model = auxiliary_client.get_vision_auxiliary_client()

        assert client is None

    def test_local_vision_uses_local_as_default_api_key(self, monkeypatch):
        """When OPENAI_API_KEY is absent, 'local' is used as the api_key."""
        _clear_local_envs(monkeypatch)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENAI_MODEL", "llava")
        # No OPENAI_API_KEY

        from agent import auxiliary_client
        fake_client = _make_fake_openai_client()
        with patch("agent.auxiliary_client.OpenAI", return_value=fake_client) as mock_oi:
            client, model = auxiliary_client.get_vision_auxiliary_client()

        # Verify OpenAI was called with api_key="local"
        call_kwargs = mock_oi.call_args
        assert call_kwargs is not None
        # api_key can be positional or keyword; check kwargs or args
        bound = call_kwargs[1] if call_kwargs[1] else {}
        assert bound.get("api_key") == "local"


# ---------------------------------------------------------------------------
# Vision explicit override: AUXILIARY_VISION_PROVIDER=local
# ---------------------------------------------------------------------------


class TestVisionLocalProviderOverride:
    """AUXILIARY_VISION_PROVIDER=local uses the local endpoint unconditionally."""

    def test_explicit_local_provider_returns_client(self, monkeypatch):
        _clear_local_envs(monkeypatch)
        monkeypatch.setenv("AUXILIARY_VISION_PROVIDER", "local")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENAI_MODEL", "deepseek-r1")  # no vision check for forced

        from agent import auxiliary_client
        fake_client = _make_fake_openai_client()
        with patch("agent.auxiliary_client.OpenAI", return_value=fake_client):
            client, model = auxiliary_client.get_vision_auxiliary_client()

        assert client is not None
        assert model == "deepseek-r1"

    def test_explicit_local_provider_no_base_url_returns_none(self, monkeypatch):
        _clear_local_envs(monkeypatch)
        monkeypatch.setenv("AUXILIARY_VISION_PROVIDER", "local")
        monkeypatch.setenv("OPENAI_MODEL", "llava")
        # No OPENAI_BASE_URL

        from agent import auxiliary_client
        with patch("agent.auxiliary_client.OpenAI") as mock_oi:
            client, model = auxiliary_client.get_vision_auxiliary_client()

        assert client is None

    def test_explicit_local_uses_auxiliary_vision_model(self, monkeypatch):
        """AUXILIARY_VISION_MODEL takes precedence over OPENAI_MODEL for forced local."""
        _clear_local_envs(monkeypatch)
        monkeypatch.setenv("AUXILIARY_VISION_PROVIDER", "local")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENAI_MODEL", "llava")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "qwen-vl:7b")

        from agent import auxiliary_client
        fake_client = _make_fake_openai_client()
        with patch("agent.auxiliary_client.OpenAI", return_value=fake_client):
            client, model = auxiliary_client.get_vision_auxiliary_client()

        # The model returned should be from AUXILIARY_VISION_MODEL
        assert model == "qwen-vl:7b"


# ---------------------------------------------------------------------------
# Text local fallback — auto mode
# ---------------------------------------------------------------------------


class TestTextLocalFallbackAuto:
    """Local text fallback activates when cloud providers are absent."""

    def test_local_text_used_when_base_url_and_model_set(self, monkeypatch):
        """If OPENAI_BASE_URL + OPENAI_MODEL but no OPENAI_API_KEY, local fallback fires."""
        _clear_local_envs(monkeypatch)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENAI_MODEL", "deepseek-v3")
        # No OPENAI_API_KEY — _try_custom_endpoint() returns None without it
        # but the new local fallback should still work

        from agent import auxiliary_client
        fake_client = _make_fake_openai_client()
        with patch("agent.auxiliary_client.OpenAI", return_value=fake_client):
            client, model = auxiliary_client.get_text_auxiliary_client()

        assert client is not None
        assert model == "deepseek-v3"

    def test_local_text_skipped_when_no_model(self, monkeypatch):
        """Without OPENAI_MODEL the local text fallback returns None."""
        _clear_local_envs(monkeypatch)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        # No OPENAI_MODEL

        from agent import auxiliary_client
        fake_client = _make_fake_openai_client()
        with patch("agent.auxiliary_client.OpenAI", return_value=fake_client):
            client, model = auxiliary_client.get_text_auxiliary_client()

        assert client is None

    def test_local_text_skipped_when_no_base_url(self, monkeypatch):
        """Without OPENAI_BASE_URL the local text fallback returns None."""
        _clear_local_envs(monkeypatch)
        monkeypatch.setenv("OPENAI_MODEL", "deepseek-v3")
        # No OPENAI_BASE_URL

        from agent import auxiliary_client
        with patch("agent.auxiliary_client.OpenAI") as mock_oi:
            client, model = auxiliary_client.get_text_auxiliary_client()

        assert client is None

    def test_local_text_uses_local_as_default_api_key(self, monkeypatch):
        """When OPENAI_API_KEY is absent, 'local' is used for local text fallback."""
        _clear_local_envs(monkeypatch)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENAI_MODEL", "deepseek-v3")

        from agent import auxiliary_client
        fake_client = _make_fake_openai_client()
        with patch("agent.auxiliary_client.OpenAI", return_value=fake_client) as mock_oi:
            client, model = auxiliary_client.get_text_auxiliary_client()

        call_kwargs = mock_oi.call_args
        assert call_kwargs is not None
        bound = call_kwargs[1] if call_kwargs[1] else {}
        assert bound.get("api_key") == "local"

    def test_cloud_provider_takes_priority_over_local_text(self, monkeypatch):
        """When OpenRouter is available it takes priority over local fallback."""
        _clear_local_envs(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-123")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENAI_MODEL", "deepseek-v3")

        from agent import auxiliary_client
        from hermes_constants import OPENROUTER_BASE_URL

        fake_client = _make_fake_openai_client()
        fake_client.base_url = OPENROUTER_BASE_URL
        with patch("agent.auxiliary_client.OpenAI", return_value=fake_client):
            client, model = auxiliary_client.get_text_auxiliary_client()

        # Model should be the OpenRouter default, not deepseek-v3
        from agent.auxiliary_client import _OPENROUTER_MODEL
        assert model == _OPENROUTER_MODEL
