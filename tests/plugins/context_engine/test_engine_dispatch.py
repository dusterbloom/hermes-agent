"""Regression tests for run_agent.py context-engine dispatch logic.

Before the fix, the dispatch condition was:
    if _engine_name == "compressor" or _engine_name not in ("lcm", "rlm"):
        pass  # fall through to built-in compressor — BUG

Any engine name that was not "lcm" or "rlm" silently fell into the no-op
branch.  E.g. context.engine: "custom" would never call load_context_engine.

After the fix the condition is simply:
    if _engine_name == "compressor":
        pass

So any other name reaches the generic else branch that calls
load_context_engine(_engine_name, ...).
"""

import json
from unittest.mock import MagicMock, patch


class TestEngineDispatch:
    """Verify that custom engine names reach load_context_engine()."""

    def _run_dispatch(self, engine_name: str, rlm_enabled: bool = False):
        """Execute just the context-engine dispatch block from run_agent.py.

        Replicated from run_agent.py lines ~1907-1966 so the logic can be
        unit-tested without spinning up a full AIAgent.
        """
        _ctx_cfg = {"engine": engine_name, "rlm": rlm_enabled, "rlm_config": {}}
        _engine_name = _ctx_cfg.get("engine", "compressor") or "compressor"
        _rlm_enabled = bool(_ctx_cfg.get("rlm", False))
        _selected_engine = None

        if _engine_name == "compressor":
            pass
        elif _engine_name == "rlm":
            try:
                from plugins.context_engine import load_context_engine  # noqa: F401
                _selected_engine = load_context_engine(
                    "rlm", rlm_config=_ctx_cfg.get("rlm_config", {})
                )
            except Exception:
                pass
        else:
            rlm_kwargs = {"rlm_config": _ctx_cfg.get("rlm_config", {})}
            if _rlm_enabled:
                try:
                    from plugins.context_engine import load_composite_engine  # noqa: F401
                    _selected_engine = load_composite_engine(
                        compression_name=_engine_name,
                        rlm_enabled=True,
                        **rlm_kwargs,
                    )
                except Exception:
                    _selected_engine = None
            if _selected_engine is None:
                try:
                    from plugins.context_engine import load_context_engine  # noqa: F401
                    _selected_engine = load_context_engine(
                        _engine_name,
                        rlm_config=rlm_kwargs.get("rlm_config"),
                    )
                except Exception:
                    pass

        return _selected_engine

    def test_compressor_name_uses_builtin(self):
        """engine=compressor must never call load_context_engine."""
        mock_engine = MagicMock(name="mock_engine")
        with patch("plugins.context_engine.load_context_engine", return_value=mock_engine) as mock_load:
            result = self._run_dispatch("compressor")
        mock_load.assert_not_called()
        assert result is None

    def test_custom_engine_name_calls_load_context_engine(self):
        """engine=custom must call load_context_engine('custom', ...)."""
        mock_engine = MagicMock(name="mock_custom_engine")
        with patch("plugins.context_engine.load_context_engine", return_value=mock_engine) as mock_load:
            result = self._run_dispatch("custom")
        mock_load.assert_called_once()
        call_args = mock_load.call_args
        assert call_args[0][0] == "custom", (
            f"load_context_engine called with engine name {call_args[0][0]!r}, expected 'custom'"
        )
        assert result is mock_engine

    def test_lcm_engine_still_calls_load_context_engine(self):
        """engine=lcm must also reach load_context_engine (not broken by fix)."""
        mock_engine = MagicMock(name="mock_lcm_engine")
        with patch("plugins.context_engine.load_context_engine", return_value=mock_engine) as mock_load:
            result = self._run_dispatch("lcm")
        mock_load.assert_called_once()
        assert mock_load.call_args[0][0] == "lcm"
        assert result is mock_engine

    def test_rlm_standalone_calls_load_context_engine_with_rlm(self):
        """engine=rlm must call load_context_engine('rlm', ...)."""
        mock_engine = MagicMock(name="mock_rlm_engine")
        with patch("plugins.context_engine.load_context_engine", return_value=mock_engine) as mock_load:
            result = self._run_dispatch("rlm")
        mock_load.assert_called_once()
        assert mock_load.call_args[0][0] == "rlm"
        assert result is mock_engine

    def test_custom_engine_with_rlm_tries_composite_first(self):
        """engine=custom with rlm=True must try load_composite_engine first."""
        mock_composite = MagicMock(name="mock_composite")
        with patch("plugins.context_engine.load_composite_engine", return_value=mock_composite) as mock_composite_load, \
             patch("plugins.context_engine.load_context_engine") as mock_plain_load:
            result = self._run_dispatch("custom", rlm_enabled=True)
        mock_composite_load.assert_called_once()
        assert mock_composite_load.call_args[1]["compression_name"] == "custom"
        mock_plain_load.assert_not_called()  # composite succeeded, no fallback needed
        assert result is mock_composite
