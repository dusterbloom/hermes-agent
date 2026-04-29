"""Regression test: config loaded in on_session_start must reach the live LcmEngine.

Fix: after rebuilding self._config in on_session_start, propagate it to
self._engine.config so compaction reads the user-supplied thresholds, not the
constructor-time defaults.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from plugins.context_engine.lcm.__init__ import LcmContextEngine
from plugins.context_engine.lcm.config import LcmConfig


class TestConfigPropagationOnSessionStart:
    """on_session_start must write the reloaded config back to the live engine."""

    def test_tau_soft_propagated_to_engine(self):
        """engine._engine.config.tau_soft reflects the value from config.yaml."""
        default_config = LcmConfig()
        assert default_config.tau_soft != 0.42, "pick a non-default value for this test"

        engine = LcmContextEngine()

        fake_cfg = {"lcm": {"tau_soft": "0.42"}}
        with patch(
            "hermes_cli.config.load_config",
            return_value=fake_cfg,
        ):
            engine.on_session_start("test-session-cfg1")

        assert engine._engine.config.tau_soft == pytest.approx(0.42), (
            f"engine._engine.config.tau_soft={engine._engine.config.tau_soft!r} "
            "but expected 0.42 — config was not propagated to the live engine"
        )

    def test_tau_hard_propagated_to_engine(self):
        """engine._engine.config.tau_hard reflects the value from config.yaml."""
        engine = LcmContextEngine()
        fake_cfg = {"lcm": {"tau_hard": "0.95"}}
        with patch(
            "hermes_cli.config.load_config",
            return_value=fake_cfg,
        ):
            engine.on_session_start("test-session-cfg2")

        assert engine._engine.config.tau_hard == pytest.approx(0.95), (
            f"engine._engine.config.tau_hard={engine._engine.config.tau_hard!r} "
            "but expected 0.95"
        )

    def test_protect_last_n_propagated_to_engine(self):
        """engine._engine.config.protect_last_n reflects config.yaml."""
        engine = LcmContextEngine()
        fake_cfg = {"lcm": {"protect_last_n": "10"}}
        with patch(
            "hermes_cli.config.load_config",
            return_value=fake_cfg,
        ):
            engine.on_session_start("test-session-cfg3")

        assert engine._engine.config.protect_last_n == 10, (
            f"engine._engine.config.protect_last_n={engine._engine.config.protect_last_n!r} "
            "but expected 10"
        )

    def test_wrapper_attributes_also_updated(self):
        """threshold_percent and protect_last_n on the wrapper also sync."""
        engine = LcmContextEngine()
        fake_cfg = {"lcm": {"tau_soft": "0.60", "protect_last_n": "8"}}
        with patch(
            "hermes_cli.config.load_config",
            return_value=fake_cfg,
        ):
            engine.on_session_start("test-session-cfg4")

        assert engine.threshold_percent == pytest.approx(0.60)
        assert engine.protect_last_n == 8

    def test_no_config_does_not_crash(self):
        """If load_config raises, on_session_start must not crash."""
        engine = LcmContextEngine()
        with patch(
            "hermes_cli.config.load_config",
            side_effect=RuntimeError("no config"),
        ):
            engine.on_session_start("test-session-no-cfg")
        # Engine should still be usable
        assert engine._engine is not None
