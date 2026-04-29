"""Regression tests for LCM session resume / _ingested_count correctness.

Fix A: _ingested_count must track input-message count, not active-entry count.
Fix B: raw message ids must be preserved across save/rebuild when active list
       contains non-contiguous raw entries (e.g. after lcm_focus).
"""
from __future__ import annotations

import pytest

from plugins.context_engine.lcm.engine import LcmEngine, ContextEntry
from plugins.context_engine.lcm.config import LcmConfig
from plugins.context_engine.lcm.__init__ import LcmContextEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_messages(n: int) -> list[dict]:
    """Return n simple alternating user/assistant messages."""
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"msg {i}"})
    return msgs


def _build_session_data(engine: LcmEngine) -> dict:
    """Build a minimal session_data dict from a live engine (mimics on_session_end).

    Uses active_messages_for_session() so raw entries carry _lcm_raw_id, exactly
    as the real persistence layer does.
    """
    lcm_meta = engine.to_session_metadata()
    # Use annotated messages so _lcm_raw_id is preserved across rebuild
    active_msgs = engine.active_messages_for_session()
    return {"messages": active_msgs, "lcm": lcm_meta}


# ---------------------------------------------------------------------------
# Fix A — _ingested_count across session rotation
# ---------------------------------------------------------------------------

class TestIngestedCountAfterRebuild:
    """_ingested_count should equal the number of *input* messages that were
    originally fed into the engine, not the active-list length."""

    def test_rebuild_sets_ingested_count_to_original_message_count(self):
        """After rebuild from disk, _ingested_count == len(original_messages)."""
        config = LcmConfig(enabled=True, protect_last_n=2)
        outer = LcmContextEngine(lcm_config=config)

        messages = _make_messages(100)
        outer.compress(messages)
        # Manually run a compaction to shrink the active list so active != 100
        block = outer._engine.find_compactable_block()
        if block:
            outer._engine.compact("summary text", level=1, block_start=block[0], block_end=block[1])

        active_after_compact = len(outer._engine.active)
        assert active_after_compact < 100, "Need compaction to have fired for this test to be meaningful"

        # Build session_data as on_session_end would
        session_data = _build_session_data(outer._engine)

        # Create a new engine instance and call on_session_start with the persisted data
        import json, tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = "test-session-abc"
            session_file = pathlib.Path(tmpdir) / "sessions" / f"session_{session_id}.json"
            session_file.parent.mkdir(parents=True)
            session_file.write_text(json.dumps(session_data), encoding="utf-8")

            new_outer = LcmContextEngine(lcm_config=config)
            new_outer.on_session_start(session_id, hermes_home=tmpdir)

        # _ingested_count must equal len(original_messages) saved in lcm metadata
        expected = len(session_data["lcm"]["original_messages"])
        assert new_outer._ingested_count == expected, (
            f"_ingested_count={new_outer._ingested_count} != "
            f"len(original_messages)={expected} — cursor wrongly rebased on active list"
        )

    def test_no_saved_file_ingested_count_is_zero(self):
        """Fresh session_start with no save file resets _ingested_count to 0."""
        import tempfile
        config = LcmConfig(enabled=True)
        outer = LcmContextEngine(lcm_config=config)
        # Simulate already having ingested some messages
        outer._ingested_count = 50

        with tempfile.TemporaryDirectory() as tmpdir:
            outer.on_session_start("nonexistent-session", hermes_home=tmpdir)

        assert outer._ingested_count == 0, (
            f"_ingested_count={outer._ingested_count} should be 0 for fresh session"
        )

    def test_no_new_messages_ingested_after_rebuild(self):
        """After rebuild, calling compress() with the same 100 messages must not
        ingest any new messages (store size must remain unchanged)."""
        config = LcmConfig(enabled=True, protect_last_n=2)
        outer = LcmContextEngine(lcm_config=config)

        messages = _make_messages(100)
        outer.compress(messages)
        block = outer._engine.find_compactable_block()
        if block:
            outer._engine.compact("summary text", level=1, block_start=block[0], block_end=block[1])

        session_data = _build_session_data(outer._engine)
        store_size_before_rebuild = len(outer._engine.store)

        import json, tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = "test-no-reingest"
            session_file = pathlib.Path(tmpdir) / "sessions" / f"session_{session_id}.json"
            session_file.parent.mkdir(parents=True)
            session_file.write_text(json.dumps(session_data), encoding="utf-8")

            new_outer = LcmContextEngine(lcm_config=config)
            new_outer.on_session_start(session_id, hermes_home=tmpdir)

        # Calling compress with the same 100 messages must not add anything to store
        new_outer.compress(messages)
        store_size_after = len(new_outer._engine.store)

        assert store_size_after == store_size_before_rebuild, (
            f"store grew from {store_size_before_rebuild} to {store_size_after} — "
            "messages were re-ingested after rebuild"
        )

    def test_new_messages_after_rebuild_are_ingested(self):
        """After rebuild with 100 messages, compress with 110 ingests exactly 10 new."""
        config = LcmConfig(enabled=True, protect_last_n=2)
        outer = LcmContextEngine(lcm_config=config)

        messages = _make_messages(100)
        outer.compress(messages)
        block = outer._engine.find_compactable_block()
        if block:
            outer._engine.compact("summary text", level=1, block_start=block[0], block_end=block[1])

        session_data = _build_session_data(outer._engine)
        store_size_at_save = len(outer._engine.store)

        import json, tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = "test-new-msgs"
            session_file = pathlib.Path(tmpdir) / "sessions" / f"session_{session_id}.json"
            session_file.parent.mkdir(parents=True)
            session_file.write_text(json.dumps(session_data), encoding="utf-8")

            new_outer = LcmContextEngine(lcm_config=config)
            new_outer.on_session_start(session_id, hermes_home=tmpdir)

        messages_110 = _make_messages(110)
        new_outer.compress(messages_110)

        store_size_after = len(new_outer._engine.store)
        assert store_size_after == store_size_at_save + 10, (
            f"Expected {store_size_at_save + 10} store entries, got {store_size_after}"
        )


# ---------------------------------------------------------------------------
# Fix B — preserve raw message ids across save/rebuild after lcm_focus
# ---------------------------------------------------------------------------

