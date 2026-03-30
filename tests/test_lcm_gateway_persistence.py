"""TDD tests for LCM metadata persistence in the gateway session layer.

RED phase: these tests are written first and should FAIL until the
implementation is added to gateway/session.py (SessionEntry + SessionStore).
"""
import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from gateway.session import SessionEntry
from gateway.config import Platform
from agent.lcm.engine import LcmEngine
from agent.lcm.config import LcmConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(**kwargs) -> SessionEntry:
    """Build a minimal SessionEntry for testing."""
    defaults = dict(
        session_key="test:key",
        session_id="20260101_000000_abcd1234",
        created_at=datetime(2026, 1, 1, 0, 0, 0),
        updated_at=datetime(2026, 1, 1, 0, 0, 0),
    )
    defaults.update(kwargs)
    return SessionEntry(**defaults)


def _make_lcm_metadata() -> dict:
    """Return a realistic lcm_metadata dict as produced by LcmEngine.to_session_metadata()."""
    return {
        "summaries": [
            {
                "node_id": 0,
                "source_ids": [0, 1],
                "child_summaries": [],
                "text": "User asked about the weather; agent replied it is sunny.",
                "level": 1,
                "tokens": 42,
            }
        ],
        "original_messages": [
            {"role": "user", "content": "What is the weather?"},
            {"role": "assistant", "content": "It is sunny today."},
            {"role": "user", "content": "Thanks!"},
        ],
        "store_size": 3,
        "pinned": [2],
        "last_summary": "User asked about the weather; agent replied it is sunny.",
    }


# ---------------------------------------------------------------------------
# 1. SessionEntry has an lcm_metadata field
# ---------------------------------------------------------------------------

class TestSessionEntryLcmMetadataField:
    def test_session_entry_stores_lcm_metadata(self):
        """SessionEntry should accept lcm_metadata as an optional dict field."""
        entry = _make_entry(lcm_metadata={"summaries": [], "pinned": []})
        assert entry.lcm_metadata is not None
        assert isinstance(entry.lcm_metadata, dict)

    def test_session_entry_lcm_metadata_defaults_to_none(self):
        """SessionEntry without lcm_metadata should have lcm_metadata=None."""
        entry = _make_entry()
        assert entry.lcm_metadata is None


# ---------------------------------------------------------------------------
# 2. Serialization: to_dict() includes 'lcm' key when lcm_metadata is set
# ---------------------------------------------------------------------------

class TestSessionEntrySerialisation:
    def test_session_save_includes_lcm_metadata(self):
        """to_dict() should include an 'lcm' key when lcm_metadata is present."""
        meta = _make_lcm_metadata()
        entry = _make_entry(lcm_metadata=meta)

        d = entry.to_dict()

        assert "lcm" in d
        assert d["lcm"] == meta

    def test_session_save_omits_lcm_key_when_none(self):
        """to_dict() should NOT include 'lcm' when lcm_metadata is None."""
        entry = _make_entry()
        d = entry.to_dict()
        assert "lcm" not in d

    def test_session_save_roundtrip_json(self):
        """Serialized entry should survive a JSON round-trip."""
        meta = _make_lcm_metadata()
        entry = _make_entry(lcm_metadata=meta)

        raw = json.dumps(entry.to_dict())
        loaded = json.loads(raw)

        assert "lcm" in loaded
        assert loaded["lcm"]["store_size"] == 3
        assert loaded["lcm"]["pinned"] == [2]


# ---------------------------------------------------------------------------
# 3. Deserialization: from_dict() restores lcm_metadata
# ---------------------------------------------------------------------------

class TestSessionEntryDeserialisation:
    def test_session_load_restores_lcm_metadata(self):
        """from_dict() should populate lcm_metadata when 'lcm' key is present."""
        meta = _make_lcm_metadata()
        data = {
            "session_key": "test:key",
            "session_id": "20260101_000000_abcd1234",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "lcm": meta,
        }

        entry = SessionEntry.from_dict(data)

        assert entry.lcm_metadata is not None
        assert entry.lcm_metadata["store_size"] == 3
        assert entry.lcm_metadata["pinned"] == [2]

    def test_session_without_lcm_metadata_loads_cleanly(self):
        """Loading a session without 'lcm' key should give lcm_metadata=None (backward compat)."""
        data = {
            "session_key": "test:key",
            "session_id": "20260101_000000_abcd1234",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            # no 'lcm' key
        }

        entry = SessionEntry.from_dict(data)

        assert entry.lcm_metadata is None


# ---------------------------------------------------------------------------
# 4. Full save/load roundtrip through JSON file (simulates SessionStore._save)
# ---------------------------------------------------------------------------

class TestSessionSaveLoadRoundtrip:
    def test_session_save_load_roundtrip(self):
        """Save a session with lcm_metadata, reload from JSON, verify metadata matches."""
        meta = _make_lcm_metadata()
        entry = _make_entry(lcm_metadata=meta)

        # Simulate what SessionStore._save does
        data = {"test:key": entry.to_dict()}
        serialized = json.dumps(data, indent=2)

        # Simulate what SessionStore._ensure_loaded_locked does
        parsed = json.loads(serialized)
        restored = SessionEntry.from_dict(parsed["test:key"])

        assert restored.lcm_metadata is not None
        assert restored.lcm_metadata["store_size"] == meta["store_size"]
        assert restored.lcm_metadata["pinned"] == meta["pinned"]
        assert restored.lcm_metadata["last_summary"] == meta["last_summary"]
        assert len(restored.lcm_metadata["summaries"]) == len(meta["summaries"])
        assert len(restored.lcm_metadata["original_messages"]) == len(meta["original_messages"])

    def test_session_save_load_roundtrip_no_lcm(self):
        """Old sessions without lcm survive the roundtrip unchanged."""
        entry = _make_entry()

        data = {"test:key": entry.to_dict()}
        serialized = json.dumps(data, indent=2)

        parsed = json.loads(serialized)
        restored = SessionEntry.from_dict(parsed["test:key"])

        assert restored.lcm_metadata is None


# ---------------------------------------------------------------------------
# 5. LcmEngine.to_session_metadata() returns expected structure
# ---------------------------------------------------------------------------

class TestLcmEngineToSessionMetadataFormat:
    def test_lcm_engine_to_session_metadata_format(self):
        """to_session_metadata() should return a dict with all expected keys."""
        config = LcmConfig()
        engine = LcmEngine(config)

        for i in range(4):
            engine.ingest({"role": "user", "content": f"message {i}"})

        engine.pin([0])
        metadata = engine.to_session_metadata()

        assert isinstance(metadata, dict)
        # Required top-level keys
        assert "summaries" in metadata
        assert "original_messages" in metadata
        assert "store_size" in metadata
        assert "pinned" in metadata
        assert "last_summary" in metadata

        # Values make sense
        assert metadata["store_size"] == 4
        assert len(metadata["original_messages"]) == 4
        assert metadata["pinned"] == [0]
        assert metadata["summaries"] == []  # no compaction yet
        assert metadata["last_summary"] is None  # no compaction yet

    def test_lcm_engine_to_session_metadata_after_compaction(self):
        """After compaction, summaries list should be populated."""
        config = LcmConfig()
        engine = LcmEngine(config)

        for i in range(4):
            engine.ingest({"role": "user", "content": f"message {i}"})

        # Direct compact (no async summarizer needed)
        engine.compact("Summary text", level=1, block_start=0, block_end=2)
        metadata = engine.to_session_metadata()

        assert len(metadata["summaries"]) == 1
        assert metadata["summaries"][0]["text"] == "Summary text"
        assert metadata["summaries"][0]["source_ids"] == [0, 1]


# ---------------------------------------------------------------------------
# 6. Full engine rebuild roundtrip via session metadata
# ---------------------------------------------------------------------------

class TestLcmEngineRebuildFromSessionRoundtrip:
    def test_lcm_engine_rebuild_from_session_roundtrip(self):
        """Create engine, compact, serialize, rebuild — verify state matches."""
        config = LcmConfig()
        engine = LcmEngine(config)

        messages_in = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am fine"},
        ]
        for msg in messages_in:
            engine.ingest(msg)

        engine.pin([0])
        engine.compact("Summary of first two messages", level=1, block_start=0, block_end=2)

        metadata = engine.to_session_metadata()
        active_messages = engine.active_messages()

        # Rebuild into a new engine instance
        engine2 = LcmEngine.rebuild_from_session(
            {"messages": active_messages, "lcm": metadata},
            config,
        )

        # Store should have all 4 original messages
        assert len(engine2.store) == 4

        # Active should have: 1 summary + 2 raw = 3
        assert len(engine2.active) == 3
        assert engine2.active[0].kind == "summary"
        assert engine2.active[1].kind == "raw"
        assert engine2.active[2].kind == "raw"

        # Pinned IDs should be restored
        assert 0 in engine2._pinned_ids

        # DAG should have the same number of nodes
        assert len(engine2.dag.nodes) == len(engine.dag.nodes)

    def test_lcm_engine_rebuild_preserves_messages(self):
        """Rebuilt engine should preserve the actual message content."""
        config = LcmConfig()
        engine = LcmEngine(config)

        engine.ingest({"role": "user", "content": "original message"})

        metadata = engine.to_session_metadata()
        active = engine.active_messages()

        engine2 = LcmEngine.rebuild_from_session(
            {"messages": active, "lcm": metadata},
            config,
        )

        msgs = engine2.active_messages()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "original message"

    def test_lcm_engine_rebuild_roundtrip_no_compaction(self):
        """Rebuild works even when there was no compaction (empty summaries)."""
        config = LcmConfig()
        engine = LcmEngine(config)

        for i in range(3):
            engine.ingest({"role": "user", "content": f"msg {i}"})

        metadata = engine.to_session_metadata()
        active = engine.active_messages()

        engine2 = LcmEngine.rebuild_from_session(
            {"messages": active, "lcm": metadata},
            config,
        )

        assert len(engine2.store) == 3
        assert len(engine2.active) == 3
        assert all(e.kind == "raw" for e in engine2.active)
