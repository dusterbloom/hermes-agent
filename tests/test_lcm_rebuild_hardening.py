"""Hardening tests for rebuild_from_session edge cases.

TDD protocol: these tests were written FIRST (RED), then implementation
was updated to make them pass (GREEN).

Each test covers a specific failure mode that should degrade gracefully
rather than crash.
"""
import pytest
from agent.lcm.engine import LcmEngine
from agent.lcm.config import LcmConfig


@pytest.fixture
def config():
    return LcmConfig(enabled=True)


class TestRebuildHardening:
    """Edge-case hardening for rebuild_from_session."""

    def test_rebuild_validates_pinned_ids_exist_in_store(self, config):
        """Pinned IDs that don't exist in the restored store should be silently dropped."""
        session_data = {
            "messages": [{"role": "user", "content": "hello"}],
            "lcm": {
                "summaries": [],
                "original_messages": [{"role": "user", "content": "hello"}],
                "pinned": [0, 99, 200],  # 99 and 200 don't exist
            },
        }
        engine = LcmEngine.rebuild_from_session(session_data, config)
        # Only ID 0 is valid (store size == 1)
        assert engine._pinned_ids == {0}
        assert 99 not in engine._pinned_ids
        assert 200 not in engine._pinned_ids

    def test_rebuild_with_missing_original_messages_key(self, config):
        """Session data without original_messages key should rebuild from active messages."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        session_data = {
            "messages": messages,
            "lcm": {
                "summaries": [],
                # no original_messages key at all
            },
        }
        engine = LcmEngine.rebuild_from_session(session_data, config)
        assert len(engine.active) == 2
        assert all(e.kind == "raw" for e in engine.active)
        assert len(engine.store) == 2

    def test_rebuild_with_empty_original_messages(self, config):
        """original_messages: [] should not crash."""
        messages = [
            {"role": "user", "content": "msg1"},
        ]
        session_data = {
            "messages": messages,
            "lcm": {
                "summaries": [],
                "original_messages": [],
            },
        }
        # Should not raise
        engine = LcmEngine.rebuild_from_session(session_data, config)
        # With no original_messages, falls through to raw-append path
        assert len(engine.active) == 1

    def test_rebuild_with_corrupt_summary_node(self, config):
        """DAG node referencing source IDs that don't exist in store should be kept
        but expand_summary should return placeholders rather than crashing."""
        session_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "[Summary]",
                    "_lcm_summary": True,
                    "_lcm_node_id": 0,
                },
            ],
            "lcm": {
                "summaries": [
                    {
                        "node_id": 0,
                        "source_ids": [999, 1000, 1001],  # none exist in store
                        "text": "Corrupt summary",
                        "level": 1,
                        "tokens": 30,
                    }
                ],
                "original_messages": [],  # empty store
            },
        }
        # Should not crash during rebuild
        engine = LcmEngine.rebuild_from_session(session_data, config)
        assert len(engine.dag) == 1
        # expand_summary should return placeholders, not raise
        expanded = engine.expand_summary(0)
        assert len(expanded) == 3
        for _mid, msg in expanded:
            assert msg.get("_lcm_pruned") is True

    def test_rebuild_with_missing_lcm_meta_key(self, config):
        """Session data without lcm key should create a fresh engine state."""
        session_data = {
            "messages": [
                {"role": "user", "content": "legacy message"},
            ]
            # no "lcm" key at all
        }
        engine = LcmEngine.rebuild_from_session(session_data, config)
        # Should still load messages
        assert len(engine.active) == 1
        assert engine.active[0].kind == "raw"
        assert len(engine.dag) == 0
        assert engine._pinned_ids == set()
        assert engine._last_summary is None

    def test_rebuild_with_extra_unknown_keys(self, config):
        """Extra unknown keys in lcm metadata should be silently ignored."""
        session_data = {
            "messages": [{"role": "user", "content": "hello"}],
            "lcm": {
                "summaries": [],
                "original_messages": [{"role": "user", "content": "hello"}],
                "future_feature_flag": True,
                "some_new_index": {"a": 1, "b": 2},
                "version": "99.0.0",
            },
        }
        # Must not raise
        engine = LcmEngine.rebuild_from_session(session_data, config)
        assert len(engine.active) == 1

    def test_rebuild_with_pinned_ids_exceeding_store_size(self, config):
        """Pinned IDs beyond the store's range should be silently dropped."""
        original_messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        session_data = {
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ],
            "lcm": {
                "summaries": [],
                "original_messages": original_messages,
                "pinned": [0, 1, 2, 3, 100],  # only 0 and 1 are valid
            },
        }
        engine = LcmEngine.rebuild_from_session(session_data, config)
        assert engine._pinned_ids == {0, 1}

    def test_rebuild_preserves_valid_state_through_corruption(self, config):
        """When some data is corrupt, valid data should still be restored correctly."""
        original_messages = [
            {"role": "user", "content": "good msg 0"},
            {"role": "assistant", "content": "good msg 1"},
        ]
        session_data = {
            "messages": [
                # One valid raw message
                {"role": "user", "content": "good msg 0"},
                # One summary entry referencing an out-of-range node ID
                {
                    "role": "user",
                    "content": "[Summary]",
                    "_lcm_summary": True,
                    "_lcm_node_id": 999,  # no DAG node with this ID
                },
            ],
            "lcm": {
                "summaries": [],  # no nodes defined — node 999 is unknown
                "original_messages": original_messages,
                "pinned": [0, 50],  # 50 is invalid
            },
        }
        # Must not crash
        engine = LcmEngine.rebuild_from_session(session_data, config)
        # Valid raw message should be present
        assert any(e.kind == "raw" for e in engine.active)
        # Valid pin should survive
        assert 0 in engine._pinned_ids
        # Invalid pin should be dropped
        assert 50 not in engine._pinned_ids

    def test_rebuild_with_none_last_summary(self, config):
        """last_summary: None should be handled without error."""
        session_data = {
            "messages": [{"role": "user", "content": "hi"}],
            "lcm": {
                "summaries": [],
                "original_messages": [{"role": "user", "content": "hi"}],
                "last_summary": None,
            },
        }
        engine = LcmEngine.rebuild_from_session(session_data, config)
        assert engine._last_summary is None

    def test_rebuild_with_dag_referencing_pruned_messages(self, config):
        """After rebuild, if the store has been pruned, DAG nodes referencing
        pruned IDs should return placeholders via expand_summary."""
        original_messages = [
            {"role": "user", "content": "msg 0"},
            {"role": "assistant", "content": "msg 1"},
            {"role": "user", "content": "msg 2"},
        ]
        session_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "[Summary]",
                    "_lcm_summary": True,
                    "_lcm_node_id": 0,
                },
                {"role": "user", "content": "recent"},
            ],
            "lcm": {
                "summaries": [
                    {
                        "node_id": 0,
                        "source_ids": [0, 1, 2],
                        "text": "Earlier chat",
                        "level": 1,
                        "tokens": 30,
                    }
                ],
                "original_messages": original_messages,
            },
        }
        engine = LcmEngine.rebuild_from_session(session_data, config)
        # Manually prune some messages from the store
        engine.store.prune(keep_ids=set(), max_size=0)

        # expand_summary must still return results (placeholders for pruned)
        expanded = engine.expand_summary(0)
        assert len(expanded) == 3
        # All should be placeholders since we pruned everything
        for _mid, msg in expanded:
            assert msg.get("_lcm_pruned") is True
