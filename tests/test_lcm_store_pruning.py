"""Tests for store pruning with max_store_size.

TDD Protocol:
- These tests are written FIRST (RED phase)
- Implementation must make them GREEN

Design constraints:
- The store is append-only; IDs must remain stable
- "Pruning" = marking entries as unavailable, not removing them from the list
- Referenced messages (in active DAG summaries) must never be pruned
- Pinned messages must never be pruned
"""
import pytest

from agent.lcm.config import LcmConfig
from agent.lcm.engine import LcmEngine
from agent.lcm.store import ImmutableStore
from agent.lcm.dag import SummaryDag


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_msg(i: int) -> dict:
    return {"role": "user", "content": f"message {i}"}


# ---------------------------------------------------------------------------
# 1. LcmConfig.max_store_size
# ---------------------------------------------------------------------------

class TestConfigMaxStoreSize:
    def test_config_has_max_store_size(self):
        """LcmConfig must have a max_store_size field with a sensible default."""
        config = LcmConfig()
        assert hasattr(config, "max_store_size")
        assert config.max_store_size >= 100  # sensible lower bound

    def test_config_max_store_size_default_is_1000(self):
        """Default should be 1000."""
        config = LcmConfig()
        assert config.max_store_size == 1000

    def test_config_max_store_size_custom(self):
        """Should accept custom value."""
        config = LcmConfig(max_store_size=50)
        assert config.max_store_size == 50


# ---------------------------------------------------------------------------
# 2. ImmutableStore.prune()
# ---------------------------------------------------------------------------

class TestStorePruneRemovesOldestUnreferenced:
    def test_store_prune_removes_oldest_unreferenced(self):
        """Add 10 messages, keep 5-9, prune with max=7.
        Messages 0-2 (oldest unreferenced) should be pruned.
        """
        store = ImmutableStore()
        for i in range(10):
            store.append(make_msg(i))

        # Messages 5-9 are "referenced" (e.g., by summaries)
        keep_ids = {5, 6, 7, 8, 9}
        store.prune(keep_ids=keep_ids, max_size=7)

        # Active (non-pruned) count should be <= max_size
        assert store.active_count <= 7

        # Messages 0, 1, 2 should be pruned (oldest unreferenced)
        assert store.get(0) is None
        assert store.get(1) is None
        assert store.get(2) is None

    def test_store_prune_preserves_referenced_messages(self):
        """Messages referenced by keep_ids must NOT be pruned even if old."""
        store = ImmutableStore()
        for i in range(10):
            store.append(make_msg(i))

        keep_ids = {0, 1, 2}  # old messages that are referenced
        store.prune(keep_ids=keep_ids, max_size=5)

        # Referenced messages must survive
        assert store.get(0) is not None
        assert store.get(1) is not None
        assert store.get(2) is not None

    def test_store_prune_preserves_tail(self):
        """The most recent messages should be last to go."""
        store = ImmutableStore()
        for i in range(10):
            store.append(make_msg(i))

        store.prune(keep_ids=set(), max_size=5)

        # Active count should be within limit
        assert store.active_count <= 5

        # Tail messages (most recent) should still be present
        assert store.get(9) is not None
        assert store.get(8) is not None


class TestStoreGetReturnNoneForPruned:
    def test_store_get_returns_none_for_pruned(self):
        """After pruning, store.get(pruned_id) returns None."""
        store = ImmutableStore()
        for i in range(10):
            store.append(make_msg(i))

        # Prune without keeping any
        store.prune(keep_ids=set(), max_size=3)

        # Pruned IDs return None
        pruned = store.pruned_ids
        assert len(pruned) > 0
        for pid in pruned:
            assert store.get(pid) is None

    def test_store_get_many_skips_pruned(self):
        """get_many should skip pruned entries."""
        store = ImmutableStore()
        for i in range(5):
            store.append(make_msg(i))

        store.prune(keep_ids=set(), max_size=3)

        pruned = store.pruned_ids
        # Requesting pruned IDs from get_many should return nothing for them
        pairs = store.get_many(list(pruned))
        assert len(pairs) == 0


class TestStorePrunePreservesPinned:
    def test_store_prune_preserves_pinned_messages(self):
        """Pinned messages must NOT be pruned regardless of age."""
        store = ImmutableStore()
        for i in range(10):
            store.append(make_msg(i))

        pinned = {0, 1}
        keep_ids = pinned  # only pinned are referenced
        store.prune(keep_ids=keep_ids, max_size=5)

        # Pinned messages survive
        assert store.get(0) is not None
        assert store.get(1) is not None


class TestStorePruneNoOp:
    def test_store_prune_no_op_when_under_limit(self):
        """When store is under max_size, prune is a no-op."""
        store = ImmutableStore()
        for i in range(5):
            store.append(make_msg(i))

        store.prune(keep_ids=set(), max_size=10)

        # Nothing should be pruned
        assert len(store.pruned_ids) == 0
        assert store.active_count == 5

    def test_store_prune_with_empty_dag(self):
        """When keep_ids is empty, oldest messages are pruned first."""
        store = ImmutableStore()
        for i in range(10):
            store.append(make_msg(i))

        store.prune(keep_ids=set(), max_size=5)

        assert store.active_count == 5
        # Oldest should be pruned
        assert store.get(0) is None
        assert store.get(1) is None
        assert store.get(2) is None
        assert store.get(3) is None
        assert store.get(4) is None


class TestStorePruneRespectsMaxSize:
    def test_store_prune_respects_max_store_size(self):
        """Active count after prune must be <= max_size."""
        store = ImmutableStore()
        for i in range(20):
            store.append(make_msg(i))

        store.prune(keep_ids=set(), max_size=7)

        assert store.active_count <= 7

    def test_store_active_count_property(self):
        """active_count returns non-pruned message count."""
        store = ImmutableStore()
        for i in range(10):
            store.append(make_msg(i))

        assert store.active_count == 10

        store.prune(keep_ids=set(), max_size=5)
        assert store.active_count <= 5

    def test_store_pruned_ids_property(self):
        """pruned_ids returns a set of all pruned MessageIds."""
        store = ImmutableStore()
        for i in range(5):
            store.append(make_msg(i))

        store.prune(keep_ids=set(), max_size=3)

        pruned = store.pruned_ids
        assert isinstance(pruned, (set, frozenset))
        # Should have pruned 2 messages (5 - 3 = 2)
        assert len(pruned) == 2


# ---------------------------------------------------------------------------
# 3. Engine: expand gracefully handles pruned messages
# ---------------------------------------------------------------------------

class TestExpandReturnsPrunedMarker:
    def test_expand_returns_pruned_message_for_dropped(self):
        """Expanding a summary whose source messages were pruned should return
        a placeholder '[Message pruned from store]' instead of the original.
        """
        config = LcmConfig(protect_last_n=0)
        engine = LcmEngine(config)

        for i in range(4):
            engine.ingest(make_msg(i))

        # Compact messages 0-3 into a summary
        node = engine.compact("summary of 0-3", level=1, block_start=0, block_end=4)

        # Manually prune message 0 from the store
        engine.store.prune(keep_ids={1, 2, 3}, max_size=3)

        # Expanding source of the summary should handle pruned gracefully
        pairs = engine.expand_summary(node.id)

        # We expect 4 results, but msg 0 is pruned
        assert len(pairs) == 4
        # The pruned message should have a placeholder
        pruned_pair = next((p for p in pairs if p[0] == 0), None)
        assert pruned_pair is not None
        assert "[Message pruned from store]" in str(pruned_pair[1].get("content", ""))

    def test_expand_non_pruned_returns_original(self):
        """Non-pruned messages in expand should return original content."""
        config = LcmConfig(protect_last_n=0)
        engine = LcmEngine(config)

        for i in range(3):
            engine.ingest(make_msg(i))

        node = engine.compact("summary", level=1, block_start=0, block_end=3)

        # Don't prune anything
        pairs = engine.expand_summary(node.id)

        assert len(pairs) == 3
        for mid, msg in pairs:
            assert "[Message pruned from store]" not in str(msg.get("content", ""))
            assert f"message {mid}" in msg.get("content", "")


# ---------------------------------------------------------------------------
# 4. Engine auto-pruning on ingest
# ---------------------------------------------------------------------------

class TestEngineAutoPrunesOnIngest:
    def test_engine_auto_prunes_on_ingest(self):
        """After ingesting messages that push store over max_store_size,
        pruning happens automatically.
        """
        config = LcmConfig(max_store_size=5)
        engine = LcmEngine(config)

        # Ingest 8 messages — should trigger auto-prune
        for i in range(8):
            engine.ingest(make_msg(i))

        # The store's active count must be within the limit
        assert engine.store.active_count <= config.max_store_size

    def test_engine_auto_prune_preserves_pinned(self):
        """Auto-pruning must not prune pinned messages."""
        config = LcmConfig(max_store_size=5)
        engine = LcmEngine(config)

        for i in range(3):
            engine.ingest(make_msg(i))

        # Pin message 0
        engine.pin([0])

        # Ingest more to trigger auto-prune
        for i in range(3, 8):
            engine.ingest(make_msg(i))

        # Pinned message 0 must survive
        assert engine.store.get(0) is not None

    def test_engine_auto_prune_preserves_referenced(self):
        """Auto-pruning must not prune messages referenced by active DAG summaries."""
        config = LcmConfig(max_store_size=5, protect_last_n=0)
        engine = LcmEngine(config)

        for i in range(4):
            engine.ingest(make_msg(i))

        # Compact first 4 into a summary — messages 0-3 are now referenced
        engine.compact("summary", level=1, block_start=0, block_end=4)

        # Ingest more to trigger auto-prune
        for i in range(4, 8):
            engine.ingest(make_msg(i))

        # Messages 0-3 are referenced by the active summary — must survive
        assert engine.store.get(0) is not None
        assert engine.store.get(1) is not None
        assert engine.store.get(2) is not None
        assert engine.store.get(3) is not None

    def test_engine_no_auto_prune_when_under_limit(self):
        """When store is under max_store_size, no pruning should happen."""
        config = LcmConfig(max_store_size=100)
        engine = LcmEngine(config)

        for i in range(5):
            engine.ingest(make_msg(i))

        # Nothing should be pruned
        assert len(engine.store.pruned_ids) == 0
        assert engine.store.active_count == 5
