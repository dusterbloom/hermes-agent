"""Unit tests for LcmEngine v2 features."""
import pytest
from unittest.mock import MagicMock, patch

from agent.lcm.engine import LcmEngine, CompactionAction, ContextEntry
from agent.lcm.config import LcmConfig
from agent.lcm.store import ImmutableStore
from agent.lcm.dag import SummaryDag


class TestLcmEngineInit:
    def test_init_with_model_info(self):
        config = LcmConfig()
        engine = LcmEngine(
            config,
            model="claude-sonnet-4",
            provider="anthropic",
            context_length=200_000,
        )
        assert engine.model == "claude-sonnet-4"
        assert engine.provider == "anthropic"
        assert engine.context_length == 200_000

    def test_init_components(self):
        config = LcmConfig()
        engine = LcmEngine(config)
        assert engine.store is not None
        assert engine.dag is not None
        assert engine.token_estimator is not None
        assert engine.summarizer is not None

    def test_context_length_property(self):
        config = LcmConfig()
        engine = LcmEngine(config, context_length=100_000)
        assert engine.context_length == 100_000

        engine.context_length = 150_000
        assert engine.context_length == 150_000


class TestLcmEngineIngest:
    def test_ingest_adds_to_store_and_active(self):
        config = LcmConfig()
        engine = LcmEngine(config)

        msg_id = engine.ingest({"role": "user", "content": "hello"})
        assert msg_id == 0
        assert len(engine.store) == 1
        assert len(engine.active) == 1
        assert engine.active[0].kind == "raw"

    def test_ingest_multiple(self):
        config = LcmConfig()
        engine = LcmEngine(config)

        for i in range(5):
            engine.ingest({"role": "user", "content": f"msg {i}"})

        assert len(engine.store) == 5
        assert len(engine.active) == 5


class TestLcmEngineTokenEstimation:
    def test_active_tokens(self):
        config = LcmConfig()
        engine = LcmEngine(config)

        engine.ingest({"role": "user", "content": "hello world"})
        tokens = engine.active_tokens()
        assert tokens > 0

    def test_active_token_breakdown(self):
        config = LcmConfig()
        engine = LcmEngine(config)

        # Add raw messages
        for i in range(3):
            engine.ingest({"role": "user", "content": f"message {i}"})

        breakdown = engine.active_token_breakdown()
        assert breakdown["total"] > 0
        assert breakdown["raw"] == breakdown["total"]
        assert breakdown["summary"] == 0
        assert breakdown["raw_count"] == 3
        assert breakdown["summary_count"] == 0


class TestLcmEngineThresholds:
    def test_check_thresholds_none(self):
        config = LcmConfig(tau_soft=0.5, tau_hard=0.85)
        engine = LcmEngine(config, context_length=1000)

        engine.ingest({"role": "user", "content": "short"})
        action = engine.check_thresholds()
        assert action == CompactionAction.NONE

    def test_check_thresholds_async(self):
        config = LcmConfig(tau_soft=0.5, tau_hard=0.85)
        engine = LcmEngine(config, context_length=100)

        # Fill past soft threshold (50%)
        engine.ingest({"role": "user", "content": "x" * 250})  # ~62 tokens
        action = engine.check_thresholds()
        assert action == CompactionAction.ASYNC

    def test_check_thresholds_blocking(self):
        config = LcmConfig(tau_soft=0.5, tau_hard=0.85)
        engine = LcmEngine(config, context_length=100)

        # Fill past hard threshold (85%)
        engine.ingest({"role": "user", "content": "x" * 400})  # ~100 tokens
        action = engine.check_thresholds()
        assert action == CompactionAction.BLOCKING


class TestLcmEngineCompact:
    def test_compact_basic(self):
        config = LcmConfig(protect_last_n=2)
        engine = LcmEngine(config)

        for i in range(6):
            engine.ingest({"role": "user", "content": f"msg {i}"})

        # Compact first 4 messages
        node = engine.compact("test summary", level=1, block_start=0, block_end=4)

        assert node is not None
        assert node.id == 0
        assert len(node.source_ids) == 4
        assert len(engine.active) == 3  # 1 summary + 2 protected
        assert len(engine.store) == 6  # All original messages preserved

    def test_compact_block_with_mock_summarizer(self):
        config = LcmConfig(protect_last_n=2)
        engine = LcmEngine(config)

        for i in range(6):
            engine.ingest({"role": "user", "content": f"msg {i}"})

        # Mock summarizer
        engine.summarizer.summarize = lambda turns, previous_summary=None: "## Goal\nTest goal"

        node = engine.compact_block(0, 4)

        assert node is not None
        assert "Test goal" in node.text
        assert engine._last_summary is not None

    def test_auto_compact(self):
        config = LcmConfig(protect_last_n=2)
        engine = LcmEngine(config)

        for i in range(6):
            engine.ingest({"role": "user", "content": f"msg {i}"})

        # Mock summarizer
        engine.summarizer.summarize = lambda turns, previous_summary=None: "Summary"

        node = engine.auto_compact()

        assert node is not None
        assert len(engine.active) < 6


class TestLcmEngineExpand:
    def test_expand(self):
        config = LcmConfig()
        engine = LcmEngine(config)

        engine.ingest({"role": "user", "content": "msg 1"})
        engine.ingest({"role": "user", "content": "msg 2"})

        pairs = engine.expand([0, 1])
        assert len(pairs) == 2
        assert pairs[0][1]["content"] == "msg 1"
        assert pairs[1][1]["content"] == "msg 2"

    def test_expand_summary(self):
        config = LcmConfig(protect_last_n=0)
        engine = LcmEngine(config)

        # Ingest and compact
        for i in range(4):
            engine.ingest({"role": "user", "content": f"msg {i}"})

        node = engine.compact("summary", level=1, block_start=0, block_end=4)

        # Expand should recover all 4 original messages
        pairs = engine.expand_summary(node.id)
        assert len(pairs) == 4

    def test_focus_summary(self):
        config = LcmConfig(protect_last_n=0)
        engine = LcmEngine(config)

        for i in range(4):
            engine.ingest({"role": "user", "content": f"msg {i}"})

        node = engine.compact("summary", level=1, block_start=0, block_end=4)

        # Focus should expand the summary back into active
        result = engine.focus_summary(node.id)
        assert result is True
        assert len(engine.active) == 4
        assert all(e.kind == "raw" for e in engine.active)

    def test_focus_summary_not_found(self):
        config = LcmConfig()
        engine = LcmEngine(config)

        result = engine.focus_summary(999)
        assert result is False


class TestLcmEngineSearch:
    def test_keyword_search(self):
        config = LcmConfig(semantic_search=False)
        engine = LcmEngine(config)

        engine.ingest({"role": "user", "content": "hello world"})
        engine.ingest({"role": "user", "content": "foo bar"})
        engine.ingest({"role": "user", "content": "hello again"})

        results = engine.search("hello", limit=5)
        assert len(results) == 2

    def test_search_no_results(self):
        config = LcmConfig(semantic_search=False)
        engine = LcmEngine(config)

        engine.ingest({"role": "user", "content": "hello world"})

        results = engine.search("xyz", limit=5)
        assert len(results) == 0


class TestLcmEnginePinning:
    def test_pin(self):
        config = LcmConfig(max_pinned=10)
        engine = LcmEngine(config)

        engine.ingest({"role": "user", "content": "msg 1"})
        engine.ingest({"role": "user", "content": "msg 2"})

        count = engine.pin([0])
        assert count == 1
        assert 0 in engine._pinned_ids

    def test_pin_respects_limit(self):
        config = LcmConfig(max_pinned=2)
        engine = LcmEngine(config)

        for i in range(5):
            engine.ingest({"role": "user", "content": f"msg {i}"})

        count = engine.pin([0, 1, 2, 3, 4])
        assert count == 2  # Only 2 can be pinned
        assert len(engine._pinned_ids) == 2

    def test_unpin(self):
        config = LcmConfig()
        engine = LcmEngine(config)

        engine.ingest({"role": "user", "content": "msg"})
        engine.pin([0])

        count = engine.unpin([0])
        assert count == 1
        assert 0 not in engine._pinned_ids

    def test_pinned_not_compacted(self):
        config = LcmConfig(protect_last_n=0)
        engine = LcmEngine(config)

        for i in range(4):
            engine.ingest({"role": "user", "content": f"msg {i}"})

        # Pin message 0
        engine.pin([0])

        # Find compactable block should skip pinned
        block = engine.find_compactable_block()
        assert block is not None
        assert block[0] == 1  # Starts after pinned entry


class TestLcmEngineSessionPersistence:
    def test_to_session_metadata(self):
        config = LcmConfig()
        engine = LcmEngine(config)

        for i in range(3):
            engine.ingest({"role": "user", "content": f"msg {i}"})

        engine.pin([0])
        metadata = engine.to_session_metadata()

        assert len(metadata["original_messages"]) == 3
        assert metadata["pinned"] == [0]
        assert metadata["store_size"] == 3

    def test_rebuild_from_session(self):
        config = LcmConfig()

        # Create and populate engine
        engine = LcmEngine(config)
        for i in range(4):
            engine.ingest({"role": "user", "content": f"msg {i}"})
        engine.pin([0])
        engine.compact("summary", level=1, block_start=0, block_end=2)

        # Serialize
        metadata = engine.to_session_metadata()
        messages = engine.active_messages()

        # Rebuild
        engine2 = LcmEngine.rebuild_from_session(
            {"messages": messages, "lcm": metadata},
            config,
        )

        assert len(engine2.store) == 4
        assert len(engine2.active) == 3  # 1 summary + 2 raw
        assert 0 in engine2._pinned_ids

    def test_rebuild_validates_pinned_ids(self):
        config = LcmConfig()

        # Create session data with invalid pinned IDs
        session_data = {
            "messages": [
                {"role": "user", "content": "msg 1"},
            ],
            "lcm": {
                "original_messages": [{"role": "user", "content": "msg 1"}],
                "pinned": [0, 99],  # 99 doesn't exist
            },
        }

        engine = LcmEngine.rebuild_from_session(session_data, config)
        assert 0 in engine._pinned_ids
        assert 99 not in engine._pinned_ids  # Invalid ID removed


class TestLcmEngineFormatting:
    def test_format_expanded(self):
        config = LcmConfig()
        engine = LcmEngine(config)

        engine.ingest({"role": "user", "content": "hello"})
        result = engine.format_expanded([0])

        assert "[msg 0]" in result
        assert "user" in result
        assert "hello" in result

    def test_format_toc(self):
        config = LcmConfig()
        engine = LcmEngine(config)

        engine.ingest({"role": "user", "content": "hello"})
        result = engine.format_toc()

        assert "Conversation Timeline" in result
        assert "msg 0" in result

    def test_format_budget(self):
        config = LcmConfig()
        engine = LcmEngine(config, context_length=1000)

        engine.ingest({"role": "user", "content": "hello"})
        result = engine.format_budget()

        assert "LCM Context Budget" in result
        assert "1,000" in result
        assert "Active entries" in result


class TestLcmEngineReset:
    def test_reset_clears_all_state(self):
        config = LcmConfig()
        engine = LcmEngine(config)

        for i in range(3):
            engine.ingest({"role": "user", "content": f"msg {i}"})
        engine.pin([0])
        engine.compact("summary", level=1, block_start=0, block_end=2)

        engine.reset()

        assert len(engine.store) == 0
        assert len(engine.active) == 0
        assert len(engine.dag.nodes) == 0
        assert len(engine._pinned_ids) == 0
        assert engine._last_summary is None
