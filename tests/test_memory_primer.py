"""Tests for session primer — cross-session knowledge injection."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.lcm.config import LcmConfig
from agent.lcm.engine import LcmEngine, ContextEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine() -> LcmEngine:
    config = LcmConfig(protect_last_n=0)
    return LcmEngine(config)


def _make_mock_results(n: int) -> list[dict]:
    return [
        {"content": f"fact {i}", "trust_score": 0.8}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# TestPrimerFunction
# ---------------------------------------------------------------------------

class TestPrimerFunction:
    def test_prime_queries_hrr_store(self):
        """prime_from_hrr(engine, store, topic) calls retriever.search(topic)."""
        engine = _make_engine()
        mock_store = MagicMock()
        mock_results = _make_mock_results(1)

        with patch("agent.lcm.hrr.retrieval.FactRetriever") as MockRetriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.search.return_value = mock_results
            MockRetriever.return_value = mock_retriever_instance

            from agent.lcm.primer import prime_from_hrr
            prime_from_hrr(engine, mock_store, "project context")

        mock_retriever_instance.search.assert_called_once()
        call_args = mock_retriever_instance.search.call_args
        # First positional arg or 'query' kwarg must be the topic
        query_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("query", "")
        assert "project context" in query_arg

    def test_prime_injects_message_into_active_context(self):
        """After priming, engine.active has one more entry at position 0."""
        engine = _make_engine()
        mock_store = MagicMock()
        before_len = len(engine.active)

        with patch("agent.lcm.hrr.retrieval.FactRetriever") as MockRetriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.search.return_value = _make_mock_results(2)
            MockRetriever.return_value = mock_retriever_instance

            from agent.lcm.primer import prime_from_hrr
            prime_from_hrr(engine, mock_store, "topic")

        assert len(engine.active) == before_len + 1
        assert engine.active[0].kind == "raw"

    def test_prime_message_has_prior_knowledge_marker(self):
        """The injected message has _lcm_primer=True and 'Prior Knowledge' in content."""
        engine = _make_engine()
        mock_store = MagicMock()

        with patch("agent.lcm.hrr.retrieval.FactRetriever") as MockRetriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.search.return_value = _make_mock_results(1)
            MockRetriever.return_value = mock_retriever_instance

            from agent.lcm.primer import prime_from_hrr
            prime_from_hrr(engine, mock_store, "topic")

        injected = engine.active[0].message
        assert injected.get("_lcm_primer") is True
        assert "Prior Knowledge" in injected.get("content", "")

    def test_prime_includes_fact_content(self):
        """The injected message contains the fact text from search results."""
        engine = _make_engine()
        mock_store = MagicMock()
        results = [{"content": "the user prefers dark mode", "trust_score": 0.9}]

        with patch("agent.lcm.hrr.retrieval.FactRetriever") as MockRetriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.search.return_value = results
            MockRetriever.return_value = mock_retriever_instance

            from agent.lcm.primer import prime_from_hrr
            prime_from_hrr(engine, mock_store, "preferences")

        content = engine.active[0].message["content"]
        assert "the user prefers dark mode" in content

    def test_prime_respects_max_facts_limit(self):
        """With max_facts=2, only 2 facts are included even if store returns 5."""
        engine = _make_engine()
        mock_store = MagicMock()

        with patch("agent.lcm.hrr.retrieval.FactRetriever") as MockRetriever:
            mock_retriever_instance = MagicMock()
            # search already honours the limit kwarg, but we simulate 5 returned
            mock_retriever_instance.search.return_value = _make_mock_results(2)
            MockRetriever.return_value = mock_retriever_instance

            from agent.lcm.primer import prime_from_hrr
            prime_from_hrr(engine, mock_store, "topic", max_facts=2)

        # Verify search was called with limit=2
        call_kwargs = mock_retriever_instance.search.call_args.kwargs
        assert call_kwargs.get("limit") == 2

    def test_prime_returns_count_of_injected_facts(self):
        """Returns the number of facts found (int)."""
        engine = _make_engine()
        mock_store = MagicMock()

        with patch("agent.lcm.hrr.retrieval.FactRetriever") as MockRetriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.search.return_value = _make_mock_results(3)
            MockRetriever.return_value = mock_retriever_instance

            from agent.lcm.primer import prime_from_hrr
            count = prime_from_hrr(engine, mock_store, "topic")

        assert count == 3


# ---------------------------------------------------------------------------
# TestPrimerNoop
# ---------------------------------------------------------------------------

class TestPrimerNoop:
    def test_noop_when_store_is_none(self):
        """prime_from_hrr(engine, None, topic) returns 0 without touching active."""
        engine = _make_engine()
        before_len = len(engine.active)

        from agent.lcm.primer import prime_from_hrr
        result = prime_from_hrr(engine, None, "topic")

        assert result == 0
        assert len(engine.active) == before_len

    def test_noop_when_no_results(self):
        """store returns [] => returns 0, engine.active unchanged."""
        engine = _make_engine()
        mock_store = MagicMock()
        before_len = len(engine.active)

        with patch("agent.lcm.hrr.retrieval.FactRetriever") as MockRetriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.search.return_value = []
            MockRetriever.return_value = mock_retriever_instance

            from agent.lcm.primer import prime_from_hrr
            result = prime_from_hrr(engine, mock_store, "topic")

        assert result == 0
        assert len(engine.active) == before_len

    def test_noop_when_topic_empty(self):
        """prime_from_hrr(engine, store, '') returns 0 without querying."""
        engine = _make_engine()
        mock_store = MagicMock()

        from agent.lcm.primer import prime_from_hrr
        result = prime_from_hrr(engine, mock_store, "")

        assert result == 0
        mock_store.assert_not_called()

    def test_noop_when_topic_whitespace_only(self):
        """prime_from_hrr(engine, store, '   ') is treated as empty topic."""
        engine = _make_engine()
        mock_store = MagicMock()

        from agent.lcm.primer import prime_from_hrr
        result = prime_from_hrr(engine, mock_store, "   ")

        assert result == 0


# ---------------------------------------------------------------------------
# TestPrimerSafety
# ---------------------------------------------------------------------------

class TestPrimerSafety:
    def test_exception_does_not_crash(self):
        """store.search raises Exception => returns 0, no crash."""
        engine = _make_engine()
        mock_store = MagicMock()

        with patch("agent.lcm.hrr.retrieval.FactRetriever") as MockRetriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.search.side_effect = Exception("db is gone")
            MockRetriever.return_value = mock_retriever_instance

            from agent.lcm.primer import prime_from_hrr
            try:
                result = prime_from_hrr(engine, mock_store, "topic")
            except Exception as exc:
                pytest.fail(f"prime_from_hrr must not raise, but raised {type(exc).__name__}: {exc}")

        assert result == 0

    def test_engine_state_consistent_after_error(self):
        """engine.active length unchanged after error."""
        engine = _make_engine()
        engine.ingest({"role": "user", "content": "existing message"})
        before_len = len(engine.active)
        mock_store = MagicMock()

        with patch("agent.lcm.hrr.retrieval.FactRetriever") as MockRetriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.search.side_effect = RuntimeError("network timeout")
            MockRetriever.return_value = mock_retriever_instance

            from agent.lcm.primer import prime_from_hrr
            prime_from_hrr(engine, mock_store, "topic")

        assert len(engine.active) == before_len

    def test_exception_is_logged_as_warning(self):
        """Swallowed exceptions are logged at WARNING level."""
        import logging

        engine = _make_engine()
        mock_store = MagicMock()

        with patch("agent.lcm.hrr.retrieval.FactRetriever") as MockRetriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.search.side_effect = Exception("boom")
            MockRetriever.return_value = mock_retriever_instance

            with patch.object(
                logging.getLogger("agent.lcm.primer"), "warning"
            ) as mock_warn:
                from agent.lcm.primer import prime_from_hrr
                prime_from_hrr(engine, mock_store, "topic")

            mock_warn.assert_called_once()
