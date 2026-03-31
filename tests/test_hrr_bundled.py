"""Tests for HRR store bundled under agent.lcm.hrr (TDD RED phase).

These tests are written FIRST (RED phase) before the implementation.
They exercise imports and basic functionality from the new bundled location.
"""
import pytest


def _has_numpy():
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# 1. Import tests (RED: will fail until agent/lcm/hrr/ is created)
# ---------------------------------------------------------------------------


class TestHrrImports:
    def test_import_holographic_module(self):
        """from agent.lcm.hrr.holographic import encode_atom, bind, unbind, bundle, similarity, encode_text"""
        from agent.lcm.hrr.holographic import (  # noqa: F401
            encode_atom,
            bind,
            unbind,
            bundle,
            similarity,
            encode_text,
        )

    def test_import_store_class(self):
        """from agent.lcm.hrr.store import MemoryStore"""
        from agent.lcm.hrr.store import MemoryStore  # noqa: F401

    def test_import_retrieval_class(self):
        """from agent.lcm.hrr.retrieval import FactRetriever"""
        from agent.lcm.hrr.retrieval import FactRetriever  # noqa: F401

    def test_package_exports(self):
        """from agent.lcm.hrr import MemoryStore, FactRetriever"""
        from agent.lcm.hrr import MemoryStore, FactRetriever  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Store CRUD tests
# ---------------------------------------------------------------------------


class TestHrrStoreCrud:
    def test_add_and_search_fact(self, tmp_path):
        """MemoryStore can add a fact and search for it via FTS5."""
        from agent.lcm.hrr.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "test.db")
        fact_id = store.add_fact("Python uses indentation for blocks", category="programming")
        assert fact_id > 0
        results = store.search_facts("indentation")
        assert len(results) >= 1
        assert any("indentation" in r["content"] for r in results)
        store.close()

    def test_fact_deduplication(self, tmp_path):
        """Adding the same content twice returns the existing fact_id."""
        from agent.lcm.hrr.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "test.db")
        id1 = store.add_fact("The sky is blue")
        id2 = store.add_fact("The sky is blue")
        assert id1 == id2
        store.close()

    def test_entity_extraction(self, tmp_path):
        """Capitalized multi-word phrases are extracted as entities."""
        from agent.lcm.hrr.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "test.db")
        store.add_fact("John Doe works at Acme Corp")
        rows = store._conn.execute("SELECT name FROM entities").fetchall()
        names = {r["name"] for r in rows}
        assert "John Doe" in names
        assert "Acme Corp" in names
        store.close()

    def test_trust_feedback(self, tmp_path):
        """record_feedback adjusts trust score."""
        from agent.lcm.hrr.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "test.db")
        fid = store.add_fact("Test fact")
        result = store.record_feedback(fid, helpful=True)
        assert result["new_trust"] > result["old_trust"]
        store.close()

    def test_remove_fact(self, tmp_path):
        """remove_fact deletes and returns True."""
        from agent.lcm.hrr.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "test.db")
        fid = store.add_fact("Temporary fact")
        assert store.remove_fact(fid) is True
        assert store.remove_fact(fid) is False  # already gone
        store.close()


# ---------------------------------------------------------------------------
# 3. Retriever tests
# ---------------------------------------------------------------------------


class TestHrrRetriever:
    def test_search_returns_scored_results(self, tmp_path):
        """FactRetriever.search() returns results with score field."""
        from agent.lcm.hrr.store import MemoryStore
        from agent.lcm.hrr.retrieval import FactRetriever

        store = MemoryStore(db_path=tmp_path / "test.db")
        store.add_fact("Rust has a borrow checker for memory safety", category="programming")
        store.add_fact("Python uses garbage collection", category="programming")
        retriever = FactRetriever(store=store)
        results = retriever.search("memory safety")
        assert len(results) >= 1
        assert all("score" in r for r in results)
        store.close()

    @pytest.mark.skipif(
        not _has_numpy(), reason="numpy required for HRR vector operations"
    )
    def test_probe_entity(self, tmp_path):
        """FactRetriever.probe() finds facts structurally connected to an entity."""
        from agent.lcm.hrr.store import MemoryStore
        from agent.lcm.hrr.retrieval import FactRetriever

        store = MemoryStore(db_path=tmp_path / "test.db")
        store.add_fact("John Smith leads the backend team")
        store.add_fact("The backend team uses PostgreSQL")
        retriever = FactRetriever(store=store)
        results = retriever.probe("John Smith")
        assert len(results) >= 1
        store.close()

    @pytest.mark.skipif(
        not _has_numpy(), reason="numpy required for HRR vector operations"
    )
    def test_reason_multi_entity(self, tmp_path):
        """FactRetriever.reason() finds facts related to multiple entities."""
        from agent.lcm.hrr.store import MemoryStore
        from agent.lcm.hrr.retrieval import FactRetriever

        store = MemoryStore(db_path=tmp_path / "test.db")
        store.add_fact("Alice and Bob collaborate on the API project")
        store.add_fact("Alice works on frontend")
        store.add_fact("Bob works on backend")
        retriever = FactRetriever(store=store)
        results = retriever.reason(["Alice", "Bob"])
        assert len(results) >= 1
        store.close()
