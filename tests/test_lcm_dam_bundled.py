"""TDD tests for the bundled DAM package at agent/lcm/dam/.

These tests are written FIRST (RED phase) before the implementation.
They exercise imports and basic functionality from the new bundled location.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Import tests (RED: will fail until agent/lcm/dam/ is created)
# ---------------------------------------------------------------------------


class TestImports:
    def test_import_dam_network_from_agent_lcm(self):
        from agent.lcm.dam.network import DenseAssociativeMemory  # noqa: F401

    def test_import_dam_encoder_from_agent_lcm(self):
        from agent.lcm.dam.encoder import MessageEncoder  # noqa: F401

    def test_import_dam_retriever_from_agent_lcm(self):
        from agent.lcm.dam.retrieval import DAMRetriever  # noqa: F401

    def test_import_dam_persistence_from_agent_lcm(self):
        from agent.lcm.dam.persistence import save_state, load_state  # noqa: F401

    def test_import_dam_schemas_from_agent_lcm(self):
        from agent.lcm.dam.schemas import (  # noqa: F401
            DAM_SEARCH,
            DAM_RECALL,
            DAM_COMPOSE,
            DAM_STATUS,
        )

    def test_dam_package_exports(self):
        """agent/lcm/dam/__init__.py must export core classes."""
        from agent.lcm.dam import DenseAssociativeMemory, MessageEncoder, DAMRetriever  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Functional tests — verify moved code works correctly
# ---------------------------------------------------------------------------


class TestDamNetworkWorksAfterMove:
    def test_learn_and_recall(self):
        from agent.lcm.dam.network import DenseAssociativeMemory

        net = DenseAssociativeMemory(nv=256, nh=16, lr=0.01)

        # Create 3 distinct patterns
        rng = np.random.default_rng(seed=42)
        patterns = rng.standard_normal((3, 256)).astype(np.float32)
        for i in range(3):
            patterns[i] /= np.linalg.norm(patterns[i])

        loss = net.learn(patterns, epochs=5)
        assert isinstance(loss, float), "learn() should return a float loss"
        assert loss >= 0.0

        # Recall should return vectors of correct shape
        v_out, s = net.recall(patterns[0])
        assert v_out.shape == (256,), "recalled vector has wrong shape"
        assert s.shape == (16,), "hidden activation has wrong shape"

    def test_get_state_roundtrip(self):
        from agent.lcm.dam.network import DenseAssociativeMemory

        net = DenseAssociativeMemory(nv=128, nh=8)
        state = net.get_state()
        net2 = DenseAssociativeMemory.from_state(state)
        assert np.allclose(net.xi, net2.xi)
        assert net.nv == net2.nv
        assert net.nh == net2.nh


class TestDamRetrieverWorksAfterMove:
    def test_sync_and_search(self):
        """Create encoder+network+retriever from new imports, verify sync+search."""
        from agent.lcm.dam.network import DenseAssociativeMemory
        from agent.lcm.dam.encoder import MessageEncoder
        from agent.lcm.dam.retrieval import DAMRetriever
        from agent.lcm.engine import LcmEngine
        from agent.lcm.config import LcmConfig
        from agent.lcm.tools import set_engine

        config = LcmConfig(enabled=True, protect_last_n=2)
        engine = LcmEngine(config)
        engine.ingest({"role": "user", "content": "Python error handling"})
        engine.ingest({"role": "assistant", "content": "Use try/except for errors"})
        engine.ingest({"role": "user", "content": "What is the weather"})
        set_engine(engine)

        try:
            net = DenseAssociativeMemory(nv=256, nh=16, lr=0.01)
            enc = MessageEncoder(nv=256)
            retriever = DAMRetriever(net, enc)

            n = retriever.sync_with_store()
            assert n == 3, f"Expected 3 messages indexed, got {n}"
            assert len(retriever._pattern_cache) == 3

            results = retriever.search("Python exception")
            assert isinstance(results, list)
        finally:
            set_engine(None)

    def test_retriever_status(self):
        from agent.lcm.dam.network import DenseAssociativeMemory
        from agent.lcm.dam.encoder import MessageEncoder
        from agent.lcm.dam.retrieval import DAMRetriever

        net = DenseAssociativeMemory(nv=256, nh=16)
        enc = MessageEncoder(nv=256)
        retriever = DAMRetriever(net, enc)

        status = retriever.get_status()
        assert status["nv"] == 256
        assert status["nh"] == 16
        assert status["patterns_stored"] == 0
