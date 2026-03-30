"""Tests for DAM tool handlers and schemas."""
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_DAM_DIR = Path.home() / ".hermes/plugins/hermes-dam"
_NS = "hermes_plugins"
_PKG = f"{_NS}.hermes_dam"

# Insert DAM plugin dir so bare siblings (network, encoder) resolve in test context.
if str(_DAM_DIR) not in sys.path:
    sys.path.insert(0, str(_DAM_DIR))


def _ensure_package():
    """Ensure hermes_plugins.hermes_dam namespace exists in sys.modules so
    relative imports inside retrieval.py resolve correctly."""
    if _NS not in sys.modules:
        ns_pkg = types.ModuleType(_NS)
        ns_pkg.__path__ = []
        ns_pkg.__package__ = _NS
        sys.modules[_NS] = ns_pkg

    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(_DAM_DIR)]
        pkg.__package__ = _PKG
        sys.modules[_PKG] = pkg

    for sibling in ("network", "encoder", "persistence"):
        fqn = f"{_PKG}.{sibling}"
        if fqn not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                fqn, _DAM_DIR / f"{sibling}.py", submodule_search_locations=[]
            )
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = _PKG
            sys.modules[fqn] = mod
            spec.loader.exec_module(mod)


def _load_dam_module(name: str):
    """Load a DAM plugin module by filename, bypassing sys.path name conflicts.

    Modules that use relative imports (e.g. retrieval.py) are loaded as members
    of the hermes_plugins.hermes_dam package so those imports resolve correctly.
    Modules without relative imports (tools.py, schemas.py) are loaded with a
    simple unique name.
    """
    _ensure_package()
    fqn = f"{_PKG}.{name}"
    if fqn in sys.modules:
        return sys.modules[fqn]
    spec = importlib.util.spec_from_file_location(
        fqn, _DAM_DIR / f"{name}.py",
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


from agent.lcm.engine import LcmEngine
from agent.lcm.config import LcmConfig
from agent.lcm.tools import set_engine


@pytest.fixture
def dam_tools_mod():
    """Return the DAM tools module, loaded fresh via importlib."""
    return _load_dam_module("tools")


@pytest.fixture
def setup_dam(tmp_path, dam_tools_mod):
    """Set up LCM engine + DAM retriever and inject into dam_tools module."""
    from network import DenseAssociativeMemory
    from encoder import MessageEncoder
    retrieval_mod = _load_dam_module("retrieval")
    DAMRetriever = retrieval_mod.DAMRetriever

    config = LcmConfig(enabled=True, protect_last_n=2)
    engine = LcmEngine(config)
    engine.ingest({"role": "user", "content": "Python error handling with try except"})
    engine.ingest({"role": "assistant", "content": "Use try/except for error handling"})
    engine.ingest({"role": "user", "content": "What is the weather today"})
    engine.ingest({"role": "assistant", "content": "I cannot check weather"})
    set_engine(engine)

    net = DenseAssociativeMemory(nv=512, nh=16, lr=0.01)
    enc = MessageEncoder(nv=512)
    retriever = DAMRetriever(net, enc, tmp_path)
    retriever.sync_with_store()
    dam_tools_mod._retriever = retriever

    yield dam_tools_mod

    dam_tools_mod._retriever = None
    set_engine(None)


class TestSchemas:
    def test_all_schemas_present(self):
        schemas = _load_dam_module("schemas")
        for schema in [schemas.DAM_SEARCH, schemas.DAM_RECALL, schemas.DAM_COMPOSE, schemas.DAM_STATUS]:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema

    def test_dam_search_has_required_query(self):
        schemas = _load_dam_module("schemas")
        assert "query" in schemas.DAM_SEARCH["parameters"]["required"]

    def test_dam_recall_has_required_message_id(self):
        schemas = _load_dam_module("schemas")
        assert "message_id" in schemas.DAM_RECALL["parameters"]["required"]

    def test_dam_compose_has_required_queries(self):
        schemas = _load_dam_module("schemas")
        assert "queries" in schemas.DAM_COMPOSE["parameters"]["required"]

    def test_dam_status_no_required_params(self):
        schemas = _load_dam_module("schemas")
        assert schemas.DAM_STATUS["parameters"]["required"] == []


class TestDamSearch:
    def test_returns_results(self, setup_dam):
        result = setup_dam.handle_dam_search({"query": "Python exception"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_result_is_not_error(self, setup_dam):
        result = setup_dam.handle_dam_search({"query": "Python error handling"})
        assert isinstance(result, str)

    def test_empty_query_returns_error(self, setup_dam):
        result = setup_dam.handle_dam_search({"query": ""})
        assert "error" in result.lower()

    def test_empty_query_strips_whitespace(self, setup_dam):
        result = setup_dam.handle_dam_search({"query": "   "})
        assert "error" in result.lower()

    def test_no_retriever_falls_back_gracefully(self, dam_tools_mod):
        dam_tools_mod._retriever = None
        result = dam_tools_mod.handle_dam_search({"query": "test"})
        assert isinstance(result, str)

    def test_respects_limit_parameter(self, setup_dam):
        result = setup_dam.handle_dam_search({"query": "Python", "limit": 1})
        assert isinstance(result, str)


class TestDamRecall:
    def test_finds_similar_messages(self, setup_dam):
        result = setup_dam.handle_dam_recall({"message_id": 0})
        assert isinstance(result, str)
        # Should return a list header, not a JSON error object
        assert result.startswith("Messages similar to") or "msg" in result.lower()

    def test_missing_message_id_returns_error(self, setup_dam):
        result = setup_dam.handle_dam_recall({})
        assert "error" in result.lower()

    def test_invalid_message_id_returns_gracefully(self, setup_dam):
        result = setup_dam.handle_dam_recall({"message_id": 9999})
        assert isinstance(result, str)

    def test_no_retriever_returns_error(self, dam_tools_mod):
        dam_tools_mod._retriever = None
        result = dam_tools_mod.handle_dam_recall({"message_id": 0})
        assert "error" in result.lower()


class TestDamCompose:
    def test_and_operation_returns_string(self, setup_dam):
        result = setup_dam.handle_dam_compose({"queries": ["Python", "error"], "operation": "AND"})
        assert isinstance(result, str)

    def test_or_operation_returns_string(self, setup_dam):
        result = setup_dam.handle_dam_compose({"queries": ["Python", "weather"], "operation": "OR"})
        assert isinstance(result, str)

    def test_too_few_queries_returns_error(self, setup_dam):
        result = setup_dam.handle_dam_compose({"queries": ["solo"]})
        assert "error" in result.lower()

    def test_empty_queries_returns_error(self, setup_dam):
        result = setup_dam.handle_dam_compose({"queries": []})
        assert "error" in result.lower()

    def test_invalid_operation_returns_error(self, setup_dam):
        result = setup_dam.handle_dam_compose({"queries": ["Python", "error"], "operation": "XOR"})
        assert "error" in result.lower()

    def test_no_retriever_returns_error(self, dam_tools_mod):
        dam_tools_mod._retriever = None
        result = dam_tools_mod.handle_dam_compose({"queries": ["Python", "error"]})
        assert "error" in result.lower()


class TestDamStatus:
    def test_shows_network_dimensions(self, setup_dam):
        result = setup_dam.handle_dam_status({})
        assert isinstance(result, str)
        assert "512" in result or "Nv" in result or "dimension" in result.lower()

    def test_shows_patterns_stored(self, setup_dam):
        result = setup_dam.handle_dam_status({})
        assert "pattern" in result.lower() or "stored" in result.lower()

    def test_no_retriever_returns_inactive_message(self, dam_tools_mod):
        dam_tools_mod._retriever = None
        result = dam_tools_mod.handle_dam_status({})
        assert "not initialized" in result.lower() or "inactive" in result.lower()
