"""Bundled Dense Associative Memory package for Hermes LCM.

Provides the core DAM components as a first-party package under agent.lcm.dam,
eliminating the need for sys.path hacks in tests and enabling clean imports.
"""
from agent.lcm.dam.network import DenseAssociativeMemory
from agent.lcm.dam.encoder import MessageEncoder
from agent.lcm.dam.retrieval import DAMRetriever
from agent.lcm.dam.persistence import save_state, load_state

__all__ = [
    "DenseAssociativeMemory",
    "MessageEncoder",
    "DAMRetriever",
    "save_state",
    "load_state",
]
