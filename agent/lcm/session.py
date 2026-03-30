"""LCM Session mixin — session persistence for LcmEngine."""
from __future__ import annotations

import logging
from typing import Any

from agent.lcm.config import LcmConfig
from agent.lcm.store import ImmutableStore
from agent.lcm.dag import SummaryDag

logger = logging.getLogger(__name__)


class LcmSessionMixin:
    """Session persistence capabilities.

    Expects the host class to provide:
    - self.store: ImmutableStore
    - self.dag: SummaryDag
    - self.active: list[ContextEntry]
    - self._pinned_ids: set[int]
    - self._async_compaction_pending: bool
    - self._last_summary: Optional[str]
    - self.summarizer: Summarizer (has .reset())
    - self.semantic_index: SemanticIndex (has .clear())
    - cls(config, model, provider, context_length) constructor
    """

    @classmethod
    def rebuild_from_session(
        cls,
        session_data: dict,
        config: LcmConfig,
        model: str = "",
        provider: str = "",
        context_length: int = 128_000,
    ) -> "LcmSessionMixin":
        """Reconstruct engine state from persisted session data.

        Two modes:
        - With ``lcm.original_messages``: the full store backup is restored
          first; raw entries in the active list reference those store positions.
        - Without it (legacy or no prior compaction): each raw message in the
          active list is appended to the store directly.
        """
        # Import here to avoid circular imports; ContextEntry lives in engine
        from agent.lcm.engine import ContextEntry

        engine = cls(config, model, provider, context_length)
        messages = session_data.get("messages", [])
        lcm_meta = session_data.get("lcm", {})
        summaries = lcm_meta.get("summaries", [])
        original_messages = lcm_meta.get("original_messages", [])

        for msg in original_messages:
            engine.store.append(msg)

        for s in sorted(summaries, key=lambda x: x.get("node_id", 0)):
            engine.dag.create_node(
                source_ids=s.get("source_ids", []),
                text=s.get("text", ""),
                level=s.get("level", 1),
                tokens=s.get("tokens", 0),
                children=s.get("child_summaries", []),
            )

        if original_messages:
            raw_in_active = sum(1 for m in messages if not m.get("_lcm_summary"))
            next_raw_id = len(engine.store) - raw_in_active
        else:
            next_raw_id = 0

        for msg in messages:
            if msg.get("_lcm_summary"):
                node_id = msg.get("_lcm_node_id", 0)
                engine.active.append(ContextEntry.summary(node_id, msg))
            else:
                if original_messages:
                    msg_id = next_raw_id
                    next_raw_id += 1
                else:
                    msg_id = engine.store.append(msg)
                engine.active.append(ContextEntry.raw(msg_id, msg))

        pinned_ids = lcm_meta.get("pinned", [])
        valid_pins = {pid for pid in pinned_ids if pid < len(engine.store)}
        engine._pinned_ids = valid_pins

        engine._last_summary = lcm_meta.get("last_summary")

        logger.info(
            "LCM: Rebuilt from session: %d messages, %d summaries, %d pinned",
            len(engine.active), len(engine.dag.nodes), len(engine._pinned_ids),
        )

        return engine

    def to_session_metadata(self) -> dict:
        """Serialize engine state for session JSON persistence."""
        summaries = []
        for node in self.dag.nodes:
            summaries.append({
                "node_id": node.id,
                "source_ids": node.source_ids,
                "child_summaries": node.child_summaries,
                "text": node.text,
                "level": node.level,
                "tokens": node.tokens,
            })

        original_messages = []
        for i in range(len(self.store)):
            msg = self.store.get(i)
            if msg is not None:
                original_messages.append(msg)

        return {
            "summaries": summaries,
            "original_messages": original_messages,
            "store_size": len(self.store),
            "pinned": sorted(self._pinned_ids),
            "last_summary": self._last_summary,
        }

    def reset(self):
        """Reset engine state for a new session."""
        self.store = ImmutableStore()
        self.dag = SummaryDag()
        self.active = []
        self._async_compaction_pending = False
        self._pinned_ids = set()
        self._last_summary = None
        self.summarizer.reset()
        self.semantic_index.clear()
