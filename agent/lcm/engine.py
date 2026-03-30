"""LCM Engine — manages active context, compaction, and expansion.

This is the core of the unified LCM system, combining:
- ImmutableStore: append-only archive of all messages
- SummaryDag: tracks summary→source mappings for reversibility
- TokenEstimator: accurate token counting
- Summarizer: structured summary generation
- SemanticIndex: optional embedding-based search
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from agent.lcm.config import LcmConfig
from agent.lcm.dag import MessageId, SummaryDag, SummaryNode
from agent.lcm.store import ImmutableStore
from agent.lcm.summarizer import Summarizer, SummarizerConfig
from agent.lcm.tokens import TokenEstimator, TokenEstimatorConfig
from agent.lcm.semantic import SemanticIndex, SemanticIndexConfig, create_semantic_index

logger = logging.getLogger(__name__)


class CompactionAction(Enum):
    NONE = auto()
    ASYNC = auto()
    BLOCKING = auto()


@dataclass
class ContextEntry:
    kind: str  # "raw" | "summary"
    msg_id: MessageId | None
    node_id: int | None
    message: dict[str, Any]

    @classmethod
    def raw(cls, msg_id: MessageId, message: dict[str, Any]) -> "ContextEntry":
        return cls(kind="raw", msg_id=msg_id, node_id=None, message=message)

    @classmethod
    def summary(cls, node_id: int, message: dict[str, Any]) -> "ContextEntry":
        return cls(kind="summary", msg_id=None, node_id=node_id, message=message)


class LcmEngine:
    """Core context management engine.

    Maintains an append-only store of all ingested messages and a mutable
    active list that may contain raw or summary entries. Compaction
    replaces a contiguous run of raw entries with a single summary entry,
    recording the mapping in the DAG so originals can always be recovered.

    This unified implementation is independent of ContextCompressor and
    provides accurate token estimation, structured summarization, and
    optional semantic search.
    """

    def __init__(
        self,
        config: LcmConfig,
        model: str = "",
        provider: str = "",
        context_length: int = 128_000,
    ) -> None:
        self.config = config
        self.model = model
        self.provider = provider

        # Core components
        self.store: ImmutableStore = ImmutableStore()
        self.dag: SummaryDag = SummaryDag()
        self.active: list[ContextEntry] = []

        # State
        self._async_compaction_pending: bool = False
        self._pinned_ids: set[int] = set()
        self._last_summary: Optional[str] = None

        # Token estimator
        token_config = TokenEstimatorConfig(
            model=model,
            provider=provider,
            context_length=context_length,
        )
        self.token_estimator = TokenEstimator(token_config)

        # Summarizer
        summarizer_config = SummarizerConfig(
            model=config.summary_model or model,
        )
        self.summarizer = Summarizer(summarizer_config, self.token_estimator)

        # Semantic index (optional)
        semantic_config = SemanticIndexConfig(enabled=config.semantic_search)
        self.semantic_index = create_semantic_index(semantic_config)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def context_length(self) -> int:
        """Get the context length from the token estimator."""
        return self.token_estimator.context_length

    @context_length.setter
    def context_length(self, value: int):
        """Set the context length."""
        self.token_estimator.context_length = value

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, message: dict[str, Any]) -> MessageId:
        """Append a message to the store and active list.

        Returns the stable MessageId assigned by the store.
        """
        msg_id = self.store.append(message)
        self.active.append(ContextEntry.raw(msg_id, message))
        return msg_id

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def active_messages(self) -> list[dict[str, Any]]:
        """Return message dicts for all active entries, in order."""
        return [entry.message for entry in self.active]

    def active_tokens(self) -> int:
        """Estimate tokens currently in the active context."""
        return self.token_estimator.estimate(self.active_messages())

    def active_token_breakdown(self) -> dict[str, int]:
        """Return token breakdown for active context."""
        raw_entries = [e for e in self.active if e.kind == "raw"]
        summary_entries = [e for e in self.active if e.kind == "summary"]

        raw_tokens = self.token_estimator.estimate([e.message for e in raw_entries])
        summary_tokens = self.token_estimator.estimate([e.message for e in summary_entries])

        return {
            "total": raw_tokens + summary_tokens,
            "raw": raw_tokens,
            "summary": summary_tokens,
            "raw_count": len(raw_entries),
            "summary_count": len(summary_entries),
        }

    # ------------------------------------------------------------------
    # Threshold checking
    # ------------------------------------------------------------------

    def check_thresholds(self, token_budget: int | None = None) -> CompactionAction:
        """Determine whether compaction is needed given the token budget.

        Args:
            token_budget: Optional override for context length. Uses
                         self.context_length if not provided.

        Returns:
            BLOCKING  — above tau_hard, must compact before next turn
            ASYNC     — above tau_soft, should compact in background (once)
            NONE      — below tau_soft, or ASYNC already pending
        """
        budget = token_budget or self.context_length
        usage = self.active_tokens()
        ratio = usage / budget if budget > 0 else 0.0

        if ratio >= self.config.tau_hard:
            logger.info(
                "LCM: Blocking compaction triggered (usage=%d, budget=%d, ratio=%.2f >= tau_hard=%.2f)",
                usage, budget, ratio, self.config.tau_hard,
            )
            return CompactionAction.BLOCKING

        if ratio >= self.config.tau_soft:
            if self._async_compaction_pending:
                return CompactionAction.NONE
            self._async_compaction_pending = True
            logger.info(
                "LCM: Async compaction suggested (usage=%d, budget=%d, ratio=%.2f >= tau_soft=%.2f)",
                usage, budget, ratio, self.config.tau_soft,
            )
            return CompactionAction.ASYNC

        return CompactionAction.NONE

    # ------------------------------------------------------------------
    # Block finding
    # ------------------------------------------------------------------

    def find_compactable_block(self) -> tuple[int, int] | None:
        """Find the oldest contiguous block of raw entries that can be compacted.

        Rules:
        - The tail of protect_last_n entries is never touched.
        - The block must start after any leading summary entries.
        - Tool-call/result pairs are never split: if the candidate end would
          leave a tool-result entry without its preceding tool-call entry,
          the boundary is moved backwards until the pair is intact.
        - Returns None if there is nothing useful to compact.

        Returns (start, end) slice indices into self.active [start, end).
        """
        protect = self.config.protect_last_n
        total = len(self.active)

        # Need at least protect + 2 entries to have anything outside the tail.
        if total < protect + 2:
            return None

        compactable_limit = total - protect  # exclusive upper bound

        # Find where the leading summaries/pinned entries end (first compactable raw entry).
        start = 0
        while start < compactable_limit and (
            self.active[start].kind == "summary"
            or self.active[start].msg_id in self._pinned_ids
        ):
            start += 1

        if start >= compactable_limit:
            # Everything compactable is already summarised or pinned.
            return None

        # The block runs from start up to the first pinned entry or compactable_limit.
        end = start
        while end < compactable_limit and self.active[end].msg_id not in self._pinned_ids:
            end += 1

        # Integrity: the block must contain at least one entry.
        if end - start < 1:
            return None

        # Adjust end so we never orphan a tool-result from its tool-call.
        # Walk backwards from end-1; if that entry is a tool result, pull end
        # back until the last included entry is not a tool result.
        end = self._retreat_from_tool_result(start, end)
        if end is None or end - start < 1:
            return None

        return (start, end)

    def _retreat_from_tool_result(self, start: int, end: int) -> int | None:
        """Retreat `end` so the block does not end mid-pair.

        A "tool" role message must be immediately preceded (in the block) by
        the assistant message carrying its tool_calls. If the proposed end
        would cut between the assistant-with-tool_calls and the tool result,
        we pull end back to exclude that pair entirely.
        """
        while end > start:
            last_entry = self.active[end - 1]
            if last_entry.message.get("role") == "tool":
                # This tool result's tool_call must also be in the block.
                # The tool_call is the entry immediately before it.
                if end - 1 > start:
                    preceding = self.active[end - 2]
                    if preceding.message.get("tool_calls"):
                        # Pair is intact — fine.
                        break
                # Pair would be split; retreat past the tool result.
                end -= 1
            else:
                break
        return end if end > start else None

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def compact(
        self,
        summary_text: str,
        level: int,
        block_start: int,
        block_end: int,
    ) -> SummaryNode:
        """Replace active[block_start:block_end] with a single summary entry.

        The original MessageIds are recorded in the DAG node so they can be
        expanded later. Returns the new SummaryNode.

        This is the low-level compact method. Use compact_block() for
        automatic summarization.
        """
        block = self.active[block_start:block_end]
        source_ids: list[MessageId] = [
            e.msg_id for e in block if e.msg_id is not None
        ]

        tokens = self.token_estimator.estimate([e.message for e in block])
        node = self.dag.create_node(
            source_ids=source_ids,
            text=summary_text,
            level=level,
            tokens=tokens,
        )

        summary_message: dict[str, Any] = {
            "role": "user",
            "content": f"[Summary of {len(source_ids)} messages]\n{summary_text}",
            "_lcm_summary": True,
            "_lcm_node_id": node.id,
        }
        summary_entry = ContextEntry.summary(node.id, summary_message)

        self.active[block_start:block_end] = [summary_entry]
        self._async_compaction_pending = False

        logger.info(
            "LCM: Compacted %d messages into summary node %d (~%d tokens saved)",
            len(source_ids), node.id, tokens - len(summary_text) // 4,
        )

        return node

    def compact_block(self, start: int, end: int) -> SummaryNode | None:
        """Compact a block with automatic summarization.

        Uses the Summarizer to generate a structured summary of the block,
        then replaces the block with a single summary entry.

        Returns the new SummaryNode, or None if summarization fails.
        """
        if start >= end:
            logger.warning("LCM: Invalid block range [%d, %d)", start, end)
            return None

        block = self.active[start:end]
        block_messages = [e.message for e in block]

        # Generate summary
        summary = self.summarizer.summarize(
            block_messages,
            previous_summary=self._last_summary,
        )

        if summary is None:
            logger.warning("LCM: Summarization failed, block not compacted")
            return None

        # Store summary for iterative updates
        self._last_summary = summary

        # Perform compaction
        return self.compact(summary, level=1, block_start=start, block_end=end)

    def auto_compact(self) -> SummaryNode | None:
        """Find and compact the oldest compactable block.

        This is the main entry point for automatic compaction triggered
        by threshold checks.

        Returns the new SummaryNode, or None if no block found or
        summarization fails.
        """
        block = self.find_compactable_block()
        if block is None:
            return None
        return self.compact_block(block[0], block[1])

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def expand(self, msg_ids: list[MessageId]) -> list[tuple[MessageId, dict[str, Any]]]:
        """Return (id, message) tuples for the given MessageIds from the store."""
        return self.store.get_many(msg_ids)

    def expand_summary(self, node_id: int) -> list[tuple[MessageId, dict[str, Any]]]:
        """Recursively expand all source messages for a DAG node."""
        all_ids = self.dag.all_source_ids(node_id)
        return self.store.get_many(all_ids)

    def focus_summary(self, node_id: int) -> bool:
        """Expand a summary entry back into raw messages in the active list.

        Returns True if successful, False if node not found.
        """
        # Find the summary entry
        summary_index: int | None = None
        for i, entry in enumerate(self.active):
            if entry.kind == "summary" and entry.node_id == node_id:
                summary_index = i
                break

        if summary_index is None:
            return False

        # Get source messages
        pairs = self.expand_summary(node_id)
        if not pairs:
            return False

        # Create raw entries
        expanded_entries = [
            ContextEntry.raw(mid, msg) for mid, msg in pairs
        ]

        # Replace summary with expanded entries
        self.active[summary_index:summary_index + 1] = expanded_entries

        # Reset async flag since context grew
        self._async_compaction_pending = False

        logger.info(
            "LCM: Expanded summary node %d into %d raw messages",
            node_id, len(expanded_entries),
        )

        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> list[tuple[int, dict[str, Any]]]:
        """Search messages in the store.

        Uses semantic search if available, otherwise falls back to keyword search.

        Returns list of (msg_id, message) tuples.
        """
        # Try semantic search first
        if self.semantic_index.is_available() and self.config.semantic_search:
            ids = self.semantic_index.search(query, k=limit)
            if ids:
                return self.store.get_many(ids)

        # Fallback: keyword search
        return self._keyword_search(query, limit)

    def _keyword_search(self, query: str, limit: int) -> list[tuple[int, dict[str, Any]]]:
        """Fallback keyword search."""
        query_lower = query.lower()
        results: list[tuple[int, dict[str, Any]]] = []

        for msg_id in range(len(self.store)):
            msg = self.store.get(msg_id)
            if msg is None:
                continue
            content = str(msg.get("content", "") or "")
            if query_lower in content.lower():
                results.append((msg_id, msg))
                if len(results) >= limit:
                    break

        return results

    def build_semantic_index(self) -> bool:
        """Build the semantic index for the store.

        Returns True if indexing succeeded.
        """
        if not self.semantic_index.is_available():
            return False
        return self.semantic_index.index(self.store)

    # ------------------------------------------------------------------
    # Pinning
    # ------------------------------------------------------------------

    def pin(self, msg_ids: list[int]) -> int:
        """Pin message IDs so they are never compacted.

        Returns the number of newly pinned IDs.
        """
        new_ids = set(msg_ids) - self._pinned_ids

        # Check limit
        if len(self._pinned_ids) + len(new_ids) > self.config.max_pinned:
            remaining = self.config.max_pinned - len(self._pinned_ids)
            new_ids = set(list(new_ids)[:remaining])

        self._pinned_ids.update(new_ids)
        return len(new_ids)

    def unpin(self, msg_ids: list[int]) -> int:
        """Unpin message IDs.

        Returns the number of IDs that were unpinned.
        """
        removed = len(self._pinned_ids.intersection(msg_ids))
        self._pinned_ids.difference_update(msg_ids)
        return removed

    def get_pinned(self) -> list[int]:
        """Return sorted list of pinned message IDs."""
        return sorted(self._pinned_ids)

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    @classmethod
    def rebuild_from_session(
        cls,
        session_data: dict,
        config: LcmConfig,
        model: str = "",
        provider: str = "",
        context_length: int = 128_000,
    ) -> "LcmEngine":
        """Reconstruct engine state from persisted session data.

        Two modes:
        - With ``lcm.original_messages``: the full store backup is restored
          first; raw entries in the active list reference those store positions.
        - Without it (legacy or no prior compaction): each raw message in the
          active list is appended to the store directly.
        """
        engine = cls(config, model, provider, context_length)
        messages = session_data.get("messages", [])
        lcm_meta = session_data.get("lcm", {})
        summaries = lcm_meta.get("summaries", [])
        original_messages = lcm_meta.get("original_messages", [])

        # Rebuild immutable store from original_messages when available.
        for msg in original_messages:
            engine.store.append(msg)

        # Rebuild DAG — must insert nodes in node_id order so the auto-assigned
        # ids from create_node match the stored node_id values.
        for s in sorted(summaries, key=lambda x: x.get("node_id", 0)):
            engine.dag.create_node(
                source_ids=s.get("source_ids", []),
                text=s.get("text", ""),
                level=s.get("level", 1),
                tokens=s.get("tokens", 0),
                children=s.get("child_summaries", []),
            )

        # Determine the store index for the first raw entry in the active list.
        # When original_messages is present the raw entries are the tail of the
        # store, so the first raw entry's id = len(store) - count_raw_in_active.
        if original_messages:
            raw_in_active = sum(1 for m in messages if not m.get("_lcm_summary"))
            next_raw_id = len(engine.store) - raw_in_active
        else:
            next_raw_id = 0  # will be assigned by store.append below

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

        # Restore pinned IDs from metadata
        pinned_ids = lcm_meta.get("pinned", [])
        # Validate: only keep IDs that exist in store
        valid_pins = {pid for pid in pinned_ids if pid < len(engine.store)}
        engine._pinned_ids = valid_pins

        # Restore last summary if available
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

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_expanded(self, msg_ids: list[MessageId]) -> str:
        """Format expanded messages as '[msg N] role: content' lines."""
        pairs = self.expand(msg_ids)
        lines: list[str] = []
        for mid, msg in pairs:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", "") or "")
            lines.append(f"[msg {mid}] {role}: {content}")
        return "\n".join(lines)

    def format_toc(self) -> str:
        """Format a table of contents of the active context."""
        if not self.active:
            return "Active context is empty."

        lines = ["Conversation Timeline:"]
        for i, entry in enumerate(self.active):
            if entry.kind == "raw":
                role = entry.message.get("role", "?")
                content = str(entry.message.get("content", "") or "")
                snippet = content[:60].replace("\n", " ")
                lines.append(f"  [{i}] msg {entry.msg_id} ({role}): {snippet}")
            else:
                content = str(entry.message.get("content", "") or "")
                snippet = content[:60].replace("\n", " ")
                lines.append(f"  [{i}] summary node={entry.node_id}: {snippet}")

        return "\n".join(lines)

    def format_budget(self) -> str:
        """Format a token budget summary."""
        breakdown = self.active_token_breakdown()

        lines = [
            "LCM Context Budget:",
            f"  Active entries : {len(self.active)} ({breakdown['raw_count']} raw, {breakdown['summary_count']} summaries)",
            f"  Active tokens  : ~{breakdown['total']}",
            f"  Raw tokens     : ~{breakdown['raw']}",
            f"  Summary tokens : ~{breakdown['summary']}",
            f"  Store total    : {len(self.store)} messages (immutable)",
            f"  Context limit  : {self.context_length:,} tokens",
            f"  Pinned IDs     : {self.get_pinned() or 'none'}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

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
