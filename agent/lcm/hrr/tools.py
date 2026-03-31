"""Unified memory tools — auto-route across LCM, DAM, and HRR layers.

Six tools replace 11 fragmented lcm_* / dam_* tools:
  memory_search  — unified search (DAM -> HRR -> keyword fallback)
  memory_pin     — pin in LCM + crystallize to HRR
  memory_expand  — expand summaries (session) or recall facts (cross-session)
  memory_forget  — compact in LCM + optionally lower HRR trust
  memory_reason  — HRR compositional query (probe/related/reason/contradict)
  memory_budget  — token budget breakdown (delegates to lcm_budget)

The old lcm_* and dam_* tools remain fully functional (backward compat).
"""

from __future__ import annotations

import logging
from typing import Any

from agent.lcm.tools import get_engine

logger = logging.getLogger(__name__)


def _require_engine():
    """Return the active engine or raise RuntimeError."""
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Memory system not initialized. No LCM engine registered.")
    return engine


# ---------------------------------------------------------------------------
# memory_search
# ---------------------------------------------------------------------------


def handle_memory_search(args: dict[str, Any]) -> str:
    """Unified search across DAM (session) and HRR (persistent) layers."""
    engine = _require_engine()
    query = str(args.get("query", "")).strip()
    source = args.get("source", "auto")
    limit = 10

    if not query:
        return "Error: query is required."

    results: list[str] = []

    # Layer 1: DAM (in-session semantic search) — skip when source == "memory"
    if source in ("auto", "session") and engine.retriever is not None:
        try:
            dam_results = engine.retriever.search(query, limit=limit)
            for msg_id, score in dam_results:
                msg = engine.store.get(msg_id)
                if msg:
                    role = msg.get("role", "unknown")
                    snippet = str(msg.get("content", "") or "")[:200].replace("\n", " ")
                    results.append(
                        f"[Session msg#{msg_id} score={score:.2f}] {role}: {snippet}"
                    )
        except Exception as exc:
            logger.debug("DAM search failed, continuing: %s", exc)

    # Layer 2: HRR (cross-session persistent knowledge) — skip when source == "session"
    if source in ("auto", "memory") and engine.hrr_store is not None:
        try:
            from agent.lcm.hrr.retrieval import FactRetriever

            retriever = FactRetriever(store=engine.hrr_store)
            hrr_results = retriever.search(query, limit=limit - len(results))
            for fact in hrr_results:
                results.append(
                    f"[Memory fact#{fact['fact_id']} trust={fact.get('trust_score', '?')}]"
                    f" {fact['content']}"
                )
        except Exception as exc:
            logger.debug("HRR search failed, continuing: %s", exc)

    # Layer 3: Keyword fallback (LCM store linear scan)
    if not results:
        try:
            from agent.lcm.tools import handle_lcm_search

            return handle_lcm_search({"query": query})
        except Exception as exc:
            logger.debug("LCM keyword fallback failed: %s", exc)

    if not results:
        return f"No results found for '{query}'."

    return "\n".join(results)


# ---------------------------------------------------------------------------
# memory_pin
# ---------------------------------------------------------------------------


def handle_memory_pin(args: dict[str, Any]) -> str:
    """Pin messages in LCM and crystallize them as persistent HRR facts."""
    engine = _require_engine()
    message_ids = args.get("message_ids", [])
    reason = str(args.get("reason", "")).strip()

    # Normalise: accept both list[int] and comma-separated string (LCM compat)
    if isinstance(message_ids, (list, tuple)):
        ids_str = ",".join(str(i) for i in message_ids)
    else:
        ids_str = str(message_ids)

    # Delegate to LCM pin
    from agent.lcm.tools import handle_lcm_pin

    result = handle_lcm_pin({"message_ids": ids_str})

    # Auto-crystallize to HRR persistent store
    if engine.hrr_store is not None:
        crystallized = 0
        for mid in (message_ids if isinstance(message_ids, (list, tuple)) else []):
            try:
                mid_int = int(mid)
                msg = engine.store.get(mid_int)
                if msg and msg.get("content"):
                    engine.hrr_store.add_fact(
                        content=str(msg["content"]),
                        category="pinned",
                        tags=reason or "memory_pin",
                    )
                    crystallized += 1
            except Exception as exc:
                logger.warning(
                    "Failed to crystallize pinned message %s: %s", mid, exc
                )
        if crystallized:
            result += f" Crystallized {crystallized} message(s) to persistent memory."

    return result


# ---------------------------------------------------------------------------
# memory_expand
# ---------------------------------------------------------------------------


def handle_memory_expand(args: dict[str, Any]) -> str:
    """Expand session summaries or recall persistent facts by query."""
    engine = _require_engine()
    source = args.get("source", "session")

    if source == "session":
        # Delegate to LCM expand — convert list[int] to comma-separated string if needed
        expand_args = dict(args)
        message_ids = expand_args.get("message_ids")
        if isinstance(message_ids, (list, tuple)):
            expand_args["message_ids"] = ",".join(str(i) for i in message_ids)
        from agent.lcm.tools import handle_lcm_expand

        return handle_lcm_expand(expand_args)

    # Cross-session: query HRR persistent store
    query = str(args.get("query", "")).strip()
    if not query:
        return "No query provided. Supply 'query' when using source='memory'."

    if engine.hrr_store is None:
        return "Persistent memory not available for cross-session recall."

    try:
        from agent.lcm.hrr.retrieval import FactRetriever

        retriever = FactRetriever(store=engine.hrr_store)
        results = retriever.search(query, limit=5)
        if not results:
            return f"No persistent memories found for '{query}'."

        lines = [
            f"- [{r.get('category', 'general')}] {r['content']}"
            f" (trust: {r.get('trust_score', '?')})"
            for r in results
        ]
        return "Persistent memories:\n" + "\n".join(lines)
    except Exception as exc:
        return f"Error querying persistent memory: {exc}"


# ---------------------------------------------------------------------------
# memory_forget
# ---------------------------------------------------------------------------


def handle_memory_forget(args: dict[str, Any]) -> str:
    """Compact messages in LCM and optionally lower HRR trust for related facts."""
    engine = _require_engine()
    lower_trust = bool(args.get("lower_trust", False))
    message_ids = args.get("message_ids", [])

    # Normalise to comma-separated string for LCM
    if isinstance(message_ids, (list, tuple)):
        ids_str = ",".join(str(i) for i in message_ids)
    else:
        ids_str = str(message_ids)

    from agent.lcm.tools import handle_lcm_forget

    forget_args = dict(args)
    forget_args["message_ids"] = ids_str
    result = handle_lcm_forget(forget_args)

    # Optionally lower HRR trust for related persistent facts
    if lower_trust and engine.hrr_store is not None:
        ids_list = (
            message_ids
            if isinstance(message_ids, (list, tuple))
            else [int(x.strip()) for x in str(message_ids).split(",") if x.strip()]
        )
        for mid in ids_list:
            try:
                mid_int = int(mid)
                msg = engine.store.get(mid_int)
                if msg and msg.get("content"):
                    content_snippet = str(msg["content"])[:100]
                    matches = engine.hrr_store.search_facts(content_snippet, limit=3)
                    for match in matches:
                        engine.hrr_store.record_feedback(match["fact_id"], helpful=False)
            except Exception as exc:
                logger.warning(
                    "Failed to lower trust for message %s: %s", mid, exc
                )

    return result


# ---------------------------------------------------------------------------
# memory_reason
# ---------------------------------------------------------------------------


def handle_memory_reason(args: dict[str, Any]) -> str:
    """Compositional reasoning over persistent HRR knowledge."""
    engine = _require_engine()
    entities = list(args.get("entities", []))
    action = str(args.get("action", "reason")).strip()

    if not entities:
        return "Error: entities list must not be empty."

    if engine.hrr_store is None:
        return (
            "Persistent memory not available. "
            "The memory_reason tool requires cross-session knowledge (hrr_store)."
        )

    try:
        from agent.lcm.hrr.retrieval import FactRetriever

        retriever = FactRetriever(store=engine.hrr_store)

        if action == "probe":
            if len(entities) != 1:
                return "Error: 'probe' action requires exactly one entity."
            results = retriever.probe(entities[0], limit=10)

        elif action == "related":
            if len(entities) != 1:
                return "Error: 'related' action requires exactly one entity."
            results = retriever.related(entities[0], limit=10)

        elif action == "contradict":
            contradictions = retriever.contradict(limit=10)
            if not contradictions:
                return "No contradictions found in persistent memory."
            lines: list[str] = []
            for c in contradictions:
                lines.append(
                    f"Contradiction (score={c['contradiction_score']}):"
                )
                lines.append(f"  A: {c['fact_a']['content']}")
                lines.append(f"  B: {c['fact_b']['content']}")
                shared = ", ".join(c.get("shared_entities", []))
                lines.append(f"  Shared entities: {shared or 'none'}")
            return "\n".join(lines)

        else:
            # Default: "reason" — multi-entity AND intersection
            results = retriever.reason(entities, limit=10)

        if not results:
            return f"No facts found for entities: {', '.join(entities)}"

        lines = [
            f"- {r['content']}"
            f" (score: {r.get('score', 0.0):.2f}, trust: {r.get('trust_score', '?')})"
            for r in results
        ]
        return f"Facts related to {', '.join(entities)}:\n" + "\n".join(lines)

    except Exception as exc:
        return f"Error in compositional reasoning: {exc}"


# ---------------------------------------------------------------------------
# memory_budget
# ---------------------------------------------------------------------------


def handle_memory_budget(args: dict[str, Any]) -> str:
    """Token budget breakdown — delegates to lcm_budget."""
    _require_engine()  # raise RuntimeError early if no engine registered
    from agent.lcm.tools import handle_lcm_budget

    return handle_lcm_budget(args)


# ---------------------------------------------------------------------------
# Schema dict for tool registration
# ---------------------------------------------------------------------------

try:
    from agent.lcm.hrr.schemas import MEMORY_TOOL_SCHEMAS
except ImportError:
    MEMORY_TOOL_SCHEMAS: dict[str, Any] = {}  # type: ignore[no-redef]

# Handler dispatch table — maps tool name to handler function
MEMORY_TOOL_HANDLERS: dict[str, Any] = {
    "memory_search": handle_memory_search,
    "memory_pin": handle_memory_pin,
    "memory_expand": handle_memory_expand,
    "memory_forget": handle_memory_forget,
    "memory_reason": handle_memory_reason,
    "memory_budget": handle_memory_budget,
}
