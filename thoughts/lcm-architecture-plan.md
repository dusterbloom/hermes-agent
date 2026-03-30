# Lossless Context Management (LCM) for Hermes-Agent

## Status (as of 2026-03-30)

### Shipped
- Core engine: immutable store, summary DAG, active context
- 7 agent tools: expand, pin, forget, search, focus, budget, toc
- Three-level escalation with deterministic fallback
- Tool-call pair protection during compaction
- Session persistence (to_session_metadata / rebuild_from_session)
- Config integration (lcm section in config.yaml)
- Agent loop integration (run_agent.py)
- DAM plugin: neural pattern-completion search (hermes-dam)
- Default: ENABLED

### Deferred
- tiktoken integration (using chars//4 estimation)
- Structured summary templates (using generic format)
- LLM-based summarization (_call_summary_llm is a stub — falls through to Level 3 deterministic)
- Cross-session DAG persistence in gateway sessions

## Architecture Plan

### What is LCM?

**Lossless Context Management** — a novel architecture from Ehrlich & Blackman (2026) that replaces destructive context trimming with a dual-state memory system combining an immutable store and a hierarchical summary DAG.

Instead of discarding old messages when context fills up, LCM:
1. **Preserves everything** in an immutable store (session JSONL)
2. **Summarizes old blocks** into the active context (what gets sent to LLM)
3. **Keeps lossless pointers** — LLM can call tools to retrieve originals anytime
4. **Intelligently compacts** using a 3-level escalation protocol

### Source: Nanobot Implementation

Located at `/home/peppi/Dev/nanobot/src/agent/lcm.rs` (865 lines, Rust).

---

## Module Design

### Directory Structure

```
agent/lcm/
├── __init__.py        # Public API: LcmEngine, LcmConfig, CompactionAction
├── config.py          # LcmConfig dataclass
├── dag.py             # SummaryDag, SummaryNode (pure data structures)
├── store.py           # ImmutableStore (append-only message log)
├── engine.py          # LcmEngine (state machine, threshold checks, compaction)
├── escalation.py      # Three-level summary escalation (pure functions)
├── refusal.py         # LLM refusal pattern detection
├── expand_tool.py     # lcm_expand tool registration
└── tools.py           # Extended agent tools: pin, forget, search, focus, budget, toc
```

### Core Data Structures

```python
# agent/lcm/dag.py

MessageId = int  # Index into the immutable store

@dataclass
class SummaryNode:
    id: int
    source_ids: list[MessageId]     # Original messages this covers
    child_summaries: list[int]      # If merged from other summaries
    text: str                       # Summary content
    tokens: int                     # Estimated token count
    level: int                      # Escalation level (1, 2, or 3)

class SummaryDag:
    """Directed acyclic graph of summary nodes."""
    nodes: list[SummaryNode]
    next_id: int

    def create_node(source_ids, text, tokens, level, children=None) -> SummaryNode
    def get(node_id) -> SummaryNode | None
    def all_source_ids(node_id) -> list[MessageId]  # Recursive
```

```python
# agent/lcm/store.py

class ImmutableStore:
    """Append-only message log. Never modified after append."""
    _messages: list[dict]

    def append(message) -> MessageId
    def get(msg_id) -> dict | None
    def get_many(msg_ids) -> list[tuple[MessageId, dict]]
    def __len__() -> int
```

```python
# agent/lcm/config.py

@dataclass
class LcmConfig:
    enabled: bool = False
    tau_soft: float = 0.50            # Async compaction threshold
    tau_hard: float = 0.85            # Blocking compaction threshold
    deterministic_target: int = 512   # Token target for Level 3 truncation
    protect_last_n: int = 4           # Never compact the last N raw messages
    summary_model: str = ""           # Override model for summarization
```

```python
# agent/lcm/engine.py

class CompactionAction(Enum):
    NONE = auto()
    ASYNC = auto()      # Compact before next LLM call (non-urgent)
    BLOCKING = auto()   # Must compact NOW before next LLM call

@dataclass
class ContextEntry:
    kind: str                    # "raw" or "summary"
    msg_id: MessageId | None
    node_id: int | None
    message: dict[str, Any]

class LcmEngine:
    config: LcmConfig
    dag: SummaryDag
    store: ImmutableStore
    active: list[ContextEntry]

    def ingest(message) -> MessageId
    def active_messages() -> list[dict]
    def active_tokens() -> int
    def check_thresholds(available_budget) -> CompactionAction
    def find_compactable_block() -> tuple[int, int] | None
    def compact(summary_text, level, block_start, block_end) -> SummaryNode
    def expand(msg_ids) -> list[tuple[MessageId, dict]]
    def expand_summary(node_id) -> list[tuple[MessageId, dict]]
    def rebuild_from_session(session_data, config) -> LcmEngine
```

---

## Agent Context Tools (Beyond expand)

LCM enables the agent to be an **active curator** of its own context, not just a passive recipient. These tools turn context into a first-class resource:

### Core Tools

| Tool | Description | Agent Use Case |
|------|-------------|---------------|
| `lcm_expand` | Retrieve originals behind a summary | "I need the exact code from earlier" |
| `lcm_pin` | Mark messages as never-compact | "This schema definition is load-bearing" |
| `lcm_forget` | Aggressively compact a block now | "That debug tangent is resolved, reclaim tokens" |
| `lcm_search` | Keyword search across immutable store | "Did we discuss rate limiting?" without expanding all |
| `lcm_focus` | Temporarily expand a topic, auto-recompact when done | "Pull back auth discussion, work on it, release" |
| `lcm_budget` | Show context usage breakdown | "How much room? What's the biggest reclaimable block?" |
| `lcm_toc` | Timeline / table of contents | "Map of what we covered for navigation" |

### Tool Schemas

```python
# agent/lcm/tools.py

LCM_PIN_SCHEMA = {
    "name": "lcm_pin",
    "description": "Pin messages so they are never compacted. Use for critical context.",
    "parameters": {
        "type": "object",
        "properties": {
            "message_ids": {"type": "string", "description": "Comma-separated message IDs to pin"},
        },
        "required": ["message_ids"],
    },
}

LCM_FORGET_SCHEMA = {
    "name": "lcm_forget",
    "description": "Aggressively compact specific messages to reclaim context budget.",
    "parameters": {
        "type": "object",
        "properties": {
            "message_ids": {"type": "string", "description": "Comma-separated message IDs to compact"},
            "reason": {"type": "string", "description": "Why these can be forgotten (for summary)"},
        },
        "required": ["message_ids"],
    },
}

LCM_SEARCH_SCHEMA = {
    "name": "lcm_search",
    "description": "Search across all messages (including compacted) without expanding them.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search terms"},
            "limit": {"type": "integer", "description": "Max results (default 10)"},
        },
        "required": ["query"],
    },
}

LCM_FOCUS_SCHEMA = {
    "name": "lcm_focus",
    "description": "Temporarily expand a summary into active context. Auto-recompacts on next threshold.",
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {"type": "integer", "description": "Summary node to expand"},
        },
        "required": ["node_id"],
    },
}

LCM_BUDGET_SCHEMA = {
    "name": "lcm_budget",
    "description": "Show context budget usage breakdown.",
    "parameters": {"type": "object", "properties": {}},
}

LCM_TOC_SCHEMA = {
    "name": "lcm_toc",
    "description": "Get a table of contents of the conversation.",
    "parameters": {"type": "object", "properties": {}},
}
```

---

## State Machine

```
NORMAL (< τ_soft 50%)
├─ Messages ingested raw into active + store
└─ No compaction triggered
        │ active tokens >= τ_soft
        ▼
ASYNC COMPACTION (50-85%)
├─ Flag set: compact before next LLM call
├─ Non-blocking — current response continues
└─ Compaction runs at next opportunity
        │ active tokens >= τ_hard
        ▼
BLOCKING COMPACTION (>85%)
├─ Must compact NOW before next LLM call
├─ Three-level escalation
└─ Guaranteed convergence via Level 3
```

### Three-Level Escalation

1. **Level 1** — LLM summarize with `mode="preserve_details"` (target: τ_soft × 80%)
2. **Level 2** — LLM summarize with `mode="bullet_points"` (half of Level 1 budget)
3. **Level 3** — Deterministic truncation (no LLM, guaranteed ≤ target tokens)

Selection: Try Level 1 → if output ≥ input or refusal detected → Level 2 → if still fails → Level 3

---

## Integration Points

### 1. AIAgent.__init__() (run_agent.py)

Where `ContextCompressor` is created, add LCM initialization gated on config:

```python
_lcm_cfg = _agent_cfg.get("lcm", {})
lcm_enabled = str(_lcm_cfg.get("enabled", False)).lower() in ("true", "1", "yes")
if lcm_enabled:
    from agent.lcm import LcmEngine, LcmConfig
    self.lcm_engine = LcmEngine(LcmConfig(**_lcm_cfg))
else:
    self.lcm_engine = None
```

### 2. Message Ingestion

Add `_append_message(self, messages, msg)` helper that appends to `messages` and calls `self.lcm_engine.ingest(msg)` when LCM is active.

Hook into all message append points:
- User message append (~line 5473)
- Assistant message append (~line 6761)
- Tool result appends (sequential + concurrent)

### 3. Compression Trigger Points

Replace compression calls with LCM-aware logic:

```python
if self.lcm_engine:
    action = self.lcm_engine.check_thresholds(available_budget)
    if action != CompactionAction.NONE:
        messages = self._lcm_compact(messages, system_message, blocking=(action == CompactionAction.BLOCKING))
elif self.compression_enabled:
    # Existing ContextCompressor path (unchanged)
```

### 4. Tool Registration

Register all LCM tools when engine is active:

```python
if self.lcm_engine:
    from agent.lcm.tools import register_all_lcm_tools
    register_all_lcm_tools(self.lcm_engine)
```

### 5. Session Persistence

Add `lcm` metadata to session JSON:

```json
{
    "messages": [...],
    "lcm": {
        "summaries": [{"node_id": 0, "source_ids": [3,4,5], "text": "...", "level": 1, "tokens": 250}],
        "pinned": [1, 2, 15],
        "store_size": 42
    }
}
```

---

## Improvements over Nanobot

| Area | Nanobot | Hermes LCM |
|------|---------|------------|
| Token counting | chars/4 estimate | tiktoken for OpenAI, cached on ContextEntry |
| Summary prompts | Generic | Structured template (Goal, Progress, Decisions, Files) |
| Tool-call awareness | Splits tool/result pairs | Boundary alignment from existing compressor |
| Pre-compaction | None | Memory flush gives LLM chance to save important context |
| Agent tools | expand only | expand, pin, forget, search, focus, budget, toc |
| Protection window | Hardcoded last 4 | Configurable protect_last_n + token-budget tail |
| Concurrency | Arc<Mutex> (Rust async) | Not needed (Python single-threaded agent loop) |
| Summary quality | Refusal detection only | + vacuous summary detection (< 5% of input tokens) |
| Iterative updates | No | Merge adjacent summaries into higher-level DAG nodes |

---

## Configuration

Add to `~/.hermes/config.yaml`:

```yaml
lcm:
  enabled: false
  tau_soft: 0.50
  tau_hard: 0.85
  deterministic_target: 512
  protect_last_n: 4
  summary_model: ""
```

When `lcm.enabled = true`, the `compression` section is ignored.

---

## Migration Strategy

### Phase 1: Ship behind flag (old compressor untouched)
- Add `agent/lcm/` modules
- Add conditional LCM path at compression trigger points
- Register tools only when LCM active
- **Default: off**

### Phase 2: Dual-path testing
- Integration tests with both paths
- Session reload verification
- Tool expansion verification

### Phase 3: Default to LCM
- Change default to `lcm.enabled: true`
- Keep old compressor as fallback

### Phase 4: Remove old compressor (future)
- After LCM stable for several releases

---

## Testing Strategy

### Unit Tests (all pure, no IO)

```
tests/agent/lcm/
├── test_dag.py          # SummaryDag operations (6 tests)
├── test_store.py        # ImmutableStore (5 tests)
├── test_engine.py       # LcmEngine core (15 tests)
├── test_escalation.py   # Escalation + deterministic truncation (6 tests)
├── test_refusal.py      # Refusal detection (3 tests)
├── test_config.py       # Config loading (2 tests)
├── test_rebuild.py      # Session reconstruction (5 tests)
└── test_tools.py        # Agent tools: pin, forget, search, focus, budget, toc (14 tests)
```

### Integration Tests

```
tests/test_lcm_integration.py
├── test_lcm_compaction_in_agent_loop
├── test_lcm_expand_tool_returns_originals
├── test_lcm_session_reload
├── test_lcm_disabled_uses_old_compressor
├── test_lcm_context_pressure_warnings
├── test_lcm_pin_survives_compaction
├── test_lcm_forget_reclaims_tokens
└── test_lcm_search_finds_compacted_messages
```

---

## Implementation Order

| Step | Files | Tests | Depends On |
|------|-------|-------|-----------|
| 1. Data structures | dag.py, store.py, config.py | test_dag, test_store, test_config | Nothing |
| 2. Refusal + deterministic truncation | refusal.py, escalation.py (partial) | test_refusal, test_escalation | Nothing |
| 3. Engine core | engine.py | test_engine | Steps 1, 2 |
| 4. Escalation with LLM | escalation.py (full) | test_escalation (mock LLM) | Steps 2, 3 |
| 5. Session rebuild | engine.py (rebuild_from_session) | test_rebuild | Steps 1, 3 |
| 6. Agent tools (all 7) | expand_tool.py, tools.py | test_tools | Step 3 |
| 7. Config integration | hermes_cli/config.py | config loading test | Step 1 |
| 8. Agent loop integration | run_agent.py | test_lcm_integration | Steps 3-7 |
| 9. Session persistence | run_agent.py | end-to-end test | Steps 5, 8 |
| 10. Polish + docs | startup logs, pressure display | manual verification | All |

---

## Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| Summary larger than original | Check summary_tokens < block_tokens before accepting |
| Summary LLM call fails | Three-level escalation, Level 3 always converges |
| Tool-call/result pair split | Boundary alignment + unit tests |
| Corrupted LCM metadata on reload | Defensive parsing, fallback to raw-messages mode |
| lcm_expand floods context | Cap at configurable token limit, truncate with note |
| Pin abuse fills context | Cap max pinned messages, warn agent when approaching limit |
| Search performance on large stores | Index on ingest, keyword-based (not embedding) for v1 |
