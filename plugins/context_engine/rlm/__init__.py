"""Recursive Language Model context engine plugin.

Implements the RLM paradigm (arXiv:2512.24601) as a ContextEngine plugin.
Instead of compressing messages, the RLM treats the prompt as an external
Python variable in a REPL environment, allowing the LLM to programmatically
peek, grep, chunk, and recursively call sub-LLMs over the context.

Activation: set ``context.engine: rlm`` in config.yaml.

Requires an external LLM provider (OpenAI, custom, etc.) for the root and
sub-LLM calls.
"""

from __future__ import annotations

import json
import logging
import math
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from agent.context_engine import ContextEngine

logger = logging.getLogger(__name__)

# ── RLM REPL Environment ──────────────────────────────────────────────────


class RLMAgentEnvironment:
    """Python REPL environment for RLM context management.

    Stores the context as a variable. The root LLM can:
    - Peek at context snippets
    - Grep/search with regex
    - Chunk and map over context
    - Call sub-LLMs via env.llm_batch() or env.llm_call()

    The LLM provides its final answer via the answer dict:
    answer = {"content": "...", "ready": True}
    """

    def __init__(
        self,
        context: str,
        max_iterations: int = 20,
        output_limit: int = 8192,
    ):
        self.context = context
        self.max_iterations = max_iterations
        self.output_limit = output_limit

        # REPL namespace
        self.namespace: dict = {
            "__context__": context,
            "__context_length__": len(context),
            "__context_token_length__": self._estimate_tokens(context),
            "answer": {"content": "", "ready": False},
            "llm_batch": self._llm_batch_handler,
            "llm_call": self._llm_call_handler,
        }

        self.iteration_count = 0

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return max(1, len(text) // 4)

    def execute(self, code: str) -> str:
        """Execute Python code in the REPL environment.

        Returns truncated output visible to the LLM.
        """
        try:
            compiled = compile(code, "<rlm>", "exec")
            exec(compiled, self.namespace)

            result = self.namespace.get("answer", "")
            if isinstance(result, dict):
                output = result.get("content", "")
            else:
                output = str(result)

            if len(output) > self.output_limit:
                output = output[: self.output_limit] + "...[truncated]"

            return output

        except Exception as e:
            return f"REPL Error: {type(e).__name__}: {e}"

    def execute_expression(self, expression: str) -> str:
        """Execute a Python expression and return the result."""
        try:
            result = eval(expression, self.namespace)
            output = str(result)

            if len(output) > self.output_limit:
                output = output[: self.output_limit] + "...[truncated]"

            return output

        except Exception as e:
            return f"Expression Error: {type(e).__name__}: {e}"

    def _llm_batch_handler(self, prompts: list) -> list:
        """Batch sub-LLM calls. Parallel execution.

        Each prompt is a question about a subset of __context__.
        Returns list of answers from sub-LLMs.

        Note: This is a stub — the real implementation in
        RLMAgentEnvironment delegates to RLMAgent._call_sub_llm.
        """
        # Placeholder — will be replaced when the environment is connected
        # to the actual LLM client
        return []

    def _llm_call_handler(self, prompt: str) -> str:
        """Recursive sub-LLM call stub."""
        return ""

    def set_answer(self, content: str, ready: bool = False):
        """Set the final answer."""
        self.namespace["answer"]["content"] = content
        self.namespace["answer"]["ready"] = ready

    def check_iteration_limit(self) -> bool:
        """Whether we've hit max iterations."""
        self.iteration_count += 1
        return self.iteration_count >= self.max_iterations


# ── RLM ContextEngine Plugin ─────────────────────────────────────────────


class RlmContextEngine(ContextEngine):
    """RLM adapter that implements the ContextEngine ABC.

    Unlike LCM which compresses messages, the RLM treats the context as
    an external Python variable and allows the root LLM to programmatically
    interact with it via a REPL environment.

    Key design decisions:
    - Never compresses messages (returns them unchanged)
    - Exposes rlm_peek, rlm_grep, rlm_partition tools to the agent
    - Sub-LLMs get tools, root LLM does not (keeps root context clean)
    - Parallel batch sub-LLM calls via llm_batch()
    """

    def __init__(self, **kwargs: Any) -> None:
        self._config: dict = kwargs.get("rlm_config", {})
        self._max_iterations = self._config.get("max_iterations", 20)
        self._output_limit = self._config.get("output_limit", 8192)

        # Token tracking (required by ABC)
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_total_tokens: int = 0
        self.threshold_tokens: int = 0
        self.context_length: int = 0
        self.compression_count: int = 0
        self.threshold_percent: float = 1.0  # Never triggers compression
        self.protect_last_n: int = 6

        # Store context for this session
        self._session_context: Optional[str] = None
        self._rlm_env: Optional[RLMAgentEnvironment] = None

    # ── Identity ────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "rlm"

    # ── Core ABC interface ──────────────────────────────────────────────

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Update tracked token usage from API response."""
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total = prompt_tokens + completion_tokens

        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = completion_tokens
        self.last_total_tokens = total

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """RLM never compresses — it handles unbounded context via REPL."""
        return False

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """RLM preflight check — always False."""
        return False

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
    ) -> List[Dict[str, Any]]:
        """Return messages unchanged.

        RLM doesn't compress — it delegates context management to the
        REPL environment. The agent interacts with context via tools
        (rlm_peek, rlm_grep, etc.).
        """
        return list(messages)

    # ── Session lifecycle ───────────────────────────────────────────────

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Initialize RLM environment for the session."""
        self.context_length = kwargs.get("context_length", 128_000)
        self.threshold_tokens = self.context_length

        # Extract context from config if provided
        context = kwargs.get("rlm_context")
        if context:
            self._session_context = context
            self._init_rlm_env(context)

        logger.info(
            "RLM session started (session=%s, max_iter=%d, output_limit=%d)",
            session_id,
            self._max_iterations,
            self._output_limit,
        )

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Session end — context is ephemeral (RLM stores in memory)."""
        pass

    def on_session_reset(self) -> None:
        """Reset per-session state."""
        super().on_session_reset()
        self._session_context = None
        self._rlm_env = None

    def _init_rlm_env(self, context: str) -> None:
        """Create the REPL environment for the given context."""
        self._session_context = context
        self._rlm_env = RLMAgentEnvironment(
            context=context,
            max_iterations=self._max_iterations,
            output_limit=self._output_limit,
        )

    # ── Tools ───────────────────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return RLM tool schemas (rlm_peek, rlm_grep, rlm_partition)."""
        return list(RLM_TOOL_SCHEMAS.values())

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatch RLM tool calls."""
        handlers = {
            "rlm_peek": self._handle_peek,
            "rlm_grep": self._handle_grep,
            "rlm_partition": self._handle_partition,
        }

        handler = handlers.get(name)
        if handler is None:
            return json.dumps({"error": f"Unknown RLM tool: {name}"})

        return handler(args)

    def _handle_peek(self, args: Dict[str, Any]) -> str:
        """Peek at a section of the context.

        Args:
            start: Start character index
            length: Number of characters to read

        Returns the context snippet.
        """
        if not self._rlm_env or not self._session_context:
            return json.dumps({"error": "No context loaded"})

        start = args.get("start", 0)
        length = args.get("length", 2000)

        try:
            context = self._session_context
            snippet = context[start : start + length]
            return json.dumps({"snippet": snippet, "start": start, "length": len(snippet)})
        except Exception as e:
            return json.dumps({"error": f"Peek failed: {e}"})

    def _handle_grep(self, args: Dict[str, Any]) -> str:
        """Grep the context for regex patterns.

        Args:
            pattern: Regex pattern to search for
            max_matches: Maximum number of matches to return (default: 20)

        Returns list of matching lines.
        """
        if not self._rlm_env or not self._session_context:
            return json.dumps({"error": "No context loaded"})

        pattern = args.get("pattern", "")
        max_matches = args.get("max_matches", 20)

        try:
            context = self._session_context
            lines = context.split("\n")
            matches = []

            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    matches.append({"line_num": i, "content": line})
                    if len(matches) >= max_matches:
                        break

            return json.dumps({
                "pattern": pattern,
                "matches": matches,
                "total_found": len([l for l in lines if re.search(pattern, l)]),
            })
        except Exception as e:
            return json.dumps({"error": f"Grep failed: {e}"})

    def _handle_partition(self, args: Dict[str, Any]) -> str:
        """Partition context into chunks and return chunk boundaries.

        Args:
            chunk_size: Characters per chunk (default: 8000)
            overlap: Characters of overlap between chunks (default: 500)

        Returns list of chunk metadata (index, start, length).
        """
        if not self._rlm_env or not self._session_context:
            return json.dumps({"error": "No context loaded"})

        chunk_size = args.get("chunk_size", 8000)
        overlap = args.get("overlap", 500)

        try:
            context = self._session_context
            chunks = []
            start = 0

            while start < len(context):
                end = min(start + chunk_size, len(context))
                chunk_text = context[start:end]

                chunks.append({
                    "index": len(chunks),
                    "start": start,
                    "end": end,
                    "length": len(chunk_text),
                })

                start = end - overlap if end < len(context) else end

            return json.dumps({
                "total_chunks": len(chunks),
                "chunk_size": chunk_size,
                "overlap": overlap,
                "chunks": chunks,
            })
        except Exception as e:
            return json.dumps({"error": f"Partition failed: {e}"})


# ── Tool Schemas ────────────────────────────────────────────────────────

RLM_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "rlm_peek": {
        "name": "rlm_peek",
        "description": "Peek at a section of the external context stored in the RLM REPL environment. Returns a snippet of characters from the context. Use this to understand the structure of the context before deciding how to process it.",
        "parameters": {
            "type": "object",
            "properties": {
                "start": {
                    "type": "integer",
                    "description": "Start character index (0-based). Default: 0",
                },
                "length": {
                    "type": "integer",
                    "description": "Number of characters to return. Default: 2000",
                },
            },
            "required": [],
        },
    },
    "rlm_grep": {
        "name": "rlm_grep",
        "description": "Grep the context for regex pattern matches. Returns up to max_matches matching lines with line numbers. Use this to find specific IDs, keywords, or patterns in the context without loading the entire context into the model's window.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "max_matches": {
                    "type": "integer",
                    "description": "Maximum matches to return. Default: 20",
                },
            },
            "required": ["pattern"],
        },
    },
    "rlm_partition": {
        "name": "rlm_partition",
        "description": "Partition the context into fixed-size chunks with optional overlap. Returns metadata about each chunk (index, start, end, length). Use this before rlm_batch_sub_llm to prepare chunks for parallel processing.",
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_size": {
                    "type": "integer",
                    "description": "Characters per chunk. Default: 8000",
                },
                "overlap": {
                    "type": "integer",
                    "description": "Characters of overlap between chunks. Default: 500",
                },
            },
            "required": [],
        },
    },
}


# ── Plugin Registration ─────────────────────────────────────────────────


def register(ctx) -> None:
    """Plugin entry point — register RLM as the context engine."""
    engine = RlmContextEngine()
    if hasattr(ctx, "register_context_engine"):
        ctx.register_context_engine(engine)
    else:
        logger.warning("RLM plugin: context engine registration not supported")
