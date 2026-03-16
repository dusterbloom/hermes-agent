"""Model capabilities registry.

Pattern-matches model names to capability structs so the rest of the agent
can make routing decisions (tool calling, vision, strict alternation, etc.)
without hard-coding model names throughout the codebase.

Only stdlib imports — no external dependencies.
"""

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ModelSizeClass(Enum):
    SMALL = "small"    # ≤3B params
    MEDIUM = "medium"  # 7–14B params
    LARGE = "large"    # 30B+ params


class ReaderTier(Enum):
    MINIMAL = "minimal"    # very short context, minimal reasoning
    STANDARD = "standard"  # normal capabilities
    ADVANCED = "advanced"  # strong reasoning, large context


@dataclass
class ModelCapabilities:
    size_class: ModelSizeClass = ModelSizeClass.MEDIUM
    tool_calling: bool = False
    vision: bool = False
    thinking: bool = False
    strict_alternation: bool = True  # most local models need this
    max_output_tokens: int = 4096
    reader_tier: ReaderTier = ReaderTier.STANDARD
    context_length: Optional[int] = None  # None = use server-reported or probed value


# ---------------------------------------------------------------------------
# Built-in pattern table
# Most-specific patterns first — first match wins.
# ---------------------------------------------------------------------------

def _cap(
    size_class: ModelSizeClass,
    tool_calling: bool,
    vision: bool,
    thinking: bool,
    strict_alternation: bool,
    max_output_tokens: int,
    reader_tier: ReaderTier,
) -> ModelCapabilities:
    """Shorthand constructor used only in the table below."""
    return ModelCapabilities(
        size_class=size_class,
        tool_calling=tool_calling,
        vision=vision,
        thinking=thinking,
        strict_alternation=strict_alternation,
        max_output_tokens=max_output_tokens,
        reader_tier=reader_tier,
    )


_L = ModelSizeClass.LARGE
_M = ModelSizeClass.MEDIUM
_S = ModelSizeClass.SMALL
_ADV = ReaderTier.ADVANCED
_STD = ReaderTier.STANDARD
_MIN = ReaderTier.MINIMAL

BUILT_IN_PATTERNS: list[tuple[str, ModelCapabilities]] = [
    # Cloud models — no strict alternation required
    ("claude",          _cap(_L, True,  True,  True,  False, 16384, _ADV)),
    ("gpt-4",           _cap(_L, True,  True,  True,  False, 16384, _ADV)),
    ("gpt-3.5",         _cap(_M, True,  False, False, False,  4096, _STD)),
    ("gemini",          _cap(_L, True,  True,  True,  False,  8192, _ADV)),

    # Qwen — more specific patterns first
    ("qwen3",           _cap(_L, True,  False, True,  True,   8192, _ADV)),
    ("qwen2.5-coder",   _cap(_L, True,  False, False, True,   8192, _ADV)),
    ("qwen2.5",         _cap(_L, True,  False, False, True,   8192, _ADV)),
    ("qwen-vl",         _cap(_L, True,  True,  False, True,   4096, _STD)),

    # Hermes (NousResearch)
    ("hermes-3",        _cap(_L, True,  False, False, True,   8192, _ADV)),
    ("hermes-2",        _cap(_M, True,  False, False, True,   4096, _STD)),

    # Llama — most specific first
    ("llama-4",         _cap(_L, True,  True,  True,  True,   8192, _ADV)),
    ("llama-3.3",       _cap(_L, True,  False, False, True,   8192, _ADV)),
    ("llama-3.2",       _cap(_M, True,  False, False, True,   4096, _STD)),
    ("llama-3.1",       _cap(_L, True,  False, False, True,   4096, _STD)),
    ("llama-3",         _cap(_M, False, False, False, True,   4096, _STD)),

    # Vision-focused models — before generic family names that may appear as substrings
    ("llava",           _cap(_M, False, True,  False, True,   4096, _STD)),
    ("bakllava",        _cap(_M, False, True,  False, True,   4096, _STD)),

    # Mistral / Mixtral — most specific first
    ("mistral-large",   _cap(_L, True,  False, False, True,   8192, _ADV)),
    ("mixtral",         _cap(_L, True,  False, False, True,   4096, _STD)),
    ("mistral",         _cap(_M, True,  False, False, True,   4096, _STD)),

    # DeepSeek — most specific first
    ("deepseek-r1",     _cap(_L, False, False, True,  True,   8192, _ADV)),
    ("deepseek-v3",     _cap(_L, True,  False, False, True,   8192, _ADV)),
    ("deepseek-coder",  _cap(_L, True,  False, False, True,   8192, _ADV)),
    ("deepseek",        _cap(_M, False, False, False, True,   4096, _STD)),

    # Phi — most specific first
    ("phi-4",           _cap(_M, True,  False, False, True,   4096, _STD)),
    ("phi-3.5",         _cap(_S, False, False, False, True,   2048, _MIN)),
    ("phi-3",           _cap(_S, False, False, False, True,   2048, _MIN)),

    # Gemma — most specific first
    ("gemma-2",         _cap(_M, False, False, False, True,   4096, _STD)),
    ("gemma",           _cap(_S, False, False, False, True,   2048, _MIN)),

    # Nemotron
    ("nemotron",        _cap(_L, True,  False, True,  True,   8192, _ADV)),

]


# ---------------------------------------------------------------------------
# Size-marker detection
# ---------------------------------------------------------------------------


def _has_size_marker(name: str, marker: str) -> bool:
    """Return True if *name* contains *marker* as a standalone size token.

    Prevents false positives:
    - ``'35b'`` must NOT match ``'3b'`` (digit before the marker)
    - ``'a3b'`` must NOT match ``'3b'`` (letter before the marker, MoE active-param suffix)
    """
    idx = name.find(marker)
    if idx < 0:
        return False
    if idx > 0:
        prev = name[idx - 1]
        if prev.isalnum():
            return False
    return True


def _apply_size_markers(name: str, caps: ModelCapabilities) -> ModelCapabilities:
    """Adjust *caps.size_class* based on explicit param-count suffixes in *name*."""
    small_markers = ("0.5b", "1b", "1.5b", "2b", "3b", "4b")
    medium_markers = ("7b", "8b", "9b", "13b", "14b")
    large_markers = ("32b", "34b", "70b", "72b")

    for marker in small_markers:
        if _has_size_marker(name, marker):
            caps.size_class = ModelSizeClass.SMALL
            return caps

    for marker in medium_markers:
        if _has_size_marker(name, marker):
            caps.size_class = ModelSizeClass.MEDIUM
            return caps

    for marker in large_markers:
        if _has_size_marker(name, marker):
            caps.size_class = ModelSizeClass.LARGE
            return caps

    return caps


# ---------------------------------------------------------------------------
# Override application
# ---------------------------------------------------------------------------


def _apply_overrides(caps: ModelCapabilities, overrides: dict) -> ModelCapabilities:
    """Return a copy of *caps* with fields from *overrides* applied.

    Only known fields (attributes present on ModelCapabilities) are applied;
    unknown keys are silently ignored.
    """
    result = copy.copy(caps)
    for key, value in overrides.items():
        if hasattr(result, key):
            setattr(result, key, value)
    return result


# ---------------------------------------------------------------------------
# Main lookup
# ---------------------------------------------------------------------------


def lookup(model_name: str, user_overrides: dict | None = None) -> ModelCapabilities:
    """Look up capabilities for a model by name.

    Args:
        model_name: The model name/ID, e.g. ``"qwen2.5-coder:32b-instruct-q4_K_M"``
                    or ``"ollama/llama-3.1:8b"``.
        user_overrides: Optional dict mapping model-name substrings to dicts of
                        capability fields to override.  First matching key wins.

    Returns:
        A :class:`ModelCapabilities` instance with best-guess values.
    """
    name = model_name.lower()

    # Strip provider prefix, e.g. "ollama/qwen2.5-coder" -> "qwen2.5-coder"
    if "/" in name:
        name = name.split("/", 1)[1]

    # Pattern match — first hit wins
    caps = ModelCapabilities()  # default fallback
    for pattern, pattern_caps in BUILT_IN_PATTERNS:
        if pattern in name:
            caps = copy.copy(pattern_caps)
            break

    # Refine size class from explicit param-count suffixes
    caps = _apply_size_markers(name, caps)

    # Apply user overrides — first matching key wins
    if user_overrides:
        for pattern, override_dict in user_overrides.items():
            if pattern.lower() in name:
                caps = _apply_overrides(caps, override_dict)
                break

    return caps


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def supports_native_tool_calling(
    model_name: str, user_overrides: dict | None = None
) -> bool:
    """Return True if the model supports native function/tool calling."""
    return lookup(model_name, user_overrides).tool_calling


def supports_vision(model_name: str, user_overrides: dict | None = None) -> bool:
    """Return True if the model supports vision/image input."""
    return lookup(model_name, user_overrides).vision


def needs_strict_alternation(
    model_name: str, user_overrides: dict | None = None
) -> bool:
    """Return True if the model requires strict user/assistant turn alternation."""
    return lookup(model_name, user_overrides).strict_alternation
