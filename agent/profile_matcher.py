"""Match hardware to recommended local model profiles."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).parent.parent / "local_profiles"


def load_profiles() -> list[dict]:
    """Load all hardware profile YAML files."""
    import yaml

    profiles = []
    if not PROFILES_DIR.exists():
        logger.warning(f"Profiles directory not found: {PROFILES_DIR}")
        return profiles

    for f in sorted(PROFILES_DIR.glob("*.yaml")):
        try:
            with open(f) as fh:
                profile = yaml.safe_load(fh)
                profile["_file"] = f.name
                profiles.append(profile)
        except Exception as e:
            logger.warning(f"Failed to load profile {f}: {e}")

    return profiles


def match_profile(
    gpu_vram_mb: int = 0,
    unified_memory_mb: int = 0,
    has_gpu: bool = False,
    gpu_type: str = "none",
) -> Optional[dict]:
    """Find the best matching hardware profile.

    Args:
        gpu_vram_mb: NVIDIA VRAM in MB (0 if no NVIDIA GPU)
        unified_memory_mb: Apple Silicon unified memory in MB (0 if not Apple)
        has_gpu: Whether any GPU was detected
        gpu_type: "nvidia" | "apple" | "none"
    """
    profiles = load_profiles()

    if gpu_type == "nvidia" and gpu_vram_mb > 0:
        # Match NVIDIA GPU by VRAM range
        for p in profiles:
            vram_range = p.get("gpu_vram_range")
            if vram_range and vram_range[0] <= gpu_vram_mb <= vram_range[1]:
                return p

    elif gpu_type == "apple" and unified_memory_mb > 0:
        # Match Apple Silicon by unified memory range
        for p in profiles:
            mem_range = p.get("unified_memory_range")
            if mem_range and mem_range[0] <= unified_memory_mb <= mem_range[1]:
                return p

    # Fallback to CPU-only profile
    for p in profiles:
        if "cpu" in p.get("_file", "").lower():
            return p

    return None


def match_profile_from_hardware() -> Optional[dict]:
    """Auto-detect hardware and return the best matching profile."""
    from agent.hardware import detect_hardware

    hw = detect_hardware()

    if hw.gpu:
        return match_profile(
            gpu_vram_mb=hw.gpu.vram_total_mb,
            unified_memory_mb=hw.gpu.vram_total_mb if hw.gpu.gpu_type == "apple" else 0,
            has_gpu=True,
            gpu_type=hw.gpu.gpu_type,
        )

    return match_profile(has_gpu=False, gpu_type="none")


def format_profile_recommendation(profile: dict) -> str:
    """Format a profile as a human-readable recommendation."""
    lines = [f"\n  Recommended Profile: {profile.get('name', 'Unknown')}"]
    lines.append(f"   {profile.get('description', '')}\n")

    models = profile.get("recommended_models", {})

    if models.get("main"):
        lines.append("   Main models (pick one):")
        for m in models["main"]:
            ctx = f" — {m.get('context', '?')} ctx" if m.get("context") else ""
            lines.append(f"     * {m['name']}{ctx}")
            if m.get("description"):
                lines.append(f"       {m['description']}")

    if models.get("auxiliary"):
        lines.append("\n   Auxiliary models:")
        for m in models["auxiliary"]:
            lines.append(f"     * {m['name']}")

    if models.get("vision"):
        lines.append("\n   Vision models:")
        for m in models["vision"]:
            lines.append(f"     * {m['name']}")

    search = profile.get("search", {})
    if search.get("setup"):
        lines.append(f"\n   Web search: {search.get('setup', '')}")

    server = profile.get("server", {})
    if server.get("recommended"):
        lines.append(f"\n   Recommended server: {server['recommended']}")

    notes = profile.get("notes")
    if notes:
        lines.append(f"\n   Note: {notes.strip()}")

    return "\n".join(lines)
