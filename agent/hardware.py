"""Hardware detection module.

Detects GPU, CPU, and RAM capabilities of the host machine.  All results are
cached at the module level for the lifetime of the process so repeated calls
are cheap.

Only stdlib is used (subprocess, platform, os, pathlib, logging, dataclasses,
typing).  No external dependencies.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GPUInfo:
    name: str           # e.g. "NVIDIA GeForce RTX 3090"
    vram_total_mb: int  # e.g. 24576
    vram_free_mb: int   # e.g. 20480
    vram_used_mb: int   # e.g. 4096
    gpu_type: str = "nvidia"  # nvidia | apple | none


@dataclass
class HardwareInfo:
    gpu: Optional[GPUInfo]  # None if no GPU detected
    cpu_cores: int           # total CPU cores
    ram_total_mb: int        # total system RAM in MB
    ram_free_mb: int         # available system RAM in MB
    platform: str            # linux | darwin | windows
    is_wsl: bool = False     # running under WSL2


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_cached_info: Optional[HardwareInfo] = None


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def is_wsl() -> bool:
    """Return True when running inside WSL (any version)."""
    try:
        with open("/proc/version", "r") as fh:
            content = fh.read().lower()
        return "microsoft" in content or "wsl" in content
    except OSError:
        return False


def detect_nvidia_gpu() -> Optional[GPUInfo]:
    """Detect NVIDIA GPU via nvidia-smi.

    Returns the first GPU found, or None if nvidia-smi is unavailable or
    returns a non-zero exit code.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        line = result.stdout.strip().split("\n")[0]  # first GPU only
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return None
        return GPUInfo(
            name=parts[0],
            vram_total_mb=int(float(parts[1])),
            vram_free_mb=int(float(parts[2])),
            vram_used_mb=int(float(parts[3])),
            gpu_type="nvidia",
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as exc:
        logger.debug("nvidia-smi not available: %s", exc)
        return None


def detect_apple_gpu() -> Optional[GPUInfo]:
    """Detect Apple Silicon GPU (unified memory).

    Returns None on any non-Darwin platform or on detection failure.
    """
    if platform.system() != "Darwin":
        return None
    try:
        # Total physical memory
        mem_result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if mem_result.returncode != 0:
            return None
        total_bytes = int(mem_result.stdout.strip())
        total_mb = total_bytes // (1024 * 1024)

        # Chip brand name
        chip_result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        chip_name = (
            chip_result.stdout.strip()
            if chip_result.returncode == 0
            else "Apple Silicon"
        )

        # Apple Silicon GPU shares unified memory.  Estimate ~75% available
        # for ML workloads; the rest is consumed by the OS and running apps.
        free_mb = int(total_mb * 0.75)
        used_mb = total_mb - free_mb

        return GPUInfo(
            name=chip_name,
            vram_total_mb=total_mb,
            vram_free_mb=free_mb,
            vram_used_mb=used_mb,
            gpu_type="apple",
        )
    except Exception as exc:
        logger.debug("Apple Silicon detection failed: %s", exc)
        return None


def detect_system_info() -> Tuple[int, int, int]:
    """Return (cpu_cores, ram_total_mb, ram_free_mb).

    Uses platform-appropriate methods:
    - Linux: /proc/meminfo
    - macOS: sysctl + vm_stat
    - Windows/fallback: os.cpu_count() only; RAM defaults to 0
    """
    cpu_cores = os.cpu_count() or 1
    system = platform.system()

    if system == "Linux":
        return _detect_linux_memory(cpu_cores)
    if system == "Darwin":
        return _detect_darwin_memory(cpu_cores)
    return _detect_fallback_memory(cpu_cores)


def _detect_linux_memory(cpu_cores: int) -> Tuple[int, int, int]:
    """Parse /proc/meminfo for total and available RAM (in MB)."""
    try:
        with open("/proc/meminfo", "r") as fh:
            lines = fh.readlines()
        mem = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                mem[key] = int(parts[1])  # values are in kB
        total_mb = mem.get("MemTotal", 0) // 1024
        # MemAvailable is the most accurate "free for applications" figure
        free_mb = mem.get("MemAvailable", mem.get("MemFree", 0)) // 1024
        return cpu_cores, total_mb, free_mb
    except Exception as exc:
        logger.debug("Could not read /proc/meminfo: %s", exc)
        return cpu_cores, 0, 0


def _detect_darwin_memory(cpu_cores: int) -> Tuple[int, int, int]:
    """Use sysctl + vm_stat to retrieve RAM totals on macOS."""
    total_mb = 0
    free_mb = 0
    try:
        mem_result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if mem_result.returncode == 0:
            total_bytes = int(mem_result.stdout.strip())
            total_mb = total_bytes // (1024 * 1024)
    except Exception as exc:
        logger.debug("sysctl hw.memsize failed: %s", exc)

    try:
        vm_result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if vm_result.returncode == 0:
            free_mb = _parse_vm_stat(vm_result.stdout)
    except Exception as exc:
        logger.debug("vm_stat failed: %s", exc)

    return cpu_cores, total_mb, free_mb


def _parse_vm_stat(output: str) -> int:
    """Parse vm_stat output and estimate free RAM in MB.

    Counts pages that are free, inactive, or speculative — these are pages
    that the OS can reclaim immediately for new allocations.
    """
    PAGE_SIZE = 4096  # standard macOS page size in bytes
    free_pages = 0
    for line in output.splitlines():
        lower = line.lower()
        if any(k in lower for k in ("pages free", "pages inactive", "pages speculative")):
            # Lines look like: "Pages free:                               1234."
            parts = line.split(":")
            if len(parts) == 2:
                try:
                    free_pages += int(parts[1].strip().rstrip("."))
                except ValueError:
                    pass
    return (free_pages * PAGE_SIZE) // (1024 * 1024)


def _detect_fallback_memory(cpu_cores: int) -> Tuple[int, int, int]:
    """Fallback memory detection: try psutil, otherwise return zeros."""
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        total_mb = vm.total // (1024 * 1024)
        free_mb = vm.available // (1024 * 1024)
        return cpu_cores, total_mb, free_mb
    except ImportError:
        logger.debug("psutil not available; RAM info unavailable on this platform")
        return cpu_cores, 0, 0


def detect_wsl_host_ip() -> Optional[str]:
    """Return the Windows host IP as seen from WSL2.

    In WSL2, /etc/resolv.conf contains the Windows host as the DNS nameserver.
    This is useful for connecting to services running on the Windows host
    (e.g. LM Studio).

    Returns None if the file is missing or contains no nameserver line.
    """
    try:
        with open("/etc/resolv.conf", "r") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith("nameserver"):
                    parts = stripped.split()
                    if len(parts) >= 2:
                        return parts[1]
        return None
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def detect_hardware(force_refresh: bool = False) -> HardwareInfo:
    """Detect hardware capabilities.  Results are cached for the session lifetime.

    Args:
        force_refresh: When True, ignore the cached result and re-detect.

    Returns:
        A populated HardwareInfo dataclass.
    """
    global _cached_info
    if _cached_info is not None and not force_refresh:
        return _cached_info

    system = platform.system()
    if system == "Linux":
        plat = "linux"
    elif system == "Darwin":
        plat = "darwin"
    else:
        plat = "windows"

    wsl = is_wsl()

    gpu = detect_nvidia_gpu()
    if gpu is None:
        gpu = detect_apple_gpu()

    cpu_cores, ram_total, ram_free = detect_system_info()

    _cached_info = HardwareInfo(
        gpu=gpu,
        cpu_cores=cpu_cores,
        ram_total_mb=ram_total,
        ram_free_mb=ram_free,
        platform=plat,
        is_wsl=wsl,
    )
    return _cached_info


def get_available_vram_mb() -> int:
    """Return free VRAM in MB, or 0 if no GPU is detected."""
    info = detect_hardware()
    if info.gpu is None:
        return 0
    return info.gpu.vram_free_mb


def format_hardware_summary(info: Optional[HardwareInfo] = None) -> str:
    """Return a human-readable hardware summary string."""
    if info is None:
        info = detect_hardware()
    lines = []
    if info.gpu:
        lines.append(
            f"GPU: {info.gpu.name} ({info.gpu.vram_total_mb}MB VRAM, {info.gpu.vram_free_mb}MB free)"
        )
    else:
        lines.append("GPU: None detected")
    lines.append(f"CPU: {info.cpu_cores} cores")
    lines.append(f"RAM: {info.ram_total_mb}MB total, {info.ram_free_mb}MB free")
    platform_label = info.platform + (" (WSL2)" if info.is_wsl else "")
    lines.append(f"Platform: {platform_label}")
    return "\n".join(lines)
