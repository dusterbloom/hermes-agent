"""Tests for agent/hardware.py — hardware detection module.

All subprocess calls and file reads are mocked so tests are fully hermetic
and do not depend on the machine running the suite.
"""

import platform
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# Import the module under test.  It lives in agent/ which must be on sys.path.
# ---------------------------------------------------------------------------
sys.path.insert(0, "/home/peppi/Dev/hermes-agent")
sys.path.insert(0, "/home/peppi/Dev/hermes-agent/agent")

from agent.hardware import (
    GPUInfo,
    HardwareInfo,
    detect_apple_gpu,
    detect_hardware,
    detect_nvidia_gpu,
    detect_system_info,
    detect_wsl_host_ip,
    format_hardware_summary,
    get_available_vram_mb,
    is_wsl,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _proc_result(stdout: str, returncode: int = 0) -> MagicMock:
    """Build a fake subprocess.CompletedProcess."""
    m = MagicMock()
    m.stdout = stdout
    m.returncode = returncode
    return m


# ---------------------------------------------------------------------------
# is_wsl()
# ---------------------------------------------------------------------------

class TestIsWsl:
    def test_wsl2_kernel_string(self):
        wsl_version = "Linux version 5.15.167.4-microsoft-standard-WSL2"
        with patch("builtins.open", mock_open(read_data=wsl_version)):
            assert is_wsl() is True

    def test_wsl_lowercase(self):
        content = "Linux version 4.19.128-microsoft-standard"
        with patch("builtins.open", mock_open(read_data=content)):
            assert is_wsl() is True

    def test_plain_linux(self):
        content = "Linux version 6.1.0-20-amd64 (debian-kernel)"
        with patch("builtins.open", mock_open(read_data=content)):
            assert is_wsl() is False

    def test_missing_proc_version(self):
        with patch("builtins.open", side_effect=OSError("no such file")):
            assert is_wsl() is False


# ---------------------------------------------------------------------------
# detect_nvidia_gpu()
# ---------------------------------------------------------------------------

class TestDetectNvidiaGpu:
    _NVIDIA_OUTPUT = "NVIDIA GeForce RTX 3090, 24576, 20480, 4096\n"

    def test_happy_path(self):
        with patch("subprocess.run", return_value=_proc_result(self._NVIDIA_OUTPUT)):
            info = detect_nvidia_gpu()
        assert info is not None
        assert info.name == "NVIDIA GeForce RTX 3090"
        assert info.vram_total_mb == 24576
        assert info.vram_free_mb == 20480
        assert info.vram_used_mb == 4096
        assert info.gpu_type == "nvidia"

    def test_nonzero_returncode_returns_none(self):
        with patch("subprocess.run", return_value=_proc_result("", returncode=1)):
            assert detect_nvidia_gpu() is None

    def test_nvidia_smi_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert detect_nvidia_gpu() is None

    def test_timeout_returns_none(self):
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)):
            assert detect_nvidia_gpu() is None

    def test_malformed_output_returns_none(self):
        with patch("subprocess.run", return_value=_proc_result("bad, output\n")):
            assert detect_nvidia_gpu() is None

    def test_multiple_gpus_uses_first(self):
        multi = "Tesla T4, 15360, 14000, 1360\nTesla T4, 15360, 13000, 2360\n"
        with patch("subprocess.run", return_value=_proc_result(multi)):
            info = detect_nvidia_gpu()
        assert info is not None
        assert info.vram_total_mb == 15360
        assert info.vram_free_mb == 14000

    def test_float_values_truncated(self):
        # nvidia-smi can occasionally emit floats
        output = "RTX 4090, 24564.0, 22000.5, 2563.5\n"
        with patch("subprocess.run", return_value=_proc_result(output)):
            info = detect_nvidia_gpu()
        assert info is not None
        assert info.vram_total_mb == 24564
        assert info.vram_free_mb == 22000
        assert info.vram_used_mb == 2563


# ---------------------------------------------------------------------------
# detect_apple_gpu()
# ---------------------------------------------------------------------------

class TestDetectAppleGpu:
    def test_non_darwin_returns_none(self):
        with patch("platform.system", return_value="Linux"):
            assert detect_apple_gpu() is None

    def test_happy_path(self):
        total_bytes = 16 * 1024 * 1024 * 1024  # 16 GB
        total_mb = total_bytes // (1024 * 1024)  # 16384

        def fake_run(cmd, **kwargs):
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "hw.memsize" in cmd_str:
                return _proc_result(str(total_bytes))
            if "brand_string" in cmd_str:
                return _proc_result("Apple M2 Pro")
            return _proc_result("", returncode=1)

        with patch("platform.system", return_value="Darwin"), \
             patch("subprocess.run", side_effect=fake_run):
            info = detect_apple_gpu()

        assert info is not None
        assert info.name == "Apple M2 Pro"
        assert info.vram_total_mb == total_mb
        assert info.gpu_type == "apple"
        # free should be 75% of total
        assert info.vram_free_mb == int(total_mb * 0.75)
        assert info.vram_used_mb == total_mb - info.vram_free_mb

    def test_memsize_fail_returns_none(self):
        with patch("platform.system", return_value="Darwin"), \
             patch("subprocess.run", return_value=_proc_result("", returncode=1)):
            assert detect_apple_gpu() is None

    def test_chip_name_fallback(self):
        total_bytes = 8 * 1024 * 1024 * 1024

        def fake_run(cmd, **kwargs):
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "hw.memsize" in cmd_str:
                return _proc_result(str(total_bytes))
            # brand_string command fails
            return _proc_result("", returncode=1)

        with patch("platform.system", return_value="Darwin"), \
             patch("subprocess.run", side_effect=fake_run):
            info = detect_apple_gpu()

        assert info is not None
        assert info.name == "Apple Silicon"


# ---------------------------------------------------------------------------
# detect_system_info()
# ---------------------------------------------------------------------------

PROC_MEMINFO = """\
MemTotal:       16384000 kB
MemFree:         2048000 kB
MemAvailable:    8192000 kB
Buffers:          512000 kB
Cached:          4096000 kB
"""


class TestDetectSystemInfo:
    def test_linux_proc_meminfo(self):
        with patch("platform.system", return_value="Linux"), \
             patch("builtins.open", mock_open(read_data=PROC_MEMINFO)), \
             patch("os.cpu_count", return_value=8):
            cores, total, free = detect_system_info()

        assert cores == 8
        # MemTotal 16384000 kB = 16000 MB
        assert total == 16000
        # MemAvailable 8192000 kB = 8000 MB
        assert free == 8000

    def test_cpu_count_none_defaults_to_one(self):
        with patch("platform.system", return_value="Linux"), \
             patch("builtins.open", mock_open(read_data=PROC_MEMINFO)), \
             patch("os.cpu_count", return_value=None):
            cores, _, _ = detect_system_info()
        assert cores == 1

    def test_darwin_sysctl(self):
        total_bytes = 8 * 1024 * 1024 * 1024  # 8 GB

        def fake_run(cmd, **kwargs):
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "hw.memsize" in cmd_str:
                return _proc_result(str(total_bytes))
            if "vm_stat" in cmd_str:
                # pages free, pages inactive, etc. — simplified
                pages_free = 100000
                return _proc_result(
                    f"Pages free: {pages_free}.\nPages inactive: 50000.\nPages speculative: 10000.\n"
                )
            return _proc_result("", returncode=1)

        with patch("platform.system", return_value="Darwin"), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("os.cpu_count", return_value=10):
            cores, total, free = detect_system_info()

        assert cores == 10
        assert total == total_bytes // (1024 * 1024)
        # free should be reasonable (> 0)
        assert free > 0

    def test_windows_fallback(self):
        """On Windows (or unknown), fall back to os/psutil or defaults."""
        with patch("platform.system", return_value="Windows"), \
             patch("os.cpu_count", return_value=4):
            cores, total, free = detect_system_info()
        assert cores == 4
        # We can't know exact values but they must be positive
        assert total > 0
        assert free >= 0


# ---------------------------------------------------------------------------
# detect_wsl_host_ip()
# ---------------------------------------------------------------------------

RESOLV_CONF = """\
# This file was automatically generated by WSL.
nameserver 172.31.48.1
options timeout:2 attempts:5
"""


class TestDetectWslHostIp:
    def test_parses_nameserver(self):
        with patch("builtins.open", mock_open(read_data=RESOLV_CONF)):
            ip = detect_wsl_host_ip()
        assert ip == "172.31.48.1"

    def test_missing_file_returns_none(self):
        with patch("builtins.open", side_effect=OSError):
            assert detect_wsl_host_ip() is None

    def test_no_nameserver_line(self):
        with patch("builtins.open", mock_open(read_data="# empty\n")):
            assert detect_wsl_host_ip() is None


# ---------------------------------------------------------------------------
# detect_hardware() — caching behaviour
# ---------------------------------------------------------------------------

class TestDetectHardwareCaching:
    def _make_patchers(self, gpu=None):
        """Return a context manager that stubs out all detection helpers."""
        import contextlib

        @contextlib.contextmanager
        def ctx():
            with patch("agent.hardware.detect_nvidia_gpu", return_value=gpu), \
                 patch("agent.hardware.detect_apple_gpu", return_value=None), \
                 patch("agent.hardware.detect_system_info", return_value=(4, 8192, 4096)), \
                 patch("agent.hardware.is_wsl", return_value=False), \
                 patch("platform.system", return_value="Linux"):
                yield

        return ctx()

    def setup_method(self):
        # Clear module-level cache before each test.
        import agent.hardware as hw
        hw._cached_info = None

    def test_returns_hardware_info(self):
        with self._make_patchers():
            info = detect_hardware()
        assert isinstance(info, HardwareInfo)
        assert info.cpu_cores == 4
        assert info.ram_total_mb == 8192
        assert info.ram_free_mb == 4096
        assert info.platform == "linux"
        assert info.is_wsl is False

    def test_result_is_cached(self):
        with self._make_patchers():
            first = detect_hardware()
            second = detect_hardware()
        assert first is second

    def test_force_refresh_bypasses_cache(self):
        with self._make_patchers():
            first = detect_hardware()
        # Change what detect_system_info returns and force refresh
        with patch("agent.hardware.detect_nvidia_gpu", return_value=None), \
             patch("agent.hardware.detect_apple_gpu", return_value=None), \
             patch("agent.hardware.detect_system_info", return_value=(8, 16384, 8000)), \
             patch("agent.hardware.is_wsl", return_value=True), \
             patch("platform.system", return_value="Linux"):
            second = detect_hardware(force_refresh=True)
        assert second is not first
        assert second.cpu_cores == 8

    def test_gpu_stored_in_info(self):
        gpu = GPUInfo("RTX 3090", 24576, 20480, 4096, "nvidia")
        with self._make_patchers(gpu=gpu):
            info = detect_hardware()
        assert info.gpu is gpu

    def test_no_gpu_stored_as_none(self):
        with self._make_patchers(gpu=None):
            info = detect_hardware()
        assert info.gpu is None

    def test_darwin_platform_label(self):
        import agent.hardware as hw
        hw._cached_info = None
        with patch("agent.hardware.detect_nvidia_gpu", return_value=None), \
             patch("agent.hardware.detect_apple_gpu", return_value=None), \
             patch("agent.hardware.detect_system_info", return_value=(8, 16384, 8000)), \
             patch("agent.hardware.is_wsl", return_value=False), \
             patch("platform.system", return_value="Darwin"):
            info = detect_hardware()
        assert info.platform == "darwin"

    def test_windows_platform_label(self):
        import agent.hardware as hw
        hw._cached_info = None
        with patch("agent.hardware.detect_nvidia_gpu", return_value=None), \
             patch("agent.hardware.detect_apple_gpu", return_value=None), \
             patch("agent.hardware.detect_system_info", return_value=(4, 8192, 4096)), \
             patch("agent.hardware.is_wsl", return_value=False), \
             patch("platform.system", return_value="Windows"):
            info = detect_hardware()
        assert info.platform == "windows"


# ---------------------------------------------------------------------------
# get_available_vram_mb()
# ---------------------------------------------------------------------------

class TestGetAvailableVramMb:
    def setup_method(self):
        import agent.hardware as hw
        hw._cached_info = None

    def test_with_gpu(self):
        gpu = GPUInfo("RTX 3090", 24576, 18000, 6576, "nvidia")
        info = HardwareInfo(gpu=gpu, cpu_cores=8, ram_total_mb=32768,
                            ram_free_mb=16000, platform="linux")
        with patch("agent.hardware.detect_hardware", return_value=info):
            assert get_available_vram_mb() == 18000

    def test_without_gpu(self):
        info = HardwareInfo(gpu=None, cpu_cores=4, ram_total_mb=8192,
                            ram_free_mb=4096, platform="linux")
        with patch("agent.hardware.detect_hardware", return_value=info):
            assert get_available_vram_mb() == 0


# ---------------------------------------------------------------------------
# format_hardware_summary()
# ---------------------------------------------------------------------------

class TestFormatHardwareSummary:
    def _make_info(self, gpu=None, wsl=False):
        return HardwareInfo(
            gpu=gpu,
            cpu_cores=8,
            ram_total_mb=16384,
            ram_free_mb=8192,
            platform="linux",
            is_wsl=wsl,
        )

    def test_with_gpu(self):
        gpu = GPUInfo("RTX 3090", 24576, 20480, 4096, "nvidia")
        summary = format_hardware_summary(self._make_info(gpu=gpu))
        assert "RTX 3090" in summary
        assert "24576" in summary
        assert "20480" in summary
        assert "8" in summary      # cpu cores
        assert "16384" in summary  # ram total
        assert "8192" in summary   # ram free

    def test_without_gpu(self):
        summary = format_hardware_summary(self._make_info())
        assert "None detected" in summary

    def test_wsl_label_shown(self):
        summary = format_hardware_summary(self._make_info(wsl=True))
        assert "WSL2" in summary

    def test_wsl_label_absent_when_false(self):
        summary = format_hardware_summary(self._make_info(wsl=False))
        assert "WSL2" not in summary

    def test_no_argument_calls_detect_hardware(self):
        info = self._make_info()
        import agent.hardware as hw
        hw._cached_info = None
        with patch("agent.hardware.detect_hardware", return_value=info) as mock_detect:
            format_hardware_summary()
            mock_detect.assert_called_once()
