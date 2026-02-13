#!/usr/bin/env python3
"""A.E.G.I.S Hardware Validation Script.

Checks CUDA/GPU availability, system RAM, swap, and Python/PyTorch versions
to verify the environment is correctly configured for AI inference workloads.
"""

import logging
import platform
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bytes_to_gib(b: int) -> float:
    return round(b / (1024 ** 3), 2)


def _read_meminfo_field(field: str) -> int:
    """Read a field from /proc/meminfo and return value in bytes."""
    meminfo = Path("/proc/meminfo")
    for line in meminfo.read_text().splitlines():
        if line.startswith(field):
            # Format: "FieldName:    12345 kB"
            parts = line.split()
            return int(parts[1]) * 1024  # kB -> bytes
    return 0

# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_python() -> None:
    log.info("--- Python ---")
    log.info("Python version : %s", platform.python_version())
    log.info("Python path    : %s", sys.executable)


def check_pytorch() -> None:
    log.info("--- PyTorch ---")
    try:
        import torch  # noqa: WPS433

        log.info("PyTorch version: %s", torch.__version__)
        log.info("CUDA built with: %s", torch.version.cuda)
        log.info("cuDNN version  : %s", torch.backends.cudnn.version())
    except ImportError:
        log.error("PyTorch is NOT installed.")


def check_cuda() -> None:
    log.info("--- CUDA / GPU ---")
    try:
        import torch  # noqa: WPS433

        if not torch.cuda.is_available():
            log.warning("CUDA is NOT available to PyTorch.")
            return

        log.info("CUDA available : True")
        log.info("CUDA version   : %s", torch.version.cuda)
        log.info("GPU count      : %s", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total = _bytes_to_gib(props.total_memory)
            free = _bytes_to_gib(torch.cuda.mem_get_info(i)[0])
            log.info(
                "  GPU %d: %s | VRAM total: %.2f GiB | VRAM free: %.2f GiB",
                i, props.name, total, free,
            )
    except ImportError:
        log.error("PyTorch is NOT installed — cannot check CUDA.")


def check_ram() -> None:
    log.info("--- System RAM ---")
    total = _read_meminfo_field("MemTotal")
    available = _read_meminfo_field("MemAvailable")
    log.info("RAM total     : %.2f GiB", _bytes_to_gib(total))
    log.info("RAM available : %.2f GiB", _bytes_to_gib(available))


def check_swap() -> None:
    log.info("--- Swap ---")
    total = _read_meminfo_field("SwapTotal")
    free = _read_meminfo_field("SwapFree")
    log.info("Swap total    : %.2f GiB", _bytes_to_gib(total))
    log.info("Swap free     : %.2f GiB", _bytes_to_gib(free))


def check_nvidia_driver() -> None:
    log.info("--- NVIDIA Driver ---")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        log.warning("nvidia-smi not found on PATH.")
        return

    result = subprocess.run(
        [nvidia_smi, "--query-gpu=driver_version,persistence_mode",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            driver, persistence = [s.strip() for s in line.split(",")]
            log.info("Driver version : %s", driver)
            log.info("Persistence    : %s", persistence)
    else:
        log.warning("nvidia-smi failed: %s", result.stderr.strip())

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 50)
    log.info("A.E.G.I.S Hardware Validation")
    log.info("=" * 50)

    check_python()
    check_pytorch()
    check_cuda()
    check_nvidia_driver()
    check_ram()
    check_swap()

    log.info("=" * 50)
    log.info("Validation complete.")


if __name__ == "__main__":
    main()
