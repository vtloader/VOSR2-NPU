"""Device / dtype selection for VOSR 2.0, with first-class Ascend NPU support.

Selection order is explicit and overridable with ``VOSR2_DEVICE``:

    VOSR2_DEVICE = npu | cuda | cpu   (case-insensitive)

Without the override: Ascend NPU (``torch_npu`` importable and
``torch.npu.is_available()``) wins, then CUDA, then CPU.

Importing ``torch_npu`` registers the ``npu`` backend with torch itself, so no
global monkey-patching (``torch_npu.contrib.transfer_to_npu``) is used: on an
NPU box ``torch.cuda.is_available()`` stays False and the two backends remain
distinguishable, which is what lets ``VOSR2_DEVICE=cpu`` be a real CPU run
rather than a silently-remapped one.

CANN itself must be sourced before Python starts (``source
/usr/local/Ascend/ascend-toolkit/set_env.sh``); torch_npu cannot set up
``LD_LIBRARY_PATH`` for an already-running process.
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("vosr2.device")

_VALID = ("npu", "cuda", "cpu")


def _import_torch_npu() -> bool:
    """Import torch_npu if present. Returns whether the NPU backend is usable."""
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    except Exception as exc:  # a broken CANN install should not kill a CPU/CUDA run
        log.debug("torch_npu present but failed to import: %s", exc)
        return False
    try:
        return bool(torch.npu.is_available())
    except Exception as exc:
        log.debug("torch.npu.is_available() failed: %s", exc)
        return False


def _detect_backend() -> str:
    requested = os.environ.get("VOSR2_DEVICE", "").strip().lower()
    if requested:
        if requested not in _VALID:
            raise ValueError(
                f"VOSR2_DEVICE={requested!r} is not valid; use one of {', '.join(_VALID)}."
            )
        if requested == "npu" and not _import_torch_npu():
            raise RuntimeError(
                "VOSR2_DEVICE=npu was requested but the NPU backend is unavailable. "
                "Check that torch_npu is installed and that CANN is sourced "
                "(source /usr/local/Ascend/ascend-toolkit/set_env.sh), then run "
                "`npu-smi info` to confirm the device is visible."
            )
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("VOSR2_DEVICE=cuda was requested but CUDA is unavailable.")
        return requested

    if _import_torch_npu():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def backend() -> str:
    """Resolved backend name: ``npu``, ``cuda`` or ``cpu``."""
    return _detect_backend()


def is_npu() -> bool:
    return backend() == "npu"


def device(index: int = 0) -> torch.device:
    b = backend()
    if b == "npu":
        n = torch.npu.device_count()
        if not 0 <= index < n:
            raise ValueError(f"NPU index {index} out of range (found {n} device(s)).")
        return torch.device(f"npu:{index}")
    if b == "cuda":
        n = torch.cuda.device_count()
        if not 0 <= index < n:
            raise ValueError(f"CUDA index {index} out of range (found {n} device(s)).")
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def device_name(index: int = 0) -> str:
    b = backend()
    try:
        if b == "npu":
            return torch.npu.get_device_name(index)
        if b == "cuda":
            return torch.cuda.get_device_name(index)
    except Exception:
        pass
    return "CPU"


def device_count() -> int:
    b = backend()
    if b == "npu":
        return torch.npu.device_count()
    if b == "cuda":
        return torch.cuda.device_count()
    return 0


def set_device(index: int = 0) -> torch.device:
    """Select and return the device, initialising the backend context."""
    dev = device(index)
    if dev.type == "npu":
        torch.npu.set_device(dev)
    elif dev.type == "cuda":
        torch.cuda.set_device(dev)
    return dev


def synchronize(dev: torch.device | None = None) -> None:
    if dev is None or dev.type == "npu":
        if backend() == "npu":
            torch.npu.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize(dev)


def empty_cache(dev: torch.device | None = None) -> None:
    """Release cached device memory. No-op on CPU."""
    b = backend()
    if b == "npu":
        torch.npu.empty_cache()
    elif b == "cuda":
        torch.cuda.empty_cache()


def memory_allocated(dev: torch.device | None = None) -> int:
    b = backend()
    if b == "npu":
        return torch.npu.memory_allocated(dev)
    if b == "cuda":
        return torch.cuda.memory_allocated(dev)
    return 0


def memory_reserved(dev: torch.device | None = None) -> int:
    b = backend()
    if b == "npu":
        return torch.npu.memory_reserved(dev)
    if b == "cuda":
        return torch.cuda.memory_reserved(dev)
    return 0


def reset_peak_memory(dev: torch.device | None = None) -> None:
    b = backend()
    if b == "npu":
        torch.npu.reset_peak_memory_stats(dev)
    elif b == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)


def peak_memory(dev: torch.device | None = None) -> int:
    b = backend()
    if b == "npu":
        return torch.npu.max_memory_allocated(dev)
    if b == "cuda":
        return torch.cuda.max_memory_allocated(dev)
    return 0


def memory_total(dev: torch.device | None = None) -> int:
    """Total HBM/VRAM in bytes, or 0 when unknown (CPU)."""
    b = backend()
    try:
        if b == "npu":
            return int(torch.npu.get_device_properties(dev or device()).total_memory)
        if b == "cuda":
            return int(torch.cuda.get_device_properties(dev or device()).total_memory)
    except Exception:
        pass
    return 0


def supports_bf16() -> bool:
    """Ascend 910 series implements bf16 natively; so does Ampere+."""
    return backend() in ("npu", "cuda")


def default_dtype() -> torch.dtype:
    """``bf16`` on NPU and Ampere+ CUDA, ``fp16`` on older CUDA, ``fp32`` on CPU."""
    b = backend()
    if b == "npu":
        return torch.bfloat16
    if b == "cuda":
        try:
            major, _ = torch.cuda.get_device_capability()
            return torch.bfloat16 if major >= 8 else torch.float16
        except Exception:
            return torch.float16
    return torch.float32


def resolve_dtype(name: str) -> torch.dtype:
    """Map a CLI/UI dtype name to a torch dtype."""
    name = (name or "default").strip().lower()
    if name in ("default", "auto"):
        return default_dtype()
    table = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError(
            f"Unknown dtype {name!r}; use one of default, fp16, bf16, fp32."
        )
    return table[name]


def describe() -> str:
    """One-line summary for logs and the UI."""
    b = backend()
    n = device_count()
    if b == "cpu":
        return "CPU"
    names = ", ".join(device_name(i) for i in range(n))
    return f"{b.upper()} ({n}x {names})"
