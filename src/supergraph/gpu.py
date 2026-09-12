"""Auto-setup for CUDA GPU offload.

supergraph never grabs a GPU implicitly. Users opt in by calling
``supergraph.gpu.setup()`` directly, by passing ``profile="pro"`` to
``SuperGraph`` (which calls setup internally), or by passing explicit
``n_gpu_layers`` kwargs to llama-cpp paths.

This module handles the dirty work the user should not have to:

  1. Discovers ``nvidia-*-cu12`` wheel directories under any visible
     ``site-packages`` and ctypes-preloads their shared libraries with
     ``RTLD_GLOBAL`` in dependency order. This bypasses the
     ``LD_LIBRARY_PATH`` requirement that otherwise forces users to
     re-launch Python with custom env vars.
  2. Runs a probe: imports onnxruntime + checks llama-cpp-python's CUDA
     build flag. If either fails, returns a structured failure with the
     original error and falls back to CPU silently.
  3. Caches the probe result process-wide so repeated calls are free.

Other supergraph modules consume the cached state via ``is_ready()`` /
``status()`` / ``n_gpu_layers_default()``.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)

# Order matters. cuda_runtime + cuda_nvrtc + nvjitlink come before
# cublas / cufft / cudnn so dependent symbols resolve. cudnn last because
# it pulls cublasLt at load-time.
_LOAD_ORDER = (
    "cuda_runtime",
    "cuda_nvrtc",
    "nvjitlink",
    "cublas",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "cudnn",
)


@dataclass
class GPUStatus:
    """Result of a GPU setup attempt. Read-only after probe completes."""

    ready: bool = False
    provider: str | None = None
    device_name: str | None = None
    error: str | None = None
    preloaded: list[str] = field(default_factory=list)


_status: GPUStatus | None = None
# Guards setup() so concurrent callers don't both run preload + probe.
# Cached result is fine to read without the lock once _status is set.
_setup_lock = threading.Lock()


def _site_packages_dirs() -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in sys.path:
        if not p:
            continue
        path = Path(p)
        if path.is_dir() and path.name == "site-packages":
            if path not in seen:
                seen.add(path)
                out.append(path)
    return out


def _find_nvidia_libs() -> list[Path]:
    """Return .so files from installed nvidia-*-cu12 wheels in load order."""
    libs: list[Path] = []
    for site in _site_packages_dirs():
        nv_root = site / "nvidia"
        if not nv_root.is_dir():
            continue
        for component in _LOAD_ORDER:
            lib_dir = nv_root / component / "lib"
            if not lib_dir.is_dir():
                continue
            # Sort reverse so versioned ".so.<X>.<Y>" beats bare ".so" symlinks
            # (some symlinks point at non-existent dev headers).
            for so in sorted(lib_dir.glob("*.so*"), reverse=True):
                if so.is_file():
                    libs.append(so)
    return libs


def _preload(libs: list[Path]) -> list[str]:
    """ctypes-preload .so files; return successfully loaded names."""
    loaded: list[str] = []
    for lib in libs:
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            loaded.append(lib.name)
        except OSError as e:
            _log.debug("gpu: skip %s (%s)", lib.name, e)
    return loaded


def _probe_onnxruntime() -> tuple[bool, str | None, str | None]:
    """Check if onnxruntime exposes CUDAExecutionProvider.

    Returns (ok, provider, error). Listing the provider is necessary but
    not sufficient - actual cudaSetDevice() may still fail at session
    creation time. Full bind verification happens at first inference.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return False, None, "onnxruntime not installed"

    try:
        available = ort.get_available_providers()
    except Exception as e:
        return False, None, f"onnxruntime.get_available_providers failed: {e}"

    if "CUDAExecutionProvider" not in available:
        return False, None, (
            f"CUDAExecutionProvider not in onnxruntime providers "
            f"(have {available}); install onnxruntime-gpu"
        )
    return True, "CUDAExecutionProvider", None


def _probe_llama_cpp() -> tuple[bool, str | None, str | None]:
    """Check llama-cpp-python's CUDA build flag. Cheap, no model load."""
    try:
        from llama_cpp import llama_supports_gpu_offload
    except ImportError:
        return False, None, "llama-cpp-python not installed"
    except Exception as e:
        # Native lib fails to load (libcudart.so.12 missing, etc.). Preload
        # was supposed to fix this; if it still fails the wheel is broken.
        return False, None, f"llama-cpp-python native lib failed to load: {e}"

    if not llama_supports_gpu_offload():
        # Reason is deterministic: wheel was built without GPU. We do not
        # claim which CUDA toolchain index the user should pull from -
        # that depends on the host driver / runtime they actually have
        # (cu12 / cu13 / Metal / Vulkan / ROCm). Link to upstream wheel
        # index so users pick the one matching their setup.
        return False, None, (
            "llama-cpp-python is installed but its wheel was built "
            "without GPU support (llama_supports_gpu_offload() returned "
            "False). Reinstall from a GPU-enabled wheel index that "
            "matches your local CUDA / Metal / Vulkan / ROCm runtime; "
            "see https://github.com/abetlen/llama-cpp-python#installation"
        )
    return True, "llama_cpp_cuda", None


def setup(probe_llama_cpp: bool = True) -> GPUStatus:
    """Discover, preload, and probe. Cached after first call.

    Pass ``probe_llama_cpp=False`` if the caller only needs ORT-GPU (NER /
    onnx embedders) and llama-cpp is not installed.
    """
    global _status
    if _status is not None:
        return _status

    with _setup_lock:
        # Re-check under the lock: another thread may have populated
        # _status while we waited.
        if _status is not None:
            return _status
        return _setup_locked(probe_llama_cpp)


def _setup_locked(probe_llama_cpp: bool) -> GPUStatus:
    """setup() body, run with _setup_lock held."""
    global _status
    libs = _find_nvidia_libs()
    preloaded = _preload(libs) if libs else []

    ort_ok, ort_provider, ort_err = _probe_onnxruntime()

    llama_ok = True
    llama_err: str | None = None
    if probe_llama_cpp:
        llama_ok, _, llama_err = _probe_llama_cpp()

    # Either path counts as success - users may install only one. Failure
    # message lists every attempt so the diagnostic is honest.
    ready = ort_ok or (probe_llama_cpp and llama_ok)
    err_lines = []
    if not ort_ok and ort_err:
        err_lines.append(f"onnxruntime: {ort_err}")
    if probe_llama_cpp and not llama_ok and llama_err:
        err_lines.append(f"llama-cpp-python: {llama_err}")

    device_name: str | None = None
    if ready:
        device_name = _read_device_name()

    _status = GPUStatus(
        ready=ready,
        provider=(ort_provider if ort_ok else ("llama_cpp_cuda" if llama_ok else None)),
        device_name=device_name,
        error="; ".join(err_lines) if err_lines and not ready else None,
        preloaded=preloaded,
    )

    if ready:
        # Surface to compute_profile so its env-var-based gate also flips.
        os.environ.setdefault("SUPERGRAPH_GPU", "1")
        _log.info(
            "gpu: ready provider=%s device=%s preloaded=%d libs",
            _status.provider, device_name, len(preloaded),
        )
    else:
        _log.warning("gpu: setup failed - %s", _status.error or "no provider")

    return _status


def is_ready() -> bool:
    """True iff ``setup()`` succeeded. Does NOT trigger setup itself."""
    return _status is not None and _status.ready


def status() -> GPUStatus | None:
    """Last cached probe result. None if ``setup()`` was never called."""
    return _status


def n_gpu_layers_default() -> int:
    """``-1`` (offload all) when GPU ready, ``0`` (CPU) otherwise."""
    return -1 if is_ready() else 0


def _read_device_name() -> str | None:
    """Best-effort device name via nvidia-smi. None on any failure."""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            first = out.stdout.strip().splitlines()
            if first:
                return first[0].strip()
    except Exception:
        pass
    return None


def reset_for_tests() -> None:
    """Clear cache. Test-only - not part of the public API."""
    global _status
    _status = None
