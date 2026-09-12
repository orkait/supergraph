from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)

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

    ready: bool = False
    provider: str | None = None
    device_name: str | None = None
    error: str | None = None
    preloaded: list[str] = field(default_factory=list)


_status: GPUStatus | None = None
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
    libs: list[Path] = []
    for site in _site_packages_dirs():
        nv_root = site / "nvidia"
        if not nv_root.is_dir():
            continue
        for component in _LOAD_ORDER:
            lib_dir = nv_root / component / "lib"
            if not lib_dir.is_dir():
                continue
            for so in sorted(lib_dir.glob("*.so*"), reverse=True):
                if so.is_file():
                    libs.append(so)
    return libs


def _preload(libs: list[Path]) -> list[str]:
    loaded: list[str] = []
    for lib in libs:
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            loaded.append(lib.name)
        except OSError as e:
            _log.debug("gpu: skip %s (%s)", lib.name, e)
    return loaded


def _probe_onnxruntime() -> tuple[bool, str | None, str | None]:
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
    try:
        from llama_cpp import llama_supports_gpu_offload
    except ImportError:
        return False, None, "llama-cpp-python not installed"
    except Exception as e:
        return False, None, f"llama-cpp-python native lib failed to load: {e}"

    if not llama_supports_gpu_offload():
        return False, None, (
            "llama-cpp-python is installed but its wheel was built "
            "without GPU support (llama_supports_gpu_offload() returned "
            "False). Reinstall from a GPU-enabled wheel index that "
            "matches your local CUDA / Metal / Vulkan / ROCm runtime; "
            "see https://github.com/abetlen/llama-cpp-python#installation"
        )
    return True, "llama_cpp_cuda", None


def setup(probe_llama_cpp: bool = True) -> GPUStatus:
    global _status
    if _status is not None:
        return _status

    with _setup_lock:
        if _status is not None:
            return _status
        return _setup_locked(probe_llama_cpp)


def _setup_locked(probe_llama_cpp: bool) -> GPUStatus:
    global _status
    libs = _find_nvidia_libs()
    preloaded = _preload(libs) if libs else []

    ort_ok, ort_provider, ort_err = _probe_onnxruntime()

    llama_ok = True
    llama_err: str | None = None
    if probe_llama_cpp:
        llama_ok, _, llama_err = _probe_llama_cpp()

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
        os.environ.setdefault("SUPERGRAPH_GPU", "1")
        _log.info(
            "gpu: ready provider=%s device=%s preloaded=%d libs",
            _status.provider, device_name, len(preloaded),
        )
    else:
        _log.warning("gpu: setup failed - %s", _status.error or "no provider")

    return _status


def is_ready() -> bool:
    return _status is not None and _status.ready


def status() -> GPUStatus | None:
    return _status


def n_gpu_layers_default() -> int:
    return -1 if is_ready() else 0


def _read_device_name() -> str | None:
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
    global _status
    _status = None
