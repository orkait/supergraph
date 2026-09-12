from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import psutil


def _apply_blas_env_cap() -> None:
    cap = os.environ.get("SUPERGRAPH_BLAS_CAP")
    if cap is None:
        try:
            n = os.cpu_count() or 2
        except Exception:
            n = 2
        cap = str(max(1, min(2, n // 4)))
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "SCIPY_OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ.setdefault(var, cap)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_apply_blas_env_cap()


_GPU_PROVIDERS = (
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
)


_overrides: dict = {
    "profile": None,
    "ner_threads": None,
    "embed_threads": None,
    "rerank_threads": None,
    "embed_batch_size": None,
    "disable_load_scaling": False,
    "disable_battery_scaling": False,
}


_ENV_KEYS = (
    "SUPERGRAPH_PROFILE", "SUPERGRAPH_NER_THREADS",
    "SUPERGRAPH_EMBED_THREADS", "SUPERGRAPH_RERANK_THREADS",
    "SUPERGRAPH_EMBED_BATCH", "SUPERGRAPH_GPU",
)
_last_env_fingerprint: tuple | None = None


def _env_fingerprint() -> tuple:
    return tuple(os.environ.get(k) for k in _ENV_KEYS)


def configure(
    *,
    profile: str | None = None,
    ner_threads: int | None = None,
    embed_threads: int | None = None,
    rerank_threads: int | None = None,
    embed_batch_size: int | None = None,
    disable_load_scaling: bool = False,
    disable_battery_scaling: bool = False,
) -> None:
    global _overrides, _last_env_fingerprint
    _overrides = {
        "profile": profile,
        "ner_threads": ner_threads,
        "embed_threads": embed_threads,
        "rerank_threads": rerank_threads,
        "embed_batch_size": embed_batch_size,
        "disable_load_scaling": disable_load_scaling,
        "disable_battery_scaling": disable_battery_scaling,
    }
    _last_env_fingerprint = None
    _compute_profile.cache_clear()


@dataclass(frozen=True)
class ComputeProfile:
    name: str
    cores: int
    logical_cores: int
    ram_gb: float
    has_gpu: bool
    gpu_provider: str | None
    on_battery: bool
    load_pct: float

    ner_threads: int
    embed_threads: int
    rerank_threads: int

    embed_batch_size: int
    defer_embeddings: bool


def _detect_cores() -> tuple[int, int]:
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = os.cpu_count() or 1

    physical = psutil.cpu_count(logical=False) or affinity
    logical = psutil.cpu_count(logical=True) or affinity

    physical = min(physical, affinity) if affinity else physical
    logical = min(logical, affinity) if affinity else logical
    return max(1, physical), max(1, logical)


def _detect_ram_gb() -> float:
    try:
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 0.0


def _detect_battery() -> bool:
    try:
        bat = psutil.sensors_battery()
    except Exception:
        return False
    return bool(bat and not bat.power_plugged)


def _detect_gpu() -> tuple[bool, str | None]:
    if os.environ.get("SUPERGRAPH_GPU") != "1":
        return False, None
    try:
        import onnxruntime as ort
    except ImportError:
        return False, None
    try:
        available = ort.get_available_providers()
    except Exception:
        return False, None
    for p in _GPU_PROVIDERS:
        if p in available:
            return True, p
    return False, None


def _env_int(key: str, fallback: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return fallback
    try:
        return max(1, int(raw))
    except ValueError:
        return fallback


def _classify(cores: int, ram_gb: float, has_gpu: bool, on_battery: bool) -> str:
    if has_gpu:
        return "gpu"
    if cores <= 2 or (ram_gb and ram_gb < 4.0):
        return "tiny"
    if cores <= 6 or (ram_gb and ram_gb < 16.0) or on_battery:
        return "laptop"
    return "desktop"


def _base_profile(name: str, physical_cores: int) -> tuple[int, int, int, int, bool]:
    if name == "tiny":
        return (1, 1, 1, 16, False)
    if name == "laptop":
        t = max(2, min(4, physical_cores // 2))
        return (2, t, t, 32, False)
    if name == "desktop":
        t = max(2, min(8, physical_cores * 6 // 10))
        return (2, t, t, 64, True)
    if name == "gpu":
        t = max(2, min(12, physical_cores * 3 // 4))
        return (2, t, t, 128, True)
    return (2, 4, 4, 32, False)


def _detect_load_pct() -> float:
    try:
        return float(psutil.cpu_percent(interval=0.1))
    except Exception:
        return 0.0


@lru_cache(maxsize=1)
def _compute_profile() -> ComputeProfile:
    ov = _overrides
    requested = (ov["profile"] or os.environ.get("SUPERGRAPH_PROFILE", "auto")).strip().lower()
    physical, logical = _detect_cores()
    ram_gb = _detect_ram_gb()
    has_gpu, gpu_provider = _detect_gpu()
    on_battery = _detect_battery()

    if requested in {"tiny", "laptop", "desktop", "gpu"}:
        name = requested
    else:
        name = _classify(physical, ram_gb, has_gpu, on_battery)

    ner_t, embed_t, rerank_t, batch, defer = _base_profile(name, physical)

    ner_locked = ov["ner_threads"] is not None
    embed_locked = ov["embed_threads"] is not None
    rerank_locked = ov["rerank_threads"] is not None

    if on_battery and name != "tiny" and not ov["disable_battery_scaling"]:
        if not embed_locked:
            embed_t = max(1, embed_t - 1)
        if not rerank_locked:
            rerank_t = max(1, rerank_t - 1)

    load_pct = _detect_load_pct()
    if load_pct > 40.0 and name != "tiny" and not ov["disable_load_scaling"]:
        if not embed_locked:
            embed_t = max(1, embed_t // 2)
        if not rerank_locked:
            rerank_t = max(1, rerank_t // 2)

    final_ner = ov["ner_threads"] if ner_locked else _env_int("SUPERGRAPH_NER_THREADS", ner_t)
    final_embed = ov["embed_threads"] if embed_locked else _env_int("SUPERGRAPH_EMBED_THREADS", embed_t)
    final_rerank = ov["rerank_threads"] if rerank_locked else _env_int("SUPERGRAPH_RERANK_THREADS", rerank_t)
    final_batch = ov["embed_batch_size"] if ov["embed_batch_size"] is not None else _env_int("SUPERGRAPH_EMBED_BATCH", batch)

    profile = ComputeProfile(
        name=name,
        cores=physical,
        logical_cores=logical,
        ram_gb=ram_gb,
        has_gpu=has_gpu,
        gpu_provider=gpu_provider,
        on_battery=on_battery,
        load_pct=load_pct,
        ner_threads=max(1, final_ner),
        embed_threads=max(1, final_embed),
        rerank_threads=max(1, final_rerank),
        embed_batch_size=max(1, final_batch),
        defer_embeddings=defer,
    )
    blas_cap = max(1, max(profile.embed_threads, profile.ner_threads, profile.rerank_threads))
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=blas_cap)
    except Exception:
        pass
    return profile


def get_profile() -> ComputeProfile:
    global _last_env_fingerprint
    fp = _env_fingerprint()
    if fp != _last_env_fingerprint:
        _compute_profile.cache_clear()
        _last_env_fingerprint = fp
    return _compute_profile()


get_profile.cache_clear = _compute_profile.cache_clear  # type: ignore[attr-defined]


def describe_profile() -> str:
    p = get_profile()
    gpu = f" gpu={p.gpu_provider}" if p.has_gpu else ""
    ram = f"{p.ram_gb:.1f}GB" if p.ram_gb else "?GB"
    bat = " (on battery)" if p.on_battery else ""
    load = f" load={p.load_pct:.0f}%" if p.load_pct else ""
    return (
        f"profile={p.name} cores={p.cores}/{p.logical_cores} ram={ram}{gpu}{bat}{load} "
        f"threads(ner/embed/rerank)={p.ner_threads}/{p.embed_threads}/{p.rerank_threads} "
        f"embed_batch={p.embed_batch_size}"
    )


def reset_profile_cache() -> None:
    global _last_env_fingerprint
    _last_env_fingerprint = None
    _compute_profile.cache_clear()
