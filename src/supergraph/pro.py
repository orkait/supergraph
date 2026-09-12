"""supergraph pro: slotted spec + live calibration + resolver.

Pro mode is a single profile that owns model selection (embedder /
reranker / ingest mode / vision / audio / NER), GPU setup, and resource
sizing for agentic-memory deployments. This module defines the data
model and the resolver. CLI lives in ``supergraph.cli``;
``SuperGraph(profile="pro")`` integration lives in ``supergraph.store``.

Hard rules baked into this design:

  - **No hard-coded RAM/TPS/disk numbers.** All resource estimates come
    from a per-host calibration cache populated by ``supergraph pro
    setup`` (which downloads each component then immediately probes it).
    Missing cache → ``ProCalibrationMissing`` raised; we don't guess.
  - **Slotted spec, not flat list.** Embedders / rerankers / ingest
    modes are mutually exclusive within a slot. Slots make the choice
    explicit.
  - **One resolver, one trip.** ``resolve(spec, host) -> ResolvedConfig``
    returns either ``fits=True`` with the maximal knobs that fit, or
    ``fits=False`` with structured shortfalls + suggestions. Callers
    raise ``ProUnsupportedHostError`` from the latter.
  - **Layered VRAM allocation: bonsai-first.** Throughput-critical path
    gets GPU first; reranker / vision get whatever VRAM remains.
  - **Linux x86_64 + NVIDIA CUDA 12 only for v1.** Apple Metal /
    AMD ROCm out of scope.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import msgspec

from supergraph.core.errors import SuperGraphError

_log = logging.getLogger(__name__)

# Schema version for the calibration cache file. Bump on any breaking
# change to the cache shape so older caches are auto-discarded instead
# of mis-parsed.
_CACHE_SCHEMA_VERSION = 1

# Default location of the calibration cache. Honors XDG_CACHE_HOME.
_DEFAULT_CACHE_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")))
    / "supergraph"
)

# Slot value sets are encoded as Literal types on ProSpec itself; msgspec
# enforces at decode time. No separate tuple registry to keep in sync.
#
# Reserved keys in CalibrationEntry.extra populated by the probe runner
# (supergraph.cli `pro setup`/`probe`). Resolver reads these to scale
# knobs without hard-coding host-RAM thresholds:
#
#   "n_ctx_min" / "n_ctx_default" / "n_ctx_max"   - bonsai context window
#                                                   measured at ram_mb_min /
#                                                   ram_mb_at_default /
#                                                   ram_mb_max
#   "n_batch_at_default"                          - bonsai n_batch at the
#                                                   default measurement
#   "embed_batch_at_default"                      - embedder batch size at
#                                                   the default measurement
#   "reranker_max_at_default"                     - reranker max_length at
#                                                   the default measurement
#
# Missing keys → resolver falls back to safe minimums.
_EXTRA_N_CTX_MIN = "n_ctx_min"
_EXTRA_N_CTX_DEFAULT = "n_ctx_default"
_EXTRA_N_CTX_MAX = "n_ctx_max"
_EXTRA_N_BATCH = "n_batch_at_default"
_EXTRA_EMBED_BATCH = "embed_batch_at_default"
_EXTRA_RERANKER_MAX = "reranker_max_at_default"

_FALLBACK_N_CTX = 2048
_FALLBACK_N_BATCH = 256
_FALLBACK_EMBED_BATCH = 16
_FALLBACK_RERANKER_MAX = 512


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------


class ProUnsupportedHostError(SuperGraphError):
    """Selected pro spec does not fit current host. ``.resolved`` carries
    the structured ``ResolvedConfig`` (with shortfalls + suggestions)."""

    def __init__(self, message: str, resolved: ResolvedConfig):
        super().__init__(message)
        self.resolved = resolved


class ProCalibrationMissing(SuperGraphError):
    """No calibration data for the selected components on this host.
    ``.missing_components`` lists the component_ids needing calibration.
    Run ``supergraph pro setup`` or ``supergraph pro probe``."""

    def __init__(self, message: str, missing_components: list[str]):
        super().__init__(message)
        self.missing_components = missing_components


class ProExtraNotInstalled(SuperGraphError):
    """``[pro]`` extra not installed. ``.missing_dists`` lists pip
    distributions that need to be present for the selected spec."""

    def __init__(self, message: str, missing_dists: list[str]):
        super().__init__(message)
        self.missing_dists = missing_dists


# ---------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------


class ProSpec(msgspec.Struct, frozen=True):
    """Slotted pro spec. Each slot picks at most one option; ``"none"``
    disables that slot. Defaults match the measured-best LoCoMo
    configuration as of v0.5.0 - subject to change as benches improve."""

    embedder: Literal[
        "jina-v5-small", "jina-v5-nano",
        "model2vec-256d", "embeddinggemma-300m",
        "fastembed-bge-small", "none",
    ] = "jina-v5-small"

    reranker: Literal["jina-v3", "none"] = "jina-v3"

    ingest_mode: Literal["bonsai", "deterministic"] = "bonsai"
    bonsai_quant: Literal["tq1_0", "tq2_0"] = "tq1_0"
    bonsai_skill: Literal["lite", "full"] = "lite"

    vision: Literal["smolvlm2-2.2b", "qwen-vl-3b", "none"] = "none"
    audio: Literal["whisper-tiny", "whisper-base", "whisper-small", "none"] = "none"

    ner: Literal["tinybert", "none"] = "tinybert"

    def component_ids(self) -> list[str]:
        """Return calibration-cache keys for each non-empty slot.

        Bonsai is one cache entry per (quant, skill) combination because
        n_ctx / n_batch / TPS all depend on both. Other slots are simpler.
        """
        ids: list[str] = []
        if self.embedder != "none":
            ids.append(f"embedder:{self.embedder}")
        if self.reranker != "none":
            ids.append(f"reranker:{self.reranker}")
        if self.ingest_mode == "bonsai":
            ids.append(f"ingest:bonsai-{self.bonsai_quant}-{self.bonsai_skill}")
        # ingest_mode="deterministic" needs no model beyond the NER slot.
        if self.vision != "none":
            ids.append(f"vision:{self.vision}")
        if self.audio != "none":
            ids.append(f"audio:{self.audio}")
        if self.ner != "none":
            ids.append(f"ner:{self.ner}")
        return ids

    def required_dists(self) -> list[str]:
        """Pip distribution names that must be importable for this spec.

        Used by ``check_extras_installed()`` to fail fast with a clear
        ``pip install 'supergraph[pro]'`` hint when extras are missing.
        Returns canonical PEP 503 normalized names.
        """
        dists: list[str] = []
        if self.ingest_mode == "bonsai" or self.vision != "none":
            dists.append("llama-cpp-python")
        if self.embedder.startswith("jina-v5"):
            dists.append("onnxruntime")
        if self.embedder == "fastembed-bge-small":
            dists.append("fastembed")
        if self.audio != "none":
            dists.append("faster-whisper")
        if self.ner == "tinybert":
            dists.append("onnxruntime")
            dists.append("tokenizers")
        # De-dup while preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for d in dists:
            if d not in seen:
                seen.add(d)
                out.append(d)
        return out


# ---------------------------------------------------------------------
# Host snapshot
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class HostSnapshot:
    """Live host capabilities. No assumptions, no static thresholds.

    All numbers come from ``capture()`` calling psutil / shutil / OS at
    the moment of resolution. ``host_signature()`` derives a stable key
    used to namespace calibration cache entries.
    """

    ram_total_mb: int
    ram_available_mb: int
    disk_free_mb: int
    cpu_cores_physical: int
    cpu_cores_logical: int
    gpu_ready: bool
    gpu_name: str | None
    gpu_vram_total_mb: int
    gpu_vram_free_mb: int
    extras_installed: frozenset[str]

    @classmethod
    def capture(
        cls,
        cache_dir: Path | None = None,
        probe_gpu: bool = True,
    ) -> HostSnapshot:
        """Snapshot RAM / disk / CPU / GPU / installed-extras live."""
        # RAM
        try:
            import psutil
            mem = psutil.virtual_memory()
            ram_total = int(mem.total / (1024 * 1024))
            ram_avail = int(mem.available / (1024 * 1024))
        except Exception:  # pragma: no cover - psutil is a core dep
            ram_total = ram_avail = 0

        # Disk: free space at the cache root, since that's where models
        # land.
        target = cache_dir or _DEFAULT_CACHE_DIR
        target.mkdir(parents=True, exist_ok=True)
        disk_free = int(shutil.disk_usage(str(target)).free / (1024 * 1024))

        # CPU
        try:
            import psutil
            phys = psutil.cpu_count(logical=False) or 0
            logical = psutil.cpu_count(logical=True) or os.cpu_count() or 0
        except Exception:  # pragma: no cover
            phys = logical = os.cpu_count() or 0

        # GPU - via supergraph.gpu (cheap if already setup).
        gpu_ready = False
        gpu_name: str | None = None
        gpu_vram_total = gpu_vram_free = 0
        if probe_gpu:
            try:
                from supergraph import gpu as _gpu
                status = _gpu.setup() if _gpu.status() is None else _gpu.status()
                gpu_ready = bool(status and status.ready)
                if gpu_ready:
                    gpu_name = (status.device_name if status else None)
                    gpu_vram_total, gpu_vram_free = _read_vram_mb()
            except Exception as e:  # pragma: no cover
                _log.debug("pro: gpu probe in HostSnapshot.capture() failed: %s", e)

        # Installed extras (canonical pip dist names).
        installed: set[str] = set()
        try:
            import importlib.metadata as im
            for d in im.distributions():
                name = d.metadata["Name"] or ""
                if name:
                    installed.add(name.lower().replace("_", "-"))
        except Exception:  # pragma: no cover
            pass

        return cls(
            ram_total_mb=ram_total,
            ram_available_mb=ram_avail,
            disk_free_mb=disk_free,
            cpu_cores_physical=phys,
            cpu_cores_logical=logical,
            gpu_ready=gpu_ready,
            gpu_name=gpu_name,
            gpu_vram_total_mb=gpu_vram_total,
            gpu_vram_free_mb=gpu_vram_free,
            extras_installed=frozenset(installed),
        )

    def host_signature(self) -> str:
        """Stable key for calibration cache namespacing.

        Includes architecture, physical cores, RAM bucket (rounded to
        nearest 1 GB), and GPU name + total VRAM. Different signature →
        cache invalidated. We round RAM because exact `available` jitters
        per process; total RAM is the durable quantity.
        """
        import platform
        ram_bucket = (self.ram_total_mb // 1024) * 1024
        parts = [
            platform.system().lower(),
            platform.machine(),
            f"cpu_{self.cpu_cores_physical}c{self.cpu_cores_logical}t",
            f"ram_{ram_bucket}mb",
        ]
        if self.gpu_ready and self.gpu_name:
            parts.append(f"gpu_{self.gpu_name.replace(' ', '_')}_{self.gpu_vram_total_mb}mb")
        else:
            parts.append("gpu_none")
        return "-".join(parts)


def _read_vram_mb() -> tuple[int, int]:
    """nvidia-smi VRAM (total, free) in MB. (0, 0) on any failure."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode != 0:
            return 0, 0
        first = out.stdout.strip().splitlines()[0]
        total_s, free_s = first.split(",")
        return int(total_s.strip()), int(free_s.strip())
    except Exception:
        return 0, 0


# ---------------------------------------------------------------------
# Calibration cache
# ---------------------------------------------------------------------


@dataclass
class CalibrationEntry:
    """One component's measured numbers. All fields populated by the
    probe runner in ``supergraph pro setup``; this module only reads.

    Conventions:
      - ``ram_mb_*`` is resident-set delta from idle baseline at the
        labelled config (e.g. ``ram_mb_n_ctx_2048_cpu``).
      - ``vram_mb_full_offload`` is delta in GPU memory free between
        before-load and after-load when offloading all layers.
      - ``tps_cpu_threads`` maps thread-count (string) → tokens-per-second
        (float). Resolver picks the entry matching host cpu_cores_logical
        (or interpolates).
      - ``tps_gpu_full_offload`` is at n_gpu_layers=-1 with default
        n_batch.
    """

    component_id: str
    measured_at: str
    ram_mb_idle: int = 0
    ram_mb_at_default: int = 0
    ram_mb_min: int = 0
    ram_mb_max: int = 0
    disk_mb: int = 0
    vram_mb_full_offload: int = 0
    tps_cpu_threads: dict[str, float] = field(default_factory=dict)
    tps_gpu_full_offload: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationCache:
    """In-memory representation of ``calibration.json``.

    Loader is forgiving: schema-version mismatch or host-signature
    mismatch returns an empty cache. Writer overwrites atomically.
    """

    schema_version: int
    supergraph_version: str
    host_signature: str
    measured_at: str
    components: dict[str, CalibrationEntry]

    @classmethod
    def empty(cls, host_signature: str) -> CalibrationCache:
        from supergraph import __version__ as gs_version
        return cls(
            schema_version=_CACHE_SCHEMA_VERSION,
            supergraph_version=gs_version,
            host_signature=host_signature,
            measured_at=datetime.now(timezone.utc).isoformat(),
            components={},
        )

    @classmethod
    def load(
        cls,
        host_signature: str,
        cache_dir: Path | None = None,
    ) -> CalibrationCache:
        """Read cache file. Returns ``empty()`` on any of:
          - file missing
          - JSON parse error
          - schema_version mismatch
          - host_signature mismatch (host changed)
          - supergraph_version mismatch (calibration semantics changed)
        """
        path = (cache_dir or _DEFAULT_CACHE_DIR) / "calibration.json"
        try:
            data = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return cls.empty(host_signature)
        except Exception as e:
            _log.warning("pro: calibration cache read failed (%s); using empty", e)
            return cls.empty(host_signature)

        if data.get("schema_version") != _CACHE_SCHEMA_VERSION:
            _log.info("pro: calibration cache schema mismatch; discarding")
            return cls.empty(host_signature)
        if data.get("host_signature") != host_signature:
            _log.info("pro: calibration cache host mismatch; discarding")
            return cls.empty(host_signature)
        from supergraph import __version__ as gs_version
        if data.get("supergraph_version") != gs_version:
            _log.info("pro: calibration cache supergraph-version mismatch; discarding")
            return cls.empty(host_signature)

        components: dict[str, CalibrationEntry] = {}
        for cid, raw in (data.get("components") or {}).items():
            try:
                components[cid] = CalibrationEntry(
                    component_id=cid,
                    measured_at=raw.get("measured_at", ""),
                    ram_mb_idle=int(raw.get("ram_mb_idle", 0)),
                    ram_mb_at_default=int(raw.get("ram_mb_at_default", 0)),
                    ram_mb_min=int(raw.get("ram_mb_min", 0)),
                    ram_mb_max=int(raw.get("ram_mb_max", 0)),
                    disk_mb=int(raw.get("disk_mb", 0)),
                    vram_mb_full_offload=int(raw.get("vram_mb_full_offload", 0)),
                    tps_cpu_threads={str(k): float(v)
                                     for k, v in (raw.get("tps_cpu_threads") or {}).items()},
                    tps_gpu_full_offload=(float(raw["tps_gpu_full_offload"])
                                          if raw.get("tps_gpu_full_offload") is not None
                                          else None),
                    extra=raw.get("extra") or {},
                )
            except (TypeError, ValueError) as e:
                _log.debug("pro: skip malformed cache entry %r (%s)", cid, e)
        return cls(
            schema_version=_CACHE_SCHEMA_VERSION,
            supergraph_version=data.get("supergraph_version", ""),
            host_signature=host_signature,
            measured_at=data.get("measured_at", ""),
            components=components,
        )

    def save(self, cache_dir: Path | None = None) -> Path:
        """Atomic write to ``calibration.json``. Returns the path written."""
        target_dir = cache_dir or _DEFAULT_CACHE_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "calibration.json"
        # Path.with_suffix REPLACES the suffix; appending ".tmp" via name
        # concatenation is the unambiguous form regardless of basename.
        tmp = path.parent / (path.name + ".tmp")
        payload = {
            "schema_version": self.schema_version,
            "supergraph_version": self.supergraph_version,
            "host_signature": self.host_signature,
            "measured_at": self.measured_at,
            "components": {
                cid: {
                    "measured_at": e.measured_at,
                    "ram_mb_idle": e.ram_mb_idle,
                    "ram_mb_at_default": e.ram_mb_at_default,
                    "ram_mb_min": e.ram_mb_min,
                    "ram_mb_max": e.ram_mb_max,
                    "disk_mb": e.disk_mb,
                    "vram_mb_full_offload": e.vram_mb_full_offload,
                    "tps_cpu_threads": e.tps_cpu_threads,
                    "tps_gpu_full_offload": e.tps_gpu_full_offload,
                    "extra": e.extra,
                }
                for cid, e in self.components.items()
            },
        }
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(path)
        return path


# ---------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedConfig:
    """Output of ``resolve()``. ``fits=False`` is the hard-stop signal.

    Knob fields (n_ctx etc.) are populated only when ``fits=True``;
    callers should treat them as undefined when fits is False. Use
    ``warnings`` for non-fatal observations (tight RAM, partial
    offload, calibration aging) and ``shortfalls`` for the reasons
    fits=False.
    """

    spec: ProSpec
    host: HostSnapshot
    fits: bool
    n_ctx: int = 0
    bonsai_n_batch: int = 0
    bonsai_n_gpu_layers: int = 0
    reranker_max_len: int = 0
    reranker_gpu_layers: int = 0
    embed_batch: int = 0
    vision_offload: bool = False
    projected_tps: dict[str, float] = field(default_factory=dict)
    ram_budget_mb: dict[str, int] = field(default_factory=dict)
    vram_budget_mb: dict[str, int] = field(default_factory=dict)
    shortfalls: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    calibration_source: Literal["measured", "missing"] = "missing"
    calibration_age_s: int | None = None


def check_extras_installed(spec: ProSpec, host: HostSnapshot) -> None:
    """Raise ``ProExtraNotInstalled`` when required pip dists are absent.

    Called before ``resolve()`` so users see the install hint instead of
    a more cryptic calibration-missing error.
    """
    required = spec.required_dists()
    missing = [d for d in required if d.lower() not in host.extras_installed]
    if missing:
        raise ProExtraNotInstalled(
            "supergraph pro spec requires the following pip distributions "
            "which are not installed: "
            + ", ".join(missing)
            + ". Install via: pip install 'supergraph[pro]'",
            missing_dists=missing,
        )


def resolve(
    spec: ProSpec,
    host: HostSnapshot | None = None,
    cache: CalibrationCache | None = None,
    cache_dir: Path | None = None,
) -> ResolvedConfig:
    """Compute the maximal knobs that fit ``spec`` on ``host``.

    Behaviour:
      - host = None → ``HostSnapshot.capture()``
      - cache = None → loads from disk (calibration.json) for the host
      - if any selected component has no calibration entry, returns
        ``ResolvedConfig(calibration_source="missing", fits=False, ...)``
        with structured shortfalls. Caller can convert to
        ``ProCalibrationMissing`` if they want a hard exception.
      - otherwise computes RAM / VRAM budgets and the largest knobs that
        fit the host's available resources. ``fits=True`` only when
        every component's *minimum* RAM/disk/VRAM budget is satisfied.
    """
    host = host or HostSnapshot.capture(cache_dir=cache_dir)
    cache = cache or CalibrationCache.load(host.host_signature(), cache_dir=cache_dir)

    # Step 1: every selected component must have calibration data.
    component_ids = spec.component_ids()
    missing = [cid for cid in component_ids if cid not in cache.components]
    if missing:
        return ResolvedConfig(
            spec=spec,
            host=host,
            fits=False,
            shortfalls=[
                f"calibration missing for: {', '.join(missing)}. "
                "Run `supergraph pro setup` to download + probe these "
                "components, or `supergraph pro probe --refresh` to "
                "recalibrate everything."
            ],
            calibration_source="missing",
        )

    # Step 2: sum minimum-knob RAM and disk requirements.
    use_gpu = host.gpu_ready
    ram_budget: dict[str, int] = {}
    vram_budget: dict[str, int] = {}
    disk_required = 0
    for cid in component_ids:
        e = cache.components[cid]
        disk_required += e.disk_mb
        if use_gpu and e.vram_mb_full_offload > 0:
            # Component will live in VRAM - host RAM only carries Python
            # overhead. ram_mb_idle is the closest proxy.
            ram_budget[cid] = e.ram_mb_idle
            vram_budget[cid] = e.vram_mb_full_offload
        else:
            # CPU mode - minimum-knob RAM is the floor we need.
            ram_budget[cid] = max(e.ram_mb_min, e.ram_mb_idle)
            vram_budget[cid] = 0

    total_ram = sum(ram_budget.values())
    total_vram = sum(vram_budget.values())

    shortfalls: list[str] = []
    suggestions: list[str] = []
    warnings: list[str] = []

    if disk_required > host.disk_free_mb:
        shortfalls.append(
            f"disk: need {disk_required} MB free for component models, "
            f"only {host.disk_free_mb} MB available at "
            f"{(cache_dir or _DEFAULT_CACHE_DIR)}"
        )
    if total_ram > host.ram_available_mb:
        shortfalls.append(
            f"RAM: minimum-knob budget {total_ram} MB exceeds "
            f"{host.ram_available_mb} MB available"
        )
        # Suggest the heaviest droppable slot.
        heaviest = max(ram_budget.items(), key=lambda kv: kv[1])
        suggestions.append(
            f"drop {heaviest[0]} (saves ~{heaviest[1]} MB RAM); "
            "see `supergraph pro check --help` for slot overrides"
        )

    if use_gpu and total_vram > host.gpu_vram_free_mb:
        # Layered allocation: bonsai-first. Try removing reranker, then
        # vision, from the GPU side.
        warnings.append(
            f"VRAM: full offload would need {total_vram} MB, only "
            f"{host.gpu_vram_free_mb} MB free; falling back to layered "
            "allocation (bonsai → reranker → vision)."
        )
        # Recompute layered budget.
        offload_priority = [
            cid for cid in component_ids if cid.startswith("ingest:bonsai-")
        ] + [
            cid for cid in component_ids if cid.startswith("reranker:")
        ] + [
            cid for cid in component_ids if cid.startswith("vision:")
        ] + [
            cid for cid in component_ids if cid.startswith("embedder:")
        ]
        used_vram = 0
        kept_on_gpu: set[str] = set()
        for cid in offload_priority:
            cost = cache.components[cid].vram_mb_full_offload
            if cost > 0 and used_vram + cost <= host.gpu_vram_free_mb:
                used_vram += cost
                kept_on_gpu.add(cid)
        # CPU-side RAM grows for components that did NOT make it onto GPU.
        for cid in component_ids:
            if cid not in kept_on_gpu and vram_budget.get(cid, 0) > 0:
                e = cache.components[cid]
                ram_budget[cid] = max(e.ram_mb_min, e.ram_mb_idle)
                vram_budget[cid] = 0
        total_ram = sum(ram_budget.values())
        total_vram = used_vram
        if total_ram > host.ram_available_mb:
            shortfalls.append(
                f"RAM: even with layered VRAM offload, host needs "
                f"{total_ram} MB but only {host.ram_available_mb} MB "
                f"available"
            )

    if shortfalls:
        return ResolvedConfig(
            spec=spec, host=host, fits=False,
            ram_budget_mb=ram_budget, vram_budget_mb=vram_budget,
            shortfalls=shortfalls, suggestions=suggestions, warnings=warnings,
            calibration_source="measured",
            calibration_age_s=_age_s(cache.measured_at),
        )

    # Step 3: pick maximal knobs that still fit. Knob values come from
    # the calibration entry's `extra` dict (set by the probe runner) so
    # we never hard-code "tier X needs N MB". The probe ran the actual
    # model at concrete knobs and recorded both the knob value and the
    # resulting RAM use; we just check which measurement still fits.
    bonsai_id = next((cid for cid in component_ids
                      if cid.startswith("ingest:bonsai-")), None)
    n_ctx = _FALLBACK_N_CTX
    bonsai_n_batch = _FALLBACK_N_BATCH
    if bonsai_id is not None:
        e = cache.components[bonsai_id]
        # ram_left = how much host RAM remains for bonsai once every
        # other component takes its share. Compare each measured tier
        # against this; pick the largest knob whose recorded RAM fits.
        ram_left = host.ram_available_mb - sum(
            v for k, v in ram_budget.items() if k != bonsai_id)
        n_ctx_max = _maybe_int(e.extra.get(_EXTRA_N_CTX_MAX))
        n_ctx_default = _maybe_int(e.extra.get(_EXTRA_N_CTX_DEFAULT))
        n_ctx_min = _maybe_int(e.extra.get(_EXTRA_N_CTX_MIN))
        n_batch_default = _maybe_int(e.extra.get(_EXTRA_N_BATCH))
        if n_ctx_max and e.ram_mb_max and e.ram_mb_max <= ram_left:
            n_ctx = n_ctx_max
            bonsai_n_batch = n_batch_default or _FALLBACK_N_BATCH
        elif n_ctx_default and e.ram_mb_at_default and e.ram_mb_at_default <= ram_left:
            n_ctx = n_ctx_default
            bonsai_n_batch = n_batch_default or _FALLBACK_N_BATCH
        elif n_ctx_min:
            n_ctx = n_ctx_min
            bonsai_n_batch = _FALLBACK_N_BATCH
        # else: leave fallbacks (cache extra empty, e.g. legacy probe).

    # Reranker max length and embedder batch read straight from the
    # default measurement; no host-RAM ladder.
    reranker_id = next((cid for cid in component_ids
                        if cid.startswith("reranker:")), None)
    reranker_max = 0
    if reranker_id is not None:
        e = cache.components[reranker_id]
        reranker_max = _maybe_int(e.extra.get(_EXTRA_RERANKER_MAX)) or _FALLBACK_RERANKER_MAX

    embedder_id = next((cid for cid in component_ids
                        if cid.startswith("embedder:")), None)
    embed_batch = _FALLBACK_EMBED_BATCH
    if embedder_id is not None:
        e = cache.components[embedder_id]
        embed_batch = _maybe_int(e.extra.get(_EXTRA_EMBED_BATCH)) or _FALLBACK_EMBED_BATCH

    # GPU layer counts: -1 if cached layered allocation kept it on GPU.
    bonsai_n_gpu_layers = -1 if (use_gpu and bonsai_id and vram_budget.get(bonsai_id, 0) > 0) else 0
    reranker_gpu_layers = -1 if (use_gpu and reranker_id and vram_budget.get(reranker_id, 0) > 0) else 0
    vision_offload = bool(use_gpu and any(
        cid.startswith("vision:") and vram_budget.get(cid, 0) > 0
        for cid in component_ids
    ))

    # Projected TPS: best-available - GPU number when offloaded, CPU
    # number at logical-thread count when not.
    projected: dict[str, float] = {}
    for cid in component_ids:
        e = cache.components[cid]
        if use_gpu and vram_budget.get(cid, 0) > 0 and e.tps_gpu_full_offload:
            projected[cid] = e.tps_gpu_full_offload
        elif e.tps_cpu_threads:
            # Pick the entry closest to host.cpu_cores_logical.
            best_key = min(
                e.tps_cpu_threads.keys(),
                key=lambda k: abs(int(k) - host.cpu_cores_logical),
            )
            projected[cid] = e.tps_cpu_threads[best_key]

    # Tightness warnings.
    if total_ram > int(host.ram_available_mb * 0.85):
        warnings.append(
            f"RAM tight: budget {total_ram} MB / available "
            f"{host.ram_available_mb} MB. Consider closing other apps "
            "or dropping a slot."
        )
    if not use_gpu and bonsai_id is not None:
        cpu_tps = projected.get(bonsai_id, 0.0)
        gpu_tps = cache.components[bonsai_id].tps_gpu_full_offload
        if gpu_tps and cpu_tps:
            warnings.append(
                f"GPU not detected: bonsai will run at ~{cpu_tps:.0f} tps "
                f"(measured GPU: ~{gpu_tps:.0f} tps). Run "
                "`supergraph pro probe` if you have CUDA installed."
            )

    age_s = _age_s(cache.measured_at)
    if age_s is not None and age_s > 30 * 86400:
        warnings.append(
            f"calibration is {age_s // 86400} days old; consider "
            "`supergraph pro probe --refresh`"
        )

    return ResolvedConfig(
        spec=spec, host=host, fits=True,
        n_ctx=n_ctx,
        bonsai_n_batch=bonsai_n_batch,
        bonsai_n_gpu_layers=bonsai_n_gpu_layers,
        reranker_max_len=reranker_max,
        reranker_gpu_layers=reranker_gpu_layers,
        embed_batch=embed_batch,
        vision_offload=vision_offload,
        projected_tps=projected,
        ram_budget_mb=ram_budget,
        vram_budget_mb=vram_budget,
        warnings=warnings,
        calibration_source="measured",
        calibration_age_s=age_s,
    )


def _age_s(measured_at: str) -> int | None:
    """Seconds since ``measured_at`` ISO timestamp. None on parse failure."""
    if not measured_at:
        return None
    try:
        when = datetime.fromisoformat(measured_at.replace("Z", "+00:00"))
        return int((datetime.now(timezone.utc) - when).total_seconds())
    except (ValueError, TypeError):
        return None


def _maybe_int(value: Any) -> int | None:
    """Best-effort int coerce. Returns None for None / non-numeric values.

    Used to read scalar knob values from CalibrationEntry.extra dicts,
    which can carry arbitrary user-supplied JSON.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
