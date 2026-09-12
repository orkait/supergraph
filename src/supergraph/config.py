"""Typed configuration for supergraph via msgspec Structs.

One file, sectioned by engine. Missing keys auto-fill from defaults.

Load order (each layer overrides the previous):
    1. config.py dataclass defaults (source of truth for types + default values)
    2. supergraph.json file (per-deployment overrides)
    3. SUPERGRAPH_* environment variables (Docker/k8s friendly)
    4. Constructor kwargs (code-level, highest priority)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import msgspec


_log = logging.getLogger(__name__)


def _coerce_remember_weights(weights) -> list[float]:
    """Accept 3 or 4 weights for REMEMBER fusion.

    Fusion interprets the list based on length + graph_signal_enabled:
      4 weights, graph on   -> [vec, bm25, recency, graph]  (additive)
      3 weights, graph on   -> [vec, bm25, graph]           (graph replaces recency)
      3 weights, graph off  -> [vec, bm25, recency]         (classic)
    See intelligence.py::_remember for the matching dispatch.
    """
    parsed = [float(w) for w in weights]
    if len(parsed) not in (3, 4):
        raise ValueError(f"remember_weights must have length 3 or 4, got {len(parsed)}")
    return parsed


class CoreConfig(msgspec.Struct, frozen=True):
    ceiling_mb: int = 256
    initial_capacity: int = 1024
    compact_threshold: float = 0.2
    string_gc_threshold: float = 3.0
    eviction_target_ratio: float = 0.8
    use_compression: bool = False
    protected_kinds: list[str] = msgspec.field(default_factory=lambda: ["schema", "config", "system"])


class VectorConfig(msgspec.Struct, frozen=True):
    embedder: str = "default"
    embedder_model: str | None = None
    embedder_dims: int | None = None
    embedder_max_length: int = 512
    gpu_layers: int = 0
    quantize_binary: bool = False
    similarity_threshold: float = 0.85
    duplicate_threshold: float = 0.95
    search_oversample: int = 16
    model2vec_model: str = "minishlab/M2V_base_output"
    model_cache_dir: str | None = "./models"


class DocumentConfig(msgspec.Struct, frozen=True):
    fts_tokenizer: str = "porter unicode61"
    chunk_max_size: int = 2000
    chunk_overlap: int = 50
    summary_max_length: int = 200
    fts_full_text: bool = True
    vision_model: str = "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"
    vision_base_url: str | None = None
    vision_max_tokens: int = 512


class DslConfig(msgspec.Struct, frozen=True):
    cost_threshold: int = 100_000
    plan_cache_size: int = 256
    auto_optimize: bool = False
    optimize_interval: int = 500
    recall_decay: float = 0.5912428069710964
    remember_weights: list[float] = msgspec.field(default_factory=lambda: [0.52, 0.25, 0.15, 0.08])
    fusion_method: str = "weighted"  # "rrf" or "weighted"
    rrf_k: float = 60.0
    recency_half_life_days: float = 7300.0  # ~20 years
    similar_to_oversample: int = 2
    lexical_search_oversample: int = 3
    nucleus_expansion: bool = False
    nucleus_hops: int = 1
    nucleus_neighbors_per_hop: int = 3
    nucleus_min_text_length: int = 20
    nucleus_allowed_kinds: list[str] = msgspec.field(default_factory=lambda: ["message", "chunk", "section"])
    sentence_query_expansion: bool = True
    enable_sentence_nodes: bool = True
    enable_rollback: bool = True
    graph_signal_enabled: bool = True
    entity_extractor: str = "tinybert_onnx"
    # NER entity extraction is opt-in. Default off: writes do not require
    # onnxruntime + the NER model and never extract entities unless a model dir
    # is set here (or via SUPERGRAPH_DSL_ENTITY_MODEL_DIR), e.g.
    # f"{os.environ.get('MODELS_DIR', './models')}/tinybert-ner".
    entity_model_dir: str | None = None
    entity_score_threshold: float = 0.6
    entity_max_length: int = 256
    reranker: str | None = None
    reranker_model_dir: str | None = f"{os.environ.get('MODELS_DIR', './models')}/jina-reranker-v3/jina-reranker-v3-Q8_0.gguf"
    reranker_projector_path: str | None = f"{os.environ.get('MODELS_DIR', './models')}/jina-reranker-v3/projector.safetensors"
    reranker_max_length: int = 2048
    # CPU-only by default. Set to -1 (offload all layers) or N (offload N
    # layers) when a CUDA-built llama-cpp-python wheel is installed AND the
    # caller wants GPU offload. Matches vector.gpu_layers and BonsaiIngestor's
    # n_gpu_layers defaults so the install is portable and never silently
    # grabs a GPU. supergraph.gpu.setup() / SuperGraph(profile="pro") flip
    # this on automatically when GPU bind succeeds.
    reranker_gpu_layers: int = 0
    cache_gc_threshold: int = 200


class VaultConfig(msgspec.Struct, frozen=True):
    enabled: bool = False
    path: str | None = None
    auto_sync: bool = True


class PersistenceConfig(msgspec.Struct, frozen=True):
    enabled: bool = True
    wal_hard_limit: int = 100_000
    auto_checkpoint_threshold: int = 50_000
    log_retention_days: int = 7
    busy_timeout_ms: int = 5000
    enable_wal: bool = True


class RetentionConfig(msgspec.Struct, frozen=True):
    blob_warm_days: int = 30
    blob_archive_days: int = 90
    blob_delete_days: int = 365


class ServerConfig(msgspec.Struct, frozen=True):
    cors_origins: list[str] = msgspec.field(default_factory=lambda: ["*"])
    ingest_root: str | None = None
    auth_token: str | None = None
    rate_limit_rpm: int = 120
    rate_limit_window: int = 60
    max_query_length: int = 10_000
    max_batch_size: int = 1000


class EvolutionConfig(msgspec.Struct, frozen=True):
    similarity_buffer_size: int = 100
    max_rules: int = 50
    min_cooldown: int = 10
    history_retention: int = 1000


class ComputeConfig(msgspec.Struct, frozen=True):
    profile: str | None = None
    ner_threads: int | None = None
    embed_threads: int | None = None
    rerank_threads: int | None = None
    embed_batch_size: int | None = None
    disable_load_scaling: bool = False
    disable_battery_scaling: bool = False


class IngestConfig(msgspec.Struct, frozen=True):
    # NL->DSL ingestion backend. None disables NL ingest (explicit opt-in);
    # "cloud" uses the litellm multi-provider CloudIngestor; "local" uses the
    # GGUF BonsaiIngestor (requires a model; wire via SuperGraph(profile="pro")).
    nl_backend: str | None = None
    # Candidate provider-prefixed model ids for the cloud chain. Empty list =
    # use resolve.DEFAULT_FREE_FIRST_CHAIN.
    nl_models: list[str] = msgspec.field(default_factory=list)
    free_first: bool = True
    nl_max_tokens: int = 1000
    nl_temperature: float = 0.0


class SuperGraphConfig(msgspec.Struct, frozen=True):
    core: CoreConfig = msgspec.field(default_factory=CoreConfig)
    vector: VectorConfig = msgspec.field(default_factory=VectorConfig)
    document: DocumentConfig = msgspec.field(default_factory=DocumentConfig)
    dsl: DslConfig = msgspec.field(default_factory=DslConfig)
    vault: VaultConfig = msgspec.field(default_factory=VaultConfig)
    persistence: PersistenceConfig = msgspec.field(default_factory=PersistenceConfig)
    retention: RetentionConfig = msgspec.field(default_factory=RetentionConfig)
    server: ServerConfig = msgspec.field(default_factory=ServerConfig)
    evolution: EvolutionConfig = msgspec.field(default_factory=EvolutionConfig)
    compute: ComputeConfig = msgspec.field(default_factory=ComputeConfig)
    ingest: IngestConfig = msgspec.field(default_factory=IngestConfig)


_decoder = msgspec.json.Decoder(SuperGraphConfig)
_encoder = msgspec.json.Encoder()

# Section name -> (config class, field name -> type)
_SECTION_MAP: dict[str, tuple[type, dict[str, type]]] = {
    "core": (CoreConfig, {f: type(getattr(CoreConfig(), f)) for f in CoreConfig.__struct_fields__}),
    "vector": (VectorConfig, {f: type(getattr(VectorConfig(), f)) for f in VectorConfig.__struct_fields__}),
    "document": (DocumentConfig, {f: type(getattr(DocumentConfig(), f)) for f in DocumentConfig.__struct_fields__}),
    "dsl": (DslConfig, {f: type(getattr(DslConfig(), f)) for f in DslConfig.__struct_fields__}),
    "vault": (VaultConfig, {f: type(getattr(VaultConfig(), f)) for f in VaultConfig.__struct_fields__}),
    "persistence": (PersistenceConfig, {f: type(getattr(PersistenceConfig(), f)) for f in PersistenceConfig.__struct_fields__}),
    "retention": (RetentionConfig, {f: type(getattr(RetentionConfig(), f)) for f in RetentionConfig.__struct_fields__}),
    "server": (ServerConfig, {f: type(getattr(ServerConfig(), f)) for f in ServerConfig.__struct_fields__}),
    "evolution": (EvolutionConfig, {f: type(getattr(EvolutionConfig(), f)) for f in EvolutionConfig.__struct_fields__}),
    "compute": (ComputeConfig, {f: type(getattr(ComputeConfig(), f)) for f in ComputeConfig.__struct_fields__}),
    "ingest": (IngestConfig, {f: type(getattr(IngestConfig(), f)) for f in IngestConfig.__struct_fields__}),
}

# Flat kwarg name -> (section, field) for constructor shortcuts
_KWARG_SHORTCUTS: dict[str, tuple[str, str]] = {
    "ceiling_mb":           ("core", "ceiling_mb"),
    "initial_capacity":     ("core", "initial_capacity"),
    "eviction_target_ratio":("core", "eviction_target_ratio"),
    "use_compression":      ("core", "use_compression"),
    "remember_weights":     ("dsl", "remember_weights"),
    "fusion_method":        ("dsl", "fusion_method"),
    "rrf_k":                ("dsl", "rrf_k"),
    "recall_decay":         ("dsl", "recall_decay"),
    "recency_half_life_days": ("dsl", "recency_half_life_days"),
    "similar_to_oversample": ("dsl", "similar_to_oversample"),
    "lexical_search_oversample": ("dsl", "lexical_search_oversample"),
    "nucleus_expansion":    ("dsl", "nucleus_expansion"),
    "nucleus_hops":         ("dsl", "nucleus_hops"),
    "nucleus_neighbors_per_hop": ("dsl", "nucleus_neighbors_per_hop"),
    "nucleus_min_text_length": ("dsl", "nucleus_min_text_length"),
    "nucleus_allowed_kinds":   ("dsl", "nucleus_allowed_kinds"),
    "sentence_query_expansion": ("dsl", "sentence_query_expansion"),
    "enable_sentence_nodes":    ("dsl", "enable_sentence_nodes"),
    "enable_rollback":          ("dsl", "enable_rollback"),
    "graph_signal_enabled": ("dsl", "graph_signal_enabled"),
    "entity_extractor":     ("dsl", "entity_extractor"),
    "entity_model_dir":     ("dsl", "entity_model_dir"),
    "entity_score_threshold": ("dsl", "entity_score_threshold"),
    "entity_max_length":    ("dsl", "entity_max_length"),
    "reranker":             ("dsl", "reranker"),
    "reranker_model_dir":   ("dsl", "reranker_model_dir"),
    "reranker_projector_path": ("dsl", "reranker_projector_path"),
    "reranker_max_length":  ("dsl", "reranker_max_length"),
    "reranker_gpu_layers":  ("dsl", "reranker_gpu_layers"),
    "quantize_binary":      ("vector", "quantize_binary"),
    "search_oversample":    ("vector", "search_oversample"),
    "similarity_threshold": ("vector", "similarity_threshold"),
    "duplicate_threshold":  ("vector", "duplicate_threshold"),
    "fts_tokenizer":        ("document", "fts_tokenizer"),
    "nl_backend":           ("ingest", "nl_backend"),
    "nl_models":            ("ingest", "nl_models"),
    "nl_max_tokens":        ("ingest", "nl_max_tokens"),
    "nl_temperature":       ("ingest", "nl_temperature"),
}


def _coerce(value: str, target_type: type):
    """Coerce a string env var value to the target type."""
    if target_type is bool or target_type is type(True):
        return value.lower() in ("1", "true", "yes")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is list:
        return [v.strip() for v in value.split(",") if v.strip()]
    if target_type is type(None):
        return value if value else None
    return value


def load_config(path: str | Path | None = None) -> SuperGraphConfig:
    """Load config overrides from a JSON file and merge onto defaults.

    The file is expected to be a partial dict - only sections/fields the user
    changed. Missing sections and fields fall back to config.py defaults.
    Returns full defaults if the file doesn't exist or is empty.
    """
    if path is None:
        return SuperGraphConfig()
    p = Path(path)
    if not p.exists():
        return SuperGraphConfig()
    raw = p.read_bytes().strip()
    if not raw:
        return SuperGraphConfig()
    try:
        overrides = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        _log.warning("config parse error in %s: %s - using defaults", p, e)
        return SuperGraphConfig()
    if not isinstance(overrides, dict) or not overrides:
        return SuperGraphConfig()
    return _rebuild_config(SuperGraphConfig(), overrides)


def save_config(config: SuperGraphConfig, path: str | Path) -> None:
    """Save config overrides to a JSON file.

    Only writes values that differ from defaults. This keeps supergraph.json
    minimal and ensures users benefit from future default improvements.
    """
    defaults = SuperGraphConfig()
    current = msgspec.json.decode(msgspec.json.encode(config))
    default_data = msgspec.json.decode(msgspec.json.encode(defaults))

    diff: dict = {}
    for section, fields in current.items():
        if not isinstance(fields, dict):
            continue
        section_diff = {}
        for field, value in fields.items():
            default_val = default_data.get(section, {}).get(field)
            if value != default_val:
                section_diff[field] = value
        if section_diff:
            diff[section] = section_diff

    p = Path(path)
    if diff:
        p.write_text(json.dumps(diff, indent=2) + "\n")
    elif p.exists():
        p.unlink()


def apply_env_overrides(config: SuperGraphConfig) -> SuperGraphConfig:
    """Override config fields from SUPERGRAPH_* environment variables.

    Convention: SUPERGRAPH_{SECTION}_{FIELD} in uppercase.
    Examples:
        SUPERGRAPH_CORE_CEILING_MB=512
        SUPERGRAPH_DSL_RECALL_DECAY=0.5
        SUPERGRAPH_DSL_REMEMBER_WEIGHTS=0.40,0.30,0.15,0.10,0.05
        SUPERGRAPH_VECTOR_SEARCH_OVERSAMPLE=10
        SUPERGRAPH_VECTOR_SIMILARITY_THRESHOLD=0.80
        SUPERGRAPH_SERVER_CORS_ORIGINS=http://localhost:3000,http://localhost:8080
        SUPERGRAPH_SERVER_AUTH_TOKEN=secret123
    """
    updates: dict[str, dict[str, object]] = {}

    for env_key, env_val in os.environ.items():
        if not env_key.startswith("SUPERGRAPH_") or not env_val:
            continue

        parts = env_key[len("SUPERGRAPH_"):].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, field = parts[0], parts[1]

        if section not in _SECTION_MAP:
            continue
        _, field_types = _SECTION_MAP[section]
        if field not in field_types:
            continue

        target_type = field_types[field]
        try:
            if field == "remember_weights":
                try:
                    parsed = _coerce_remember_weights(env_val.split(","))
                except ValueError as e:
                    _log.warning("invalid SUPERGRAPH_DSL_REMEMBER_WEIGHTS: %s", e)
                    continue
                updates.setdefault(section, {})[field] = parsed
            elif field == "protected_kinds" or field == "cors_origins":
                updates.setdefault(section, {})[field] = [v.strip() for v in env_val.split(",")]
            else:
                updates.setdefault(section, {})[field] = _coerce(env_val, target_type)
        except (ValueError, TypeError) as e:
            _log.warning("invalid env var %s=%s: %s", env_key, env_val, e)

    if not updates:
        return config

    return _rebuild_config(config, updates)


def merge_kwargs(config: SuperGraphConfig, **kwargs) -> SuperGraphConfig:
    """Override config fields from SuperGraph(...) constructor kwargs.

    Two shapes of kwargs are accepted:

    1. Flat shortcuts for tuning knobs (see _KWARG_SHORTCUTS), e.g.
       ceiling_mb, remember_weights, recall_decay, search_oversample.
    2. Top-level convenience kwargs that map to multi-field config
       updates: embedder, ingest_root, vault, retention (dict).
       These are the primary public API - SuperGraph(embedder=...) is
       how users construct the store.
    """
    updates: dict[str, dict[str, object]] = {}

    for kwarg_name, (section, field) in _KWARG_SHORTCUTS.items():
        if kwarg_name in kwargs:
            val = kwargs[kwarg_name]
            current_val = getattr(getattr(config, section), field)
            if val != current_val:
                updates.setdefault(section, {})[field] = val

    if "embedder" in kwargs:
        emb = kwargs["embedder"]
        emb_name = emb if isinstance(emb, str) else "custom"
        if emb is None:
            emb_name = "none"
        updates.setdefault("vector", {})["embedder"] = emb_name

    if "ingest_root" in kwargs and kwargs["ingest_root"] is not None:
        updates.setdefault("server", {})["ingest_root"] = kwargs["ingest_root"]

    if "vault" in kwargs and kwargs["vault"] is not None:
        updates.setdefault("vault", {})["enabled"] = True
        updates["vault"]["path"] = kwargs["vault"]

    if "retention" in kwargs and kwargs["retention"] is not None:
        r = kwargs["retention"]
        for key in ("blob_warm_days", "blob_archive_days", "blob_delete_days"):
            if key in r:
                updates.setdefault("retention", {})[key] = r[key]

    if not updates:
        return config

    return _rebuild_config(config, updates)


def _rebuild_config(
    config: SuperGraphConfig,
    updates: dict[str, dict[str, object]],
) -> SuperGraphConfig:
    """Rebuild a frozen config with section-level field overrides.

    Unknown keys (e.g. fields from older supergraph versions that have
    since been removed) are silently dropped + logged, so loading an old
    tuned_config.json does not crash with ``TypeError: Unexpected
    keyword argument``. Config forward-compat > config fidelity.
    """
    sections = {}
    for section_name in SuperGraphConfig.__struct_fields__:
        current = getattr(config, section_name)
        if section_name not in updates:
            sections[section_name] = current
            continue
        known = set(current.__struct_fields__)
        field_overrides = updates[section_name]
        for k in list(field_overrides):
            if k not in known:
                _log.warning("config: dropping unknown key %s.%s", section_name, k)
                field_overrides.pop(k)
        current_dict = {f: getattr(current, f) for f in known}
        current_dict.update(field_overrides)
        if section_name == "dsl" and "remember_weights" in current_dict:
            current_dict["remember_weights"] = _coerce_remember_weights(current_dict["remember_weights"])
        section_cls = type(current)
        sections[section_name] = section_cls(**current_dict)

    return SuperGraphConfig(**sections)
