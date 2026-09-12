"""FastAPI server for the supergraph playground."""

from __future__ import annotations

import os
import threading
import time as _time
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError as e:
    raise ImportError(
        "supergraph.server requires the `playground` extra. "
        "Install with: pip install 'supergraph[playground]'"
    ) from e

from supergraph import SuperGraph
from supergraph.core.errors import SuperGraphError
from supergraph.core.types import encode_json

app = FastAPI(title="supergraph playground")

_CORS_ORIGINS = os.environ.get("SUPERGRAPH_CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Auth + Rate Limiting Middleware ---

_AUTH_TOKEN = os.environ.get("SUPERGRAPH_AUTH_TOKEN")
_RATE_LIMIT_RPM = int(os.environ.get("SUPERGRAPH_RATE_LIMIT_RPM", "120"))

_rate_buckets: dict[str, list[float]] = defaultdict(list)
_rate_cleanup_counter = 0


def _check_rate_limit(client_ip: str) -> bool:
    """Returns True if request is allowed."""
    global _rate_cleanup_counter
    now = _time.time()
    window = float(_RATE_LIMIT_WINDOW)
    bucket = _rate_buckets[client_ip]
    _rate_buckets[client_ip] = [t for t in bucket if now - t < window]
    if len(_rate_buckets[client_ip]) >= _RATE_LIMIT_RPM:
        return False
    _rate_buckets[client_ip].append(now)

    _rate_cleanup_counter += 1
    if _rate_cleanup_counter % 100 == 0:
        stale = [ip for ip, ts in _rate_buckets.items() if not ts or now - max(ts) > float(_RATE_LIMIT_WINDOW)]
        for ip in stale:
            del _rate_buckets[ip]

    return True


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    if _AUTH_TOKEN is not None:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header[7:] != _AUTH_TOKEN:
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"},
            headers={"Retry-After": str(_RATE_LIMIT_WINDOW)},
        )

    return await call_next(request)


# ---------------------------------------------------------------------------
# Module-level store (created lazily)
# ---------------------------------------------------------------------------

_store: SuperGraph | None = None


def _get_store() -> SuperGraph:
    global _store
    if _store is None:
        db_path = os.environ.get("SUPERGRAPH_DB_PATH")
        config_path = os.environ.get("SUPERGRAPH_CONFIG")
        ingest_root = os.environ.get("SUPERGRAPH_INGEST_ROOT", os.getcwd())
        kwargs: dict = {}
        if db_path:
            kwargs["path"] = db_path
        if config_path:
            kwargs["config_path"] = config_path
        if ingest_root:
            kwargs["ingest_root"] = ingest_root
        # SUPERGRAPH_NER falsy -> entity_model_dir=None, skips the NER extractor.
        if os.environ.get("SUPERGRAPH_NER", "1").strip().lower() in ("0", "false", "off", "no"):
            kwargs["entity_model_dir"] = None
        # FastAPI serves these sync endpoints from a threadpool, so multiple
        # requests can hit this shared store concurrently. The storage engine
        # is single-writer - concurrent writers corrupt node_count / id_to_slot
        # and crash ColumnStore._grow. queued=True serializes execute() through
        # one worker, making the shared instance safe across request threads.
        kwargs["queued"] = True
        _store = SuperGraph(**kwargs)
    return _store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_payload(result) -> dict[str, Any]:
    return {
        "kind": result.kind,
        "data": result.data,
        "count": result.count,
        "elapsed_us": result.elapsed_us,
    }


def _json_bytes_response(payload: Any) -> Response:
    return Response(content=encode_json(payload), media_type="application/json")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ExecuteRequest(BaseModel):
    query: str
    namespace: str | None = None


class BatchRequest(BaseModel):
    queries: list[str]
    namespace: str | None = None


class ConfigRequest(BaseModel):
    ceiling_mb: int | None = None
    cost_threshold: int | None = None
    eviction_target_ratio: float | None = None


class NLIngestRequest(BaseModel):
    text: str
    msg_id: str | None = None
    session_id: str = "default"
    role: str = "user"
    dry_run: bool = False


class MediaIngestRequest(BaseModel):
    id: str
    mime: str
    data_b64: str
    namespace: str | None = None
    prompt: str | None = None
    models: list[str] | None = None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

_MAX_QUERY_LENGTH = int(os.environ.get("SUPERGRAPH_MAX_QUERY_LENGTH", "10000"))
_MAX_BATCH_SIZE = int(os.environ.get("SUPERGRAPH_MAX_BATCH_SIZE", "1000"))
_RATE_LIMIT_WINDOW = int(os.environ.get("SUPERGRAPH_RATE_LIMIT_WINDOW", "60"))


def _validate_query(query: str) -> str | None:
    """Validate query input. Returns error message or None if valid."""
    if not query or not query.strip():
        return "Empty query"
    if len(query) > _MAX_QUERY_LENGTH:
        return f"Query exceeds maximum length ({_MAX_QUERY_LENGTH} chars)"
    if "\x00" in query:
        return "Query contains null bytes"
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/execute")
def execute(req: ExecuteRequest):
    err = _validate_query(req.query)
    if err:
        return _json_bytes_response({"kind": "error", "data": err, "count": 0, "elapsed_us": 0})
    store = _get_store()
    try:
        result = store.execute(req.query, namespace=req.namespace)
    except Exception as exc:
        # Catch all exceptions (including ValueError, KeyError, etc.) and
        # return them as soft 200 errors. Pre-fix only SuperGraphError was
        # caught, so a programmer bug anywhere in the executor surfaced as
        # an HTTP 500 with a stack trace leaked to the client, and in
        # batch mode caused the whole batch to abort (bug #80). Logging
        # here gives us the observability we used to get from the 500.
        import logging
        logging.getLogger(__name__).warning(
            "execute: %s: %s", type(exc).__name__, exc,
        )
        return _json_bytes_response({
            "kind": "error",
            "data": f"{type(exc).__name__}: {exc}",
            "count": 0,
            "elapsed_us": 0,
        })
    return _json_bytes_response(_result_payload(result))


@app.post("/api/execute-batch")
def execute_batch(req: BatchRequest):
    if len(req.queries) > _MAX_BATCH_SIZE:
        return _json_bytes_response([{"kind": "error", "data": f"Batch exceeds {_MAX_BATCH_SIZE} queries", "count": 0, "elapsed_us": 0}])
    store = _get_store()
    results = []
    import logging as _logging
    for q in req.queries:
        err = _validate_query(q)
        if err:
            results.append({"kind": "error", "data": err, "count": 0, "elapsed_us": 0})
            continue
        try:
            r = store.execute(q, namespace=req.namespace)
            results.append(_result_payload(r))
        except Exception as exc:
            # Same widening as /api/execute — a single unexpected error
            # shouldn't abort the remaining queries in the batch.
            _logging.getLogger(__name__).warning(
                "execute-batch: %s: %s", type(exc).__name__, exc,
            )
            results.append({
                "kind": "error",
                "data": f"{type(exc).__name__}: {exc}",
                "count": 0,
                "elapsed_us": 0,
            })
    return _json_bytes_response(results)


def _detect_cpu_quota() -> int:
    """Effective CPU count from the cgroup CFS quota (cgroup v2 then v1), falling
    back to affinity/cpu_count. Quota reflects the real limit; affinity sees host cores."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return max(1, round(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            return max(1, round(q / p))
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 2


_bonsai = None
_bonsai_lock = threading.Lock()


def _get_bonsai():
    """Lazily build a CPU BonsaiIngestor (n_gpu_layers=0) from
    SUPERGRAPH_BONSAI_GGUF. Bypasses the CUDA-gated pro resolver, which refuses
    to build on a host without a GPU - Bonsai itself runs fine on CPU."""
    global _bonsai
    if _bonsai is None:
        with _bonsai_lock:
            if _bonsai is None:
                gguf = os.environ.get("SUPERGRAPH_BONSAI_GGUF")
                if not gguf:
                    raise SuperGraphError(
                        "Bonsai not configured: set SUPERGRAPH_BONSAI_GGUF to the GGUF path"
                    )
                from supergraph.bonsai_ingestor import BonsaiIngestor, _DEFAULT_LITE_PROMPT_PATH
                threads_env = os.environ.get("SUPERGRAPH_BONSAI_THREADS")
                if threads_env:
                    n_threads = int(threads_env)
                else:
                    # Use the cgroup CFS quota, not the visible core count. llama.cpp's
                    # default (and sched_getaffinity) read HOST cores, which massively
                    # oversubscribes a quota-limited container (e.g. Railway: 24 vCPU
                    # quota on a 48-core host). Reserve one only when there's headroom.
                    avail = _detect_cpu_quota()
                    n_threads = avail - 1 if avail > 2 else avail
                    n_threads = max(1, n_threads)
                _bonsai = BonsaiIngestor(
                    model_path=gguf,
                    gs=_get_store(),
                    skill_path=str(_DEFAULT_LITE_PROMPT_PATH),
                    n_gpu_layers=0,
                    n_threads=n_threads,
                )
    return _bonsai


@app.post("/api/ingest")
def ingest(req: NLIngestRequest):
    """Natural-language -> DSL, routed by ``config.ingest.nl_backend``.

    ``"cloud"`` runs the litellm multi-provider CloudIngestor (PR #198/#201);
    anything else falls back to the local Ternary-Bonsai GGUF. The backend is
    env-driven - ``SUPERGRAPH_INGEST_NL_BACKEND=cloud`` - so the same image
    serves both without a code change.

    Pre-fix this endpoint called Bonsai unconditionally, which meant the
    [cloud-cpu] image (no GGUF by design) could only reach NL ingestion
    in-process or via MCP. HTTP is the only surface a deployed supergraph
    exposes, so on Railway that left NL ingest unreachable entirely.

    ``dry_run=True`` returns the synthesized DSL without writing.
    """
    import logging
    import uuid
    store = _get_store()
    backend = store._config.ingest.nl_backend
    try:
        msg_id = req.msg_id or f"msg_{uuid.uuid4().hex[:16]}"
        if backend == "cloud":
            result = store.ingest_nl(
                req.text,
                msg_id=msg_id,
                session_id=req.session_id,
                role=req.role,
                dry_run=req.dry_run,
            )
        else:
            result = _get_bonsai().ingest(
                req.text,
                msg_id=msg_id,
                session_id=req.session_id,
                role=req.role,
                dry_run=req.dry_run,
            )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "ingest[%s]: %s: %s", backend or "local", type(exc).__name__, exc,
        )
        return _json_bytes_response({
            "kind": "error",
            "data": f"{type(exc).__name__}: {exc}",
            "backend": backend or "local",
        })
    import dataclasses
    try:
        payload = dataclasses.asdict(result) if dataclasses.is_dataclass(result) else dict(result)
    except Exception:
        payload = {k: getattr(result, k) for k in ("statements", "executed", "parsed", "rejected")
                   if hasattr(result, k)}
    # Which engine ran is not otherwise visible in the response, and the two
    # produce different rejection profiles - worth one key to avoid guessing.
    return _json_bytes_response({
        "kind": "ingest", "data": payload, "backend": backend or "local",
    })


_MAX_MEDIA_BYTES = int(os.environ.get("SUPERGRAPH_MAX_MEDIA_BYTES", str(20 * 1024 * 1024)))


def _dsl_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


@app.post("/api/ingest-media")
def ingest_media(req: MediaIngestRequest):
    """Understand media (image/audio) into descriptive text via a cloud multimodal
    model, then store it as an embedded DOCUMENT - so it is searchable like any
    other memory. supergraph owns the understanding; callers just hand over bytes."""
    import base64
    import logging

    from supergraph.ingest.media import MediaUnsupported, understand_media

    try:
        data = base64.b64decode(req.data_b64, validate=True)
    except Exception as exc:
        return _json_bytes_response({"kind": "error", "data": f"invalid data_b64: {exc}"})
    if len(data) > _MAX_MEDIA_BYTES:
        return _json_bytes_response({"kind": "error", "data": f"media exceeds {_MAX_MEDIA_BYTES} bytes"})
    try:
        text = understand_media(data, req.mime, models=req.models, prompt=req.prompt)
    except MediaUnsupported as exc:
        return _json_bytes_response({"kind": "error", "data": str(exc)})
    except Exception as exc:
        logging.getLogger(__name__).warning("ingest-media: %s: %s", type(exc).__name__, exc)
        return _json_bytes_response({"kind": "error", "data": f"{type(exc).__name__}: {exc}"})
    if not text:
        return _json_bytes_response({"kind": "error", "data": "empty understanding"})
    store = _get_store()
    dsl = (
        f"CREATE NODE {_dsl_str(req.id)} kind = \"media\" "
        f"media_mime = {_dsl_str(req.mime)} DOCUMENT {_dsl_str(text)}"
    )
    result = store.execute(dsl, namespace=req.namespace)
    return _json_bytes_response({
        "kind": "media",
        "data": {"id": req.id, "mime": req.mime, "text": text, "chars": len(text),
                 "stored": getattr(result, "kind", None)},
    })


@app.get("/api/graph")
def get_graph():
    store = _get_store()
    nodes = store.get_all_nodes()
    edges = store.get_all_edges()
    return _json_bytes_response({"nodes": nodes, "edges": edges})


@app.post("/api/reset")
def reset():
    """Reset the in-memory graph. For persistent DBs, wipes memory and WAL
    but preserves the script metadata so Run All can repopulate cleanly."""
    store = _get_store()
    store.reset_store(preserve_config=True)
    return {"ok": True}


@app.get("/api/script")
def get_script():
    store = _get_store()
    script = store.get_script()
    return {"script": script}


@app.put("/api/script")
def put_script(req: ExecuteRequest):
    store = _get_store()
    store.set_script(req.query)
    return {"ok": True}


@app.post("/api/config")
def config(req: ConfigRequest):
    store = _get_store()
    changes = {}
    if req.ceiling_mb is not None:
        changes["ceiling_mb"] = req.ceiling_mb
    if req.cost_threshold is not None:
        changes["cost_threshold"] = req.cost_threshold
    if req.eviction_target_ratio is not None:
        changes["eviction_target_ratio"] = req.eviction_target_ratio
    
    if changes:
        store.update_runtime_config(changes)
        
    return {"ok": True}


@app.get("/api/logs")
def get_logs(
    limit: int = 50,
    tag: str | None = None,
    source: str | None = None,
    trace_id: str | None = None,
    since: str | None = None,
):
    """Query the enriched query log."""
    store = _get_store()
    conn = store._conn
    if conn is None:
        return []

    sql = "SELECT id, timestamp, query, elapsed_us, result_count, error, tag, trace_id, source, phase FROM query_log"
    params: list = []
    conditions = []

    if tag:
        conditions.append("tag = ?")
        params.append(tag)
    if source:
        conditions.append("source LIKE ?")
        params.append(f"%{source}%")
    if trace_id:
        conditions.append("trace_id = ?")
        params.append(trace_id)
    if since:
        import datetime
        dt = datetime.datetime.fromisoformat(since)
        conditions.append("timestamp >= ?")
        params.append(dt.timestamp())

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [
        {
            "id": r[0], "timestamp": r[1], "query": r[2],
            "elapsed_us": r[3], "result_count": r[4], "error": r[5],
            "tag": r[6], "trace_id": r[7], "source": r[8], "phase": r[9],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Static files helper
# ---------------------------------------------------------------------------


def mount_static(application: FastAPI, path: str | Path) -> None:
    """Mount a StaticFiles directory on / if it exists."""
    p = Path(path)
    if p.is_dir():
        application.mount(
            "/",
            StaticFiles(directory=str(p), html=True),
            name="playground",
        )
