"""MCP server exposing supergraph as agent-callable tools.

Default mode: in-process SuperGraph() (fast, no HTTP, no playground server).
Remote mode: set SUPERGRAPH_URL=http://host:7200 to forward calls to a
running `supergraph playground` instance.

Launch via the console script (installed by the [mcp] extra):

    supergraph-mcp

or the equivalent module form:

    python -m supergraph.mcp.server

Environment:
    SUPERGRAPH_DB_PATH    path for in-process db (default ./supergraph-mcp.db)
    SUPERGRAPH_PROFILE    "pro" to enable Pro mode (Bonsai + Jina + NER)
    SUPERGRAPH_URL        optional remote playground URL; if set, in-process
                          store is bypassed and DSL is forwarded over HTTP
    SUPERGRAPH_AUTH_TOKEN optional bearer token for the remote URL
    SUPERGRAPH_INGEST_NL_BACKEND  "cloud" to enable the sg_ingest structured
                          NL->graph tool (needs a provider key, e.g. GROQ_API_KEY)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    sys.stderr.write(
        "missing dep 'mcp'. install with: uv pip install 'supergraph[mcp]'\n"
    )
    raise

logger = logging.getLogger("supergraph.mcp")

mcp = FastMCP("supergraph")

_REMOTE_URL = os.environ.get("SUPERGRAPH_URL")
_AUTH_TOKEN = os.environ.get("SUPERGRAPH_AUTH_TOKEN")

_store = None


def _get_store():
    """Lazy in-process SuperGraph singleton."""
    global _store
    if _store is None:
        from supergraph import SuperGraph

        kwargs: dict[str, Any] = {
            "path": os.environ.get("SUPERGRAPH_DB_PATH", "./supergraph-mcp.db"),
        }
        profile = os.environ.get("SUPERGRAPH_PROFILE")
        if profile:
            kwargs["profile"] = profile
        _store = SuperGraph(**kwargs)
    return _store


def _execute_remote(dsl: str) -> dict:
    """Forward DSL to a running playground HTTP API."""
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if _AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {_AUTH_TOKEN}"
    req = urllib.request.Request(
        f"{_REMOTE_URL.rstrip('/')}/api/execute",
        data=json.dumps({"query": dsl}).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _execute(dsl: str) -> dict:
    """Run DSL via in-process or remote, returning a structured dict.

    Errors are caught and returned as {"error": "...", "dsl": dsl} so the
    MCP client sees a clean response instead of a transport-level crash.
    """
    try:
        if _REMOTE_URL:
            return _execute_remote(dsl)
        r = _get_store().execute(dsl)
        return {
            "kind": r.kind,
            "count": r.count,
            "data": r.data,
            "elapsed_us": r.elapsed_us,
            "meta": getattr(r, "meta", None),
        }
    except Exception as e:
        logger.exception("supergraph execute failed")
        return {
            "error": f"{type(e).__name__}: {e}",
            "dsl": dsl,
        }


def _esc(s: str) -> str:
    """Escape a string for embedding in a DSL double-quoted literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _ingest_nl(text: str) -> dict:
    """Structured NL ingestion via gs.ingest_nl (in-process only).

    Extracts entities / relationships / beliefs through the configured LLM
    backend and writes them as nodes + edges. Enable by launching with
    SUPERGRAPH_INGEST_NL_BACKEND=cloud and a provider key (e.g. GROQ_API_KEY).
    """
    if _REMOTE_URL:
        return {"error": "sg_ingest needs the in-process LLM pipeline; "
                         "not available in remote (SUPERGRAPH_URL) mode. Use sg_execute."}
    try:
        r = _get_store().ingest_nl(text)
        return {
            "executed": r.executed,
            "rejected": len(r.rejected),
            "statements": r.statements,
        }
    except Exception as e:
        logger.exception("supergraph ingest_nl failed")
        return {"error": f"{type(e).__name__}: {e}"}


# ── the eight research verbs ──────────────────────────────────────────────────
# Lean by design: one smart read (sg_search), one write (sg_ingest), the research
# frontier (sg_gaps), plus relate/forget/explore and a raw-DSL escape hatch.
# The vector/lexical/fused choice, namespaces, SYS verbs and embedder knobs are
# hidden - drop to sg_execute for those.


@mcp.tool()
def sg_ingest(content: str, structure: bool = False) -> dict:
    """Add evidence to the research corpus so it is retrievable by meaning and
    keyword. The default path is cheap - embeds + indexes the text as one
    evidence record, idempotent on identical content (re-ingest is a no-op).

    structure=True instead runs the slower NL->graph extraction (entities +
    relations via the configured LLM); use it sparingly, only when the text is
    worth structuring. structure mode is in-process only and needs
    SUPERGRAPH_INGEST_NL_BACKEND=cloud + a provider key (e.g. GROQ_API_KEY).

    Example: sg_ingest("Nike signed a sponsorship deal with the national team")
    """
    if structure:
        return _ingest_nl(content)
    node_id = "ev:" + hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
    r = _execute(f'CREATE NODE "{node_id}" kind="evidence" DOCUMENT "{_esc(content)}"')
    if isinstance(r, dict) and "error" in r:
        if "exist" in str(r["error"]).lower():
            return {"id": node_id, "ingested": False, "note": "already present"}
        return r
    return {"id": node_id, "ingested": True}


@mcp.tool()
def sg_search(query: str, limit: int = 10) -> dict:
    """Retrieve the most relevant evidence for a query. The one smart read - it
    fuses semantic, keyword, recency and graph signals (with reranking when
    configured); you never pick a retrieval mode. Use this for all lookups.
    """
    return _execute(f'REMEMBER "{_esc(query)}" LIMIT {int(limit)}')


@mcp.tool()
def sg_answer(question: str) -> dict:
    """Ask a natural-language question; get a synthesised answer grounded in the
    corpus (retrieve + read). Requires an LLM reader configured on the store
    (best in Pro mode, SUPERGRAPH_PROFILE=pro).
    """
    return _execute(f'ANSWER "{_esc(question)}"')


@mcp.tool()
def sg_gaps(limit: int = 10) -> dict:
    """The research frontier - what the corpus is weakest on, so you know what to
    investigate next: under-confident records (kind="gather") and sparse /
    under-sourced entities (kind="expand"). Deduped, capped at ``limit``.
    """
    out: list[dict] = []
    seen: set[str] = set()

    def _add(nid: Any, name: Any, kind: str, reason: str) -> None:
        if nid and nid not in seen:
            seen.add(nid)
            out.append({"id": nid, "name": name, "kind": kind, "reason": reason})

    lc = _execute(f"NODES WHERE confidence < 0.6 LIMIT {int(limit)}")
    for row in (lc.get("data") or []) if isinstance(lc, dict) else []:
        _add(row.get("id"), row.get("name"), "gather", "low confidence")
    sp = _execute(f'NODES WHERE kind = "entity" AND INDEGREE < 1 LIMIT {int(limit)}')
    for row in (sp.get("data") or []) if isinstance(sp, dict) else []:
        _add(row.get("id"), row.get("name"), "expand", "sparse / under-sourced")
    gaps = out[:limit]
    return {"gaps": gaps, "count": len(gaps)}


@mcp.tool()
def sg_relate(source: str, target: str, kind: str) -> dict:
    """Record a connection you know between two records (by id)."""
    return _execute(f'CREATE EDGE "{_esc(source)}" -> "{_esc(target)}" kind = "{_esc(kind)}"')


@mcp.tool()
def sg_forget(node_id: str) -> dict:
    """Retract a record - it stops being visible to search and answer."""
    return _execute(f'RETRACT "{_esc(node_id)}"')


@mcp.tool()
def sg_explore(node_id: str, depth: int = 2, limit: int = 50) -> dict:
    """Follow connections outward from a record - the surrounding subgraph."""
    return _execute(
        f'RECALL FROM "{_esc(node_id)}" DEPTH {int(depth)} LIMIT {int(limit)}'
    )


@mcp.tool()
def sg_execute(dsl: str) -> dict:
    """Escape hatch: run raw DSL. Reach for this only when the eight verbs above
    don't cover what you need (NODES WHERE / COUNT / SYS / namespaces / pure
    lexical or vector ranking). See the supergraph-dsl skill for the grammar.
    """
    return _execute(dsl)


@mcp.resource("graph://stats")
def stats_resource() -> str:
    """Live node count + connection mode."""
    n = _execute("COUNT NODES")
    return json.dumps(
        {
            "nodes": n.get("count", 0) if "error" not in n else None,
            "error": n.get("error"),
            "mode": "remote" if _REMOTE_URL else "in-process",
            "remote_url": _REMOTE_URL,
            "db_path": os.environ.get("SUPERGRAPH_DB_PATH", "./supergraph-mcp.db"),
            "profile": os.environ.get("SUPERGRAPH_PROFILE"),
        }
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
