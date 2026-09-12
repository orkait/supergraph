from __future__ import annotations


from supergraph.query.escape import dsl_identifier, dsl_literal
from supergraph.query.filters import F, compile_where
from supergraph.query.runtime import Query, register_compiler


def _where_clause(params: dict) -> str:
    w = compile_where(params.get("where"))
    return f" WHERE {w}" if w else ""


def _check_entity_type(entity: str) -> str:
    upper = entity.upper()
    if upper not in ("NODE", "EDGE"):
        raise ValueError(f"entity must be 'NODE' or 'EDGE', got {entity!r}")
    return upper


_STATS_TARGETS = {"NODES", "EDGES", "MEMORY", "WAL"}


def stats(target: str | None = None) -> Query:
    if target is not None:
        upper = target.upper()
        if upper not in _STATS_TARGETS:
            raise ValueError(f"stats target must be in {sorted(_STATS_TARGETS)}, got {target!r}")
        target = upper
    return Query(_verb="sys_stats", _params={"target": target}, _kind="sys")


def _compile_stats(p: dict) -> str:
    out = "SYS STATS"
    if p.get("target"):
        out += f" {p['target']}"
    return out


register_compiler("sys_stats", _compile_stats)


def kinds() -> Query:
    return Query(_verb="sys_kinds", _params={}, _kind="sys")


def _compile_kinds(p: dict) -> str:
    return "SYS KINDS"


register_compiler("sys_kinds", _compile_kinds)


def edge_kinds() -> Query:
    return Query(_verb="sys_edge_kinds", _params={}, _kind="sys")


def _compile_edge_kinds(p: dict) -> str:
    return "SYS EDGE KINDS"


register_compiler("sys_edge_kinds", _compile_edge_kinds)


def describe(entity: str, name: str) -> Query:
    upper = _check_entity_type(entity)
    return Query(_verb="sys_describe", _params={"entity": upper, "name": name}, _kind="sys")


def _compile_describe(p: dict) -> str:
    return f"SYS DESCRIBE {p['entity']} {dsl_literal(p['name'])}"


register_compiler("sys_describe", _compile_describe)


def slow_queries(*, since: str | None = None, limit: int | None = None) -> Query:
    params: dict = {}
    if since is not None: params["since"] = since
    if limit is not None: params["limit"] = limit
    return Query(_verb="sys_slow", _params=params, _kind="sys")


def _compile_slow(p: dict) -> str:
    out = "SYS SLOW QUERIES"
    if "since" in p:
        out += f" SINCE {dsl_literal(p['since'])}"
    if p.get("limit") is not None:
        out += f" LIMIT {p['limit']}"
    return out


register_compiler("sys_slow", _compile_slow)


def frequent_queries(*, limit: int | None = None) -> Query:
    params: dict = {}
    if limit is not None: params["limit"] = limit
    return Query(_verb="sys_frequent", _params=params, _kind="sys")


def _compile_frequent(p: dict) -> str:
    out = "SYS FREQUENT QUERIES"
    if p.get("limit") is not None:
        out += f" LIMIT {p['limit']}"
    return out


register_compiler("sys_frequent", _compile_frequent)


def failed_queries(*, limit: int | None = None) -> Query:
    params: dict = {}
    if limit is not None: params["limit"] = limit
    return Query(_verb="sys_failed", _params=params, _kind="sys")


def _compile_failed(p: dict) -> str:
    out = "SYS FAILED QUERIES"
    if p.get("limit") is not None:
        out += f" LIMIT {p['limit']}"
    return out


register_compiler("sys_failed", _compile_failed)


def explain(query: Query) -> Query:
    if query._kind != "read":
        raise ValueError("explain() only accepts read queries")
    return Query(_verb="sys_explain", _params={"inner": query}, _kind="sys")


def _compile_explain(p: dict) -> str:
    return f"SYS EXPLAIN {p['inner'].dsl()}"


register_compiler("sys_explain", _compile_explain)


def register_node_kind(
    name: str,
    *,
    required: dict[str, str] | list[str],
    optional: dict[str, str] | list[str] | None = None,
    embed: str | None = None,
) -> Query:
    return Query(
        _verb="sys_register_node_kind",
        _params={
            "name": name,
            "required": required,
            "optional": optional,
            "embed": embed,
        },
        _kind="sys",
    )


def _format_typed_idents(items) -> str:
    if isinstance(items, dict):
        return ", ".join(f"{dsl_identifier(n)}:{dsl_identifier(t)}" for n, t in items.items())
    return ", ".join(dsl_identifier(n) for n in items)


def _compile_register_node_kind(p: dict) -> str:
    parts = [
        f"SYS REGISTER NODE KIND {dsl_literal(p['name'])}",
        "REQUIRED",
        _format_typed_idents(p["required"]),
    ]
    if p.get("optional"):
        parts += ["OPTIONAL", _format_typed_idents(p["optional"])]
    if p.get("embed"):
        parts += ["EMBED", dsl_identifier(p["embed"])]
    return " ".join(parts)


register_compiler("sys_register_node_kind", _compile_register_node_kind)


def register_edge_kind(name: str, *, from_kinds: list[str], to_kinds: list[str]) -> Query:
    if not from_kinds or not to_kinds:
        raise ValueError("register_edge_kind() requires non-empty from_kinds / to_kinds")
    return Query(
        _verb="sys_register_edge_kind",
        _params={"name": name, "from_kinds": list(from_kinds), "to_kinds": list(to_kinds)},
        _kind="sys",
    )


def _compile_register_edge_kind(p: dict) -> str:
    froms = ", ".join(dsl_literal(s) for s in p["from_kinds"])
    tos = ", ".join(dsl_literal(s) for s in p["to_kinds"])
    return f"SYS REGISTER EDGE KIND {dsl_literal(p['name'])} FROM {froms} TO {tos}"


register_compiler("sys_register_edge_kind", _compile_register_edge_kind)


def unregister(entity: str, name: str) -> Query:
    upper = _check_entity_type(entity)
    return Query(_verb="sys_unregister", _params={"entity": upper, "name": name}, _kind="sys")


def _compile_unregister(p: dict) -> str:
    return f"SYS UNREGISTER {p['entity']} KIND {dsl_literal(p['name'])}"


register_compiler("sys_unregister", _compile_unregister)


def checkpoint() -> Query:
    return Query(_verb="sys_checkpoint", _params={}, _kind="sys")


register_compiler("sys_checkpoint", lambda p: "SYS CHECKPOINT")


def rebuild_indices() -> Query:
    return Query(_verb="sys_rebuild", _params={}, _kind="sys")


register_compiler("sys_rebuild", lambda p: "SYS REBUILD INDICES")


_CLEAR_TARGETS = {"LOG", "CACHE"}


def clear(target: str) -> Query:
    upper = target.upper()
    if upper not in _CLEAR_TARGETS:
        raise ValueError(f"clear target must be in {sorted(_CLEAR_TARGETS)}, got {target!r}")
    return Query(_verb="sys_clear", _params={"target": upper}, _kind="sys")


register_compiler("sys_clear", lambda p: f"SYS CLEAR {p['target']}")


_WAL_ACTIONS = {"STATUS", "REPLAY"}


def wal(action: str) -> Query:
    upper = action.upper()
    if upper not in _WAL_ACTIONS:
        raise ValueError(f"wal action must be in {sorted(_WAL_ACTIONS)}, got {action!r}")
    return Query(_verb="sys_wal", _params={"action": upper}, _kind="sys")


register_compiler("sys_wal", lambda p: f"SYS WAL {p['action']}")


def expire(*, where: F | dict | None = None) -> Query:
    params: dict = {}
    if where is not None: params["where"] = where
    return Query(_verb="sys_expire", _params=params, _kind="sys")


def _compile_expire(p: dict) -> str:
    return "SYS EXPIRE" + _where_clause(p)


register_compiler("sys_expire", _compile_expire)


def contradictions(*, field: str, group_by: str, where: F | dict | None = None) -> Query:
    dsl_identifier(field)
    dsl_identifier(group_by)
    params: dict = {"field": field, "group_by": group_by}
    if where is not None: params["where"] = where
    return Query(_verb="sys_contradictions", _params=params, _kind="sys")


def _compile_contradictions(p: dict) -> str:
    return f"SYS CONTRADICTIONS{_where_clause(p)} FIELD {p['field']} GROUP BY {p['group_by']}"


register_compiler("sys_contradictions", _compile_contradictions)


def snapshot(name: str) -> Query:
    return Query(_verb="sys_snapshot", _params={"name": name}, _kind="sys")


register_compiler("sys_snapshot", lambda p: f"SYS SNAPSHOT {dsl_literal(p['name'])}")


def rollback_to(name: str) -> Query:
    return Query(_verb="sys_rollback", _params={"name": name}, _kind="sys")


register_compiler("sys_rollback", lambda p: f"SYS ROLLBACK TO {dsl_literal(p['name'])}")


def snapshots() -> Query:
    return Query(_verb="sys_snapshots", _params={}, _kind="sys")


register_compiler("sys_snapshots", lambda p: "SYS SNAPSHOTS")


def duplicates(*, where: F | dict | None = None, threshold: float | None = None) -> Query:
    params: dict = {}
    if where is not None: params["where"] = where
    if threshold is not None: params["threshold"] = threshold
    return Query(_verb="sys_duplicates", _params=params, _kind="sys")


def _compile_duplicates(p: dict) -> str:
    out = "SYS DUPLICATES" + _where_clause(p)
    if "threshold" in p:
        out += f" THRESHOLD {p['threshold']}"
    return out


register_compiler("sys_duplicates", _compile_duplicates)


def embedders() -> Query:
    return Query(_verb="sys_embedders", _params={}, _kind="sys")


register_compiler("sys_embedders", lambda p: "SYS EMBEDDERS")


def connect(*, where: F | dict | None = None, threshold: float | None = None) -> Query:
    params: dict = {}
    if where is not None: params["where"] = where
    if threshold is not None: params["threshold"] = threshold
    return Query(_verb="sys_connect", _params=params, _kind="sys")


def _compile_connect(p: dict) -> str:
    out = "SYS CONNECT" + _where_clause(p)
    if "threshold" in p:
        out += f" THRESHOLD {p['threshold']}"
    return out


register_compiler("sys_connect", _compile_connect)


def consolidate(*, threshold: float | None = None, min_cluster_size: int | None = None) -> Query:
    params: dict = {}
    if threshold is not None: params["threshold"] = threshold
    if min_cluster_size is not None: params["min_cluster_size"] = min_cluster_size
    return Query(_verb="sys_consolidate", _params=params, _kind="sys")


def _compile_consolidate(p: dict) -> str:
    out = "SYS CONSOLIDATE"
    if "threshold" in p:
        out += f" THRESHOLD {p['threshold']}"
    if "min_cluster_size" in p:
        out += f" MIN_CLUSTER_SIZE {p['min_cluster_size']}"
    return out


register_compiler("sys_consolidate", _compile_consolidate)


def reembed() -> Query:
    return Query(_verb="sys_reembed", _params={}, _kind="sys")


register_compiler("sys_reembed", lambda p: "SYS REEMBED")


def status() -> Query:
    return Query(_verb="sys_status", _params={}, _kind="sys")


register_compiler("sys_status", lambda p: "SYS STATUS")


def retain() -> Query:
    return Query(_verb="sys_retain", _params={}, _kind="sys")


register_compiler("sys_retain", lambda p: "SYS RETAIN")


def health() -> Query:
    return Query(_verb="sys_health", _params={}, _kind="sys")


register_compiler("sys_health", lambda p: "SYS HEALTH")


_OPTIMIZE_TARGETS = {"COMPACT", "STRINGS", "EDGES", "VECTORS", "BLOBS", "CACHE"}


def optimize(target: str | None = None) -> Query:
    if target is not None:
        upper = target.upper()
        if upper not in _OPTIMIZE_TARGETS:
            raise ValueError(f"optimize target must be in {sorted(_OPTIMIZE_TARGETS)}, got {target!r}")
        target = upper
    return Query(_verb="sys_optimize", _params={"target": target}, _kind="sys")


def _compile_optimize(p: dict) -> str:
    out = "SYS OPTIMIZE"
    if p.get("target"):
        out += f" {p['target']}"
    return out


register_compiler("sys_optimize", _compile_optimize)


def evict(*, limit: int | None = None) -> Query:
    params: dict = {}
    if limit is not None: params["limit"] = limit
    return Query(_verb="sys_evict", _params=params, _kind="sys")


def _compile_evict(p: dict) -> str:
    out = "SYS EVICT"
    if p.get("limit") is not None:
        out += f" LIMIT {p['limit']}"
    return out


register_compiler("sys_evict", _compile_evict)


def log(
    *,
    where: F | dict | None = None,
    since: str | None = None,
    trace: str | None = None,
    limit: int | None = None,
) -> Query:
    given = [x for x in (where, since, trace) if x is not None]
    if len(given) > 1:
        raise ValueError("log() accepts at most one of where=, since=, trace=")
    params: dict = {}
    if where is not None: params["where"] = where
    if since is not None: params["since"] = since
    if trace is not None: params["trace"] = trace
    if limit is not None: params["limit"] = limit
    return Query(_verb="sys_log", _params=params, _kind="sys")


def _compile_log(p: dict) -> str:
    parts = ["SYS LOG"]
    if "where" in p:
        w = compile_where(p["where"])
        if w:
            parts.append(f"WHERE {w}")
    elif "since" in p:
        parts.append(f"SINCE {dsl_literal(p['since'])}")
    elif "trace" in p:
        parts.append(f"TRACE {dsl_literal(p['trace'])}")
    if p.get("limit") is not None:
        parts.append(f"LIMIT {p['limit']}")
    return " ".join(parts)


register_compiler("sys_log", _compile_log)
