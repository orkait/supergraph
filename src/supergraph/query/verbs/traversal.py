from __future__ import annotations

from typing import Any

from supergraph.query.escape import dsl_identifier, dsl_node_id
from supergraph.query.filters import F, compile_where
from supergraph.query.runtime import Query, register_compiler


def _where_clause(params: dict) -> str:
    w = compile_where(params.get("where"))
    return f" WHERE {w}" if w else ""


def _require_depth(depth: Any, label: str) -> None:
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
        raise ValueError(f"{label} depth must be a non-negative int, got {depth!r}")


def traverse(from_id: str, *, depth: int, where: F | dict | None = None, limit: int | None = None) -> Query:
    _require_depth(depth, "traverse()")
    params: dict = {"from_id": from_id, "depth": depth}
    if where is not None: params["where"] = where
    if limit is not None: params["limit"] = limit
    return Query(_verb="traverse", _params=params, _kind="read")


def _compile_traverse(p: dict) -> str:
    out = f"TRAVERSE FROM {dsl_node_id(p['from_id'])} DEPTH {p['depth']}"
    out += _where_clause(p)
    if p.get("limit") is not None:
        out += f" LIMIT {p['limit']}"
    return out


register_compiler("traverse", _compile_traverse)


def subgraph(from_id: str, *, depth: int) -> Query:
    _require_depth(depth, "subgraph()")
    return Query(_verb="subgraph", _params={"from_id": from_id, "depth": depth}, _kind="read")


def _compile_subgraph(p: dict) -> str:
    return f"SUBGRAPH FROM {dsl_node_id(p['from_id'])} DEPTH {p['depth']}"


register_compiler("subgraph", _compile_subgraph)


def path(a: str, b: str, *, max_depth: int, where: F | dict | None = None) -> Query:
    _require_depth(max_depth, "path()")
    params: dict = {"a": a, "b": b, "max_depth": max_depth}
    if where is not None: params["where"] = where
    return Query(_verb="path", _params=params, _kind="read")


def _compile_path(p: dict) -> str:
    out = f"PATH FROM {dsl_node_id(p['a'])} TO {dsl_node_id(p['b'])} MAX_DEPTH {p['max_depth']}"
    out += _where_clause(p)
    return out


register_compiler("path", _compile_path)


def paths(a: str, b: str, *, max_depth: int, where: F | dict | None = None) -> Query:
    _require_depth(max_depth, "paths()")
    params: dict = {"a": a, "b": b, "max_depth": max_depth}
    if where is not None: params["where"] = where
    return Query(_verb="paths", _params=params, _kind="read")


def _compile_paths(p: dict) -> str:
    out = f"PATHS FROM {dsl_node_id(p['a'])} TO {dsl_node_id(p['b'])} MAX_DEPTH {p['max_depth']}"
    out += _where_clause(p)
    return out


register_compiler("paths", _compile_paths)


def shortest_path(a: str, b: str, *, max_depth: int | None = None, where: F | dict | None = None) -> Query:
    if max_depth is not None:
        _require_depth(max_depth, "shortest_path()")
    params: dict = {"a": a, "b": b}
    if max_depth is not None: params["max_depth"] = max_depth
    if where is not None: params["where"] = where
    return Query(_verb="shortest_path", _params=params, _kind="read")


def _compile_shortest_path(p: dict) -> str:
    out = f"SHORTEST PATH FROM {dsl_node_id(p['a'])} TO {dsl_node_id(p['b'])}"
    if "max_depth" in p:
        out += f" MAX_DEPTH {p['max_depth']}"
    out += _where_clause(p)
    return out


register_compiler("shortest_path", _compile_shortest_path)


def distance(a: str, b: str, *, max_depth: int) -> Query:
    _require_depth(max_depth, "distance()")
    return Query(_verb="distance", _params={"a": a, "b": b, "max_depth": max_depth}, _kind="read")


def _compile_distance(p: dict) -> str:
    return f"DISTANCE FROM {dsl_node_id(p['a'])} TO {dsl_node_id(p['b'])} MAX_DEPTH {p['max_depth']}"


register_compiler("distance", _compile_distance)


def weighted_shortest_path(a: str, b: str, *, max_depth: int | None = None, where: F | dict | None = None) -> Query:
    if max_depth is not None:
        _require_depth(max_depth, "weighted_shortest_path()")
    params: dict = {"a": a, "b": b}
    if max_depth is not None: params["max_depth"] = max_depth
    if where is not None: params["where"] = where
    return Query(_verb="weighted_shortest_path", _params=params, _kind="read")


def _compile_weighted_shortest_path(p: dict) -> str:
    out = f"WEIGHTED SHORTEST PATH FROM {dsl_node_id(p['a'])} TO {dsl_node_id(p['b'])}"
    if "max_depth" in p:
        out += f" MAX_DEPTH {p['max_depth']}"
    out += _where_clause(p)
    return out


register_compiler("weighted_shortest_path", _compile_weighted_shortest_path)


def weighted_distance(a: str, b: str, *, max_depth: int | None = None) -> Query:
    if max_depth is not None:
        _require_depth(max_depth, "weighted_distance()")
    params: dict = {"a": a, "b": b}
    if max_depth is not None: params["max_depth"] = max_depth
    return Query(_verb="weighted_distance", _params=params, _kind="read")


def _compile_weighted_distance(p: dict) -> str:
    out = f"WEIGHTED DISTANCE FROM {dsl_node_id(p['a'])} TO {dsl_node_id(p['b'])}"
    if "max_depth" in p:
        out += f" MAX_DEPTH {p['max_depth']}"
    return out


register_compiler("weighted_distance", _compile_weighted_distance)


def ancestors(id: str, *, depth: int, where: F | dict | None = None) -> Query:
    _require_depth(depth, "ancestors()")
    params: dict = {"id": id, "depth": depth}
    if where is not None: params["where"] = where
    return Query(_verb="ancestors", _params=params, _kind="read")


def _compile_ancestors(p: dict) -> str:
    out = f"ANCESTORS OF {dsl_node_id(p['id'])} DEPTH {p['depth']}"
    out += _where_clause(p)
    return out


register_compiler("ancestors", _compile_ancestors)


def descendants(id: str, *, depth: int, where: F | dict | None = None) -> Query:
    _require_depth(depth, "descendants()")
    params: dict = {"id": id, "depth": depth}
    if where is not None: params["where"] = where
    return Query(_verb="descendants", _params=params, _kind="read")


def _compile_descendants(p: dict) -> str:
    out = f"DESCENDANTS OF {dsl_node_id(p['id'])} DEPTH {p['depth']}"
    out += _where_clause(p)
    return out


register_compiler("descendants", _compile_descendants)


def common_neighbors(a: str, b: str, *, where: F | dict | None = None) -> Query:
    params: dict = {"a": a, "b": b}
    if where is not None: params["where"] = where
    return Query(_verb="common_neighbors", _params=params, _kind="read")


def _compile_common_neighbors(p: dict) -> str:
    out = f"COMMON NEIGHBORS OF {dsl_node_id(p['a'])} AND {dsl_node_id(p['b'])}"
    out += _where_clause(p)
    return out


register_compiler("common_neighbors", _compile_common_neighbors)


def match(pattern, *, limit: int | None = None) -> Query:
    from supergraph.query.pattern import Pattern
    if isinstance(pattern, Pattern):
        if len(pattern.steps) < 2:
            raise ValueError(
                "match() requires a pattern with at least one arrow; "
                "use ``P.node(\"a\").to(P.var(\"b\"))`` or similar"
            )
        text = pattern.to_dsl()
    elif isinstance(pattern, str):
        if not pattern.strip():
            raise ValueError("match() requires a non-empty pattern string")
        text = pattern
    else:
        raise TypeError(f"match() expects Pattern or str, got {type(pattern).__name__}")
    params: dict = {"pattern": text}
    if limit is not None: params["limit"] = limit
    return Query(_verb="match", _params=params, _kind="read")


def _compile_match(p: dict) -> str:
    out = f"MATCH {p['pattern']}"
    if p.get("limit") is not None:
        out += f" LIMIT {p['limit']}"
    return out


register_compiler("match", _compile_match)


def what_if_retract(id: str) -> Query:
    return Query(_verb="what_if_retract", _params={"id": id}, _kind="read")


def _compile_what_if_retract(p: dict) -> str:
    return f"WHAT IF RETRACT {dsl_node_id(p['id'])}"


register_compiler("what_if_retract", _compile_what_if_retract)


def aggregate_nodes(
    *,
    select,
    where: F | dict | None = None,
    group_by: list[str] | None = None,
    having=None,
    order_by=None,
    order_dir: str | None = None,
    limit: int | None = None,
) -> Query:
    from supergraph.query.agg import AggFunc, HavingExpr
    if not isinstance(select, (list, tuple)) or not select:
        raise ValueError("aggregate_nodes() requires select=[...] non-empty list")
    select_strs = []
    for item in select:
        if isinstance(item, AggFunc):
            select_strs.append(item.to_dsl())
        elif isinstance(item, str):
            select_strs.append(item)
        else:
            raise TypeError(f"select items must be AggFunc or str, got {type(item).__name__}")
    if group_by is not None:
        if not isinstance(group_by, (list, tuple)) or not group_by:
            raise ValueError("aggregate_nodes() group_by must be non-empty list if given")
        for g in group_by:
            dsl_identifier(g)
    if order_dir is not None and order_dir.upper() not in ("ASC", "DESC"):
        raise ValueError("aggregate_nodes() order_dir must be 'ASC' or 'DESC'")
    if isinstance(having, HavingExpr):
        having = having.to_dsl()
    if isinstance(order_by, AggFunc):
        order_by = order_by.to_dsl()
    params: dict = {"select": select_strs}
    if where is not None: params["where"] = where
    if group_by is not None: params["group_by"] = list(group_by)
    if having is not None: params["having"] = having
    if order_by is not None: params["order_by"] = order_by
    if order_dir is not None: params["order_dir"] = order_dir.upper()
    if limit is not None: params["limit"] = limit
    return Query(_verb="aggregate_nodes", _params=params, _kind="read")


def _compile_aggregate_nodes(p: dict) -> str:
    out = "AGGREGATE NODES"
    out += _where_clause(p)
    if p.get("group_by"):
        out += " GROUP BY " + ", ".join(p["group_by"])
    out += " SELECT " + ", ".join(p["select"])
    if p.get("having"):
        out += f" HAVING {p['having']}"
    if p.get("order_by"):
        out += f" ORDER BY {p['order_by']}"
        if p.get("order_dir"):
            out += f" {p['order_dir']}"
    if p.get("limit") is not None:
        out += f" LIMIT {p['limit']}"
    return out


register_compiler("aggregate_nodes", _compile_aggregate_nodes)


def count_edges(*, where: F | dict | None = None) -> Query:
    params: dict = {}
    if where is not None: params["where"] = where
    return Query(_verb="count_edges", _params=params, _kind="read")


def _compile_count_edges(p: dict) -> str:
    out = "COUNT EDGES"
    out += _where_clause(p)
    return out


register_compiler("count_edges", _compile_count_edges)
