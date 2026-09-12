from __future__ import annotations


from supergraph.query.escape import dsl_literal, dsl_node_id
from supergraph.query.filters import F, compile_where
from supergraph.query.runtime import Query, register_compiler


def _where_clause(params: dict) -> str:
    w = compile_where(params.get("where"))
    return f" WHERE {w}" if w else ""


def node(id: str, *, with_document: bool = False) -> Query:
    return Query(_verb="node", _params={"id": id, "with_document": with_document}, _kind="read")


def _compile_node(p: dict) -> str:
    out = f"NODE {dsl_node_id(p['id'])}"
    if p.get("with_document"):
        out += " WITH DOCUMENT"
    return out


register_compiler("node", _compile_node)


def nodes(
    *,
    kind: str | None = None,
    where: F | dict | None = None,
    limit: int | None = None,
    offset: int | None = None,
    order_by: str | None = None,
) -> Query:
    if kind is not None:
        kind_f = F.eq("kind", kind)
        if where is None:
            where = kind_f
        else:
            where_f = F.from_dict(where) if isinstance(where, dict) else where
            where = kind_f & where_f
    params: dict = {}
    if where is not None:
        params["where"] = where
    if limit is not None:
        params["limit"] = limit
    if offset is not None:
        params["offset"] = offset
    if order_by is not None:
        params["order_by"] = order_by
    return Query(_verb="nodes", _params=params, _kind="read")


def _compile_nodes(p: dict) -> str:
    out = "NODES"
    out += _where_clause(p)
    if p.get("order_by"):
        out += f" ORDER BY {p['order_by']}"
    if p.get("limit") is not None:
        out += f" LIMIT {p['limit']}"
    if p.get("offset") is not None:
        out += f" OFFSET {p['offset']}"
    return out


register_compiler("nodes", _compile_nodes)


def remember(
    text: str,
    *,
    limit: int | None = None,
    tokens: int | None = None,
    at: str | None = None,
    where: F | dict | None = None,
) -> Query:
    if not isinstance(text, str) or not text:
        raise ValueError("remember() requires a non-empty query text")
    params: dict = {"text": text}
    if at is not None: params["at"] = at
    if tokens is not None: params["tokens"] = tokens
    if limit is not None: params["limit"] = limit
    if where is not None: params["where"] = where
    return Query(_verb="remember", _params=params, _kind="read")


def _compile_remember(p: dict) -> str:
    out = f"REMEMBER {dsl_literal(p['text'])}"
    if p.get("at") is not None:
        out += f' AT "{p["at"]}"'
    if p.get("tokens") is not None:
        out += f" TOKENS {p['tokens']}"
    if p.get("limit") is not None:
        out += f" LIMIT {p['limit']}"
    out += _where_clause(p)
    return out


register_compiler("remember", _compile_remember)


def answer(
    text: str,
    *,
    limit: int | None = None,
    tokens: int | None = None,
    at: str | None = None,
    where: F | dict | None = None,
    using: str | None = None,
) -> Query:
    if not isinstance(text, str) or not text:
        raise ValueError("answer() requires a non-empty query text")
    params: dict = {"text": text}
    if at is not None: params["at"] = at
    if tokens is not None: params["tokens"] = tokens
    if limit is not None: params["limit"] = limit
    if where is not None: params["where"] = where
    if using is not None: params["using"] = using
    return Query(_verb="answer", _params=params, _kind="read")


def _compile_answer(p: dict) -> str:
    out = f"ANSWER {dsl_literal(p['text'])}"
    if p.get("at") is not None:
        out += f' AT "{p["at"]}"'
    if p.get("tokens") is not None:
        out += f" TOKENS {p['tokens']}"
    if p.get("limit") is not None:
        out += f" LIMIT {p['limit']}"
    out += _where_clause(p)
    if p.get("using") is not None:
        out += f' USING {dsl_literal(p["using"])}'
    return out


register_compiler("answer", _compile_answer)


def recall(from_id: str, *, depth: int, limit: int | None = None, where: F | dict | None = None) -> Query:
    if not isinstance(depth, int) or depth < 0:
        raise ValueError(f"recall() depth must be a non-negative int, got {depth!r}")
    params: dict = {"from_id": from_id, "depth": depth}
    if limit is not None: params["limit"] = limit
    if where is not None: params["where"] = where
    return Query(_verb="recall", _params=params, _kind="read")


def _compile_recall(p: dict) -> str:
    out = f"RECALL FROM {dsl_node_id(p['from_id'])} DEPTH {p['depth']}"
    if p.get("limit") is not None:
        out += f" LIMIT {p['limit']}"
    out += _where_clause(p)
    return out


register_compiler("recall", _compile_recall)


def similar(
    *,
    text: str | None = None,
    node: str | None = None,
    vec: list[float] | None = None,
    limit: int | None = None,
    where: F | dict | None = None,
) -> Query:
    given = [x for x in (text, node, vec) if x is not None]
    if len(given) != 1:
        raise ValueError("similar() requires exactly one of text=, node=, vec=")
    params: dict = {}
    if text is not None: params["text"] = text
    if node is not None: params["node"] = node
    if vec is not None: params["vec"] = list(vec)
    if limit is not None: params["limit"] = limit
    if where is not None: params["where"] = where
    return Query(_verb="similar", _params=params, _kind="read")


def _compile_similar(p: dict) -> str:
    if "text" in p:
        target = dsl_literal(p["text"])
    elif "node" in p:
        target = f"NODE {dsl_node_id(p['node'])}"
    else:
        target = "[" + ", ".join(dsl_literal(v) for v in p["vec"]) + "]"
    out = f"SIMILAR TO {target}"
    if p.get("limit") is not None:
        out += f" LIMIT {p['limit']}"
    out += _where_clause(p)
    return out


register_compiler("similar", _compile_similar)


def lexical(text: str, *, limit: int | None = None, where: F | dict | None = None) -> Query:
    if not isinstance(text, str) or not text:
        raise ValueError("lexical() requires a non-empty query text")
    params: dict = {"text": text}
    if limit is not None: params["limit"] = limit
    if where is not None: params["where"] = where
    return Query(_verb="lexical", _params=params, _kind="read")


def _compile_lexical(p: dict) -> str:
    out = f"LEXICAL SEARCH {dsl_literal(p['text'])}"
    if p.get("limit") is not None:
        out += f" LIMIT {p['limit']}"
    out += _where_clause(p)
    return out


register_compiler("lexical", _compile_lexical)


def edges(
    node: str,
    *,
    direction: str = "FROM",
    where: F | dict | None = None,
    limit: int | None = None,
) -> Query:
    if direction not in ("FROM", "TO"):
        raise ValueError(f'edges() direction must be "FROM" or "TO", got {direction!r}')
    params: dict = {"node": node, "direction": direction}
    if where is not None: params["where"] = where
    if limit is not None: params["limit"] = limit
    return Query(_verb="edges", _params=params, _kind="read")


def _compile_edges(p: dict) -> str:
    out = f"EDGES {p['direction']} {dsl_node_id(p['node'])}"
    out += _where_clause(p)
    if p.get("limit") is not None:
        out += f" LIMIT {p['limit']}"
    return out


register_compiler("edges", _compile_edges)


def count_nodes(*, where: F | dict | None = None) -> Query:
    params: dict = {}
    if where is not None: params["where"] = where
    return Query(_verb="count_nodes", _params=params, _kind="read")


def _compile_count_nodes(p: dict) -> str:
    out = "COUNT NODES"
    out += _where_clause(p)
    return out


register_compiler("count_nodes", _compile_count_nodes)
