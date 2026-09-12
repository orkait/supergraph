"""VAULT verbs. Access via ``q.vault.*``.

Grammar:
  vault_new:       "VAULT" "NEW" STRING vault_kind? vault_tags?
  vault_read:      "VAULT" "READ" STRING
  vault_write:     "VAULT" "WRITE" STRING "SECTION" STRING "CONTENT" STRING
  vault_append:    "VAULT" "APPEND" STRING "SECTION" STRING "CONTENT" STRING
  vault_search:    "VAULT" "SEARCH" STRING limit? where?
  vault_backlinks: "VAULT" "BACKLINKS" STRING
  vault_list:      "VAULT" "LIST" where? order? limit?
  vault_sync:      "VAULT" "SYNC"
  vault_daily:     "VAULT" "DAILY"
  vault_archive:   "VAULT" "ARCHIVE" STRING
"""
from __future__ import annotations

from supergraph.query.escape import dsl_literal
from supergraph.query.filters import F, compile_where
from supergraph.query.runtime import Query, register_compiler


def _where_clause(params: dict) -> str:
    w = compile_where(params.get("where"))
    return f" WHERE {w}" if w else ""


def new(title: str, *, kind: str | None = None, tags: str | None = None) -> Query:
    params: dict = {"title": title}
    if kind is not None: params["kind"] = kind
    if tags is not None: params["tags"] = tags
    return Query(_verb="vault_new", _params=params, _kind="vault")


def _compile_new(p: dict) -> str:
    out = f"VAULT NEW {dsl_literal(p['title'])}"
    if "kind" in p:
        out += f" KIND {dsl_literal(p['kind'])}"
    if "tags" in p:
        out += f" TAGS {dsl_literal(p['tags'])}"
    return out


register_compiler("vault_new", _compile_new)


def read(id: str) -> Query:
    return Query(_verb="vault_read", _params={"id": id}, _kind="vault")


register_compiler("vault_read", lambda p: f"VAULT READ {dsl_literal(p['id'])}")


def write(id: str, *, section: str, content: str) -> Query:
    return Query(
        _verb="vault_write",
        _params={"id": id, "section": section, "content": content},
        _kind="vault",
    )


def _compile_write(p: dict) -> str:
    return (
        f"VAULT WRITE {dsl_literal(p['id'])} "
        f"SECTION {dsl_literal(p['section'])} "
        f"CONTENT {dsl_literal(p['content'])}"
    )


register_compiler("vault_write", _compile_write)


def append(id: str, *, section: str, content: str) -> Query:
    return Query(
        _verb="vault_append",
        _params={"id": id, "section": section, "content": content},
        _kind="vault",
    )


def _compile_append(p: dict) -> str:
    return (
        f"VAULT APPEND {dsl_literal(p['id'])} "
        f"SECTION {dsl_literal(p['section'])} "
        f"CONTENT {dsl_literal(p['content'])}"
    )


register_compiler("vault_append", _compile_append)


def search(text: str, *, limit: int | None = None, where: F | dict | None = None) -> Query:
    params: dict = {"text": text}
    if limit is not None: params["limit"] = limit
    if where is not None: params["where"] = where
    return Query(_verb="vault_search", _params=params, _kind="vault")


def _compile_search(p: dict) -> str:
    out = f"VAULT SEARCH {dsl_literal(p['text'])}"
    if p.get("limit") is not None:
        out += f" LIMIT {p['limit']}"
    out += _where_clause(p)
    return out


register_compiler("vault_search", _compile_search)


def backlinks(id: str) -> Query:
    return Query(_verb="vault_backlinks", _params={"id": id}, _kind="vault")


register_compiler("vault_backlinks", lambda p: f"VAULT BACKLINKS {dsl_literal(p['id'])}")


def list_(*, where: F | dict | None = None, order_by: str | None = None, limit: int | None = None) -> Query:
    params: dict = {}
    if where is not None: params["where"] = where
    if order_by is not None: params["order_by"] = order_by
    if limit is not None: params["limit"] = limit
    return Query(_verb="vault_list", _params=params, _kind="vault")


def _compile_list(p: dict) -> str:
    out = "VAULT LIST"
    out += _where_clause(p)
    if p.get("order_by"):
        out += f" ORDER BY {p['order_by']}"
    if p.get("limit") is not None:
        out += f" LIMIT {p['limit']}"
    return out


register_compiler("vault_list", _compile_list)


def sync() -> Query:
    return Query(_verb="vault_sync", _params={}, _kind="vault")


register_compiler("vault_sync", lambda p: "VAULT SYNC")


def daily() -> Query:
    return Query(_verb="vault_daily", _params={}, _kind="vault")


register_compiler("vault_daily", lambda p: "VAULT DAILY")


def archive(id: str) -> Query:
    return Query(_verb="vault_archive", _params={"id": id}, _kind="vault")


register_compiler("vault_archive", lambda p: f"VAULT ARCHIVE {dsl_literal(p['id'])}")
