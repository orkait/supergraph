"""SYS EVOLVE sub-namespace. Access via ``q.sys.evolve.*``.

Grammar:
  sys_evolve: "EVOLVE" evolve_command
  evolve_rule: "RULE" STRING evolve_when_clause evolve_then_clause+ cooldown? priority?
  evolve_when_clause: "WHEN" cond ("AND" cond)*
  evolve_condition: IDENTIFIER OP NUMBER
  evolve_then_clause: "THEN" action
  evolve_action:
    | "SET" IDENT "=" NUMBER | "[" NUMBER+ "]"
    | "ADJUST" IDENT "BY" NUMBER "UNTIL" NUMBER
    | "ADJUST" IDENT "BY" NUMBER
    | "ADD" IDENT STRING
    | "REMOVE" IDENT STRING
    | "RUN" IDENT+
  EVOLVE_OP: ">= | <= | == | != | > | <"

Action + condition shapes are power-user; we accept raw DSL strings for
them and render verbatim. Users compose ``when=`` and ``then=`` lists.
"""
from __future__ import annotations

from supergraph.query.escape import dsl_literal
from supergraph.query.runtime import Query, register_compiler


def rule(
    name: str,
    *,
    when,
    then,
    cooldown: int | None = None,
    priority: int | None = None,
) -> Query:
    """Accepts typed ``EvolveCondition`` / ``EvolveAction`` or raw strings."""
    from supergraph.query.evolve_expr import EvolveCondition, EvolveAction
    if not isinstance(when, (list, tuple)) or not when:
        raise ValueError("evolve.rule() requires when=[...] non-empty list")
    if not isinstance(then, (list, tuple)) or not then:
        raise ValueError("evolve.rule() requires then=[...] non-empty list")

    def _norm_when(item):
        if isinstance(item, EvolveCondition):
            return item.to_dsl()
        if isinstance(item, str):
            return item
        raise TypeError(f"when items must be EvolveCondition or str, got {type(item).__name__}")

    def _norm_then(item):
        if isinstance(item, EvolveAction):
            return item.to_dsl()
        if isinstance(item, str):
            return item
        raise TypeError(f"then items must be EvolveAction or str, got {type(item).__name__}")

    when_strs = [_norm_when(c) for c in when]
    then_strs = [_norm_then(a) for a in then]
    params: dict = {"name": name, "when": when_strs, "then": then_strs}
    if cooldown is not None: params["cooldown"] = cooldown
    if priority is not None: params["priority"] = priority
    return Query(_verb="sys_evolve_rule", _params=params, _kind="sys")


def _compile_rule(p: dict) -> str:
    parts = [f"SYS EVOLVE RULE {dsl_literal(p['name'])}", "WHEN", " AND ".join(p["when"])]
    for t in p["then"]:
        parts += ["THEN", t]
    if "cooldown" in p:
        parts += ["COOLDOWN", str(p["cooldown"])]
    if "priority" in p:
        parts += ["PRIORITY", str(p["priority"])]
    return " ".join(parts)


register_compiler("sys_evolve_rule", _compile_rule)


def list_() -> Query:
    return Query(_verb="sys_evolve_list", _params={}, _kind="sys")


register_compiler("sys_evolve_list", lambda p: "SYS EVOLVE LIST")


def show(name: str) -> Query:
    return Query(_verb="sys_evolve_show", _params={"name": name}, _kind="sys")


register_compiler("sys_evolve_show", lambda p: f"SYS EVOLVE SHOW {dsl_literal(p['name'])}")


def enable(name: str) -> Query:
    return Query(_verb="sys_evolve_enable", _params={"name": name}, _kind="sys")


register_compiler("sys_evolve_enable", lambda p: f"SYS EVOLVE ENABLE {dsl_literal(p['name'])}")


def disable(name: str) -> Query:
    return Query(_verb="sys_evolve_disable", _params={"name": name}, _kind="sys")


register_compiler("sys_evolve_disable", lambda p: f"SYS EVOLVE DISABLE {dsl_literal(p['name'])}")


def delete(name: str) -> Query:
    return Query(_verb="sys_evolve_delete", _params={"name": name}, _kind="sys")


register_compiler("sys_evolve_delete", lambda p: f"SYS EVOLVE DELETE {dsl_literal(p['name'])}")


def history(*, limit: int | None = None) -> Query:
    params: dict = {}
    if limit is not None: params["limit"] = limit
    return Query(_verb="sys_evolve_history", _params=params, _kind="sys")


def _compile_history(p: dict) -> str:
    out = "SYS EVOLVE HISTORY"
    if p.get("limit") is not None:
        out += f" LIMIT {p['limit']}"
    return out


register_compiler("sys_evolve_history", _compile_history)


def reset() -> Query:
    return Query(_verb="sys_evolve_reset", _params={}, _kind="sys")


register_compiler("sys_evolve_reset", lambda p: "SYS EVOLVE RESET")
