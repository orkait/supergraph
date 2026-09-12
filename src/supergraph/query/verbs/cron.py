"""SYS CRON sub-namespace. Access via ``q.sys.cron.*``.

Grammar:
  sys_cron: "CRON" cron_command
  cron_add:    "ADD" STRING "SCHEDULE" STRING "QUERY" STRING
  cron_delete: "DELETE" STRING
  cron_enable: "ENABLE" STRING
  cron_disable: "DISABLE" STRING
  cron_list:   "LIST"
  cron_run:    "RUN" STRING
"""
from __future__ import annotations

from supergraph.query.escape import dsl_literal
from supergraph.query.runtime import Query, register_compiler


def add(name: str, *, schedule: str, query: str) -> Query:
    if not isinstance(schedule, str) or not schedule:
        raise ValueError("cron.add() requires schedule= as a non-empty str")
    if not isinstance(query, str) or not query:
        raise ValueError("cron.add() requires query= as a non-empty str")
    return Query(
        _verb="sys_cron_add",
        _params={"name": name, "schedule": schedule, "query": query},
        _kind="sys",
    )


def _compile_cron_add(p: dict) -> str:
    return (
        f"SYS CRON ADD {dsl_literal(p['name'])} "
        f"SCHEDULE {dsl_literal(p['schedule'])} "
        f"QUERY {dsl_literal(p['query'])}"
    )


register_compiler("sys_cron_add", _compile_cron_add)


def delete(name: str) -> Query:
    return Query(_verb="sys_cron_delete", _params={"name": name}, _kind="sys")


register_compiler("sys_cron_delete", lambda p: f"SYS CRON DELETE {dsl_literal(p['name'])}")


def enable(name: str) -> Query:
    return Query(_verb="sys_cron_enable", _params={"name": name}, _kind="sys")


register_compiler("sys_cron_enable", lambda p: f"SYS CRON ENABLE {dsl_literal(p['name'])}")


def disable(name: str) -> Query:
    return Query(_verb="sys_cron_disable", _params={"name": name}, _kind="sys")


register_compiler("sys_cron_disable", lambda p: f"SYS CRON DISABLE {dsl_literal(p['name'])}")


def list_() -> Query:
    return Query(_verb="sys_cron_list", _params={}, _kind="sys")


register_compiler("sys_cron_list", lambda p: "SYS CRON LIST")


def run(name: str) -> Query:
    return Query(_verb="sys_cron_run", _params={"name": name}, _kind="sys")


register_compiler("sys_cron_run", lambda p: f"SYS CRON RUN {dsl_literal(p['name'])}")
