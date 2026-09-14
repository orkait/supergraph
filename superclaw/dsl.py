from __future__ import annotations

import time
from typing import Any, Protocol

from superclaw.settings import LIMITS
from supergraph.core.errors import SuperGraphError
from supergraph.core.types import Result

_SKIPPABLE = ("duplicate", "not found", "already exist")

MS_PER_SECOND = 1000
MS_PER_DAY = 86_400_000


class Store(Protocol):
    def execute(self, query: str, *, namespace: str | None = None) -> Result: ...
    def close(self) -> None: ...


def lit(value: Any) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def rows(result: Result) -> list[dict[str, Any]]:
    data = getattr(result, "data", None)
    return data if isinstance(data, list) else []


def now_ms() -> int:
    return int(time.time() * MS_PER_SECOND)


def edge(gs: Store, source: str, target: str, kind: str, namespace: str | None = None, **fields: Any) -> bool:
    extra = "".join(f" {name} = {value if isinstance(value, (int, float)) else lit(value)}" for name, value in fields.items())
    try:
        gs.execute(f"CREATE EDGE {lit(source)} -> {lit(target)} kind = {lit(kind)}{extra}", namespace=namespace)
    except SuperGraphError as e:
        if any(marker in str(e).lower() for marker in _SKIPPABLE):
            return False
        raise
    return True


def age(at_ms: int, now: int | None = None) -> str:
    days = max(0, ((now if now is not None else now_ms()) - at_ms) // MS_PER_DAY)
    if days == 0:
        return "today"
    if days < LIMITS.days_per_month:
        return f"{days}d ago"
    if days < LIMITS.days_per_year:
        return f"{days // LIMITS.days_per_month}mo ago"
    return f"{days // LIMITS.days_per_year}y ago"
