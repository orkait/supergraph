from __future__ import annotations

import time
from typing import Any

from supergraph.core.errors import SuperGraphError

from superclaw.settings import LIMITS

_SKIPPABLE = ("duplicate", "not found", "already exist")

MS_PER_SECOND = 1000
MS_PER_DAY = 86_400_000


def lit(value: Any) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def rows(result: Any) -> list[dict[str, Any]]:
    data = getattr(result, "data", None)
    return data if isinstance(data, list) else []


def now_ms() -> int:
    return int(time.time() * MS_PER_SECOND)


def edge(gs: Any, source: str, target: str, kind: str, namespace: str | None = None, **fields: Any) -> bool:
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
