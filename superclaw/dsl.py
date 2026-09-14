from __future__ import annotations

import time
from typing import Any

from superclaw.settings import LIMITS

MS_PER_SECOND = 1000
MS_PER_DAY = 86_400_000


def lit(value: Any) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def rows(result: Any) -> list[dict]:
    data = getattr(result, "data", None)
    return data if isinstance(data, list) else []


def now_ms() -> int:
    return int(time.time() * MS_PER_SECOND)


def age(at_ms: int, now: int | None = None) -> str:
    days = max(0, ((now if now is not None else now_ms()) - at_ms) // MS_PER_DAY)
    if days == 0:
        return "today"
    if days < LIMITS.days_per_month:
        return f"{days}d ago"
    if days < LIMITS.days_per_year:
        return f"{days // LIMITS.days_per_month}mo ago"
    return f"{days // LIMITS.days_per_year}y ago"
