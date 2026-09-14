from __future__ import annotations

from superclaw.settings import MILLION, THOUSAND


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def oneline(text: str) -> str:
    return " ".join(text.split())


def compact(size: int) -> str:
    if size >= MILLION:
        return f"{size / MILLION:.1f}M"
    if size >= THOUSAND:
        return f"{size / THOUSAND:.1f}K"
    return str(size)


def count(n: int, one: str, many: str = "") -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"
