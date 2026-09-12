
from __future__ import annotations

from typing import Callable


_len_provider: Callable[[], int] | None = None
_clear_provider: Callable[[], None] | None = None


def register(len_fn: Callable[[], int], clear_fn: Callable[[], None]) -> None:
    global _len_provider, _clear_provider
    _len_provider = len_fn
    _clear_provider = clear_fn


def size() -> int:
    return _len_provider() if _len_provider else 0


def clear() -> None:
    if _clear_provider:
        _clear_provider()
