"""Cross-layer plan-cache hook.

``core/optimizer`` needs to know two things about the parser's plan cache
(for health + cache-gc): how big it is, and how to clear it. Importing
``supergraph.dsl.parser`` from ``core`` would invert the layering - core
is meant to be leaf-wards, not top-wards.

This module is the indirection. The DSL parser registers a provider on
its own import; core only knows about this tiny interface.
"""

from __future__ import annotations

from typing import Callable


_len_provider: Callable[[], int] | None = None
_clear_provider: Callable[[], None] | None = None


def register(len_fn: Callable[[], int], clear_fn: Callable[[], None]) -> None:
    """Register plan-cache accessors. Called by supergraph.dsl.parser once at
    import time; safe to call again (later call wins)."""
    global _len_provider, _clear_provider
    _len_provider = len_fn
    _clear_provider = clear_fn


def size() -> int:
    """Plan-cache size. 0 when no provider registered."""
    return _len_provider() if _len_provider else 0


def clear() -> None:
    """Clear the plan cache. No-op when no provider registered."""
    if _clear_provider:
        _clear_provider()
