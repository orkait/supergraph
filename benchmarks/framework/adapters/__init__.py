"""Adapter registry.

Only the supergraph adapter is registered. The adapter protocol
(adapter.py) remains generic so external adapters can still be
plugged in programmatically.
"""

from .supergraph_ import SuperGraphAdapter

AVAILABLE: dict[str, type] = {
    "supergraph": SuperGraphAdapter,
}


def get_adapter(name: str) -> type:
    if name not in AVAILABLE:
        raise ValueError(
            f"Unknown adapter: {name!r}. Available: {sorted(AVAILABLE.keys())}"
        )
    return AVAILABLE[name]
