
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
