"""Core types for supergraph query results and graph elements."""

from dataclasses import dataclass, field
from typing import Any, TypeAlias

import msgspec
import numpy as np

NodeData: TypeAlias = dict[str, Any]


def _json_fallback(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


_JSON_ENCODER = msgspec.json.Encoder(enc_hook=_json_fallback)


def encode_json(payload: Any) -> bytes:
    return _JSON_ENCODER.encode(payload)


@dataclass(slots=True)
class Edge:
    source: str
    target: str
    kind: str
    data: dict = field(default_factory=dict)


@dataclass(slots=True)
class Result:
    kind: str              # "node", "nodes", "edges", "path", "paths", "match",
                           # "subgraph", "distance", "stats", "plan", "schema",
                           # "log_entries", "ok", "error"
    data: Any              # varies by kind
    count: int             # number of items in data
    elapsed_us: int = 0    # execution time in microseconds
    meta: dict = field(default_factory=dict)  # evolution events, future extensions

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        d = {
            "kind": self.kind,
            "data": self.data,
            "count": self.count,
            "elapsed_us": self.elapsed_us,
        }
        if self.meta:
            d["meta"] = self.meta
        return d

    def to_json(self) -> str:
        """Compact JSON string."""
        return encode_json(self.to_dict()).decode("utf-8")
