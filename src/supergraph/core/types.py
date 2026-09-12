
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
    kind: str
    data: Any
    count: int
    elapsed_us: int = 0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
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
        return encode_json(self.to_dict()).decode("utf-8")
