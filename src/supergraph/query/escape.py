from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any


def _escape_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def dsl_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("NaN is not a valid DSL NUMBER")
        if math.isinf(value):
            raise ValueError("Infinity is not a valid DSL NUMBER")
        return repr(value)
    if isinstance(value, str):
        return _escape_string(value)
    if isinstance(value, datetime):
        return _escape_string(value.isoformat())
    if isinstance(value, date):
        return _escape_string(value.isoformat())
    from supergraph.query.time_expr import TimeExpr
    if isinstance(value, TimeExpr):
        return value.to_dsl()
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("empty list is not a valid DSL literal (use a WHERE op like __in with a non-empty list)")
        return "(" + ", ".join(dsl_literal(v) for v in value) + ")"
    raise TypeError(
        f"unsupported DSL value type: {type(value).__name__}. "
        f"supported: str, int, float, bool, None, list, tuple, datetime, date"
    )


def dsl_identifier(name: str) -> str:
    if not name or not isinstance(name, str):
        raise ValueError(f"identifier must be a non-empty str, got {name!r}")
    first = name[0]
    if not (first.isalpha() or first == "_"):
        raise ValueError(f"identifier must start with letter or underscore: {name!r}")
    for ch in name[1:]:
        if not (ch.isalnum() or ch == "_"):
            raise ValueError(f"identifier contains invalid character {ch!r}: {name!r}")
    return name


def dsl_field_ref(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"field ref must be a non-empty str, got {name!r}")
    parts = name.split(".")
    if len(parts) > 2:
        raise ValueError(f"field ref supports at most one dot level, got {name!r}")
    for p in parts:
        dsl_identifier(p)
    return name


def dsl_variable(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"variable name must be a non-empty str, got {name!r}")
    stripped = name[1:] if name.startswith("$") else name
    dsl_identifier(stripped)
    return f"${stripped}"


def dsl_time_unit(unit: str) -> str:
    if unit not in ("s", "m", "h", "d"):
        raise ValueError(f"time unit must be one of 's','m','h','d', got {unit!r}")
    return unit


def dsl_node_ref(ref: str) -> str:
    if not isinstance(ref, str) or not ref:
        raise ValueError(f"node ref must be a non-empty str, got {ref!r}")
    if ref.startswith("$"):
        return dsl_variable(ref)
    return dsl_node_id(ref)


def dsl_node_id(node_id: str) -> str:
    if not isinstance(node_id, str) or not node_id:
        raise ValueError(f"node id must be a non-empty str, got {node_id!r}")
    return _escape_string(node_id)
