from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
CHECKS: dict[str, type | tuple[type, ...]] = {
    "string": str, "integer": int, "number": (int, float), "boolean": bool,
    "array": list, "object": dict, "null": type(None),
}


class SchemaError(ValueError):
    pass


def load(path: Path) -> dict[str, Any]:
    try:
        schema = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise SchemaError(f"{path}: {e}") from e
    if not isinstance(schema, dict):
        raise SchemaError(f"{path}: a JSON Schema must be an object")
    return schema


def instruction(schema: dict[str, Any]) -> str:
    return (
        "<output_schema>\nYour final message must be one JSON value matching this JSON Schema, and nothing else: "
        "no prose before or after it, no code fence.\n"
        + json.dumps(schema, indent=2)
        + "\n</output_schema>"
    )


def extract(text: str) -> Any:
    body = text.strip()
    fenced = FENCE.search(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        return json.loads(body)
    except ValueError as e:
        raise SchemaError(f"the final answer is not JSON: {e}") from e


def _type_problem(value: Any, expected: str, where: str) -> str:
    if expected not in CHECKS:
        return ""
    if expected in ("number", "integer") and isinstance(value, bool):
        return f"{where}: expected {expected}, got boolean"
    return "" if isinstance(value, CHECKS[expected]) else f"{where}: expected {expected}, got {type(value).__name__}"


def problems(value: Any, schema: dict[str, Any], where: str = "$") -> list[str]:
    found: list[str] = []
    expected = schema.get("type")
    if isinstance(expected, str):
        problem = _type_problem(value, expected, where)
        if problem:
            return [problem]
    if isinstance(value, dict):
        for key in schema.get("required") or []:
            if key not in value:
                found.append(f"{where}: missing required property {key!r}")
        for key, sub in (schema.get("properties") or {}).items():
            if key in value and isinstance(sub, dict):
                found += problems(value[key], sub, f"{where}.{key}")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            found += problems(item, schema["items"], f"{where}[{index}]")
    return found
