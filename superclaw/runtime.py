from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Message:
    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    is_error: bool = False


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Completion:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""


class Provider(Protocol):
    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> Completion: ...


def to_wire(messages: list[Message]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            wire.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        elif m.role == "assistant" and m.tool_calls:
            wire.append({
                "role": "assistant",
                "content": m.content or None,
                "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                    for c in m.tool_calls
                ],
            })
        else:
            wire.append({"role": m.role, "content": m.content})
    return wire


def approx_tokens(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace()) // 4


def estimate_tokens(messages: list[Message], tools: list[dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        total += approx_tokens(m.content) + 4
        for c in m.tool_calls:
            total += approx_tokens(c.name) + approx_tokens(c.arguments) + 4
    for t in tools:
        total += approx_tokens(json.dumps(t)) + 4
    return total
