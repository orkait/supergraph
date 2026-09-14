from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

from superclaw.settings import LIMITS

ToolSchema: TypeAlias = dict[str, Any]
ToolArgs: TypeAlias = dict[str, Any]
TextSink: TypeAlias = Callable[[str], None]
CancelCheck: TypeAlias = Callable[[], bool]


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
    images: list[str] = field(default_factory=list)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Completion:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""


class Cancelled(RuntimeError):
    pass


class Provider(Protocol):
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        on_text: TextSink | None = None,
        max_tokens: int | None = None,
        cancelled: CancelCheck | None = None,
    ) -> Completion: ...


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
        elif m.images:
            blocks: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": url}} for url in m.images]
            if m.content:
                blocks.insert(0, {"type": "text", "text": m.content})
            wire.append({"role": m.role, "content": blocks})
        else:
            wire.append({"role": m.role, "content": m.content})
    return wire


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


THOUSAND = 1000
PER_MILLION = 1_000_000


def compact(count: int) -> str:
    if count >= PER_MILLION:
        return f"{count / PER_MILLION:.1f}M"
    if count >= THOUSAND:
        return f"{count / THOUSAND:.1f}K"
    return str(count)


def count(n: int, one: str, many: str = "") -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


_ASCII_INK = re.compile(r"[!-~]")
_NON_ASCII = re.compile(r"[^\x00-\x7f]")


def approx_tokens(text: str) -> int:
    ink = len(_ASCII_INK.findall(text))
    foreign = sum(len(ch.encode("utf-8")) for ch in _NON_ASCII.findall(text))
    return (ink + 3) // 4 + foreign if text else 0


def message_tokens(m: Message) -> int:
    overhead = LIMITS.message_overhead_tokens
    images = len(m.images) * LIMITS.image_tokens
    return approx_tokens(m.content) + overhead + images + sum(approx_tokens(c.name) + approx_tokens(c.arguments) + overhead for c in m.tool_calls)


def estimate_tokens(messages: list[Message], tools: list[dict[str, Any]]) -> int:
    return sum(message_tokens(m) for m in messages) + sum(approx_tokens(json.dumps(t)) + LIMITS.message_overhead_tokens for t in tools)
