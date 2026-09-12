
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Message:
    role: str
    content: str
    timestamp: float | None = None


@dataclass
class Session:
    session_id: str
    messages: list[Message]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryContext:
    question: str
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    retrieved_memories: list[str]
    answer: str | None = None
    elapsed_ms: float = 0.0
    tokens_used: int = 0
    raw: Any = None


class MemoryAdapter(Protocol):

    name: str
    version: str

    def reset(self) -> None: ...
    def ingest(self, session: Session) -> float: ...
    def query(self, question: str, k: int = 5) -> QueryResult: ...
    def close(self) -> None: ...


class TimedOperation:

    def __init__(self) -> None:
        self._start_ns = 0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "TimedOperation":
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter_ns() - self._start_ns) / 1_000_000
