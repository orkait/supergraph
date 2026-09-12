from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from superclaw.redaction import redact


class SideEffect(str, Enum):
    NONE = "none"
    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    NETWORK = "network"


class Permission(str, Enum):
    ALLOW = "allow"
    PROMPT = "prompt"
    DENY = "deny"


@dataclass(frozen=True)
class Safety:
    side_effect: SideEffect
    permission: Permission
    reason: str


@dataclass
class Result:
    ok: bool
    output: str
    changed_files: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False

    @classmethod
    def success(cls, output: str, **kw: Any) -> "Result":
        return cls(True, output, **kw)

    @classmethod
    def error(cls, output: str, **kw: Any) -> "Result":
        return cls(False, output, **kw)


class FileTracker:
    def __init__(self) -> None:
        self._hashes: dict[Path, str] = {}

    @staticmethod
    def _hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def record(self, path: Path, content: bytes) -> None:
        self._hashes[path] = self._hash(content)

    def seen(self, path: Path) -> bool:
        return path in self._hashes

    def conflict(self, path: Path, current: bytes) -> str:
        known = self._hashes.get(path)
        if known is None:
            return "read the file before editing it"
        if known != self._hash(current):
            return "the file changed on disk since you last read it; read it again before editing"
        return ""


@dataclass
class ToolContext:
    workspace: Path
    session_id: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    files: FileTracker = field(default_factory=FileTracker)


class PathEscapes(ValueError):
    pass


def jail(workspace: Path, path: str) -> Path:
    root = Path(workspace).resolve()
    candidate = Path(path)
    target = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if target != root and root not in target.parents:
        raise PathEscapes(f"{path!r} escapes the workspace")
    return target


def relative(workspace: Path, target: Path) -> str:
    return target.relative_to(Path(workspace).resolve()).as_posix()


class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    safety: Safety
    deferred: bool = False

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        raise NotImplementedError

    def summary(self) -> str:
        return self.description.split(". ", 1)[0].rstrip(".")

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


MAX_OUTPUT_BYTES = 64 * 1024
_TRUNCATION_MARKER = "\n\n[... output truncated: {dropped} chars omitted ...]\n\n"


def _cap(output: str) -> tuple[str, bool]:
    if len(output) <= MAX_OUTPUT_BYTES:
        return output, False
    keep = MAX_OUTPUT_BYTES // 2
    dropped = len(output) - 2 * keep
    return output[:keep] + _TRUNCATION_MARKER.format(dropped=dropped) + output[-keep:], True


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def tools(self, visible: Callable[[Tool], bool] | None = None) -> list[Tool]:
        return [t for _, t in sorted(self._tools.items()) if visible is None or visible(t)]

    def definitions(self, visible: Callable[[Tool], bool] | None = None, loaded: set[str] | None = None) -> list[dict[str, Any]]:
        if loaded is None:
            return [t.definition() for t in self.tools(visible)]
        return [t.definition() for t in self.tools(visible) if not t.deferred or t.name in loaded]

    def deferred(self, visible: Callable[[Tool], bool] | None = None) -> list[Tool]:
        return [t for t in self.tools(visible) if t.deferred]

    def run(self, name: str, args: dict[str, Any], ctx: ToolContext) -> Result:
        tool = self._tools.get(name)
        if tool is None:
            return Result.error(f"unknown tool {name!r}; available: {', '.join(self.names())}")
        try:
            res = tool.run(args, ctx)
        except PathEscapes as e:
            return Result.error(f"Error: {e}")
        except Exception as e:
            return Result.error(f"Error: {name} failed: {type(e).__name__}: {e}")
        res.output, redacted = redact(res.output)
        res.output, capped = _cap(res.output)
        res.truncated = res.truncated or capped
        if redacted:
            res.meta["redacted"] = True
        return res
