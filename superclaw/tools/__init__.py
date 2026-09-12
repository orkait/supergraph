from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


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


@dataclass
class ToolContext:
    workspace: Path
    session_id: str = ""
    state: dict[str, Any] = field(default_factory=dict)


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

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        raise NotImplementedError

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

    def definitions(self, visible: Callable[[Tool], bool] | None = None) -> list[dict[str, Any]]:
        return [
            t.definition()
            for name, t in sorted(self._tools.items())
            if visible is None or visible(t)
        ]

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
        res.output, capped = _cap(res.output)
        res.truncated = res.truncated or capped
        return res
