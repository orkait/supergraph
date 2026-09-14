from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from collections.abc import Callable, Sequence

from superclaw.redaction import redact
from superclaw.runtime import approx_tokens
from superclaw.settings import LIMITS
from superclaw.tools.budget import Budget, Budgeted, Category, budget_output


class Observations(Protocol):
    def save(self, session_id: str, tool: str, call_id: str, body: str) -> str: ...

    def load(self, ref: str) -> Any: ...


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
class Display:
    summary: str = ""
    kind: str = ""
    preview: str = ""


@dataclass(frozen=True)
class Artifact:
    ref: str
    complete: bool


@dataclass(frozen=True)
class Diagnostics:
    category: str
    original_chars: int
    model_chars: int
    original_tokens: int
    model_tokens: int
    truncated: bool
    redacted: bool
    reason: str


@dataclass
class Result:
    ok: bool
    output: str
    changed_files: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    display: Display = field(default_factory=Display)
    artifact: Artifact | None = None
    diagnostics: Diagnostics | None = None

    @classmethod
    def success(cls, output: str, **kw: Any) -> Result:
        return cls(True, output, **kw)

    @classmethod
    def error(cls, output: str, **kw: Any) -> Result:
        return cls(False, output, **kw)


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    digest: str
    message: int

    def covers(self, start: int, end: int) -> bool:
        return self.start <= start and end <= self.end


class FileTracker:
    def __init__(self) -> None:
        self._hashes: dict[Path, str] = {}
        self._windows: dict[Path, list[Window]] = {}
        self.cursor = 0

    @staticmethod
    def _hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def record(self, path: Path, content: bytes) -> None:
        digest = self._hash(content)
        if self._hashes.get(path) != digest:
            self._windows.pop(path, None)
        self._hashes[path] = digest

    def shown(self, path: Path, start: int, end: int) -> None:
        self._windows.setdefault(path, []).append(Window(start, end, self._hashes[path], self.cursor))

    def in_context(self, path: Path, start: int, end: int, current: bytes) -> Window | None:
        digest = self._hash(current)
        return next((w for w in self._windows.get(path, []) if w.digest == digest and w.covers(start, end)), None)

    def evict(self, messages: set[int]) -> None:
        self._rewrite(lambda w: None if w.message in messages else w)

    def compacted(self, system_end: int, removed: int) -> None:
        cut = system_end + removed
        self._rewrite(lambda w: None if w.message < cut else replace(w, message=w.message - removed + 1))

    def _rewrite(self, fn: Callable[[Window], Window | None]) -> None:
        for path in list(self._windows):
            kept = [moved for w in self._windows[path] if (moved := fn(w)) is not None]
            if kept:
                self._windows[path] = kept
            else:
                del self._windows[path]

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
    extra_dirs: tuple[Path, ...] = ()
    state: dict[str, Any] = field(default_factory=dict)
    files: FileTracker = field(default_factory=FileTracker)
    cancelled: Callable[[], bool] | None = None

    @property
    def roots(self) -> tuple[Path, ...]:
        return (self.workspace, *self.extra_dirs)


class PathEscapes(ValueError):
    pass


def roots_of(roots: Path | str | Sequence[Path]) -> list[Path]:
    one = isinstance(roots, (str, Path))
    return [Path(roots).resolve()] if one else [Path(r).resolve() for r in roots]


def jail(roots: Path | str | Sequence[Path], path: str) -> Path:
    allowed = roots_of(roots)
    candidate = Path(path)
    target = (candidate if candidate.is_absolute() else allowed[0] / candidate).resolve()
    if not any(target == root or root in target.parents for root in allowed):
        raise PathEscapes(f"{path!r} escapes the workspace")
    return target


def relative(roots: Path | str | Sequence[Path], target: Path) -> str:
    for root in roots_of(roots):
        if target == root:
            return "."
        if root in target.parents:
            return target.relative_to(root).as_posix()
    return str(target)


class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    safety: Safety
    deferred: bool = False
    output_category: Category = Category.DEFAULT

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        raise NotImplementedError

    def category(self, args: dict[str, Any]) -> Category:
        return self.output_category

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


def truncation_notice(budgeted: Budgeted, artifact: Artifact | None) -> str:
    where = f"recall §{artifact.ref} for the full output" if artifact else "the omitted part is not recoverable"
    return (f"\n[superclaw] output shortened from {budgeted.original_tokens:,} tokens ({budgeted.original_chars:,} chars) "
            f"to {budgeted.retained_tokens:,} tokens; {where}")


def ref_trailer(ref: str) -> str:
    return f"\n[§{ref}]"


class Registry:
    def __init__(self, observations: Observations | None = None, budget: Budget | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self.observations = observations
        self._budget = budget

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

    def run(self, name: str, args: dict[str, Any], ctx: ToolContext, call_id: str = "") -> Result:
        tool = self._tools.get(name)
        if tool is None:
            return Result.error(f"unknown tool {name!r}; available: {', '.join(self.names())}")
        try:
            res = tool.run(args, ctx)
        except PathEscapes as e:
            return Result.error(f"Error: {e}")
        except Exception as e:
            return Result.error(f"Error: {name} failed: {type(e).__name__}: {e}")
        return self.finalize(name, args, res, ctx, call_id)

    def finalize(self, name: str, args: dict[str, Any], res: Result, ctx: ToolContext, call_id: str = "") -> Result:
        tool = self._tools.get(name)
        previous = res.diagnostics
        boundary, redacted = redact(res.output)
        category = tool.category(args) if tool else Category.DEFAULT
        budgeted = budget_output(boundary, category, self._budget)
        res.output = budgeted.text
        body = redact(str(res.meta.pop("full", "")))[0] or boundary
        if self.observations and res.ok and len(body) > LIMITS.obs_min_chars:
            res.artifact = Artifact(self.observations.save(ctx.session_id, name, call_id, body), complete=True)
        if budgeted.truncated:
            ctx.files.evict({ctx.files.cursor})
            res.output += truncation_notice(budgeted, res.artifact)
        elif res.artifact:
            res.output += ref_trailer(res.artifact.ref)
        res.truncated = res.truncated or budgeted.truncated
        if redacted:
            res.meta["redacted"] = True
        res.diagnostics = Diagnostics(
            category=category.value,
            original_chars=previous.original_chars if previous else max(len(body), budgeted.original_chars),
            model_chars=len(res.output),
            original_tokens=previous.original_tokens if previous else max(approx_tokens(body), budgeted.original_tokens),
            model_tokens=approx_tokens(res.output),
            truncated=res.truncated, redacted=redacted or bool(previous and previous.redacted), reason=budgeted.reason,
        )
        return res
