from __future__ import annotations

import re
from pathlib import Path, PurePath
from typing import Any

from superclaw.tools import (
    Permission,
    Result,
    Safety,
    SideEffect,
    Tool,
    ToolContext,
    jail,
    relative,
)

IGNORED_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
MAX_READ_BYTES = 256 * 1024
MAX_ENTRIES = 500


def _read(side_effect_reason: str) -> Safety:
    return Safety(SideEffect.READ, Permission.ALLOW, side_effect_reason)


def _write(side_effect_reason: str) -> Safety:
    return Safety(SideEffect.WRITE, Permission.PROMPT, side_effect_reason)


def _walk(root: Path, max_depth: int | None):
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for entry in entries:
            if entry.name in IGNORED_DIRS:
                continue
            yield entry
            if entry.is_dir() and not entry.is_symlink() and (max_depth is None or depth + 1 < max_depth):
                stack.append((entry, depth + 1))


class ReadFile(Tool):
    name = "read_file"
    description = "Read a file with line numbers. offset and limit select a line range."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "1-based first line.", "minimum": 1},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    safety = _read("Reads a file inside the workspace.")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        target = jail(ctx.workspace, args["path"])
        if not target.exists():
            return Result.error(f"Error: file not found: {args['path']}")
        if target.is_dir():
            return Result.error(f"Error: {args['path']} is a directory; use list_directory")
        data = target.read_bytes()
        ctx.files.record(target, data)
        truncated = len(data) > MAX_READ_BYTES
        text = data[:MAX_READ_BYTES].decode("utf-8", errors="replace")
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        start = max(1, int(args.get("offset") or 1))
        limit = args.get("limit")
        end = start - 1 + int(limit) if limit else len(lines)
        window = lines[start - 1:end]
        out = "\n".join(f"{start + i}→{line}" for i, line in enumerate(window))
        if truncated:
            out += f"\n[... file truncated at {MAX_READ_BYTES} bytes ...]"
        return Result.success(out, truncated=truncated)


class WriteFile(Tool):
    name = "write_file"
    description = "Create a file. Overwriting needs overwrite=true and a prior read."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "description": {"type": "string", "description": "Why, one short line."},
            "content": {"type": "string"},
            "overwrite": {"type": "boolean", "default": False},
        },
        "required": ["path", "description", "content"],
        "additionalProperties": False,
    }
    safety = _write("Creates or overwrites a file.")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        target = jail(ctx.workspace, args["path"])
        rel = relative(ctx.workspace, target)
        if target.exists():
            if not args.get("overwrite"):
                return Result.error(f"Error: {rel} already exists; pass overwrite=true to replace it")
            if problem := ctx.files.conflict(target, target.read_bytes()):
                return Result.error(f"Error: {rel}: {problem}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args["content"])
        ctx.files.record(target, args["content"].encode())
        return Result.success(f"Wrote {len(args['content'])} chars to {rel}", changed_files=[rel])


class EditFile(Tool):
    name = "edit_file"
    description = "Replace an exact, unique string in a file you have read. replace_all for every occurrence."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "description": {"type": "string", "description": "Why, one short line."},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["path", "description", "old_string", "new_string"],
        "additionalProperties": False,
    }
    safety = _write("Edits an existing file in place.")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        target = jail(ctx.workspace, args["path"])
        rel = relative(ctx.workspace, target)
        if not target.is_file():
            return Result.error(f"Error: file not found: {rel}")
        old = args["old_string"]
        if old == "":
            return Result.error("Error: old_string must not be empty")
        data = target.read_bytes()
        if problem := ctx.files.conflict(target, data):
            return Result.error(f"Error: {rel}: {problem}")
        text = data.decode("utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return Result.error(f"Error: old_string not found in {rel}")
        if count > 1 and not args.get("replace_all"):
            return Result.error(
                f"Error: old_string matches {count} times in {rel}; include more surrounding context or pass replace_all=true"
            )
        updated = text.replace(old, args["new_string"])
        target.write_text(updated)
        ctx.files.record(target, updated.encode())
        return Result.success(f"Replaced {count} occurrence(s) in {rel}", changed_files=[rel])


class ListDirectory(Tool):
    name = "list_directory"
    deferred = True
    description = "List a directory, optionally recursive."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list. Defaults to workspace root.", "default": "."},
            "recursive": {"type": "boolean", "description": "Whether to list recursively.", "default": False},
            "max_depth": {"type": "integer", "description": "Maximum recursion depth when recursive is true.", "default": 2, "minimum": 1, "maximum": 5},
        },
        "additionalProperties": False,
    }
    safety = _read("Lists directory entries inside the workspace.")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        base = jail(ctx.workspace, args.get("path") or ".")
        if not base.is_dir():
            return Result.error(f"Error: not a directory: {args.get('path') or '.'}")
        depth = int(args.get("max_depth") or 2) if args.get("recursive") else 1
        rows = []
        for entry in _walk(base, depth):
            rel = entry.relative_to(base).as_posix()
            rows.append(rel + "/" if entry.is_dir() else rel)
        rows.sort()
        truncated = len(rows) > MAX_ENTRIES
        out = "\n".join(rows[:MAX_ENTRIES])
        if truncated:
            out += f"\n[... {len(rows) - MAX_ENTRIES} more entries ...]"
        return Result.success(out or "(empty)", truncated=truncated)


class Glob(Tool):
    name = "glob"
    deferred = True
    description = "Find files by glob pattern."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": 'Glob pattern, for example "**/*.py".'},
            "cwd": {"type": "string", "description": "Directory to scan. Defaults to workspace root.", "default": "."},
            "limit": {"type": "integer", "description": "Maximum matches to return.", "default": 100, "minimum": 1, "maximum": 1000},
            "include_dirs": {"type": "boolean", "description": "Whether directory matches should be included.", "default": False},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    safety = _read("Matches file names inside the workspace.")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        base = jail(ctx.workspace, args.get("cwd") or ".")
        limit = int(args.get("limit") or 100)
        rows = []
        for match in sorted(base.glob(args["pattern"])):
            if any(part in IGNORED_DIRS for part in match.relative_to(base).parts):
                continue
            if match.is_dir() and not args.get("include_dirs"):
                continue
            rows.append(match.relative_to(base).as_posix())
        truncated = len(rows) > limit
        out = "\n".join(rows[:limit])
        if truncated:
            out += f"\n[... {len(rows) - limit} more matches ...]"
        return Result.success(out or "(no matches)", truncated=truncated)


class Grep(Tool):
    name = "grep"
    description = "Search file contents with a regex."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "glob": {"type": "string", "description": 'File-name filter such as "*.py".'},
            "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"], "default": "content"},
            "case_insensitive": {"type": "boolean", "default": False},
            "head_limit": {"type": "integer", "default": 50},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    safety = _read("Searches file contents inside the workspace.")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        base = jail(ctx.workspace, args.get("path") or ".")
        flags = re.IGNORECASE if args.get("case_insensitive") else 0
        try:
            rx = re.compile(args["pattern"], flags)
        except re.error as e:
            return Result.error(f"Error: invalid regex: {e}")
        mode = args.get("output_mode") or "content"
        head = int(args.get("head_limit") or 50)
        name_filter = args.get("glob")
        files = [base] if base.is_file() else [p for p in _walk(base, None) if p.is_file()]
        root = Path(ctx.workspace).resolve()
        rows: list[str] = []
        emitted = 0
        truncated = False
        for f in sorted(files):
            rel = f.relative_to(root).as_posix()
            if name_filter and not PurePath(rel).match(name_filter):
                continue
            try:
                text = f.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            hits = [(i, line) for i, line in enumerate(text.splitlines(), 1) if rx.search(line)]
            if not hits:
                continue
            if mode == "files_with_matches":
                rows.append(rel)
            elif mode == "count":
                rows.append(f"{rel}:{len(hits)}")
            else:
                for i, line in hits:
                    if emitted >= head:
                        truncated = True
                        break
                    rows.append(f"{rel}:{i}:{line}")
                    emitted += 1
        out = "\n".join(rows)
        if truncated:
            out += f"\n[... more matches; raise head_limit or narrow the pattern ...]"
        return Result.success(out or "(no matches)", truncated=truncated)


def core_file_tools() -> list[Tool]:
    return [ReadFile(), WriteFile(), EditFile(), ListDirectory(), Glob(), Grep()]
