from __future__ import annotations

import difflib
import re
from pathlib import Path, PurePath
from typing import Any

from superclaw.settings import LIMITS
from superclaw.tools import (
    Category,
    Display,
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


def _clip_line(line: str) -> str:
    cap = LIMITS.read_line_chars
    return line if len(line) <= cap else f"{line[:cap]}… [+{len(line) - cap:,} chars]"


def _diff_display(rel: str, before: str, after: str, summary: str) -> Display:
    diff = "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile=rel, tofile=rel))
    return Display(summary=summary, kind="diff", preview=diff if len(diff) <= LIMITS.diff_preview_bytes else "")


def _is_binary(path: Path) -> bool:
    with path.open("rb") as handle:
        return b"\0" in handle.read(LIMITS.binary_sniff_bytes)


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
    description = "Read a file with line numbers (offset, limit). Ranges already in your context are skipped unless force=true."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "1-based first line.", "minimum": 1},
            "limit": {"type": "integer", "minimum": 1},
            "force": {"type": "boolean", "default": False},
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    safety = _read("Reads a file inside the workspace.")
    output_category = Category.FILE

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        target = jail(ctx.workspace, args["path"])
        rel = args["path"]
        if not target.exists():
            return Result.error(f"Error: file not found: {rel}")
        if target.is_dir():
            return Result.error(f"Error: {rel} is a directory; use list_directory")
        data = target.read_bytes()
        truncated = len(data) > LIMITS.read_file_bytes
        lines = data[:LIMITS.read_file_bytes].decode("utf-8", errors="replace").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        start = max(1, int(args.get("offset") or 1))
        end = min(len(lines), start - 1 + int(args.get("limit") or LIMITS.read_file_lines))
        if not args.get("force") and (window := ctx.files.in_context(target, start, end, data)):
            return Result.success(f"{rel} lines {start}-{end} are already in your context (sent unchanged as lines {window.start}-{window.end}); use them, or pass force=true to re-send.")
        ctx.files.record(target, data)
        ctx.files.shown(target, start, end)
        out = "\n".join(f"{start + i}→{_clip_line(line)}" for i, line in enumerate(lines[start - 1:end]))
        if end < len(lines):
            out += f"\n[{len(lines) - end:,} more lines; call read_file with offset={end + 1} to continue]"
        if truncated:
            out += f"\n[file is {len(data):,} bytes; only the first {LIMITS.read_file_bytes:,} were read]"
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
        before = ""
        if target.exists():
            if not args.get("overwrite"):
                return Result.error(f"Error: {rel} already exists; pass overwrite=true to replace it")
            data = target.read_bytes()
            if problem := ctx.files.conflict(target, data):
                return Result.error(f"Error: {rel}: {problem}")
            before = data.decode("utf-8", errors="replace")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args["content"])
        ctx.files.record(target, args["content"].encode())
        summary = f"Wrote {len(args['content'])} chars to {rel}"
        return Result.success(summary, changed_files=[rel], display=_diff_display(rel, before, args["content"], summary))


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
        summary = f"Replaced {count} occurrence(s) in {rel}"
        return Result.success(summary, changed_files=[rel], display=_diff_display(rel, text, updated, summary))


class ListDirectory(Tool):
    name = "list_directory"
    deferred = True
    description = "List a directory, optionally recursive."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list. Defaults to workspace root.", "default": "."},
            "recursive": {"type": "boolean", "description": "Whether to list recursively.", "default": False},
            "max_depth": {"type": "integer", "description": "Maximum recursion depth when recursive is true.", "default": LIMITS.list_directory_depth, "minimum": 1, "maximum": LIMITS.list_directory_max_depth},
        },
        "additionalProperties": False,
    }
    safety = _read("Lists directory entries inside the workspace.")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        base = jail(ctx.workspace, args.get("path") or ".")
        if not base.is_dir():
            return Result.error(f"Error: not a directory: {args.get('path') or '.'}")
        depth = int(args.get("max_depth") or LIMITS.list_directory_depth) if args.get("recursive") else 1
        rows = []
        for entry in _walk(base, depth):
            rel = entry.relative_to(base).as_posix()
            rows.append(rel + "/" if entry.is_dir() else rel)
        rows.sort()
        truncated = len(rows) > LIMITS.list_directory_entries
        out = "\n".join(rows[:LIMITS.list_directory_entries])
        if truncated:
            out += f"\n[... {len(rows) - LIMITS.list_directory_entries} more entries ...]"
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
            "limit": {"type": "integer", "description": "Maximum matches to return.", "default": LIMITS.glob_limit, "minimum": 1, "maximum": LIMITS.glob_max_limit},
            "include_dirs": {"type": "boolean", "description": "Whether directory matches should be included.", "default": False},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    safety = _read("Matches file names inside the workspace.")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        base = jail(ctx.workspace, args.get("cwd") or ".")
        limit = int(args.get("limit") or LIMITS.glob_limit)
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
            "head_limit": {"type": "integer", "default": LIMITS.grep_head_limit},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    safety = _read("Searches file contents inside the workspace.")
    output_category = Category.SEARCH

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        base = jail(ctx.workspace, args.get("path") or ".")
        try:
            rx = re.compile(args["pattern"], re.IGNORECASE if args.get("case_insensitive") else 0)
        except re.error as e:
            return Result.error(f"Error: invalid regex: {e}")
        mode = args.get("output_mode") or "content"
        head = int(args.get("head_limit") or LIMITS.grep_head_limit)
        root = Path(ctx.workspace).resolve()
        rows: list[str] = []
        truncated = False
        for rel, hits in self._matches(base, root, rx, args.get("glob")):
            if mode == "files_with_matches":
                rows.append(rel)
            elif mode == "count":
                rows.append(f"{rel}:{len(hits)}")
            else:
                room = head - len(rows)
                rows += [f"{rel}:{i}:{line}" for i, line in hits[:room]]
                truncated = truncated or len(hits) > room
        out = "\n".join(rows)
        if truncated:
            out += "\n[... more matches; raise head_limit or narrow the pattern ...]"
        return Result.success(out or "(no matches)", truncated=truncated)

    @staticmethod
    def _matches(base: Path, root: Path, rx: re.Pattern[str], name_filter: str | None):
        files = [base] if base.is_file() else [p for p in _walk(base, None) if p.is_file()]
        for f in sorted(files):
            rel = f.relative_to(root).as_posix()
            if name_filter and not PurePath(rel).match(name_filter):
                continue
            try:
                if f.stat().st_size > LIMITS.read_file_bytes or _is_binary(f):
                    continue
                text = f.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            hits = [(i, line) for i, line in enumerate(text.splitlines(), 1) if rx.search(line)]
            if hits:
                yield rel, hits


def core_file_tools() -> list[Tool]:
    return [ReadFile(), WriteFile(), EditFile(), ListDirectory(), Glob(), Grep()]
