from __future__ import annotations

import hashlib
from typing import Any

from supergraph.core.errors import SuperGraphError

from superclaw.tools import Permission, Result, Safety, SideEffect, Tool, ToolContext

_DEFAULT_LIMIT = 5


def _lit(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _rows(result: Any) -> list[dict]:
    data = getattr(result, "data", None)
    return data if isinstance(data, list) else []


class Memory:
    def __init__(self, gs: Any) -> None:
        self._gs = gs

    def note(self, text: str) -> str:
        node_id = "mem:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        try:
            self._gs.execute(f'CREATE NODE {_lit(node_id)} kind = "memory" DOCUMENT {_lit(text)}')
        except SuperGraphError as e:
            if "exist" not in str(e).lower():
                raise
        return node_id

    def _search(self, query: str, limit: int) -> list[dict]:
        try:
            return _rows(self._gs.execute(f"REMEMBER {_lit(query)} LIMIT {int(limit)}"))
        except SuperGraphError:
            return []

    def _doc(self, node_id: str) -> str:
        try:
            data = self._gs.execute(f"NODE {_lit(node_id)} WITH DOCUMENT").data or {}
        except SuperGraphError:
            return ""
        return (data.get("_document") or "").strip()

    def hits(self, query: str, limit: int = _DEFAULT_LIMIT) -> list[tuple[str, str]]:
        found = [(r["id"], self._doc(r["id"])) for r in self._search(query, limit)]
        return [(node_id, text) for node_id, text in found if text]

    def recall(self, query: str, limit: int = _DEFAULT_LIMIT) -> str:
        return "\n".join(f"- {text}" for _, text in self.hits(query, limit))

    def search_tool(self) -> Tool:
        return _MemorySearch(self)

    def note_tool(self) -> Tool:
        return _MemoryNote(self)


class _MemorySearch(Tool):
    name = "memory_search"
    description = "Search long-term memory for facts, decisions, preferences or history relevant to the task."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to recall."},
            "limit": {"type": "integer", "description": "Maximum results.", "default": _DEFAULT_LIMIT, "minimum": 1, "maximum": 20},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.READ, Permission.ALLOW, "Reads from long-term memory.")

    def __init__(self, memory: Memory) -> None:
        self._m = memory

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        limit = int(args.get("limit") or _DEFAULT_LIMIT)
        hits = self._m.hits(str(args.get("query") or ""), limit)
        lines = [f"{node_id}: {text}" for node_id, text in hits]
        return Result.success("\n".join(lines) if lines else "No matching memories.")


class _MemoryNote(Tool):
    name = "memory_note"
    description = (
        "Record a durable fact the user states or a decision worth keeping, so a later session recalls it. "
        "Do not record transient details."
    )
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The fact to remember, as one self-contained sentence."}},
        "required": ["text"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.NONE, Permission.ALLOW, "Writes to the agent's own long-term memory.")

    def __init__(self, memory: Memory) -> None:
        self._m = memory

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        text = str(args.get("text") or "").strip()
        if not text:
            return Result.error("Error: text must not be empty")
        return Result.success(self._m.note(text))
