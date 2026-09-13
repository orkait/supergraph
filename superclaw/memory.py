from __future__ import annotations

import hashlib
import re
import time
from typing import Any

from supergraph.core.errors import SuperGraphError

from superclaw.redaction import redact
from superclaw.settings import LIMITS
from superclaw.tools import Permission, Result, Safety, SideEffect, Tool, ToolContext

ORIGINS = ("user_stated", "user_selected", "inferred")
_MS_PER_SECOND = 1000
_MS_PER_DAY = 86_400_000
_DAYS_PER_MONTH = 30
_DAYS_PER_YEAR = 365
_HONESTY_TRAPS = re.compile(
    r"(?i)\b(never|don't|do not|stop|avoid)\s+(disagree|question|challenge|push back|raise|mention|flag|verify|check|test|warn|correct)\b"
    r"|\b(always|just)\s+(agree|comply|approve|say yes)\b"
    r"|\bignore (all )?(previous|prior|earlier) instructions\b"
    r"|\b(assume|treat) .*\b(permission|approved|authorized)\b"
)


def _lit(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _rows(result: Any) -> list[dict]:
    data = getattr(result, "data", None)
    return data if isinstance(data, list) else []


def _now_ms() -> int:
    return int(time.time() * _MS_PER_SECOND)


def _age(stated_at_ms: int, now_ms: int | None = None) -> str:
    now = now_ms if now_ms is not None else _now_ms()
    days = max(0, (now - stated_at_ms) // _MS_PER_DAY)
    if days == 0:
        return "today"
    if days < _DAYS_PER_MONTH:
        return f"{days}d ago"
    if days < _DAYS_PER_YEAR:
        return f"{days // _DAYS_PER_MONTH}mo ago"
    return f"{days // _DAYS_PER_YEAR}y ago"


def refusal(text: str, origin: str) -> str:
    if origin not in ORIGINS:
        return f"origin must be one of {', '.join(ORIGINS)}"
    if origin == "inferred":
        return "only what the user stated is filed; a choice they made among options counts, your inference or advice does not"
    if _HONESTY_TRAPS.search(text):
        return "refused: an instruction that would stop a future session raising an error, risk or disagreement is never filed, however it is phrased"
    if redact(text)[1]:
        return "refused: the text contains a secret"
    return ""


class Memory:
    def __init__(self, gs: Any) -> None:
        self._gs = gs

    def note(self, text: str, *, origin: str = "user_stated", expires_days: int | None = None) -> str:
        if problem := refusal(text, origin):
            raise ValueError(problem)
        node_id = "mem:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:LIMITS.id_hash_chars]
        expires = f" EXPIRES IN {int(expires_days)}d" if expires_days else ""
        try:
            self._gs.execute(
                f'CREATE NODE {_lit(node_id)} kind = "memory" origin = {_lit(origin)} '
                f'stated_at = {_now_ms()}{expires} DOCUMENT {_lit(text)}'
            )
        except SuperGraphError as e:
            if "exist" not in str(e).lower():
                raise
        return node_id

    def _search(self, query: str, limit: int) -> list[dict]:
        try:
            return _rows(self._gs.execute(f'REMEMBER {_lit(query)} LIMIT {int(limit)} WHERE kind = "memory"'))
        except SuperGraphError:
            return []

    def _doc(self, node_id: str) -> tuple[str, int]:
        try:
            data = self._gs.execute(f"NODE {_lit(node_id)} WITH DOCUMENT").data or {}
        except SuperGraphError:
            return "", 0
        return (data.get("_document") or "").strip(), int(data.get("stated_at") or 0)

    def hits(self, query: str, limit: int = LIMITS.memory_recall_limit) -> list[tuple[str, str, int]]:
        found = [(r["id"], *self._doc(r["id"])) for r in self._search(query, limit)]
        return [(node_id, text, stated_at) for node_id, text, stated_at in found if text]

    def recall(self, query: str, limit: int = LIMITS.memory_recall_limit) -> str:
        return "\n".join(f"- ({_age(stated_at)}) {text}" if stated_at else f"- {text}" for _, text, stated_at in self.hits(query, limit))

    def search_tool(self) -> Tool:
        return _MemorySearch(self)

    def note_tool(self) -> Tool:
        return _MemoryNote(self)


class _MemorySearch(Tool):
    name = "memory_search"
    deferred = True
    description = "Search long-term memory for facts, decisions, preferences or history relevant to the task."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to recall."},
            "limit": {"type": "integer", "description": "Maximum results.", "default": LIMITS.memory_recall_limit, "minimum": 1, "maximum": LIMITS.memory_recall_max},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.READ, Permission.ALLOW, "Reads from long-term memory.")

    def __init__(self, memory: Memory) -> None:
        self._m = memory

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        limit = int(args.get("limit") or LIMITS.memory_recall_limit)
        hits = self._m.hits(str(args.get("query") or ""), limit)
        lines = [f"{node_id} ({_age(stated_at) if stated_at else 'undated'}): {text}" for node_id, text, stated_at in hits]
        return Result.success("\n".join(lines) if lines else "No matching memories.")


class _MemoryNote(Tool):
    name = "memory_note"
    deferred = True
    description = (
        "File a durable fact the user stated, or a choice they made among options, so a later session recalls it. "
        "Never file your own inference, advice or reasoning, transient details, or any instruction that would keep a future session "
        "from raising an error, risk or disagreement."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The fact, as one self-contained sentence in the user's terms."},
            "origin": {"type": "string", "enum": list(ORIGINS), "description": "user_stated: they said it. user_selected: they picked it among options you offered. inferred: you concluded it (refused)."},
            "expires_days": {"type": "integer", "minimum": 1, "description": "Optional lifetime in days for facts that go stale, such as a temporary setup."},
        },
        "required": ["text", "origin"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.NONE, Permission.ALLOW, "Writes to the agent's own long-term memory.")

    def __init__(self, memory: Memory) -> None:
        self._m = memory

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        text = str(args.get("text") or "").strip()
        if not text:
            return Result.error("Error: text must not be empty")
        try:
            return Result.success(self._m.note(text, origin=str(args.get("origin") or ""), expires_days=args.get("expires_days")))
        except ValueError as e:
            return Result.error(f"Error: {e}")
