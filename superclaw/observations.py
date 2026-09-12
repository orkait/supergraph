from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from supergraph.core.errors import SuperGraphError

from superclaw.runtime import approx_tokens
from superclaw.session import NAMESPACE, _lit
from superclaw.settings import LIMITS
from superclaw.tools import Permission, Result, Safety, SideEffect, Tool, ToolContext

REF = re.compile(r"§([0-9a-f]{8,})")
KIND = "obs"


def ref_in(text: str) -> str:
    match = REF.search(text)
    return match.group(1) if match else ""


@dataclass(frozen=True)
class Observation:
    ref: str
    tool: str
    body: str

    @property
    def tokens(self) -> int:
        return approx_tokens(self.body)

    def chunk(self, index: int) -> tuple[str, int]:
        size = LIMITS.recall_chunk_tokens * LIMITS.chars_per_token
        total = max(1, -(-len(self.body) // size))
        index = min(max(0, index), total - 1)
        return self.body[index * size:(index + 1) * size], total


class ObservationStore:
    def __init__(self, gs: Any) -> None:
        self._gs = gs

    def _x(self, query: str):
        return self._gs.execute(query, namespace=NAMESPACE)

    def _document(self, ref: str) -> str | None:
        try:
            data = self._x(f'NODE {_lit(KIND + ":" + ref)} WITH DOCUMENT').data
        except SuperGraphError:
            return None
        return None if not data else data.get("_document") or ""

    def save(self, session_id: str, tool: str, call_id: str, body: str) -> str:
        digest = hashlib.sha256(f"{tool}\0{call_id}\0{body}".encode()).hexdigest()
        ref = digest[:LIMITS.ref_hex_chars]
        while (existing := self._document(ref)) is not None and existing != body:
            ref = digest[:len(ref) + LIMITS.ref_hex_step]
        if existing is None:
            self._x(
                f'CREATE NODE {_lit(KIND + ":" + ref)} kind = {_lit(KIND)} sid = {_lit(session_id)} tool = {_lit(tool)} '
                f'call_id = {_lit(call_id)} tokens = {approx_tokens(body)} chars = {len(body)} DOCUMENT {_lit(body)}'
            )
        return ref

    def load(self, ref: str) -> Observation | None:
        body = self._document(ref)
        if body is None:
            return None
        row = self._x(f'NODE {_lit(KIND + ":" + ref)}').data or {}
        return Observation(ref, str(row.get("tool", "")), body)

    def search(self, query: str, limit: int = LIMITS.recall_search_limit) -> list[Observation]:
        try:
            rows = self._x(f'REMEMBER {_lit(query)} LIMIT {int(limit)} WHERE kind = {_lit(KIND)}').data or []
        except SuperGraphError:
            return []
        found = [self.load(str(r["id"]).removeprefix(KIND + ":")) for r in rows]
        return [o for o in found if o]


class Recall(Tool):
    name = "recall"
    deferred = True
    description = (
        "Bring back an earlier tool result that was shortened or pruned. Pass ref (the §id shown in the result) and an optional chunk, "
        "or a query to search every stored result by meaning."
    )
    parameters = {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "The id after § in a result."},
            "chunk": {"type": "integer", "minimum": 0, "default": 0},
            "query": {"type": "string", "description": "Find results by meaning when the id is unknown."},
        },
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.READ, Permission.ALLOW, "Reads stored tool results.")

    def __init__(self, store: ObservationStore) -> None:
        self._store = store

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        ref = str(args.get("ref") or "").lstrip("§").strip()
        if ref:
            obs = self._store.load(ref)
            if obs is None:
                return Result.error(f"Error: no stored result §{ref}; pass a query to search instead")
            text, total = obs.chunk(int(args.get("chunk") or 0))
            index = min(int(args.get("chunk") or 0), total - 1)
            head = f"[§{obs.ref} {obs.tool}, chunk {index + 1} of {total}, {obs.tokens:,} tokens in total]\n"
            return Result.success(head + text)
        query = str(args.get("query") or "").strip()
        if not query:
            return Result.error("Error: pass ref or query")
        hits = self._store.search(query)
        if not hits:
            return Result.success("No stored result matches.")
        lines = [f"§{o.ref} {o.tool} {o.tokens:,} tokens: {o.body[:LIMITS.recall_preview_chars]!r}" for o in hits]
        return Result.success("\n".join(lines))
