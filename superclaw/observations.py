from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from supergraph.core.errors import SuperGraphError

from superclaw.dsl import edge
from superclaw.runtime import approx_tokens
from superclaw.session import NAMESPACE, _lit
from superclaw.settings import LEARNED_EDGE, LIMITS, PRODUCED_EDGE
from superclaw.tools import Permission, Result, Safety, SideEffect, Tool, ToolContext, jail

REF = re.compile(r"§([0-9a-f]{8,})")
KIND = "obs"
KERNEL_KIND = "kernel"


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

    def save(self, session_id: str, tool: str, call_id: str, body: str, expires_days: int = 0) -> str:
        digest = hashlib.sha256(f"{tool}\0{call_id}\0{body}".encode()).hexdigest()
        ref = digest[:LIMITS.ref_hex_chars]
        while (existing := self._document(ref)) is not None and existing != body:
            ref = digest[:len(ref) + LIMITS.ref_hex_step]
        if existing is None:
            expires = f" EXPIRES IN {int(expires_days)}d" if expires_days else ""
            self._x(
                f'CREATE NODE {_lit(KIND + ":" + ref)} kind = {_lit(KIND)} sid = {_lit(session_id)} tool = {_lit(tool)} '
                f'call_id = {_lit(call_id)} tokens = {approx_tokens(body)} chars = {len(body)}{expires} DOCUMENT {_lit(body)}'
            )
            if session_id:
                edge(self._gs, f"{KIND}:{ref}", f"session:{session_id}", PRODUCED_EDGE, NAMESPACE)
        return ref

    def load(self, ref: str) -> Observation | None:
        body = self._document(ref)
        if body is None:
            return None
        row = self._x(f'NODE {_lit(KIND + ":" + ref)}').data or {}
        return Observation(ref, str(row.get("tool", "")), body)

    def save_kernel(self, session_id: str, blob: str) -> None:
        node = _lit(KERNEL_KIND + ":" + session_id)
        if self.load_kernel(session_id):
            self._x(f"DELETE NODE {node}")
        self._x(f'CREATE NODE {node} kind = {_lit(KERNEL_KIND)} sid = {_lit(session_id)} DOCUMENT {_lit(blob)}')

    def load_kernel(self, session_id: str) -> str:
        try:
            data = self._x(f'NODE {_lit(KERNEL_KIND + ":" + session_id)} WITH DOCUMENT').data
        except SuperGraphError:
            return ""
        return (data or {}).get("_document") or ""

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
        "a query to search every stored result by meaning, or a path to see which earlier sessions read or wrote that file and what they learned and stored."
    )
    parameters = {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "The id after § in a result."},
            "chunk": {"type": "integer", "minimum": 0, "default": 0},
            "query": {"type": "string", "description": "Find results by meaning when the id is unknown."},
            "path": {"type": "string", "description": "Workspace path; lists the sessions that touched it, their facts and stored results."},
            "doc": {"type": "string", "description": "An ingested document id (doc:...); reads its chunk at chunk."},
        },
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.READ, Permission.ALLOW, "Reads stored tool results.")

    def __init__(self, store: ObservationStore, sessions: Any | None = None, facts: Any | None = None, documents: Any | None = None) -> None:
        self._store = store
        self._sessions = sessions
        self._facts = facts
        self._documents = documents

    def chunk(self, doc: str, index: int) -> Result:
        found = self._documents.read(doc, index) if self._documents else None
        if found is None:
            return Result.error(f"Error: no chunk {index} in {doc}; recall with a query lists chunks")
        total = self._documents.load(doc).chunks if self._documents.load(doc) else index + 1
        return Result.success(f"[{found.id}, chunk {index + 1} of {total}]\n{found.text}")

    def around(self, path: str, ctx: ToolContext) -> Result:
        if self._sessions is None:
            return Result.error("Error: no session store in this run")
        target = str(jail(ctx.roots, path))
        sessions = self._sessions.touching(target)[: LIMITS.recall_path_sessions]
        if not sessions:
            return Result.success(f"No earlier session touched {path}.")
        lines = [f"{path}: touched by {len(sessions)} session(s)"]
        for session in sessions:
            linked = self._sessions.around(session["id"])
            verbs = sorted({verb for file, verb in self._sessions.files_of(session["id"]) if file == target})
            lines.append(f"{session['id']} {session.get('title') or '(untitled)'!r}: {', '.join(verbs) or 'touched'}")
            for node in linked.get(LEARNED_EDGE, []):
                fact = self._facts.load(node) if self._facts else None
                if fact:
                    lines.append(f"  fact {fact.line()}")
            refs = [node.removeprefix(f"{KIND}:") for node in linked.get(PRODUCED_EDGE, [])]
            if refs:
                lines.append("  results: " + ", ".join(f"§{ref}" for ref in refs[: LIMITS.recall_path_refs]))
        return Result.success("\n".join(lines))

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        if args.get("path"):
            return self.around(str(args["path"]), ctx)
        if args.get("doc"):
            return self.chunk(str(args["doc"]), int(args.get("chunk") or 0))
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
        chunks = self._documents.search(query) if self._documents else []
        if not hits and not chunks:
            return Result.success("No stored result matches.")
        lines = [f"§{o.ref} {o.tool} {o.tokens:,} tokens: {o.body[:LIMITS.recall_preview_chars]!r}" for o in hits]
        lines += [f"{c.id}: {c.text[:LIMITS.chunk_preview_chars]!r}" for c in chunks]
        return Result.success("\n".join(lines))
