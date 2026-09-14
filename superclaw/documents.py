from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from superclaw.dsl import Result, Store, edge, lit, rows
from superclaw.session import NAMESPACE
from superclaw.settings import CHUNK_KIND, DOCUMENT_KIND, INGESTED_EDGE, LIMITS
from supergraph.core.errors import SuperGraphError


@dataclass(frozen=True)
class Document:
    id: str
    path: str
    chunks: int
    parser: str
    confidence: float


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    text: str


def ident(path: Path) -> str:
    stamp = f"{path}\0{path.stat().st_size}\0{path.stat().st_mtime_ns}"
    return f"{DOCUMENT_KIND[:3]}:" + hashlib.sha1(stamp.encode("utf-8"), usedforsecurity=False).hexdigest()[: LIMITS.id_hash_chars]


class Documents:
    def __init__(self, gs: Store) -> None:
        self._gs = gs

    def _x(self, query: str) -> Result:
        return self._gs.execute(query, namespace=NAMESPACE)

    def ingest(self, path: Path, *, session_id: str = "", ttl_days: int = LIMITS.ingest_ttl_days, pin: bool = False) -> Document:
        node = ident(path)
        try:
            data = self._x(f"INGEST {lit(str(path))} AS {lit(node)} KIND {lit(DOCUMENT_KIND)}").data or {}
        except SuperGraphError as e:
            if "exist" not in str(e).lower():
                raise
            existing = self.load(node)
            if existing is None:
                raise
            data = {"chunks": existing.chunks, "parser": existing.parser, "confidence": existing.confidence}
        if not pin:
            self.expire(node, ttl_days)
        if session_id:
            edge(self._gs, node, f"session:{session_id}", INGESTED_EDGE, NAMESPACE)
        self._x(f"UPDATE NODE {lit(node)} SET path = {lit(str(path))} pinned = {int(pin)}")
        return Document(node, str(path), int(data.get("chunks") or 0), str(data.get("parser") or ""), float(data.get("confidence") or 0))

    def expire(self, node: str, ttl_days: int) -> None:
        self._x(f"UPSERT NODE {lit(node)} kind = {lit(DOCUMENT_KIND)} EXPIRES IN {int(ttl_days)}d")
        for child in rows(self._x(f"NODES WHERE id LIKE {lit(node + ':%')}")):
            self._x(f"UPSERT NODE {lit(str(child['id']))} EXPIRES IN {int(ttl_days)}d")

    def load(self, node: str) -> Document | None:
        try:
            data = self._x(f"NODE {lit(node)}").data or {}
        except SuperGraphError:
            return None
        if not data:
            return None
        chunks = len(rows(self._x(f"NODES WHERE kind = {lit(CHUNK_KIND)} AND id LIKE {lit(node + ':chunk:%')}")))
        return Document(node, str(data.get("path") or data.get("source") or ""), chunks, str(data.get("parser") or ""), float(data.get("confidence") or 0))

    def read(self, node: str, index: int = 0) -> Chunk | None:
        chunk = f"{node}:chunk:{int(index)}"
        try:
            data = self._x(f"NODE {lit(chunk)} WITH DOCUMENT").data or {}
        except SuperGraphError:
            return None
        text = (data.get("_document") or "").strip()
        return Chunk(chunk, node, text) if text else None

    def search(self, query: str, limit: int = LIMITS.doc_search_limit) -> list[Chunk]:
        try:
            found = rows(self._x(f"REMEMBER {lit(query)} LIMIT {int(limit)} WHERE kind = {lit(CHUNK_KIND)}"))
        except SuperGraphError:
            return []
        chunks = []
        for row in found:
            node = str(row["id"])
            doc, _, index = node.rpartition(":chunk:")
            if (chunk := self.read(doc, int(index or 0))) is not None:
                chunks.append(chunk)
        return chunks
