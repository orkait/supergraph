"""Research: the lean, high-altitude SDK for supergraph as a deep-research memory.

Eight verbs - ingest, search, answer, gaps, relate, forget, explore, execute -
that speak research, not graph-DB. Everything underneath (DSL, the vector/lexical/
graph fusion, the embedder, chunking, NER, SYS verbs) is hidden. Power users drop
to ``execute`` for raw DSL. This is the surface to hand to other people; the full
``SuperGraph`` class stays the engine behind it.

Single embedder, single coherent corpus - deliberate. A deep researcher needs all
evidence in one comparable space so it can be synthesised; it is not a generic
multi-tenant search engine.
"""
from __future__ import annotations

import hashlib
from typing import Any

from supergraph.core.errors import SuperGraphError
from supergraph.store import SuperGraph


def _lit(value: str) -> str:
    """Render a string as a DSL double-quoted literal (supergraph's escaping)."""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _rows(result: Any) -> list[dict]:
    data = getattr(result, "data", result)
    return data if isinstance(data, list) else []


class Research:
    """A deep-research memory. Wrap a SuperGraph, or open one with ``Research.open``."""

    def __init__(self, store: SuperGraph) -> None:
        self._gs = store

    @classmethod
    def open(cls, path: str | None = None, **kwargs: Any) -> "Research":
        return cls(SuperGraph(path=path, **kwargs))

    # ── write ────────────────────────────────────────────────────────────────
    def ingest(self, content: str | list[str], *, structure: bool = False) -> str | list[str]:
        """Add evidence to the corpus. Embeds + indexes the text so it is
        retrievable by meaning and keyword. Pass a list to ingest many at once.

        ``structure=True`` runs the slower NL->graph extraction (entities +
        relations) instead of storing the text as a single evidence node - use it
        sparingly, the cheap default is right for gathering volume.
        """
        if isinstance(content, list):
            return [self._ingest_one(c, structure=structure) for c in content]
        return self._ingest_one(content, structure=structure)

    def _ingest_one(self, content: str, *, structure: bool) -> str:
        if structure:
            res = self._gs.ingest_nl(content)
            return getattr(res, "msg_id", "") or ""
        node_id = "ev:" + hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
        try:
            self._gs.execute(
                f"CREATE NODE {_lit(node_id)} kind = \"evidence\" DOCUMENT {_lit(content)}"
            )
        except SuperGraphError as e:
            if "exist" not in str(e).lower():
                raise  # re-ingesting identical content is a no-op, anything else is real
        return node_id

    # ── read ─────────────────────────────────────────────────────────────────
    def search(self, query: str, *, limit: int = 10) -> list[dict]:
        """Retrieve the most relevant evidence for a query. One smart read - it
        fuses semantic, keyword, and graph signals; you never pick a mode."""
        return _rows(self._gs.execute(f"REMEMBER {_lit(query)} LIMIT {int(limit)}"))

    def answer(self, question: str, *, limit: int | None = None) -> Any:
        """Ask a natural-language question; get a synthesised answer grounded in
        the corpus. Requires an LLM reader configured on the store."""
        return getattr(self._gs.ask(question, limit=limit), "data", None)

    def gaps(
        self, *, limit: int = 10, max_confidence: float = 0.6,
        contradiction_field: str | None = None, group_by: str = "id",
    ) -> list[dict]:
        """The research frontier - what the corpus is weakest on. Aggregates
        under-confidence (gather), sparse/under-sourced entities (expand), and
        optionally contradictions (resolve). Deduped, capped at ``limit``."""
        out: list[dict] = []
        seen: set[str] = set()

        def _add(node_id: str, name: str | None, kind: str, reason: str) -> None:
            if node_id and node_id not in seen:
                seen.add(node_id)
                out.append({"id": node_id, "name": name, "kind": kind, "reason": reason})

        for row in _rows(self._gs.execute(
            f"NODES WHERE confidence < {float(max_confidence)} LIMIT {int(limit)}"
        )):
            _add(row.get("id"), row.get("name"), "gather", "low confidence")
        for row in _rows(self._gs.execute(
            f'NODES WHERE kind = "entity" AND INDEGREE < 1 LIMIT {int(limit)}'
        )):
            _add(row.get("id"), row.get("name"), "expand", "sparse / under-sourced")
        if contradiction_field:
            try:
                for row in _rows(self._gs.execute(
                    f"SYS CONTRADICTIONS FIELD {contradiction_field} GROUP BY {group_by}"
                )):
                    grp = str(row.get(group_by) or row.get("group") or "")
                    _add(grp, grp, "resolve", "contradiction")
            except SuperGraphError:
                pass
        return out[:limit]

    # ── graph ────────────────────────────────────────────────────────────────
    def relate(self, source: str, target: str, *, kind: str) -> None:
        """Record a connection you know between two records."""
        self._gs.execute(
            f"CREATE EDGE {_lit(source)} -> {_lit(target)} kind = {_lit(kind)}"
        )

    def explore(self, node_id: str, *, depth: int = 2, limit: int = 50) -> list[dict]:
        """Follow connections outward from a record - the surrounding subgraph."""
        return _rows(self._gs.execute(
            f"RECALL FROM {_lit(node_id)} DEPTH {int(depth)} LIMIT {int(limit)}"
        ))

    def forget(self, node_id: str) -> None:
        """Retract a record - it stops being visible to search and answer."""
        self._gs.execute(f"RETRACT {_lit(node_id)}")

    # ── escape hatch ───────────────────────────────────────────────────────────
    def execute(self, dsl: str) -> Any:
        """Raw DSL. Reach for this only when the eight verbs above don't cover it."""
        return self._gs.execute(dsl)

    def close(self) -> None:
        self._gs.close()


__all__ = ["Research"]
