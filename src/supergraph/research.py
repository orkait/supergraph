from __future__ import annotations

import hashlib
from typing import Any

from supergraph.core.errors import SuperGraphError
from supergraph.store import SuperGraph


def _lit(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _rows(result: Any) -> list[dict]:
    data = getattr(result, "data", result)
    return data if isinstance(data, list) else []


class Research:

    def __init__(self, store: SuperGraph) -> None:
        self._gs = store

    @classmethod
    def open(cls, path: str | None = None, **kwargs: Any) -> "Research":
        return cls(SuperGraph(path=path, **kwargs))

    def ingest(self, content: str | list[str], *, structure: bool = False) -> str | list[str]:
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
                raise
        return node_id

    def search(self, query: str, *, limit: int = 10) -> list[dict]:
        return _rows(self._gs.execute(f"REMEMBER {_lit(query)} LIMIT {int(limit)}"))

    def answer(self, question: str, *, limit: int | None = None) -> Any:
        return getattr(self._gs.ask(question, limit=limit), "data", None)

    def gaps(
        self, *, limit: int = 10, max_confidence: float = 0.6,
        contradiction_field: str | None = None, group_by: str = "id",
    ) -> list[dict]:
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

    def relate(self, source: str, target: str, *, kind: str) -> None:
        self._gs.execute(
            f"CREATE EDGE {_lit(source)} -> {_lit(target)} kind = {_lit(kind)}"
        )

    def explore(self, node_id: str, *, depth: int = 2, limit: int = 50) -> list[dict]:
        return _rows(self._gs.execute(
            f"RECALL FROM {_lit(node_id)} DEPTH {int(depth)} LIMIT {int(limit)}"
        ))

    def forget(self, node_id: str) -> None:
        self._gs.execute(f"RETRACT {_lit(node_id)}")

    def execute(self, dsl: str) -> Any:
        return self._gs.execute(dsl)

    def close(self) -> None:
        self._gs.close()


__all__ = ["Research"]
