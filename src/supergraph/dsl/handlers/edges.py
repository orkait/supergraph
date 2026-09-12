"""Edge query + CRUD handlers."""

from supergraph.dsl.handlers._registry import handles
from supergraph.dsl.ast_nodes import (
    EdgesQuery, CreateEdge, UpdateEdge, DeleteEdge, DeleteEdges,
)
from supergraph.core.types import Result


class EdgeHandlers:

    @handles(EdgesQuery)
    def _edges(self, q: EdgesQuery) -> Result:
        kind = self._extract_kind_from_where(q.where)
        if q.direction == "FROM":
            edges = self.store.get_edges_from(q.node_id, kind=kind)
        else:
            edges = self.store.get_edges_to(q.node_id, kind=kind)
        if q.where and not self._is_simple_kind_filter(q.where):
            edges = [e for e in edges if self._eval_where(q.where.expr, e)]
        edges = [
            e for e in edges
            if self._is_visible_by_id(e["source"]) and self._is_visible_by_id(e["target"])
        ]
        if q.limit:
            edges = edges[:q.limit.value]
        return Result(kind="edges", data=edges, count=len(edges))

    @handles(CreateEdge, write=True)
    def _create_edge(self, q: CreateEdge) -> Result:
        data = {fp.name: fp.value for fp in q.fields}
        kind = data.pop("kind", "default")
        src_node = self.store.get_node(q.source)
        tgt_node = self.store.get_node(q.target)
        if src_node and tgt_node:
            self.schema.validate_edge(kind, src_node["kind"], tgt_node["kind"])
        self.store.put_edge(q.source, q.target, kind, data if data else None)
        return Result(kind="edges", data=[{"source": q.source, "target": q.target, "kind": kind}], count=1)

    @handles(UpdateEdge, write=True)
    def _update_edge(self, q: UpdateEdge) -> Result:
        # Look up source/target string IDs without interning. Pre-fix,
        # interning happened inside the edge-type loop on every iteration
        # AND happened before checking whether the endpoints exist —
        # calling ``UPDATE EDGE`` against two nonexistent nodes leaked
        # two string entries per call (bug #41). Short-circuit when either
        # endpoint is unknown.
        if q.source not in self.store.string_table:
            return Result(kind="ok", data={"source": q.source, "target": q.target, "updated": 0}, count=0)
        if q.target not in self.store.string_table:
            return Result(kind="ok", data={"source": q.source, "target": q.target, "updated": 0}, count=0)
        src_str_id = self.store.string_table.intern(q.source)
        tgt_str_id = self.store.string_table.intern(q.target)
        src_slot = self.store.id_to_slot.get(src_str_id)
        tgt_slot = self.store.id_to_slot.get(tgt_str_id)
        if src_slot is None or tgt_slot is None:
            return Result(kind="ok", data={"source": q.source, "target": q.target, "updated": 0}, count=0)

        kind = self._extract_kind_from_where(q.where)
        update_data = {fp.name: fp.value for fp in q.fields}
        updated = 0
        for etype in ([kind] if kind else list(self.store._edges_by_type.keys())):
            if etype not in self.store._edges_by_type:
                continue
            for i, (s, t, d) in enumerate(self.store._edges_by_type[etype]):
                if s == src_slot and t == tgt_slot:
                    new_data = {**d, **update_data}
                    self.store._edges_by_type[etype][i] = (s, t, new_data)
                    self.store._edge_data_idx.setdefault(etype, {})[(src_slot, tgt_slot)] = new_data
                    updated += 1
        if updated > 0:
            self.store._edges_dirty = True
            self.store._ensure_edges_built()
        return Result(kind="ok", data={"source": q.source, "target": q.target, "updated": updated}, count=updated)

    @handles(DeleteEdge, write=True)
    def _delete_edge(self, q: DeleteEdge) -> Result:
        kind = self._extract_kind_from_where(q.where)
        if kind:
            self.store.delete_edge(q.source, q.target, kind)
        else:
            # Drop matching edges across every known kind in one CSR
            # rebuild. Pre-fix this loop called delete_edge() per type,
            # each triggering a full edge rebuild — O(E * T) where T is
            # the number of edge types (bug #42).
            pairs = [(q.source, q.target, etype) for etype in list(self.store._edges_by_type.keys())]
            self.store.delete_edges_bulk(pairs)
        return Result(kind="ok", data=None, count=1)

    @handles(DeleteEdges, write=True)
    def _delete_edges(self, q: DeleteEdges) -> Result:
        kind = self._extract_kind_from_where(q.where)
        if q.direction == "FROM":
            edges = self.store.get_edges_from(q.node_id, kind=kind)
        else:
            edges = self.store.get_edges_to(q.node_id, kind=kind)
        for e in edges:
            self.store.delete_edge(e["source"], e["target"], e["kind"])
        return Result(kind="edges", data=edges, count=len(edges))
