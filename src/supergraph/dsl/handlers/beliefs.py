"""Belief handlers: ASSERT, RETRACT, PROPAGATE."""

import time
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix

from supergraph.core.edges import resize_csr

from supergraph.dsl.handlers._registry import handles
from supergraph.dsl.ast_nodes import AssertStmt, RetractStmt, PropagateStmt
from supergraph.core.types import Result
from supergraph.core.errors import NodeNotFound


class BeliefHandlers:

    @handles(AssertStmt, write=True)
    def _assert(self, q: AssertStmt) -> Result:
        """ASSERT: upsert with reserved __confidence__, __source__, __retracted__=0."""
        data = {fp.name: fp.value for fp in q.fields}
        kind = data.pop("kind", "default")
        self.schema.validate_node(kind, data)
        self.store.upsert_node(q.id, kind, data)
        str_id = self.store.string_table.intern(q.id)
        slot = self.store.id_to_slot[str_id]
        if q.confidence is not None:
            self.store.columns.set_reserved(slot, "__confidence__", q.confidence)
        if q.source is not None:
            self.store.columns.set_reserved(slot, "__source__", q.source)
        self.store.columns.set_reserved(slot, "__retracted__", 0)
        event_at = getattr(q, 'event_at', None)
        if event_at is not None:
            from supergraph.dsl.handlers.mutations import MutationHandlers
            ms = MutationHandlers._parse_event_at(event_at)
            if ms is not None:
                self.store.columns.set_reserved(slot, "__event_at__", ms)
        node = self.store.get_node(q.id)
        return Result(kind="node", data=node, count=1)

    @handles(RetractStmt, write=True)
    def _retract(self, q: RetractStmt) -> Result:
        """RETRACT: mark node as retracted (invisible via live_mask).

        If the retracted node is kind='document', also retract all outgoing
        chunk and image nodes.
        """
        if q.id not in self.store.string_table:
            raise NodeNotFound(q.id)
        str_id = self.store.string_table.intern(q.id)
        slot = self.store.id_to_slot.get(str_id)
        if slot is None or slot in self.store.node_tombstones:
            raise NodeNotFound(q.id)
        now_ms = int(time.time() * 1000)
        self.store.columns.set_reserved(slot, "__retracted__", 1)
        self.store.columns.set_reserved(slot, "__retracted_at__", now_ms)
        if q.reason:
            self.store.columns.set_reserved(slot, "__retract_reason__", q.reason)
        if self._vector_store:
            self._vector_store.remove(slot)

        kind_str_id = int(self.store.node_kinds[slot])
        kind_name = self.store.string_table.lookup(kind_str_id)
        if kind_name == "document":
            for child_id, child_slot in self._collect_doc_children(q.id):
                if child_slot not in self.store.node_tombstones:
                    self.store.columns.set_reserved(child_slot, "__retracted__", 1)
                    self.store.columns.set_reserved(child_slot, "__retracted_at__", now_ms)
                    if q.reason:
                        self.store.columns.set_reserved(child_slot, "__retract_reason__", q.reason)

        return Result(kind="ok", data={"id": q.id}, count=1)

    @handles(PropagateStmt, write=True)
    def _propagate(self, q: PropagateStmt) -> Result:
        """PROPAGATE: BFS forward belief chaining via algos.graph.propagate_values."""
        from supergraph.algos.graph import propagate_values

        src_slot = self._resolve_slot(q.node_id)
        if src_slot is None:
            raise NodeNotFound(q.node_id)

        n = self.store._next_slot
        field = q.field

        if not self.store.columns.has_column(field):
            return Result(kind="ok", data={"updated": 0}, count=0)
        if not self.store.columns._presence[field][src_slot]:
            return Result(kind="ok", data={"updated": 0}, count=0)

        combined = self.store.edge_matrices.get(None)
        if combined is None:
            return Result(kind="ok", data={"updated": 0}, count=0)

        mat = combined
        if mat.shape[0] < n:
            mat = resize_csr(mat, n)

        dtype = self.store.columns._dtypes[field]
        raw = self.store.columns._columns[field][src_slot]
        if dtype == "float64":
            source_value = float(raw)
        elif dtype == "int64":
            source_value = float(int(raw))
        else:
            return Result(kind="ok", data={"updated": 0}, count=0)

        updated_slots, new_values = propagate_values(
            matrix=mat,
            source=src_slot,
            source_value=source_value,
            depth=q.depth,
            existing_values=self.store.columns._columns[field],
            presence=self.store.columns._presence[field],
            blocked_slots=self.store.node_tombstones,
        )

        now_ms = int(time.time() * 1000)
        for nb, propagated in zip(updated_slots, new_values):
            self.store.columns.set(nb, {field: propagated})
            self.store.columns.set_reserved(nb, "__updated_at__", now_ms)

        return Result(
            kind="ok", data={"updated": len(updated_slots)}, count=len(updated_slots)
        )
