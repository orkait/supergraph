"""Core in-memory graph engine.

Manages nodes (numpy arrays + ColumnStore) and edges (EdgeMatrices with CSR)
with secondary indices, tombstone-based deletion, and memory ceiling
enforcement.
"""

from __future__ import annotations

import time
import numpy as np

from supergraph.core.edges import EdgeMatrices
from supergraph.core.errors import SuperGraphError, NodeExists, NodeNotFound
from supergraph.core.memory import (
    BYTES_PER_NODE_ESTIMATE,
    BYTES_PER_EDGE_ESTIMATE,
    DEFAULT_CEILING_BYTES,
    check_ceiling,
    measure,
)
from supergraph.core.columns import ColumnStore
from supergraph.core.strings import StringTable


class CoreStore:
    """In-memory graph store backed by numpy arrays and sparse matrices."""

    def __init__(self, ceiling_bytes: int = DEFAULT_CEILING_BYTES, capacity: int = 1024, use_compression: bool = False):
        self.string_table = StringTable()
        self._edge_matrices = EdgeMatrices(use_compression=use_compression)
        self._ceiling_bytes = ceiling_bytes

        # Node storage - pre-allocate
        self._capacity = capacity
        self._count = 0  # number of live nodes (not counting tombstones)
        self._next_slot = 0  # next slot to fill
        self.node_ids = np.full(self._capacity, -1, dtype=np.int32)
        self.node_kinds = np.zeros(self._capacity, dtype=np.int32)
        self.node_tombstones: set[int] = set()
        self.id_to_slot: dict[int, int] = {}

        # Edge storage
        self._edges_by_type: dict[str, list[tuple]] = {}
        self._edge_keys: set[tuple[int, int, str]] = set()  # (src_slot, tgt_slot, kind) for O(1) duplicate check
        self._edge_data_idx: dict[str, dict[tuple[int, int], dict]] = {}  # edge_type -> {(src, tgt): data}
        self._edges_dirty = False  # deferred CSR rebuild flag

        # Secondary indices
        self.secondary_indices: dict[str, dict] = {}
        self._indexed_fields: set[str] = set()

        # Columnar acceleration layer
        self.columns = ColumnStore(self.string_table, self._capacity)

        # Active context (for BIND/DISCARD CONTEXT)
        self._active_context: str | None = None

        # Active namespace (for BIND/DISCARD NAMESPACE). Unlike context,
        # namespaced nodes are excluded from the default (unbound) view, giving
        # an isolated partition that does not pollute the general memory.
        self._active_namespace: str | None = None

        # Named snapshots storage (for SYS SNAPSHOT/ROLLBACK)
        self._snapshots: dict[str, dict] = {}
        self._tombstone_mask_cache: tuple[int, int, np.ndarray] | None = None  # (n, len(tombstones), mask)

        # Dirty tracking for incremental checkpoint
        self._dirty_nodes = True
        self._dirty_edges = True
        self._dirty_strings = True
        # _dirty_columns is a property delegating to self.columns.dirty

        # Live mask / live slots version cache - invalidated on every mutation
        self._live_version: int = 0
        self._live_mask_cache: dict[str | None, tuple[int, np.ndarray]] = {}
        self._live_slots_cache: dict[str | None, tuple[int, np.ndarray]] = {}

        # Self-calibrating ceiling estimate - updated every 1000 puts via
        # _recalibrate_ceiling_estimate().  Starts at the static default and
        # converges toward the real per-node footprint for this schema.
        self._bytes_per_node_estimate: float = float(BYTES_PER_NODE_ESTIMATE)
        self._bytes_per_edge_estimate: float = float(BYTES_PER_EDGE_ESTIMATE)
        self._calibrate_at_count: int = 1000  # next node count that triggers recalibration

    @property
    def _dirty_columns(self) -> bool:
        return self.columns.dirty

    @_dirty_columns.setter
    def _dirty_columns(self, value: bool) -> None:
        self.columns.dirty = bool(value)

    # -- slot management -----------------------------------------------------

    def _alloc_slot(self) -> int:
        """Get next available slot, reusing tombstones or expanding."""
        if self.node_tombstones:
            slot = self.node_tombstones.pop()
            self._tombstone_mask_cache = None
            return slot
        if self._next_slot >= self._capacity:
            self._grow()
        slot = self._next_slot
        self._next_slot += 1
        return slot

    def _invalidate_live_cache(self) -> None:
        """Increment write version so live mask/slots caches are rebuilt on next read."""
        self._live_version += 1

    def _recalibrate_ceiling_estimate(self) -> None:
        """Update bytes-per-node estimate from a live measurement of this store.

        Called every 1000 put_node operations.  Uses skip_csr=True to avoid
        triggering an expensive CSR rebuild mid-write.  Vector store memory is
        NOT included here because CoreStore has no reference to it - the
        estimate therefore covers the core footprint only (arrays, columns,
        string table, edge lists).  For schemas with no embedder this is
        essentially exact; for schemas with embeddings the ceiling will be
        somewhat conservative, which is acceptable (we prefer false-positives
        over OOM).
        """
        if self._count < 100:
            return
        report = measure(self, vector_store=None, skip_csr=True)
        core_bytes = report["core_total"]
        self._bytes_per_node_estimate = core_bytes / self._count
        raw_edge_count = sum(len(v) for v in self._edges_by_type.values())
        if raw_edge_count > 0:
            edge_bytes = report["edge_lists"] + report["edge_data_idx"]
            self._bytes_per_edge_estimate = edge_bytes / raw_edge_count
        # Schedule next calibration: double the interval up to 10k (amortised O(1))
        self._calibrate_at_count = self._count + min(self._count, 10_000)

    def _grow(self):
        """Double array capacity."""
        new_cap = self._capacity * 2

        new_ids = np.full(new_cap, -1, dtype=np.int32)
        new_ids[: self._capacity] = self.node_ids
        self.node_ids = new_ids

        new_kinds = np.zeros(new_cap, dtype=np.int32)
        new_kinds[: self._capacity] = self.node_kinds
        self.node_kinds = new_kinds

        self.columns.grow(new_cap)
        self._capacity = new_cap

    # -- properties ----------------------------------------------------------

    @property
    def node_count(self) -> int:
        return self._count

    @property
    def edge_matrices(self) -> EdgeMatrices:
        """Auto-rebuilds CSR matrices if dirty."""
        self._ensure_edges_built()
        return self._edge_matrices

    @property
    def edge_count(self) -> int:
        # Use raw count to avoid CSR rebuild for simple counting
        return sum(len(v) for v in self._edges_by_type.values())

    # -- node CRUD -----------------------------------------------------------

    def put_node(self, id: str, kind: str, data: dict) -> int:
        """Add a node. Returns slot index. Raises NodeExists if ID exists."""
        # Fast path: if the id is already known, check for existing live
        # slot BEFORE interning the kind. intern() is monotonic — once we
        # mint a new ID there's no rolling back — so we defer interning of
        # the ``kind`` string until after the ceiling check cannot fire.
        # Pre-fix, a failed put_node (ceiling exceeded, allocation failure)
        # left the kind string in the table forever, wasting an int32 slot
        # per failure and inflating gc_strings pressure (bug #2).
        if id in self.string_table:
            str_id = self.string_table.intern(id)  # idempotent for existing ids
            if str_id in self.id_to_slot:
                slot = self.id_to_slot[str_id]
                if slot not in self.node_tombstones:
                    raise NodeExists(id)

        raw_edge_count = sum(len(v) for v in self._edges_by_type.values())
        check_ceiling(
            self._count, raw_edge_count, 1, 0, self._ceiling_bytes,
            bytes_per_node=int(self._bytes_per_node_estimate),
            bytes_per_edge=int(self._bytes_per_edge_estimate),
        )

        # Ceiling passed — safe to intern both id and kind. Order matters:
        # ``id`` might already be interned (lookup path above) or new.
        str_id = self.string_table.intern(id)
        kind_id = self.string_table.intern(kind)
        slot = self._alloc_slot()

        self.node_ids[slot] = str_id
        self.node_kinds[slot] = kind_id
        self.columns.set(slot, data)
        self.id_to_slot[str_id] = slot
        self._count += 1
        self._dirty_nodes = True
        self._dirty_columns = True
        self._dirty_strings = True
        self._invalidate_live_cache()
        now_ms = int(time.time() * 1000)
        self.columns.set_reserved(slot, "__created_at__", now_ms)
        self.columns.set_reserved(slot, "__updated_at__", now_ms)
        # Tag the active namespace so the node is isolated to it. Single write
        # chokepoint: covers CREATE / UPSERT(new) / ASSERT uniformly.
        if self._active_namespace:
            self.columns.set_reserved(slot, "__namespace__", self._active_namespace)

        # Update secondary indices
        for field in self._indexed_fields:
            if field in data:
                val = data[field]
                self.secondary_indices[field].setdefault(val, []).append(slot)

        # Recalibrate bytes-per-node estimate periodically
        if self._count >= self._calibrate_at_count:
            self._recalibrate_ceiling_estimate()

        return slot

    def get_node(self, id: str) -> dict | None:
        """Get node data by ID. Returns None if not found."""
        if id not in self.string_table:
            return None
        str_id = self.string_table.intern(id)
        slot = self.id_to_slot.get(str_id)
        if slot is None or slot in self.node_tombstones:
            return None
        return self._materialize_slot(slot)

    def update_node(self, id: str, data: dict):
        """Update node data. Raises NodeNotFound if missing."""
        if id not in self.string_table:
            raise NodeNotFound(id)
        str_id = self.string_table.intern(id)
        slot = self.id_to_slot.get(str_id)
        if slot is None or slot in self.node_tombstones:
            raise NodeNotFound(id)

        # Remove old values from secondary indices
        for field in self._indexed_fields:
            if self.columns.has_column(field) and self.columns._presence[field][slot]:
                dtype = self.columns._dtypes[field]
                raw = self.columns._columns[field][slot]
                if dtype == "int32_interned":
                    old_val = self.string_table.lookup(int(raw))
                elif dtype == "float64":
                    old_val = float(raw)
                else:
                    old_val = int(raw)
                idx_list = self.secondary_indices[field].get(old_val, [])
                if slot in idx_list:
                    idx_list.remove(slot)

        # Update columns
        self.columns.set(slot, data)

        # Add new values to secondary indices
        for field in self._indexed_fields:
            if field in data:
                self.secondary_indices[field].setdefault(data[field], []).append(slot)
        self.columns.set_reserved(slot, "__updated_at__", int(time.time() * 1000))
        self._dirty_columns = True
        self._dirty_strings = True

    def upsert_node(self, id: str, kind: str, data: dict) -> int:
        """Create or update node."""
        if id in self.string_table:
            str_id = self.string_table.intern(id)
            slot = self.id_to_slot.get(str_id)
            if slot is not None and slot not in self.node_tombstones:
                self.update_node(id, data)
                return slot
        return self.put_node(id, kind, data)

    def delete_nodes_bulk(
        self,
        ids: list[str],
        vector_store=None,
        document_store=None,
    ) -> list[str]:
        """Bulk tombstone by ID with a single edge cascade rebuild.

        Much faster than looping delete_node() for DELETE NODES WHERE:
        avoids O(N*E) cascade cost by running one filter pass at the end.
        Unknown or already-tombstoned IDs are skipped silently.
        Returns the list of IDs actually removed.
        """
        if not ids:
            return []

        # Filter + intern in one pass. Each iteration is O(1) against the
        # string table and dict, so the loop body is already cheap; the
        # win here is just list-comp construction over repeated .append.
        intern = self.string_table.intern
        id_to_slot = self.id_to_slot
        tombstones = self.node_tombstones
        triples = [
            (id_, str_id, slot)
            for id_ in ids
            if id_ in self.string_table
            for str_id in (intern(id_),)
            for slot in (id_to_slot.get(str_id),)
            if slot is not None and slot not in tombstones
        ]
        if not triples:
            return []
        deleted_ids = [t[0] for t in triples]
        str_ids_to_remove = [t[1] for t in triples]
        slots_to_remove = [t[2] for t in triples]

        if not slots_to_remove:
            return []

        slot_set = set(slots_to_remove)

        if self._indexed_fields:
            # Secondary-index cleanup. Per-bucket list filter is O(|bucket|)
            # but we touch each bucket at most once per field, not once per
            # slot. Previously: O(slots * bucket_size * fields) - `slot in
            # list` + `list.remove` on each removed slot.
            lookup = self.string_table.lookup
            slot_idx = np.asarray(slots_to_remove, dtype=np.int64)
            for field in self._indexed_fields:
                if not self.columns.has_column(field):
                    continue
                bucket = self.secondary_indices.get(field)
                if bucket is None:
                    continue
                pres_col = self.columns._presence[field]
                data_col = self.columns._columns[field]
                dtype = self.columns._dtypes[field]
                live = pres_col[slot_idx]
                if not live.any():
                    continue
                raws = data_col[slot_idx[live]]
                if dtype == "int32_interned":
                    vals = [lookup(int(r)) for r in raws]
                elif dtype == "float64":
                    vals = [float(r) for r in raws]
                else:
                    vals = [int(r) for r in raws]
                # Group slots by val so we filter each bucket list once.
                by_val: dict = {}
                for s, v in zip(slot_idx[live].tolist(), vals):
                    by_val.setdefault(v, set()).add(int(s))
                for v, dead in by_val.items():
                    lst = bucket.get(v)
                    if lst is None:
                        continue
                    bucket[v] = [s for s in lst if s not in dead]

        import logging as _logging
        _rm_log = _logging.getLogger(__name__)
        # Batch-delete documents + FTS in one transaction (N queries -> 2).
        if document_store is not None and slots_to_remove:
            try:
                document_store.delete_documents_batch(slots_to_remove)
            except Exception as _ds_err:
                _rm_log.debug("doc batch delete failed: %s", _ds_err)
        # Vector store has no batch remove (usearch Index.remove is per-key).
        if vector_store is not None:
            for slot in slots_to_remove:
                try:
                    vector_store.remove(slot)
                except Exception as _vs_err:
                    _rm_log.debug("vector remove failed slot=%s: %s", slot, _vs_err)
        self.columns.clear_slots(slots_to_remove)

        self.node_tombstones.update(slot_set)
        self._tombstone_mask_cache = None
        for str_id in str_ids_to_remove:
            self.id_to_slot.pop(str_id, None)
        self._count -= len(slots_to_remove)
        self._dirty_nodes = True
        self._dirty_columns = True
        self._dirty_edges = True
        self._invalidate_live_cache()

        from supergraph.algos.edges_ops import (
            cascade_filter_edges,
            rebuild_edge_keys_set,
            rebuild_edge_data_idx,
        )
        new_edges, any_removed = cascade_filter_edges(self._edges_by_type, slot_set)
        self._edges_by_type = new_edges
        if any_removed:
            self._edge_keys = rebuild_edge_keys_set(self._edges_by_type)
            self._edge_data_idx = rebuild_edge_data_idx(self._edges_by_type)

        self._edges_dirty = True
        self._ensure_edges_built()
        return deleted_ids

    def delete_node(self, id: str):
        """Delete node and cascade-delete all edges. Raises NodeNotFound."""
        if id not in self.string_table:
            raise NodeNotFound(id)
        str_id = self.string_table.intern(id)
        slot = self.id_to_slot.get(str_id)
        if slot is None or slot in self.node_tombstones:
            raise NodeNotFound(id)

        # Remove from indices
        for field in self._indexed_fields:
            if self.columns.has_column(field) and self.columns._presence[field][slot]:
                dtype = self.columns._dtypes[field]
                raw = self.columns._columns[field][slot]
                if dtype == "int32_interned":
                    val = self.string_table.lookup(int(raw))
                elif dtype == "float64":
                    val = float(raw)
                else:
                    val = int(raw)
                idx_list = self.secondary_indices[field].get(val, [])
                if slot in idx_list:
                    idx_list.remove(slot)

        # Tombstone
        self.columns.clear(slot)
        self.node_tombstones.add(slot)
        self._tombstone_mask_cache = None
        del self.id_to_slot[str_id]
        self._count -= 1
        self._dirty_nodes = True
        self._dirty_columns = True
        self._dirty_edges = True
        self._invalidate_live_cache()

        # Cascade-delete edges touching this node
        self._cascade_delete_edges(slot)

    # -- edge CRUD -----------------------------------------------------------

    def put_edge(
        self, source_id: str, target_id: str, kind: str, data: dict | None = None
    ):
        """Add an edge. Both nodes must exist.

        Semantics (simple directed graph with edge data):

          - New (src, tgt, kind) triple: inserted.
          - Existing triple with ``data`` equal to the stored value: raises
            ``SuperGraphError("Duplicate edge: ...")`` — caller is asking
            the store to do something it has already done.
          - Existing triple with ``data`` that differs from the stored
            value: **silently ignored**; the first write wins. Use
            ``UpdateEdge`` / ``UPDATE EDGE`` DSL to modify edge fields on
            an existing edge.

        This last case is the fix for bug #13. Pre-fix, a second CREATE
        EDGE with different data appended a duplicate row in
        ``_edges_by_type``, clobbered ``_edge_data_idx`` with the newer
        data, and let the dynamic edge buffer accumulate a second CSR
        entry. That cascaded into scipy CSR summing duplicate weights
        (#18), inflated COUNT EDGES (#19) and SYS STATS (#26), and
        polluted traversal-matrix weight semantics. One place, six bugs.
        """
        if source_id not in self.string_table:
            raise NodeNotFound(source_id)
        if target_id not in self.string_table:
            raise NodeNotFound(target_id)

        src_str_id = self.string_table.intern(source_id)
        tgt_str_id = self.string_table.intern(target_id)

        src_slot = self.id_to_slot.get(src_str_id)
        tgt_slot = self.id_to_slot.get(tgt_str_id)

        if src_slot is None or src_slot in self.node_tombstones:
            raise NodeNotFound(source_id)
        if tgt_slot is None or tgt_slot in self.node_tombstones:
            raise NodeNotFound(target_id)

        # Duplicate detection — O(1) set lookup. Before doing any ceiling
        # check or mutation, see if the (src, tgt, kind) triple is already
        # known. The pre-existing behavior was to raise on exact-match and
        # silently insert a duplicate on data-differs; the new behavior is
        # to raise on exact-match and return early on data-differs.
        edge_key = (src_slot, tgt_slot, kind)
        edge_data = data or {}
        if edge_key in self._edge_keys:
            existing = self._edge_data_idx.get(kind, {}).get((src_slot, tgt_slot))
            if existing == edge_data:
                raise SuperGraphError(
                    f"Duplicate edge: {source_id} -> {target_id} kind={kind}"
                )
            # Same endpoints, different data. First-write-wins; callers that
            # need to mutate fields use UPDATE EDGE.
            return

        # New edge — proceed with insertion. Ceiling check happens here so
        # the failure mode for "we hit the edge-count limit" doesn't pay the
        # lookup cost on the duplicate hot path above.
        raw_edge_count = sum(len(v) for v in self._edges_by_type.values())
        check_ceiling(
            self._count, raw_edge_count, 0, 1, self._ceiling_bytes,
            bytes_per_node=int(self._bytes_per_node_estimate),
            bytes_per_edge=int(self._bytes_per_edge_estimate),
        )

        self._edge_keys.add(edge_key)
        self._edges_by_type.setdefault(kind, []).append(
            (src_slot, tgt_slot, edge_data)
        )
        self._edge_data_idx.setdefault(kind, {})[(src_slot, tgt_slot)] = edge_data

        # Add to dynamic edge buffer (L0) instead of forcing full CSR rebuild.
        # Only reached for genuinely-new edges (duplicate branch above returns
        # early), so the buffer can't accumulate parallel entries for the
        # same pair — fixes bug #15 as a consequence of fixing #13.
        weight = float(edge_data.get("weight", 1.0))
        self._edge_matrices.add_dynamic(src_slot, tgt_slot, kind, self._next_slot, weight)
        if self._edge_matrices._pending_edge_count > 10000:
            self._edges_dirty = True

        self._dirty_edges = True
        self._dirty_strings = True
        self._invalidate_live_cache()

    def delete_edge(self, source_id: str, target_id: str, kind: str):
        """Delete a specific edge. Triggers CSR rebuild — prefer ``delete_edges_bulk``
        when dropping multiple edges of the same source/target pair (e.g.,
        VAULT wikilink resync) to avoid N rebuilds."""
        # Avoid interning unknown ids — both saves string_table entries on
        # missing edges and short-circuits the work when either endpoint
        # does not exist.
        if source_id not in self.string_table or target_id not in self.string_table:
            return
        src_str_id = self.string_table.intern(source_id)
        tgt_str_id = self.string_table.intern(target_id)
        src_slot = self.id_to_slot.get(src_str_id)
        tgt_slot = self.id_to_slot.get(tgt_str_id)
        if src_slot is None or tgt_slot is None:
            return

        if kind in self._edges_by_type:
            self._edges_by_type[kind] = [
                (s, t, d)
                for s, t, d in self._edges_by_type[kind]
                if not (s == src_slot and t == tgt_slot)
            ]
            if not self._edges_by_type[kind]:
                del self._edges_by_type[kind]
        if kind in self._edge_data_idx:
            self._edge_data_idx[kind].pop((src_slot, tgt_slot), None)
            if not self._edge_data_idx[kind]:
                del self._edge_data_idx[kind]
        self._edge_keys.discard((src_slot, tgt_slot, kind))
        self._rebuild_edges()
        self._dirty_edges = True

    def delete_edges_bulk(self, edges: list[tuple[str, str, str]]) -> int:
        """Delete multiple edges, rebuilding CSR exactly once at the end.

        Fixes the N-rebuild cost of iterating ``delete_edge`` in callers
        that need to drop many edges at once (bugs #42, #57). Examples:

        - ``DELETE EDGE "a" -> "b"`` with no kind: the DSL handler
          previously called delete_edge once per edge-type, triggering
          O(E) rebuilds for a single logical operation.
        - Vault sync resyncing a note's outgoing wikilinks: one rebuild
          per link instead of one rebuild for the whole resync.

        Returns the number of edges actually removed. Missing edges are
        skipped silently (they're a valid input to an idempotent sync
        primitive).
        """
        if not edges:
            return 0

        # Resolve every endpoint up front so we can reject bad input
        # without mutating state halfway through.
        resolved: list[tuple[int, int, str]] = []
        for source_id, target_id, kind in edges:
            if source_id not in self.string_table or target_id not in self.string_table:
                continue
            src_slot = self.id_to_slot.get(self.string_table.intern(source_id))
            tgt_slot = self.id_to_slot.get(self.string_table.intern(target_id))
            if src_slot is None or tgt_slot is None:
                continue
            resolved.append((src_slot, tgt_slot, kind))

        if not resolved:
            return 0

        # Group by kind so we filter each edge list exactly once.
        by_kind: dict[str, set[tuple[int, int]]] = {}
        for src_slot, tgt_slot, kind in resolved:
            by_kind.setdefault(kind, set()).add((src_slot, tgt_slot))

        removed = 0
        for kind, pairs in by_kind.items():
            if kind not in self._edges_by_type:
                continue
            before = len(self._edges_by_type[kind])
            self._edges_by_type[kind] = [
                (s, t, d)
                for s, t, d in self._edges_by_type[kind]
                if (s, t) not in pairs
            ]
            after = len(self._edges_by_type[kind])
            removed += before - after
            if not self._edges_by_type[kind]:
                del self._edges_by_type[kind]
            if kind in self._edge_data_idx:
                for pair in pairs:
                    self._edge_data_idx[kind].pop(pair, None)
                if not self._edge_data_idx[kind]:
                    del self._edge_data_idx[kind]
            for pair in pairs:
                self._edge_keys.discard((pair[0], pair[1], kind))

        if removed > 0:
            self._rebuild_edges()
            self._dirty_edges = True
        return removed

    def get_edges_from(self, id: str, kind: str | None = None) -> list[dict]:
        """Get outgoing edges from a node. Uses CSR for neighbor lookup."""
        self._ensure_edges_built()
        if id not in self.string_table:
            return []
        str_id = self.string_table.intern(id)
        slot = self.id_to_slot.get(str_id)
        if slot is None or slot in self.node_tombstones:
            return []

        result = []
        types_to_check = [kind] if kind else self._edge_matrices.edge_types
        for etype in types_to_check:
            neighbors = self._edge_matrices.neighbors_out(slot, etype)
            if len(neighbors) == 0:
                continue
            idx = self._edge_data_idx.get(etype, {})
            for nb in neighbors:
                nb = int(nb)
                tgt_id = self._slot_to_id(nb)
                if tgt_id is not None:
                    d = idx.get((slot, nb), {})
                    result.append({"source": id, "target": tgt_id, "kind": etype, **d})
        return result

    def get_edges_to(self, id: str, kind: str | None = None) -> list[dict]:
        """Get incoming edges to a node. Uses transpose CSR to skip irrelevant edge types."""
        self._ensure_edges_built()
        if id not in self.string_table:
            return []
        str_id = self.string_table.intern(id)
        slot = self.id_to_slot.get(str_id)
        if slot is None or slot in self.node_tombstones:
            return []

        result = []
        types_to_check = [kind] if kind else self._edge_matrices.edge_types
        for etype in types_to_check:
            incoming = self._edge_matrices.neighbors_in(slot, etype)
            if len(incoming) == 0:
                continue
            idx = self._edge_data_idx.get(etype, {})
            for nb in incoming:
                nb = int(nb)
                src_id = self._slot_to_id(nb)
                if src_id is not None:
                    d = idx.get((nb, slot), {})
                    result.append({"source": src_id, "target": id, "kind": etype, **d})
        return result

    # -- cascade / rebuild ---------------------------------------------------

    def _cascade_delete_edges(self, slot: int):
        """Remove all edges involving this slot from pending edges."""
        any_removed = False
        for etype in list(self._edges_by_type.keys()):
            old_len = len(self._edges_by_type[etype])
            self._edges_by_type[etype] = [
                (s, t, d)
                for s, t, d in self._edges_by_type[etype]
                if s != slot and t != slot
            ]
            if not self._edges_by_type[etype]:
                del self._edges_by_type[etype]
            if len(self._edges_by_type.get(etype, [])) != old_len:
                any_removed = True
        if any_removed:
            self._edge_keys = {
                (s, t, k)
                for k, edges in self._edges_by_type.items()
                for s, t, _d in edges
            }
            self._edge_data_idx = {
                k: {(s, t): d for s, t, d in edges}
                for k, edges in self._edges_by_type.items()
            }
        self._edges_dirty = True
        self._ensure_edges_built()

    def _ensure_edges_built(self):
        """Lazily rebuild CSR matrices only when dirty."""
        if not self._edges_dirty:
            return
        self._edges_dirty = False
        num_nodes = max(self._next_slot, 1)
        self._edge_matrices.rebuild(self._edges_by_type, num_nodes)

    def _rebuild_edges(self):
        """Force rebuild EdgeMatrices from pending edge lists."""
        self._edges_dirty = False
        num_nodes = max(self._next_slot, 1)
        self._edge_matrices.rebuild(self._edges_by_type, num_nodes)
        self._edge_data_idx = {
            k: {(s, t): d for s, t, d in edges}
            for k, edges in self._edges_by_type.items()
        }

    # -- helpers -------------------------------------------------------------

    def _slot_to_id(self, slot: int) -> str | None:
        """Convert slot index back to string ID."""
        if slot >= self._next_slot or slot in self.node_tombstones:
            return None
        str_id = int(self.node_ids[slot])
        if str_id == -1:
            return None
        return self.string_table.lookup(str_id)

    # -- secondary indices ---------------------------------------------------

    def add_index(self, field: str):
        """Build secondary index on a field."""
        self._indexed_fields.add(field)
        index: dict = {}
        if not self.columns.has_column(field):
            self.secondary_indices[field] = index
            return
        for slot in range(self._next_slot):
            if slot not in self.node_tombstones and self.columns._presence[field][slot]:
                dtype = self.columns._dtypes[field]
                raw = self.columns._columns[field][slot]
                if dtype == "int32_interned":
                    val = self.string_table.lookup(int(raw))
                elif dtype == "float64":
                    val = float(raw)
                else:
                    val = int(raw)
                index.setdefault(val, []).append(slot)
        self.secondary_indices[field] = index

    def reindex_slots(self, field: str, slots) -> None:
        """Update the secondary index for ``field`` for a specific set of slots.

        Used by bulk UpdateNodes so a field change to a small subset of
        rows doesn't trigger a full-table re-scan of the secondary index
        (bug #32). Caller guarantees every slot in ``slots`` is live.

        The implementation removes the slots from whatever value buckets
        they currently appear in (we don't know the old value), then
        re-inserts them under their current column value.
        """
        if field not in self._indexed_fields:
            return
        if not self.columns.has_column(field):
            return
        idx = self.secondary_indices.setdefault(field, {})
        slot_set = {int(s) for s in slots}

        # Scan-and-drop: secondary indices store ``dict[value, list[slot]]``
        # with no reverse map, so removing a slot requires walking each
        # bucket. The cost is O(|indexed_values|) not O(N rows) — still
        # cheaper than the full rebuild when |affected_slots| << N.
        empty_vals: list = []
        for val, slot_list in idx.items():
            new_list = [s for s in slot_list if s not in slot_set]
            if len(new_list) == len(slot_list):
                continue
            if new_list:
                idx[val] = new_list
            else:
                empty_vals.append(val)
        for v in empty_vals:
            del idx[v]

        # Re-insert under current values
        dtype = self.columns._dtypes[field]
        for slot in slot_set:
            if not self.columns._presence[field][slot]:
                continue
            raw = self.columns._columns[field][slot]
            if dtype == "int32_interned":
                val = self.string_table.lookup(int(raw))
            elif dtype == "float64":
                val = float(raw)
            else:
                val = int(raw)
            idx.setdefault(val, []).append(slot)

    def query_by_index(self, field: str, value) -> list[int]:
        """Query secondary index. Returns slot indices."""
        if field not in self.secondary_indices:
            return []
        return self.secondary_indices[field].get(value, [])

    # -- bulk queries --------------------------------------------------------

    def get_all_edges(self) -> list[dict]:
        """Get all edges across all types."""
        result = []
        for etype, edge_list in self._edges_by_type.items():
            for src_slot, tgt_slot, data in edge_list:
                src_id = self._slot_to_id(src_slot)
                tgt_id = self._slot_to_id(tgt_slot)
                if src_id and tgt_id:
                    result.append({"source": src_id, "target": tgt_id, "kind": etype, **data})
        return result

    def _tombstone_mask(self, n: int) -> np.ndarray:
        """Cached tombstone boolean mask. Invalidated when tombstones change."""
        if not self.node_tombstones:
            return np.zeros(n, dtype=bool)
        cache = self._tombstone_mask_cache
        tomb_len = len(self.node_tombstones)
        if cache is not None and cache[0] == n and cache[1] == tomb_len:
            return cache[2]
        tomb_arr = np.array(list(self.node_tombstones), dtype=np.int32)
        mask = np.zeros(n, dtype=bool)
        valid = tomb_arr[tomb_arr < n]
        if len(valid) > 0:
            mask[valid] = True
        self._tombstone_mask_cache = (n, tomb_len, mask)
        return mask

    def compute_live_mask(self, n: int) -> np.ndarray:
        """Unified visibility: tombstones + TTL + retracted."""
        from supergraph.algos.visibility import full_live_mask
        import time as _time
        expires = self.columns.get_column("__expires_at__", n)
        retracted = self.columns.get_column("__retracted__", n)
        expires_col = expires[0] if expires is not None else None
        expires_pres = expires[1] if expires is not None else None
        retracted_col = retracted[0] if retracted is not None else None
        retracted_pres = retracted[1] if retracted is not None else None
        return full_live_mask(
            node_ids=self.node_ids,
            tombstones=self.node_tombstones,
            n=n,
            now_ms=int(_time.time() * 1000),
            expires_col=expires_col,
            expires_pres=expires_pres,
            retracted_col=retracted_col,
            retracted_pres=retracted_pres,
        )

    def _has_time_sensitive_visibility(self) -> bool:
        """True when TTL or retracted columns exist - prevents stale cache hits."""
        return (
            self.columns.has_column("__expires_at__")
            or self.columns.has_column("__retracted__")
        )

    def _live_slots(self, kind: str | None = None) -> np.ndarray:
        """Return numpy array of live slot indices, optionally filtered by kind.

        Includes tombstone, TTL-expiry, and retraction visibility. Results are
        cached by mutation version UNLESS time-sensitive columns exist (TTL /
        retracted), since those change visibility without a write.
        """
        n = self._next_slot
        if n == 0:
            return np.empty(0, dtype=np.int32)

        time_sensitive = self._has_time_sensitive_visibility()
        if not time_sensitive:
            cached = self._live_slots_cache.get(kind)
            if cached is not None and cached[0] == self._live_version:
                return cached[1]

        if kind and kind not in self.string_table:
            return np.empty(0, dtype=np.int32)

        # Full visibility: tombstones + TTL + retracted
        mask = self.compute_live_mask(n)

        if kind is not None:
            kind_id = self.string_table.intern(kind)
            mask &= self.node_kinds[:n] == kind_id

        slots = np.nonzero(mask)[0]
        if not time_sensitive:
            self._live_slots_cache[kind] = (self._live_version, slots)
        return slots

    def _live_mask(self, kind: str | None = None) -> np.ndarray:
        """Return boolean mask of live slots, optionally filtered by kind.

        Includes tombstone, TTL-expiry, and retraction visibility. Results are
        cached by mutation version UNLESS time-sensitive columns exist.
        """
        n = self._next_slot
        if n == 0:
            return np.empty(0, dtype=bool)

        time_sensitive = self._has_time_sensitive_visibility()
        if not time_sensitive:
            cached = self._live_mask_cache.get(kind)
            if cached is not None and cached[0] == self._live_version:
                return cached[1]

        # Full visibility: tombstones + TTL + retracted
        mask = self.compute_live_mask(n)

        if kind is not None:
            if kind not in self.string_table:
                mask = np.zeros(n, dtype=bool)
            else:
                kind_id = self.string_table.intern(kind)
                mask = mask & (self.node_kinds[:n] == kind_id)

        if not time_sensitive:
            self._live_mask_cache[kind] = (self._live_version, mask)
        return mask

    def _materialize_slot(self, slot: int) -> dict | None:
        """Build a full node dict from column arrays at a slot index."""
        if slot in self.node_tombstones:
            return None
        str_id = int(self.node_ids[slot])
        if str_id == -1:
            return None
        lookup = self.string_table.lookup
        cols = self.columns._columns
        pres = self.columns._presence
        dtypes = self.columns._dtypes
        d = {
            "id": lookup(str_id),
            "kind": lookup(int(self.node_kinds[slot])),
        }
        for field, col in cols.items():
            if field[0] == "_" and field[-1] == "_":
                continue
            if pres[field][slot]:
                dtype = dtypes[field]
                raw = col[slot]
                if dtype == "int32_interned":
                    d[field] = lookup(int(raw))
                elif dtype == "float64":
                    d[field] = float(raw)
                else:
                    d[field] = int(raw)
        return d

    def _materialize_bulk(self, slots: np.ndarray) -> list[dict]:
        """Vectorized bulk materialization - delegates to algos.materialization."""
        from supergraph.algos.materialization import materialize_bulk
        return materialize_bulk(
            slots=slots,
            node_ids=self.node_ids,
            node_kinds=self.node_kinds,
            id_to_str=self.string_table._id_to_str,
            columns=self.columns._columns,
            presence=self.columns._presence,
            dtypes=self.columns._dtypes,
        )

    def get_all_nodes(self, kind: str | None = None, predicate=None) -> list[dict]:
        """Get all live nodes, optionally filtered by kind and/or predicate.

        Args:
            kind: Optional kind string for numpy-accelerated filtering.
            predicate: Optional callable(raw_data_dict) -> bool. When provided,
                the predicate receives a dict of data fields (no id/kind).
        """
        slots = self._live_slots(kind)
        if len(slots) == 0:
            return []

        if predicate is None:
            return self._materialize_bulk(slots)

        result = []
        for node in self._materialize_bulk(slots):
            raw = {k: v for k, v in node.items() if k not in ("id", "kind")}
            if predicate(raw):
                result.append(node)
        return result

    def count_nodes(self, kind: str | None = None, predicate=None) -> int:
        """Count live nodes without building dicts. Uses numpy.

        Args:
            kind: Optional kind string for numpy-accelerated filtering.
            predicate: Optional callable(raw_data_dict) -> bool. When provided,
                counts only nodes whose raw data passes the predicate.
        """
        slots = self._live_slots(kind)
        if predicate is None:
            return len(slots)
        return sum(
            1 for node in self._materialize_bulk(slots)
            if predicate({k: v for k, v in node.items() if k not in ("id", "kind")})
        )

    def query_node_ids(self, kind: str | None = None, predicate=None) -> list[str]:
        """Return node IDs matching criteria without full dict construction.

        Useful for DELETE NODES WHERE - only the ID is needed to delete.
        """
        slots = self._live_slots(kind)
        if len(slots) == 0:
            return []
        if predicate is None:
            id_to_str = self.string_table._id_to_str
            return [id_to_str[s] for s in self.node_ids[slots].tolist() if s >= 0]
        return [
            node["id"]
            for node in self._materialize_bulk(slots)
            if predicate({k: v for k, v in node.items() if k not in ("id", "kind")})
        ]

    # -- field operations ----------------------------------------------------

    def increment_field(self, id: str, field: str, amount: int | float):
        """Increment a numeric field. Raises NodeNotFound or TypeError."""
        if id not in self.string_table:
            raise NodeNotFound(id)
        str_id = self.string_table.intern(id)
        slot = self.id_to_slot.get(str_id)
        if slot is None or slot in self.node_tombstones:
            raise NodeNotFound(id)

        if not self.columns.has_column(field) or not self.columns._presence[field][slot]:
            current = 0
        else:
            dtype = self.columns._dtypes[field]
            if dtype not in ("int64", "float64"):
                raise TypeError(f"Field '{field}' is not numeric: {dtype}")
            raw = self.columns._columns[field][slot]
            current = float(raw) if dtype == "float64" else int(raw)

        new_val = current + amount
        self.columns.set_field(slot, field, new_val)
        self.columns.set_reserved(slot, "__updated_at__", int(time.time() * 1000))
        self._dirty_columns = True

    def reset_dirty_flags(self):
        """Reset all dirty tracking flags after checkpoint."""
        self._dirty_nodes = False
        self.columns.dirty = False
        self._dirty_edges = False
        self._dirty_strings = False

    # -----------------------------------------------------------------
    # Snapshot / restore primitive
    # -----------------------------------------------------------------
    #
    # Centralizes the "save full mutable store state, do something, possibly
    # restore" pattern that was previously hand-rolled at four sites:
    # SYS SNAPSHOT (executor_system._snapshot/_rollback), MERGE rollback
    # (mutations._merge), BATCH rollback (mutations._batch), and the
    # counterfactual WHAT IF RETRACT (intelligence._counterfactual). Each of
    # those sites missed the same fields — string_table, secondary_indices,
    # _indexed_fields — so a SYS COMPACT STRINGS after a SYS ROLLBACK could
    # leave the restored columns referencing string IDs that no longer
    # existed (bug #29) and the identity-init in string_gc would silently
    # remap them out of bounds (bug #101). Same root cause for the BATCH
    # (#8), MERGE (#36), and counterfactual (#49) rollbacks.
    #
    # Anything the store itself owns belongs here. Vector state lives on
    # runtime and is the caller's responsibility — the snapshot returned
    # by ``make_snapshot`` is intentionally store-only so that callers can
    # compose it with their own vector save/restore logic.

    def make_snapshot(self) -> "StoreSnapshot":
        """Capture a deep copy of every mutable store-owned field.

        Cheap-ish: O(N) in node count + indexed-value count. Not free —
        callers wrapping a single small mutation should consider whether a
        narrower snapshot would do.
        """
        ns = self._next_slot
        return StoreSnapshot(
            # String table: capture both list and reverse map. Restoration
            # rebuilds the StringTable from the list, regenerating the
            # reverse map. We keep both copies so a future caller that wants
            # to inspect the snapshot doesn't pay the rebuild cost.
            strings=self.string_table.to_list(),
            # Columns has its own snapshot machinery; we just hold the result
            # opaquely and hand it back at restore time.
            columns=self.columns.snapshot_arrays(),
            # Per-slot arrays — slice + copy so the snapshot is independent
            # of subsequent mutations to the underlying buffer.
            node_ids=self.node_ids[:ns].copy(),
            node_kinds=self.node_kinds[:ns].copy(),
            # Sets and dicts: shallow copies are sufficient because the
            # values inside (slot ints, tuple keys, edge-data dicts) are
            # never mutated in place — they're replaced wholesale by the
            # callers we care about. If that invariant breaks elsewhere,
            # bump these to deep copies.
            tombstones=set(self.node_tombstones),
            edges_by_type={k: list(v) for k, v in self._edges_by_type.items()},
            edge_keys=set(self._edge_keys),
            edge_data_idx={
                k: dict(v) for k, v in self._edge_data_idx.items()
            },
            id_to_slot=dict(self.id_to_slot),
            # Index state. secondary_indices values are dicts of lists; we
            # copy the lists since they're mutated in place by put_node.
            secondary_indices={
                field: {val: list(slots) for val, slots in idx.items()}
                for field, idx in self.secondary_indices.items()
            },
            indexed_fields=set(self._indexed_fields),
            # Scalar bookkeeping
            next_slot=ns,
            count=self._count,
            capacity=self._capacity,
            active_context=self._active_context,
        )

    def restore_snapshot(self, snap: "StoreSnapshot") -> None:
        """Restore from a snapshot produced by ``make_snapshot``.

        Re-allocates buffers as needed if the store has grown since the
        snapshot. Does not touch vector store / runtime — caller handles
        that separately because vector state lives outside this class.
        """
        # Grow capacity FIRST so column restore + node array assignment land
        # in correctly-sized buffers. Pre-fix sites that grew between snap
        # and restore would have to special-case this; centralizing here
        # means everyone gets it right.
        while self._capacity < snap.capacity:
            self._grow()

        # Rebuild StringTable from the captured list. This is the field the
        # four hand-rolled sites missed. Without it, a SYS COMPACT STRINGS
        # after a SYS ROLLBACK could compact away strings that the restored
        # columns still reference, producing dangling string IDs (bugs #29,
        # #101).
        self.string_table = StringTable.from_list(list(snap.strings))
        # Rewire ColumnStore to the new StringTable; ColumnStore holds a
        # reference for value lookups and that reference must stay current.
        self.columns._string_table = self.string_table

        self.columns.restore_arrays(snap.columns)
        self.columns.grow(self._capacity)

        # Restore the per-slot arrays. Clear any slots the snapshot didn't
        # cover so leftover bytes from later inserts don't survive the
        # rollback.
        ns_snap = snap.next_slot
        self.node_ids[:ns_snap] = snap.node_ids
        self.node_kinds[:ns_snap] = snap.node_kinds
        if ns_snap < self._next_slot:
            self.node_ids[ns_snap:self._next_slot] = -1
            self.node_kinds[ns_snap:self._next_slot] = 0

        self.node_tombstones = set(snap.tombstones)
        self._edges_by_type = {k: list(v) for k, v in snap.edges_by_type.items()}
        self._edge_keys = set(snap.edge_keys)
        # Restore edge_data_idx directly so weighted lookups don't have to
        # wait for _rebuild_edges. _rebuild_edges still re-derives this from
        # _edges_by_type, but until that runs the in-flight lookups via
        # get_edge_data are accurate.
        self._edge_data_idx = {k: dict(v) for k, v in snap.edge_data_idx.items()}
        self.id_to_slot = dict(snap.id_to_slot)
        # Index state — also missed by all four hand-rolled sites. Without
        # restoring these, a rollback that removed a node would leave its
        # slot in the secondary index forever, producing phantom hits.
        self.secondary_indices = {
            field: {val: list(slots) for val, slots in idx.items()}
            for field, idx in snap.secondary_indices.items()
        }
        self._indexed_fields = set(snap.indexed_fields)

        self._next_slot = ns_snap
        self._count = snap.count
        self._active_context = snap.active_context

        # Mark everything dirty + invalidate caches. Callers usually trigger
        # _rebuild_edges as well, but we leave that to them so a no-op
        # rollback (snap == current state) doesn't pay the cost.
        self._dirty_nodes = True
        self._dirty_edges = True
        self._dirty_strings = True
        self._invalidate_live_cache()


# Module-level dataclass: out of the class so other code can import the type
# for annotations without pulling in the heavy CoreStore import path.
from dataclasses import dataclass, field as _field
from typing import Any as _Any


@dataclass(slots=True)
class StoreSnapshot:
    """Immutable-by-convention snapshot of CoreStore mutable state.

    Returned by ``CoreStore.make_snapshot``. Restore with
    ``CoreStore.restore_snapshot``. Does not include vector store state —
    that lives on the runtime and is the caller's responsibility.
    """
    strings: list[str]
    columns: _Any  # opaque dict from ColumnStore.snapshot_arrays()
    node_ids: _Any  # np.ndarray
    node_kinds: _Any  # np.ndarray
    tombstones: set[int]
    edges_by_type: dict[str, list[tuple]]
    edge_keys: set[tuple[int, int, str]]
    edge_data_idx: dict[str, dict[tuple[int, int], dict]]
    id_to_slot: dict[int, int]
    secondary_indices: dict[str, dict]
    indexed_fields: set[str]
    next_slot: int
    count: int
    capacity: int
    active_context: str | None
