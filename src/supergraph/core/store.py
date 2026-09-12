
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

    def __init__(self, ceiling_bytes: int = DEFAULT_CEILING_BYTES, capacity: int = 1024, use_compression: bool = False):
        self.string_table = StringTable()
        self._edge_matrices = EdgeMatrices(use_compression=use_compression)
        self._ceiling_bytes = ceiling_bytes

        self._capacity = capacity
        self._count = 0
        self._next_slot = 0
        self.node_ids = np.full(self._capacity, -1, dtype=np.int32)
        self.node_kinds = np.zeros(self._capacity, dtype=np.int32)
        self.node_tombstones: set[int] = set()
        self.id_to_slot: dict[int, int] = {}

        self._edges_by_type: dict[str, list[tuple]] = {}
        self._edge_keys: set[tuple[int, int, str]] = set()
        self._edge_data_idx: dict[str, dict[tuple[int, int], dict]] = {}
        self._edges_dirty = False

        self.secondary_indices: dict[str, dict] = {}
        self._indexed_fields: set[str] = set()

        self.columns = ColumnStore(self.string_table, self._capacity)

        self._active_context: str | None = None

        self._active_namespace: str | None = None

        self._snapshots: dict[str, dict] = {}
        self._tombstone_mask_cache: tuple[int, int, np.ndarray] | None = None

        self._dirty_nodes = True
        self._dirty_edges = True
        self._dirty_strings = True

        self._live_version: int = 0
        self._live_mask_cache: dict[str | None, tuple[int, np.ndarray]] = {}
        self._live_slots_cache: dict[str | None, tuple[int, np.ndarray]] = {}

        self._bytes_per_node_estimate: float = float(BYTES_PER_NODE_ESTIMATE)
        self._bytes_per_edge_estimate: float = float(BYTES_PER_EDGE_ESTIMATE)
        self._calibrate_at_count: int = 1000

    @property
    def _dirty_columns(self) -> bool:
        return self.columns.dirty

    @_dirty_columns.setter
    def _dirty_columns(self, value: bool) -> None:
        self.columns.dirty = bool(value)


    def _alloc_slot(self) -> int:
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
        self._live_version += 1

    def _recalibrate_ceiling_estimate(self) -> None:
        if self._count < 100:
            return
        report = measure(self, vector_store=None, skip_csr=True)
        core_bytes = report["core_total"]
        self._bytes_per_node_estimate = core_bytes / self._count
        raw_edge_count = sum(len(v) for v in self._edges_by_type.values())
        if raw_edge_count > 0:
            edge_bytes = report["edge_lists"] + report["edge_data_idx"]
            self._bytes_per_edge_estimate = edge_bytes / raw_edge_count
        self._calibrate_at_count = self._count + min(self._count, 10_000)

    def _grow(self):
        new_cap = self._capacity * 2

        new_ids = np.full(new_cap, -1, dtype=np.int32)
        new_ids[: self._capacity] = self.node_ids
        self.node_ids = new_ids

        new_kinds = np.zeros(new_cap, dtype=np.int32)
        new_kinds[: self._capacity] = self.node_kinds
        self.node_kinds = new_kinds

        self.columns.grow(new_cap)
        self._capacity = new_cap


    @property
    def node_count(self) -> int:
        return self._count

    @property
    def edge_matrices(self) -> EdgeMatrices:
        self._ensure_edges_built()
        return self._edge_matrices

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self._edges_by_type.values())


    def put_node(self, id: str, kind: str, data: dict) -> int:
        if id in self.string_table:
            str_id = self.string_table.intern(id)
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
        if self._active_namespace:
            self.columns.set_reserved(slot, "__namespace__", self._active_namespace)

        for field in self._indexed_fields:
            if field in data:
                val = data[field]
                self.secondary_indices[field].setdefault(val, []).append(slot)

        if self._count >= self._calibrate_at_count:
            self._recalibrate_ceiling_estimate()

        return slot

    def get_node(self, id: str) -> dict | None:
        if id not in self.string_table:
            return None
        str_id = self.string_table.intern(id)
        slot = self.id_to_slot.get(str_id)
        if slot is None or slot in self.node_tombstones:
            return None
        return self._materialize_slot(slot)

    def update_node(self, id: str, data: dict):
        if id not in self.string_table:
            raise NodeNotFound(id)
        str_id = self.string_table.intern(id)
        slot = self.id_to_slot.get(str_id)
        if slot is None or slot in self.node_tombstones:
            raise NodeNotFound(id)

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

        self.columns.set(slot, data)

        for field in self._indexed_fields:
            if field in data:
                self.secondary_indices[field].setdefault(data[field], []).append(slot)
        self.columns.set_reserved(slot, "__updated_at__", int(time.time() * 1000))
        self._dirty_columns = True
        self._dirty_strings = True

    def upsert_node(self, id: str, kind: str, data: dict) -> int:
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
        if not ids:
            return []

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
        if document_store is not None and slots_to_remove:
            try:
                document_store.delete_documents_batch(slots_to_remove)
            except Exception as _ds_err:
                _rm_log.debug("doc batch delete failed: %s", _ds_err)
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
        if id not in self.string_table:
            raise NodeNotFound(id)
        str_id = self.string_table.intern(id)
        slot = self.id_to_slot.get(str_id)
        if slot is None or slot in self.node_tombstones:
            raise NodeNotFound(id)

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

        self.columns.clear(slot)
        self.node_tombstones.add(slot)
        self._tombstone_mask_cache = None
        del self.id_to_slot[str_id]
        self._count -= 1
        self._dirty_nodes = True
        self._dirty_columns = True
        self._dirty_edges = True
        self._invalidate_live_cache()

        self._cascade_delete_edges(slot)


    def put_edge(
        self, source_id: str, target_id: str, kind: str, data: dict | None = None
    ):
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

        edge_key = (src_slot, tgt_slot, kind)
        edge_data = data or {}
        if edge_key in self._edge_keys:
            existing = self._edge_data_idx.get(kind, {}).get((src_slot, tgt_slot))
            if existing == edge_data:
                raise SuperGraphError(
                    f"Duplicate edge: {source_id} -> {target_id} kind={kind}"
                )
            return

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

        weight = float(edge_data.get("weight", 1.0))
        self._edge_matrices.add_dynamic(src_slot, tgt_slot, kind, self._next_slot, weight)
        if self._edge_matrices._pending_edge_count > 10000:
            self._edges_dirty = True

        self._dirty_edges = True
        self._dirty_strings = True
        self._invalidate_live_cache()

    def delete_edge(self, source_id: str, target_id: str, kind: str):
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
        if not edges:
            return 0

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


    def _cascade_delete_edges(self, slot: int):
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
        if not self._edges_dirty:
            return
        self._edges_dirty = False
        num_nodes = max(self._next_slot, 1)
        self._edge_matrices.rebuild(self._edges_by_type, num_nodes)

    def _rebuild_edges(self):
        self._edges_dirty = False
        num_nodes = max(self._next_slot, 1)
        self._edge_matrices.rebuild(self._edges_by_type, num_nodes)
        self._edge_data_idx = {
            k: {(s, t): d for s, t, d in edges}
            for k, edges in self._edges_by_type.items()
        }


    def _slot_to_id(self, slot: int) -> str | None:
        if slot >= self._next_slot or slot in self.node_tombstones:
            return None
        str_id = int(self.node_ids[slot])
        if str_id == -1:
            return None
        return self.string_table.lookup(str_id)


    def add_index(self, field: str):
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
        if field not in self._indexed_fields:
            return
        if not self.columns.has_column(field):
            return
        idx = self.secondary_indices.setdefault(field, {})
        slot_set = {int(s) for s in slots}

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
        if field not in self.secondary_indices:
            return []
        return self.secondary_indices[field].get(value, [])


    def get_all_edges(self) -> list[dict]:
        result = []
        for etype, edge_list in self._edges_by_type.items():
            for src_slot, tgt_slot, data in edge_list:
                src_id = self._slot_to_id(src_slot)
                tgt_id = self._slot_to_id(tgt_slot)
                if src_id and tgt_id:
                    result.append({"source": src_id, "target": tgt_id, "kind": etype, **data})
        return result

    def _tombstone_mask(self, n: int) -> np.ndarray:
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
        return (
            self.columns.has_column("__expires_at__")
            or self.columns.has_column("__retracted__")
        )

    def _live_slots(self, kind: str | None = None) -> np.ndarray:
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

        mask = self.compute_live_mask(n)

        if kind is not None:
            kind_id = self.string_table.intern(kind)
            mask &= self.node_kinds[:n] == kind_id

        slots = np.nonzero(mask)[0]
        if not time_sensitive:
            self._live_slots_cache[kind] = (self._live_version, slots)
        return slots

    def _live_mask(self, kind: str | None = None) -> np.ndarray:
        n = self._next_slot
        if n == 0:
            return np.empty(0, dtype=bool)

        time_sensitive = self._has_time_sensitive_visibility()
        if not time_sensitive:
            cached = self._live_mask_cache.get(kind)
            if cached is not None and cached[0] == self._live_version:
                return cached[1]

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
        slots = self._live_slots(kind)
        if predicate is None:
            return len(slots)
        return sum(
            1 for node in self._materialize_bulk(slots)
            if predicate({k: v for k, v in node.items() if k not in ("id", "kind")})
        )

    def query_node_ids(self, kind: str | None = None, predicate=None) -> list[str]:
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


    def increment_field(self, id: str, field: str, amount: int | float):
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
        self._dirty_nodes = False
        self.columns.dirty = False
        self._dirty_edges = False
        self._dirty_strings = False


    def make_snapshot(self) -> "StoreSnapshot":
        ns = self._next_slot
        return StoreSnapshot(
            strings=self.string_table.to_list(),
            columns=self.columns.snapshot_arrays(),
            node_ids=self.node_ids[:ns].copy(),
            node_kinds=self.node_kinds[:ns].copy(),
            tombstones=set(self.node_tombstones),
            edges_by_type={k: list(v) for k, v in self._edges_by_type.items()},
            edge_keys=set(self._edge_keys),
            edge_data_idx={
                k: dict(v) for k, v in self._edge_data_idx.items()
            },
            id_to_slot=dict(self.id_to_slot),
            secondary_indices={
                field: {val: list(slots) for val, slots in idx.items()}
                for field, idx in self.secondary_indices.items()
            },
            indexed_fields=set(self._indexed_fields),
            next_slot=ns,
            count=self._count,
            capacity=self._capacity,
            active_context=self._active_context,
        )

    def restore_snapshot(self, snap: "StoreSnapshot") -> None:
        while self._capacity < snap.capacity:
            self._grow()

        self.string_table = StringTable.from_list(list(snap.strings))
        self.columns._string_table = self.string_table

        self.columns.restore_arrays(snap.columns)
        self.columns.grow(self._capacity)

        ns_snap = snap.next_slot
        self.node_ids[:ns_snap] = snap.node_ids
        self.node_kinds[:ns_snap] = snap.node_kinds
        if ns_snap < self._next_slot:
            self.node_ids[ns_snap:self._next_slot] = -1
            self.node_kinds[ns_snap:self._next_slot] = 0

        self.node_tombstones = set(snap.tombstones)
        self._edges_by_type = {k: list(v) for k, v in snap.edges_by_type.items()}
        self._edge_keys = set(snap.edge_keys)
        self._edge_data_idx = {k: dict(v) for k, v in snap.edge_data_idx.items()}
        self.id_to_slot = dict(snap.id_to_slot)
        self.secondary_indices = {
            field: {val: list(slots) for val, slots in idx.items()}
            for field, idx in snap.secondary_indices.items()
        }
        self._indexed_fields = set(snap.indexed_fields)

        self._next_slot = ns_snap
        self._count = snap.count
        self._active_context = snap.active_context

        self._dirty_nodes = True
        self._dirty_edges = True
        self._dirty_strings = True
        self._invalidate_live_cache()


from dataclasses import dataclass
from typing import Any as _Any


@dataclass(slots=True)
class StoreSnapshot:
    strings: list[str]
    columns: _Any
    node_ids: _Any
    node_kinds: _Any
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
