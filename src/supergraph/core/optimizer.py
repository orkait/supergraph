"""Self-balancing optimizer: 6 operations that clean up accumulated pressure.

All operations assume exclusive access (no concurrent reads/writes).
The caller (SuperGraph.execute) enforces this via the _optimizing lock.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

from supergraph.algos.compact import (
    apply_slot_remap_to_edges as _algo_apply_slot_remap,
    build_live_mask as _algo_build_live_mask,
    slot_remap_plan as _algo_slot_remap_plan,
)
from supergraph.algos.eviction import (
    needs_optimization as _algo_needs_optimization,
    rank_evictable_slots as _algo_rank_evictable,
)
from supergraph.core.store import CoreStore
from supergraph.core.strings import StringTable


def health_check(store: CoreStore, vector_store=None, document_store=None) -> dict:
    """Compute pressure metrics. Lightweight, no mutations."""
    n = store._next_slot
    tombstone_count = len(store.node_tombstones)
    tombstone_ratio = tombstone_count / max(n, 1)

    string_count = len(store.string_table)
    live_nodes = store.node_count
    string_bloat = string_count / max(live_nodes, 1)

    dead_vectors = 0
    total_vectors = 0
    if vector_store is not None:
        total_vectors = vector_store.count()
        cap = min(n, vector_store._capacity)
        if cap > 0 and store.node_tombstones:
            has_vec = vector_store._has_vector[:cap]
            tomb_mask = np.zeros(cap, dtype=bool)
            tomb_arr = np.fromiter(store.node_tombstones, dtype=np.int32)
            tomb_arr = tomb_arr[tomb_arr < cap]
            if tomb_arr.size:
                tomb_mask[tomb_arr] = True
            dead_vectors = int(np.sum(has_vec & tomb_mask))

    edge_keys_count = len(store._edge_keys)
    actual_edges = sum(len(v) for v in store._edges_by_type.values())
    stale_edge_keys = edge_keys_count - actual_edges

    from supergraph.core import plan_cache as _plan_cache_hook
    cache_size = _plan_cache_hook.size()

    return {
        "tombstone_ratio": round(tombstone_ratio, 3),
        "tombstone_count": tombstone_count,
        "total_slots": n,
        "live_nodes": live_nodes,
        "string_count": string_count,
        "string_bloat": round(string_bloat, 2),
        "dead_vectors": dead_vectors,
        "total_vectors": total_vectors,
        "stale_edge_keys": max(stale_edge_keys, 0),
        "cache_size": cache_size,
    }


def needs_optimization(health: dict, compact_threshold: float = 0.2,
                       string_gc_threshold: float = 3.0,
                       cache_gc_threshold: int = 200) -> list[str]:
    return _algo_needs_optimization(
        health, compact_threshold, string_gc_threshold, cache_gc_threshold,
    )


def compact_tombstones(store: CoreStore, vector_store=None, document_store=None) -> dict:
    """Shift live slots down to eliminate tombstone gaps.

    Renumbers all slot references: node_ids, node_kinds, columns,
    edges, id_to_slot, vectors, DocumentStore, FTS5.
    Uses numpy fancy indexing for bulk array operations.
    """
    if not store.node_tombstones:
        return {"compacted": 0}

    n = store._next_slot

    live_mask = _algo_build_live_mask(store.node_ids, store.node_tombstones, n)
    old_to_new, new_count = _algo_slot_remap_plan(live_mask)
    live_slots_arr = np.nonzero(live_mask)[0]

    if new_count == n:
        return {"compacted": 0}

    old_to_new_dict = {int(old): int(old_to_new[old]) for old in live_slots_arr}

    new_capacity = max(new_count * 2, store._capacity)

    # Remap node arrays via fancy indexing
    new_ids = np.full(new_capacity, -1, dtype=np.int32)
    new_ids[:new_count] = store.node_ids[live_slots_arr]
    new_kinds = np.zeros(new_capacity, dtype=np.int32)
    new_kinds[:new_count] = store.node_kinds[live_slots_arr]

    # Remap columns via fancy indexing (no Python loop over slots)
    for field in list(store.columns._columns.keys()):
        old_col = store.columns._columns[field]
        old_pres = store.columns._presence[field]
        dtype_str = store.columns._dtypes[field]
        new_col = store.columns._make_sentinel_array(dtype_str, new_capacity)
        new_pres = np.zeros(new_capacity, dtype=bool)
        new_col[:new_count] = old_col[live_slots_arr]
        new_pres[:new_count] = old_pres[live_slots_arr]
        store.columns._columns[field] = new_col
        store.columns._presence[field] = new_pres
    store.columns._capacity = new_capacity

    for etype in list(store._edges_by_type.keys()):
        remapped = _algo_apply_slot_remap(
            store._edges_by_type[etype], old_to_new, n,
        )
        if remapped:
            store._edges_by_type[etype] = remapped
        else:
            del store._edges_by_type[etype]

    store._edge_keys = {
        (s, t, k) for k, edges in store._edges_by_type.items() for s, t, _d in edges
    }

    # Remap vectors. Vectorised: mask live_slots against _has_vector, then
    # gather vectors + new-slot numbers via numpy. Pre-fix, every live slot
    # triggered two per-slot method calls (has_vector + get_vector) plus a
    # Python list build.
    if vector_store is not None:
        has_vec = vector_store._has_vector
        valid = live_slots_arr < len(has_vec)
        live_clip = live_slots_arr[valid]
        keep = has_vec[live_clip]
        old_slots_arr = live_clip[keep]

        # Reset the index + presence bitmap at the new capacity.
        vector_store._has_vector = np.zeros(new_capacity, dtype=bool)
        vector_store._capacity = new_capacity
        from usearch.index import Index
        vector_store._index = Index(ndim=vector_store._dims, metric="cos", dtype="f32")

        if old_slots_arr.size:
            vecs = np.vstack([
                vector_store.get_vector(int(s)) for s in old_slots_arr
            ])
            new_slots = old_to_new[old_slots_arr].astype(np.int64)
            vector_store._index.add(new_slots, vecs)
            vector_store._has_vector[new_slots] = True

    # Remap DocumentStore slot keys via temp-table batch (O(1) statements vs O(N))
    if document_store is not None:
        conn = document_store._conn

        # Build live-slot temp table - used for both orphan deletion and remap.
        conn.execute(
            "CREATE TEMP TABLE _live_slots (slot INT PRIMARY KEY)"
        )
        conn.executemany(
            "INSERT INTO _live_slots VALUES (?)",
            [(s,) for s in old_to_new_dict.keys()]
        )

        # Delete tombstoned slot rows. Tombstoned slots are not in _live_slots
        # but may still occupy rows in document tables; leaving them causes
        # UNIQUE constraint violations when a live slot remaps onto that number.
        _ORPHAN_COLS = [
            ("documents", "slot"),
            ("summaries", "slot"),
            ("images", "slot"),
            ("doc_metadata", "doc_slot"),
        ]
        for tbl, col in _ORPHAN_COLS:
            conn.execute(
                f"DELETE FROM {tbl} WHERE {col} >= 0 AND {col} < ?"  # noqa: S608
                f" AND {col} NOT IN (SELECT slot FROM _live_slots)",
                (n,),
            )

        # Remap live slots that change position (old_slot != new_slot)
        remapped = [(old, new) for old, new in old_to_new_dict.items() if old != new]
        if remapped:
            conn.execute(
                "CREATE TEMP TABLE _slot_remap (old_slot INT PRIMARY KEY, new_slot INT)"
            )
            conn.executemany("INSERT INTO _slot_remap VALUES (?,?)", remapped)

            _REMAP_COLS = [
                ("documents", "slot"),
                ("summaries", "slot"),
                ("summaries", "doc_slot"),
                ("images", "slot"),
                ("doc_metadata", "doc_slot"),
            ]
            # Pass 1: old slot → negative sentinel (avoids collision with valid new slots)
            for tbl, col in _REMAP_COLS:
                conn.execute(
                    f"UPDATE {tbl} SET {col} = -(({col}) + 1)"  # noqa: S608
                    f" WHERE {col} IN (SELECT old_slot FROM _slot_remap)"
                )
            # Pass 2: negative sentinel → new slot
            for tbl, col in _REMAP_COLS:
                conn.execute(
                    f"UPDATE {tbl} SET {col} = ("  # noqa: S608
                    f"  SELECT new_slot FROM _slot_remap"
                    f"  WHERE old_slot = -(({tbl}.{col}) + 1)"
                    f") WHERE {col} IN (SELECT -(old_slot + 1) FROM _slot_remap)"
                )
            conn.execute("DROP TABLE _slot_remap")

        conn.execute("DROP TABLE _live_slots")

        # Rebuild FTS from remapped summaries
        conn.execute("DELETE FROM doc_fts")
        rows = conn.execute("SELECT slot, summary FROM summaries").fetchall()
        for slot, summary in rows:
            conn.execute("INSERT INTO doc_fts (rowid, summary) VALUES (?, ?)", (slot, summary))
        conn.commit()

    # Apply to store
    store.node_ids = new_ids
    store.node_kinds = new_kinds
    store._capacity = new_capacity
    store.node_tombstones.clear()
    store.id_to_slot = {}
    for slot in range(new_count):
        str_id = int(new_ids[slot])
        if str_id >= 0:
            store.id_to_slot[str_id] = slot
    store._next_slot = new_count
    store._edges_dirty = True
    store._ensure_edges_built()

    # Rebuild secondary indices
    for field in list(store._indexed_fields):
        store.add_index(field)

    return {"compacted": n - new_count}


def gc_strings(store: CoreStore) -> dict:
    """Rebuild string table with only referenced strings.

    Shell over supergraph.algos.string_gc: collects referenced ids,
    builds remap plan, applies in-place remap to store arrays.
    """
    from supergraph.algos.string_gc import (
        collect_referenced_ids,
        build_remap_plan,
        apply_remap_to_array,
    )

    n = store._next_slot

    live_mask = store.node_ids[:n] >= 0
    if store.node_tombstones:
        tomb_arr = np.array(list(store.node_tombstones), dtype=np.int32)
        valid = tomb_arr[tomb_arr < n]
        if len(valid) > 0:
            live_mask[valid] = False

    interned_cols = [
        (store.columns._columns[field], store.columns._presence[field])
        for field in store.columns._columns
        if store.columns._dtypes[field] == "int32_interned"
    ]
    extra_ids = {
        store.string_table.intern(etype)
        for etype in store._edges_by_type
        if etype in store.string_table
    }

    referenced = collect_referenced_ids(
        node_ids=store.node_ids,
        node_kinds=store.node_kinds,
        live_mask=live_mask,
        n=n,
        interned_columns=interned_cols,
        extra_ids=extra_ids,
    )

    old_table = store.string_table
    old_count = len(old_table)
    if len(referenced) == old_count:
        return {"strings_freed": 0}

    new_strings, remap_arr = build_remap_plan(
        referenced=referenced,
        old_table_len=old_count,
        lookup_fn=old_table.lookup,
    )

    ids = store.node_ids[:n]
    apply_remap_to_array(ids, remap_arr, ids >= 0)

    kinds = store.node_kinds[:n]
    apply_remap_to_array(kinds, remap_arr, live_mask)

    for field in store.columns._columns:
        if store.columns._dtypes[field] == "int32_interned":
            col = store.columns._columns[field][:n]
            pres = store.columns._presence[field][:n]
            apply_remap_to_array(col, remap_arr, pres)

    store.id_to_slot = {
        int(remap_arr[old_str_id]): slot
        for old_str_id, slot in store.id_to_slot.items()
    }

    new_table = StringTable.from_list(new_strings)
    store.string_table = new_table
    store.columns._string_table = new_table

    from supergraph.core import plan_cache as _plan_cache_hook
    _plan_cache_hook.clear()

    for field in list(store._indexed_fields):
        store.add_index(field)

    return {"strings_freed": old_count - len(new_strings)}


def defrag_edges(store: CoreStore) -> dict:
    """Rebuild edge keys set and CSR matrices from clean edge lists."""
    store._edge_keys = {
        (s, t, k) for k, edges in store._edges_by_type.items() for s, t, _d in edges
    }
    store._edges_dirty = True
    store._ensure_edges_built()
    return {"edge_types": len(store._edges_by_type)}


def cleanup_vectors(store: CoreStore, vector_store) -> dict:
    """Remove HNSW entries for tombstoned/retracted slots."""
    if vector_store is None:
        return {"removed": 0}

    n = store._next_slot
    live = store.compute_live_mask(n)
    removed = 0

    for slot in range(min(n, vector_store._capacity)):
        if vector_store._has_vector[slot] and not live[slot]:
            vector_store.remove(slot)
            removed += 1

    return {"removed": removed}


def sweep_orphans(store: CoreStore, document_store) -> dict:
    """Delete DocumentStore rows not referenced by live slots."""
    if document_store is None:
        return {"cleaned": 0}

    live = store.compute_live_mask(store._next_slot)
    live_slots = set(int(s) for s in np.nonzero(live)[0])
    cleaned = document_store.orphan_cleanup(live_slots)
    return {"cleaned": cleaned}


def clear_caches(store: CoreStore) -> dict:
    """Clear plan cache and edge combination/transpose caches."""
    from supergraph.core import plan_cache as _plan_cache_hook
    _plan_cache_hook.clear()
    store.edge_matrices._cache.clear()
    store.edge_matrices._transpose_cache.clear()
    return {"cleared": True}


def compact_tombstones_safe(
    store: CoreStore,
    schema,
    conn,
    vector_store=None,
    document_store=None,
) -> dict:
    """Crash-safe compact_tombstones using an atomic ATTACH transaction.
    
    Eliminates the double-checkpoint IO trap. If a crash occurs, SQLite 
    rolls back both the blobs and the documents DB automatically.
    """
    if conn is None:
        return compact_tombstones(store, vector_store, document_store)
    if not store.node_tombstones:
        return {"compacted": 0}

    from supergraph.persistence.serializer import checkpoint as _checkpoint

    n = store._next_slot
    live_mask = _algo_build_live_mask(store.node_ids, store.node_tombstones, n)
    old_to_new, new_count = _algo_slot_remap_plan(live_mask)
    live_slots_arr = np.nonzero(live_mask)[0]

    if new_count == n:
        return {"compacted": 0}

    old_to_new_dict = {int(old): int(old_to_new[old]) for old in live_slots_arr}

    attached = False
    if document_store is not None and not document_store._temp:
        # Flush any pending transactions in DocumentStore before attaching
        document_store._conn.commit()
        safe_doc_path = document_store._path.replace("'", "''")
        conn.execute(f"ATTACH DATABASE '{safe_doc_path}' AS docs")
        attached = True

    try:
        conn.execute("BEGIN IMMEDIATE")

        # 1. Atomic Document Remapping via ATTACH
        if attached:
            conn.execute("CREATE TEMP TABLE _live_slots (slot INT PRIMARY KEY)")
            conn.executemany("INSERT INTO _live_slots VALUES (?)", [(s,) for s in old_to_new_dict.keys()])

            _ORPHAN_COLS = [
                ("docs.documents", "slot"),
                ("docs.summaries", "slot"),
                ("docs.images", "slot"),
                ("docs.doc_metadata", "doc_slot"),
            ]
            for tbl, col in _ORPHAN_COLS:
                conn.execute(
                    f"DELETE FROM {tbl} WHERE {col} >= 0 AND {col} < ? "
                    f"AND {col} NOT IN (SELECT slot FROM _live_slots)",
                    (n,),
                )

            remapped = [(old, new) for old, new in old_to_new_dict.items() if old != new]
            if remapped:
                conn.execute("CREATE TEMP TABLE _slot_remap (old_slot INT PRIMARY KEY, new_slot INT)")
                conn.executemany("INSERT INTO _slot_remap VALUES (?,?)", remapped)

                _REMAP_COLS = [
                    ("docs.documents", "slot"),
                    ("docs.summaries", "slot"),
                    ("docs.summaries", "doc_slot"),
                    ("docs.images", "slot"),
                    ("docs.doc_metadata", "doc_slot"),
                ]
                for tbl, col in _REMAP_COLS:
                    conn.execute(f"UPDATE {tbl} SET {col} = -(({col}) + 1) WHERE {col} IN (SELECT old_slot FROM _slot_remap)")
                for tbl, col in _REMAP_COLS:
                    conn.execute(f"UPDATE {tbl} SET {col} = (SELECT new_slot FROM _slot_remap WHERE old_slot = -(({tbl}.{col}) + 1)) WHERE {col} IN (SELECT -(old_slot + 1) FROM _slot_remap)")

                conn.execute("DROP TABLE _slot_remap")
            conn.execute("DROP TABLE _live_slots")

            conn.execute("DELETE FROM docs.doc_fts")
            rows = conn.execute("SELECT slot, summary FROM docs.summaries").fetchall()
            for slot, summary in rows:
                conn.execute("INSERT INTO docs.doc_fts (rowid, summary) VALUES (?, ?)", (slot, summary))

        # 2. In-Memory Array Remapping
        new_capacity = max(new_count * 2, store._capacity)
        new_ids = np.full(new_capacity, -1, dtype=np.int32)
        new_ids[:new_count] = store.node_ids[live_slots_arr]
        new_kinds = np.zeros(new_capacity, dtype=np.int32)
        new_kinds[:new_count] = store.node_kinds[live_slots_arr]

        new_columns = {}
        new_presences = {}
        for field in list(store.columns._columns.keys()):
            old_col = store.columns._columns[field]
            old_pres = store.columns._presence[field]
            dtype_str = store.columns._dtypes[field]
            new_col = store.columns._make_sentinel_array(dtype_str, new_capacity)
            new_pres = np.zeros(new_capacity, dtype=bool)
            new_col[:new_count] = old_col[live_slots_arr]
            new_pres[:new_count] = old_pres[live_slots_arr]
            new_columns[field] = new_col
            new_presences[field] = new_pres

        new_edges = {}
        for etype in list(store._edges_by_type.keys()):
            remapped_edges = _algo_apply_slot_remap(store._edges_by_type[etype], old_to_new, n)
            if remapped_edges:
                new_edges[etype] = remapped_edges

        # Vector Remapping
        if vector_store is not None:
            old_slots = []
            vecs = []
            for old_slot in live_slots_arr:
                old_slot = int(old_slot)
                if vector_store.has_vector(old_slot):
                    old_slots.append(old_slot)
                    vecs.append(vector_store.get_vector(old_slot))

            vector_store._has_vector = np.zeros(new_capacity, dtype=bool)
            vector_store._capacity = new_capacity
            from usearch.index import Index
            vector_store._index = Index(ndim=vector_store._dims, metric="cos", dtype="f32")
            if old_slots:
                new_slots = np.array([old_to_new_dict[s] for s in old_slots], dtype=np.int64)
                vector_store._index.add(new_slots, np.vstack(vecs))
                vector_store._has_vector[new_slots] = True

        # 3. Apply to Store Pointers
        store.node_ids = new_ids
        store.node_kinds = new_kinds
        store._capacity = new_capacity
        store.node_tombstones.clear()
        store.id_to_slot = {}
        for slot in range(new_count):
            str_id = int(new_ids[slot])
            if str_id >= 0:
                store.id_to_slot[str_id] = slot

        for field in list(store.columns._columns.keys()):
            store.columns._columns[field] = new_columns[field]
            store.columns._presence[field] = new_presences[field]
        store.columns._capacity = new_capacity

        store._edges_by_type = new_edges
        store._edge_keys = {(s, t, k) for k, edges in store._edges_by_type.items() for s, t, _d in edges}
        store._next_slot = new_count
        store._edges_dirty = True
        store._ensure_edges_built()

        for field in list(store._indexed_fields):
            store.add_index(field)

        store._dirty_nodes = True
        store._dirty_columns = True
        store._dirty_edges = True
        store._dirty_strings = True

        # 4. Atomic Flush of SuperGraph Blobs
        _checkpoint(store, schema, conn, force=True)

    except Exception as e:
        conn.execute("ROLLBACK")
        raise e
    finally:
        if attached:
            conn.execute("DETACH DATABASE docs")

    return {"compacted": n - new_count}


def optimize_all(
    store: CoreStore,
    vector_store=None,
    document_store=None,
    schema=None,
    conn=None,
) -> dict:
    """Run all 6 optimization operations in safe order.

    Short-circuits on the first failure; returns results so far plus
    {"error": "..."}. When schema and conn are both provided the compact
    step routes through compact_tombstones_safe.
    """
    if conn is not None and schema is not None:
        compact_step = lambda: compact_tombstones_safe(
            store, schema, conn, vector_store, document_store
        )
    else:
        compact_step = lambda: compact_tombstones(
            store, vector_store, document_store
        )

    results: dict = {}
    steps: list[tuple[str, callable]] = [
        ("compact", compact_step),
        ("strings", lambda: gc_strings(store)),
        ("edges",   lambda: defrag_edges(store)),
        ("vectors", lambda: cleanup_vectors(store, vector_store)),
        ("blobs",   lambda: sweep_orphans(store, document_store)),
        ("cache",   lambda: clear_caches(store)),
    ]
    for name, op in steps:
        try:
            results[name] = op()
        except Exception as e:
            results["error"] = f"{name}: {type(e).__name__}: {e}"
            break
    return results


def _get_evictable_slots_sorted(store: CoreStore, protected_kinds: set) -> list[int]:
    """Shell: resolve store column arrays, delegate to algos.eviction."""
    n = store._next_slot
    if n == 0:
        return []

    live = store.compute_live_mask(n)
    updated_col = store.columns.get_column("__updated_at__", n)
    if updated_col is not None:
        col_data, col_pres, _ = updated_col
        return _algo_rank_evictable(
            live_mask=live,
            kind_ids=store.node_kinds[:n],
            kind_lookup=store.string_table.lookup,
            updated_at=col_data[:n],
            updated_at_present=col_pres[:n],
            protected_kinds=protected_kinds,
        )
    return _algo_rank_evictable(
        live_mask=live,
        kind_ids=store.node_kinds[:n],
        kind_lookup=store.string_table.lookup,
        updated_at=None,
        updated_at_present=None,
        protected_kinds=protected_kinds,
    )


def _evict_nodes(store: CoreStore, slots_to_evict: list[int], vector_store=None, document_store=None) -> int:
    """Helper: actually evict the given slots."""
    if not slots_to_evict:
        return 0

    evicted = 0
    for slot in slots_to_evict:
        str_id = int(store.node_ids[slot])
        if str_id < 0:
            continue
        try:
            _ = store.string_table.lookup(str_id)
        except KeyError:
            continue

        if vector_store is not None:
            vector_store.remove(slot)
        if document_store is not None:
            try:
                document_store.delete_document(slot)
            except Exception as err:
                logger.debug("document_store.delete_document(%s) failed during eviction: %s", slot, err)

        # Tombstone
        store.columns.clear(slot)
        store.node_tombstones.add(slot)
        if str_id in store.id_to_slot:
            del store.id_to_slot[str_id]
        store._count -= 1
        evicted += 1

    # Cascade edge cleanup for all evicted slots
    if evicted > 0:
        store._edge_keys = {
            (s, t, k)
            for k, edges in store._edges_by_type.items()
            for s, t, _d in edges
            if s not in store.node_tombstones and t not in store.node_tombstones
        }
        for etype in list(store._edges_by_type.keys()):
            store._edges_by_type[etype] = [
                (s, t, d) for s, t, d in store._edges_by_type[etype]
                if s not in store.node_tombstones and t not in store.node_tombstones
            ]
            if not store._edges_by_type[etype]:
                del store._edges_by_type[etype]
        store._edges_dirty = True
        store._ensure_edges_built()

    return evicted


def evict_oldest(store: CoreStore, target_bytes: int, vector_store=None, document_store=None, protected_kinds: set | None = None) -> dict:
    """Evict oldest non-protected nodes until memory usage is at or below target_bytes."""
    from supergraph.core.memory import measure
    
    PROTECTED_KINDS = protected_kinds if protected_kinds is not None else {"schema", "config", "system"}
    
    before = measure(store, vector_store)
    if before["total"] <= target_bytes:
        return {"evicted": 0, "bytes_before": before["total"], "bytes_after": before["total"]}

    candidates = _get_evictable_slots_sorted(store, PROTECTED_KINDS)
    evicted = 0
    
    to_evict = []
    for slot in candidates:
        to_evict.append(slot)
        # Check if we're under target periodically to avoid measuring every node
        if len(to_evict) % 100 == 0:
            _evict_nodes(store, to_evict, vector_store, document_store)
            evicted += len(to_evict)
            to_evict.clear()
            
            current = measure(store, vector_store)
            if current["total"] <= target_bytes:
                break
                
    if to_evict:
        _evict_nodes(store, to_evict, vector_store, document_store)
        evicted += len(to_evict)

    after = measure(store, vector_store)
    return {"evicted": evicted, "bytes_before": before["total"], "bytes_after": after["total"]}


def evict_by_count(store: CoreStore, limit: int, vector_store=None, document_store=None, protected_kinds: set | None = None) -> dict:
    """Evict exactly limit number of oldest non-protected nodes."""
    from supergraph.core.memory import measure
    
    if limit <= 0:
        before = measure(store, vector_store)
        return {"evicted": 0, "bytes_before": before["total"], "bytes_after": before["total"]}
        
    PROTECTED_KINDS = protected_kinds if protected_kinds is not None else {"schema", "config", "system"}
    
    before = measure(store, vector_store)
    
    candidates = _get_evictable_slots_sorted(store, PROTECTED_KINDS)
    to_evict = candidates[:limit]
    
    evicted = _evict_nodes(store, to_evict, vector_store, document_store)
    
    after = measure(store, vector_store)
    return {"evicted": evicted, "bytes_before": before["total"], "bytes_after": after["total"]}
