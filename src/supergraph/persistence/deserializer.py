"""Deserialize a CoreStore + SchemaRegistry from sqlite.

load() reads the full graph state from the blobs and metadata tables,
reconstructing the in-memory CoreStore with arrays, edge matrices,
secondary indices, and tombstones.
"""

import os
from pathlib import Path
from urllib.parse import unquote

import msgspec.json as mjson
import numpy as np

from supergraph.core.store import CoreStore
from supergraph.core.strings import StringTable
from supergraph.core.schema import SchemaRegistry
from supergraph.persistence.database import SCHEMA_VERSION
from supergraph.core.errors import VersionMismatch


def load(conn, use_compression: bool = False, db_path: str | Path | None = None) -> tuple[CoreStore, SchemaRegistry]:
    """Load graph from sqlite. Returns (store, schema).

    Raises VersionMismatch if schema_version doesn't match.
    Returns empty store if no data found.
    """
    # Check version
    row = conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
    if row is None:
        # Fresh database, return empty store
        return CoreStore(use_compression=use_compression), SchemaRegistry()

    if int(row[0]) != SCHEMA_VERSION:
        raise VersionMismatch(found=row[0], expected=SCHEMA_VERSION)

    # Load string table
    strings_row = conn.execute("SELECT data FROM blobs WHERE key='strings'").fetchone()
    if strings_row is None:
        return CoreStore(use_compression=use_compression), SchemaRegistry()

    string_table = StringTable.from_list(mjson.decode(strings_row[0]))

    # Load store metadata. Missing row means corrupted / partially-written
    # database - fall back to empty store rather than surfacing
    # `TypeError: 'NoneType' object is not subscriptable`.
    meta_row = conn.execute("SELECT data FROM blobs WHERE key='store_meta'").fetchone()
    if meta_row is None:
        return CoreStore(use_compression=use_compression), SchemaRegistry()
    meta = mjson.decode(meta_row[0])

    # Create store and set its internals
    store = CoreStore(use_compression=use_compression)
    store.string_table = string_table
    store.columns._string_table = string_table
    store._next_slot = meta["next_slot"]
    store._count = meta["count"]

    # Ensure capacity
    capacity = meta.get("capacity", max(meta["next_slot"] * 2, 1024))
    store._capacity = capacity
    store.columns._capacity = capacity

    # Load node arrays. Each blob is required once store_meta exists - a
    # checkpoint writes them in a single transaction. If any are missing we
    # refuse to half-load: bail to empty so the caller hits NodeNotFound
    # instead of a silent zero-length array shadowing real data.
    ids_row = conn.execute("SELECT data, dtype FROM blobs WHERE key='node_ids'").fetchone()
    kinds_row = conn.execute("SELECT data, dtype FROM blobs WHERE key='node_kinds'").fetchone()
    tomb_row = conn.execute("SELECT data FROM blobs WHERE key='tombstones'").fetchone()
    if ids_row is None or kinds_row is None or tomb_row is None:
        return CoreStore(use_compression=use_compression), SchemaRegistry()

    loaded_ids = np.frombuffer(ids_row[0], dtype=np.dtype(ids_row[1])).copy()
    store.node_ids = np.full(capacity, -1, dtype=np.int32)
    store.node_ids[:len(loaded_ids)] = loaded_ids

    loaded_kinds = np.frombuffer(kinds_row[0], dtype=np.dtype(kinds_row[1])).copy()
    store.node_kinds = np.zeros(capacity, dtype=np.int32)
    store.node_kinds[:len(loaded_kinds)] = loaded_kinds

    store.node_tombstones = set(mjson.decode(tomb_row[0]))

    # Rebuild id_to_slot from node_ids array. Single vectorised filter
    # instead of a Python loop over every slot.
    n = store._next_slot
    ids = store.node_ids[:n]
    live = ids >= 0
    if store.node_tombstones:
        tomb_arr = np.fromiter(store.node_tombstones, dtype=np.int64)
        tomb_arr = tomb_arr[tomb_arr < n]
        if tomb_arr.size:
            live[tomb_arr] = False
    live_slots = np.nonzero(live)[0]
    store.id_to_slot = dict(zip(ids[live_slots].astype(int).tolist(),
                                live_slots.astype(int).tolist()))

    # Load raw edge lists
    store._edges_by_type = {}
    edge_rows = conn.execute(
        "SELECT key, data FROM blobs WHERE key LIKE 'raw_edges:%'"
    ).fetchall()
    for key, data in edge_rows:
        # Symmetric unquote — see serializer for why etypes are URL-quoted
        # on write (bug #7). Legacy databases that wrote unquoted etypes
        # still decode correctly because alnum characters are unaffected
        # by quote/unquote round-tripping.
        etype = unquote(key[len("raw_edges:"):])
        # mjson.decode returns tuples as lists; wrap to tuples in one shot.
        # Previously did a per-edge generator + tuple repack.
        store._edges_by_type[etype] = [tuple(e) for e in mjson.decode(data)]

    # Rebuild edge keys set and edge matrices
    store._edge_keys = {
        (s, t, k)
        for k, edges in store._edges_by_type.items()
        for s, t, _d in edges
    }
    store._rebuild_edges()

    # Load indexed field names (indices rebuilt after columns are loaded)
    idx_row = conn.execute("SELECT data FROM blobs WHERE key='indexed_fields'").fetchone()
    indexed_fields = mjson.decode(idx_row[0]) if idx_row else []

    # Load column store data
    col_rows = conn.execute(
        "SELECT key, data, dtype FROM blobs WHERE key LIKE 'columns:%'"
    ).fetchall()

    if col_rows:
        col_blobs: dict[str, dict] = {}
        for key, data, dtype in col_rows:
            parts = key.split(":", 2)
            if len(parts) == 3:
                field_name = unquote(parts[1])
                sub_key = parts[2]
                col_blobs.setdefault(field_name, {})[sub_key] = (data, dtype)

        for field_name, blobs in col_blobs.items():
            if "data" in blobs and "presence" in blobs and "dtype" in blobs:
                data_blob, data_np_dtype = blobs["data"]
                pres_blob, pres_np_dtype = blobs["presence"]
                col_dtype_raw = blobs["dtype"][0]
                col_dtype_str = col_dtype_raw.decode() if isinstance(col_dtype_raw, bytes) else col_dtype_raw

                col_arr = np.frombuffer(data_blob, dtype=np.dtype(data_np_dtype)).copy()
                pres_arr = np.frombuffer(pres_blob, dtype=np.dtype(pres_np_dtype)).copy()

                full_col = store.columns._make_sentinel_array(col_dtype_str, capacity)
                full_col[:len(col_arr)] = col_arr
                store.columns._columns[field_name] = full_col

                full_pres = np.zeros(capacity, dtype=bool)
                full_pres[:len(pres_arr)] = pres_arr
                store.columns._presence[field_name] = full_pres

                store.columns._dtypes[field_name] = col_dtype_str

    # Rebuild secondary indices now that columns are loaded
    if idx_row:
        for field in indexed_fields:
            store.add_index(field)

    # Load vector index if present
    dims_row = conn.execute("SELECT data FROM blobs WHERE key='vector_dims'").fetchone()
    if dims_row:
        from supergraph.vector.store import VectorStore
        dims = int(dims_row[0])
        
        vector_path = None
        if db_path:
            vector_path = os.path.join(str(db_path), "vectors.usearch")
            
        store.vectors = VectorStore(dims=dims, capacity=capacity, path=vector_path)

        index_row = conn.execute("SELECT data, dtype FROM blobs WHERE key='vector_index'").fetchone()
        if index_row:
            data, dtype = index_row
            if dtype == "marker" and data == b"FILE_BASED":
                # Handled by __init__(path=...) above
                pass
            else:
                store.vectors.load(data)

        pres_row = conn.execute("SELECT data, dtype FROM blobs WHERE key='vector_presence'").fetchone()
        if pres_row:
            loaded = np.frombuffer(pres_row[0], dtype=np.dtype(pres_row[1])).copy()
            store.vectors._has_vector[:len(loaded)] = loaded
    else:
        store.vectors = None

    # Load schema
    schema = SchemaRegistry()
    schema_row = conn.execute("SELECT data FROM blobs WHERE key='schema'").fetchone()
    if schema_row:
        schema = SchemaRegistry.from_dict(mjson.decode(schema_row[0]))

    return store, schema
