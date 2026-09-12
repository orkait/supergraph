"""Serialize a CoreStore + SchemaRegistry to sqlite.

checkpoint() writes the graph state into the blobs and metadata
tables within a single transaction. Skips clean subsystems unless
force=True.
"""

import time
from urllib.parse import quote

import msgspec.json as mjson

from supergraph.persistence.database import SCHEMA_VERSION
from supergraph.core.store import CoreStore
from supergraph.core.schema import SchemaRegistry


def checkpoint(store: CoreStore, schema: SchemaRegistry, conn, force: bool = False):
    """Write graph state to sqlite. Skips clean subsystems unless force=True."""
    with conn:
        if force or store._dirty_strings:
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         ("strings", mjson.encode(store.string_table.to_list()), "json"))

        if force or store._dirty_nodes:
            kinds_data = store.node_kinds[:store._next_slot]
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         ("node_kinds", kinds_data.tobytes(), str(kinds_data.dtype)))

            ids_data = store.node_ids[:store._next_slot]
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         ("node_ids", ids_data.tobytes(), str(ids_data.dtype)))

            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         ("tombstones", mjson.encode(list(store.node_tombstones)), "json"))

            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         ("store_meta", mjson.encode({
                             "next_slot": store._next_slot,
                             "count": store._count,
                             "capacity": store._capacity,
                         }), "json"))

        if force or store._dirty_columns:
            conn.execute("DELETE FROM blobs WHERE key LIKE 'columns:%'")
            for field in store.columns._columns:
                col_data = store.columns._columns[field][:store._next_slot]
                pres_data = store.columns._presence[field][:store._next_slot]
                dtype_str = store.columns._dtypes[field]
                safe_field = quote(field, safe="")
                conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                             (f"columns:{safe_field}:data", col_data.tobytes(), str(col_data.dtype)))
                conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                             (f"columns:{safe_field}:presence", pres_data.tobytes(), str(pres_data.dtype)))
                conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                             (f"columns:{safe_field}:dtype", dtype_str.encode(), "text"))

        if force or store._dirty_edges:
            # Clean up any legacy edges:* blobs written by older versions.
            # The serializer used to write CSR matrix bytes under edges:*
            # keys but the deserializer only ever read raw_edges:* (bug #6).
            # Keep the DELETE so upgrading databases don't accumulate cruft;
            # drop the writes themselves.
            conn.execute("DELETE FROM blobs WHERE key LIKE 'edges:%'")
            conn.execute("DELETE FROM blobs WHERE key LIKE 'edge_data:%'")
            conn.execute("DELETE FROM blobs WHERE key LIKE 'raw_edges:%'")

            for etype, edge_list in store._edges_by_type.items():
                # URL-quote edge type names that contain colons or other
                # separator characters. Without this, an etype like
                # ``"category:subcategory"`` produces a key like
                # ``raw_edges:category:subcategory`` whose structure
                # collides with the ``LIKE 'raw_edges:%'`` scan during
                # load (bug #7). ``quote`` is imported at module level
                # alongside the columns-loop usage on line 50; pre-fix I
                # had a redundant local import inside this loop which
                # shadowed the module-level ``quote`` and broke the
                # earlier columns loop with ``UnboundLocalError``.
                safe_etype = quote(etype, safe="")
                serializable = [(int(s), int(t), d) for s, t, d in edge_list]
                conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                            (f"raw_edges:{safe_etype}", mjson.encode(serializable), "json"))

        # Always write secondary index field names (tiny)
        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("indexed_fields", mjson.encode(list(store._indexed_fields)), "json"))

        # Schema (always write - tiny)
        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("schema", mjson.encode(schema.to_dict()), "json"))

        # Vector index (2-tier: file-based with path ref, or inline blob for in-memory)
        vectors = getattr(store, 'vectors', None)
        if vectors is not None and vectors.count() > 0:
            vector_path = getattr(vectors, '_path', None)
            if vector_path:
                # Save to file, store marker in blobs
                vectors.save(vector_path)
                conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                             ("vector_index", b"FILE_BASED", "marker"))
            else:
                # In-memory mode: store as blob
                v_data = vectors.save()
                conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                             ("vector_index", v_data, "usearch"))

            pres = vectors._has_vector[:store._next_slot]
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         ("vector_presence", pres.tobytes(), str(pres.dtype)))
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         ("vector_dims", str(vectors.dims).encode(), "text"))
        else:
            conn.execute("DELETE FROM blobs WHERE key IN ('vector_index', 'vector_presence', 'vector_dims')")

        # Version + timestamp
        conn.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                     ("schema_version", str(SCHEMA_VERSION)))
        conn.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                     ("last_checkpoint_time", str(time.time())))

        # Clear WAL after checkpoint
        conn.execute("DELETE FROM wal")
