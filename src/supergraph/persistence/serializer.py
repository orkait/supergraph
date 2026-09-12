
import time
from urllib.parse import quote

import msgspec.json as mjson

from supergraph.persistence.database import SCHEMA_VERSION
from supergraph.core.store import CoreStore
from supergraph.core.schema import SchemaRegistry


def checkpoint(store: CoreStore, schema: SchemaRegistry, conn, force: bool = False):
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
            conn.execute("DELETE FROM blobs WHERE key LIKE 'edges:%'")
            conn.execute("DELETE FROM blobs WHERE key LIKE 'edge_data:%'")
            conn.execute("DELETE FROM blobs WHERE key LIKE 'raw_edges:%'")

            for etype, edge_list in store._edges_by_type.items():
                safe_etype = quote(etype, safe="")
                serializable = [(int(s), int(t), d) for s, t, d in edge_list]
                conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                            (f"raw_edges:{safe_etype}", mjson.encode(serializable), "json"))

        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("indexed_fields", mjson.encode(list(store._indexed_fields)), "json"))

        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("schema", mjson.encode(schema.to_dict()), "json"))

        vectors = getattr(store, 'vectors', None)
        if vectors is not None and vectors.count() > 0:
            vector_path = getattr(vectors, '_path', None)
            if vector_path:
                vectors.save(vector_path)
                conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                             ("vector_index", b"FILE_BASED", "marker"))
            else:
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

        conn.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                     ("schema_version", str(SCHEMA_VERSION)))
        conn.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                     ("last_checkpoint_time", str(time.time())))

        conn.execute("DELETE FROM wal")
