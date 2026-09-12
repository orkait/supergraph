
import time

import pytest

from supergraph.core.errors import VersionMismatch
from supergraph.persistence.database import (
    SCHEMA_VERSION,
    get_metadata,
    open_database,
    set_metadata,
)
from supergraph.persistence.deserializer import load
from supergraph.persistence.serializer import checkpoint
from supergraph.core.schema import SchemaRegistry
from supergraph.core.store import CoreStore


def _populated_store():
    s = CoreStore()
    s.put_node("alice", "person", {"age": 30, "city": "NYC"})
    s.put_node("bob", "person", {"age": 25, "city": "LA"})
    s.put_node("acme", "org", {"industry": "tech"})
    s.put_edge("alice", "bob", "knows")
    s.put_edge("alice", "acme", "works_at")
    return s


def _populated_schema():
    schema = SchemaRegistry()
    schema.register_node_kind("person", required=["age"], optional=["city"])
    schema.register_node_kind("org", required=["industry"])
    schema.register_edge_kind("knows", from_kinds=["person"], to_kinds=["person"])
    schema.register_edge_kind("works_at", from_kinds=["person"], to_kinds=["org"])
    return schema


class TestOpenDatabase:
    def test_creates_tables(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "blobs" in tables
        assert "wal" in tables
        assert "query_log" in tables
        assert "metadata" in tables
        conn.close()

    def test_idempotent_open(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn1 = open_database(db_path)
        set_metadata(conn1, "hello", "world")
        conn1.close()

        conn2 = open_database(db_path)
        assert get_metadata(conn2, "hello") == "world"
        conn2.close()

    def test_wal_journal_mode(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()


class TestMetadata:
    def test_set_and_get(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        set_metadata(conn, "version", "42")
        assert get_metadata(conn, "version") == "42"
        conn.close()

    def test_get_missing_returns_none(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        assert get_metadata(conn, "nonexistent") is None
        conn.close()

    def test_overwrite(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        set_metadata(conn, "key", "v1")
        set_metadata(conn, "key", "v2")
        assert get_metadata(conn, "key") == "v2"
        conn.close()

    def test_multiple_keys(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        set_metadata(conn, "a", "1")
        set_metadata(conn, "b", "2")
        assert get_metadata(conn, "a") == "1"
        assert get_metadata(conn, "b") == "2"
        conn.close()


class TestRoundTripEmpty:
    def test_empty_store(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        store = CoreStore()
        schema = SchemaRegistry()
        checkpoint(store, schema, conn)

        loaded_store, loaded_schema = load(conn)
        assert loaded_store.node_count == 0
        assert loaded_store.edge_count == 0
        assert loaded_schema.list_node_kinds() == []
        assert loaded_schema.list_edge_kinds() == []
        conn.close()


class TestRoundTripNodesOnly:
    def test_nodes_preserved(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        store = CoreStore()
        store.put_node("alice", "person", {"age": 30, "city": "NYC"})
        store.put_node("bob", "person", {"age": 25, "city": "LA"})
        store.put_node("acme", "org", {"industry": "tech"})

        checkpoint(store, SchemaRegistry(), conn)
        loaded, _ = load(conn)

        assert loaded.node_count == 3
        alice = loaded.get_node("alice")
        assert alice is not None
        assert alice["kind"] == "person"
        assert alice["age"] == 30
        assert alice["city"] == "NYC"

        bob = loaded.get_node("bob")
        assert bob is not None
        assert bob["age"] == 25

        acme = loaded.get_node("acme")
        assert acme is not None
        assert acme["kind"] == "org"
        assert acme["industry"] == "tech"
        conn.close()

    def test_node_ids_consistent(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        store = CoreStore()
        store.put_node("x", "t", {"v": 1})
        store.put_node("y", "t", {"v": 2})

        checkpoint(store, SchemaRegistry(), conn)
        loaded, _ = load(conn)

        assert loaded.get_node("x")["v"] == 1
        assert loaded.get_node("y")["v"] == 2
        assert loaded.get_node("nonexistent") is None
        conn.close()


class TestRoundTripWithEdges:
    def test_edges_preserved(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        store = _populated_store()

        checkpoint(store, SchemaRegistry(), conn)
        loaded, _ = load(conn)

        assert loaded.node_count == 3
        assert loaded.edge_count == 2

        edges_from_alice = loaded.get_edges_from("alice")
        targets = {e["target"] for e in edges_from_alice}
        assert targets == {"bob", "acme"}

        edges_to_bob = loaded.get_edges_to("bob")
        assert len(edges_to_bob) == 1
        assert edges_to_bob[0]["source"] == "alice"
        assert edges_to_bob[0]["kind"] == "knows"
        conn.close()

    def test_edge_kinds_preserved(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        store = _populated_store()

        checkpoint(store, SchemaRegistry(), conn)
        loaded, _ = load(conn)

        knows_edges = loaded.get_edges_from("alice", kind="knows")
        assert len(knows_edges) == 1
        assert knows_edges[0]["target"] == "bob"

        works_edges = loaded.get_edges_from("alice", kind="works_at")
        assert len(works_edges) == 1
        assert works_edges[0]["target"] == "acme"
        conn.close()

    def test_no_edges_for_unconnected_node(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        store = _populated_store()

        checkpoint(store, SchemaRegistry(), conn)
        loaded, _ = load(conn)

        edges_from_bob = loaded.get_edges_from("bob")
        assert edges_from_bob == []
        conn.close()


class TestRoundTripWithIndices:
    def test_indices_rebuilt(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        store = CoreStore()
        store.put_node("a", "person", {"city": "NYC"})
        store.put_node("b", "person", {"city": "LA"})
        store.put_node("c", "person", {"city": "NYC"})
        store.add_index("city")

        checkpoint(store, SchemaRegistry(), conn)
        loaded, _ = load(conn)

        nyc_slots = loaded.query_by_index("city", "NYC")
        assert len(nyc_slots) == 2
        la_slots = loaded.query_by_index("city", "LA")
        assert len(la_slots) == 1
        conn.close()

    def test_multiple_indices(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        store = CoreStore()
        store.put_node("a", "person", {"city": "NYC", "age": 30})
        store.put_node("b", "person", {"city": "LA", "age": 25})
        store.add_index("city")
        store.add_index("age")

        checkpoint(store, SchemaRegistry(), conn)
        loaded, _ = load(conn)

        assert len(loaded.query_by_index("city", "NYC")) == 1
        assert len(loaded.query_by_index("age", 30)) == 1
        conn.close()


class TestRoundTripWithTombstones:
    def test_tombstones_preserved(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        store = CoreStore()
        store.put_node("a", "t", {"v": 1})
        store.put_node("b", "t", {"v": 2})
        store.put_node("c", "t", {"v": 3})
        store.delete_node("b")

        checkpoint(store, SchemaRegistry(), conn)
        loaded, _ = load(conn)

        assert loaded.node_count == 2
        assert loaded.get_node("a") is not None
        assert loaded.get_node("b") is None
        assert loaded.get_node("c") is not None
        conn.close()

    def test_tombstoned_edges_not_restored(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        store = _populated_store()
        store.delete_node("bob")

        checkpoint(store, SchemaRegistry(), conn)
        loaded, _ = load(conn)

        assert loaded.node_count == 2
        assert loaded.edge_count == 1
        edges = loaded.get_edges_from("alice")
        assert len(edges) == 1
        assert edges[0]["target"] == "acme"
        conn.close()


class TestRoundTripWithSchema:
    def test_schema_preserved(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        store = _populated_store()
        schema = _populated_schema()

        checkpoint(store, schema, conn)
        _, loaded_schema = load(conn)

        assert sorted(loaded_schema.list_node_kinds()) == ["org", "person"]
        assert sorted(loaded_schema.list_edge_kinds()) == ["knows", "works_at"]

        person_def = loaded_schema.describe_node_kind("person")
        assert "age" in person_def["required"]
        assert "city" in person_def["optional"]

        knows_def = loaded_schema.describe_edge_kind("knows")
        assert "person" in knows_def["from_kinds"]
        assert "person" in knows_def["to_kinds"]
        conn.close()

    def test_empty_schema_preserved(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        store = CoreStore()
        schema = SchemaRegistry()

        checkpoint(store, schema, conn)
        _, loaded_schema = load(conn)

        assert loaded_schema.list_node_kinds() == []
        assert loaded_schema.list_edge_kinds() == []
        conn.close()


class TestRoundTripLargeStore:
    def test_100_plus_nodes(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        store = CoreStore()
        n = 150
        for i in range(n):
            store.put_node(f"node_{i}", "thing", {"index": i, "label": f"label_{i}"})
        for i in range(n - 1):
            store.put_edge(f"node_{i}", f"node_{i+1}", "next")

        checkpoint(store, SchemaRegistry(), conn)
        loaded, _ = load(conn)

        assert loaded.node_count == n
        assert loaded.edge_count == n - 1

        for i in [0, 50, 99, 149]:
            node = loaded.get_node(f"node_{i}")
            assert node is not None
            assert node["index"] == i
            assert node["label"] == f"label_{i}"

        edges = loaded.get_edges_from("node_0")
        assert len(edges) == 1
        assert edges[0]["target"] == "node_1"

        edges = loaded.get_edges_from("node_149")
        assert edges == []
        conn.close()


class TestWALBehavior:
    def test_wal_entries_added(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        now = time.time()
        conn.execute(
            "INSERT INTO wal (timestamp, statement) VALUES (?, ?)",
            (now, "CREATE node alice"),
        )
        conn.execute(
            "INSERT INTO wal (timestamp, statement) VALUES (?, ?)",
            (now + 1, "CREATE node bob"),
        )
        conn.commit()

        rows = conn.execute("SELECT * FROM wal ORDER BY seq").fetchall()
        assert len(rows) == 2
        assert rows[0][2] == "CREATE node alice"
        assert rows[1][2] == "CREATE node bob"
        conn.close()

    def test_wal_cleared_after_checkpoint(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        now = time.time()
        conn.execute(
            "INSERT INTO wal (timestamp, statement) VALUES (?, ?)",
            (now, "CREATE node alice"),
        )
        conn.commit()

        rows = conn.execute("SELECT * FROM wal").fetchall()
        assert len(rows) == 1

        store = CoreStore()
        store.put_node("alice", "person", {"age": 30})
        checkpoint(store, SchemaRegistry(), conn)

        rows = conn.execute("SELECT * FROM wal").fetchall()
        assert len(rows) == 0
        conn.close()

    def test_wal_entries_survive_without_checkpoint(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = open_database(db_path)
        now = time.time()
        conn.execute(
            "INSERT INTO wal (timestamp, statement) VALUES (?, ?)",
            (now, "CREATE node alice"),
        )
        conn.commit()
        conn.close()

        conn2 = open_database(db_path)
        rows = conn2.execute("SELECT * FROM wal").fetchall()
        assert len(rows) == 1
        assert rows[0][2] == "CREATE node alice"
        conn2.close()

    def test_wal_autoincrement(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        now = time.time()
        for i in range(5):
            conn.execute(
                "INSERT INTO wal (timestamp, statement) VALUES (?, ?)",
                (now + i, f"stmt_{i}"),
            )
        conn.commit()

        rows = conn.execute("SELECT seq FROM wal ORDER BY seq").fetchall()
        seqs = [r[0] for r in rows]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == 5
        conn.close()


class TestVersionCheck:
    def test_version_mismatch_raises(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        store = CoreStore()
        store.put_node("x", "t", {})
        checkpoint(store, SchemaRegistry(), conn)

        conn.execute(
            "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
            ("schema_version", "999"),
        )
        conn.commit()

        with pytest.raises(VersionMismatch) as exc_info:
            load(conn)
        assert exc_info.value.expected == SCHEMA_VERSION
        conn.close()

    def test_fresh_database_returns_empty(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        store, schema = load(conn)
        assert store.node_count == 0
        assert schema.list_node_kinds() == []
        conn.close()

    def test_correct_version_loads_fine(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        store = CoreStore()
        store.put_node("a", "t", {"v": 1})
        checkpoint(store, SchemaRegistry(), conn)

        loaded, _ = load(conn)
        assert loaded.get_node("a")["v"] == 1
        conn.close()


class TestQueryLog:
    def test_insert_log_entry(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        now = time.time()
        conn.execute(
            "INSERT INTO query_log (timestamp, query, elapsed_us, result_count, error) "
            "VALUES (?, ?, ?, ?, ?)",
            (now, "MATCH (n:person)", 150, 3, None),
        )
        conn.commit()

        rows = conn.execute("SELECT * FROM query_log").fetchall()
        assert len(rows) == 1
        assert rows[0][2] == "MATCH (n:person)"
        assert rows[0][3] == 150
        assert rows[0][4] == 3
        assert rows[0][5] is None
        conn.close()

    def test_insert_log_entry_with_error(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        now = time.time()
        conn.execute(
            "INSERT INTO query_log (timestamp, query, elapsed_us, result_count, error) "
            "VALUES (?, ?, ?, ?, ?)",
            (now, "BAD QUERY", 10, 0, "Parse error"),
        )
        conn.commit()

        rows = conn.execute("SELECT * FROM query_log").fetchall()
        assert len(rows) == 1
        assert rows[0][5] == "Parse error"
        conn.close()

    def test_query_log_by_timestamp(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        t1 = 1000.0
        t2 = 2000.0
        t3 = 3000.0
        conn.execute(
            "INSERT INTO query_log (timestamp, query, elapsed_us, result_count, error) "
            "VALUES (?, ?, ?, ?, ?)",
            (t1, "query_1", 100, 1, None),
        )
        conn.execute(
            "INSERT INTO query_log (timestamp, query, elapsed_us, result_count, error) "
            "VALUES (?, ?, ?, ?, ?)",
            (t2, "query_2", 200, 2, None),
        )
        conn.execute(
            "INSERT INTO query_log (timestamp, query, elapsed_us, result_count, error) "
            "VALUES (?, ?, ?, ?, ?)",
            (t3, "query_3", 300, 3, None),
        )
        conn.commit()

        rows = conn.execute(
            "SELECT query FROM query_log WHERE timestamp > ? ORDER BY timestamp",
            (t1,),
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][0] == "query_2"
        assert rows[1][0] == "query_3"
        conn.close()

    def test_query_log_by_elapsed(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        now = time.time()
        conn.execute(
            "INSERT INTO query_log (timestamp, query, elapsed_us, result_count, error) "
            "VALUES (?, ?, ?, ?, ?)",
            (now, "fast", 50, 1, None),
        )
        conn.execute(
            "INSERT INTO query_log (timestamp, query, elapsed_us, result_count, error) "
            "VALUES (?, ?, ?, ?, ?)",
            (now, "slow", 5000, 10, None),
        )
        conn.commit()

        rows = conn.execute(
            "SELECT query FROM query_log WHERE elapsed_us > 1000"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "slow"
        conn.close()

    def test_query_log_by_error(self, tmp_path):
        conn = open_database(tmp_path / "test.db")
        now = time.time()
        conn.execute(
            "INSERT INTO query_log (timestamp, query, elapsed_us, result_count, error) "
            "VALUES (?, ?, ?, ?, ?)",
            (now, "good_query", 100, 5, None),
        )
        conn.execute(
            "INSERT INTO query_log (timestamp, query, elapsed_us, result_count, error) "
            "VALUES (?, ?, ?, ?, ?)",
            (now, "bad_query", 10, 0, "SyntaxError"),
        )
        conn.commit()

        rows = conn.execute(
            "SELECT query FROM query_log WHERE error IS NOT NULL"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "bad_query"
        conn.close()


class TestEndToEnd:
    def test_checkpoint_then_modify_then_reload_original(self, tmp_path):
        db_path = tmp_path / "test.db"

        conn1 = open_database(db_path)
        store = _populated_store()
        schema = _populated_schema()
        checkpoint(store, schema, conn1)
        conn1.close()

        store.put_node("dave", "person", {"age": 40})
        store.delete_node("bob")
        assert store.node_count == 3

        conn2 = open_database(db_path)
        loaded, loaded_schema = load(conn2)
        conn2.close()

        assert loaded.node_count == 3
        assert loaded.get_node("alice") is not None
        assert loaded.get_node("bob") is not None
        assert loaded.get_node("acme") is not None
        assert loaded.get_node("dave") is None

    def test_loaded_store_supports_queries(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = open_database(db_path)

        store = _populated_store()
        store.add_index("city")
        checkpoint(store, SchemaRegistry(), conn)
        conn.close()

        conn2 = open_database(db_path)
        loaded, _ = load(conn2)
        conn2.close()

        all_nodes = loaded.get_all_nodes()
        assert len(all_nodes) == 3

        persons = loaded.get_all_nodes(kind="person")
        assert len(persons) == 2

        nyc_slots = loaded.query_by_index("city", "NYC")
        assert len(nyc_slots) == 1

        edges = loaded.get_edges_from("alice")
        assert len(edges) == 2

    def test_loaded_store_supports_mutations(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = open_database(db_path)

        store = _populated_store()
        checkpoint(store, SchemaRegistry(), conn)
        conn.close()

        conn2 = open_database(db_path)
        loaded, schema = load(conn2)

        loaded.put_node("dave", "person", {"age": 40, "city": "SF"})
        assert loaded.node_count == 4
        assert loaded.get_node("dave")["age"] == 40

        loaded.put_edge("dave", "alice", "knows")
        assert loaded.edge_count == 3

        edges_to_alice = loaded.get_edges_to("alice")
        sources = {e["source"] for e in edges_to_alice}
        assert "dave" in sources

        checkpoint(loaded, schema, conn2)

        loaded2, _ = load(conn2)
        assert loaded2.node_count == 4
        assert loaded2.get_node("dave") is not None
        assert loaded2.edge_count == 3
        conn2.close()

    def test_multiple_checkpoints(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = open_database(db_path)

        store = CoreStore()
        store.put_node("alice", "person", {"age": 30})
        checkpoint(store, SchemaRegistry(), conn)

        loaded1, _ = load(conn)
        assert loaded1.node_count == 1

        store.put_node("bob", "person", {"age": 25})
        store.put_edge("alice", "bob", "knows")
        checkpoint(store, SchemaRegistry(), conn)

        loaded2, _ = load(conn)
        assert loaded2.node_count == 2
        assert loaded2.edge_count == 1
        assert loaded2.get_node("bob") is not None
        conn.close()

    def test_checkpoint_clears_stale_edge_blobs(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = open_database(db_path)

        store = _populated_store()
        checkpoint(store, SchemaRegistry(), conn)

        store.delete_edge("alice", "bob", "knows")
        checkpoint(store, SchemaRegistry(), conn)

        loaded, _ = load(conn)
        assert loaded.edge_count == 1
        knows_edges = loaded.get_edges_from("alice", kind="knows")
        assert knows_edges == []
        conn.close()

    def test_update_node_then_checkpoint(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = open_database(db_path)

        store = CoreStore()
        store.put_node("alice", "person", {"age": 30})
        checkpoint(store, SchemaRegistry(), conn)

        store.update_node("alice", {"age": 31, "promoted": True})
        checkpoint(store, SchemaRegistry(), conn)

        loaded, _ = load(conn)
        alice = loaded.get_node("alice")
        assert alice["age"] == 31
        assert alice["promoted"] == 1
        conn.close()

    def test_delete_node_then_checkpoint(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = open_database(db_path)

        store = _populated_store()
        checkpoint(store, SchemaRegistry(), conn)

        store.delete_node("bob")
        checkpoint(store, SchemaRegistry(), conn)

        loaded, _ = load(conn)
        assert loaded.node_count == 2
        assert loaded.get_node("bob") is None
        assert loaded.edge_count == 1
        conn.close()


class TestColumnPersistence:
    def test_checkpoint_and_load_preserves_columns(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = open_database(db_path)

        store = CoreStore()
        schema = SchemaRegistry()
        store.put_node("n1", "fn", {"score": 42, "name": "main"})
        store.put_node("n2", "fn", {"score": 99, "name": "helper"})

        checkpoint(store, schema, conn)
        store2, schema2 = load(conn)

        assert store2.columns.has_column("score")
        assert store2.columns.has_column("name")
        mask = store2.columns.get_mask("score", "=", 42, store2._next_slot)
        assert mask[0] and not mask[1]
        mask2 = store2.columns.get_mask("name", "=", "main", store2._next_slot)
        assert mask2[0] and not mask2[1]
        conn.close()

    def test_backward_compat_no_column_data(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = open_database(db_path)

        store = CoreStore()
        schema = SchemaRegistry()
        store.put_node("n1", "fn", {"score": 42})

        checkpoint(store, schema, conn)
        conn.execute("DELETE FROM blobs WHERE key LIKE 'columns:%'")
        conn.commit()

        store2, schema2 = load(conn)
        assert not store2.columns.has_column("score")
        store2.put_node("n2", "fn", {"score": 99})
        assert store2.columns.has_column("score")
        conn.close()

    def test_column_dtypes_preserved(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = open_database(db_path)

        store = CoreStore()
        schema = SchemaRegistry()
        store.put_node("n1", "fn", {"line": 42, "weight": 3.14, "name": "x"})

        checkpoint(store, schema, conn)
        store2, _ = load(conn)

        assert store2.columns._dtypes["line"] == "int64"
        assert store2.columns._dtypes["weight"] == "float64"
        assert store2.columns._dtypes["name"] == "int32_interned"
        conn.close()


def test_wal_manager_is_wired(tmp_path):
    from supergraph import SuperGraph
    gs = SuperGraph(path=str(tmp_path))
    assert hasattr(gs, '_wal'), "SuperGraph must have _wal attribute (WALManager)"
    from supergraph.wal import WALManager
    assert isinstance(gs._wal, WALManager)
    gs.close()


def test_wal_manager_no_inline_methods(tmp_path):
    from supergraph import SuperGraph
    gs = SuperGraph(path=str(tmp_path))
    assert not hasattr(gs, '_wal_append'), "inline _wal_append must be deleted"
    assert not hasattr(gs, '_replay_wal'), "inline _replay_wal must be deleted"
    assert not hasattr(gs, '_maybe_auto_checkpoint'), "inline _maybe_auto_checkpoint must be deleted"
    assert not hasattr(gs, '_rotate_query_log'), "inline _rotate_query_log must be deleted"
    assert not hasattr(gs, '_log_query'), "inline _log_query must be deleted"
    gs.close()


def test_supergraph_has_public_api():
    from supergraph import SuperGraph
    gs = SuperGraph()
    assert hasattr(gs, 'get_all_nodes'), "missing get_all_nodes()"
    assert hasattr(gs, 'get_all_edges'), "missing get_all_edges()"
    assert hasattr(SuperGraph, 'cost_threshold'), "missing cost_threshold property"
    assert hasattr(SuperGraph, 'ceiling_mb'), "missing ceiling_mb property"


def test_cost_threshold_property():
    from supergraph import SuperGraph
    gs = SuperGraph()
    original = gs.cost_threshold
    gs.cost_threshold = 50_000
    assert gs.cost_threshold == 50_000
    gs.cost_threshold = original


def test_ceiling_mb_property():
    from supergraph import SuperGraph
    gs = SuperGraph()
    gs.ceiling_mb = 512
    assert gs.ceiling_mb == 512
