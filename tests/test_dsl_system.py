"""End-to-end tests for system DSL: DSL string -> parse -> SystemExecutor -> verify Result."""

import time
import pytest

from supergraph.core.store import CoreStore
from supergraph.core.schema import SchemaRegistry
from supergraph.core.runtime import RuntimeState
from supergraph.dsl.parser import parse
from supergraph.dsl.executor_system import SystemExecutor
from supergraph.persistence.database import open_database
from supergraph.core.errors import SuperGraphError


@pytest.fixture
def setup():
    store = CoreStore()
    store.put_node("fn_a", "function", {"name": "a"})
    store.put_node("fn_b", "function", {"name": "b"})
    store.put_node("cls_x", "class", {"name": "X"})
    store.put_edge("fn_a", "fn_b", "calls")

    schema = SchemaRegistry()
    return store, schema


@pytest.fixture
def setup_with_db(tmp_path, setup):
    store, schema = setup
    db_path = tmp_path / "test.db"
    conn = open_database(db_path)
    return store, schema, conn


def execute_sys(store, schema, query, conn=None):
    ast = parse(query)
    runtime = RuntimeState(store=store, schema=schema, conn=conn)
    executor = SystemExecutor(runtime)
    return executor.execute(ast)


# =============================================
# SYS STATS
# =============================================

class TestSysStats:
    def test_stats_all(self, setup):
        store, schema = setup
        r = execute_sys(store, schema, "SYS STATS")
        assert r.kind == "stats"
        assert r.data["node_count"] == 3
        assert r.data["edge_count"] == 1
        assert "memory_bytes" in r.data
        assert "ceiling_bytes" in r.data
        assert "wal_entries" in r.data
        assert "uptime_seconds" in r.data
        assert "edge_counts_by_type" in r.data
        assert r.data["edge_counts_by_type"]["calls"] == 1
        assert r.elapsed_us >= 0

    def test_stats_nodes(self, setup):
        store, schema = setup
        r = execute_sys(store, schema, "SYS STATS NODES")
        assert r.kind == "stats"
        assert r.data["node_count"] == 3
        assert "edge_count" not in r.data
        assert "memory_bytes" not in r.data

    def test_stats_edges(self, setup):
        store, schema = setup
        r = execute_sys(store, schema, "SYS STATS EDGES")
        assert r.kind == "stats"
        assert r.data["edge_count"] == 1
        assert "node_count" not in r.data
        assert "memory_bytes" not in r.data

    def test_stats_memory(self, setup):
        store, schema = setup
        r = execute_sys(store, schema, "SYS STATS MEMORY")
        assert r.kind == "stats"
        assert "memory_bytes" in r.data
        assert "ceiling_bytes" in r.data
        assert r.data["memory_bytes"] > 0
        assert "node_count" not in r.data

    def test_stats_wal_no_conn(self, setup):
        store, schema = setup
        r = execute_sys(store, schema, "SYS STATS WAL")
        assert r.data["wal_entries"] == 0

    def test_stats_wal_with_conn(self, setup_with_db):
        store, schema, conn = setup_with_db
        r = execute_sys(store, schema, "SYS STATS WAL", conn=conn)
        assert r.data["wal_entries"] == 0


# =============================================
# SYS KINDS
# =============================================

class TestSysKinds:
    def test_kinds_empty(self, setup):
        store, schema = setup
        r = execute_sys(store, schema, "SYS KINDS")
        assert r.kind == "schema"
        assert r.data == []
        assert r.count == 0

    def test_kinds_after_register(self, setup):
        store, schema = setup
        execute_sys(
            store, schema,
            'SYS REGISTER NODE KIND "function" REQUIRED name OPTIONAL file, line',
        )
        r = execute_sys(store, schema, "SYS KINDS")
        assert r.kind == "schema"
        assert "function" in r.data
        assert r.count == 1


# =============================================
# SYS EDGE KINDS
# =============================================

class TestSysEdgeKinds:
    def test_edge_kinds_after_register(self, setup):
        store, schema = setup
        execute_sys(
            store, schema,
            'SYS REGISTER EDGE KIND "calls" FROM "function" TO "function"',
        )
        r = execute_sys(store, schema, "SYS EDGE KINDS")
        assert r.kind == "schema"
        assert "calls" in r.data
        assert r.count == 1


# =============================================
# SYS DESCRIBE
# =============================================

class TestSysDescribe:
    def test_describe_node_kind(self, setup):
        store, schema = setup
        execute_sys(
            store, schema,
            'SYS REGISTER NODE KIND "function" REQUIRED name OPTIONAL file, line',
        )
        r = execute_sys(store, schema, 'SYS DESCRIBE NODE "function"')
        assert r.kind == "schema"
        assert r.count == 1
        assert r.data["kind"] == "function"
        assert "name" in r.data["required"]
        assert "file" in r.data["optional"]
        assert "line" in r.data["optional"]

    def test_describe_unknown_node(self, setup):
        store, schema = setup
        r = execute_sys(store, schema, 'SYS DESCRIBE NODE "unknown"')
        assert r.kind == "schema"
        assert r.data is None
        assert r.count == 0


# =============================================
# SYS REGISTER / UNREGISTER
# =============================================

class TestSysRegister:
    def test_register_node_kind(self, setup):
        store, schema = setup
        r = execute_sys(
            store, schema,
            'SYS REGISTER NODE KIND "function" REQUIRED name OPTIONAL file, line',
        )
        assert r.kind == "ok"
        assert schema.list_node_kinds() == ["function"]
        defn = schema.describe_node_kind("function")
        assert "name" in defn["required"]
        assert "file" in defn["optional"]
        assert "line" in defn["optional"]

    def test_register_edge_kind(self, setup):
        store, schema = setup
        r = execute_sys(
            store, schema,
            'SYS REGISTER EDGE KIND "calls" FROM "function" TO "function"',
        )
        assert r.kind == "ok"
        assert schema.list_edge_kinds() == ["calls"]
        defn = schema.describe_edge_kind("calls")
        assert "function" in defn["from_kinds"]
        assert "function" in defn["to_kinds"]

    def test_unregister_node_kind(self, setup):
        store, schema = setup
        execute_sys(
            store, schema,
            'SYS REGISTER NODE KIND "function" REQUIRED name',
        )
        assert schema.list_node_kinds() == ["function"]
        r = execute_sys(store, schema, 'SYS UNREGISTER NODE KIND "function"')
        assert r.kind == "ok"
        assert schema.list_node_kinds() == []


# =============================================
# SYS CHECKPOINT / REBUILD / CLEAR
# =============================================

class TestSysMaintenance:
    def test_checkpoint(self, setup):
        store, schema = setup
        r = execute_sys(store, schema, "SYS CHECKPOINT")
        assert r.kind == "ok"

    def test_rebuild_indices(self, setup):
        store, schema = setup
        store.add_index("name")
        r = execute_sys(store, schema, "SYS REBUILD INDICES")
        assert r.kind == "ok"
        # After rebuild, index should still work
        slots = store.query_by_index("name", "a")
        assert len(slots) == 1

    def test_clear_cache(self, setup):
        store, schema = setup
        r = execute_sys(store, schema, "SYS CLEAR CACHE")
        assert r.kind == "ok"

    def test_clear_log(self, setup_with_db):
        store, schema, conn = setup_with_db
        # Insert a log entry
        conn.execute(
            "INSERT INTO query_log (timestamp, query, elapsed_us, result_count) "
            "VALUES (?, ?, ?, ?)",
            (time.time(), 'NODE "x"', 100, 1),
        )
        conn.commit()
        row = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()
        assert row[0] == 1

        r = execute_sys(store, schema, "SYS CLEAR LOG", conn=conn)
        assert r.kind == "ok"
        row = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()
        assert row[0] == 0


# =============================================
# SYS WAL
# =============================================

class TestSysWal:
    def test_wal_status_no_conn(self, setup):
        store, schema = setup
        r = execute_sys(store, schema, "SYS WAL STATUS")
        assert r.kind == "stats"
        assert r.data["wal_entries"] == 0
        assert r.data["wal_bytes"] == 0

    def test_wal_status_with_conn(self, setup_with_db):
        store, schema, conn = setup_with_db
        # Insert a WAL entry
        conn.execute(
            "INSERT INTO wal (timestamp, statement) VALUES (?, ?)",
            (time.time(), 'CREATE NODE "x" kind = "test"'),
        )
        conn.commit()

        r = execute_sys(store, schema, "SYS WAL STATUS", conn=conn)
        assert r.kind == "stats"
        assert r.data["wal_entries"] == 1
        assert r.data["wal_bytes"] > 0

    def test_wal_replay(self, setup):
        store, schema = setup
        r = execute_sys(store, schema, "SYS WAL REPLAY")
        assert r.kind == "ok"


# =============================================
# SYS EXPLAIN
# =============================================

class TestSysExplain:
    def test_explain_traverse(self, setup):
        store, schema = setup
        r = execute_sys(
            store, schema,
            'SYS EXPLAIN TRAVERSE FROM "fn_a" DEPTH 3 WHERE kind = "calls"',
        )
        assert r.kind == "plan"
        assert "estimated_frontier" in r.data
        assert "hops" in r.data
        assert r.data["rejected"] is False

    def test_explain_nodes(self, setup):
        store, schema = setup
        r = execute_sys(store, schema, 'SYS EXPLAIN NODES WHERE name = "a"')
        assert r.kind == "plan"
        assert r.data["type"] == "scan"
        assert r.data["estimated_nodes"] == 3

    def test_explain_nodes_with_index(self, setup):
        store, schema = setup
        store.add_index("name")
        r = execute_sys(store, schema, 'SYS EXPLAIN NODES WHERE name = "a"')
        assert r.kind == "plan"
        assert r.data["type"] == "index_lookup"


# =============================================
# Query log operations (SLOW / FREQUENT / FAILED)
# =============================================

class TestQueryLog:
    def _insert_log_entries(self, conn):
        """Insert sample query log entries for testing."""
        now = time.time()
        entries = [
            (now - 100, 'NODE "fn_a"', 500, 1, None),
            (now - 90, 'NODE "fn_b"', 2000, 1, None),
            (now - 80, 'NODES', 1500, 5, None),
            (now - 70, 'NODE "fn_a"', 300, 1, None),
            (now - 60, 'NODES', 800, 3, None),
            (now - 50, 'NODE "bad"', 100, 0, "NodeNotFound: bad"),
            (now - 40, 'EDGES FROM "fn_a"', 3000, 2, None),
            (now - 30, 'NODE "oops"', 50, 0, "NodeNotFound: oops"),
        ]
        conn.executemany(
            "INSERT INTO query_log (timestamp, query, elapsed_us, result_count, error) "
            "VALUES (?, ?, ?, ?, ?)",
            entries,
        )
        conn.commit()

    def test_slow_queries(self, setup_with_db):
        store, schema, conn = setup_with_db
        self._insert_log_entries(conn)
        r = execute_sys(store, schema, "SYS SLOW QUERIES LIMIT 3", conn=conn)
        assert r.kind == "log_entries"
        assert r.count == 3
        # Should be ordered by elapsed_us DESC
        assert r.data[0]["elapsed_us"] >= r.data[1]["elapsed_us"]
        assert r.data[1]["elapsed_us"] >= r.data[2]["elapsed_us"]

    def test_slow_queries_no_conn(self, setup):
        store, schema = setup
        r = execute_sys(store, schema, "SYS SLOW QUERIES LIMIT 5")
        assert r.kind == "log_entries"
        assert r.data == []
        assert r.count == 0

    def test_frequent_queries(self, setup_with_db):
        store, schema, conn = setup_with_db
        self._insert_log_entries(conn)
        r = execute_sys(store, schema, "SYS FREQUENT QUERIES LIMIT 2", conn=conn)
        assert r.kind == "log_entries"
        assert r.count == 2
        # Most frequent should be first
        assert r.data[0]["count"] >= r.data[1]["count"]

    def test_frequent_queries_no_conn(self, setup):
        store, schema = setup
        r = execute_sys(store, schema, "SYS FREQUENT QUERIES LIMIT 5")
        assert r.kind == "log_entries"
        assert r.data == []

    def test_failed_queries(self, setup_with_db):
        store, schema, conn = setup_with_db
        self._insert_log_entries(conn)
        r = execute_sys(store, schema, "SYS FAILED QUERIES LIMIT 5", conn=conn)
        assert r.kind == "log_entries"
        assert r.count == 2
        # All entries should have an error
        for entry in r.data:
            assert entry["error"] is not None

    def test_failed_queries_no_conn(self, setup):
        store, schema = setup
        r = execute_sys(store, schema, "SYS FAILED QUERIES LIMIT 5")
        assert r.kind == "log_entries"
        assert r.data == []


# =============================================
# Unknown command
# =============================================

class TestUnknownCommand:
    def test_unknown_raises(self, setup):
        store, schema = setup
        executor = SystemExecutor(RuntimeState(store=store, schema=schema))
        with pytest.raises(SuperGraphError, match="Unknown system command"):
            executor.execute("not an AST node")


def test_executor_base_split_integrity():
    """All ExecutorBase helpers must be accessible from Executor after split."""
    from supergraph.dsl.executor import Executor
    from supergraph.core.store import CoreStore
    from supergraph.core.schema import SchemaRegistry
    from supergraph.core.runtime import RuntimeState
    store = CoreStore()
    schema = SchemaRegistry()
    ex = Executor(RuntimeState(store=store, schema=schema))
    # VisibilityMixin
    assert hasattr(ex, '_compute_live_mask')
    assert hasattr(ex, '_resolve_slot')
    assert hasattr(ex, '_is_slot_visible')
    assert hasattr(ex, '_apply_ttl')
    # FilteringMixin
    assert hasattr(ex, '_eval_where')
    assert hasattr(ex, '_try_column_filter')
    assert hasattr(ex, '_try_column_nodes')
    assert hasattr(ex, '_try_index_lookup')
