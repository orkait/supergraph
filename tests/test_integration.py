"""Integration tests for the SuperGraph public API.

Tests exercise the full stack: DSL parsing, execution, persistence,
and error handling through the SuperGraph facade.
"""

import pytest

from supergraph import SuperGraph
from supergraph.core.errors import BatchRollback, CeilingExceeded


# ── Test 1: Full workflow - schema, bulk load, query, mutate, verify ──


def test_full_workflow(tmp_path):
    with SuperGraph(path=str(tmp_path / "db")) as g:
        # Register schema
        g.execute('SYS REGISTER NODE KIND "function" REQUIRED name OPTIONAL file, line')
        g.execute('SYS REGISTER NODE KIND "class" REQUIRED name OPTIONAL file')
        g.execute('SYS REGISTER EDGE KIND "calls" FROM "function" TO "function"')

        # Bulk load
        g.execute('CREATE NODE "fn_main" kind = "function" name = "main" file = "main.py"')
        g.execute('CREATE NODE "fn_helper" kind = "function" name = "helper" file = "utils.py"')
        g.execute('CREATE NODE "fn_parse" kind = "function" name = "parse" file = "parser.py"')
        g.execute('CREATE NODE "cls_app" kind = "class" name = "App" file = "main.py"')

        g.execute('CREATE EDGE "fn_main" -> "fn_helper" kind = "calls"')
        g.execute('CREATE EDGE "fn_main" -> "fn_parse" kind = "calls"')
        g.execute('CREATE EDGE "fn_helper" -> "fn_parse" kind = "calls"')

        # Verify counts
        assert g.node_count == 4
        assert g.edge_count == 3

        # Query single node
        r = g.execute('NODE "fn_main"')
        assert r.data["name"] == "main"

        # Filter nodes by kind
        r = g.execute('NODES WHERE kind = "function"')
        assert r.count == 3

        # Outgoing edges
        r = g.execute('EDGES FROM "fn_main" WHERE kind = "calls"')
        assert r.count == 2

        # Shortest path (fn_main -> fn_parse is a direct edge)
        r = g.execute('SHORTEST PATH FROM "fn_main" TO "fn_parse" WHERE kind = "calls"')
        assert r.data is not None
        assert len(r.data) == 2  # direct edge: [fn_main, fn_parse]

        # Ancestors of fn_parse (fn_main and fn_helper both call it)
        r = g.execute('ANCESTORS OF "fn_parse" DEPTH 2 WHERE kind = "calls"')
        assert r.count >= 2  # fn_main and fn_helper

        # Mutate: update a field
        g.execute('UPDATE NODE "fn_main" SET line = 1')
        r = g.execute('NODE "fn_main"')
        assert r.data["line"] == 1

        # Delete a node
        g.execute('DELETE NODE "cls_app"')
        assert g.node_count == 3

        # Stats
        r = g.execute('SYS STATS')
        assert r.data["node_count"] == 3


# ── Test 2: Persistence round-trip ────────────────────────────────────


def test_persistence_roundtrip(tmp_path):
    db_path = str(tmp_path / "db")

    # Create and populate
    with SuperGraph(path=db_path) as g:
        g.execute('CREATE NODE "a" kind = "x" name = "alpha"')
        g.execute('CREATE NODE "b" kind = "x" name = "beta"')
        g.execute('CREATE EDGE "a" -> "b" kind = "link"')
        g.checkpoint()

    # Reload and verify
    with SuperGraph(path=db_path) as g:
        assert g.node_count == 2
        assert g.edge_count == 1

        r = g.execute('NODE "a"')
        assert r.data["name"] == "alpha"

        r = g.execute('EDGES FROM "a" WHERE kind = "link"')
        assert r.count == 1
        assert r.data[0]["target"] == "b"


# ── Test 3: WAL recovery ─────────────────────────────────────────────


def test_wal_recovery(tmp_path):
    db_path = str(tmp_path / "db")

    # Create, mutate without explicit checkpoint.
    # Writes are appended to the WAL table before execution, so they
    # survive even if we skip the normal close() checkpoint path.
    g = SuperGraph(path=db_path)
    g.execute('CREATE NODE "a" kind = "x" name = "alpha"')
    g.execute('CREATE NODE "b" kind = "x" name = "beta"')

    # Simulate a crash: close the sqlite connection directly and
    # prevent close() from doing a full checkpoint. Release the
    # single-owner path lock the same way a crashed process would
    # (OS cleanup; we do it explicitly here).
    g._conn.close()
    g._runtime.conn = None
    if g._path_lock is not None:
        g._path_lock.release()
        g._path_lock = None

    # Reopen - the constructor should replay the WAL
    with SuperGraph(path=db_path) as g2:
        assert g2.node_count == 2
        r = g2.execute('NODE "a"')
        assert r.data is not None


# ── Test 4: Batch with rollback ───────────────────────────────────────


def test_batch_rollback():
    g = SuperGraph()
    g.execute('CREATE NODE "a" kind = "x" name = "alpha"')

    # The second CREATE inside the batch creates "a" again, which
    # already exists - triggers NodeExists, which causes rollback.
    with pytest.raises(BatchRollback):
        g.execute(
            'BEGIN\n'
            'CREATE NODE "b" kind = "x" name = "beta"\n'
            'CREATE NODE "a" kind = "x" name = "duplicate"\n'
            'COMMIT'
        )

    # "b" was created before the failing statement but should be
    # rolled back along with it.
    r = g.execute('NODE "b"')
    assert r.data is None
    assert g.node_count == 1


# ── Test 5: Memory ceiling ───────────────────────────────────────────


def test_memory_ceiling():
    g = SuperGraph(ceiling_mb=1)  # ~1 MB ceiling

    with pytest.raises(CeilingExceeded):
        for i in range(100_000):
            g.execute(f'CREATE NODE "n{i}" kind = "x" name = "node{i}"')


# ── Test 6: In-memory only mode ──────────────────────────────────────


def test_in_memory_mode():
    g = SuperGraph()  # No path -> in-memory
    g.execute('CREATE NODE "a" kind = "x" name = "alpha"')
    assert g.node_count == 1
    g.checkpoint()  # should be a no-op
    g.close()       # should succeed without error


# ── Test 7: System queries disabled ──────────────────────────────────


def test_system_queries_disabled():
    g = SuperGraph(allow_system_queries=False)
    g.execute('CREATE NODE "a" kind = "x" name = "alpha"')  # user query works

    with pytest.raises(PermissionError):
        g.execute('SYS STATS')


# ── Test 8: Execute batch ────────────────────────────────────────────


def test_execute_batch():
    g = SuperGraph()
    results = g.execute_batch([
        'CREATE NODE "a" kind = "x" name = "alpha"',
        'CREATE NODE "b" kind = "x" name = "beta"',
        'CREATE EDGE "a" -> "b" kind = "link"',
    ])
    assert len(results) == 3
    assert all(r.kind in ("ok", "node", "nodes", "edges") for r in results)
    assert g.node_count == 2


# ── Test 9: Context manager persistence ──────────────────────────────


def test_context_manager(tmp_path):
    db_path = str(tmp_path / "db")
    with SuperGraph(path=db_path) as g:
        g.execute('CREATE NODE "a" kind = "x" name = "alpha"')
    # __exit__ calls close() which calls checkpoint()

    with SuperGraph(path=db_path) as g:
        assert g.node_count == 1


# ── Test 10: Complex query patterns ──────────────────────────────────


def test_complex_queries():
    g = SuperGraph()

    # Build a linear call chain: fn0 -> fn1 -> fn2 -> ... -> fn9
    for i in range(10):
        g.execute(f'CREATE NODE "fn{i}" kind = "function" name = "func{i}"')
    for i in range(9):
        g.execute(f'CREATE EDGE "fn{i}" -> "fn{i + 1}" kind = "calls"')

    # Path finding
    r = g.execute('PATH FROM "fn0" TO "fn5" MAX_DEPTH 10 WHERE kind = "calls"')
    assert r.data is not None
    assert r.data[0] == "fn0"
    assert r.data[-1] == "fn5"

    # Distance
    r = g.execute('DISTANCE FROM "fn0" TO "fn5" MAX_DEPTH 10')
    assert r.data == 5

    # Traverse (includes start node)
    r = g.execute('TRAVERSE FROM "fn0" DEPTH 3 WHERE kind = "calls"')
    assert r.count == 4  # fn0, fn1, fn2, fn3

    # Descendants (excludes start node)
    r = g.execute('DESCENDANTS OF "fn0" DEPTH 3 WHERE kind = "calls"')
    assert r.count == 3  # fn1, fn2, fn3

    # Ancestors (excludes start node)
    r = g.execute('ANCESTORS OF "fn5" DEPTH 2 WHERE kind = "calls"')
    assert r.count == 2  # fn4, fn3

    # MATCH single hop
    r = g.execute('MATCH ("fn0") -[kind = "calls"]-> (b)')
    assert r.count == 1
    assert r.data["bindings"][0]["b"] == "fn1"

    # MATCH multi-hop
    r = g.execute('MATCH ("fn0") -[kind = "calls"]-> (b) -[kind = "calls"]-> (c)')
    assert r.count == 1
    assert r.data["bindings"][0]["b"] == "fn1"
    assert r.data["bindings"][0]["c"] == "fn2"


# ── Test 11: UPSERT behavior ────────────────────────────────────────


def test_upsert():
    g = SuperGraph()
    g.execute('UPSERT NODE "a" kind = "x" name = "v1"')
    r = g.execute('NODE "a"')
    assert r.data["name"] == "v1"

    g.execute('UPSERT NODE "a" kind = "x" name = "v2"')
    r = g.execute('NODE "a"')
    assert r.data["name"] == "v2"
    assert g.node_count == 1  # still just one node


# ── Test 12: DELETE NODES with WHERE ─────────────────────────────────


def test_delete_nodes_where():
    g = SuperGraph()
    g.execute('CREATE NODE "a" kind = "x" name = "keep"')
    g.execute('CREATE NODE "b" kind = "y" name = "delete"')
    g.execute('CREATE NODE "c" kind = "y" name = "delete"')
    g.execute('DELETE NODES WHERE kind = "y"')
    assert g.node_count == 1
    r = g.execute('NODE "a"')
    assert r.data is not None


# ── Test 13: INCREMENT ───────────────────────────────────────────────


def test_increment():
    g = SuperGraph()
    g.execute('CREATE NODE "a" kind = "x" name = "alpha" hits = 0')
    g.execute('INCREMENT NODE "a" hits BY 1')
    g.execute('INCREMENT NODE "a" hits BY 5')
    r = g.execute('NODE "a"')
    assert r.data["hits"] == 6


# ── Test 14: COMMON NEIGHBORS ────────────────────────────────────────


def test_common_neighbors():
    g = SuperGraph()
    g.execute('CREATE NODE "a" kind = "x" name = "a"')
    g.execute('CREATE NODE "b" kind = "x" name = "b"')
    g.execute('CREATE NODE "c" kind = "x" name = "c"')
    g.execute('CREATE NODE "d" kind = "x" name = "d"')

    g.execute('CREATE EDGE "a" -> "c" kind = "link"')
    g.execute('CREATE EDGE "a" -> "d" kind = "link"')
    g.execute('CREATE EDGE "b" -> "c" kind = "link"')
    g.execute('CREATE EDGE "b" -> "d" kind = "link"')

    r = g.execute('COMMON NEIGHBORS OF "a" AND "b" WHERE kind = "link"')
    assert r.count == 2
    names = {n["name"] for n in r.data}
    assert names == {"c", "d"}
