
import pytest

from supergraph import SuperGraph
from supergraph.core.errors import BatchRollback, CeilingExceeded


def test_full_workflow(tmp_path):
    with SuperGraph(path=str(tmp_path / "db")) as g:
        g.execute('SYS REGISTER NODE KIND "function" REQUIRED name OPTIONAL file, line')
        g.execute('SYS REGISTER NODE KIND "class" REQUIRED name OPTIONAL file')
        g.execute('SYS REGISTER EDGE KIND "calls" FROM "function" TO "function"')

        g.execute('CREATE NODE "fn_main" kind = "function" name = "main" file = "main.py"')
        g.execute('CREATE NODE "fn_helper" kind = "function" name = "helper" file = "utils.py"')
        g.execute('CREATE NODE "fn_parse" kind = "function" name = "parse" file = "parser.py"')
        g.execute('CREATE NODE "cls_app" kind = "class" name = "App" file = "main.py"')

        g.execute('CREATE EDGE "fn_main" -> "fn_helper" kind = "calls"')
        g.execute('CREATE EDGE "fn_main" -> "fn_parse" kind = "calls"')
        g.execute('CREATE EDGE "fn_helper" -> "fn_parse" kind = "calls"')

        assert g.node_count == 4
        assert g.edge_count == 3

        r = g.execute('NODE "fn_main"')
        assert r.data["name"] == "main"

        r = g.execute('NODES WHERE kind = "function"')
        assert r.count == 3

        r = g.execute('EDGES FROM "fn_main" WHERE kind = "calls"')
        assert r.count == 2

        r = g.execute('SHORTEST PATH FROM "fn_main" TO "fn_parse" WHERE kind = "calls"')
        assert r.data is not None
        assert len(r.data) == 2

        r = g.execute('ANCESTORS OF "fn_parse" DEPTH 2 WHERE kind = "calls"')
        assert r.count >= 2

        g.execute('UPDATE NODE "fn_main" SET line = 1')
        r = g.execute('NODE "fn_main"')
        assert r.data["line"] == 1

        g.execute('DELETE NODE "cls_app"')
        assert g.node_count == 3

        r = g.execute('SYS STATS')
        assert r.data["node_count"] == 3


def test_persistence_roundtrip(tmp_path):
    db_path = str(tmp_path / "db")

    with SuperGraph(path=db_path) as g:
        g.execute('CREATE NODE "a" kind = "x" name = "alpha"')
        g.execute('CREATE NODE "b" kind = "x" name = "beta"')
        g.execute('CREATE EDGE "a" -> "b" kind = "link"')
        g.checkpoint()

    with SuperGraph(path=db_path) as g:
        assert g.node_count == 2
        assert g.edge_count == 1

        r = g.execute('NODE "a"')
        assert r.data["name"] == "alpha"

        r = g.execute('EDGES FROM "a" WHERE kind = "link"')
        assert r.count == 1
        assert r.data[0]["target"] == "b"


def test_wal_recovery(tmp_path):
    db_path = str(tmp_path / "db")

    g = SuperGraph(path=db_path)
    g.execute('CREATE NODE "a" kind = "x" name = "alpha"')
    g.execute('CREATE NODE "b" kind = "x" name = "beta"')

    g._conn.close()
    g._runtime.conn = None
    if g._path_lock is not None:
        g._path_lock.release()
        g._path_lock = None

    with SuperGraph(path=db_path) as g2:
        assert g2.node_count == 2
        r = g2.execute('NODE "a"')
        assert r.data is not None


def test_batch_rollback():
    g = SuperGraph()
    g.execute('CREATE NODE "a" kind = "x" name = "alpha"')

    with pytest.raises(BatchRollback):
        g.execute(
            'BEGIN\n'
            'CREATE NODE "b" kind = "x" name = "beta"\n'
            'CREATE NODE "a" kind = "x" name = "duplicate"\n'
            'COMMIT'
        )

    r = g.execute('NODE "b"')
    assert r.data is None
    assert g.node_count == 1


def test_memory_ceiling():
    g = SuperGraph(ceiling_mb=1)

    with pytest.raises(CeilingExceeded):
        for i in range(100_000):
            g.execute(f'CREATE NODE "n{i}" kind = "x" name = "node{i}"')


def test_in_memory_mode():
    g = SuperGraph()
    g.execute('CREATE NODE "a" kind = "x" name = "alpha"')
    assert g.node_count == 1
    g.checkpoint()
    g.close()


def test_system_queries_disabled():
    g = SuperGraph(allow_system_queries=False)
    g.execute('CREATE NODE "a" kind = "x" name = "alpha"')

    with pytest.raises(PermissionError):
        g.execute('SYS STATS')


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


def test_context_manager(tmp_path):
    db_path = str(tmp_path / "db")
    with SuperGraph(path=db_path) as g:
        g.execute('CREATE NODE "a" kind = "x" name = "alpha"')

    with SuperGraph(path=db_path) as g:
        assert g.node_count == 1


def test_complex_queries():
    g = SuperGraph()

    for i in range(10):
        g.execute(f'CREATE NODE "fn{i}" kind = "function" name = "func{i}"')
    for i in range(9):
        g.execute(f'CREATE EDGE "fn{i}" -> "fn{i + 1}" kind = "calls"')

    r = g.execute('PATH FROM "fn0" TO "fn5" MAX_DEPTH 10 WHERE kind = "calls"')
    assert r.data is not None
    assert r.data[0] == "fn0"
    assert r.data[-1] == "fn5"

    r = g.execute('DISTANCE FROM "fn0" TO "fn5" MAX_DEPTH 10')
    assert r.data == 5

    r = g.execute('TRAVERSE FROM "fn0" DEPTH 3 WHERE kind = "calls"')
    assert r.count == 4

    r = g.execute('DESCENDANTS OF "fn0" DEPTH 3 WHERE kind = "calls"')
    assert r.count == 3

    r = g.execute('ANCESTORS OF "fn5" DEPTH 2 WHERE kind = "calls"')
    assert r.count == 2

    r = g.execute('MATCH ("fn0") -[kind = "calls"]-> (b)')
    assert r.count == 1
    assert r.data["bindings"][0]["b"] == "fn1"

    r = g.execute('MATCH ("fn0") -[kind = "calls"]-> (b) -[kind = "calls"]-> (c)')
    assert r.count == 1
    assert r.data["bindings"][0]["b"] == "fn1"
    assert r.data["bindings"][0]["c"] == "fn2"


def test_upsert():
    g = SuperGraph()
    g.execute('UPSERT NODE "a" kind = "x" name = "v1"')
    r = g.execute('NODE "a"')
    assert r.data["name"] == "v1"

    g.execute('UPSERT NODE "a" kind = "x" name = "v2"')
    r = g.execute('NODE "a"')
    assert r.data["name"] == "v2"
    assert g.node_count == 1


def test_delete_nodes_where():
    g = SuperGraph()
    g.execute('CREATE NODE "a" kind = "x" name = "keep"')
    g.execute('CREATE NODE "b" kind = "y" name = "delete"')
    g.execute('CREATE NODE "c" kind = "y" name = "delete"')
    g.execute('DELETE NODES WHERE kind = "y"')
    assert g.node_count == 1
    r = g.execute('NODE "a"')
    assert r.data is not None


def test_increment():
    g = SuperGraph()
    g.execute('CREATE NODE "a" kind = "x" name = "alpha" hits = 0')
    g.execute('INCREMENT NODE "a" hits BY 1')
    g.execute('INCREMENT NODE "a" hits BY 5')
    r = g.execute('NODE "a"')
    assert r.data["hits"] == 6


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
