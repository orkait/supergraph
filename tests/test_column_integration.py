
import pytest
from supergraph.core.store import CoreStore
from supergraph.core.schema import SchemaRegistry
from supergraph.core.runtime import RuntimeState
from supergraph.dsl.parser import parse
from supergraph.dsl.executor import Executor
from supergraph.dsl.executor_system import SystemExecutor


def _make_runtime(store):
    return RuntimeState(store=store, schema=SchemaRegistry())


@pytest.fixture
def graph():
    store = CoreStore()
    store.put_node("fn1", "function", {"name": "main", "line": 1, "score": 100})
    store.put_node("fn2", "function", {"name": "helper", "line": 10, "score": 50})
    store.put_node("fn3", "function", {"name": "parse", "line": 5, "score": 200})
    store.put_node("cls1", "class", {"name": "App", "line": 20, "score": 75})
    store.put_node("cls2", "class", {"name": "Base", "line": 1, "score": 25})
    return Executor(_make_runtime(store))


def execute(executor, query):
    ast = parse(query)
    return executor.execute(ast)


class TestColumnFilterNodes:
    def test_where_int_gt(self, graph):
        r = execute(graph, 'NODES WHERE score > 90')
        ids = {n["id"] for n in r.data}
        assert ids == {"fn1", "fn3"}

    def test_where_int_eq(self, graph):
        r = execute(graph, 'NODES WHERE line = 1')
        ids = {n["id"] for n in r.data}
        assert ids == {"fn1", "cls2"}

    def test_where_string_eq(self, graph):
        r = execute(graph, 'NODES WHERE name = "main"')
        assert r.count == 1
        assert r.data[0]["id"] == "fn1"

    def test_where_kind_and_int(self, graph):
        r = execute(graph, 'NODES WHERE kind = "function" AND score > 90')
        ids = {n["id"] for n in r.data}
        assert ids == {"fn1", "fn3"}

    def test_where_or(self, graph):
        r = execute(graph, 'NODES WHERE score = 100 OR score = 25')
        ids = {n["id"] for n in r.data}
        assert ids == {"fn1", "cls2"}

    def test_where_not(self, graph):
        r = execute(graph, 'NODES WHERE kind = "function" AND NOT score > 90')
        ids = {n["id"] for n in r.data}
        assert ids == {"fn2"}

    def test_where_in(self, graph):
        r = execute(graph, 'NODES WHERE name IN ("main", "parse")')
        ids = {n["id"] for n in r.data}
        assert ids == {"fn1", "fn3"}

    def test_where_null_eq(self, graph):
        graph.store.put_node("bare", "function", {"name": "bare", "line": 99})
        r = execute(graph, 'NODES WHERE score = NULL')
        ids = {n["id"] for n in r.data}
        assert "bare" in ids

    def test_where_null_neq(self, graph):
        graph.store.put_node("bare", "function", {"name": "bare", "line": 99})
        r = execute(graph, 'NODES WHERE score != NULL')
        ids = {n["id"] for n in r.data}
        assert "bare" not in ids
        assert len(ids) == 5

    def test_contains_falls_back(self, graph):
        r = execute(graph, 'NODES WHERE name CONTAINS "main"')
        assert r.count == 1
        assert r.data[0]["id"] == "fn1"

    def test_like_falls_back(self, graph):
        r = execute(graph, 'NODES WHERE name LIKE "ma%"')
        assert r.count == 1

    def test_order_by_with_limit(self, graph):
        r = execute(graph, 'NODES WHERE kind = "function" ORDER BY score DESC LIMIT 2')
        assert r.data[0]["id"] == "fn3"
        assert r.data[1]["id"] == "fn1"

    def test_order_by_asc(self, graph):
        r = execute(graph, 'NODES WHERE kind = "function" ORDER BY score ASC LIMIT 3')
        assert r.data[0]["score"] == 50
        assert r.data[1]["score"] == 100

    def test_offset(self, graph):
        r = execute(graph, 'NODES WHERE kind = "function" ORDER BY score ASC LIMIT 2 OFFSET 1')
        assert len(r.data) == 2

    def test_order_by_no_where(self, graph):
        r = execute(graph, 'NODES ORDER BY score DESC LIMIT 2')
        assert r.data[0]["score"] == 200
        assert r.data[1]["score"] == 100


class TestColumnFilterCount:
    def test_count_with_filter(self, graph):
        r = execute(graph, 'COUNT NODES WHERE score > 50')
        assert r.data == 3

    def test_count_with_kind_and_filter(self, graph):
        r = execute(graph, 'COUNT NODES WHERE kind = "function" AND score > 50')
        assert r.data == 2


class TestColumnFilterDelete:
    def test_delete_with_filter(self, graph):
        r = execute(graph, 'DELETE NODES WHERE score < 50')
        assert r.count == 1
        assert r.data[0]["id"] == "cls2"
        r2 = execute(graph, 'COUNT NODES')
        assert r2.data == 4


from supergraph.core.schema import SchemaRegistry
from supergraph.dsl.executor_system import SystemExecutor
from supergraph.core.errors import BatchRollback


class TestBatchRollbackColumns:
    def test_rollback_restores_columns(self):
        store = CoreStore()
        store.put_node("n1", "fn", {"score": 10})
        executor = Executor(_make_runtime(store))

        with pytest.raises(BatchRollback):
            execute(executor, '''BEGIN
CREATE NODE "n2" kind = "fn" score = 99
CREATE NODE "n1" kind = "fn" score = 50
COMMIT''')

        assert store.get_node("n2") is None
        mask = store.columns.get_mask("score", "=", 10, store._next_slot)
        assert mask[0]


class TestSysRebuildColumns:
    def test_rebuild_restores_secondary_indices(self):
        store = CoreStore()
        store.put_node("n1", "fn", {"score": 42})
        store.add_index("score")
        sys_exec = SystemExecutor(_make_runtime(store))

        store.secondary_indices["score"] = {}
        assert store.query_by_index("score", 42) == []

        ast = parse("SYS REBUILD INDICES")
        sys_exec.execute(ast)

        assert store.columns.has_column("score")
        mask = store.columns.get_mask("score", "=", 42, store._next_slot)
        assert mask[0]

        slots = store.query_by_index("score", 42)
        assert len(slots) == 1


class TestColumnMemoryStats:
    def test_stats_includes_column_memory(self):
        store = CoreStore()
        store.put_node("n1", "fn", {"score": 42})
        sys_exec = SystemExecutor(_make_runtime(store))

        ast = parse("SYS STATS MEMORY")
        r = sys_exec.execute(ast)
        assert "column_memory_bytes" in r.data
        assert r.data["column_memory_bytes"] > 0


from supergraph import SuperGraph


class TestEndToEnd:
    def test_supergraph_round_trip(self, tmp_path):
        with SuperGraph(str(tmp_path / "db"), ceiling_mb=64) as gs:
            gs.execute('CREATE NODE "fn1" kind = "function" name = "main" score = 100')
            gs.execute('CREATE NODE "fn2" kind = "function" name = "helper" score = 50')
            gs.execute('CREATE NODE "fn3" kind = "function" name = "parse" score = 200')
            gs.checkpoint()

        with SuperGraph(str(tmp_path / "db"), ceiling_mb=64) as gs:
            r = gs.execute('NODES WHERE score > 90')
            ids = {n["id"] for n in r.data}
            assert ids == {"fn1", "fn3"}

            r = gs.execute('COUNT NODES WHERE score > 50')
            assert r.data == 2

    def test_typed_schema_end_to_end(self, tmp_path):
        with SuperGraph(str(tmp_path / "db")) as gs:
            gs.execute('SYS REGISTER NODE KIND "function" REQUIRED name:string, line:int OPTIONAL score:float')
            gs.execute('CREATE NODE "fn1" kind = "function" name = "main" line = 1 score = 9.5')

            r = gs.execute('NODES WHERE name = "main"')
            assert r.count == 1

            r = gs.execute('COUNT NODES WHERE line = 1')
            assert r.data == 1

    def test_columns_survive_wal_replay(self, tmp_path):
        with SuperGraph(str(tmp_path / "db")) as gs:
            gs.execute('CREATE NODE "n1" kind = "fn" score = 42')

        with SuperGraph(str(tmp_path / "db")) as gs:
            r = gs.execute('NODES WHERE score = 42')
            assert r.count == 1


def test_compact_preserves_document_slots(tmp_path):
    from supergraph import SuperGraph
    gs = SuperGraph(path=str(tmp_path))

    gs.execute('CREATE NODE "a" kind = "doc"')
    gs.execute('CREATE NODE "b" kind = "doc"')
    gs.execute('CREATE NODE "c" kind = "doc"')

    for nid, summary in [("a", "alpha content"), ("b", "beta content"), ("c", "gamma content")]:
        str_id = gs._store.string_table.intern(nid)
        slot = gs._store.id_to_slot[str_id]
        gs._document_store.put_summary(slot, summary)

    gs.execute('DELETE NODE "a"')

    gs.execute("SYS OPTIMIZE COMPACT")

    r = gs.execute('LEXICAL SEARCH "beta" LIMIT 5')
    ids = [n["id"] for n in r.data]
    assert "b" in ids, f"'b' missing after compaction; got {ids}"

    r2 = gs.execute('LEXICAL SEARCH "gamma" LIMIT 5')
    ids2 = [n["id"] for n in r2.data]
    assert "c" in ids2, f"'c' missing after compaction; got {ids2}"

    gs.close()
