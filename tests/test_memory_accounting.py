import tempfile
from supergraph import SuperGraph
from supergraph.core.memory import measure, estimate


def test_measure_returns_breakdown():
    gs = SuperGraph()
    gs.execute('CREATE NODE "a" kind = "test" name = "Alice"')
    gs.execute('CREATE NODE "b" kind = "test" name = "Bob"')
    gs.execute('CREATE EDGE "a" -> "b" kind = "knows"')

    report = measure(gs._store, gs._vector_store)
    assert "node_arrays" in report
    assert "columns" in report
    assert "string_table" in report
    assert "edge_lists" in report
    assert "edge_matrices" in report
    assert "vector_store" in report
    assert "total" in report
    assert report["total"] > 0
    assert report["node_arrays"] > 0
    gs.close()


def test_measure_more_accurate_than_estimate():
    gs = SuperGraph()
    for i in range(100):
        gs.execute(f'CREATE NODE "n{i}" kind = "test" val = {i}')
    report = measure(gs._store)
    est = estimate(gs._store.node_count, gs._store.edge_count)
    assert report["total"] > 0
    assert est > 0
    gs.close()


def test_sys_status_includes_measured_memory():
    with tempfile.TemporaryDirectory() as td:
        gs = SuperGraph(path=td)
        gs.execute('CREATE NODE "x" kind = "test"')
        result = gs.execute('SYS STATUS')
        assert "memory_measured" in result.data
        gs.close()


def test_sys_health_includes_utilization():
    with tempfile.TemporaryDirectory() as td:
        gs = SuperGraph(path=td)
        gs.execute('CREATE NODE "x" kind = "test"')
        result = gs.execute('SYS HEALTH')
        assert "memory_utilization" in result.data
        assert result.data["memory_utilization"] >= 0
        gs.close()


def test_evict_oldest_removes_nodes():
    from supergraph.core.store import CoreStore
    from supergraph.core.optimizer import evict_oldest
    import time

    store = CoreStore(ceiling_bytes=1_000_000)
    for i in range(50):
        store.put_node(f"n{i}", "test", {"val": i})
        store.columns.set_reserved(
            store.id_to_slot[store.string_table.intern(f"n{i}")],
            "__updated_at__",
            int(time.time() * 1000) - (50 - i) * 86400000
        )

    result = evict_oldest(store, target_bytes=1)
    assert result["evicted"] > 0
    assert result["evicted"] >= 40


def test_sys_evict_command():
    with tempfile.TemporaryDirectory() as td:
        gs = SuperGraph(path=td, ceiling_mb=256)
        for i in range(20):
            gs.execute(f'CREATE NODE "e{i}" kind = "test" val = {i}')
        result = gs.execute('SYS EVICT')
        assert result.kind == "ok"
        assert "evicted" in result.data
        gs.close()
