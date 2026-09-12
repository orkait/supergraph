"""Tests for the self-balancing optimizer."""

import pytest
from supergraph import SuperGraph, OptimizationInProgress
from supergraph.dsl.parser import parse_uncached
from supergraph.dsl.ast_nodes import SysHealth, SysOptimize


class TestParsing:
    def test_parse_health(self):
        ast = parse_uncached('SYS HEALTH')
        assert isinstance(ast, SysHealth)

    def test_parse_optimize_all(self):
        ast = parse_uncached('SYS OPTIMIZE')
        assert isinstance(ast, SysOptimize)
        assert ast.target is None

    def test_parse_optimize_compact(self):
        ast = parse_uncached('SYS OPTIMIZE COMPACT')
        assert isinstance(ast, SysOptimize)
        assert ast.target == "COMPACT"

    def test_parse_optimize_all_targets(self):
        for target in ("COMPACT", "STRINGS", "EDGES", "VECTORS", "BLOBS", "CACHE"):
            ast = parse_uncached(f'SYS OPTIMIZE {target}')
            assert ast.target == target


class TestHealth:
    def test_health_returns_metrics(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        gs.execute('CREATE NODE "a" kind = "test"')
        result = gs.execute('SYS HEALTH')
        assert result.kind == "health"
        data = result.data
        assert "tombstone_ratio" in data
        assert "string_bloat" in data
        assert "dead_vectors" in data
        assert "recommended" in data
        assert data["live_nodes"] == 1
        gs.close()

    def test_health_detects_tombstone_pressure(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        for i in range(10):
            gs.execute(f'CREATE NODE "n{i}" kind = "test"')
        for i in range(8):
            gs.execute(f'DELETE NODE "n{i}"')
        result = gs.execute('SYS HEALTH')
        assert result.data["tombstone_ratio"] == 0.8
        assert "COMPACT" in result.data["recommended"]
        gs.close()


class TestOptimizeCompact:
    def test_compact_removes_tombstones(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        for i in range(10):
            gs.execute(f'CREATE NODE "n{i}" kind = "test" value = {i}')
        for i in range(5):
            gs.execute(f'DELETE NODE "n{i}"')
        assert gs._store._next_slot == 10
        assert len(gs._store.node_tombstones) == 5

        result = gs.execute('SYS OPTIMIZE COMPACT')
        assert result.data["compacted"] == 5
        assert gs._store._next_slot == 5
        assert len(gs._store.node_tombstones) == 0

        # Remaining nodes still accessible
        for i in range(5, 10):
            node = gs.execute(f'NODE "n{i}"')
            assert node.data is not None
            assert node.data["value"] == i
        gs.close()

    def test_compact_preserves_edges(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        gs.execute('CREATE NODE "a" kind = "test"')
        gs.execute('CREATE NODE "b" kind = "test"')
        gs.execute('CREATE NODE "c" kind = "test"')
        gs.execute('CREATE EDGE "a" -> "c" kind = "link"')
        gs.execute('DELETE NODE "b"')

        gs.execute('SYS OPTIMIZE COMPACT')
        edges = gs.execute('EDGES FROM "a"')
        assert len(edges.data) == 1
        assert edges.data[0]["target"] == "c"
        gs.close()

    def test_compact_no_tombstones_is_noop(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        gs.execute('CREATE NODE "a" kind = "test"')
        result = gs.execute('SYS OPTIMIZE COMPACT')
        assert result.data["compacted"] == 0
        gs.close()

    def test_compact_preserves_snapshots(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        gs.execute('CREATE NODE "a" kind = "test"')
        gs.execute('CREATE NODE "b" kind = "test"')
        gs.execute('SYS SNAPSHOT "before"')
        gs.execute('DELETE NODE "a"')
        gs.execute('SYS OPTIMIZE COMPACT')
        snapshots = gs.execute('SYS SNAPSHOTS')
        assert "before" in snapshots.data
        gs.execute('SYS ROLLBACK TO "before"')
        assert gs.execute('NODE "a"').data is not None
        assert gs.execute('NODE "b"').data is not None
        gs.close()


class TestOptimizeStrings:
    def test_gc_frees_dead_strings(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        for i in range(20):
            gs.execute(f'CREATE NODE "tmp{i}" kind = "temp" label = "unique_string_{i}"')
        for i in range(20):
            gs.execute(f'DELETE NODE "tmp{i}"')
        gs.execute('CREATE NODE "keep" kind = "live"')

        old_string_count = len(gs._store.string_table)
        result = gs.execute('SYS OPTIMIZE STRINGS')
        new_string_count = len(gs._store.string_table)
        assert result.data["strings_freed"] > 0
        assert new_string_count < old_string_count

        node = gs.execute('NODE "keep"')
        assert node.data is not None
        assert node.data["kind"] == "live"
        gs.close()


class TestOptimizeEdges:
    def test_defrag_rebuilds(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        gs.execute('CREATE NODE "a" kind = "test"')
        gs.execute('CREATE NODE "b" kind = "test"')
        gs.execute('CREATE EDGE "a" -> "b" kind = "link"')
        result = gs.execute('SYS OPTIMIZE EDGES')
        assert result.data["edge_types"] == 1
        gs.close()


class TestOptimizeVectors:
    def test_cleanup_removes_dead_vectors(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        gs.execute('CREATE NODE "a" kind = "test" VECTOR [1.0, 0.0, 0.0, 0.0]')
        gs.execute('CREATE NODE "b" kind = "test" VECTOR [0.0, 1.0, 0.0, 0.0]')
        # RETRACT now immediately removes the vector (no ghost until optimize)
        str_id = gs._store.string_table.intern("a")
        slot = gs._store.id_to_slot[str_id]
        assert gs._vector_store.has_vector(slot)
        gs.execute('RETRACT "a"')
        assert not gs._vector_store.has_vector(slot)

        result = gs.execute('SYS OPTIMIZE VECTORS')
        assert result.data["removed"] == 0
        gs.close()


class TestOptimizeAll:
    def test_optimize_all(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        for i in range(5):
            gs.execute(f'CREATE NODE "n{i}" kind = "test"')
        for i in range(3):
            gs.execute(f'DELETE NODE "n{i}"')
        result = gs.execute('SYS OPTIMIZE')
        assert "compact" in result.data
        assert "strings" in result.data
        assert "edges" in result.data
        assert "vectors" in result.data
        assert "blobs" in result.data
        assert "cache" in result.data
        gs.close()


class TestOptimizeLock:
    def test_lock_rejects_during_optimize(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        gs._optimizer._optimizing = True
        with pytest.raises(OptimizationInProgress):
            gs.execute('NODE "x"')
        gs._optimizer._optimizing = False
        gs.close()


class TestAutoOptimize:
    def test_auto_optimize_triggers_at_interval(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)

        # Build pressure manually (auto_optimize disabled by default)
        for i in range(10):
            gs.execute(f'CREATE NODE "n{i}" kind = "test"')
        for i in range(8):
            gs.execute(f'DELETE NODE "n{i}"')

        assert gs._store._next_slot == 10
        assert len(gs._store.node_tombstones) == 8

        # Simulate what auto-optimize does: health check sets flag
        gs._optimizer._check_health()
        assert gs._optimizer._needs_optimize is True

        # Next query runs auto-optimize at safe point
        result = gs.execute('NODE "n8"')
        assert gs._optimizer._needs_optimize is False
        assert len(gs._store.node_tombstones) == 0
        assert result.data is not None
        gs.close()

    def test_auto_optimize_disabled_by_default(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        assert gs._config.dsl.auto_optimize is False
        gs.close()
