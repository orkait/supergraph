"""Locks the evolution-engine wiring fixes, snapshot survival, and WAL replay surfacing."""
import sqlite3

import pytest

from supergraph import SuperGraph


@pytest.mark.needs_embedder
class TestSimilarityBufferWired:
    def test_buffer_shared_with_runtime(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"))
        assert gs._runtime.similarity_buffer is gs._similarity_buffer
        gs.close()

    def test_similar_populates_buffer(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"))
        gs.execute(
            'SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text'
        )
        for i in range(5):
            gs.execute(f'CREATE NODE "d{i}" kind = "doc" text = "sample document {i}"')

        assert len(gs._similarity_buffer) == 0
        gs.execute('SIMILAR TO "sample document 2" LIMIT 3')
        assert len(gs._similarity_buffer) == 1
        assert 0.0 <= gs._similarity_buffer[0] <= 1.0
        gs.close()

    def test_remember_populates_buffer(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"))
        gs.execute(
            'SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text'
        )
        for i in range(5):
            gs.execute(f'CREATE NODE "d{i}" kind = "doc" text = "sample document {i}"')

        assert len(gs._similarity_buffer) == 0
        gs.execute('REMEMBER "sample" LIMIT 3')
        assert len(gs._similarity_buffer) == 1
        gs.close()

    def test_avg_similarity_signal_reads_buffer(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"))
        gs.execute(
            'SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text'
        )
        for i in range(5):
            gs.execute(f'CREATE NODE "d{i}" kind = "doc" text = "sample document {i}"')

        engine = gs._evolution_engine
        signals_before = engine.compute_signals()
        assert signals_before["avg_similarity"] == 0.0

        gs.execute('SIMILAR TO "sample document 2" LIMIT 3')
        gs.execute('SIMILAR TO "sample document 1" LIMIT 3')

        signals_after = engine.compute_signals()
        assert signals_after["avg_similarity"] > 0.0
        gs.close()


class TestEdgeDensitySignal:
    def test_edge_density_returns_real_value(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        for i in range(5):
            gs.execute(f'CREATE NODE "n{i}" kind = "test"')
        for i in range(4):
            gs.execute(f'CREATE EDGE "n{i}" -> "n{i+1}" kind = "link"')

        engine = gs._evolution_engine
        signals = engine.compute_signals()
        assert signals["edge_density"] == pytest.approx(0.8, abs=0.01)
        gs.close()

    def test_edge_density_zero_on_empty_graph(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        engine = gs._evolution_engine
        signals = engine.compute_signals()
        assert signals["edge_density"] == 0.0
        gs.close()


@pytest.mark.needs_embedder
class TestSimilarityThresholdWired:
    def test_default_is_none(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"))
        assert gs._executor._similarity_threshold is None
        gs.close()

    def test_evolution_set_writes_to_executor(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"))
        gs.execute(
            'SYS EVOLVE RULE "thresh" WHEN memory_pct >= 0 '
            'THEN SET similarity_threshold = 0.9 COOLDOWN 10'
        )
        engine = gs._evolution_engine
        engine.evaluate(engine.compute_signals())
        assert gs._executor._similarity_threshold == pytest.approx(0.9, abs=0.001)
        gs.close()

    def test_threshold_filters_similar_results(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"))
        gs.execute(
            'SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text'
        )
        texts = [
            "apple fruit sweet",
            "banana yellow tropical",
            "orange citrus vitamin",
            "car vehicle transport",
            "cat animal pet",
        ]
        for i, t in enumerate(texts):
            gs.execute(f'CREATE NODE "d{i}" kind = "doc" text = "{t}"')

        gs._executor._similarity_threshold = 0.99
        res = gs.execute('SIMILAR TO "apple fruit" LIMIT 10')
        for node in res.data:
            assert node["_similarity_score"] >= 0.99 - 0.001
        gs.close()


class TestDuplicateThresholdWired:
    def test_default_override_is_none(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"))
        assert gs._sys_executor._duplicate_threshold_override is None
        gs.close()

    def test_evolution_set_writes_override(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"))
        gs.execute(
            'SYS EVOLVE RULE "dup" WHEN memory_pct >= 0 '
            'THEN SET duplicate_threshold = 0.99 COOLDOWN 10'
        )
        engine = gs._evolution_engine
        engine.evaluate(engine.compute_signals())
        assert gs._sys_executor._duplicate_threshold_override == pytest.approx(0.99)
        gs.close()

    def test_dsl_explicit_threshold_still_wins(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"))
        gs._sys_executor._duplicate_threshold_override = 0.99
        gs.close()


class TestProtectedKindsWired:
    def test_default_is_none(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        assert gs._sys_executor._protected_kinds is None
        gs.close()

    def test_evolution_add_writes_to_sys_executor(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        gs.execute(
            'SYS EVOLVE RULE "pk" WHEN memory_pct >= 0 '
            'THEN ADD protected_kinds "vip" COOLDOWN 10'
        )
        engine = gs._evolution_engine
        engine.evaluate(engine.compute_signals())

        kinds = gs._sys_executor._protected_kinds
        assert kinds is not None
        assert "vip" in kinds
        assert "schema" in kinds
        assert "config" in kinds
        gs.close()

    def test_evict_respects_runtime_protected_kinds(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        gs._sys_executor._protected_kinds = {"schema", "config", "system", "vip"}

        gs.execute('CREATE NODE "important" kind = "vip" data = "keep me"')
        for i in range(10):
            gs.execute(f'CREATE NODE "n{i}" kind = "throwaway"')

        gs.execute('SYS EVICT LIMIT 5')
        assert gs.execute('NODE "important"').data is not None
        gs.close()


class TestSnapshotSurvivesCompact:
    def test_snapshot_not_cleared_by_compact(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        gs.execute('CREATE NODE "a" kind = "x"')
        gs.execute('CREATE NODE "b" kind = "x"')
        gs.execute('SYS SNAPSHOT "pre"')
        gs.execute('DELETE NODE "a"')
        gs.execute('SYS OPTIMIZE COMPACT')
        snaps = gs.execute('SYS SNAPSHOTS').data
        assert "pre" in snaps
        gs.close()

    def test_rollback_after_compact_restores_deleted(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        gs.execute('CREATE NODE "keep" kind = "x"')
        gs.execute('CREATE NODE "also_keep" kind = "x"')
        gs.execute('CREATE NODE "gone" kind = "x"')
        gs.execute('SYS SNAPSHOT "pre"')

        gs.execute('DELETE NODE "gone"')
        gs.execute('SYS OPTIMIZE COMPACT')
        gs.execute('SYS ROLLBACK TO "pre"')

        assert gs.execute('NODE "keep"').data is not None
        assert gs.execute('NODE "also_keep"').data is not None
        assert gs.execute('NODE "gone"').data is not None
        gs.close()

    def test_rollback_preserves_edges_across_compact(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        gs.execute('CREATE NODE "a" kind = "x"')
        gs.execute('CREATE NODE "b" kind = "x"')
        gs.execute('CREATE NODE "c" kind = "x"')
        gs.execute('CREATE EDGE "a" -> "b" kind = "link"')
        gs.execute('CREATE EDGE "b" -> "c" kind = "link"')
        gs.execute('SYS SNAPSHOT "full"')

        gs.execute('DELETE NODE "a"')
        gs.execute('SYS OPTIMIZE COMPACT')
        gs.execute('SYS ROLLBACK TO "full"')

        edges_from_a = gs.execute('EDGES FROM "a"').data
        assert any(e["target"] == "b" for e in edges_from_a)
        gs.close()


class TestWalReplayErrorSurfacing:
    def test_status_includes_replay_error_count_zero(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        status = gs.execute('SYS STATUS').data
        assert status.get("wal_replay_errors") == 0
        gs.close()

    def test_corrupt_wal_surfaces_replay_error(self, tmp_path):
        db_dir = tmp_path / "db"
        gs = SuperGraph(path=str(db_dir), embedder=None)
        gs.execute('CREATE NODE "a" kind = "test"')
        gs.close()

        conn = sqlite3.connect(str(db_dir / "supergraph.db"))
        conn.execute(
            "INSERT INTO wal (timestamp, statement) VALUES (?, ?)",
            (0.0, "TOTALLY INVALID DSL $$$"),
        )
        conn.commit()
        conn.close()

        gs2 = SuperGraph(path=str(db_dir), embedder=None)
        assert gs2._wal.replay_error_count == 1
        status = gs2.execute('SYS STATUS').data
        assert status["wal_replay_errors"] == 1
        assert "wal_replay_error_details" in status
        assert len(status["wal_replay_error_details"]) == 1
        assert "INVALID" in status["wal_replay_error_details"][0]["statement"]
        gs2.close()

    def test_replay_errors_cleared_on_reopen(self, tmp_path):
        db_dir = tmp_path / "db"
        gs = SuperGraph(path=str(db_dir), embedder=None)
        gs.execute('CREATE NODE "a" kind = "test"')
        gs.close()

        conn = sqlite3.connect(str(db_dir / "supergraph.db"))
        conn.execute(
            "INSERT INTO wal (timestamp, statement) VALUES (?, ?)",
            (0.0, "BAD STATEMENT"),
        )
        conn.commit()
        conn.close()

        gs2 = SuperGraph(path=str(db_dir), embedder=None)
        assert gs2._wal.replay_error_count == 1
        gs2.close()

        gs3 = SuperGraph(path=str(db_dir), embedder=None)
        assert gs3._wal.replay_error_count == 0
        gs3.close()
