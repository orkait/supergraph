"""Tests for graph intelligence: RECALL, PROPAGATE, COUNTERFACTUAL, SNAPSHOT, CONTEXT."""
import pytest
from supergraph import SuperGraph
from supergraph.core.errors import NodeNotFound


class TestRecall:
    def test_recall_returns_connected_nodes(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "cue" kind = "concept" name = "Paris"')
        g.execute('CREATE NODE "m1" kind = "memory" name = "Eiffel"')
        g.execute('CREATE NODE "m2" kind = "memory" name = "Louvre"')
        g.execute('CREATE NODE "m3" kind = "memory" name = "Unrelated"')
        g.execute('CREATE EDGE "cue" -> "m1" kind = "related"')
        g.execute('CREATE EDGE "cue" -> "m2" kind = "related"')
        result = g.execute('RECALL FROM "cue" DEPTH 1 LIMIT 10')
        ids = [n["id"] for n in result.data]
        assert "m1" in ids
        assert "m2" in ids
        assert "m3" not in ids

    def test_recall_with_where(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "cue" kind = "concept"')
        g.execute('CREATE NODE "m1" kind = "memory"')
        g.execute('CREATE NODE "m2" kind = "other"')
        g.execute('CREATE EDGE "cue" -> "m1" kind = "r"')
        g.execute('CREATE EDGE "cue" -> "m2" kind = "r"')
        result = g.execute('RECALL FROM "cue" DEPTH 1 LIMIT 10 WHERE kind = "memory"')
        assert all(n["kind"] == "memory" for n in result.data)

    def test_recall_has_activation_score(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "cue" kind = "concept"')
        g.execute('CREATE NODE "m1" kind = "memory"')
        g.execute('CREATE EDGE "cue" -> "m1" kind = "r"')
        result = g.execute('RECALL FROM "cue" DEPTH 1 LIMIT 10')
        assert len(result.data) > 0
        assert "_activation_score" in result.data[0]

    def test_recall_nonexistent_raises(self):
        g = SuperGraph(ceiling_mb=256)
        with pytest.raises(NodeNotFound):
            g.execute('RECALL FROM "nonexistent" DEPTH 1 LIMIT 10')

    def test_recall_depth_2(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "a" kind = "concept"')
        g.execute('CREATE NODE "b" kind = "memory"')
        g.execute('CREATE NODE "c" kind = "memory"')
        g.execute('CREATE EDGE "a" -> "b" kind = "r"')
        g.execute('CREATE EDGE "b" -> "c" kind = "r"')
        result = g.execute('RECALL FROM "a" DEPTH 2 LIMIT 10')
        ids = [n["id"] for n in result.data]
        assert "c" in ids  # reachable at depth 2

    def test_recall_reaches_incoming_neighbors_from_sink_node(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "msg" kind = "memory" content = "Caroline moved from Sweden"')
        g.execute('CREATE NODE "ent" kind = "entity" name = "Caroline"')
        g.execute('CREATE EDGE "msg" -> "ent" kind = "mentions"')
        result = g.execute('RECALL FROM "ent" DEPTH 1 LIMIT 10')
        ids = [n["id"] for n in result.data]
        assert "msg" in ids

    def test_recall_spreads_bidirectionally(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "a" kind = "concept"')
        g.execute('CREATE NODE "b" kind = "concept"')
        g.execute('CREATE NODE "c" kind = "concept"')
        g.execute('CREATE EDGE "a" -> "b" kind = "r"')
        g.execute('CREATE EDGE "b" -> "c" kind = "r"')
        result = g.execute('RECALL FROM "b" DEPTH 1 LIMIT 10')
        ids = {n["id"] for n in result.data}
        assert ids == {"a", "c"}


class TestPropagate:
    def test_propagate_updates_descendants(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('SYS REGISTER NODE KIND "belief" REQUIRED confidence:float')
        g.execute('CREATE NODE "root" kind = "belief" confidence = 0.9')
        g.execute('CREATE NODE "child" kind = "belief" confidence = 0.5')
        g.execute('CREATE EDGE "root" -> "child" kind = "supports"')
        result = g.execute('PROPAGATE "root" FIELD confidence DEPTH 1')
        assert result.data["updated"] >= 1


class TestCounterfactual:
    def test_what_if_does_not_commit(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "b1" kind = "belief" value = "x"')
        g.execute('CREATE NODE "c1" kind = "conclusion"')
        g.execute('CREATE EDGE "b1" -> "c1" kind = "supports"')
        result = g.execute('WHAT IF RETRACT "b1"')
        assert result.data["affected_count"] >= 1
        # Original still exists
        assert g.execute('NODE "b1"').data is not None

    def test_what_if_nonexistent_raises(self):
        g = SuperGraph(ceiling_mb=256)
        with pytest.raises(NodeNotFound):
            g.execute('WHAT IF RETRACT "nonexistent"')


class TestSnapshot:
    def test_snapshot_and_rollback(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "n1" kind = "test" value = "original"')
        g.execute('SYS SNAPSHOT "before"')
        g.execute('UPDATE NODE "n1" SET value = "modified"')
        assert g.execute('NODE "n1"').data["value"] == "modified"
        g.execute('SYS ROLLBACK TO "before"')
        assert g.execute('NODE "n1"').data["value"] == "original"

    def test_snapshots_list(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('SYS SNAPSHOT "snap1"')
        g.execute('SYS SNAPSHOT "snap2"')
        result = g.execute('SYS SNAPSHOTS')
        assert len(result.data) >= 2

    def test_rollback_to_nonexistent_raises(self):
        g = SuperGraph(ceiling_mb=256)
        with pytest.raises(Exception):
            g.execute('SYS ROLLBACK TO "nonexistent"')


class TestBindContext:
    def test_context_isolates_creates(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "global" kind = "fact" name = "visible"')
        g.execute('BIND CONTEXT "session-1"')
        g.execute('CREATE NODE "local" kind = "hypothesis" name = "maybe"')
        # Only context nodes visible while bound
        result = g.execute('NODES')
        assert len(result.data) == 1
        assert result.data[0]["id"] == "local"

    def test_discard_context_deletes_nodes(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "global" kind = "fact" name = "keep"')
        g.execute('BIND CONTEXT "session-1"')
        g.execute('CREATE NODE "local" kind = "temp" name = "discard"')
        g.execute('DISCARD CONTEXT "session-1"')
        # Back to global view, local deleted
        result = g.execute('NODES')
        assert len(result.data) == 1
        assert result.data[0]["id"] == "global"


def test_combined_transpose_cached():
    """get_combined_transpose() must return the same object on repeated calls."""
    from supergraph.core.edges import EdgeMatrices
    em = EdgeMatrices()
    em.rebuild({"knows": [(0, 1, {}), (1, 2, {})]}, num_nodes=3)
    t1 = em.get_combined_transpose()
    t2 = em.get_combined_transpose()
    assert t1 is t2, "transpose must be the same cached object, not recomputed"


def test_combined_transpose_invalidated_on_rebuild():
    """Rebuild must invalidate the combined transpose cache."""
    from supergraph.core.edges import EdgeMatrices
    em = EdgeMatrices()
    em.rebuild({"knows": [(0, 1, {})]}, num_nodes=2)
    t1 = em.get_combined_transpose()
    em.rebuild({"knows": [(0, 1, {}), (1, 0, {})]}, num_nodes=2)
    t2 = em.get_combined_transpose()
    assert t1 is not t2, "after rebuild, transpose cache must be refreshed"


def test_combined_spread_matrix_cached():
    """Spread matrix cache should be stable across repeated calls."""
    from supergraph.core.edges import EdgeMatrices
    em = EdgeMatrices()
    em.rebuild({"knows": [(0, 1, {}), (1, 2, {})]}, num_nodes=3)
    s1 = em.get_combined_spread()
    s2 = em.get_combined_spread()
    assert s1 is s2


def test_combined_spread_matrix_invalidated_on_rebuild():
    """Spread matrix cache must refresh after rebuild."""
    from supergraph.core.edges import EdgeMatrices
    em = EdgeMatrices()
    em.rebuild({"knows": [(0, 1, {})]}, num_nodes=2)
    s1 = em.get_combined_spread()
    em.rebuild({"knows": [(0, 1, {}), (1, 0, {})]}, num_nodes=2)
    s2 = em.get_combined_spread()
    assert s1 is not s2
