"""PR 2: traversal verbs + remaining reads.

Every verb: emit + parser roundtrip + at least one injection case.
"""
import pytest

from supergraph import q, F
from supergraph.dsl.parser import parse


def _roundtrip(query_obj):
    dsl = query_obj.dsl()
    try:
        parse(dsl)
    except Exception as e:
        pytest.fail(f"parser rejected {dsl!r}: {e}")
    return dsl


class TestTraverse:
    def test_basic(self):
        assert _roundtrip(q.traverse("n1", depth=3)) == 'TRAVERSE FROM "n1" DEPTH 3'

    def test_with_where_limit(self):
        dsl = _roundtrip(q.traverse("n1", depth=3, where=F.eq("kind", "msg"), limit=10))
        assert 'TRAVERSE FROM "n1" DEPTH 3' in dsl
        assert 'WHERE kind = "msg"' in dsl
        assert "LIMIT 10" in dsl

    def test_negative_depth_raises(self):
        with pytest.raises(ValueError):
            q.traverse("n1", depth=-1)


class TestSubgraph:
    def test_basic(self):
        assert _roundtrip(q.subgraph("n1", depth=2)) == 'SUBGRAPH FROM "n1" DEPTH 2'


class TestPathFamily:
    def test_path(self):
        assert _roundtrip(q.path("a", "b", max_depth=5)) == 'PATH FROM "a" TO "b" MAX_DEPTH 5'

    def test_path_with_where(self):
        dsl = _roundtrip(q.path("a", "b", max_depth=5, where=F.eq("kind", "memory")))
        assert 'WHERE kind = "memory"' in dsl

    def test_paths(self):
        assert _roundtrip(q.paths("a", "b", max_depth=5)) == 'PATHS FROM "a" TO "b" MAX_DEPTH 5'

    def test_shortest_path_no_max(self):
        assert _roundtrip(q.shortest_path("a", "b")) == 'SHORTEST PATH FROM "a" TO "b"'

    def test_shortest_path_with_max(self):
        dsl = _roundtrip(q.shortest_path("a", "b", max_depth=5))
        assert "MAX_DEPTH 5" in dsl

    def test_distance(self):
        assert _roundtrip(q.distance("a", "b", max_depth=5)) == 'DISTANCE FROM "a" TO "b" MAX_DEPTH 5'

    def test_weighted_shortest_path(self):
        dsl = _roundtrip(q.weighted_shortest_path("a", "b", max_depth=5))
        assert dsl == 'WEIGHTED SHORTEST PATH FROM "a" TO "b" MAX_DEPTH 5'

    def test_weighted_distance(self):
        dsl = _roundtrip(q.weighted_distance("a", "b", max_depth=5))
        assert dsl == 'WEIGHTED DISTANCE FROM "a" TO "b" MAX_DEPTH 5'


class TestAncestorsDescendants:
    def test_ancestors(self):
        assert _roundtrip(q.ancestors("n1", depth=3)) == 'ANCESTORS OF "n1" DEPTH 3'

    def test_descendants(self):
        assert _roundtrip(q.descendants("n1", depth=3)) == 'DESCENDANTS OF "n1" DEPTH 3'

    def test_ancestors_with_where(self):
        dsl = _roundtrip(q.ancestors("n1", depth=3, where=F.eq("kind", "memory")))
        assert "WHERE" in dsl

    def test_negative_depth(self):
        with pytest.raises(ValueError):
            q.ancestors("n1", depth=-1)


class TestCommonNeighbors:
    def test_basic(self):
        assert _roundtrip(q.common_neighbors("a", "b")) == 'COMMON NEIGHBORS OF "a" AND "b"'

    def test_with_where(self):
        dsl = _roundtrip(q.common_neighbors("a", "b", where=F.eq("kind", "m")))
        assert "WHERE" in dsl


class TestMatch:
    def test_basic(self):
        dsl = _roundtrip(q.match('("fn_main") -[kind = "calls"]-> (callee)'))
        assert dsl.startswith("MATCH ")

    def test_with_limit(self):
        dsl = _roundtrip(q.match('("x") -[]-> (y)', limit=10))
        assert "LIMIT 10" in dsl

    def test_empty_pattern_raises(self):
        with pytest.raises(ValueError):
            q.match("")


class TestCounterfactual:
    def test_basic(self):
        assert _roundtrip(q.what_if_retract("fact:x")) == 'WHAT IF RETRACT "fact:x"'


class TestAggregate:
    def test_basic(self):
        dsl = _roundtrip(q.aggregate_nodes(select=["COUNT()"]))
        assert dsl == "AGGREGATE NODES SELECT COUNT()"

    def test_full(self):
        dsl = _roundtrip(q.aggregate_nodes(
            select=["COUNT()", "AVG(importance)"],
            where=F.eq("kind", "memory"),
            group_by=["topic"],
            limit=10,
        ))
        assert 'WHERE kind = "memory"' in dsl
        assert "GROUP BY topic" in dsl
        assert "SELECT COUNT(), AVG(importance)" in dsl
        assert "LIMIT 10" in dsl

    def test_empty_select_raises(self):
        with pytest.raises(ValueError):
            q.aggregate_nodes(select=[])

    def test_bad_group_by_identifier(self):
        with pytest.raises(ValueError):
            q.aggregate_nodes(select=["COUNT()"], group_by=["bad-name"])


class TestCountEdges:
    def test_basic(self):
        assert _roundtrip(q.count_edges()) == "COUNT EDGES"

    def test_with_where(self):
        dsl = _roundtrip(q.count_edges(where=F.eq("kind", "next")))
        assert 'WHERE kind = "next"' in dsl
