from __future__ import annotations

import pytest

from supergraph import q, F, P, agg
from supergraph.dsl.parser import parse


def _rt(query_obj):
    dsl = query_obj.dsl()
    parse(dsl)
    return dsl


class TestWithKwargSetPaths:

    def test_with_where_sets(self):
        out = q.nodes().with_(where=F.eq("k", "m"))
        assert 'WHERE kind = "m"' not in out.dsl()
        assert 'k = "m"' in out.dsl()

    def test_with_tokens_sets(self):
        out = q.remember("x").with_(tokens=1000)
        assert "TOKENS 1000" in out.dsl()

    def test_with_at_sets(self):
        out = q.remember("x").with_(at="2024-01")
        assert 'AT "2024-01"' in out.dsl()


class TestReadsBranches:
    def test_nodes_kind_plus_where_dict(self):
        dsl = _rt(q.nodes(kind="memory", where={"topic": "travel"}, limit=5))
        assert 'kind = "memory"' in dsl
        assert 'topic = "travel"' in dsl

    def test_nodes_kind_plus_where_F(self):
        dsl = _rt(q.nodes(kind="memory", where=F.gt("importance", 0.5), limit=5))
        assert 'kind = "memory"' in dsl
        assert "importance > 0.5" in dsl

    def test_nodes_with_offset(self):
        dsl = _rt(q.nodes(kind="memory", limit=10, offset=5))
        assert "LIMIT 10 OFFSET 5" in dsl

    def test_edges_with_limit(self):
        dsl = _rt(q.edges("n1", limit=10))
        assert 'EDGES FROM "n1" LIMIT 10' in dsl


class TestSysBranches:
    def test_slow_with_limit(self):
        dsl = _rt(q.sys.slow_queries(limit=5))
        assert "LIMIT 5" in dsl

    def test_contradictions_non_dict_typed_ident(self):
        dsl = _rt(q.sys.register_node_kind("k", required=["a", "b"]))
        assert "REQUIRED a, b" in dsl

    def test_log_where_path(self):
        dsl = _rt(q.sys.log(where=F.eq("k", "m")))
        assert 'WHERE k = "m"' in dsl


class TestTraversalOrderByAgg:
    def test_aggregate_order_by_typed_agg(self):
        dsl = _rt(q.aggregate_nodes(
            select=[agg.count()],
            order_by=agg.count(),
            order_dir="DESC",
        ))
        assert "ORDER BY COUNT() DESC" in dsl

    def test_aggregate_order_by_string(self):
        dsl = _rt(q.aggregate_nodes(
            select=[agg.count()],
            order_by="COUNT()",
        ))
        assert "ORDER BY COUNT()" in dsl


class TestCreateNodeVector:
    def test_vector_clause_create_node(self):
        dsl = _rt(q.create_node("n1", kind="memory", vector=[0.1, 0.2, 0.3]))
        assert "VECTOR [0.1, 0.2, 0.3]" in dsl

    def test_vector_clause_create_node_auto(self):
        dsl = _rt(q.create_node_auto(kind="memory", vector=[0.5, 0.5]))
        assert "VECTOR [0.5, 0.5]" in dsl

    def test_vector_clause_upsert(self):
        dsl = _rt(q.upsert_node("n1", kind="memory", vector=[0.1, 0.1]))
        assert "VECTOR [0.1, 0.1]" in dsl


class TestWriteValidationBranches:
    def test_create_node_empty_kind(self):
        with pytest.raises(ValueError, match="non-empty"):
            q.create_node("n1", kind="")

    def test_create_node_auto_empty_kind(self):
        with pytest.raises(ValueError, match="non-empty"):
            q.create_node_auto(kind="")

    def test_create_edge_empty_kind(self):
        with pytest.raises(ValueError, match="non-empty"):
            q.create_edge("a", "b", kind="")

    def test_upsert_expires_mutex(self):
        with pytest.raises(ValueError, match="expires_in OR expires_at"):
            q.upsert_node("n1", kind="m", expires_in="1h", expires_at="2024")

    def test_delete_nodes_where_collapses_to_empty(self):
        from supergraph.query.verbs.writes import _compile_delete_nodes
        with pytest.raises(ValueError, match="empty WHERE"):
            _compile_delete_nodes({"where": {}})

    def test_update_nodes_none_where(self):
        with pytest.raises(ValueError, match="where"):
            q.update_nodes(where=None, set={"x": 1})

    def test_assert_empty_kind(self):
        with pytest.raises(ValueError, match="non-empty"):
            q.assert_("f1", kind="")


class TestEvolveRunEmpty:
    def test_evolve_run_at_compile_time_raises(self):
        from supergraph.query.evolve_expr import EvolveAction
        a = EvolveAction("run", None, ())
        with pytest.raises(ValueError, match="at least one identifier"):
            a.to_dsl()

    def test_evolve_unknown_kind(self):
        from supergraph.query.evolve_expr import EvolveAction
        a = EvolveAction("bogus", None, None)
        with pytest.raises(ValueError, match="unknown"):
            a.to_dsl()


class TestSysLastBranches:
    def test_frequent_with_limit(self):
        dsl = _rt(q.sys.frequent_queries(limit=10))
        assert "LIMIT 10" in dsl

    def test_log_since_path(self):
        dsl = _rt(q.sys.log(since="2024-01-01"))
        assert 'SINCE "2024-01-01"' in dsl

    def test_failed_with_limit(self):
        dsl = _rt(q.sys.failed_queries(limit=5))
        assert "LIMIT 5" in dsl


class TestCreateNodeAutoExpires:
    def test_auto_expires_in(self):
        dsl = _rt(q.create_node_auto(kind="m", expires_in="1h"))
        assert "EXPIRES IN 1h" in dsl

    def test_auto_expires_at(self):
        dsl = _rt(q.create_node_auto(kind="m", expires_at="2024-03-15"))
        assert 'EXPIRES AT "2024-03-15"' in dsl

    def test_auto_expires_mutex(self):
        with pytest.raises(ValueError, match="expires_in OR expires_at"):
            q.create_node_auto(kind="m", expires_in="1h", expires_at="2024")


class TestUpsertExpires:
    def test_upsert_expires_in(self):
        dsl = _rt(q.upsert_node("n1", kind="m", expires_in="30s"))
        assert "EXPIRES IN 30s" in dsl

    def test_upsert_expires_at(self):
        dsl = _rt(q.upsert_node("n1", kind="m", expires_at="2024-03-15"))
        assert 'EXPIRES AT "2024-03-15"' in dsl


class TestPatternToStepDirect:
    def test_to_accepts_bare_step(self):
        from supergraph.query.pattern import _Step
        right = _Step(bound_id=None, var_name="x", where=None)
        left = P.node("a")
        combined = left.to(right)
        dsl = combined.to_dsl()
        assert '("a") -[]-> (x)' in dsl
