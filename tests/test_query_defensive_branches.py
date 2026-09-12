"""Hits every defensive error / NotImplemented / dunder path not already
exercised by happy-path tests. Pushes coverage toward 100%.
"""
from __future__ import annotations

import pytest

from supergraph import q, F, P, agg, Time, EvolveWhen, EvolveThen, Query
from supergraph.query.filters import F as _F, _Leaf, _Degree
from supergraph.query.escape import dsl_field_ref, dsl_variable, dsl_node_ref
from supergraph.query.time_expr import Time as _Time


# -- F algebra defensive paths ---------------------------------------------

class TestFOperatorsWithWrongType:
    def test_and_with_non_F_returns_notimplemented(self):
        # Python's operator protocol: returning NotImplemented from __and__
        # makes Python try the right-hand type; if also fails -> TypeError.
        with pytest.raises(TypeError):
            _ = F.eq("k", "m") & 42

    def test_or_with_non_F_returns_notimplemented(self):
        with pytest.raises(TypeError):
            _ = F.eq("k", "m") | 42


class TestFBaseRaises:
    def test_base_to_dsl_raises(self):
        base = _F()  # instantiate raw base class
        with pytest.raises(NotImplementedError):
            base.to_dsl()


class TestFValidation:
    def test_not_in_requires_sequence(self):
        with pytest.raises(TypeError):
            F.not_in("topic", "not-a-list")

    def test_not_in_empty(self):
        with pytest.raises(ValueError):
            F.not_in("topic", [])

    def test_like_requires_str(self):
        with pytest.raises(TypeError):
            F.like("topic", 42)

    def test_similar_score_text_must_be_str(self):
        with pytest.raises(TypeError):
            F.similar_score("x", 42, gt=0.5)

    def test_similar_score_gt_must_be_number(self):
        with pytest.raises(TypeError):
            F.similar_score("x", "t", gt="bad")


class TestFDegreeInvalid:
    def test_degree_non_number_n(self):
        d = _Degree("INDEGREE", None, ">", "bad")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            d.to_dsl()


class TestFLeafUnknownOp:
    def test_unknown_op_raises_on_compile(self):
        leaf = _Leaf("x", "bogus_op", 1)
        with pytest.raises(ValueError, match="unknown op"):
            leaf.to_dsl()


class TestFFromDictBadKeys:
    def test_and_non_list(self):
        with pytest.raises(ValueError):
            F.from_dict({"__and__": "not-a-list"})

    def test_or_non_list(self):
        with pytest.raises(ValueError):
            F.from_dict({"__or__": "not-a-list"})

    def test_not_non_dict(self):
        with pytest.raises(ValueError):
            F.from_dict({"__not__": "not-a-dict"})

    def test_from_dict_multiple_and_keys(self):
        # Multi-entry dict compiles to AND
        f = F.from_dict({"k": "m", "x__gt": 0.5})
        out = f.to_dsl()
        assert "kind" not in out  # wasn't asked for
        assert "k = \"m\"" in out
        assert "x > 0.5" in out

    def test_from_dict_and_grouping(self):
        f = F.from_dict({"__and__": [{"a": 1}, {"b": 2}]})
        assert "a = 1" in f.to_dsl()
        assert "b = 2" in f.to_dsl()


# -- Escape helper defensive paths -----------------------------------------

class TestEscapeHelpers:
    def test_field_ref_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            dsl_field_ref("")

    def test_field_ref_non_str_raises(self):
        with pytest.raises(ValueError):
            dsl_field_ref(42)  # type: ignore[arg-type]

    def test_field_ref_too_many_dots(self):
        with pytest.raises(ValueError, match="at most one dot"):
            dsl_field_ref("a.b.c")

    def test_variable_empty_raises(self):
        with pytest.raises(ValueError):
            dsl_variable("")

    def test_variable_non_str_raises(self):
        with pytest.raises(ValueError):
            dsl_variable(42)  # type: ignore[arg-type]

    def test_node_ref_empty_raises(self):
        with pytest.raises(ValueError):
            dsl_node_ref("")

    def test_node_ref_non_str_raises(self):
        with pytest.raises(ValueError):
            dsl_node_ref(42)  # type: ignore[arg-type]


# -- TimeExpr dunder -------------------------------------------------------

class TestTimeExprRepr:
    def test_repr(self):
        t = _Time.now()
        assert "TimeExpr" in repr(t)
        assert "NOW" in repr(t)


# -- Pattern defensive paths -----------------------------------------------

class TestPatternTo:
    def test_to_multi_step_pattern_rejected(self):
        left = P.node("a")
        right = P.node("b").to(P.var("c"))
        with pytest.raises(ValueError, match="single-step"):
            left.to(right)

    def test_to_non_pattern_non_step(self):
        left = P.node("a")
        with pytest.raises(TypeError):
            left.to(42)  # type: ignore[arg-type]

    def test_to_with_edge_dict(self):
        p = P.node("a").to(P.var("b"), edge={"kind": "calls"})
        dsl = p.to_dsl()
        assert '-[kind = "calls"]->' in dsl

    def test_var_with_where_dict(self):
        p = P.var("x", where={"kind": "fn"})
        dsl = p.to_dsl()
        assert 'WHERE kind = "fn"' in dsl


# -- Query runtime defensive paths -----------------------------------------

class TestQueryUnknownCompiler:
    def test_compile_unknown_verb_raises(self):
        bad = Query(_verb="nonexistent_verb", _params={}, _kind="read")
        with pytest.raises(RuntimeError, match="no compiler"):
            bad.dsl()


class TestQueryModifiersValidation:
    def test_tokens_negative(self):
        with pytest.raises(ValueError):
            q.remember("x").tokens(-1)

    def test_at_empty(self):
        with pytest.raises(ValueError):
            q.remember("x").at("")

    def test_at_non_str(self):
        with pytest.raises(ValueError):
            q.remember("x").at(42)  # type: ignore[arg-type]

    def test_order_by_empty(self):
        with pytest.raises(ValueError):
            q.nodes().order_by("")

    def test_where_non_f_non_dict(self):
        with pytest.raises(TypeError):
            q.nodes().where(42)  # type: ignore[arg-type]


class TestQueryWithKwargNoneRemovesModifier:
    def test_with_limit_none_removes(self):
        base = q.nodes(limit=10)
        cleared = base.with_(limit=None)
        assert "LIMIT" not in cleared.dsl()

    def test_with_tokens_none_removes(self):
        base = q.remember("x", tokens=100)
        cleared = base.with_(tokens=None)
        assert "TOKENS" not in cleared.dsl()

    def test_with_at_none_removes(self):
        base = q.remember("x", at="2024")
        cleared = base.with_(at=None)
        assert "AT" not in cleared.dsl()

    def test_with_order_by_none_removes(self):
        base = q.nodes(order_by="x DESC")
        cleared = base.with_(order_by=None)
        assert "ORDER BY" not in cleared.dsl()

    def test_with_where_none_removes(self):
        base = q.nodes(where=F.eq("k", "m"))
        cleared = base.with_(where=None)
        assert "WHERE" not in cleared.dsl()


class TestQueryReprBadCompile:
    def test_repr_handles_compile_error(self):
        # A Query with an unregistered verb compiles-errors; repr must not raise
        bad = Query(_verb="definitely_not_registered", _params={}, _kind="read")
        r = repr(bad)
        assert "compile-error" in r or "Query" in r


# -- Verbs defensive paths -------------------------------------------------

class TestVerbDefensive:
    def test_ingest_empty_file_raises(self):
        with pytest.raises(ValueError):
            q.ingest("")

    def test_similar_all_three_given(self):
        with pytest.raises(ValueError):
            q.similar(text="a", node="b", vec=[0.1])

    def test_match_non_str_non_pattern(self):
        with pytest.raises(TypeError):
            q.match(42)  # type: ignore[arg-type]

    def test_match_empty_string(self):
        with pytest.raises(ValueError):
            q.match("")

    def test_match_whitespace_string(self):
        with pytest.raises(ValueError):
            q.match("   ")

    def test_aggregate_non_agg_non_str_select(self):
        with pytest.raises(TypeError):
            q.aggregate_nodes(select=[42])  # type: ignore[list-item]

    def test_aggregate_empty_select(self):
        with pytest.raises(ValueError):
            q.aggregate_nodes(select=[])

    def test_aggregate_empty_group_by(self):
        with pytest.raises(ValueError):
            q.aggregate_nodes(select=[agg.count()], group_by=[])

    def test_aggregate_bad_order_dir(self):
        with pytest.raises(ValueError):
            q.aggregate_nodes(select=[agg.count()], order_dir="SIDEWAYS")

    def test_evolve_rule_bad_when_item(self):
        with pytest.raises(TypeError):
            q.sys.evolve.rule("r", when=[42], then=["RUN X"])  # type: ignore[list-item]

    def test_evolve_rule_bad_then_item(self):
        with pytest.raises(TypeError):
            q.sys.evolve.rule("r", when=["x > 0"], then=[42])  # type: ignore[list-item]


class TestEvolveActionBadInput:
    def test_set_list_with_bool(self):
        a = EvolveThen.set("x", [True])
        with pytest.raises(TypeError):
            a.to_dsl()

    def test_set_list_empty(self):
        a = EvolveThen.set("x", [])
        with pytest.raises(ValueError):
            a.to_dsl()

    def test_set_unsupported_value_type(self):
        a = EvolveThen.set("x", "str-not-number")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            a.to_dsl()

    def test_run_empty_tokens(self):
        with pytest.raises(ValueError):
            EvolveThen.run()


class TestAggRepr:
    def test_str_returns_dsl(self):
        assert str(agg.count()) == "COUNT()"


class TestSysDefensive:
    def test_stats_bad_target(self):
        with pytest.raises(ValueError):
            q.sys.stats("GARBAGE")

    def test_optimize_bad_target(self):
        with pytest.raises(ValueError):
            q.sys.optimize("UNKNOWN")

    def test_clear_bad_target(self):
        with pytest.raises(ValueError):
            q.sys.clear("EDGES")  # grammar: only LOG/CACHE

    def test_wal_bad_action(self):
        with pytest.raises(ValueError):
            q.sys.wal("BACKUP")

    def test_describe_bad_entity(self):
        with pytest.raises(ValueError):
            q.sys.describe("SCHEMA", "x")

    def test_register_edge_kind_empty_from(self):
        with pytest.raises(ValueError):
            q.sys.register_edge_kind("e", from_kinds=[], to_kinds=["a"])


class TestTimeValidation:
    def test_now_minus_bad_unit(self):
        with pytest.raises(ValueError):
            Time.now_minus(1, "year")

    def test_now_minus_negative(self):
        with pytest.raises(ValueError):
            Time.now_minus(-1, "d")


class TestTraversalValidation:
    def test_traverse_negative_depth(self):
        with pytest.raises(ValueError):
            q.traverse("n", depth=-1)

    def test_subgraph_negative_depth(self):
        with pytest.raises(ValueError):
            q.subgraph("n", depth=-1)

    def test_path_negative_max_depth(self):
        with pytest.raises(ValueError):
            q.path("a", "b", max_depth=-1)

    def test_ancestors_negative_depth(self):
        with pytest.raises(ValueError):
            q.ancestors("n", depth=-1)

    def test_descendants_negative_depth(self):
        with pytest.raises(ValueError):
            q.descendants("n", depth=-1)


class TestWriteValidation:
    def test_delete_nodes_none_where(self):
        with pytest.raises(ValueError):
            q.delete_nodes(where=None)

    def test_update_nodes_empty_set(self):
        with pytest.raises(ValueError):
            q.update_nodes(where=F.eq("k", "m"), set={})

    def test_update_edge_empty_set(self):
        with pytest.raises(ValueError):
            q.update_edge("a", "b", set={})

    def test_delete_edges_bad_direction(self):
        with pytest.raises(ValueError):
            q.delete_edges("n", direction="SIDEWAYS")

    def test_edges_bad_direction(self):
        with pytest.raises(ValueError):
            q.edges("n", direction="SIDEWAYS")

    def test_increment_bool_by(self):
        with pytest.raises(ValueError):
            q.increment("n", "hits", by=True)

    def test_increment_non_number_by(self):
        with pytest.raises(ValueError):
            q.increment("n", "hits", by="1")  # type: ignore[arg-type]

    def test_propagate_negative_depth(self):
        with pytest.raises(ValueError):
            q.propagate("n", field="c", depth=-1)

    def test_ingest_vision_model_without_vision_using(self):
        with pytest.raises(ValueError):
            q.ingest("x.pdf", vision_model="smolvlm", using="pymupdf4llm")

    def test_ingest_unknown_using(self):
        with pytest.raises(ValueError):
            q.ingest("x.pdf", using="fantasy")

    def test_ingest_empty_file(self):
        with pytest.raises(ValueError):
            q.ingest("")

    def test_batch_empty(self):
        with pytest.raises(ValueError):
            q.batch()

    def test_var_non_write_query(self):
        with pytest.raises(ValueError):
            q.var("x", q.nodes())


class TestCreateNodeExpires:
    def test_expires_in_invalid_format(self):
        with pytest.raises(ValueError, match="NUMBER"):
            q.create_node("n", kind="m", expires_in="forever").dsl()

    def test_expires_in_non_str(self):
        with pytest.raises(ValueError):
            q.create_node("n", kind="m", expires_in=42).dsl()  # type: ignore[arg-type]


class TestCronValidation:
    def test_cron_add_empty_schedule(self):
        with pytest.raises(ValueError):
            q.sys.cron.add("n", schedule="", query="SYS STATUS")

    def test_cron_add_empty_query(self):
        with pytest.raises(ValueError):
            q.sys.cron.add("n", schedule="* * * * *", query="")


class TestEvolveValidation:
    def test_evolve_condition_invalid_op(self):
        c = EvolveWhen.cond("x", "~=", 1)
        with pytest.raises(ValueError, match="EVOLVE_OP"):
            c.to_dsl()

    def test_evolve_condition_non_numeric_value(self):
        c = EvolveWhen.cond("x", ">", "bad")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            c.to_dsl()

    def test_evolve_rule_empty_when(self):
        with pytest.raises(ValueError):
            q.sys.evolve.rule("r", when=[], then=["RUN X"])

    def test_evolve_rule_empty_then(self):
        with pytest.raises(ValueError):
            q.sys.evolve.rule("r", when=["x > 0"], then=[])


class TestRegisterVerbValidation:
    def test_register_verb_bad_name(self):
        from supergraph.query import register_verb
        with pytest.raises(ValueError):
            register_verb("not a valid identifier")(lambda: None)


class TestRawValidation:
    def test_raw_non_str(self):
        with pytest.raises(ValueError):
            q.raw(42)  # type: ignore[arg-type]

    def test_raw_empty(self):
        with pytest.raises(ValueError):
            q.raw("")
