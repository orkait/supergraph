import pytest

from supergraph import q, F, P, agg, EvolveWhen, EvolveThen
from supergraph.dsl.parser import parse


def _roundtrip(query_obj):
    dsl = query_obj.dsl()
    try:
        parse(dsl)
    except Exception as e:
        pytest.fail(f"parser rejected {dsl!r}: {e}")
    return dsl


class TestPattern:
    def test_single_bound_step(self):
        p = P.node("fn_main")
        assert p.to_dsl() == '("fn_main")'

    def test_single_step_rejected_by_match(self):
        p = P.node("fn_main")
        with pytest.raises(ValueError, match="at least one arrow"):
            q.match(p)

    def test_single_var_step(self):
        p = P.var("callee")
        assert p.to_dsl() == "(callee)"

    def test_var_with_where(self):
        p = P.var("callee", where=F.eq("kind", "fn"))
        assert p.to_dsl() == '(callee WHERE kind = "fn")'

    def test_two_step_pattern_no_edge(self):
        p = P.node("fn_main").to(P.var("callee"))
        dsl = _roundtrip(q.match(p))
        assert 'MATCH ("fn_main") -[]-> (callee)' in dsl

    def test_two_step_with_edge_filter(self):
        p = P.node("fn_main").to(P.var("callee"), edge=F.eq("kind", "calls"))
        dsl = _roundtrip(q.match(p))
        assert 'MATCH ("fn_main") -[kind = "calls"]-> (callee)' in dsl

    def test_multi_hop(self):
        p = (P.node("a")
             .to(P.var("b"), edge=F.eq("kind", "r1"))
             .to(P.var("c")))
        dsl = _roundtrip(q.match(p))
        assert '("a") -[kind = "r1"]-> (b) -[]-> (c)' in dsl

    def test_match_still_accepts_string(self):
        dsl = _roundtrip(q.match('("a") -[]-> (b)'))
        assert "MATCH" in dsl

    def test_pattern_immutability(self):
        base = P.node("a")
        extended = base.to(P.var("b"))
        assert base.to_dsl() == '("a")'
        assert extended.to_dsl() == '("a") -[]-> (b)'

    def test_pattern_rejects_bad_var_name(self):
        with pytest.raises(ValueError):
            P.var("1bad")

    def test_pattern_rejects_empty_node_id(self):
        with pytest.raises(ValueError):
            P.node("")


class TestAgg:
    def test_count(self):
        assert agg.count().to_dsl() == "COUNT()"

    def test_count_distinct(self):
        assert agg.count_distinct("topic").to_dsl() == "COUNT DISTINCT(topic)"

    def test_sum_avg_min_max(self):
        assert agg.sum("x").to_dsl() == "SUM(x)"
        assert agg.avg("x").to_dsl() == "AVG(x)"
        assert agg.min("x").to_dsl() == "MIN(x)"
        assert agg.max("x").to_dsl() == "MAX(x)"

    def test_agg_invalid_field(self):
        with pytest.raises(ValueError):
            agg.sum("bad-field")

    def test_agg_in_select(self):
        dsl = _roundtrip(q.aggregate_nodes(
            select=[agg.count(), agg.avg("importance")],
            group_by=["topic"],
        ))
        assert "SELECT COUNT(), AVG(importance)" in dsl

    def test_agg_mixed_string_and_typed_in_select(self):
        dsl = _roundtrip(q.aggregate_nodes(
            select=[agg.count(), "AVG(importance)"],
        ))
        assert "SELECT COUNT(), AVG(importance)" in dsl

    def test_having_via_comparison(self):
        h = agg.avg("importance") > 0.5
        assert h.to_dsl() == "AVG(importance) > 0.5"

    def test_having_in_aggregate(self):
        dsl = _roundtrip(q.aggregate_nodes(
            select=[agg.count()],
            having=agg.count() >= 10,
        ))
        assert "HAVING COUNT() >= 10" in dsl


class TestEvolveTyped:
    def test_cond_basic(self):
        c = EvolveWhen.cond("recall_hit_rate", "<=", 0.4)
        assert c.to_dsl() == "recall_hit_rate <= 0.4"

    def test_cond_invalid_op(self):
        with pytest.raises(ValueError):
            EvolveWhen.cond("x", "~=", 0.1).to_dsl()

    def test_then_set_scalar(self):
        a = EvolveThen.set("target", 0.5)
        assert a.to_dsl() == "SET target = 0.5"

    def test_then_set_list(self):
        a = EvolveThen.set("weights", [0.5, 0.3, 0.2])
        assert a.to_dsl() == "SET weights = [0.5, 0.3, 0.2]"

    def test_then_adjust(self):
        assert EvolveThen.adjust("x", 0.1).to_dsl() == "ADJUST x BY 0.1"

    def test_then_adjust_until(self):
        assert EvolveThen.adjust_until("x", 0.1, 1.0).to_dsl() == "ADJUST x BY 0.1 UNTIL 1.0"

    def test_then_add(self):
        assert EvolveThen.add("tag", "priority").to_dsl() == 'ADD tag "priority"'

    def test_then_remove(self):
        assert EvolveThen.remove("tag", "stale").to_dsl() == 'REMOVE tag "stale"'

    def test_then_run(self):
        assert EvolveThen.run("SYS", "REEMBED").to_dsl() == "RUN SYS REEMBED"

    def test_rule_typed_end_to_end(self):
        dsl = _roundtrip(q.sys.evolve.rule(
            "r1",
            when=[EvolveWhen.cond("recall_hit_rate", "<=", 0.4)],
            then=[EvolveThen.run("SYS", "REEMBED")],
            cooldown=86400,
            priority=1,
        ))
        assert 'SYS EVOLVE RULE "r1"' in dsl
        assert "WHEN recall_hit_rate <= 0.4" in dsl
        assert "THEN RUN SYS REEMBED" in dsl
        assert "COOLDOWN 86400" in dsl
        assert "PRIORITY 1" in dsl

    def test_rule_mixed_typed_and_strings(self):
        dsl = _roundtrip(q.sys.evolve.rule(
            "r2",
            when=["x > 0.5", EvolveWhen.cond("y", "<", 1.0)],
            then=[EvolveThen.set("z", 0.1)],
        ))
        assert "WHEN x > 0.5 AND y < 1.0" in dsl
        assert "THEN SET z = 0.1" in dsl

    def test_rule_rejects_non_expression_types(self):
        with pytest.raises(TypeError):
            q.sys.evolve.rule("r", when=[42], then=["RUN X"])


class TestImmutability:
    def test_pattern_chain_immutable(self):
        base = P.node("a")
        _ = base.to(P.var("b"))
        assert base.to_dsl() == '("a")'

    def test_agg_comparison_no_mutation(self):
        fn = agg.avg("x")
        _ = fn > 0.5
        assert fn.to_dsl() == "AVG(x)"

    def test_evolve_cond_immutable(self):
        c = EvolveWhen.cond("x", ">", 0.5)
        with pytest.raises(Exception):
            c.value = 99
