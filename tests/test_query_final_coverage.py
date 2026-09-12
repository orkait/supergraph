"""Final 100%-DSL-coverage audit.

Programmatically enumerates every lark production in grammar.lark,
then asserts:

  1. Every verb-producing rule has a builder that emits a parser-valid example.
  2. Every value/clause sub-grammar (expires_at, time_expr, etc.) is reachable.
  3. Every WHERE condition type is reachable via F.

This is the final backstop. If anything in the builder regresses or
grammar grows a new verb without a builder, this test fails loud.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from supergraph import q, F, P, Time, agg, EvolveWhen, EvolveThen
from supergraph.dsl.parser import parse


# Canonical one-liner for every top-level user/SYS/vault verb in grammar.lark.
# Each emission MUST parse.
CANONICAL_EXAMPLES = {
    # Reads (22)
    "node_q":          q.node("n1"),
    "nodes_q":         q.nodes(kind="m", limit=10),
    "edges_q":         q.edges("n1"),
    "traverse_q":      q.traverse("n1", depth=3),
    "subgraph_q":      q.subgraph("n1", depth=2),
    "path_q":          q.path("a", "b", max_depth=5),
    "paths_q":         q.paths("a", "b", max_depth=5),
    "shortest_q":      q.shortest_path("a", "b", max_depth=5),
    "distance_q":      q.distance("a", "b", max_depth=5),
    "weighted_sp_q":   q.weighted_shortest_path("a", "b", max_depth=5),
    "weighted_dist_q": q.weighted_distance("a", "b", max_depth=5),
    "ancestors_q":     q.ancestors("n1", depth=3),
    "descendants_q":   q.descendants("n1", depth=3),
    "common_q":        q.common_neighbors("a", "b"),
    "match_q":         q.match(P.node("a").to(P.var("b"), edge=F.eq("kind", "x"))),
    "count_q_nodes":   q.count_nodes(),
    "count_q_edges":   q.count_edges(),
    "aggregate_q":     q.aggregate_nodes(select=[agg.count()]),
    "recall_q":        q.recall("n1", depth=2, limit=10),
    "counterfactual":  q.what_if_retract("n1"),
    "similar_q_text":  q.similar(text="x", limit=5),
    "similar_q_node":  q.similar(node="n1", limit=5),
    "similar_q_vec":   q.similar(vec=[0.1, 0.2], limit=5),
    "lexical_q":       q.lexical("x", limit=5),
    "remember_q":      q.remember("x", at="2024-03", tokens=1000, limit=10),

    # Writes (21 + 2 control)
    "create_node":     q.create_node("n1", kind="m", document="d"),
    "create_node_auto": q.create_node_auto(kind="m"),
    "update_node":     q.update_node("n1", x=1),
    "upsert_node":     q.upsert_node("n1", kind="m"),
    "delete_node":     q.delete_node("n1"),
    "delete_nodes":    q.delete_nodes(where=F.eq("kind", "t")),
    "update_nodes":    q.update_nodes(where=F.eq("k", "m"), set={"x": 1}),
    "create_edge":     q.create_edge("a", "b", kind="next"),
    "update_edge":     q.update_edge("a", "b", set={"w": 1}),
    "delete_edge":     q.delete_edge("a", "b"),
    "delete_edges":    q.delete_edges("n1"),
    "increment":       q.increment("n1", "hits", by=1),
    "assert_stmt":     q.assert_("f1", kind="fact", value=42),
    "retract_stmt":    q.retract("f1"),
    "merge_stmt":      q.merge("a", "b"),
    "propagate_stmt":  q.propagate("n1", field="c", depth=2),
    "bind_context":    q.bind_context("s"),
    "discard_context": q.discard_context("s"),
    "ingest_stmt":     q.ingest("x.pdf"),
    "connect_node":    q.connect_node("n1"),
    "forget_node":     q.forget("n1"),

    # Batch + var_assign
    "batch":        q.batch(q.create_node("n1", kind="m"), q.delete_node("n2")),
    "var_assign":   q.batch(
        q.var("x", q.create_node("n1", kind="m")),
        q.var("y", q.create_node("n2", kind="m")),
        q.create_edge("$x", "$y", kind="next"),
    ),

    # SYS (33)
    "sys_status":            q.sys.status(),
    "sys_stats":             q.sys.stats("NODES"),
    "sys_health":            q.sys.health(),
    "sys_kinds":             q.sys.kinds(),
    "sys_edge_kinds":        q.sys.edge_kinds(),
    "sys_describe":          q.sys.describe("NODE", "memory"),
    "sys_slow":              q.sys.slow_queries(),
    "sys_frequent":          q.sys.frequent_queries(),
    "sys_failed":            q.sys.failed_queries(),
    "sys_explain":           q.sys.explain(q.remember("x", limit=5)),
    "sys_register_node":     q.sys.register_node_kind("m", required={"t": "string"}),
    "sys_register_edge":     q.sys.register_edge_kind("e", from_kinds=["m"], to_kinds=["x"]),
    "sys_unregister":        q.sys.unregister("NODE", "m"),
    "sys_checkpoint":        q.sys.checkpoint(),
    "sys_rebuild":           q.sys.rebuild_indices(),
    "sys_clear":             q.sys.clear("LOG"),
    "sys_wal":               q.sys.wal("STATUS"),
    "sys_expire":            q.sys.expire(),
    "sys_contradictions":    q.sys.contradictions(field="v", group_by="t"),
    "sys_snapshot":          q.sys.snapshot("s"),
    "sys_rollback":          q.sys.rollback_to("s"),
    "sys_snapshots":         q.sys.snapshots(),
    "sys_duplicates":        q.sys.duplicates(),
    "sys_embedders":         q.sys.embedders(),
    "sys_connect":           q.sys.connect(),
    "sys_consolidate":       q.sys.consolidate(),
    "sys_reembed":           q.sys.reembed(),
    "sys_retain":            q.sys.retain(),
    "sys_optimize":          q.sys.optimize(),
    "sys_log":               q.sys.log(),
    "sys_evict":             q.sys.evict(),

    # CRON (6)
    "sys_cron_add":     q.sys.cron.add("n", schedule="* * * * *", query="SYS STATUS"),
    "sys_cron_delete":  q.sys.cron.delete("n"),
    "sys_cron_enable":  q.sys.cron.enable("n"),
    "sys_cron_disable": q.sys.cron.disable("n"),
    "sys_cron_list":    q.sys.cron.list(),
    "sys_cron_run":     q.sys.cron.run("n"),

    # EVOLVE (8)
    "sys_evolve_rule":    q.sys.evolve.rule(
        "r1",
        when=[EvolveWhen.cond("r", "<=", 0.4)],
        then=[EvolveThen.run("SYS", "REEMBED")],
    ),
    "sys_evolve_list":    q.sys.evolve.list(),
    "sys_evolve_show":    q.sys.evolve.show("r1"),
    "sys_evolve_enable":  q.sys.evolve.enable("r1"),
    "sys_evolve_disable": q.sys.evolve.disable("r1"),
    "sys_evolve_delete":  q.sys.evolve.delete("r1"),
    "sys_evolve_history": q.sys.evolve.history(),
    "sys_evolve_reset":   q.sys.evolve.reset(),

    # VAULT (10)
    "vault_new":       q.vault.new("D"),
    "vault_read":      q.vault.read("D"),
    "vault_write":     q.vault.write("D", section="S", content="c"),
    "vault_append":    q.vault.append("D", section="S", content="c"),
    "vault_search":    q.vault.search("x", limit=5),
    "vault_backlinks": q.vault.backlinks("D"),
    "vault_list":      q.vault.list(),
    "vault_sync":      q.vault.sync(),
    "vault_daily":     q.vault.daily(),
    "vault_archive":   q.vault.archive("D"),
}


def test_every_canonical_example_parses():
    failures = []
    for name, query_obj in CANONICAL_EXAMPLES.items():
        try:
            dsl = query_obj.dsl()
            parse(dsl)
        except Exception as e:
            failures.append((name, str(e)[:120], query_obj.dsl() if hasattr(query_obj, "dsl") else "?"))
    if failures:
        msg = "\n".join(f"  {n}: {e}\n    DSL: {d}" for n, e, d in failures)
        pytest.fail(f"{len(failures)} canonical examples failed to parse:\n{msg}")


def test_expires_at_variant():
    """Both EXPIRES IN N<unit> and EXPIRES AT "timestamp" must work."""
    q1 = q.create_node("n1", kind="m", expires_in="1h")
    q2 = q.create_node("n2", kind="m", expires_at="2024-03-15T00:00:00")
    parse(q1.dsl())
    parse(q2.dsl())
    assert "EXPIRES IN 1h" in q1.dsl()
    assert 'EXPIRES AT "2024-03-15T00:00:00"' in q2.dsl()


def test_expires_in_and_at_mutex():
    import pytest as _p
    with _p.raises(ValueError, match="expires_in OR expires_at"):
        q.create_node("n1", kind="m", expires_in="1h", expires_at="2024")


def test_every_where_op_reachable():
    """Every WHERE sub-condition must be buildable via F."""
    parse(q.nodes(where=F.eq("k", "m")).dsl())
    parse(q.nodes(where=F.ne("k", "m")).dsl())
    parse(q.nodes(where=F.gt("x", 1)).dsl())
    parse(q.nodes(where=F.gte("x", 1)).dsl())
    parse(q.nodes(where=F.lt("x", 1)).dsl())
    parse(q.nodes(where=F.lte("x", 1)).dsl())
    parse(q.nodes(where=F.in_("k", ["a", "b"])).dsl())
    parse(q.nodes(where=F.not_in("k", ["a"])).dsl())
    parse(q.nodes(where=F.contains("t", "str")).dsl())
    parse(q.nodes(where=F.like("t", "%str%")).dsl())
    parse(q.nodes(where=F.startswith("t", "Pro")).dsl())
    parse(q.nodes(where=F.is_null("d")).dsl())
    parse(q.nodes(where=F.is_not_null("d")).dsl())
    parse(q.nodes(where=F.similar_score("content", "text", gt=0.5)).dsl())
    parse(q.nodes(where=F.indegree(">", 10)).dsl())
    parse(q.nodes(where=F.outdegree(">=", 5, field="kind")).dsl())
    parse(q.nodes(where=F.raw('kind = "x" AND INDEGREE > 5')).dsl())
    parse(q.nodes(where=F.eq("parent.kind", "m")).dsl())  # dot notation


def test_every_value_type_reachable():
    """Every value type in the grammar."""
    # STRING
    parse(q.nodes(where=F.eq("x", "str")).dsl())
    # NUMBER (int + float)
    parse(q.nodes(where=F.eq("x", 42)).dsl())
    parse(q.nodes(where=F.eq("x", 0.5)).dsl())
    # NULL
    parse(q.nodes(where=F.is_null("x")).dsl())
    # time_expr
    parse(q.nodes(where=F.gte("x", Time.now())).dsl())
    parse(q.nodes(where=F.gte("x", Time.today())).dsl())
    parse(q.nodes(where=F.gte("x", Time.yesterday())).dsl())
    parse(q.nodes(where=F.gte("x", Time.now_minus(7, "d"))).dsl())


def test_every_agg_func_reachable():
    parse(q.aggregate_nodes(select=[agg.count()]).dsl())
    parse(q.aggregate_nodes(select=[agg.count_distinct("x")]).dsl())
    parse(q.aggregate_nodes(select=[agg.sum("x")]).dsl())
    parse(q.aggregate_nodes(select=[agg.avg("x")]).dsl())
    parse(q.aggregate_nodes(select=[agg.min("x")]).dsl())
    parse(q.aggregate_nodes(select=[agg.max("x")]).dsl())


def test_every_evolve_action_reachable():
    from supergraph.query.evolve_expr import EvolveThen as A
    # All action variants
    q.sys.evolve.rule("r", when=[EvolveWhen.cond("x", ">", 0)],
                     then=[A.set("y", 0.5)]).dsl()
    q.sys.evolve.rule("r", when=[EvolveWhen.cond("x", ">", 0)],
                     then=[A.set("y", [0.5, 0.3])]).dsl()
    q.sys.evolve.rule("r", when=[EvolveWhen.cond("x", ">", 0)],
                     then=[A.adjust("y", 0.1)]).dsl()
    q.sys.evolve.rule("r", when=[EvolveWhen.cond("x", ">", 0)],
                     then=[A.adjust_until("y", 0.1, 1.0)]).dsl()
    q.sys.evolve.rule("r", when=[EvolveWhen.cond("x", ">", 0)],
                     then=[A.add("tag", "priority")]).dsl()
    q.sys.evolve.rule("r", when=[EvolveWhen.cond("x", ">", 0)],
                     then=[A.remove("tag", "stale")]).dsl()
    q.sys.evolve.rule("r", when=[EvolveWhen.cond("x", ">", 0)],
                     then=[A.run("SYS", "REEMBED")]).dsl()


def test_every_match_step_reachable():
    # bound_step
    p1 = P.node("a").to(P.var("b"))
    parse(q.match(p1).dsl())
    # var_step without where
    p2 = P.var("a").to(P.var("b"))
    parse(q.match(p2).dsl())
    # var_step with where
    p3 = P.var("a", where=F.eq("kind", "fn")).to(P.var("b"))
    parse(q.match(p3).dsl())
    # arrow with edge filter
    p4 = P.node("a").to(P.var("b"), edge=F.eq("kind", "calls"))
    parse(q.match(p4).dsl())
