from __future__ import annotations

import re
from pathlib import Path


from supergraph.query.runtime import _COMPILERS


EXPECTED_VERBS = {
    "node", "nodes", "edges",
    "traverse", "subgraph",
    "path", "paths", "shortest_path", "distance",
    "weighted_shortest_path", "weighted_distance",
    "ancestors", "descendants", "common_neighbors",
    "match", "count_nodes", "count_edges", "aggregate_nodes",
    "recall", "what_if_retract",
    "similar", "lexical", "remember", "answer",
    "create_node", "create_node_auto",
    "update_node", "upsert_node",
    "delete_node", "delete_nodes",
    "create_edge", "update_edge", "delete_edge", "delete_edges",
    "increment", "assert_", "retract", "update_nodes_write",
    "merge", "propagate",
    "bind_context", "discard_context",
    "ingest", "connect_node", "forget",
    "update_nodes",
    "sys_stats", "sys_kinds", "sys_edge_kinds", "sys_describe",
    "sys_slow", "sys_frequent", "sys_failed", "sys_explain",
    "sys_register_node_kind", "sys_register_edge_kind", "sys_unregister",
    "sys_checkpoint", "sys_rebuild", "sys_clear", "sys_wal",
    "sys_expire", "sys_contradictions",
    "sys_snapshot", "sys_rollback", "sys_snapshots",
    "sys_duplicates", "sys_embedders",
    "sys_connect", "sys_consolidate", "sys_reembed",
    "sys_status", "sys_retain", "sys_health",
    "sys_optimize", "sys_evict", "sys_log",
    "sys_cron_add", "sys_cron_delete", "sys_cron_enable",
    "sys_cron_disable", "sys_cron_list", "sys_cron_run",
    "sys_evolve_rule", "sys_evolve_list", "sys_evolve_show",
    "sys_evolve_enable", "sys_evolve_disable", "sys_evolve_delete",
    "sys_evolve_history", "sys_evolve_reset",
    "vault_new", "vault_read", "vault_write", "vault_append",
    "vault_search", "vault_backlinks", "vault_list",
    "vault_sync", "vault_daily", "vault_archive",
    "batch", "raw",
}


_ALIASES_TO_DROP = {"update_nodes_write"}


def test_every_expected_verb_has_a_compiler():
    expected = EXPECTED_VERBS - _ALIASES_TO_DROP
    missing = expected - set(_COMPILERS.keys())
    assert not missing, f"verbs without compilers: {sorted(missing)}"


def test_no_orphan_compilers():
    expected = EXPECTED_VERBS - _ALIASES_TO_DROP
    orphan = set(_COMPILERS.keys()) - expected
    assert not orphan, f"compilers not in EXPECTED_VERBS (update this test): {sorted(orphan)}"


def test_grammar_rule_count_sanity():
    grammar_path = Path(__file__).resolve().parent.parent / "src" / "supergraph" / "dsl" / "grammar.lark"
    text = grammar_path.read_text()

    rule_defs = re.findall(r"^([a-z_][a-z_0-9]*):", text, re.MULTILINE)
    rules = set(rule_defs)
    assert 100 <= len(rules) <= 250, f"unexpected grammar rule count {len(rules)}; review EXPECTED_VERBS"
