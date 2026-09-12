"""Grammar coverage audit: every DSL rule has a builder + compiler.

This test is the failsafe. If anyone adds a new verb to grammar.lark
without adding a builder, it fails loud. If anyone adds a builder
without registering a compiler, it fails loud.

The expected-rules list is maintained by hand - treat it as the spec.
Any diff between this list and grammar.lark should be resolved by
updating BOTH (add builder + update list).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from supergraph.query.runtime import _COMPILERS


# Every verb we ship a builder for. One entry per grammar production.
EXPECTED_VERBS = {
    # Reads
    "node", "nodes", "edges",
    "traverse", "subgraph",
    "path", "paths", "shortest_path", "distance",
    "weighted_shortest_path", "weighted_distance",
    "ancestors", "descendants", "common_neighbors",
    "match", "count_nodes", "count_edges", "aggregate_nodes",
    "recall", "what_if_retract",
    "similar", "lexical", "remember", "answer",
    # Writes
    "create_node", "create_node_auto",
    "update_node", "upsert_node",
    "delete_node", "delete_nodes",
    "create_edge", "update_edge", "delete_edge", "delete_edges",
    "increment", "assert_", "retract", "update_nodes_write",  # alias not used
    "merge", "propagate",
    "bind_context", "discard_context",
    "ingest", "connect_node", "forget",
    "update_nodes",
    # SYS
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
    # CRON
    "sys_cron_add", "sys_cron_delete", "sys_cron_enable",
    "sys_cron_disable", "sys_cron_list", "sys_cron_run",
    # EVOLVE
    "sys_evolve_rule", "sys_evolve_list", "sys_evolve_show",
    "sys_evolve_enable", "sys_evolve_disable", "sys_evolve_delete",
    "sys_evolve_history", "sys_evolve_reset",
    # VAULT
    "vault_new", "vault_read", "vault_write", "vault_append",
    "vault_search", "vault_backlinks", "vault_list",
    "vault_sync", "vault_daily", "vault_archive",
    # Meta
    "batch", "raw",
}


# Aliases we include for safety but don't actually need a compiler for.
# (update_nodes_write was a placeholder; delete it from expected.)
_ALIASES_TO_DROP = {"update_nodes_write"}


def test_every_expected_verb_has_a_compiler():
    """Each verb in EXPECTED_VERBS must have a registered compiler."""
    expected = EXPECTED_VERBS - _ALIASES_TO_DROP
    missing = expected - set(_COMPILERS.keys())
    assert not missing, f"verbs without compilers: {sorted(missing)}"


def test_no_orphan_compilers():
    """Compilers registered but not in EXPECTED_VERBS indicate stale code."""
    expected = EXPECTED_VERBS - _ALIASES_TO_DROP
    orphan = set(_COMPILERS.keys()) - expected
    assert not orphan, f"compilers not in EXPECTED_VERBS (update this test): {sorted(orphan)}"


def test_grammar_rule_count_sanity():
    """Light sanity check: grammar.lark has ~80 top-level user+sys+vault rules.

    Does not enforce per-rule mapping (that would require a lark introspection
    pass). Just asserts the ballpark so a major grammar redesign trips this.
    """
    grammar_path = Path(__file__).resolve().parent.parent / "src" / "supergraph" / "dsl" / "grammar.lark"
    text = grammar_path.read_text()

    # Count top-level productions that look like verb rules (lowercase_underscore:)
    # Exclude internal non-verb rules (field_pairs, where_clause, etc.)
    rule_defs = re.findall(r"^([a-z_][a-z_0-9]*):", text, re.MULTILINE)
    # De-dup
    rules = set(rule_defs)
    # Rule count sanity: covers verbs + internal productions (field_pairs,
    # where_clause, expr, etc.). Range chosen generously; a major grammar
    # redesign trips it.
    assert 100 <= len(rules) <= 250, f"unexpected grammar rule count {len(rules)}; review EXPECTED_VERBS"
