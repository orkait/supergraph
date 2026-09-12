"""supergraph query builder: typed, composable, 100% DSL coverage.

Public surface:

  from supergraph.query import q, F, Query, register_verb

  q.remember("European history", limit=10)
  q.nodes(where=F.eq("kind", "memory") & F.gt("importance", 0.5))

User guide: https://supergraph-docs.orkait.com/query-builder
"""
from __future__ import annotations

from typing import Any

from supergraph.query import plugins
from supergraph.query.filters import F
from supergraph.query.runtime import Query
from supergraph.query.time_expr import Time, TimeExpr
from supergraph.query.pattern import P, Pattern
from supergraph.query.agg import agg, AggFunc, HavingExpr
from supergraph.query.evolve_expr import (
    EvolveWhen, EvolveThen, EvolveCondition, EvolveAction,
)
from supergraph.query.verbs import (
    reads as _reads,
    writes as _writes,
    traversal as _traversal,
    sys as _sys,
    cron as _cron,
    evolve as _evolve,
    vault as _vault,
)


class _CronNamespace:
    add     = staticmethod(_cron.add)
    delete  = staticmethod(_cron.delete)
    enable  = staticmethod(_cron.enable)
    disable = staticmethod(_cron.disable)
    run     = staticmethod(_cron.run)
    list    = staticmethod(_cron.list_)


class _EvolveNamespace:
    rule    = staticmethod(_evolve.rule)
    show    = staticmethod(_evolve.show)
    enable  = staticmethod(_evolve.enable)
    disable = staticmethod(_evolve.disable)
    delete  = staticmethod(_evolve.delete)
    history = staticmethod(_evolve.history)
    reset   = staticmethod(_evolve.reset)
    list    = staticmethod(_evolve.list_)


class _SysNamespace:
    # Scalar / inspection
    status       = staticmethod(_sys.status)
    stats        = staticmethod(_sys.stats)
    health       = staticmethod(_sys.health)
    kinds        = staticmethod(_sys.kinds)
    edge_kinds   = staticmethod(_sys.edge_kinds)
    describe     = staticmethod(_sys.describe)
    embedders    = staticmethod(_sys.embedders)
    # Query introspection
    slow_queries     = staticmethod(_sys.slow_queries)
    frequent_queries = staticmethod(_sys.frequent_queries)
    failed_queries   = staticmethod(_sys.failed_queries)
    explain          = staticmethod(_sys.explain)
    # Schema
    register_node_kind = staticmethod(_sys.register_node_kind)
    register_edge_kind = staticmethod(_sys.register_edge_kind)
    unregister         = staticmethod(_sys.unregister)
    # Maintenance
    checkpoint       = staticmethod(_sys.checkpoint)
    rebuild_indices  = staticmethod(_sys.rebuild_indices)
    clear            = staticmethod(_sys.clear)
    wal              = staticmethod(_sys.wal)
    expire           = staticmethod(_sys.expire)
    contradictions   = staticmethod(_sys.contradictions)
    snapshot         = staticmethod(_sys.snapshot)
    rollback_to      = staticmethod(_sys.rollback_to)
    snapshots        = staticmethod(_sys.snapshots)
    duplicates       = staticmethod(_sys.duplicates)
    connect          = staticmethod(_sys.connect)
    consolidate      = staticmethod(_sys.consolidate)
    reembed          = staticmethod(_sys.reembed)
    retain           = staticmethod(_sys.retain)
    optimize         = staticmethod(_sys.optimize)
    evict            = staticmethod(_sys.evict)
    log              = staticmethod(_sys.log)
    # Nested namespaces
    cron             = _CronNamespace()
    evolve           = _EvolveNamespace()


class _VaultNamespace:
    new       = staticmethod(_vault.new)
    read      = staticmethod(_vault.read)
    write     = staticmethod(_vault.write)
    append    = staticmethod(_vault.append)
    search    = staticmethod(_vault.search)
    backlinks = staticmethod(_vault.backlinks)
    list      = staticmethod(_vault.list_)
    sync      = staticmethod(_vault.sync)
    daily     = staticmethod(_vault.daily)
    archive   = staticmethod(_vault.archive)

register_verb = plugins.register_verb


class _QNamespace:
    """Attribute-access facade for verb functions + plugin fallthrough.

    ``q.remember(...)`` and ``q.create_node(...)`` look up built-in verbs;
    anything else falls through to the plugin registry so third-party
    packages can register custom verbs without modifying core.
    """

    # --- Reads ---
    node           = staticmethod(_reads.node)
    nodes          = staticmethod(_reads.nodes)
    remember       = staticmethod(_reads.remember)
    answer         = staticmethod(_reads.answer)
    recall         = staticmethod(_reads.recall)
    similar        = staticmethod(_reads.similar)
    lexical        = staticmethod(_reads.lexical)
    edges          = staticmethod(_reads.edges)
    count_nodes    = staticmethod(_reads.count_nodes)

    # --- Traversal ---
    traverse       = staticmethod(_traversal.traverse)
    subgraph       = staticmethod(_traversal.subgraph)
    path           = staticmethod(_traversal.path)
    paths          = staticmethod(_traversal.paths)
    shortest_path  = staticmethod(_traversal.shortest_path)
    distance       = staticmethod(_traversal.distance)
    weighted_shortest_path = staticmethod(_traversal.weighted_shortest_path)
    weighted_distance      = staticmethod(_traversal.weighted_distance)
    ancestors      = staticmethod(_traversal.ancestors)
    descendants    = staticmethod(_traversal.descendants)
    common_neighbors = staticmethod(_traversal.common_neighbors)
    match          = staticmethod(_traversal.match)
    what_if_retract = staticmethod(_traversal.what_if_retract)
    aggregate_nodes = staticmethod(_traversal.aggregate_nodes)
    count_edges    = staticmethod(_traversal.count_edges)

    # --- Writes ---
    create_node    = staticmethod(_writes.create_node)
    create_node_auto = staticmethod(_writes.create_node_auto)
    create_edge    = staticmethod(_writes.create_edge)
    delete_node    = staticmethod(_writes.delete_node)
    update_node    = staticmethod(_writes.update_node)
    upsert_node    = staticmethod(_writes.upsert_node)
    delete_nodes   = staticmethod(_writes.delete_nodes)
    update_nodes   = staticmethod(_writes.update_nodes)
    update_edge    = staticmethod(_writes.update_edge)
    delete_edge    = staticmethod(_writes.delete_edge)
    delete_edges   = staticmethod(_writes.delete_edges)
    increment      = staticmethod(_writes.increment)
    assert_        = staticmethod(_writes.assert_)
    retract        = staticmethod(_writes.retract)
    merge          = staticmethod(_writes.merge)
    propagate      = staticmethod(_writes.propagate)
    forget         = staticmethod(_writes.forget)
    connect_node   = staticmethod(_writes.connect_node)
    ingest         = staticmethod(_writes.ingest)

    # --- Control ---
    bind_context   = staticmethod(_writes.bind_context)
    discard_context = staticmethod(_writes.discard_context)
    begin          = staticmethod(_writes.begin)
    commit         = staticmethod(_writes.commit)
    batch          = staticmethod(_writes.batch)
    var            = staticmethod(_writes.var)

    # --- Namespaces ---
    sys            = _SysNamespace()
    vault          = _VaultNamespace()

    # --- Escape hatch ---
    @staticmethod
    def raw(dsl: str, **params: Any) -> Query:
        """Pass-through DSL with optional ``:name`` parameter substitution.

        Every referenced ``:name`` in ``dsl`` must be provided in ``params``;
        values are escaped through the normal DSL literal path.
        """
        from supergraph.query.escape import dsl_literal
        if not isinstance(dsl, str) or not dsl:
            raise ValueError("q.raw() requires a non-empty DSL string")
        import re
        # Match :name but not node-id colons inside quoted strings.
        # Strategy: only recognise :name when preceded by start/space/= /comma/open-paren.
        placeholders = set(re.findall(r"(?<=[\s=,(])\:(\w+)|(?<=^)\:(\w+)", dsl))
        # Also allow plain :name at any position as legacy pattern
        placeholders = set(re.findall(r":(\w+)", dsl)) & set(re.findall(r":([a-zA-Z_]\w*)", dsl))
        # Simpler: treat any :word as placeholder (callers should quote node ids themselves).
        placeholders = set(re.findall(r"(?<!\\):([a-zA-Z_]\w*)", dsl))
        missing = placeholders - set(params)
        if missing:
            raise ValueError(f"q.raw() missing params: {sorted(missing)}")
        unused = set(params) - placeholders
        if unused:
            raise ValueError(f"q.raw() unused params: {sorted(unused)}")
        out = dsl
        for name, value in params.items():
            out = out.replace(f":{name}", dsl_literal(value))
        return Query(_verb="raw", _params={"text": out}, _kind="raw")

    def __getattr__(self, name: str) -> Any:
        fn = plugins._lookup(name)
        if fn is not None:
            return fn
        known = sorted(n for n in dir(self) if not n.startswith("_"))
        registered = plugins._registered_names()
        raise AttributeError(
            f"q has no verb {name!r}. "
            f"Built-in: {known}. Registered: {registered}"
        )


q = _QNamespace()


__all__ = [
    "q", "F", "Query", "Time", "TimeExpr",
    "P", "Pattern",
    "agg", "AggFunc", "HavingExpr",
    "EvolveWhen", "EvolveThen", "EvolveCondition", "EvolveAction",
    "register_verb",
]
