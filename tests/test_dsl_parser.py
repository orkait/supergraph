
import pytest
from supergraph.dsl.parser import parse, parse_uncached, clear_cache, _plan_cache
from supergraph.dsl.ast_nodes import (
    NodeQuery, NodesQuery, EdgesQuery, TraverseQuery, SubgraphQuery,
    PathQuery, PathsQuery, ShortestPathQuery, DistanceQuery,
    AncestorsQuery, DescendantsQuery, CommonNeighborsQuery,
    MatchQuery, MatchPattern, CreateNode, UpdateNode, UpsertNode, DeleteNode, DeleteNodes,
    CreateEdge, DeleteEdge, DeleteEdges, Increment, Batch,
    FieldPair, Condition, DegreeCondition,
    NotExpr, AndExpr, OrExpr, WhereClause, SysStats, SysKinds, SysEdgeKinds, SysDescribe,
    SysSlowQueries, SysFrequentQueries, SysFailedQueries,
    SysExplain, SysRegisterNodeKind, SysRegisterEdgeKind,
    SysUnregister, SysCheckpoint, SysRebuild, SysClear, SysWal,
)
from supergraph.core.errors import QueryError


class TestNodeQuery:
    def test_node_simple(self):
        r = parse('NODE "user:42"')
        assert isinstance(r, NodeQuery)
        assert r.id == "user:42"


class TestNodesQuery:
    def test_nodes_with_where_and_limit(self):
        r = parse('NODES WHERE kind = "function" LIMIT 10')
        assert isinstance(r, NodesQuery)
        assert r.where is not None
        assert isinstance(r.where, WhereClause)
        cond = r.where.expr
        assert isinstance(cond, Condition)
        assert cond.field == "kind"
        assert cond.op == "="
        assert cond.value == "function"
        assert r.limit is not None
        assert r.limit.value == 10

    def test_nodes_no_filter(self):
        r = parse("NODES")
        assert isinstance(r, NodesQuery)
        assert r.where is None
        assert r.limit is None

    def test_nodes_limit_only(self):
        r = parse("NODES LIMIT 5")
        assert isinstance(r, NodesQuery)
        assert r.where is None
        assert r.limit.value == 5


class TestEdgesQuery:
    def test_edges_from_with_where(self):
        r = parse('EDGES FROM "user:1" WHERE kind = "calls"')
        assert isinstance(r, EdgesQuery)
        assert r.direction == "FROM"
        assert r.node_id == "user:1"
        assert r.where is not None
        cond = r.where.expr
        assert isinstance(cond, Condition)
        assert cond.field == "kind"
        assert cond.value == "calls"

    def test_edges_to(self):
        r = parse('EDGES TO "user:1"')
        assert isinstance(r, EdgesQuery)
        assert r.direction == "TO"
        assert r.node_id == "user:1"
        assert r.where is None


class TestTraverseQuery:
    def test_traverse_with_where(self):
        r = parse('TRAVERSE FROM "root" DEPTH 3 WHERE kind = "calls"')
        assert isinstance(r, TraverseQuery)
        assert r.start_id == "root"
        assert r.depth == 3
        assert r.where is not None

    def test_traverse_no_where(self):
        r = parse('TRAVERSE FROM "root" DEPTH 2')
        assert isinstance(r, TraverseQuery)
        assert r.start_id == "root"
        assert r.depth == 2
        assert r.where is None


class TestSubgraphQuery:
    def test_subgraph(self):
        r = parse('SUBGRAPH FROM "node:1" DEPTH 2')
        assert isinstance(r, SubgraphQuery)
        assert r.start_id == "node:1"
        assert r.depth == 2


class TestPathQuery:
    def test_path(self):
        r = parse('PATH FROM "a" TO "b" MAX_DEPTH 5')
        assert isinstance(r, PathQuery)
        assert r.from_id == "a"
        assert r.to_id == "b"
        assert r.max_depth == 5
        assert r.where is None

    def test_path_with_where(self):
        r = parse('PATH FROM "a" TO "b" MAX_DEPTH 5 WHERE kind = "calls"')
        assert isinstance(r, PathQuery)
        assert r.where is not None


class TestPathsQuery:
    def test_paths(self):
        r = parse('PATHS FROM "a" TO "b" MAX_DEPTH 4')
        assert isinstance(r, PathsQuery)
        assert r.from_id == "a"
        assert r.to_id == "b"
        assert r.max_depth == 4
        assert r.where is None


class TestShortestPathQuery:
    def test_shortest_path(self):
        r = parse('SHORTEST PATH FROM "a" TO "b"')
        assert isinstance(r, ShortestPathQuery)
        assert r.from_id == "a"
        assert r.to_id == "b"
        assert r.where is None

    def test_shortest_path_with_where(self):
        r = parse('SHORTEST PATH FROM "a" TO "b" WHERE kind = "calls"')
        assert isinstance(r, ShortestPathQuery)
        assert r.where is not None


class TestDistanceQuery:
    def test_distance(self):
        r = parse('DISTANCE FROM "a" TO "b" MAX_DEPTH 10')
        assert isinstance(r, DistanceQuery)
        assert r.from_id == "a"
        assert r.to_id == "b"
        assert r.max_depth == 10


class TestAncestorsQuery:
    def test_ancestors(self):
        r = parse('ANCESTORS OF "node:1" DEPTH 3')
        assert isinstance(r, AncestorsQuery)
        assert r.node_id == "node:1"
        assert r.depth == 3
        assert r.where is None

    def test_ancestors_with_where(self):
        r = parse('ANCESTORS OF "node:1" DEPTH 3 WHERE kind = "calls"')
        assert isinstance(r, AncestorsQuery)
        assert r.where is not None


class TestDescendantsQuery:
    def test_descendants(self):
        r = parse('DESCENDANTS OF "node:1" DEPTH 3')
        assert isinstance(r, DescendantsQuery)
        assert r.node_id == "node:1"
        assert r.depth == 3
        assert r.where is None


class TestCommonNeighborsQuery:
    def test_common_neighbors(self):
        r = parse('COMMON NEIGHBORS OF "a" AND "b"')
        assert isinstance(r, CommonNeighborsQuery)
        assert r.node_a == "a"
        assert r.node_b == "b"
        assert r.where is None

    def test_common_neighbors_with_where(self):
        r = parse('COMMON NEIGHBORS OF "a" AND "b" WHERE kind = "calls"')
        assert isinstance(r, CommonNeighborsQuery)
        assert r.where is not None


class TestMatchQuery:
    def test_match_basic(self):
        r = parse('MATCH ("src") -[kind = "calls"]-> (x) LIMIT 10')
        assert isinstance(r, MatchQuery)
        assert isinstance(r.pattern, MatchPattern)
        assert len(r.pattern.steps) == 2
        assert len(r.pattern.arrows) == 1
        assert r.pattern.steps[0].bound_id == "src"
        assert r.pattern.steps[1].variable == "x"
        assert isinstance(r.pattern.arrows[0].expr, Condition)
        assert r.pattern.arrows[0].expr.field == "kind"
        assert r.pattern.arrows[0].expr.value == "calls"
        assert r.limit is not None
        assert r.limit.value == 10

    def test_match_no_limit(self):
        r = parse('MATCH ("a") -[kind = "knows"]-> (b)')
        assert isinstance(r, MatchQuery)
        assert r.limit is None


class TestCreateNode:
    def test_create_node(self):
        r = parse('CREATE NODE "user:1" kind = "function" name = "foo"')
        assert isinstance(r, CreateNode)
        assert r.id == "user:1"
        assert len(r.fields) == 2
        assert r.fields[0] == FieldPair(name="kind", value="function")
        assert r.fields[1] == FieldPair(name="name", value="foo")

    def test_create_node_no_fields(self):
        r = parse('CREATE NODE "user:1"')
        assert isinstance(r, CreateNode)
        assert r.id == "user:1"
        assert r.fields == []


class TestUpdateNode:
    def test_update_node(self):
        r = parse('UPDATE NODE "user:1" SET name = "bar"')
        assert isinstance(r, UpdateNode)
        assert r.id == "user:1"
        assert len(r.fields) == 1
        assert r.fields[0] == FieldPair(name="name", value="bar")


class TestUpsertNode:
    def test_upsert_node(self):
        r = parse('UPSERT NODE "user:1" kind = "function"')
        assert isinstance(r, UpsertNode)
        assert r.id == "user:1"
        assert len(r.fields) == 1
        assert r.fields[0] == FieldPair(name="kind", value="function")


class TestDeleteNode:
    def test_delete_node(self):
        r = parse('DELETE NODE "user:1"')
        assert isinstance(r, DeleteNode)
        assert r.id == "user:1"


class TestDeleteNodes:
    def test_delete_nodes(self):
        r = parse('DELETE NODES WHERE kind = "test"')
        assert isinstance(r, DeleteNodes)
        assert r.where is not None
        cond = r.where.expr
        assert isinstance(cond, Condition)
        assert cond.field == "kind"
        assert cond.value == "test"


class TestCreateEdge:
    def test_create_edge(self):
        r = parse('CREATE EDGE "src" -> "tgt" kind = "calls"')
        assert isinstance(r, CreateEdge)
        assert r.source == "src"
        assert r.target == "tgt"
        assert len(r.fields) == 1
        assert r.fields[0] == FieldPair(name="kind", value="calls")

    def test_create_edge_no_fields(self):
        r = parse('CREATE EDGE "src" -> "tgt"')
        assert isinstance(r, CreateEdge)
        assert r.fields == []


class TestDeleteEdge:
    def test_delete_edge(self):
        r = parse('DELETE EDGE "src" -> "tgt"')
        assert isinstance(r, DeleteEdge)
        assert r.source == "src"
        assert r.target == "tgt"
        assert r.where is None

    def test_delete_edge_with_where(self):
        r = parse('DELETE EDGE "src" -> "tgt" WHERE kind = "calls"')
        assert isinstance(r, DeleteEdge)
        assert r.where is not None


class TestDeleteEdges:
    def test_delete_edges_from(self):
        r = parse('DELETE EDGES FROM "src" WHERE kind = "calls"')
        assert isinstance(r, DeleteEdges)
        assert r.direction == "FROM"
        assert r.node_id == "src"
        assert r.where is not None

    def test_delete_edges_no_where(self):
        r = parse('DELETE EDGES FROM "src"')
        assert isinstance(r, DeleteEdges)
        assert r.direction == "FROM"
        assert r.where is None


class TestIncrement:
    def test_increment(self):
        r = parse('INCREMENT NODE "user:1" hits BY 1')
        assert isinstance(r, Increment)
        assert r.node_id == "user:1"
        assert r.field == "hits"
        assert r.amount == 1

    def test_increment_float(self):
        r = parse('INCREMENT NODE "user:1" score BY 0.5')
        assert isinstance(r, Increment)
        assert r.amount == 0.5


class TestBatch:
    def test_batch(self):
        q = 'BEGIN\nCREATE NODE "a" kind = "x"\nCREATE NODE "b" kind = "y"\nCOMMIT'
        r = parse(q)
        assert isinstance(r, Batch)
        assert len(r.statements) == 2
        assert isinstance(r.statements[0], CreateNode)
        assert isinstance(r.statements[1], CreateNode)


class TestFilterExpressions:
    def test_and_expr(self):
        r = parse('NODES WHERE kind = "function" AND name = "foo"')
        assert isinstance(r, NodesQuery)
        expr = r.where.expr
        assert isinstance(expr, AndExpr)
        assert len(expr.operands) == 2
        assert isinstance(expr.operands[0], Condition)
        assert isinstance(expr.operands[1], Condition)

    def test_or_expr(self):
        r = parse('NODES WHERE kind = "function" OR kind = "method"')
        assert isinstance(r, NodesQuery)
        expr = r.where.expr
        assert isinstance(expr, OrExpr)
        assert len(expr.operands) == 2

    def test_not_expr(self):
        r = parse('NODES WHERE NOT kind = "test"')
        assert isinstance(r, NodesQuery)
        expr = r.where.expr
        assert isinstance(expr, NotExpr)
        assert isinstance(expr.operand, Condition)

    def test_complex_expr(self):
        r = parse('NODES WHERE kind = "function" AND (name = "foo" OR name = "bar")')
        assert isinstance(r, NodesQuery)
        expr = r.where.expr
        assert isinstance(expr, AndExpr)
        assert len(expr.operands) == 2
        assert isinstance(expr.operands[0], Condition)
        assert isinstance(expr.operands[1], OrExpr)

    def test_null_value(self):
        r = parse("NODES WHERE name = NULL")
        assert isinstance(r, NodesQuery)
        cond = r.where.expr
        assert isinstance(cond, Condition)
        assert cond.value is None

    def test_comparison_operators(self):
        r = parse('NODES WHERE count > 5')
        cond = r.where.expr
        assert cond.op == ">"
        assert cond.value == 5

        r = parse('NODES WHERE count >= 5')
        cond = r.where.expr
        assert cond.op == ">="

        r = parse('NODES WHERE count < 5')
        cond = r.where.expr
        assert cond.op == "<"

        r = parse('NODES WHERE count <= 5')
        cond = r.where.expr
        assert cond.op == "<="

        r = parse('NODES WHERE count != 5')
        cond = r.where.expr
        assert cond.op == "!="

    def test_indegree_condition(self):
        r = parse("NODES WHERE INDEGREE > 5")
        assert isinstance(r, NodesQuery)
        cond = r.where.expr
        assert isinstance(cond, DegreeCondition)
        assert cond.degree_type == "INDEGREE"
        assert cond.edge_kind is None
        assert cond.op == ">"
        assert cond.value == 5

    def test_outdegree_condition(self):
        r = parse("NODES WHERE OUTDEGREE >= 3")
        assert isinstance(r, NodesQuery)
        cond = r.where.expr
        assert isinstance(cond, DegreeCondition)
        assert cond.degree_type == "OUTDEGREE"
        assert cond.op == ">="
        assert cond.value == 3

    def test_degree_with_edge_kind(self):
        r = parse("NODES WHERE INDEGREE calls > 5")
        assert isinstance(r, NodesQuery)
        cond = r.where.expr
        assert isinstance(cond, DegreeCondition)
        assert cond.degree_type == "INDEGREE"
        assert cond.edge_kind == "calls"
        assert cond.op == ">"
        assert cond.value == 5

    def test_numeric_value(self):
        r = parse("NODES WHERE count = 42")
        cond = r.where.expr
        assert cond.value == 42
        assert isinstance(cond.value, int)

    def test_float_value(self):
        r = parse("NODES WHERE score > 3.14")
        cond = r.where.expr
        assert cond.value == 3.14
        assert isinstance(cond.value, float)


class TestSystemQueries:
    def test_sys_stats(self):
        r = parse("SYS STATS")
        assert isinstance(r, SysStats)
        assert r.target is None

    def test_sys_stats_nodes(self):
        r = parse("SYS STATS NODES")
        assert isinstance(r, SysStats)
        assert r.target == "NODES"

    def test_sys_stats_edges(self):
        r = parse("SYS STATS EDGES")
        assert isinstance(r, SysStats)
        assert r.target == "EDGES"

    def test_sys_stats_memory(self):
        r = parse("SYS STATS MEMORY")
        assert isinstance(r, SysStats)
        assert r.target == "MEMORY"

    def test_sys_stats_wal(self):
        r = parse("SYS STATS WAL")
        assert isinstance(r, SysStats)
        assert r.target == "WAL"

    def test_sys_kinds(self):
        r = parse("SYS KINDS")
        assert isinstance(r, SysKinds)

    def test_sys_edge_kinds(self):
        r = parse("SYS EDGE KINDS")
        assert isinstance(r, SysEdgeKinds)

    def test_sys_describe_node(self):
        r = parse('SYS DESCRIBE NODE "function"')
        assert isinstance(r, SysDescribe)
        assert r.entity_type == "NODE"
        assert r.name == "function"

    def test_sys_describe_edge(self):
        r = parse('SYS DESCRIBE EDGE "calls"')
        assert isinstance(r, SysDescribe)
        assert r.entity_type == "EDGE"
        assert r.name == "calls"

    def test_sys_slow_queries(self):
        r = parse("SYS SLOW QUERIES")
        assert isinstance(r, SysSlowQueries)
        assert r.since is None
        assert r.limit is None

    def test_sys_slow_queries_with_since(self):
        r = parse('SYS SLOW QUERIES SINCE "2024-01-01"')
        assert isinstance(r, SysSlowQueries)
        assert r.since == "2024-01-01"

    def test_sys_slow_queries_with_limit(self):
        r = parse("SYS SLOW QUERIES LIMIT 5")
        assert isinstance(r, SysSlowQueries)
        assert r.limit is not None
        assert r.limit.value == 5

    def test_sys_frequent_queries(self):
        r = parse("SYS FREQUENT QUERIES")
        assert isinstance(r, SysFrequentQueries)
        assert r.limit is None

    def test_sys_frequent_queries_with_limit(self):
        r = parse("SYS FREQUENT QUERIES LIMIT 10")
        assert isinstance(r, SysFrequentQueries)
        assert r.limit.value == 10

    def test_sys_failed_queries(self):
        r = parse("SYS FAILED QUERIES")
        assert isinstance(r, SysFailedQueries)
        assert r.limit is None

    def test_sys_explain(self):
        r = parse('SYS EXPLAIN NODE "user:1"')
        assert isinstance(r, SysExplain)
        assert isinstance(r.query, NodeQuery)
        assert r.query.id == "user:1"

    def test_sys_register_node_kind(self):
        r = parse('SYS REGISTER NODE KIND "function" REQUIRED kind, name')
        assert isinstance(r, SysRegisterNodeKind)
        assert r.kind == "function"
        assert r.required == [("kind", None), ("name", None)]
        assert r.optional == []

    def test_sys_register_node_kind_with_optional(self):
        r = parse('SYS REGISTER NODE KIND "function" REQUIRED kind, name OPTIONAL description')
        assert isinstance(r, SysRegisterNodeKind)
        assert r.kind == "function"
        assert r.required == [("kind", None), ("name", None)]
        assert r.optional == [("description", None)]

    def test_register_with_typed_fields(self):
        r = parse('SYS REGISTER NODE KIND "function" REQUIRED name:string, line:int OPTIONAL score:float')
        assert isinstance(r, SysRegisterNodeKind)
        assert r.required == [("name", "string"), ("line", "int")]
        assert r.optional == [("score", "float")]

    def test_register_mixed_typed_untyped(self):
        r = parse('SYS REGISTER NODE KIND "function" REQUIRED name:string, description')
        assert r.required == [("name", "string"), ("description", None)]

    def test_register_untyped_still_works(self):
        r = parse('SYS REGISTER NODE KIND "thing" REQUIRED name, value')
        assert r.required == [("name", None), ("value", None)]

    def test_sys_register_edge_kind(self):
        r = parse('SYS REGISTER EDGE KIND "calls" FROM "function", "method" TO "function"')
        assert isinstance(r, SysRegisterEdgeKind)
        assert r.kind == "calls"
        assert r.from_kinds == ["function", "method"]
        assert r.to_kinds == ["function"]

    def test_sys_unregister_node(self):
        r = parse('SYS UNREGISTER NODE KIND "function"')
        assert isinstance(r, SysUnregister)
        assert r.entity_type == "NODE"
        assert r.kind == "function"

    def test_sys_unregister_edge(self):
        r = parse('SYS UNREGISTER EDGE KIND "calls"')
        assert isinstance(r, SysUnregister)
        assert r.entity_type == "EDGE"
        assert r.kind == "calls"

    def test_sys_checkpoint(self):
        r = parse("SYS CHECKPOINT")
        assert isinstance(r, SysCheckpoint)

    def test_sys_rebuild(self):
        r = parse("SYS REBUILD INDICES")
        assert isinstance(r, SysRebuild)

    def test_sys_clear_log(self):
        r = parse("SYS CLEAR LOG")
        assert isinstance(r, SysClear)
        assert r.target == "LOG"

    def test_sys_clear_cache(self):
        r = parse("SYS CLEAR CACHE")
        assert isinstance(r, SysClear)
        assert r.target == "CACHE"

    def test_sys_wal_status(self):
        r = parse("SYS WAL STATUS")
        assert isinstance(r, SysWal)
        assert r.action == "STATUS"

    def test_sys_wal_replay(self):
        r = parse("SYS WAL REPLAY")
        assert isinstance(r, SysWal)
        assert r.action == "REPLAY"


class TestErrorHandling:
    def test_invalid_query_raises_query_error(self):
        with pytest.raises(QueryError):
            parse("INVALID GARBAGE QUERY")

    def test_empty_string_raises_query_error(self):
        with pytest.raises(QueryError):
            parse("")

    def test_partial_query_raises_query_error(self):
        with pytest.raises(QueryError):
            parse("NODE")


class TestPlanCache:
    def setup_method(self):
        clear_cache()

    def test_cache_returns_same_result(self):
        r1 = parse('NODE "user:1"')
        r2 = parse('NODE "user:1"')
        assert r1 is r2

    def test_cache_normalizes_whitespace(self):
        r1 = parse('NODE   "user:1"')
        r2 = parse('NODE "user:1"')
        assert r1 is r2

    def test_parse_uncached_returns_fresh(self):
        r1 = parse('NODE "user:1"')
        r2 = parse_uncached('NODE "user:1"')
        assert r1 is not r2
        assert isinstance(r2, NodeQuery)

    def test_cache_eviction(self):
        from supergraph.dsl.parser import PlanCache
        cache = PlanCache(maxsize=2)
        cache.get_or_parse('NODE "a"')
        cache.get_or_parse('NODE "b"')
        cache.get_or_parse('NODE "c"')
        assert len(cache) == 2

    def test_clear_cache(self):
        parse('NODE "user:1"')
        assert len(_plan_cache) > 0
        clear_cache()
        assert len(_plan_cache) == 0


import time as _time


class TestRelativeTimeExpressions:
    def test_parse_now(self):
        ast = parse('NODES WHERE __created_at__ > NOW()')
        assert ast.where is not None
        cond = ast.where.expr
        assert isinstance(cond.value, (int, float))
        assert cond.value > 0

    def test_parse_now_minus_7d(self):
        ast = parse('NODES WHERE __created_at__ > NOW() - 7d')
        cond = ast.where.expr
        assert isinstance(cond.value, (int, float))
        now_ms = int(_time.time() * 1000)
        seven_days_ms = 7 * 86400000
        assert abs(cond.value - (now_ms - seven_days_ms)) < 2000

    def test_parse_today(self):
        ast = parse('NODES WHERE __created_at__ > TODAY')
        cond = ast.where.expr
        assert isinstance(cond.value, (int, float))

    def test_parse_yesterday(self):
        ast = parse('NODES WHERE __updated_at__ > YESTERDAY')
        cond = ast.where.expr
        assert isinstance(cond.value, (int, float))

    def test_parse_now_minus_30m(self):
        ast = parse('NODES WHERE __updated_at__ > NOW() - 30m')
        cond = ast.where.expr
        now_ms = int(_time.time() * 1000)
        assert abs(cond.value - (now_ms - 30 * 60000)) < 2000
