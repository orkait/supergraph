
import pytest
from supergraph.core.store import CoreStore
from supergraph.core.schema import SchemaRegistry
from supergraph.core.runtime import RuntimeState
from supergraph.dsl.parser import parse
from supergraph.dsl.executor import Executor


@pytest.fixture
def graph():
    store = CoreStore()
    store.put_node("fn_main", "function", {"name": "main", "file": "src/main.py", "line": 1})
    store.put_node("fn_helper", "function", {"name": "helper", "file": "src/utils.py", "line": 10})
    store.put_node("fn_parse", "function", {"name": "parse", "file": "src/parser.py", "line": 5})
    store.put_node("cls_app", "class", {"name": "App", "file": "src/main.py", "line": 20})
    store.put_node("cls_base", "class", {"name": "Base", "file": "src/base.py", "line": 1})
    store.put_node("fn_run", "function", {"name": "run", "file": "src/main.py", "line": 50, "hits": 0})

    store.put_edge("fn_main", "fn_helper", "calls")
    store.put_edge("fn_main", "fn_parse", "calls")
    store.put_edge("fn_helper", "fn_parse", "calls")
    store.put_edge("fn_main", "fn_run", "calls")
    store.put_edge("cls_app", "cls_base", "extends")
    store.put_edge("fn_main", "cls_app", "uses")

    return Executor(RuntimeState(store=store, schema=SchemaRegistry()))


def execute(executor, query):
    ast = parse(query)
    return executor.execute(ast)


class TestNodeQuery:
    def test_node_returns_data(self, graph):
        r = execute(graph, 'NODE "fn_main"')
        assert r.kind == "node"
        assert r.count == 1
        assert r.data["id"] == "fn_main"
        assert r.data["kind"] == "function"
        assert r.data["name"] == "main"
        assert r.data["file"] == "src/main.py"
        assert r.elapsed_us >= 0

    def test_node_nonexistent(self, graph):
        r = execute(graph, 'NODE "nonexistent"')
        assert r.kind == "node"
        assert r.count == 0
        assert r.data is None


class TestNodesQuery:
    def test_all_nodes(self, graph):
        r = execute(graph, "NODES")
        assert r.kind == "nodes"
        assert r.count == 6
        ids = {n["id"] for n in r.data}
        assert ids == {"fn_main", "fn_helper", "fn_parse", "cls_app", "cls_base", "fn_run"}

    def test_where_kind(self, graph):
        r = execute(graph, 'NODES WHERE kind = "function"')
        assert r.count == 4
        assert all(n["kind"] == "function" for n in r.data)

    def test_where_kind_with_limit(self, graph):
        r = execute(graph, 'NODES WHERE kind = "function" LIMIT 2')
        assert r.count == 2
        assert all(n["kind"] == "function" for n in r.data)

    def test_where_field(self, graph):
        r = execute(graph, 'NODES WHERE name = "main"')
        assert r.count == 1
        assert r.data[0]["id"] == "fn_main"

    def test_where_or(self, graph):
        r = execute(graph, 'NODES WHERE (kind = "function" OR kind = "class")')
        assert r.count == 6

    def test_where_and(self, graph):
        r = execute(graph, 'NODES WHERE kind = "function" AND file = "src/main.py"')
        ids = {n["id"] for n in r.data}
        assert ids == {"fn_main", "fn_run"}

    def test_where_not(self, graph):
        r = execute(graph, 'NODES WHERE NOT kind = "class"')
        assert r.count == 4
        assert all(n["kind"] != "class" for n in r.data)


class TestEdgesQuery:
    def test_edges_from_with_kind(self, graph):
        r = execute(graph, 'EDGES FROM "fn_main" WHERE kind = "calls"')
        assert r.kind == "edges"
        targets = {e["target"] for e in r.data}
        assert targets == {"fn_helper", "fn_parse", "fn_run"}
        assert all(e["source"] == "fn_main" for e in r.data)

    def test_edges_to_with_kind(self, graph):
        r = execute(graph, 'EDGES TO "fn_parse" WHERE kind = "calls"')
        sources = {e["source"] for e in r.data}
        assert "fn_main" in sources
        assert "fn_helper" in sources
        assert all(e["target"] == "fn_parse" for e in r.data)

    def test_edges_from_all_types(self, graph):
        r = execute(graph, 'EDGES FROM "fn_main"')
        assert r.count == 4
        kinds = {e["kind"] for e in r.data}
        assert "calls" in kinds
        assert "uses" in kinds


class TestTraverseQuery:
    def test_traverse_depth_1(self, graph):
        r = execute(graph, 'TRAVERSE FROM "fn_main" DEPTH 1 WHERE kind = "calls"')
        assert r.kind == "nodes"
        ids = {n["id"] for n in r.data}
        assert "fn_main" in ids
        assert "fn_helper" in ids
        assert "fn_parse" in ids
        assert "fn_run" in ids

    def test_traverse_depth_2(self, graph):
        r = execute(graph, 'TRAVERSE FROM "fn_main" DEPTH 2 WHERE kind = "calls"')
        ids = {n["id"] for n in r.data}
        assert "fn_main" in ids
        assert "fn_helper" in ids
        assert "fn_parse" in ids
        assert "fn_run" in ids

    def test_traverse_includes_depth_metadata(self, graph):
        r = execute(graph, 'TRAVERSE FROM "fn_main" DEPTH 1 WHERE kind = "calls"')
        for n in r.data:
            assert "_depth" in n
            if n["id"] == "fn_main":
                assert n["_depth"] == 0
            else:
                assert n["_depth"] == 1


class TestSubgraphQuery:
    def test_subgraph_depth_1(self, graph):
        r = execute(graph, 'SUBGRAPH FROM "fn_main" DEPTH 1')
        assert r.kind == "subgraph"
        assert "nodes" in r.data
        assert "edges" in r.data
        node_ids = {n["id"] for n in r.data["nodes"]}
        assert "fn_main" in node_ids
        assert len(r.data["edges"]) > 0


class TestPathQuery:
    def test_path_exists(self, graph):
        r = execute(graph, 'PATH FROM "fn_main" TO "fn_parse" MAX_DEPTH 3')
        assert r.kind == "path"
        assert r.data is not None
        assert r.data[0] == "fn_main"
        assert r.data[-1] == "fn_parse"

    def test_path_with_kind_filter(self, graph):
        r = execute(graph, 'PATH FROM "fn_main" TO "fn_parse" MAX_DEPTH 3 WHERE kind = "calls"')
        assert r.kind == "path"
        assert r.data is not None
        assert "fn_main" in r.data
        assert "fn_parse" in r.data

    def test_path_no_connection(self, graph):
        r = execute(graph, 'PATH FROM "cls_base" TO "fn_main" MAX_DEPTH 5')
        assert r.data is None or r.count == 0


class TestShortestPathQuery:
    def test_shortest_path_direct(self, graph):
        r = execute(graph, 'SHORTEST PATH FROM "fn_main" TO "fn_parse" WHERE kind = "calls"')
        assert r.kind == "path"
        assert r.data is not None
        assert len(r.data) == 2
        assert r.data[0] == "fn_main"
        assert r.data[1] == "fn_parse"

    def test_shortest_path_no_route(self, graph):
        r = execute(graph, 'SHORTEST PATH FROM "fn_parse" TO "fn_main" WHERE kind = "calls"')
        assert r.data is None


class TestDistanceQuery:
    def test_distance_direct(self, graph):
        r = execute(graph, 'DISTANCE FROM "fn_main" TO "fn_parse" MAX_DEPTH 5')
        assert r.kind == "distance"
        assert r.data == 1

    def test_distance_no_path(self, graph):
        r = execute(graph, 'DISTANCE FROM "fn_parse" TO "fn_main" MAX_DEPTH 5')
        assert r.data == -1


class TestAncestorsQuery:
    def test_ancestors(self, graph):
        r = execute(graph, 'ANCESTORS OF "fn_parse" DEPTH 2 WHERE kind = "calls"')
        assert r.kind == "subgraph"
        ids = {n["id"] for n in r.data["nodes"] if not n.get("_query_anchor")}
        assert "fn_main" in ids
        assert "fn_helper" in ids
        assert "fn_parse" not in ids
        assert len(r.data["edges"]) > 0


class TestDescendantsQuery:
    def test_descendants(self, graph):
        r = execute(graph, 'DESCENDANTS OF "fn_main" DEPTH 2 WHERE kind = "calls"')
        assert r.kind == "subgraph"
        ids = {n["id"] for n in r.data["nodes"] if not n.get("_query_anchor")}
        assert "fn_helper" in ids
        assert "fn_parse" in ids
        assert "fn_run" in ids
        assert "fn_main" not in ids
        assert len(r.data["edges"]) > 0


class TestCommonNeighborsQuery:
    def test_common_neighbors(self, graph):
        r = execute(graph, 'COMMON NEIGHBORS OF "fn_main" AND "fn_helper" WHERE kind = "calls"')
        assert r.kind == "nodes"
        ids = {n["id"] for n in r.data}
        assert "fn_parse" in ids


class TestDegreeConditions:
    def test_indegree_filter(self, graph):
        r = execute(graph, 'NODES WHERE INDEGREE > 1')
        ids = {n["id"] for n in r.data}
        assert "fn_parse" in ids

    def test_outdegree_typed_filter(self, graph):
        r = execute(graph, 'NODES WHERE OUTDEGREE calls > 2')
        ids = {n["id"] for n in r.data}
        assert "fn_main" in ids
        assert "fn_helper" not in ids


class TestMatchQuery:
    def test_match_bound_start(self, graph):
        r = execute(graph, 'MATCH ("fn_main") -[kind = "calls"]-> (b)')
        assert r.kind == "match"
        assert r.count > 0
        for binding in r.data["bindings"]:
            assert "b" in binding
        bound_ids = {b["b"] for b in r.data["bindings"]}
        assert "fn_helper" in bound_ids
        assert "fn_parse" in bound_ids
        assert "fn_run" in bound_ids
        assert len(r.data["edges"]) > 0

    def test_match_untyped_reports_stored_edge_kind(self, graph):
        r = execute(graph, 'MATCH ("fn_main") -[]-> (b)')
        by_target = {e["target"]: e["kind"] for e in r.data["edges"]}
        assert by_target["fn_helper"] == "calls"
        assert by_target["cls_app"] == "uses"

    def test_match_typed_reports_stored_edge_kind(self, graph):
        r = execute(graph, 'MATCH ("fn_main") -[kind = "uses"]-> (b)')
        assert [e["kind"] for e in r.data["edges"]] == ["uses"]
