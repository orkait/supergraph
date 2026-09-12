
import pytest
from supergraph.core.store import CoreStore
from supergraph.core.schema import SchemaRegistry
from supergraph.core.runtime import RuntimeState
from supergraph.dsl.parser import parse
from supergraph.dsl.executor import Executor
from supergraph.core.errors import NodeExists, NodeNotFound, BatchRollback


@pytest.fixture
def executor():
    store = CoreStore()
    return Executor(RuntimeState(store=store, schema=SchemaRegistry()))


def execute(executor, query):
    ast = parse(query)
    return executor.execute(ast)


class TestCreateNode:
    def test_create_node(self, executor):
        r = execute(executor, 'CREATE NODE "x" kind = "function" name = "foo"')
        assert r.kind == "node"
        assert r.data["id"] == "x"
        assert r.data["kind"] == "function"
        node = executor.store.get_node("x")
        assert node is not None
        assert node["kind"] == "function"
        assert node["name"] == "foo"

    def test_create_node_duplicate(self, executor):
        execute(executor, 'CREATE NODE "x" kind = "function" name = "foo"')
        with pytest.raises(NodeExists):
            execute(executor, 'CREATE NODE "x" kind = "function" name = "bar"')


class TestUpdateNode:
    def test_update_node(self, executor):
        execute(executor, 'CREATE NODE "x" kind = "function" name = "foo"')
        r = execute(executor, 'UPDATE NODE "x" SET name = "bar"')
        assert r.kind == "node"
        assert r.data["id"] == "x"
        assert r.data["name"] == "bar"

    def test_update_missing_node(self, executor):
        with pytest.raises(NodeNotFound):
            execute(executor, 'UPDATE NODE "missing" SET name = "bar"')


class TestUpsertNode:
    def test_upsert_creates_new(self, executor):
        r = execute(executor, 'UPSERT NODE "y" kind = "class" name = "Widget"')
        assert r.kind == "node"
        assert r.data["id"] == "y"
        node = executor.store.get_node("y")
        assert node is not None
        assert node["kind"] == "class"
        assert node["name"] == "Widget"

    def test_upsert_updates_existing(self, executor):
        execute(executor, 'CREATE NODE "y" kind = "class" name = "Widget"')
        execute(executor, 'UPSERT NODE "y" kind = "class" name = "SuperWidget"')
        node = executor.store.get_node("y")
        assert node["name"] == "SuperWidget"


class TestDeleteNode:
    def test_delete_node(self, executor):
        execute(executor, 'CREATE NODE "x" kind = "function" name = "foo"')
        r = execute(executor, 'DELETE NODE "x"')
        assert r.kind == "ok"
        assert executor.store.get_node("x") is None

    def test_delete_missing_node(self, executor):
        with pytest.raises(NodeNotFound):
            execute(executor, 'DELETE NODE "missing"')


class TestDeleteNodes:
    def test_delete_nodes_by_kind(self, executor):
        execute(executor, 'CREATE NODE "a" kind = "function" name = "a"')
        execute(executor, 'CREATE NODE "b" kind = "function" name = "b"')
        execute(executor, 'CREATE NODE "c" kind = "class" name = "c"')

        r = execute(executor, 'DELETE NODES WHERE kind = "function"')
        assert r.kind == "nodes"
        assert r.count == 2
        assert executor.store.get_node("a") is None
        assert executor.store.get_node("b") is None
        assert executor.store.get_node("c") is not None


class TestCreateEdge:
    def test_create_edge(self, executor):
        execute(executor, 'CREATE NODE "a" kind = "function" name = "a"')
        execute(executor, 'CREATE NODE "b" kind = "function" name = "b"')
        r = execute(executor, 'CREATE EDGE "a" -> "b" kind = "calls"')
        assert r.kind == "edges"
        assert r.data[0]["source"] == "a"
        assert r.data[0]["target"] == "b"
        edges = executor.store.get_edges_from("a", kind="calls")
        assert len(edges) == 1
        assert edges[0]["target"] == "b"


class TestDeleteEdge:
    def test_delete_edge(self, executor):
        execute(executor, 'CREATE NODE "a" kind = "function" name = "a"')
        execute(executor, 'CREATE NODE "b" kind = "function" name = "b"')
        execute(executor, 'CREATE EDGE "a" -> "b" kind = "calls"')

        r = execute(executor, 'DELETE EDGE "a" -> "b" WHERE kind = "calls"')
        assert r.kind == "ok"
        edges = executor.store.get_edges_from("a", kind="calls")
        assert len(edges) == 0


class TestDeleteEdges:
    def test_delete_edges_from(self, executor):
        execute(executor, 'CREATE NODE "a" kind = "function" name = "a"')
        execute(executor, 'CREATE NODE "b" kind = "function" name = "b"')
        execute(executor, 'CREATE NODE "c" kind = "function" name = "c"')
        execute(executor, 'CREATE EDGE "a" -> "b" kind = "calls"')
        execute(executor, 'CREATE EDGE "a" -> "c" kind = "calls"')

        r = execute(executor, 'DELETE EDGES FROM "a" WHERE kind = "calls"')
        assert r.kind == "edges"
        assert r.count == 2
        edges = executor.store.get_edges_from("a", kind="calls")
        assert len(edges) == 0


class TestIncrement:
    def test_increment(self, executor):
        execute(executor, 'CREATE NODE "x" kind = "function" name = "foo" hits = 0')
        r = execute(executor, 'INCREMENT NODE "x" hits BY 1')
        assert r.kind == "ok"
        node = executor.store.get_node("x")
        assert node["hits"] == 1

    def test_increment_multiple(self, executor):
        execute(executor, 'CREATE NODE "x" kind = "function" name = "foo" hits = 0')
        execute(executor, 'INCREMENT NODE "x" hits BY 1')
        execute(executor, 'INCREMENT NODE "x" hits BY 5')
        node = executor.store.get_node("x")
        assert node["hits"] == 6


class TestBatch:
    def test_batch_success(self, executor):
        batch = 'BEGIN\nCREATE NODE "a" kind = "test"\nCREATE NODE "b" kind = "test"\nCREATE EDGE "a" -> "b" kind = "calls"\nCOMMIT'
        r = execute(executor, batch)
        assert r.kind == "ok"
        assert executor.store.get_node("a") is not None
        assert executor.store.get_node("b") is not None
        edges = executor.store.get_edges_from("a", kind="calls")
        assert len(edges) == 1

    def test_batch_rollback_on_failure(self, executor):
        execute(executor, 'CREATE NODE "existing" kind = "test"')
        initial_count = executor.store.node_count

        batch = 'BEGIN\nCREATE NODE "new_node" kind = "test"\nCREATE NODE "existing" kind = "test"\nCOMMIT'
        with pytest.raises(BatchRollback):
            execute(executor, batch)

        assert executor.store.get_node("new_node") is None
        assert executor.store.node_count == initial_count


class TestNullHandling:
    def test_null_condition(self, executor):
        execute(executor, 'CREATE NODE "a" kind = "function" name = "foo"')
        execute(executor, 'CREATE NODE "b" kind = "function"')

        r = execute(executor, 'NODES WHERE name = NULL')
        ids = {n["id"] for n in r.data}
        assert "b" in ids
        assert "a" not in ids

    def test_not_null_condition(self, executor):
        execute(executor, 'CREATE NODE "a" kind = "function" name = "foo"')
        execute(executor, 'CREATE NODE "b" kind = "function"')

        r = execute(executor, 'NODES WHERE name != NULL')
        ids = {n["id"] for n in r.data}
        assert "a" in ids
        assert "b" not in ids
