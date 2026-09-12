
import pytest

import supergraph.server as server_module
from supergraph.server import app
from supergraph.core.store import CoreStore

from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _reset_store():
    server_module._store = None
    yield
    server_module._store = None


@pytest.fixture
def client():
    return TestClient(app)


class TestExecute:
    def test_execute_create_node(self, client):
        resp = client.post(
            "/api/execute",
            json={"query": 'CREATE NODE "a" kind = "person" name = "Alice"'},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["kind"] == "node"
        assert body["data"]["id"] == "a"

    def test_execute_query_node(self, client):
        client.post(
            "/api/execute",
            json={"query": 'CREATE NODE "a" kind = "person" name = "Alice"'},
        )
        resp = client.post(
            "/api/execute",
            json={"query": 'NODE "a"'},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["kind"] == "node"
        assert body["data"]["name"] == "Alice"
        assert body["count"] == 1

    def test_execute_invalid_query(self, client):
        resp = client.post(
            "/api/execute",
            json={"query": "THIS IS NOT VALID DSL"},
        )
        assert resp.status_code == 200
        assert resp.json()["kind"] == "error"
        body = resp.json()
        assert body["kind"] == "error"


class TestExecuteBatch:
    def test_execute_batch(self, client):
        resp = client.post(
            "/api/execute-batch",
            json={
                "queries": [
                    'CREATE NODE "x" kind = "item" val = 1',
                    'CREATE NODE "y" kind = "item" val = 2',
                ]
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2
        assert body[0]["kind"] == "node"
        assert body[1]["kind"] == "node"


class TestGetGraph:
    def test_get_graph(self, client):
        client.post(
            "/api/execute",
            json={"query": 'CREATE NODE "a" kind = "person" name = "Alice"'},
        )
        client.post(
            "/api/execute",
            json={"query": 'CREATE NODE "b" kind = "person" name = "Bob"'},
        )
        client.post(
            "/api/execute",
            json={"query": 'CREATE EDGE "a" -> "b" kind = "knows"'},
        )
        resp = client.get("/api/graph")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["nodes"]) == 2
        assert len(body["edges"]) == 1
        edge = body["edges"][0]
        assert edge["source"] == "a"
        assert edge["target"] == "b"
        assert edge["kind"] == "knows"


class TestReset:
    def test_reset(self, client):
        client.post(
            "/api/execute",
            json={"query": 'CREATE NODE "a" kind = "person"'},
        )
        resp = client.post("/api/reset")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

        graph = client.get("/api/graph").json()
        assert len(graph["nodes"]) == 0
        assert len(graph["edges"]) == 0


class TestConfig:
    def test_config(self, client):
        resp = client.post("/api/config", json={"ceiling_mb": 512})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}


class TestGetAllEdges:
    def test_get_all_edges(self):
        store = CoreStore()
        store.put_node("a", "person", {"name": "Alice"})
        store.put_node("b", "person", {"name": "Bob"})
        store.put_edge("a", "b", "knows")

        edges = store.get_all_edges()
        assert len(edges) == 1
        assert edges[0]["source"] == "a"
        assert edges[0]["target"] == "b"
        assert edges[0]["kind"] == "knows"

    def test_get_all_edges_empty(self):
        store = CoreStore()
        assert store.get_all_edges() == []

    def test_get_all_edges_multiple_types(self):
        store = CoreStore()
        store.put_node("a", "person", {})
        store.put_node("b", "person", {})
        store.put_node("c", "person", {})
        store.put_edge("a", "b", "knows")
        store.put_edge("a", "c", "follows")

        edges = store.get_all_edges()
        assert len(edges) == 2
        kinds = {e["kind"] for e in edges}
        assert kinds == {"knows", "follows"}


