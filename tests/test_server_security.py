import os
import pytest


@pytest.fixture
def auth_client():
    os.environ["SUPERGRAPH_AUTH_TOKEN"] = "test-secret-token"
    os.environ.pop("SUPERGRAPH_DB_PATH", None)

    import importlib
    import supergraph.server as srv
    srv._store = None
    importlib.reload(srv)

    from fastapi.testclient import TestClient
    client = TestClient(srv.app)
    yield client

    os.environ.pop("SUPERGRAPH_AUTH_TOKEN", None)
    srv._store = None


@pytest.fixture
def open_client():
    os.environ.pop("SUPERGRAPH_AUTH_TOKEN", None)
    os.environ.pop("SUPERGRAPH_DB_PATH", None)

    import importlib
    import supergraph.server as srv
    srv._store = None
    importlib.reload(srv)

    from fastapi.testclient import TestClient
    client = TestClient(srv.app)
    yield client
    srv._store = None


def test_no_auth_allows_access(open_client):
    resp = open_client.post("/api/execute", json={"query": "SYS STATS"})
    assert resp.status_code == 200


def test_auth_rejects_missing_token(auth_client):
    resp = auth_client.post("/api/execute", json={"query": "SYS STATS"})
    assert resp.status_code == 401


def test_auth_rejects_wrong_token(auth_client):
    resp = auth_client.post(
        "/api/execute",
        json={"query": "SYS STATS"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_auth_accepts_correct_token(auth_client):
    resp = auth_client.post(
        "/api/execute",
        json={"query": "SYS STATS"},
        headers={"Authorization": "Bearer test-secret-token"},
    )
    assert resp.status_code == 200


def test_rate_limit_returns_429(open_client):
    import supergraph.server as srv
    original = srv._RATE_LIMIT_RPM
    srv._RATE_LIMIT_RPM = 3
    srv._rate_buckets.clear()
    try:
        for i in range(3):
            resp = open_client.post("/api/execute", json={"query": "SYS STATS"})
            assert resp.status_code == 200
        resp = open_client.post("/api/execute", json={"query": "SYS STATS"})
        assert resp.status_code == 429
        assert "Rate limit" in resp.json()["error"]
    finally:
        srv._RATE_LIMIT_RPM = original
        srv._rate_buckets.clear()


def test_rejects_empty_query(open_client):
    resp = open_client.post("/api/execute", json={"query": ""})
    assert resp.status_code == 200
    data = resp.json()
    assert data["kind"] == "error"
    assert "Empty" in data["data"]


def test_rejects_oversized_query(open_client):
    resp = open_client.post("/api/execute", json={"query": "A" * 20000})
    assert resp.status_code == 200
    data = resp.json()
    assert data["kind"] == "error"
    assert "maximum length" in data["data"]


def test_rejects_null_bytes(open_client):
    resp = open_client.post("/api/execute", json={"query": "NODE \x00"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["kind"] == "error"
    assert "null" in data["data"]


def test_batch_limit(open_client):
    resp = open_client.post("/api/execute-batch", json={"queries": ["SYS STATS"] * 1001})
    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["kind"] == "error"
    assert "1000" in data[0]["data"]
