import os
import tempfile
import pytest


@pytest.fixture
def client():
    os.environ.pop("SUPERGRAPH_AUTH_TOKEN", None)
    os.environ.pop("SUPERGRAPH_DB_PATH", None)

    import importlib
    import supergraph.server as srv
    srv._store = None
    importlib.reload(srv)

    from fastapi.testclient import TestClient
    c = TestClient(srv.app)
    yield c
    srv._store = None


@pytest.fixture
def persistent_client():
    os.environ.pop("SUPERGRAPH_AUTH_TOKEN", None)

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["SUPERGRAPH_DB_PATH"] = tmpdir

        import importlib
        import supergraph.server as srv
        srv._store = None
        importlib.reload(srv)

        from fastapi.testclient import TestClient
        c = TestClient(srv.app)
        yield c

        srv._store = None
        os.environ.pop("SUPERGRAPH_DB_PATH", None)


class TestApiLogs:

    def test_get_logs_empty_inmemory(self, client):
        resp = client.get("/api/logs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_logs_empty_on_fresh_store(self, persistent_client):
        resp = persistent_client.get("/api/logs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) == 0

    def test_get_logs_after_queries(self, persistent_client):
        persistent_client.post("/api/execute", json={"query": 'CREATE NODE "log_test" kind = "test"'})
        persistent_client.post("/api/execute", json={"query": 'NODE "log_test"'})

        resp = persistent_client.get("/api/logs?limit=10")
        assert resp.status_code == 200
        logs = resp.json()
        assert len(logs) >= 2

    def test_get_logs_have_expected_fields(self, persistent_client):
        persistent_client.post("/api/execute", json={"query": 'SYS STATS'})

        resp = persistent_client.get("/api/logs?limit=5")
        logs = resp.json()
        assert len(logs) >= 1
        entry = logs[0]
        for key in ("id", "timestamp", "query", "elapsed_us", "result_count", "error", "tag", "trace_id", "source", "phase"):
            assert key in entry, f"missing key: {key}"

    def test_get_logs_filter_by_tag(self, persistent_client):
        persistent_client.post("/api/execute", json={"query": 'CREATE NODE "ft" kind = "test"'})
        persistent_client.post("/api/execute", json={"query": 'NODE "ft"'})

        resp = persistent_client.get("/api/logs?tag=read&limit=10")
        assert resp.status_code == 200
        logs = resp.json()
        for log in logs:
            assert log["tag"] == "read"

    def test_get_logs_filter_by_source(self, persistent_client):
        persistent_client.post("/api/execute", json={"query": 'SYS STATS'})
        resp = persistent_client.get("/api/logs?source=user&limit=10")
        assert resp.status_code == 200
        logs = resp.json()
        for log in logs:
            assert "user" in log["source"]

    def test_get_logs_limit(self, persistent_client):
        for i in range(5):
            persistent_client.post("/api/execute", json={"query": f'CREATE NODE "lim{i}" kind = "test"'})

        resp = persistent_client.get("/api/logs?limit=3")
        assert resp.status_code == 200
        logs = resp.json()
        assert len(logs) <= 3

    def test_get_logs_default_limit(self, persistent_client):
        resp = persistent_client.get("/api/logs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestApiScript:

    def test_get_script_empty_inmemory(self, client):
        resp = client.get("/api/script")
        assert resp.status_code == 200
        assert resp.json()["script"] is None

    def test_put_and_get_script(self, persistent_client):
        script = 'CREATE NODE "x" kind = "test"\nNODE "x"'
        resp = persistent_client.put("/api/script", json={"query": script})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        resp = persistent_client.get("/api/script")
        assert resp.status_code == 200
        assert resp.json()["script"] == script

    def test_get_script_empty_on_fresh_store(self, persistent_client):
        resp = persistent_client.get("/api/script")
        assert resp.status_code == 200
        assert resp.json()["script"] is None

    def test_put_script_overwrite(self, persistent_client):
        persistent_client.put("/api/script", json={"query": "old script"})
        persistent_client.put("/api/script", json={"query": "new script"})

        resp = persistent_client.get("/api/script")
        assert resp.status_code == 200
        assert resp.json()["script"] == "new script"

    def test_put_script_multiline(self, persistent_client):
        script = "CREATE NODE \"a\" kind=\"x\"\nCREATE NODE \"b\" kind=\"x\"\nEDGE \"a\" -> \"b\" label=\"link\""
        persistent_client.put("/api/script", json={"query": script})
        resp = persistent_client.get("/api/script")
        assert resp.json()["script"] == script

    def test_put_script_response_shape(self, persistent_client):
        resp = persistent_client.put("/api/script", json={"query": "any content"})
        assert resp.status_code == 200
        body = resp.json()
        assert "ok" in body
        assert body["ok"] is True
