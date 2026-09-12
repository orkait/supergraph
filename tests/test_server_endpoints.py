"""Tests for /api/logs and /api/script server endpoints."""
import os
import tempfile
import pytest


@pytest.fixture
def client():
    """Create a test client without auth (in-memory store)."""
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
    """Create a test client backed by a temporary on-disk DB.

    Logs and script metadata only work when SuperGraph has a real SQLite
    connection (path != None), so tests that need them use this fixture.
    """
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
        """GET /api/logs returns empty list when store has no DB connection."""
        resp = client.get("/api/logs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_logs_empty_on_fresh_store(self, persistent_client):
        """GET /api/logs returns empty list on a fresh persistent store."""
        resp = persistent_client.get("/api/logs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) == 0

    def test_get_logs_after_queries(self, persistent_client):
        """Logs appear after executing queries against a persistent store."""
        persistent_client.post("/api/execute", json={"query": 'CREATE NODE "log_test" kind = "test"'})
        persistent_client.post("/api/execute", json={"query": 'NODE "log_test"'})

        resp = persistent_client.get("/api/logs?limit=10")
        assert resp.status_code == 200
        logs = resp.json()
        assert len(logs) >= 2

    def test_get_logs_have_expected_fields(self, persistent_client):
        """Each log entry has the expected keys."""
        persistent_client.post("/api/execute", json={"query": 'SYS STATS'})

        resp = persistent_client.get("/api/logs?limit=5")
        logs = resp.json()
        assert len(logs) >= 1
        entry = logs[0]
        for key in ("id", "timestamp", "query", "elapsed_us", "result_count", "error", "tag", "trace_id", "source", "phase"):
            assert key in entry, f"missing key: {key}"

    def test_get_logs_filter_by_tag(self, persistent_client):
        """Filter logs by tag returns only entries with that tag."""
        persistent_client.post("/api/execute", json={"query": 'CREATE NODE "ft" kind = "test"'})
        persistent_client.post("/api/execute", json={"query": 'NODE "ft"'})

        resp = persistent_client.get("/api/logs?tag=read&limit=10")
        assert resp.status_code == 200
        logs = resp.json()
        for log in logs:
            assert log["tag"] == "read"

    def test_get_logs_filter_by_source(self, persistent_client):
        """Filter logs by source returns only matching entries."""
        persistent_client.post("/api/execute", json={"query": 'SYS STATS'})
        resp = persistent_client.get("/api/logs?source=user&limit=10")
        assert resp.status_code == 200
        logs = resp.json()
        for log in logs:
            assert "user" in log["source"]

    def test_get_logs_limit(self, persistent_client):
        """The limit parameter caps the number of returned entries."""
        for i in range(5):
            persistent_client.post("/api/execute", json={"query": f'CREATE NODE "lim{i}" kind = "test"'})

        resp = persistent_client.get("/api/logs?limit=3")
        assert resp.status_code == 200
        logs = resp.json()
        assert len(logs) <= 3

    def test_get_logs_default_limit(self, persistent_client):
        """Without a limit param the endpoint uses its default (50) without error."""
        resp = persistent_client.get("/api/logs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestApiScript:

    def test_get_script_empty_inmemory(self, client):
        """GET /api/script returns null script when store has no DB connection."""
        resp = client.get("/api/script")
        assert resp.status_code == 200
        assert resp.json()["script"] is None

    def test_put_and_get_script(self, persistent_client):
        """Save a script then retrieve it verbatim."""
        script = 'CREATE NODE "x" kind = "test"\nNODE "x"'
        resp = persistent_client.put("/api/script", json={"query": script})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        resp = persistent_client.get("/api/script")
        assert resp.status_code == 200
        assert resp.json()["script"] == script

    def test_get_script_empty_on_fresh_store(self, persistent_client):
        """GET /api/script returns null when no script has been saved yet."""
        resp = persistent_client.get("/api/script")
        assert resp.status_code == 200
        assert resp.json()["script"] is None

    def test_put_script_overwrite(self, persistent_client):
        """Putting a new script overwrites the previously stored one."""
        persistent_client.put("/api/script", json={"query": "old script"})
        persistent_client.put("/api/script", json={"query": "new script"})

        resp = persistent_client.get("/api/script")
        assert resp.status_code == 200
        assert resp.json()["script"] == "new script"

    def test_put_script_multiline(self, persistent_client):
        """Multi-line scripts are stored and retrieved without modification."""
        script = "CREATE NODE \"a\" kind=\"x\"\nCREATE NODE \"b\" kind=\"x\"\nEDGE \"a\" -> \"b\" label=\"link\""
        persistent_client.put("/api/script", json={"query": script})
        resp = persistent_client.get("/api/script")
        assert resp.json()["script"] == script

    def test_put_script_response_shape(self, persistent_client):
        """PUT /api/script always returns {ok: true}."""
        resp = persistent_client.put("/api/script", json={"query": "any content"})
        assert resp.status_code == 200
        body = resp.json()
        assert "ok" in body
        assert body["ok"] is True
