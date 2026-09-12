"""POST /api/ingest routes by config.ingest.nl_backend.

The cloud NL->DSL engine landed in PR #198 and was surfaced through the SDK and
MCP in PR #201, but never through HTTP - /api/ingest called the local Bonsai
GGUF unconditionally. The [cloud-cpu] image ships no GGUF by design, so on a
deployed instance (where HTTP is the only surface) NL ingestion was unreachable
and returned "Bonsai not configured".
"""
from dataclasses import dataclass, field

import pytest


@dataclass
class _FakeResult:
    statements: list = field(default_factory=list)
    executed: int = 0
    rejected: list = field(default_factory=list)


class _FakeIngestConfig:
    def __init__(self, backend):
        self.nl_backend = backend


class _FakeConfig:
    def __init__(self, backend):
        self.ingest = _FakeIngestConfig(backend)


class _FakeStore:
    """Stands in for GraphStore - only the ingest surface is exercised."""

    def __init__(self, backend):
        self._config = _FakeConfig(backend)
        self.calls = []

    def ingest_nl(self, text, **kw):
        self.calls.append((text, kw))
        return _FakeResult(statements=['CREATE NODE "x" kind = "y"'], executed=1)


def _client(monkeypatch, backend):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import graphstore.server as server

    store = _FakeStore(backend)
    monkeypatch.setattr(server, "_get_store", lambda: store)
    return TestClient(server.app), server, store


def test_cloud_backend_routes_to_ingest_nl_not_bonsai(monkeypatch):
    client, server, store = _client(monkeypatch, "cloud")
    monkeypatch.setattr(
        server, "_get_bonsai",
        lambda: pytest.fail("cloud backend must not touch the local Bonsai GGUF"),
    )
    with client:
        r = client.post("/api/ingest", json={"text": "Kai met Priya in Bangalore"}).json()

    assert r["kind"] == "ingest"
    assert r["backend"] == "cloud"
    assert r["data"]["executed"] == 1
    text, kw = store.calls[0]
    assert text == "Kai met Priya in Bangalore"
    assert kw["session_id"] == "default" and kw["role"] == "user"
    assert kw["msg_id"].startswith("msg_")   # auto-generated when omitted


def test_cloud_backend_passes_through_dry_run_and_ids(monkeypatch):
    client, server, store = _client(monkeypatch, "cloud")
    with client:
        client.post("/api/ingest", json={
            "text": "t", "msg_id": "msg:fixed", "session_id": "s1",
            "role": "assistant", "dry_run": True,
        })
    _, kw = store.calls[0]
    assert kw == {"msg_id": "msg:fixed", "session_id": "s1",
                  "role": "assistant", "dry_run": True}


def test_local_backend_still_uses_bonsai(monkeypatch):
    client, server, store = _client(monkeypatch, None)

    class _FakeBonsai:
        def __init__(self):
            self.calls = []

        def ingest(self, text, **kw):
            self.calls.append(text)
            return _FakeResult(executed=7)

    fake = _FakeBonsai()
    monkeypatch.setattr(server, "_get_bonsai", lambda: fake)
    with client:
        r = client.post("/api/ingest", json={"text": "local path"}).json()

    assert r["kind"] == "ingest" and r["backend"] == "local"
    assert r["data"]["executed"] == 7
    assert fake.calls == ["local path"]
    assert store.calls == []


def test_missing_bonsai_surfaces_as_error_payload(monkeypatch):
    """The exact production failure: no GGUF on a [cloud-cpu] image."""
    from graphstore.core.errors import GraphStoreError

    client, server, _ = _client(monkeypatch, None)

    def _boom():
        raise GraphStoreError("Bonsai not configured: set GRAPHSTORE_BONSAI_GGUF to the GGUF path")

    monkeypatch.setattr(server, "_get_bonsai", _boom)
    with client:
        r = client.post("/api/ingest", json={"text": "x"}).json()

    assert r["kind"] == "error"
    assert r["backend"] == "local"
    assert "Bonsai not configured" in r["data"]
