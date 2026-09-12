"""MCP sg_ingest tool: structured NL->graph via gs.ingest_nl."""
import types


def test_mcp_sg_ingest_returns_structured(monkeypatch):
    import supergraph.mcp.server as srv

    res = types.SimpleNamespace(executed=3, rejected=[], statements=["a", "b", "c"])
    fake = types.SimpleNamespace(ingest_nl=lambda text: res)
    monkeypatch.setattr(srv, "_REMOTE_URL", None)
    monkeypatch.setattr(srv, "_get_store", lambda: fake)

    out = srv._ingest_nl("Alice works at OpenAI")
    assert out["executed"] == 3
    assert out["rejected"] == 0
    assert out["statements"] == ["a", "b", "c"]


def test_mcp_sg_ingest_disabled_returns_error(monkeypatch):
    import supergraph.mcp.server as srv

    def _raise(text):
        raise ValueError("NL ingestion is disabled: set config.ingest.nl_backend='cloud'")

    monkeypatch.setattr(srv, "_REMOTE_URL", None)
    monkeypatch.setattr(srv, "_get_store", lambda: types.SimpleNamespace(ingest_nl=_raise))
    out = srv._ingest_nl("x")
    assert "error" in out and "nl_backend" in out["error"]


def test_mcp_sg_ingest_remote_unsupported(monkeypatch):
    import supergraph.mcp.server as srv
    monkeypatch.setattr(srv, "_REMOTE_URL", "http://host:7200")
    out = srv._ingest_nl("x")
    assert "error" in out and "remote" in out["error"].lower()
