from supergraph.config import SuperGraphConfig, IngestConfig, apply_env_overrides


def test_ingest_defaults():
    cfg = SuperGraphConfig()
    assert cfg.ingest.nl_backend is None
    assert cfg.ingest.free_first is True
    assert cfg.ingest.nl_max_tokens == 1000


def test_ingest_env_override(monkeypatch):
    monkeypatch.setenv("SUPERGRAPH_INGEST_NL_BACKEND", "cloud")
    monkeypatch.setenv("SUPERGRAPH_INGEST_NL_MAX_TOKENS", "2000")
    cfg = apply_env_overrides(SuperGraphConfig())
    assert cfg.ingest.nl_backend == "cloud"
    assert cfg.ingest.nl_max_tokens == 2000


def test_ingest_config_is_frozen():
    import msgspec
    c = IngestConfig()
    assert isinstance(c, msgspec.Struct)


def test_supergraph_nl_kwarg_shortcuts():
    from supergraph import SuperGraph
    gs = SuperGraph(
        embedder="none",
        nl_backend="cloud",
        nl_models=["groq/llama-3.1-8b-instant"],
        nl_max_tokens=500,
    )
    assert gs._config.ingest.nl_backend == "cloud"
    assert gs._config.ingest.nl_models == ["groq/llama-3.1-8b-instant"]
    assert gs._config.ingest.nl_max_tokens == 500
