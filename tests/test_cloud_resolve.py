from supergraph.ingest.llm.resolve import (
    resolve_model, build_provider_chain, DEFAULT_FREE_FIRST_CHAIN,
)


def test_resolve_groq_prefix():
    r = resolve_model("groq/llama-3.3-70b-versatile")
    assert r["litellm_model"] == "groq/llama-3.3-70b-versatile"
    assert r["api_base"] is None
    assert "api_key" in r


def test_resolve_cloudflare_carries_account_id(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-key")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-1")
    r = resolve_model("cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast")
    assert r["litellm_model"].startswith("cloudflare/")
    assert r["api_key"] == "cf-key"
    assert r["account_id"] == "acct-1"


def test_resolve_alias_maps_to_free():
    r = resolve_model("gpt-4o-mini")
    assert r["litellm_model"] == "groq/llama-3.1-8b-instant"


def test_resolve_bare_id_defaults_openrouter():
    r = resolve_model("some/unknown-model")
    assert r["litellm_model"] == "openrouter/some/unknown-model"


def test_resolve_nvidia_nim_prefix(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nvapi-x")
    r = resolve_model("nvidia_nim/meta/llama-3.3-70b-instruct")
    assert r["litellm_model"] == "nvidia_nim/meta/llama-3.3-70b-instruct"
    assert r["api_base"] is None
    assert r["api_key"] == "nvapi-x"


def test_build_chain_drops_keyless_and_orders_free_first(monkeypatch):
    for k in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "CLOUDFLARE_API_KEY",
              "GOOGLE_AISTUDIO_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    chain = build_provider_chain(
        ["openrouter/x/paid-model", "groq/llama-3.1-8b-instant"],
        free_first=True,
    )
    pids = [c["pid"] for c in chain]
    assert pids == ["groq/llama-3.1-8b-instant", "openrouter/x/paid-model"]
    assert all(c["api_key"] for c in chain)


def test_build_chain_empty_when_no_keys(monkeypatch):
    for k in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "CLOUDFLARE_API_KEY",
              "GOOGLE_AISTUDIO_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert build_provider_chain(DEFAULT_FREE_FIRST_CHAIN) == []


def test_resolve_local_prefix_routes_openai_compatible_base(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8081/v1/")
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    r = resolve_model("local/bonsai2-small")
    assert r["litellm_model"] == "openai/bonsai2-small"
    assert r["api_base"] == "http://127.0.0.1:8081/v1"
    assert r["api_key"]
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "sk-local")
    assert resolve_model("local/bonsai2-small")["api_key"] == "sk-local"


def test_build_chain_drops_local_without_base_url(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "sk-local")
    assert build_provider_chain(["local/bonsai2-small"], free_first=False) == []
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8081/v1")
    chain = build_provider_chain(["local/bonsai2-small"], free_first=False)
    assert [c["pid"] for c in chain] == ["local/bonsai2-small"] and chain[0]["api_base"] == "http://127.0.0.1:8081/v1"
