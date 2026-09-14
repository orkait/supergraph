import json
import os
from types import SimpleNamespace

import pytest

from superclaw.catalog import describe, models_for, resolve
from superclaw.compaction import PRUNE_MARKER, SUMMARY_LABEL, compact, cut_point, project, prune_tool_results
from superclaw.meter import ContextMeter
from superclaw.tui.status import RunStats
from superclaw.models import ModelInfo, lookup, priced
from superclaw.provider import LitellmProvider, hint, parse_response
from superclaw.runtime import Message, ToolCall, Usage, approx_tokens, message_tokens, to_wire
from superclaw.settings import LIMITS, PRICED_FILE, PROVIDERS, Settings, read_opencode_key
from supergraph.ingest.llm.resolve import build_provider_chain, resolve_model


def user(text):
    return Message(role="user", content=text)


def assistant(text="", calls=()):
    return Message(role="assistant", content=text, tool_calls=list(calls))


def tool(call_id, text, is_error=False):
    return Message(role="tool", content=text, tool_call_id=call_id, is_error=is_error)


def _resp(content, prompt=10, cached=0):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=5, prompt_tokens_details=SimpleNamespace(cached_tokens=cached)),
    )


def test_catalog_pricing_and_provider_fallback(monkeypatch, tmp_path):
    assert approx_tokens("abcd efgh") == 2 and approx_tokens("日本") == 6
    info = lookup("openrouter/deepseek/deepseek-v4-flash")
    assert info.known and info.context_window > LIMITS.context_window_fallback and lookup("nobody/no-such-model").context_window == LIMITS.context_window_fallback
    priced_dir = tmp_path / "priced"
    assert priced(info.id, priced_dir) is None and lookup(info.id, priced_dir) == info
    import superclaw.models as models_mod

    real_catalog = models_mod._catalog
    models_mod._catalog = lambda: (_ for _ in ()).throw(AssertionError("catalog must not load when priced"))
    try:
        assert lookup(info.id, priced_dir) == info and priced(info.id, priced_dir) == info
    finally:
        models_mod._catalog = real_catalog
    stale = json.loads((priced_dir / PRICED_FILE).read_text())
    stale[info.id]["at"] = 0
    (priced_dir / PRICED_FILE).write_text(json.dumps(stale))
    assert priced(info.id, priced_dir) is None
    ollama = next(p for p in PROVIDERS if p.name == "ollama")
    payload = json.dumps({"data": [{"id": "glm-5.2", "context_length": 200000, "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                                    "supported_parameters": ["tools"]}, {"id": "nomic-embed-text"}]}).encode()
    calls = []

    def fetch(url, headers):
        calls.append((url, headers.get("Authorization")))
        return payload

    live = models_for(ollama, "k", tmp_path, fetch=fetch)
    assert [m.id for m in live] == ["ollama/glm-5.2"] and live[0].tools and calls == [(ollama.models_url, "Bearer k")]
    assert models_for(ollama, "k", tmp_path, fetch=fetch) == live and len(calls) == 1 and models_for(ollama, "", tmp_path, online=False) == live
    assert lookup("ollama/glm-5.2", tmp_path).context_window == 200000 and lookup("ollama/glm-5.2", tmp_path).input_per_token == 1e-6
    assert resolve("glm", live, "ollama").id == "ollama/glm-5.2" and resolve("glm-5.2", live, "ollama") and resolve("zzz", live, "ollama") is None
    assert describe(live[0], "|") == "200.0K ctx | tools | $1.00/2.00 | live" and lookup("ollama/none", tmp_path).known is False
    assert "/setup" in hint("OpenrouterException: Invalid API Key", tui=True) and "superclaw models" in hint("x is not a valid model ID", tui=False) and hint("boom", tui=True) == ""
    opencode = next(p for p in PROVIDERS if p.name == "opencode")
    oc_body = json.dumps({"data": [{"id": "deepseek-v4-flash"}, {"id": "text-embedding-3"}]}).encode()
    oc_models = models_for(opencode, "sk-oc", tmp_path / "oc", fetch=lambda url, headers: oc_body)
    assert [m.id for m in oc_models] == ["opencode/deepseek-v4-flash"] and opencode.default_model == "opencode/deepseek-v4-flash"
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-oc-test")
    resolved = resolve_model("opencode/deepseek-v4-flash")
    assert resolved["litellm_model"] == "openai/deepseek-v4-flash" and resolved["api_base"].endswith("/zen/v1") and resolved["api_key"] == "sk-oc-test"
    auth = tmp_path / "opencode" / "auth.json"
    auth.parent.mkdir(parents=True)
    auth.write_text(json.dumps({"opencode-go": {"type": "api", "key": "oc-ambient"}}))
    assert read_opencode_key({"OPENCODE_AUTH_PATH": str(auth)}) == "oc-ambient"
    assert read_opencode_key({"OPENCODE_AUTH_PATH": str(tmp_path / "none.json")}) == ""
    auth.write_text(json.dumps({"opencode-go": {"type": "oauth", "key": "z"}}))
    assert read_opencode_key({"OPENCODE_AUTH_PATH": str(auth)}) == ""
    auth.write_text(json.dumps({"opencode-go": {"type": "api", "key": "oc-ambient"}}))
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_BASE", raising=False)
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(auth))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg2"))
    Settings.from_env()
    assert os.environ["OPENCODE_API_KEY"] == "oc-ambient" and os.environ["OPENCODE_API_BASE"] == "https://opencode.ai/zen/go/v1"
    go = resolve_model("opencode/deepseek-v4-flash")
    assert go["api_key"] == "oc-ambient" and go["api_base"].endswith("/zen/go/v1") and go["extra_headers"]["x-opencode-session"].startswith("ses_")
    chain = build_provider_chain(["opencode/deepseek-v4-flash"], free_first=False)
    assert chain[0]["extra_headers"]["x-opencode-session"].startswith("ses_")
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-zen")
    monkeypatch.delenv("OPENCODE_API_BASE", raising=False)
    assert resolve_model("opencode/deepseek-v4-flash")["api_base"].endswith("/zen/v1") and "extra_headers" not in resolve_model("opencode/deepseek-v4-flash")
    assert ModelInfo("m", 1000, 100, input_per_token=1.0, output_per_token=10.0, cache_read_per_token=0.1).cost(Usage(100, 1, 40)) == 74
    assert Settings.from_env({"SUPERCLAW_MODEL": info.id}).window() == info.context_window and Settings.from_env({"SUPERCLAW_CONTEXT_WINDOW": "4096"}).window() == 4096
    assert Settings.from_env({"SUPERCLAW_MODEL": info.id}).output_cap() == min(info.max_output_tokens, LIMITS.completion_max_tokens) and Settings.from_env({"SUPERCLAW_MAX_OUTPUT_TOKENS": "1234"}).output_cap() == 1234
    big, small = ContextMeter(1_000_000), ContextMeter(40_000)
    assert big.limit() == int(1_000_000 * LIMITS.compaction_trigger_share) and big.pressure(700_000) and not big.pressure(500_000)
    assert small.limit() == int(40_000 * LIMITS.compaction_trigger_share) and small.pressure(25_000) and not small.pressure(23_000)
    assert ContextMeter(1_000, reserve=100).limit() == 600 and ContextMeter(0).pressure(10_000) is False
    stats = RunStats(window=1000)
    assert stats.cache_hit == 0.0
    stats.cached, stats.sent = 900, 1200
    assert stats.cache_hit == 0.75
    assert Settings.from_env({"SUPERCLAW_MODEL": "nobody/no-such-model"}).output_cap() == LIMITS.max_output_tokens_fallback
    assert parse_response(_resp("hello", prompt=50, cached=30)).usage.cache_read_tokens == 30
    import superclaw.provider as mod

    seen = []

    def fake(**kw):
        seen.append(kw["model"])
        if kw["model"] == "bad/model":
            raise RuntimeError("provider down")
        return _resp("ok")

    monkeypatch.setattr(mod, "_completion", fake)
    chain = [{"litellm_model": "bad/model", "api_key": "k", "api_base": None}, {"litellm_model": "good/model", "api_key": "k", "api_base": None}]
    assert LitellmProvider(chain).complete([user("hi")], []).text == "ok" and seen == ["bad/model", "good/model"]
    with pytest.raises(RuntimeError):
        LitellmProvider(chain[:1]).complete([user("hi")], [])
    kw: dict = {}

    def capture(**kwargs):
        kw.update(kwargs)
        return _resp("ok")

    monkeypatch.setattr(mod, "_completion", capture)
    one = [{"litellm_model": "m", "api_key": "k", "api_base": None}]
    LitellmProvider(one, effort="high").complete([user("hi")], [])
    assert kw["reasoning_effort"] == "high"
    kw.clear()
    LitellmProvider(one).complete([user("hi")], [])
    assert "reasoning_effort" not in kw
    kw.clear()
    LitellmProvider([{"litellm_model": "m", "api_key": "k", "api_base": None, "extra_headers": {"x-opencode-session": "ses_x"}}]).complete([user("hi")], [])
    assert kw["extra_headers"] == {"x-opencode-session": "ses_x"}
    assert Settings.from_env({"SUPERCLAW_EFFORT": "high"}).effort == "high" and Settings.from_env({"SUPERCLAW_EFFORT": "bogus"}).effort == "" and Settings.from_env({}).effort == ""
    assert Settings.from_env({"SUPERCLAW_FALLBACK_MODELS": "a/b, c/d"}).fallback_models == ("a/b", "c/d") and Settings.from_env({}).fallback_models == ()
    import superclaw.app as app_mod

    asked: list[list[str]] = []
    import supergraph.ingest.llm.resolve as resolve_mod

    monkeypatch.setattr(resolve_mod, "build_provider_chain", lambda models, **kw: asked.append(list(models)) or [])
    assert app_mod.connect_provider("m/a", "", ("m/b", "m/c")) is None and asked == [["m/a", "m/b", "m/c"]]
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    keyed = app_mod.connect_provider("openrouter/x/y")
    assert keyed is not None and asked == [["m/a", "m/b", "m/c"]]
    monkeypatch.setattr(resolve_mod, "build_provider_chain", lambda models, **kw: asked.append(list(models)) or [{"litellm_model": "x/y"}])
    assert keyed.model == "x/y" and asked[-1] == ["openrouter/x/y"]

    def part(index, cid, name, args):
        return SimpleNamespace(index=index, id=cid, function=SimpleNamespace(name=name, arguments=args))

    def piece(content=None, calls=None, finish=None):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=calls), finish_reason=finish)], usage=None)

    def streamed(**kwargs):
        assert kwargs["stream"] is True and kwargs["stream_options"] == {"include_usage": True}
        return iter([piece("<think>hidden"), piece(" more</think>Hel"), piece("lo", [part(0, "call_9", "read_file", '{"pa')]),
                     piece(None, [part(0, None, None, 'th": "a"}')], "tool_calls"),
                     SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3, prompt_tokens_details=SimpleNamespace(cached_tokens=4)))])

    monkeypatch.setattr(mod, "_completion", streamed)
    frags: list[str] = []
    done = LitellmProvider(one).complete([user("hi")], [], on_text=frags.append)
    assert frags == ["Hel", "lo"] and done.text == "Hello" and done.usage.cache_read_tokens == 4 and done.finish_reason == "tool_calls"
    assert [(c.id, c.name, c.arguments) for c in done.tool_calls] == [("call_9", "read_file", '{"path": "a"}')]
    assert LitellmProvider(one, stream=False).streams is False and Settings.from_env({}).stream and not Settings.from_env({"SUPERCLAW_STREAM": "off"}).stream


def test_meter_cut_prune_and_compaction():
    shot = Message(role="user", content="look", images=["data:image/png;base64,AAA"])
    assert to_wire([shot])[0]["content"] == [{"type": "text", "text": "look"},
                                             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]
    assert message_tokens(shot) == approx_tokens("look") + LIMITS.message_overhead_tokens + LIMITS.image_tokens
    meter = ContextMeter(window=1000, reserve=100)
    meter.observe(Usage(input_tokens=880, output_tokens=10))
    meter.append(user("x" * 80))
    assert meter.pressure(estimate=1) and ContextMeter(window=4000, reserve=16_384).pressure(estimate=2399) is False
    msgs = [Message(role="system", content="s"), user("u0"), assistant("a0"), user("u1"), assistant("", [ToolCall("c", "t", "{}")]), tool("c", "r" * 400), assistant("a1"), user("u2")]
    assert cut_point(msgs, keep_tokens=8) == 7 and cut_point(msgs, keep_tokens=115) == 4 and cut_point(msgs, keep_tokens=10_000) == 1
    big = "x" * (LIMITS.prune_threshold_chars + 10)
    cited = prune_tool_results([tool("c1", big + "\n[§abcdef12]"), tool("c2", "small")], upto=2)
    assert [i for i, _, _ in cited] == [0] and cited[0][2] == "abcdef12" and "recall §abcdef12 to expand" in cited[0][1]
    assert PRUNE_MARKER.format(recall="") in prune_tool_results([tool("c1", big)], upto=1)[0][1]
    msgs = [Message(role="system", content="s")]
    for i in range(6):
        msgs += [user(f"u{i}"), assistant(f"a{i}")]
    seen = []
    res = compact(msgs, keep_tokens=16, summarize=lambda m: seen.append(m) or "SUM")
    assert res.removed == 9 and res.messages[1].content.startswith(SUMMARY_LABEL + "\nSUM") and [m.content for m in res.messages[2:]] == ["a4", "u5", "a5"]
    assert all(f"[user #{i}]\nu{i // 2}" in seen[0] for i in range(0, 9, 2))
    msgs = [Message(role="system", content="s"), user(f"{SUMMARY_LABEL}\nOLD FACTS"),
            assistant("", [ToolCall("c1", "skill", '{"name": "bench"}'), ToolCall("c2", "read_file", '{"path": "a"}')]), tool("c1", "skill body"), tool("c2", "file text"),
            assistant("", [ToolCall("c3", "edit_file", '{"path": "src/x.py", "old_string": "a", "new_string": "b"}')]), tool("c3", "Error: old_string not found", is_error=True),
            user("u"), assistant("a"), user("u2"), assistant("a2")]
    brief = project(msgs[1:7])
    assert brief.startswith("[previous summary]\nOLD FACTS") and "skill body" not in brief and "[tool_error #5] edit_file" in brief
    body = compact(msgs, keep_tokens=12, summarize=lambda b: "SUM", plan_text="Current Plan:\n1. [pending] x").messages[1].content
    assert "Current Plan:" in body and "Skills loaded: bench" in body and "Files edited: src/x.py" in body
