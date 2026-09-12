from types import SimpleNamespace

import pytest

from superclaw.compaction import PRUNE_MARKER, RESUME_NOTE, SUMMARY_LABEL, compact, cut_point, project, prune_tool_results, render_transcript
from superclaw.guards import Guards, ends_with_continuation_cue, error_signature
from superclaw.meter import ContextMeter
from superclaw.models import ModelInfo, lookup
from superclaw.provider import LitellmProvider, parse_response
from superclaw.runtime import Message, ToolCall, Usage, approx_tokens, estimate_tokens, to_wire
from superclaw.settings import LIMITS, Settings


def user(text):
    return Message(role="user", content=text)


def assistant(text="", calls=()):
    return Message(role="assistant", content=text, tool_calls=list(calls))


def tool(call_id, text, is_error=False):
    return Message(role="tool", content=text, tool_call_id=call_id, is_error=is_error)


def _resp(content, tool_calls=None, prompt=10, completion=5, cached=0):
    calls = [SimpleNamespace(id=tc["id"], type="function", function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"])) for tc in tool_calls or []]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls or None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, prompt_tokens_details=SimpleNamespace(cached_tokens=cached)),
    )


def test_wire_shape_and_token_estimate():
    msgs = [Message(role="system", content="sys"), user("hi"), assistant("", [ToolCall("c1", "read_file", '{"path": "a"}')]), tool("c1", "1→x"), assistant("done")]
    assert to_wire(msgs) == [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "1→x"}, {"role": "assistant", "content": "done"},
    ]
    assert approx_tokens("") == 0 and approx_tokens("abcd efgh") == 2 and approx_tokens("a" * 400) == 100 and approx_tokens("日本") == 6
    tools = [{"type": "function", "function": {"name": "t", "description": "d" * 40, "parameters": {}}}]
    assert estimate_tokens(msgs, tools) > estimate_tokens(msgs, []) >= 20


def test_model_catalog_resolves_window_and_prices_with_cached_input_at_the_cache_rate():
    info = lookup("openrouter/deepseek/deepseek-v4-flash")
    assert info.known and info.context_window > LIMITS.context_window_fallback and info.input_per_token > 0
    assert lookup("bogus-gateway/deepseek-v4-flash").known
    unknown = lookup("nobody/no-such-model")
    assert not unknown.known and unknown.context_window == LIMITS.context_window_fallback
    priced = ModelInfo("m", 1000, 100, input_per_token=1.0, output_per_token=10.0, cache_read_per_token=0.1)
    assert priced.cost(Usage(input_tokens=100, output_tokens=1, cache_read_tokens=40)) == 60 + 4 + 10
    env = {"SUPERCLAW_MODEL": "openrouter/deepseek/deepseek-v4-flash", "SUPERCLAW_BUDGET_USD": "0.5"}
    s = Settings.from_env(env)
    assert s.context_window == 0 and s.window() == info.context_window and s.budget_usd == 0.5
    assert Settings.from_env({**env, "SUPERCLAW_CONTEXT_WINDOW": "4096"}).window() == 4096


def test_provider_parses_usage_and_falls_back_across_the_chain(monkeypatch):
    c = parse_response(_resp("hello", prompt=50, cached=30))
    assert c.text == "hello" and c.tool_calls == [] and (c.usage.input_tokens, c.usage.cache_read_tokens) == (50, 30)
    c = parse_response(_resp(None, [{"id": "c1", "name": "read_file", "arguments": "{}"}, {"id": "c2", "name": "", "arguments": "{}"}]))
    assert [(t.id, t.name) for t in c.tool_calls] == [("c1", "read_file")]
    import superclaw.provider as mod

    seen = []

    def fake(**kw):
        seen.append(kw)
        if kw["model"] == "bad/model":
            raise RuntimeError("provider down")
        return _resp("ok")

    monkeypatch.setattr(mod, "_completion", fake)
    chain = [{"litellm_model": "bad/model", "api_key": "k", "api_base": None}, {"litellm_model": "good/model", "api_key": "k", "api_base": None}]
    tools = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {}}}]
    assert LitellmProvider(chain).complete([Message(role="user", content="hi")], tools).text == "ok"
    assert [kw["model"] for kw in seen] == ["bad/model", "good/model"] and seen[-1]["tools"] == tools and seen[-1]["tool_choice"] == "auto"
    LitellmProvider(chain[1:]).complete([Message(role="user", content="u")], [])
    assert "tools" not in seen[-1]
    monkeypatch.setattr(mod, "_completion", lambda **kw: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(RuntimeError):
        LitellmProvider(chain[:1]).complete([Message(role="user", content="hi")], [])


def test_cut_point_walks_back_by_tokens_and_prune_shortens_only_older_oversized_results():
    msgs = [Message(role="system", content="s"), user("u0"), assistant("a0"), user("u1"), assistant("", [ToolCall("c", "t", "{}")]), tool("c", "r" * 400), assistant("a1"), user("u2")]
    assert cut_point(msgs, keep_tokens=8) == 7 and cut_point(msgs, keep_tokens=115) == 4 and cut_point(msgs, keep_tokens=10_000) == 1
    big = "x" * (LIMITS.prune_threshold_chars + 10)
    msgs = [user("u"), assistant("", [ToolCall("c1", "t", "{}")]), tool("c1", big), tool("c2", "small"), assistant("", [ToolCall("c3", "t", "{}")]), tool("c3", big)]
    pruned = prune_tool_results(msgs, upto=4)
    assert [i for i, _ in pruned] == [2] and pruned[0][1].startswith("x" * LIMITS.prune_head_chars) and PRUNE_MARKER in pruned[0][1]
    meter = ContextMeter(window=1000, reserve=100)
    assert meter.used(estimate=50) == 50 and not meter.pressure(estimate=50)
    meter.observe(Usage(input_tokens=880, output_tokens=10))
    meter.append(user("x" * 80))
    assert meter.used(estimate=1) == 890 + approx_tokens("x" * 80) + LIMITS.message_overhead_tokens and meter.pressure(estimate=1)
    meter.reset()
    assert meter.used(estimate=7) == 7


def test_compaction_keeps_the_suffix_intact_and_projects_user_messages_verbatim():
    msgs = [Message(role="system", content="s"), user("a"), assistant("b")]
    assert not compact(msgs, keep_tokens=100, summarize=lambda m: "SUM").compacted
    msgs = [Message(role="system", content="s")]
    for i in range(6):
        msgs += [user(f"u{i}"), assistant(f"a{i}")]
    seen = []
    res = compact(msgs, keep_tokens=16, summarize=lambda m: seen.append(m) or "SUM")
    assert res.compacted and res.removed == 9 and res.messages[0].content == "s"
    assert res.messages[1].role == "user" and res.messages[1].content.startswith(SUMMARY_LABEL + "\nSUM") and res.messages[1].content.rstrip().endswith(RESUME_NOTE)
    assert [m.content for m in res.messages[2:]] == ["a4", "u5", "a5"] and all(f"[user #{i}]\nu{i // 2}" in seen[0] for i in range(0, 9, 2))
    msgs = [Message(role="system", content="s"), user("u0"), assistant("a0"), user("u1"), assistant("", [ToolCall("c", "t", "{}")]), tool("c", "r"), assistant("a1")]
    assert compact(msgs, keep_tokens=12, summarize=lambda m: "SUM").messages[2].tool_calls[0].id == "c"
    msgs = [Message(role="system", content="s"), user(f"{SUMMARY_LABEL}\nOLD FACTS"),
            assistant("", [ToolCall("c1", "skill", '{"name": "bench"}'), ToolCall("c2", "read_file", '{"path": "a"}')]), tool("c1", "skill body"), tool("c2", "file text"),
            assistant("", [ToolCall("c3", "edit_file", '{"path": "src/x.py", "old_string": "a", "new_string": "b"}')]), tool("c3", "Error: old_string not found", is_error=True),
            user("u"), assistant("a"), user("u2"), assistant("a2")]
    brief = project(msgs[1:7])
    assert brief.startswith("[previous summary]\nOLD FACTS") and "skill body" not in brief and "[tool_error #5] edit_file\nError: old_string not found" in brief
    body = compact(msgs, keep_tokens=12, summarize=lambda b: "SUM", plan_text="Current Plan:\n1. [pending] x").messages[1].content
    assert "Current Plan:" in body and "Skills loaded: bench" in body and "Files edited: src/x.py" in body
    text = render_transcript([assistant("", [ToolCall("c", "t", "a" * 1000)]), tool("c", "b" * 5000)])
    assert "a" * 500 in text and "a" * 600 not in text and "b" * 2000 in text and "b" * 2100 not in text


def test_guards_count_streaks_and_fire_each_reminder_once():
    g = Guards()
    assert [g.observe_turn("", 0) for _ in range(3)] == [False, False, True]
    outcomes = [g.observe_tool_result("edit_file", True, "Error: old_string not found in a.py") for _ in range(6)]
    assert [o.hint for o in outcomes] == [False, True, False, False, False, False] and [o.stop for o in outcomes] == [False] * 5 + [True]
    g.observe_tool_result("bash", False, "ok")
    assert g.observe_tool_result("bash", True, "Error: y").count == 1
    assert error_signature("Error: line 12 in /tmp/a/x.py") == error_signature("Error: line 99 in /tmp/b/y.py")
    g = Guards(stale_tool_calls=3)
    g.observe_tool_call("update_plan")
    for _ in range(3):
        g.observe_tool_call("read_file")
    assert g.stale_plan_reminder(pending=True) is not None and g.stale_plan_reminder(pending=True) is None and g.stale_plan_reminder(pending=False) is None
    g = Guards()
    assert [g.tool_only_reminder("", 2) is None for _ in range(7)] == [True] * 5 + [False, True]
    assert ends_with_continuation_cue("Now let me check the config:") and ends_with_continuation_cue("Done. Next, I'll run the tests:")
    assert not ends_with_continuation_cue("Here is the summary:") and not ends_with_continuation_cue("Let me know if you need anything")
