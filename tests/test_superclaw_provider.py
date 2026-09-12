from types import SimpleNamespace

import pytest

from superclaw.provider import LitellmProvider, parse_response
from superclaw.runtime import Message


def _resp(content, tool_calls=None, finish="stop", prompt=10, completion=5):
    calls = []
    for tc in tool_calls or []:
        calls.append(SimpleNamespace(
            id=tc["id"],
            type="function",
            function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"]),
        ))
    msg = SimpleNamespace(content=content, tool_calls=calls or None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


class TestParseResponse:
    def test_plain_text(self):
        c = parse_response(_resp("hello"))
        assert c.text == "hello"
        assert c.tool_calls == []
        assert c.finish_reason == "stop"
        assert c.usage.input_tokens == 10 and c.usage.output_tokens == 5

    def test_tool_calls(self):
        c = parse_response(_resp(None, [{"id": "c1", "name": "read_file", "arguments": '{"path": "a"}'}], finish="tool_calls"))
        assert c.text == ""
        assert len(c.tool_calls) == 1
        assert (c.tool_calls[0].id, c.tool_calls[0].name, c.tool_calls[0].arguments) == ("c1", "read_file", '{"path": "a"}')

    def test_dropped_tool_call_without_name_is_skipped(self):
        c = parse_response(_resp(None, [{"id": "c1", "name": "", "arguments": "{}"}]))
        assert c.tool_calls == []


class TestChainFallback:
    def test_uses_first_working_provider(self, monkeypatch):
        calls = []

        def fake_completion(**kw):
            calls.append(kw["model"])
            if kw["model"] == "bad/model":
                raise RuntimeError("provider down")
            return _resp("ok")

        import superclaw.provider as mod
        monkeypatch.setattr(mod, "_completion", fake_completion)
        p = LitellmProvider(chain=[
            {"litellm_model": "bad/model", "api_key": "k", "api_base": None},
            {"litellm_model": "good/model", "api_key": "k", "api_base": None},
        ])
        out = p.complete([Message(role="user", content="hi")], tools=[])
        assert out.text == "ok"
        assert calls == ["bad/model", "good/model"]

    def test_all_fail_raises(self, monkeypatch):
        import superclaw.provider as mod
        monkeypatch.setattr(mod, "_completion", lambda **kw: (_ for _ in ()).throw(RuntimeError("down")))
        p = LitellmProvider(chain=[{"litellm_model": "x", "api_key": "k", "api_base": None}])
        with pytest.raises(RuntimeError):
            p.complete([Message(role="user", content="hi")], tools=[])

    def test_passes_tools_and_wire_messages(self, monkeypatch):
        seen = {}

        def fake_completion(**kw):
            seen.update(kw)
            return _resp("ok")

        import superclaw.provider as mod
        monkeypatch.setattr(mod, "_completion", fake_completion)
        p = LitellmProvider(chain=[{"litellm_model": "m", "api_key": "k", "api_base": None}])
        tools = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {}}}]
        p.complete([Message(role="system", content="s"), Message(role="user", content="u")], tools=tools)
        assert seen["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        assert seen["tools"] == tools
        assert seen["tool_choice"] == "auto"

    def test_no_tools_omits_tool_choice(self, monkeypatch):
        seen = {}
        import superclaw.provider as mod
        monkeypatch.setattr(mod, "_completion", lambda **kw: seen.update(kw) or _resp("ok"))
        p = LitellmProvider(chain=[{"litellm_model": "m", "api_key": "k", "api_base": None}])
        p.complete([Message(role="user", content="u")], tools=[])
        assert "tools" not in seen
