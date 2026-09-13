import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from textual.widgets import Markdown

from supergraph import SuperGraph
from supergraph.core.errors import StoreInUse

from superclaw import catalog
from superclaw.app import Runtime, build_registry, build_runtime
from superclaw.memory import Memory
from superclaw.observations import ObservationStore
from superclaw.policy import Mode, Policy
from superclaw.report import doctor_lines
from superclaw.runtime import Completion, ToolCall
from superclaw.session import SessionStore
from superclaw.settings import ASCII, PROVIDERS, UNICODE, Settings, choose_glyphs
from superclaw.tui import PermissionScreen, SuperclawApp
from superclaw.tui.app import WORDMARK_ART
from superclaw.tui.cards import ToolCard
from superclaw.tui.models import ModelScreen
from superclaw.tui.setup import SetupScreen


class Scripted:
    def __init__(self, *completions):
        self.queue = list(completions)

    def complete(self, messages, tools):
        return self.queue.pop(0) if self.queue else Completion(text="(exhausted)")


@pytest.fixture
def rt(tmp_path):
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    memory = Memory(gs)
    rt = Runtime(gs=gs, store=SessionStore(gs), memory=memory, registry=build_registry(memory, ObservationStore(gs), tmp_path),
                 policy=Policy(tmp_path, Mode.ASK), provider=None, workspace=tmp_path, model="fake/model",
                 settings=Settings.from_env({"XDG_CONFIG_HOME": str(tmp_path / "cfg"), "XDG_CACHE_HOME": str(tmp_path / "cache"), "LANG": "C.UTF-8"}))
    yield rt
    gs.close()


async def _wait_for(pilot, predicate, timeout=5.0):
    for _ in range(int(timeout / 0.05)):
        if predicate():
            return
        await pilot.pause(0.05)
    raise AssertionError("condition not met in time")


def test_prompt_renders_answer_and_permission_modal_gates_writes(rt, tmp_path, monkeypatch):
    for provider in PROVIDERS:
        monkeypatch.delenv(provider.env, raising=False)
    monkeypatch.setattr("superclaw.report.keyed_providers", lambda: [])
    locked = tmp_path / "locked"
    holder = SuperGraph(path=str(locked), embedder="none", enable_sentence_nodes=False)
    lean = build_runtime(replace(rt.settings, db_path=locked), tmp_path, Mode.ASK, require_provider=False, open_store=False)
    assert lean.gs is None and lean.store is None and doctor_lines(lean, "/setup")[3].startswith(f"store {locked}")
    lean.close()
    with pytest.raises(StoreInUse):
        SuperGraph(path=str(locked), embedder="none")
    holder.close()
    lines = doctor_lines(rt, "/setup")
    assert lines[1].startswith("sandbox ") and f"model {rt.model}" in lines[2]
    assert str(rt.settings.db_path) in lines[3] and lines[4] == "mcp 0 tools · 0 servers" and lines[-1].endswith("none; /setup")
    monkeypatch.setattr(catalog, "_get", lambda url, headers: b'{"data": [{"id": "deepseek/deepseek-v4-flash", "context_length": 1048576, "supported_parameters": ["tools"]}]}')
    sid = rt.store.create(cwd=str(tmp_path), model=rt.model)
    setup_app = SuperclawApp(rt, sid)

    async def connect():
        async with setup_app.run_test(size=(100, 34)) as pilot:
            await _wait_for(pilot, lambda: isinstance(setup_app.screen, SetupScreen))
            await pilot.press(*"sk-or-v1-test", "enter")
            await _wait_for(pilot, lambda: isinstance(setup_app.screen, ModelScreen) and setup_app.screen.rows)
            await pilot.press(*"v4-flash", "enter")
            await _wait_for(pilot, lambda: not isinstance(setup_app.screen, (SetupScreen, ModelScreen)))
            assert rt.provider is not None and "OPENROUTER_API_KEY=sk-or-v1-test" in rt.settings.credentials.read_text()
            assert oct(rt.settings.credentials.stat().st_mode)[-3:] == "600" and rt.model.startswith("openrouter/deepseek/deepseek-v4-flash")
            await pilot.press(*"/model groq/llama-3.3-70b-versatile", "enter")
            await _wait_for(pilot, lambda: isinstance(setup_app.screen, SetupScreen))
            assert setup_app.screen.provider.name == "groq" and rt.model.startswith("openrouter/")
            await pilot.press("escape", *"/model list", "enter")
            await pilot.pause(0.2)
            assert any(str(n.content).startswith(rt.settings.glyphs.prompt) and rt.model in str(n.content) for n in setup_app.query(".note"))

    asyncio.run(connect())
    rt.provider = Scripted(
        Completion(tool_calls=[ToolCall("c1", "write_file", json.dumps({"path": "out.txt", "content": "hi"}))]),
        Completion(text="wrote **out.txt**"),
    )
    app = SuperclawApp(rt, sid)

    async def drive():
        async with app.run_test(size=(100, 30)) as pilot:
            assert app.query_one("#welcome").display and not app.query_one("#transcript").display
            await pilot.press("/")
            await _wait_for(pilot, lambda: app.query_one("#palette").option_count > 0)
            await pilot.press("backspace", *"write it", "enter")
            await _wait_for(pilot, lambda: isinstance(app.screen, PermissionScreen))
            assert app.stats.timer.paused_at and app.running
            await pilot.press("a")
            await _wait_for(pilot, lambda: len(app.query(Markdown)) == 1 and not app.running)
            assert (tmp_path / "out.txt").read_text() == "hi"
            card = app.query_one(ToolCard)
            assert card.tool == "write_file" and card.ok and "+hi" in str(card.query_one(".body").content)
            assert app.query_one("#transcript").display and not app.query_one("#welcome").display
            assert "done in" in str(app.query(".note").last().content) and app.stats.timer.calls == 1
            await pilot.press(*"/rename my work", "enter")
            await pilot.pause(0.05)
            assert rt.store.get(sid)["title"] == "my work" and "my work" in str(app.query_one("#title").content)
            await pilot.press(*"/tools", "enter")
            await pilot.pause(0.05)
            assert any("write_file" in str(n.content) and "write" in str(n.content) for n in app.query(".note"))
            await pilot.press(*"/export", "enter")
            await pilot.pause(0.05)
            assert (rt.workspace / f"superclaw-transcript-{sid}.md").read_text().startswith("# superclaw transcript")
            profiles = tmp_path / ".superclaw" / "agents"
            profiles.mkdir(parents=True)
            (profiles / "reader.md").write_text("---\nname: reader\ndescription: Reads only.\ntools: read_file\n---\nOnly read.")
            await pilot.press(*"/agent reader", "enter")
            await pilot.pause(0.05)
            assert rt.agent.name == "reader" and rt.policy.allow_tools == frozenset({"read_file"})
            await pilot.press(*"/agent none", "enter")
            await pilot.pause(0.05)
            assert rt.agent is None and rt.policy.allow_tools == frozenset()
            await pilot.press(*"/attach out.txt", "enter")
            await pilot.pause(0.05)
            assert "out.txt" in app.pending.text and "goes with your next message" in str(app.query(".note").last().content)
            await pilot.press(*"/attach nope.txt", "enter")
            await pilot.pause(0.05)
            assert "not a file" in str(app.query(".error").last().content)
            assert "write it" in app.history and app.hist_index == len(app.history)

    asyncio.run(drive())
    assert [e["type"] for e in rt.store.events(sid)] == ["prompt", "message", "message", "tool_result", "message"]
    commands = tmp_path / ".superclaw" / "commands"
    commands.mkdir(parents=True)
    (commands / "pr.md").write_text("---\ndescription: Open a PR.\n---\nOpen a PR for issue $1.")
    rt.provider = Scripted(Completion(text="opened"))
    slash_app = SuperclawApp(rt, sid)

    async def slash():
        async with slash_app.run_test(size=(100, 30)) as pilot:
            await pilot.press(*"/pr")
            await _wait_for(pilot, lambda: slash_app.query_one("#palette").option_count > 0)
            assert "Open a PR." in str(slash_app.query_one("#palette").get_option_at_index(0).prompt)
            await pilot.press(*" 42", "enter")
            await _wait_for(pilot, lambda: not slash_app.running and len(slash_app.query(Markdown)) == 1)
            assert slash_app.history[-1] == "/pr 42"

    asyncio.run(slash())
    assert [e["payload"]["content"] for e in rt.store.events(sid) if e["type"] == "message" and e["payload"]["role"] == "user"][-1] == "Open a PR for issue 42."
    drawn = {ch for path in Path(SuperclawApp.__module__.replace(".", "/")).parent.glob("*.py") for ch in path.read_text() if not ch.isascii()}
    allowed = {ch for value in vars(UNICODE).values() if isinstance(value, str) for ch in value} | set("".join(WORDMARK_ART)) | {"§"}
    assert drawn <= allowed and choose_glyphs({"LANG": "C"}) is ASCII and choose_glyphs({"LANG": "C.UTF-8", "SUPERCLAW_ASCII": "1"}) is ASCII
