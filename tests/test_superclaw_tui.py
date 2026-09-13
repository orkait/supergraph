import asyncio
import json

import pytest
from textual.widgets import Markdown

from supergraph import SuperGraph

from superclaw.app import Runtime, build_registry
from superclaw.memory import Memory
from superclaw.observations import ObservationStore
from superclaw.policy import Mode, Policy
from superclaw.runtime import Completion, ToolCall
from superclaw.session import SessionStore
from superclaw.settings import Settings
from superclaw.tui import PermissionScreen, SuperclawApp
from superclaw.tui.cards import ToolCard
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
                 settings=Settings.from_env({"XDG_CONFIG_HOME": str(tmp_path / "cfg")}))
    yield rt
    gs.close()


async def _wait_for(pilot, predicate, timeout=5.0):
    for _ in range(int(timeout / 0.05)):
        if predicate():
            return
        await pilot.pause(0.05)
    raise AssertionError("condition not met in time")


def test_prompt_renders_answer_and_permission_modal_gates_writes(rt, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    sid = rt.store.create(cwd=str(tmp_path), model=rt.model)
    setup_app = SuperclawApp(rt, sid)

    async def connect():
        async with setup_app.run_test(size=(100, 34)) as pilot:
            await _wait_for(pilot, lambda: isinstance(setup_app.screen, SetupScreen))
            await pilot.press(*"sk-or-v1-test", "enter")
            await _wait_for(pilot, lambda: not isinstance(setup_app.screen, SetupScreen))
            assert rt.provider is not None and "OPENROUTER_API_KEY=sk-or-v1-test" in rt.settings.credentials.read_text()
            assert oct(rt.settings.credentials.stat().st_mode)[-3:] == "600"

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
            assert card.tool == "write_file" and card.status == "✓" and "+hi" in str(card.query_one(".body").content)
            assert app.query_one("#transcript").display and not app.query_one("#welcome").display
            assert "done in" in str(app.query(".note").last().content) and app.stats.timer.calls == 1

    asyncio.run(drive())
    assert [e["type"] for e in rt.store.events(sid)] == ["prompt", "message", "message", "tool_result", "message"]
