import asyncio
import json

import pytest
from textual.widgets import Markdown

from supergraph import SuperGraph

from superclaw.app import Runtime, build_registry
from superclaw.memory import Memory
from superclaw.policy import Mode, Policy
from superclaw.runtime import Completion, ToolCall
from superclaw.session import SessionStore
from superclaw.tui import PermissionScreen, SuperclawApp


class Scripted:
    def __init__(self, *completions):
        self.queue = list(completions)

    def complete(self, messages, tools):
        return self.queue.pop(0) if self.queue else Completion(text="(exhausted)")


@pytest.fixture
def rt(tmp_path):
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    memory = Memory(gs)
    rt = Runtime(gs=gs, store=SessionStore(gs), memory=memory, registry=build_registry(memory, tmp_path),
                 policy=Policy(tmp_path, Mode.ASK), provider=None, workspace=tmp_path, model="fake/model")
    yield rt
    gs.close()


async def _wait_for(pilot, predicate, timeout=5.0):
    for _ in range(int(timeout / 0.05)):
        if predicate():
            return
        await pilot.pause(0.05)
    raise AssertionError("condition not met in time")


def test_prompt_renders_answer_and_permission_modal_gates_writes(rt, tmp_path):
    rt.provider = Scripted(
        Completion(tool_calls=[ToolCall("c1", "write_file", json.dumps({"path": "out.txt", "content": "hi"}))]),
        Completion(text="wrote **out.txt**"),
    )
    sid = rt.store.create(cwd=str(tmp_path), model=rt.model)
    app = SuperclawApp(rt, sid)

    async def drive():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press(*"write it", "enter")
            await _wait_for(pilot, lambda: isinstance(app.screen, PermissionScreen))
            await pilot.press("a")
            await _wait_for(pilot, lambda: len(app.query(Markdown)) == 1)
            assert (tmp_path / "out.txt").read_text() == "hi"
            assert not app.query_one("#prompt").disabled

    asyncio.run(drive())
    assert [e["type"] for e in rt.store.events(sid)] == ["prompt", "message", "message", "tool_result", "message"]
