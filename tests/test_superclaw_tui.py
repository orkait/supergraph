import asyncio
import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from textual.events import Paste, TextSelected
from textual.selection import SELECT_ALL
from textual.widgets import Input, OptionList, Static

from supergraph import SuperGraph
from supergraph.core.errors import StoreInUse

from superclaw import catalog, clipboard, render, viz
from superclaw.app import Callbacks, Runtime, build_registry, build_runtime, run_once
from superclaw.memory import Memory
from superclaw.observations import ObservationStore
from superclaw.policy import Mode, Policy
from superclaw.render import Stream, markdown
from superclaw.report import doctor_lines
from superclaw.runtime import Completion, ToolCall
from superclaw.session import SessionStore
from superclaw.config import OPTIONS
from superclaw.sandbox import available
from superclaw.settings import ASCII, PROVIDERS, RENDERER_ENV, RENDERER_INLINE, SANDBOX_ENV, SANDBOX_OFF, SANDBOX_ON, UNICODE, Settings, choose_glyphs
from superclaw.tools import ToolContext
from superclaw.tui import PermissionScreen, SuperclawApp
from superclaw.tui.app import WORDMARK_ART, context_overview, describe
from superclaw.meter import ContextMeter
from superclaw.tui.cards import ToolCard, target_of
from superclaw.tui.config import ConfigScreen
from superclaw.tui.status import RunStats, StatusBar
from superclaw.tui.models import ModelScreen
from superclaw.tui.setup import SetupScreen
from superclaw.tui.status import WorkingLine


class Scripted:
    def __init__(self, *completions):
        self.queue = list(completions)

    def complete(self, messages, tools, **kw):
        return self.queue.pop(0) if self.queue else Completion(text="(exhausted)")


@pytest.fixture
def rt(tmp_path):
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    memory = Memory(gs)
    rt = Runtime(gs=gs, store=SessionStore(gs), memory=memory, registry=build_registry(memory, ObservationStore(gs), tmp_path),
                 policy=Policy(tmp_path, Mode.ASK), provider=None, workspace=tmp_path, model="fake/model",
                 settings=Settings.from_env({"XDG_CONFIG_HOME": str(tmp_path / "cfg"), "XDG_CACHE_HOME": str(tmp_path / "cache"), "XDG_DATA_HOME": str(tmp_path / "data"), "LANG": "C.UTF-8"}))
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
        monkeypatch.delenv(provider.credential_env, raising=False)
    monkeypatch.setattr("superclaw.report.keyed_providers", lambda: [])
    locked = tmp_path / "locked"
    holder = SuperGraph(path=str(locked), embedder="none", enable_sentence_nodes=False)
    lean = build_runtime(replace(rt.settings, db_path=locked), tmp_path, Mode.ASK, require_provider=False, open_store=False)
    assert lean.gs is None and lean.store is None and doctor_lines(lean, "/setup")[3].startswith(f"store {locked}") and doctor_lines(lean, "/setup")[5].startswith("graph not opened")
    lean.close()
    full = build_runtime(replace(rt.settings, db_path=tmp_path / "full"), tmp_path, Mode.ASK, require_provider=False)
    assert full.registry.run("bash", {"command": "echo ok"}, ToolContext(workspace=tmp_path)).output == "ok"
    outcome: dict[str, object] = {}

    def off_main_thread() -> None:
        outcome["python"] = full.registry.run("python", {"code": "print(2 + 2)"}, ToolContext(workspace=tmp_path)).output
        outcome["slow"] = full.kernel.execute("import time; time.sleep(5)", 0.3)

    worker = threading.Thread(target=off_main_thread)
    worker.start()
    worker.join(30)
    assert "4" in str(outcome["python"]) and outcome["slow"] == (False, "Error: kernel timed out after 0.3s; the namespace was reset")
    full.close()
    with pytest.raises(StoreInUse):
        SuperGraph(path=str(locked), embedder="none")
    holder.close()
    lines = doctor_lines(rt, "/setup")
    assert lines[1].startswith("sandbox ") and f"model {rt.model}" in lines[2]
    assert str(rt.settings.db_path) in lines[3] and lines[4] == "mcp 0 tools · 0 servers" and lines[-1].endswith("none; /setup")
    assert lines[5].startswith("graph no embedder, lexical recall only · ") and lines[5].endswith(" edges")
    assert lines[6].startswith("health tombstones ") and "last maintain" in lines[6] and doctor_lines(lean, "/setup")[6].startswith("health not opened")
    diagram = "Flow:\n\n╭───╮\n│ a │\n╰───╯\n  │\n  ▼\n╭───╮\n│ b │\n╰───╯\n\nDone."
    drawn = markdown(diagram, 88, 92, UNICODE).plain.splitlines()
    assert "╭───╮" in drawn and drawn[drawn.index("╭───╮") + 3] == "  │", "an unfenced diagram keeps its own line breaks"
    narrow = markdown("| a | bb |\n|---|---:|\n| 1 | 2 |", 88, 92, UNICODE).plain.splitlines()
    assert narrow == ["a │ bb", "──┼───", "1 │  2"], narrow
    wide = markdown(f"| k | v |\n|---|---|\n| one | {'w ' * 60}|", 40, 40, UNICODE).plain.splitlines()
    assert wide[1].startswith("────┼") and wide[3].startswith("    │ ") and len([r for r in wide if "┼" in r]) == 1
    fenced = markdown("```python\nx = 1\n```", 88, 92, UNICODE).plain.splitlines()
    assert fenced[0].endswith("python") and fenced[1].endswith("x = 1")
    stream = Stream(88, 92, UNICODE)
    body = "## Head\n\nSome prose here.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```py\nx = 1\n```\n\n- one\n- two\n"
    for size in range(9, len(body) + 9, 9):
        partial = stream.update(body[:size])
    assert partial.plain.strip() == markdown(body, 88, 92, UNICODE).plain.strip(), "streaming must converge on the whole-document render"
    assert render.settled_at("a\n\n```\nb\n") == 3 and render.settled_at("a\n\nb\n") == 3
    assert viz.bar(0.5, 1.0, 8, UNICODE) == "████    " and viz.bar(0.55, 1.0, 8, UNICODE).startswith("████")
    assert viz.gauge(1, 4, 8, UNICODE) == "██░░░░░░" and viz.gauge(1, 4, 8, ASCII) == "@@......"
    assert viz.spark([1, 2, 3], UNICODE) == "▁▄█" and viz.spark([1, 2, 3], ASCII) == ".=@"
    assert viz.tree("r", {"r": ["a", "b"], "a": ["c"]}, UNICODE) == ["r", "├── a", "│   ╰── c", "╰── b"]
    assert viz.columns([1, 2], 1, UNICODE) == ["▄█"] and viz.heat([[0, 1]], UNICODE) == ["  ██"]
    drawing = viz.flow([[("a", "")], [("b", ""), ("c", "")]], [("a", "b"), ("a", "c")], UNICODE)
    assert drawing[0].strip() == "╭───╮" and any("┴" in row for row in drawing) and drawing[-2] == "│ b │   │ c │"
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
            await pilot.press("escape", *"/model local/bonsai2-small", "enter")
            await _wait_for(pilot, lambda: isinstance(setup_app.screen, SetupScreen) and setup_app.screen.provider.name == "local")
            url_input = setup_app.screen.query_one("#key")
            assert url_input.placeholder == "LOCAL_LLM_BASE_URL" and not url_input.password and rt.model.startswith("openrouter/")
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
            working = app.query_one(WorkingLine)
            assert app.stats.timer.paused_at and app.running and working.label == "waiting for you" and working.detail == "write_file  out.txt"
            await pilot.press("a")
            await _wait_for(pilot, lambda: len(app.query(".answer")) == 1 and not app.running)
            assert (tmp_path / "out.txt").read_text() == "hi" and working.has_class("hidden")
            card = app.query_one(ToolCard)
            assert card.tool == "write_file" and card.ok and "+hi" in str(card.query_one(".body").content) and str(card.query_one(".head").content).endswith("s")
            assert app.query_one("#transcript").display and not app.query_one("#welcome").display
            assert "done in" in str(app.query(".note").last().content) and app.stats.timer.calls == 1
            app.show_tool_result("read_file", {"path": "out.txt"})
            await pilot.pause(0.05)
            local = app.query(ToolCard).last()
            assert " line" in str(local.query_one(".head").content) and str(local.query_one(".body").content) == "" and "ctrl+o" in str(local.query_one(".more").content)
            await pilot.press("ctrl+o")
            await pilot.pause(0.05)
            assert app.verbose and "hi" in str(local.query_one(".body").content) and "full tool output" in str(app.query(".note").last().content)
            await pilot.press("ctrl+o")
            await pilot.pause(0.05)
            assert not app.verbose and str(local.query_one(".body").content) == ""
            await pilot.press(*"/rename my work", "enter")
            await pilot.pause(0.05)
            assert rt.store.get(sid)["title"] == "my work" and "my work" in str(app.query_one("#status").content)
            bar = str(app.query_one("#status").content)
            first, second = bar.split("\n")
            assert rt.model in first and "ask (shift+tab to cycle)" in second and "sandbox off" in second and "my work" in second
            assert first.startswith(app.short_cwd(app.size.width)) and not app.query("#title")
            near = RunStats(window=1_000_000, used=700_000)
            near.limit = ContextMeter(1_000_000).limit()
            assert "until compaction" in app.query_one("#status", StatusBar).context(near)
            far = RunStats(window=1_000_000, used=100_000)
            far.limit = ContextMeter(1_000_000).limit()
            assert app.query_one("#status", StatusBar).context(far).endswith("10%")
            written: list[str] = []
            monkeypatch.setattr(clipboard, "put", lambda value: written.append(value) or True)
            app.action_copy_selection()
            await pilot.pause(0.05)
            assert not written and "nothing selected" in str(app.query(".error").last().content)
            app.screen.selections = {app.query(".note").last(): SELECT_ALL}
            assert app.copy_selection() and written and "my work" in written[-1]
            app.post_message(TextSelected())
            await _wait_for(pilot, lambda: "to the clipboard" in str(app.query(".note").last().content))
            assert len(written) == 2 and written[-1] == written[0]
            await pilot.press(*"/tui sideways", "enter")
            await pilot.pause(0.05)
            assert "usage: /tui default|fullscreen" in str(app.query(".error").last().content)
            await pilot.press(*"/tui default", "enter")
            await pilot.pause(0.05)
            assert f"{RENDERER_ENV}={RENDERER_INLINE}" in rt.settings.credentials.read_text()
            assert "next time you start superclaw" in str(app.query(".note").last().content)
            await pilot.press(*"/sandbox sideways", "enter")
            await pilot.pause(0.05)
            assert "sandbox takes on or off" in str(app.query(".error").last().content)
            await pilot.press(*"/sandbox off", "enter")
            await pilot.pause(0.05)
            assert f"{SANDBOX_ENV}={SANDBOX_OFF}" in rt.settings.credentials.read_text() and not rt.settings.sandbox
            assert not rt.policy.sandboxed and rt.sandbox == "" and rt.registry.get("bash").backend is None
            assert "every command asks first outside unsafe mode" in str(app.query(".note").last().content)
            await pilot.press(*"/sandbox", "enter")
            await pilot.pause(0.05)
            assert str(app.query(".note").last().content).startswith(f"sandbox {SANDBOX_OFF}; usage:")
            await pilot.press(*f"/sandbox {SANDBOX_ON}", "enter")
            await pilot.pause(0.05)
            note = str(app.query(".note").last().content) if available() else str(app.query(".error").last().content)
            assert (rt.policy.sandboxed and "bubblewrap" in note) if available() else "bubblewrap is not installed" in note
            await pilot.press(*"/config repo_map on", "enter")
            await pilot.pause(0.05)
            assert rt.settings.repo_map and "SUPERCLAW_REPO_MAP=on" in rt.settings.credentials.read_text()
            await pilot.press(*"/config nope", "enter")
            await pilot.pause(0.05)
            assert "unknown setting 'nope'" in str(app.query(".error").last().content)
            await pilot.press(*"/config budget_tokens twelve", "enter")
            await pilot.pause(0.05)
            assert "budget_tokens takes a number" in str(app.query(".error").last().content)
            await pilot.press(*"/config", "enter")
            await _wait_for(pilot, lambda: isinstance(app.screen, ConfigScreen))
            table = app.screen.query_one("#settings", OptionList)
            assert table.option_count == len(OPTIONS) and str(table.get_option_at_index(5).prompt).startswith("repo_map")
            table.highlighted = 5
            await pilot.press("enter")
            await _wait_for(pilot, lambda: not rt.settings.repo_map)
            assert "repo_map off" in str(app.screen.query_one("#note", Static).content)
            table.highlighted = next(i for i, o in enumerate(OPTIONS) if o.key == "budget_tokens")
            await pilot.press("enter", *"40000", "enter")
            await _wait_for(pilot, lambda: rt.token_budget == 40000)
            assert rt.settings.budget_tokens == 40000
            await pilot.press("escape")
            await _wait_for(pilot, lambda: not isinstance(app.screen, ConfigScreen))
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
            prompt = app.query_one("#prompt", Input)
            await pilot.press(*"/attach out.txt", "enter")
            await pilot.pause(0.05)
            assert prompt.value == "[File #1] " and "[File #1] is out.txt" in str(app.query(".note").last().content) and len(app.clips) == 1
            prompt.value = ""
            await pilot.press(*"/attach nope.txt", "enter")
            await pilot.pause(0.05)
            assert "not a file" in str(app.query(".error").last().content) and prompt.value == ""
            (tmp_path / "shot.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 8)
            prompt.post_message(Paste(f"file://{tmp_path}/shot.png"))
            await pilot.pause(0.1)
            assert prompt.value == "[Image #1] " and len(app.clips) == 2
            prompt.post_message(Paste("line one\nline two\nline three\nline four"))
            await pilot.pause(0.1)
            assert prompt.value == "[Image #1] [Pasted text #1 +4 lines] " and len(app.clips) == 3
            prompt.post_message(Paste("short"))
            await pilot.pause(0.1)
            assert prompt.value == "[Image #1] [Pasted text #1 +4 lines] short"
            monkeypatch.setattr("superclaw.clipboard.image_bytes", lambda: (b"\x89PNG\r\n\x1a\n" + b"1" * 8, "image/png"))
            app.paste_clipboard()
            await pilot.pause(0.1)
            assert prompt.value.endswith("short[Image #2] ") and len(app.clips) == 4 and list(rt.settings.clipboard_dir.glob("clip-*.png"))
            monkeypatch.setattr("superclaw.clipboard.image_bytes", lambda: None)
            monkeypatch.setattr("superclaw.clipboard.text", lambda: "from the clipboard")
            app.paste_clipboard()
            await pilot.pause(0.1)
            assert prompt.value.endswith("[Image #2] from the clipboard")
            prompt.value = "see [Image #2] now"
            prompt.cursor_position = 14
            await pilot.press("backspace")
            assert prompt.value == "see  now" and prompt.cursor_position == 4
            prompt.value = "a [Pasted text #1 +4 lines] b"
            prompt.cursor_position = 10
            await pilot.press("ctrl+w", "x")
            assert prompt.value == "a x b"
            prompt.value = "[File #1]"
            prompt.cursor_position = 3
            await pilot.press("y")
            assert prompt.value == "[File #1]y"
            picked = app.clips.select("keep [Image #2] and [Pasted text #1 +4 lines], the rest was deleted")
            assert len(picked.images) == 1 and "line four" in picked.prompt and "out.txt" not in picked.prompt and picked.prompt.startswith("keep [Image #2]")
            assert "write it" in app.history and app.hist_index == len(app.history)
            app.render_event({"type": "context", "memories": 2, "skills": 3, "repo_files": 4, "history": 5, "prompt_tokens": 6})
            assert working.label == "thinking" and working.detail == "2 memories · 3 skills · repo map 4 files · 5 earlier messages"
            app.render_event({"type": "tool_call", "id": "t9", "name": "bash", "args": {"command": "pytest -q\necho done"}})
            assert working.label == "bash" and working.detail == "pytest -q"
            app.render_event({"type": "permission_request", "tool": "bash", "args": {"command": "pytest -q"}, "reason": "r", "risk": "low", "categories": [], "prefix": []})
            assert working.label == "waiting for you" and working.detail == "bash  pytest -q"
            assert target_of("bash", {"command": "export X=1\ndocker run repo pytest", "description": "Run the suite in the container"}) == "Run the suite in the container"
            assert target_of("bash", {"command": "ls -la", "description": "List"}) == "ls -la" and target_of("read_file", {"path": "a/b.py"}) == "a/b.py"
            app.render_event({"type": "permission_decision", "tool": "bash", "decision": "allow"})
            assert working.label == "bash" and "permission bash: allow" in str(app.query(".note").last().content)
            app.render_event({"type": "tool_result", "id": "t9", "ok": True, "output": "3 passed\n", "display": {}, "ref": ""})
            await pilot.pause(0.05)
            assert working.label == "thinking" and working.detail == "" and "1 line" in str(app.query(ToolCard).last().query_one(".head").content)
            app.render_event({"type": "text_delta", "text": "he"})
            assert working.label == "writing"
            app.cancel_flag.set()
            app.phase("cancelling")
            app.render_event({"type": "context", "memories": 0, "skills": 0, "repo_files": 0, "history": 0, "prompt_tokens": 1})
            app.render_event({"type": "tool_result", "id": "t9", "ok": True, "output": "", "display": {}, "ref": ""})
            assert working.label == "cancelling"
            app.cancel_flag.clear()
            app.stats.used, app.stats.cost = 5100, 0.0026
            prompt.value = ""
            await pilot.press(*"/clear", "enter")
            await pilot.pause(0.05)
            assert app.session_id != sid and app.stats.used == 0 and app.stats.cost == 0 and app.query_one("#welcome").display and not app.query_one("#transcript").display
            app.open_session(sid)
            await pilot.pause(0.05)
            assert app.session_id == sid and app.query_one("#transcript").display and not app.query_one("#welcome").display
            said = [str(w.content) for w in app.query(Static) if "user" in w.classes]
            assert any("write it" in line for line in said) and not any(line.startswith(f"{app.glyphs.prompt} [hook]") for line in said)
            assert app.query(ToolCard) and str(app.query(ToolCard).first().query_one(".head").content).startswith(app.glyphs.ok)

    asyncio.run(drive())
    assert [e["type"] for e in rt.store.events(sid) if e["type"] != "usage"] == ["prompt", "message", "message", "tool_result", "message"]
    rt.memory.note("The user's name is Kai.")
    rt.provider, events = Scripted(Completion(text="you are Kai")), []
    run_once(rt, "what is my name", sid, Callbacks(on_event=events.append))
    assert events[0]["type"] == "context" and events[0]["memories"] == 1 and events[0]["history"] == 4 and events[0]["prompt_tokens"] > 0 and events[0]["facts"] == 0 and {"skills", "repo_files"} <= set(events[0])
    assert context_overview({"type": "context", "memories": 0, "skills": 0, "repo_files": 0, "history": 0, "prompt_tokens": 1}, UNICODE) == "fresh context"
    assert context_overview({"type": "context", "memories": 1, "skills": 1, "repo_files": 1, "history": 1, "prompt_tokens": 1}, UNICODE) == "1 memory · 1 skill · repo map 1 file · 1 earlier message"
    commands = tmp_path / ".superclaw" / "commands"
    commands.mkdir(parents=True)
    (commands / "pr.md").write_text("---\ndescription: Open a PR.\n---\nOpen a PR for issue $1.")
    (tmp_path / ".superclaw" / "skills" / "tidy").mkdir(parents=True)
    (tmp_path / ".superclaw" / "skills" / "tidy" / "SKILL.md").write_text("---\nname: tidy\ndescription: Tidy the code.\n---\nTIDY STEPS")
    rt.provider = Scripted(Completion(text="opened"), Completion(text="tidied"))
    slash_app = SuperclawApp(rt, sid)

    async def slash():
        async with slash_app.run_test(size=(100, 30)) as pilot:
            replayed = len(slash_app.query(".answer"))
            assert replayed and slash_app.query(ToolCard)
            await pilot.press(*"/pr")
            await _wait_for(pilot, lambda: slash_app.query_one("#palette").option_count > 0)
            assert "Open a PR." in str(slash_app.query_one("#palette").get_option_at_index(0).prompt)
            await pilot.press(*" 42", "enter")
            await _wait_for(pilot, lambda: not slash_app.running and len(slash_app.query(".answer")) == replayed + 1)
            assert slash_app.history[-1] == "/pr 42"
            await pilot.press(*"/ti")
            await _wait_for(pilot, lambda: slash_app.query_one("#palette").option_count > 0)
            assert str(slash_app.query_one("#palette").get_option_at_index(0).prompt).startswith("/tidy [args]") and "skill: Tidy the code." in str(slash_app.query_one("#palette").get_option_at_index(0).prompt)
            await pilot.press(*"dy now", "enter")
            await _wait_for(pilot, lambda: not slash_app.running and len(slash_app.query(".answer")) == replayed + 2)
            assert slash_app.history[-1] == "/tidy now"
            await pilot.press(*"/qu")
            await _wait_for(pilot, lambda: slash_app.query_one("#palette").option_count > 0)
            assert "/exit" in str(slash_app.query_one("#palette").get_option_at_index(0).prompt)
            await pilot.press("ctrl+u", *":q", "enter")

    asyncio.run(slash())
    assert slash_app.return_code == 0
    sent = [e["payload"]["content"] for e in rt.store.events(sid) if e["type"] == "message" and e["payload"]["role"] == "user"]
    assert sent[-2] == "Open a PR for issue 42." and sent[-1] == '<skill name="tidy">\nTIDY STEPS\n</skill>\n\nFollow the skill above for this request. now'
    drawn = {ch for path in Path(SuperclawApp.__module__.replace(".", "/")).parent.glob("*.py") for ch in path.read_text() if not ch.isascii()}
    allowed = {ch for value in vars(UNICODE).values() if isinstance(value, str) for ch in value} | set("".join(WORDMARK_ART)) | {"§"}
    assert drawn <= allowed and choose_glyphs({"LANG": "C"}) is ASCII and choose_glyphs({"LANG": "C.UTF-8", "SUPERCLAW_ASCII": "1"}) is ASCII
    drift = describe({"type": "prompt_drift", "previous": "fdfc90c91635c220", "current": "f7d6fdb31b7e0552"}, UNICODE)
    assert drift == "system prompt changed since this session last ran; the prompt event holds the new one" and describe({"type": "budget", "left": 3}, UNICODE) == "budget: left=3"
