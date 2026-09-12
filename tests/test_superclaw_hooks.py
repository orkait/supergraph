import json
import os

from superclaw.hooks import Dispatcher, load_hooks
from superclaw.loop import Options, run
from superclaw.policy import Mode, Policy
from superclaw.runtime import Completion, ToolCall
from superclaw.tools import Registry
from superclaw.tools.files import core_file_tools


class Scripted:
    def __init__(self, *completions):
        self.queue = list(completions)

    def complete(self, messages, tools):
        return self.queue.pop(0) if self.queue else Completion(text="(exhausted)")


def _script(path, body):
    path.write_text("#!/usr/bin/env bash\n" + body)
    os.chmod(path, 0o755)
    return str(path)


def test_hooks_block_annotate_and_ask_to_continue(tmp_path):
    (tmp_path / "a.txt").write_text("x\n")
    block = _script(tmp_path / "block.sh", 'grep -q \'"tool": "write_file"\' && { echo "no writes today" >&2; exit 2; }; exit 0\n')
    note = _script(tmp_path / "note.sh", 'echo \'{"additionalContext": "read counted"}\'\n')
    stop = _script(tmp_path / "stop.sh", 'grep -q "first" && { echo "say more" >&2; exit 2; }; exit 0\n')
    config = tmp_path / "hooks.json"
    config.write_text(json.dumps({"enabled": True, "hooks": [
        {"id": "block-writes", "event": "beforeTool", "matcher": "^write_file$", "command": [block]},
        {"id": "note-reads", "event": "afterTool", "matcher": "read_file", "command": [note]},
        {"id": "stop-once", "event": "stop", "command": [stop]},
        {"id": "off", "event": "beforeTool", "command": ["/bin/false"], "enabled": False},
    ]}))
    assert load_hooks([tmp_path / "missing.json"]) == []
    assert [h.id for h in load_hooks([config])] == ["block-writes", "note-reads", "stop-once"]
    (tmp_path / "disabled.json").write_text(json.dumps({"hooks": [{"event": "stop", "command": ["/bin/true"]}]}))
    assert load_hooks([tmp_path / "disabled.json"]) == []

    reg = Registry()
    for t in core_file_tools():
        reg.register(t)
    provider = Scripted(
        Completion(tool_calls=[ToolCall("c1", "read_file", json.dumps({"path": "a.txt"})),
                               ToolCall("c2", "write_file", json.dumps({"path": "b.txt", "description": "d", "content": "y"}))]),
        Completion(text="first answer"),
        Completion(text="final answer"),
    )
    res = run("go", provider, Options(registry=reg, policy=Policy(tmp_path, Mode.AUTO, sandboxed=True), workspace=tmp_path,
                                      system_prompt="S", hooks=Dispatcher(load_hooks([config]), tmp_path)))
    tool_msgs = [m.content for m in res.messages if m.role == "tool"]
    assert "[hook] read counted" in tool_msgs[0]
    assert "blocked by hook block-writes: no writes today" in tool_msgs[1]
    assert not (tmp_path / "b.txt").exists()
    assert res.final_answer == "final answer"
    assert any("stop hook (stop-once)" in m.content and "say more" in m.content for m in res.messages if m.role == "user")
