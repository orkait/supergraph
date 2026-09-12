import json

from superclaw.intent import Kind, parse_kind
from superclaw.loop import Options, run
from superclaw.policy import Action, Mode, Policy
from superclaw.runtime import Completion, ToolCall, Usage
from superclaw.tools import Registry
from superclaw.tools.files import core_file_tools
from superclaw.tools.shell import Bash


class Scripted:
    def __init__(self, *completions):
        self.queue = list(completions)

    def complete(self, messages, tools):
        return self.queue.pop(0) if self.queue else Completion(text="(exhausted)")


def registry():
    reg = Registry()
    for t in (*core_file_tools(), Bash()):
        reg.register(t)
    return reg


def test_request_kind_limits_tools(tmp_path):
    assert parse_kind('{"kind": "diagnose"}') is Kind.DIAGNOSE and parse_kind("garbage") is Kind.CHANGE
    reg = registry()
    p = Policy(tmp_path, Mode.AUTO, sandboxed=True)
    p.request_kind = Kind.ANSWER
    assert [d["function"]["name"] for d in reg.definitions(p.visible)] == ["glob", "grep", "list_directory", "read_file"]
    assert p.evaluate(reg.get("write_file"), {"path": "a", "content": ""}).action == Action.DENY
    p.request_kind = Kind.DIAGNOSE
    names = [d["function"]["name"] for d in reg.definitions(p.visible)]
    assert "bash" in names and "edit_file" not in names
    p.request_kind = Kind.CHANGE
    assert p.evaluate(reg.get("edit_file"), {"path": "a", "old_string": "x", "new_string": "y"}).action == Action.ALLOW


def test_token_budget_stops_the_run(tmp_path):
    turn = Completion(tool_calls=[ToolCall("c", "read_file", json.dumps({"path": "nope"}))], usage=Usage(600, 100))
    provider = Scripted(turn, turn, turn, Completion(text="late"))
    res = run("go", provider, Options(registry=registry(), policy=Policy(tmp_path, Mode.AUTO, sandboxed=True), workspace=tmp_path,
                                      system_prompt="S", token_budget=1000))
    assert res.stop_reason == "budget" and res.incomplete and res.turns == 3
    assert "1400 tokens" in res.final_answer
