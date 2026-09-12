import json
from pathlib import Path

from supergraph import SuperGraph

from superclaw.app import build_registry
from superclaw.loop import Options, run
from superclaw.memory import Memory
from superclaw.policy import Mode, Policy
from superclaw.runtime import Completion, ToolCall, approx_tokens
from superclaw.tools import ToolContext


class Scripted:
    def __init__(self, *completions):
        self.queue = list(completions)
        self.tools_seen = []

    def complete(self, messages, tools):
        self.tools_seen.append([t["function"]["name"] for t in tools])
        return self.queue.pop(0) if self.queue else Completion(text="(exhausted)")


def test_deferred_tools_load_through_tool_search(tmp_path):
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    reg = build_registry(Memory(gs), Path(tmp_path))
    eager = reg.definitions(loaded=set())
    names = [d["function"]["name"] for d in eager]
    assert names == ["bash", "edit_file", "grep", "read_file", "tool_search", "write_file"]
    assert approx_tokens(json.dumps(eager)) < 1000
    assert "- update_plan:" in reg.get("tool_search").description
    res = reg.run("tool_search", {"query": "select:update_plan,glob"}, ToolContext(workspace=tmp_path))
    assert res.meta["load_tools"] == ["update_plan", "glob"] and '"name": "update_plan"' in res.output
    assert not reg.run("tool_search", {"query": "zzz"}, ToolContext(workspace=tmp_path)).ok
    provider = Scripted(Completion(tool_calls=[ToolCall("c1", "tool_search", json.dumps({"query": "plan"}))]), Completion(text="ok"))
    run("go", provider, Options(registry=reg, policy=Policy(tmp_path, Mode.AUTO, sandboxed=True), workspace=tmp_path, system_prompt="S"))
    assert "update_plan" not in provider.tools_seen[0] and "update_plan" in provider.tools_seen[1]
    gs.close()
