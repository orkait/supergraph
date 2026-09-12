import json

import pytest

from supergraph import SuperGraph

from superclaw.compaction import SUMMARY_LABEL
from superclaw.loop import Options, run
from superclaw.policy import Mode, Policy
from superclaw.runtime import Completion, ToolCall
from superclaw.session import SessionStore
from superclaw.tools import Registry
from superclaw.tools.files import core_file_tools
from superclaw.tools.plan import UpdatePlan


class Scripted:
    def __init__(self, *completions):
        self.queue = list(completions)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append((list(messages), tools))
        if not self.queue:
            return Completion(text="(script exhausted)")
        return self.queue.pop(0)


def call(name, cid="c1", **args):
    return ToolCall(cid, name, json.dumps(args))


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    return tmp_path


def options(ws, mode="auto", **kw):
    reg = Registry()
    for t in core_file_tools():
        reg.register(t)
    reg.register(UpdatePlan())
    return Options(registry=reg, policy=Policy(ws, Mode(mode)), workspace=ws, system_prompt="SYS", **kw)


def test_tool_round_trip_and_session_persistence(ws):
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    store = SessionStore(gs)
    sid = store.create(cwd=str(ws), model="m")
    provider = Scripted(Completion(tool_calls=[call("read_file", path="a.txt")]), Completion(text="it says hello"))
    res = run("read a.txt", provider, options(ws, session=store, session_id=sid))
    assert res.final_answer == "it says hello"
    assert res.turns == 2
    assert [m.role for m in res.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert "1→hello" in res.messages[3].content
    assert [e["type"] for e in store.events(sid)] == ["prompt", "message", "message", "tool_result", "message"]
    assert store.replay(sid)[1].tool_calls[0].name == "read_file"
    gs.close()


def test_prompted_tool_is_denied_headless_and_allowed_via_callback(ws):
    provider = Scripted(Completion(tool_calls=[call("write_file", path="b.txt", content="x")]), Completion(text="done"))
    res = run("write", provider, options(ws, mode="ask"))
    assert "denied" in res.messages[3].content
    assert not (ws / "b.txt").exists()

    seen = []
    provider = Scripted(Completion(tool_calls=[call("write_file", path="b.txt", content="x")]), Completion(text="done"))
    run("write", provider, options(ws, mode="ask", on_permission=lambda req: seen.append(req["tool"]) or "allow"))
    assert seen == ["write_file"]
    assert (ws / "b.txt").read_text() == "x"


def test_same_error_streak_halts_the_run(ws):
    bad = [Completion(tool_calls=[call("edit_file", f"c{i}", path="a.txt", old_string="zzz", new_string="y")]) for i in range(8)]
    res = run("edit", Scripted(*bad), options(ws))
    assert res.stop_reason == "tool_failure_loop"
    assert "6 times" in res.final_answer
    hints = [m for m in res.messages if m.role == "user" and "match it exactly" in m.content]
    assert len(hints) == 1


def test_completion_gate_nudges_then_marks_incomplete(ws):
    provider = Scripted(
        Completion(tool_calls=[call("update_plan", plan=[{"content": "step", "status": "pending"}])]),
        Completion(text="Now let me do the step:"),
        Completion(text="still not done"),
        Completion(text="still not done"),
        Completion(text="still not done"),
    )
    res = run("do it", provider, options(ws, require_completion_signal=True))
    assert res.incomplete
    assert "pending" in res.incomplete_reason
    assert sum("the task is not finished" in m.content for m in res.messages if m.role == "user") == 3


def test_max_turns_asks_for_a_final_answer_without_tools(ws):
    provider = Scripted(*[Completion(tool_calls=[call("read_file", f"c{i}", path="a.txt")]) for i in range(2)], Completion(text="summary"))
    res = run("loop", provider, options(ws, max_turns=2))
    assert res.final_answer == "summary"
    assert provider.requests[-1][1] == []
    assert "tool-turn limit" in provider.requests[-1][0][-1].content


def test_compaction_triggers_and_uses_summarizer(ws):
    provider = Scripted(*[Completion(tool_calls=[call("read_file", f"c{i}", path="a.txt")]) for i in range(4)], Completion(text="ok"))
    res = run("go", provider, options(ws, context_window=600, preserve_last=2, summarize=lambda msgs: "SUMMARY"))
    summaries = [m for m in res.messages if m.content.startswith(SUMMARY_LABEL)]
    assert summaries and "SUMMARY" in summaries[0].content
    assert res.final_answer == "ok"
