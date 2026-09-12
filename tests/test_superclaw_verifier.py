import json

from superclaw.guards import Guards, ends_with_promise
from superclaw.loop import Options, run
from superclaw.policy import Mode, Policy
from superclaw.runtime import Completion, ToolCall
from superclaw.tools import Registry
from superclaw.tools.files import core_file_tools
from superclaw.verifier import parse_verdict


class Scripted:
    def __init__(self, *completions):
        self.queue = list(completions)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(list(messages))
        return self.queue.pop(0) if self.queue else Completion(text="(exhausted)")


def options(ws, **kw):
    reg = Registry()
    for t in core_file_tools():
        reg.register(t)
    return Options(registry=reg, policy=Policy(ws, Mode.AUTO, sandboxed=True), workspace=ws, system_prompt="SYS", **kw)


def test_parse_verdict_shapes():
    assert parse_verdict('{"passed": true, "reason": "ok", "nextAction": ""}').passed
    v = parse_verdict("thinking... {\"passed\": false, \"reason\": \"tests not run\", \"nextAction\": \"run pytest\"}")
    assert (v.passed, v.next_action) == (False, "run pytest")
    assert not parse_verdict("no json here").passed


def test_verifier_rejects_then_accepts(tmp_path):
    provider = Scripted(
        Completion(text="Fixed the bug."),
        Completion(text=json.dumps({"passed": False, "reason": "tests were never run", "nextAction": "run pytest -q"})),
        Completion(tool_calls=[ToolCall("c1", "read_file", json.dumps({"path": "missing"}))]),
        Completion(text="Ran them; all green."),
        Completion(text=json.dumps({"passed": True, "reason": "covered", "nextAction": ""})),
    )
    res = run("fix the bug and prove it", provider, options(tmp_path, verify=True, require_completion_signal=True))
    assert res.final_answer == "Ran them; all green."
    assert not res.incomplete
    nudges = [m for m in res.messages if m.role == "user" and "verifier:" in m.content]
    assert len(nudges) == 1 and "run pytest -q" in nudges[0].content
    verifier_requests = [r for r in provider.requests if r[0].content.startswith("You are the completion verifier")]
    assert len(verifier_requests) == 2
    assert "fix the bug and prove it" in verifier_requests[0][1].content


def test_promise_shaped_final_is_nudged_once(tmp_path):
    provider = Scripted(Completion(text="Done with part one.\n\nNext, I'll run the tests."), Completion(text="All done."))
    res = run("do it", provider, options(tmp_path))
    assert res.final_answer == "All done."
    assert sum("describes work still to do" in m.content for m in res.messages if m.role == "user") == 1
    assert ends_with_promise("Let me know if you need anything") is False


def test_identical_call_reminder_fires_at_three():
    g = Guards()
    assert g.observe_identical("read_file", '{"path":"a"}') is None
    assert g.observe_identical("read_file", '{"path":"a"}') is None
    assert "3 times" in g.observe_identical("read_file", '{"path":"a"}')
    assert g.observe_identical("read_file", '{"path":"b"}') is None
